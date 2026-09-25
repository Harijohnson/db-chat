"""
web.py - a browser front-end for the chat-db agent.

This does NOT reimplement the agent. It imports everything from main.py
(the system prompt, the tool schema, chat_once, dispatch_tool) and only adds
a thin web "harness" around it:

  * a FastAPI server with a small chat UI at  http://localhost:8000
  * a per-browser session that keeps its own message history
  * the terminal confirm() (y/N) is replaced by an Approve/Decline button
    shown in the browser before any write hits the database

Because the agent can pause mid-turn to ask for confirmation, each user turn
runs in a background thread and the browser polls /api/events for what the
agent is doing (tool calls, results, the confirm prompt, and the final answer).

Run it with:  uvicorn web:app --reload
Then open:    http://localhost:8000
"""

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import main  # the agent lives here; we reuse it as-is

BASE_DIR = Path(__file__).parent
CONFIRM_TIMEOUT = 300  # seconds to wait for the browser to Approve/Decline

app = FastAPI(title="chat-db web")

# Each agent turn runs in its own thread. We stash the active Session on a
# thread-local so main.confirm() (called deep inside a tool) can find the
# right browser to ask.
_local = threading.local()


# ---------------------------------------------------------------------------
# Session: one per browser tab. Holds history + a stream of events.
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, model):
        self.messages = [{"role": "system", "content": main.SYSTEM_PROMPT}]
        self.model = model
        self.events = []            # everything the agent has done this session
        self.pending = {}           # confirm_id -> {"event": Event, "approved": bool}
        self.busy = False           # True while a turn is running
        self.lock = threading.Lock()

    def emit(self, event):
        """Add an event for the browser to pick up on its next poll."""
        with self.lock:
            self.events.append(event)

    def events_after(self, cursor):
        with self.lock:
            return self.events[cursor:], len(self.events)


SESSIONS = {}


def get_session(session_id):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session; reload the page.")
    return session


# ---------------------------------------------------------------------------
# Web version of confirm(): pause the agent and ask the browser instead of
# the terminal. This replaces main.confirm (see the override at the bottom).
# ---------------------------------------------------------------------------

def web_confirm(action):
    session = getattr(_local, "session", None)
    if session is None:
        # No browser attached to this thread; refuse the write to stay safe.
        return False

    confirm_id = uuid.uuid4().hex
    gate = threading.Event()
    session.pending[confirm_id] = {"event": gate, "approved": False}
    session.emit({"type": "confirm", "id": confirm_id, "action": action})

    # Block this turn's thread until the browser answers (or we give up).
    gate.wait(timeout=CONFIRM_TIMEOUT)
    return session.pending.pop(confirm_id, {}).get("approved", False)


# ---------------------------------------------------------------------------
# The agent loop, web flavour: same shape as main.run_agent, but it emits
# structured events instead of printing to a terminal.
# ---------------------------------------------------------------------------

def run_turn(session, user_input):
    _local.session = session
    session.messages.append({"role": "user", "content": user_input})
    try:
        while True:
            message = main.chat_once(session.messages, session.model)
            if message is None:
                session.emit({
                    "type": "error",
                    "content": f"Could not reach Ollama at {main.OLLAMA_URL}. "
                               f"Is it running? Start it with: ollama serve",
                })
                return

            session.messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                session.emit({
                    "type": "answer",
                    "content": (message.get("content") or "").strip(),
                })
                return

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                session.emit({"type": "tool", "name": name, "arguments": arguments})
                result = main.dispatch_tool(name, arguments)
                session.emit({"type": "result", "content": str(result)})

                session.messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                })
    except Exception as error:  # noqa: BLE001 - surface any crash to the browser
        session.emit({"type": "error", "content": f"Agent error: {error}"})
    finally:
        session.emit({"type": "done"})
        session.busy = False


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class MessageIn(BaseModel):
    text: str
    session_id: str | None = None
    model: str | None = None


class ConfirmIn(BaseModel):
    session_id: str
    id: str
    approved: bool


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/info")
def info():
    return {
        "model": main.DEFAULT_MODEL,
        "tables": {t: sorted(cols) for t, cols in main.ALLOWED_TABLES.items()},
        "tools": [
            {"name": t["function"]["name"], "description": t["function"]["description"]}
            for t in main.TOOLS
        ],
        "supabase_configured": bool(main.SUPABASE_URL and main.SUPABASE_KEY),
    }


@app.post("/api/message")
def post_message(body: MessageIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message.")

    if body.session_id and body.session_id in SESSIONS:
        session = SESSIONS[body.session_id]
        session_id = body.session_id
    else:
        session_id = uuid.uuid4().hex
        session = Session(model=body.model or main.DEFAULT_MODEL)
        SESSIONS[session_id] = session

    if body.model:
        session.model = body.model

    if session.busy:
        raise HTTPException(status_code=409, detail="The agent is still working.")

    session.busy = True
    threading.Thread(target=run_turn, args=(session, text), daemon=True).start()
    return {"session_id": session_id}


@app.get("/api/events")
def get_events(session_id: str, cursor: int = 0):
    session = get_session(session_id)
    events, new_cursor = session.events_after(cursor)
    return {"events": events, "cursor": new_cursor, "busy": session.busy}


@app.post("/api/confirm")
def post_confirm(body: ConfirmIn):
    session = get_session(body.session_id)
    pending = session.pending.get(body.id)
    if pending is None:
        raise HTTPException(status_code=404, detail="No such confirmation (it may have timed out).")
    pending["approved"] = body.approved
    pending["event"].set()  # unblock the waiting agent thread
    return {"ok": True}


@app.post("/api/reset")
def post_reset(session_id: str):
    session = get_session(session_id)
    del session.messages[1:]  # keep the system prompt
    session.emit({"type": "info", "content": "Conversation history cleared."})
    return {"ok": True}


# Swap the terminal confirm for the browser one. Because the tools look up
# `confirm` on the main module at call time, this reroutes every write's
# approval to the browser without touching main.py.
main.confirm = web_confirm
