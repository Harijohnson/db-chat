"""
chat-db - a tiny local chat agent that does CRUD on a Supabase database.

You chat in plain English ("add a student named hari with pass Pass@123",
"rename student hari to harikrishnan", "delete student harikrishnan",
"add an event on 10 Oct 10 AM", "move the 10 Oct event to 1 AM") and a local
Ollama model decides which database tool to call. The tools talk to Supabase.

Run it with:  python main.py
Type /help once it is running to see the chat commands.
"""

import os
import json
from datetime import date

import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:1.7b"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# The only tables the agent may touch, and the only columns it may use per
# table. This stops the model from inventing table or column names and keeps
# it from ever writing to something it should not.
ALLOWED_TABLES = {
    "programs": {"id", "name", "description"},
    "students": {"id", "name", "password", "program_id"},
    "events": {"id", "title", "starts_at", "program_id"},
}

# How many rows a lookup returns at most, so results stay small for the model.
FIND_LIMIT = 50

SYSTEM_PROMPT = (
    "You are a database assistant for a school. You manage three tables in a\n"
    "Supabase database by calling tools.\n"
    "\n"
    f"Today's date is {date.today().isoformat()}. Turn relative dates like\n"
    '"10 Oct" into a full ISO datetime, e.g. "2026-10-10T10:00:00".\n'
    "\n"
    "Tables and their columns:\n"
    "  programs(id, name, description)\n"
    "  students(id, name, password, program_id)\n"
    "  events(id, title, starts_at, program_id)   -- starts_at is ISO datetime\n"
    "\n"
    "Tools:\n"
    "  db_find(table, match)    look up rows. Call this FIRST when the user\n"
    "                           refers to an existing record but you do not\n"
    "                           know its id.\n"
    "  db_insert(table, values) add one row.\n"
    "  db_update(table, match, values)  change the rows that fit `match`.\n"
    "  db_delete(table, match)  delete the rows that fit `match`.\n"
    "\n"
    "Rules:\n"
    "1. Never invent an id. If you need one, call db_find first.\n"
    "2. When adding a student, put the given password in the password column.\n"
    "3. For events, always set starts_at to a full ISO datetime.\n"
    "4. To rename/update or delete a record, pass a `match` such as\n"
    '   {"name": "hari"} or {"id": 3}.\n'
    "5. Make the fewest tool calls needed, then reply in one short sentence."
)


# ---------------------------------------------------------------------------
# Supabase client (created lazily so /help etc. work without credentials)
# ---------------------------------------------------------------------------

_client = None


def client():
    """Return a cached Supabase client, or raise a clear error if unconfigured."""
    global _client
    if _client is None:
        if not (SUPABASE_URL and SUPABASE_KEY):
            raise RuntimeError(
                "Supabase is not configured. Set SUPABASE_URL and "
                "SUPABASE_SERVICE_KEY in your .env file."
            )
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _validate(table, columns=()):
    """Return an error string if the table or any column is not allowed."""
    if table not in ALLOWED_TABLES:
        return f"Unknown table '{table}'. Allowed tables: {', '.join(ALLOWED_TABLES)}."

    allowed = ALLOWED_TABLES[table]
    unknown = [c for c in columns if c not in allowed]
    if unknown:
        return (
            f"Unknown column(s) {unknown} for table '{table}'. "
            f"Allowed columns: {', '.join(sorted(allowed))}."
        )
    return None


def _rows(response):
    """Pull the list of row dicts out of a Supabase response."""
    return getattr(response, "data", None) or []


def db_find(table, match=None):
    """Return rows from `table` that match every key/value in `match`."""
    match = match or {}
    error = _validate(table, match.keys())
    if error:
        return error

    query = client().table(table).select("*")
    for column, value in match.items():
        query = query.eq(column, value)

    rows = _rows(query.limit(FIND_LIMIT).execute())
    if not rows:
        return "No matching rows."
    return json.dumps(rows, default=str)


def db_insert(table, values):
    """Insert one row into `table`, after user confirmation."""
    error = _validate(table, values.keys())
    if error:
        return error

    if not confirm(f"insert into {table}: {values}"):
        return "User declined the insert."

    rows = _rows(client().table(table).insert(values).execute())
    return f"Inserted: {json.dumps(rows, default=str)}"


def db_update(table, match, values):
    """Update rows in `table` that fit `match`, after user confirmation."""
    error = _validate(table, list(match.keys()) + list(values.keys()))
    if error:
        return error
    if not match:
        return "Refusing to update without a match (that would change every row)."

    if not confirm(f"update {table} where {match} -> set {values}"):
        return "User declined the update."

    query = client().table(table).update(values)
    for column, value in match.items():
        query = query.eq(column, value)

    rows = _rows(query.execute())
    return f"Updated {len(rows)} row(s): {json.dumps(rows, default=str)}"


def db_delete(table, match):
    """Delete rows in `table` that fit `match`, after user confirmation."""
    error = _validate(table, match.keys())
    if error:
        return error
    if not match:
        return "Refusing to delete without a match (that would wipe the table)."

    if not confirm(f"delete from {table} where {match}"):
        return "User declined the delete."

    query = client().table(table).delete()
    for column, value in match.items():
        query = query.eq(column, value)

    rows = _rows(query.execute())
    return f"Deleted {len(rows)} row(s): {json.dumps(rows, default=str)}"


# ---------------------------------------------------------------------------
# Tool registry - schema sent to the model + dispatch table
# ---------------------------------------------------------------------------

_TABLE_PROP = {
    "type": "string",
    "description": "One of: programs, students, events",
}
_MATCH_PROP = {
    "type": "object",
    "description": 'Columns to match on, e.g. {"name": "hari"} or {"id": 3}',
}
_VALUES_PROP = {
    "type": "object",
    "description": 'Column values to write, e.g. {"name": "hari", "password": "Pass@123"}',
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "db_find",
            "description": "Look up rows in a table (omit match to list rows)",
            "parameters": {
                "type": "object",
                "properties": {"table": _TABLE_PROP, "match": _MATCH_PROP},
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "db_insert",
            "description": "Insert one new row into a table",
            "parameters": {
                "type": "object",
                "properties": {"table": _TABLE_PROP, "values": _VALUES_PROP},
                "required": ["table", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "db_update",
            "description": "Update rows that match, setting new column values",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": _TABLE_PROP,
                    "match": _MATCH_PROP,
                    "values": _VALUES_PROP,
                },
                "required": ["table", "match", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "db_delete",
            "description": "Delete rows that match",
            "parameters": {
                "type": "object",
                "properties": {"table": _TABLE_PROP, "match": _MATCH_PROP},
                "required": ["table", "match"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "db_find": lambda args: db_find(args["table"], args.get("match")),
    "db_insert": lambda args: db_insert(args["table"], args["values"]),
    "db_update": lambda args: db_update(args["table"], args["match"], args["values"]),
    "db_delete": lambda args: db_delete(args["table"], args["match"]),
}


def dispatch_tool(name, arguments):
    """Run the tool `name` with `arguments` and return its result as text."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Unknown tool: {name}"

    try:
        return func(arguments)
    except KeyError as missing:
        return f"Missing argument {missing} for tool {name}."
    except Exception as error:  # noqa: BLE001 - report any DB error back to the model
        return f"Tool {name} failed: {error}"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def confirm(action):
    """Ask the user to approve a write to the database."""
    answer = input(f"\n[confirm] Allow {action}? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def chat_once(messages, model):
    """Send the conversation to Ollama and return the assistant message.

    Returns None if Ollama cannot be reached, after printing a hint.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "think": False,
                "stream": False,
            },
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\nCould not reach Ollama at {OLLAMA_URL}.")
        print("Is it running? Start it with:  ollama serve")
        return None
    except requests.exceptions.RequestException as error:
        print(f"\nRequest to Ollama failed: {error}")
        return None

    return response.json()["message"]


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(user_input, messages, model):
    """Handle one user turn, running tools until the model gives an answer."""
    messages.append({"role": "user", "content": user_input})

    while True:
        message = chat_once(messages, model)
        if message is None:
            return

        # Keep the assistant's message in the conversation history.
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            print(f"\nAI: {message.get('content', '').strip()}")
            return

        # The model asked for one or more tools. Run each and feed results back.
        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]

            print(f"\n[tool] {name}({arguments})")
            result = dispatch_tool(name, arguments)
            print(f"[result] {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                }
            )


# ---------------------------------------------------------------------------
# Chat commands
# ---------------------------------------------------------------------------

HELP_TEXT = """\
Available commands:
  /help              Show this help
  /tools             List the database tools the agent can call
  /tables            Show the tables and columns the agent knows about
  /model [name]      Show the current model, or switch to `name`
  /reset             Clear the conversation history (keeps the system prompt)
  /history           Show how many messages are in the current context
  /clear             Clear the terminal screen
  /exit, /quit       Leave chat-db

Anything that does not start with `/` is sent to the agent as a request.
Examples:
  add a student named hari with password Pass@123
  rename student hari to harikrishnan
  delete student harikrishnan
  add an event on 10 Oct 10 AM
  move the 10 Oct event to 1 AM"""


def handle_command(command, messages, model):
    """Handle a `/command`. Returns (keep_running, model)."""
    parts = command.split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("/exit", "/quit"):
        print("Bye!")
        return False, model

    if cmd == "/help":
        print(HELP_TEXT)

    elif cmd == "/tools":
        for tool in TOOLS:
            function = tool["function"]
            print(f"  {function['name']:<12} {function['description']}")

    elif cmd == "/tables":
        for table, columns in ALLOWED_TABLES.items():
            print(f"  {table:<10} {', '.join(sorted(columns))}")

    elif cmd == "/model":
        if args:
            model = args[0]
            print(f"Model switched to: {model}")
        else:
            print(f"Current model: {model}")

    elif cmd == "/reset":
        del messages[1:]  # keep the system prompt at index 0
        print("Conversation history cleared.")

    elif cmd == "/history":
        print(f"{len(messages)} messages in context (including the system prompt).")

    elif cmd == "/clear":
        os.system("cls" if os.name == "nt" else "clear")

    else:
        print(f"Unknown command: {cmd}. Type /help for the list.")

    return True, model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    model = DEFAULT_MODEL
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("chat-db - talk to your Supabase database")
    print("Ask a question, or type /help for commands. /exit to quit.")

    if not (SUPABASE_URL and SUPABASE_KEY):
        print(
            "\n[warning] SUPABASE_URL / SUPABASE_SERVICE_KEY are not set in .env - "
            "database tools will fail until you add them."
        )

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            keep_running, model = handle_command(user_input, messages, model)
            if not keep_running:
                break
            continue

        run_agent(user_input, messages, model)


if __name__ == "__main__":
    main()
