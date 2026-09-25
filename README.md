# chat-db

A tiny **local** chat agent that lets you manage a Supabase database in plain
English. You type things like *"add a student named hari with password
Pass@123"* or *"move the 10 Oct event to 1 AM"*, and a locally-running Ollama
model decides which database tool to call to make it happen.

Everything except the database itself runs on your machine: the model runs in
Ollama, the orchestration runs in [main.py](main.py), and only the final
database reads/writes leave your computer (to your Supabase project over HTTPS).

You can talk to it two ways — in the **terminal** (`python main.py`) or in your
**browser** (`uvicorn web:app`). Both drive the *same* agent; the browser front
end just reuses [main.py](main.py) behind a small web page. See
[Use it in the browser](#11-use-it-in-the-browser-web-ui).

---

## Table of contents

1. [The big picture](#1-the-big-picture)
2. [How one message flows through the agent](#2-how-one-message-flows-through-the-agent)
3. [The agent loop, line by line](#3-the-agent-loop-line-by-line)
4. [The four database tools](#4-the-four-database-tools)
5. [How the model knows what to do (the system prompt)](#5-how-the-model-knows-what-to-do-the-system-prompt)
6. [Safety guards](#6-safety-guards)
7. [The database schema](#7-the-database-schema)
8. [Setup](#8-setup)
9. [Usage & worked examples](#9-usage--worked-examples)
10. [Chat commands](#10-chat-commands)
11. [Use it in the browser (web UI)](#11-use-it-in-the-browser-web-ui)
12. [Configuration](#12-configuration)
13. [Troubleshooting](#13-troubleshooting)
14. [Security notes](#14-security-notes)
15. [File map](#15-file-map)

---

## 1. The big picture

There are **three** moving parts:

```
   ┌──────────────┐        ┌──────────────────────┐        ┌───────────────┐
   │  You (chat)  │  <-->  │   main.py (agent)    │  <-->  │    Ollama     │
   └──────────────┘        │                      │        │ qwen3:1.7b    │
                           │  - holds the convo   │        │ (local LLM)   │
                           │  - runs the tools    │        └───────────────┘
                           │  - asks you to       │
                           │    confirm writes    │        ┌───────────────┐
                           └──────────┬───────────┘  <-->  │   Supabase    │
                                      │                    │  (Postgres)   │
                                      └── db_find/insert/   └───────────────┘
                                          update/delete
```

- **Ollama** is the "brain". It is a large language model running locally. It
  does **not** touch the database itself — it can only *ask* to, by emitting a
  structured **tool call** (e.g. `db_insert(table="students", values={...})`).
- **main.py** is the "hands". It sends your message plus the list of available
  tools to Ollama, and when Ollama asks for a tool, main.py actually runs it
  against Supabase and feeds the result back to the model.
- **Supabase** is the database. main.py talks to it with the official
  `supabase-py` client.

This pattern — *LLM proposes a tool call, your code executes it, result goes
back to the LLM* — is the core idea of a "tool-calling agent". The loop repeats
until the model stops asking for tools and just answers in words.

---

## 2. How one message flows through the agent

Say you type **`add a student named hari with password Pass@123`**. Here is
exactly what happens:

```
  You: add a student named hari with password Pass@123
        │
        ▼
  main.py appends {"role": "user", "content": "..."} to the message list
        │
        ▼
  main.py POSTs the whole conversation + the TOOLS schema to Ollama
        │
        ▼
  Ollama replies with a tool_call, NOT text:
     db_insert(table="students", values={"name":"hari","password":"Pass@123"})
        │
        ▼
  main.py runs db_insert(...):
     - checks "students" and its columns are allowed          (safety)
     - asks YOU: "[confirm] Allow insert into students...? [y/N]"
     - on "y", calls supabase.table("students").insert(...).execute()
        │
        ▼
  main.py appends {"role":"tool", ...result...} to the message list
        │
        ▼
  main.py POSTs the conversation AGAIN (now with the tool result)
        │
        ▼
  Ollama sees the insert succeeded and replies with plain text:
     "Added student hari."
        │
        ▼
  main.py prints:  AI: Added student hari.
```

The key insight: the model is called **in a loop**. Each time, it either asks
for another tool (loop continues) or gives a final text answer (loop ends).

---

## 3. The agent loop, line by line

The heart of the program is `run_agent()` in [main.py](main.py). Simplified:

```python
def run_agent(user_input, messages, model):
    messages.append({"role": "user", "content": user_input})   # 1

    while True:                                                 # 2
        message = chat_once(messages, model)                    # 3
        messages.append(message)                                # 4

        tool_calls = message.get("tool_calls")                  # 5
        if not tool_calls:                                      # 6
            print(f"\nAI: {message.get('content', '').strip()}")
            return

        for tool_call in tool_calls:                            # 7
            name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            result = dispatch_tool(name, arguments)             # 8
            messages.append(                                    # 9
                {"role": "tool", "tool_name": name, "content": str(result)}
            )
```

1. **Record your message.** `messages` is the full conversation history — a list
   of dicts with `role` (`system` / `user` / `assistant` / `tool`). The model
   sees this entire list every turn; that is how it "remembers" context.
2. **Loop** until the model gives a text answer.
3. **`chat_once`** POSTs the conversation to Ollama and returns the assistant's
   reply message.
4. **Store the reply** in history (so the next request includes it).
5. **Check for tool calls.** Ollama puts them under `message["tool_calls"]`.
6. **No tool calls?** The model is done — print its text and return.
7. **Otherwise run each requested tool.** `arguments` already arrives as a Python
   dict from Ollama.
8. **`dispatch_tool`** looks the tool name up in `TOOL_FUNCTIONS` and runs it.
9. **Feed the result back** as a `role: "tool"` message, then loop — the model
   reads the result and decides what to do next.

`chat_once()` is the only network call to the model. It sends:

```python
{
    "model": model,          # "qwen3:1.7b"
    "messages": messages,    # the whole conversation so far
    "tools": TOOLS,          # the JSON schema describing the 4 db tools
    "think": False,
    "stream": False,
}
```

Passing `tools=TOOLS` is what lets the model emit structured tool calls instead
of guessing. If Ollama is not running, `chat_once` prints a friendly hint and
returns `None`, and the loop exits cleanly.

---

## 4. The four database tools

Instead of a separate tool for every operation on every table (which would be
~15 tools and confuses a small model), chat-db uses **four generic tools** that
work on any of the allowed tables. Each is a plain Python function in
[main.py](main.py):

| Tool | Signature | What it does |
|---|---|---|
| `db_find` | `db_find(table, match=None)` | Reads rows. With no `match`, lists rows (up to `FIND_LIMIT`). With `match`, filters by equality on each column. |
| `db_insert` | `db_insert(table, values)` | Inserts one row. Asks you to confirm first. |
| `db_update` | `db_update(table, match, values)` | Updates rows matching `match`, setting `values`. Confirms first. **Refuses if `match` is empty.** |
| `db_delete` | `db_delete(table, match)` | Deletes rows matching `match`. Confirms first. **Refuses if `match` is empty.** |

`match` and `values` are just dicts of `column -> value`, e.g.
`match={"name": "hari"}`, `values={"name": "harikrishnan"}`.

Under the hood each one builds a `supabase-py` query. For example `db_update`:

```python
query = client().table(table).update(values)
for column, value in match.items():
    query = query.eq(column, value)      # chain a .eq() per match key
rows = query.execute().data
```

So `db_update("students", {"name":"hari"}, {"name":"harikrishnan"})` becomes,
in effect:

```sql
UPDATE students SET name = 'harikrishnan' WHERE name = 'hari';
```

### How the tools reach the model

Two structures wire the functions to the LLM:

- **`TOOLS`** — a list of JSON-schema descriptions (name, description, parameter
  types). This is sent to Ollama so it knows the tools exist and their argument
  shapes. Shared property definitions (`_TABLE_PROP`, `_MATCH_PROP`,
  `_VALUES_PROP`) keep the descriptions consistent and give the model examples.
- **`TOOL_FUNCTIONS`** — a dispatch table mapping each tool name to the real
  Python function. `dispatch_tool()` uses it to run the right function and
  catches errors (missing arguments, DB failures) so a bad call is reported
  back to the model as text instead of crashing the program.

---

## 5. How the model knows what to do (the system prompt)

The model only knows about your tables because `SYSTEM_PROMPT` (top of
[main.py](main.py)) tells it. That prompt is the first message in every
conversation and it:

- Lists the three tables and their exact columns.
- **Injects today's date** (`date.today()`), so the model can turn *"10 Oct"*
  into a full ISO datetime like `2026-10-10T10:00:00`.
- Explains when to use each tool — crucially, *"call `db_find` FIRST when the
  user refers to an existing record but you don't know its id."*
- Gives rules: never invent an id, store the given password in the `password`
  column, always give events a full `starts_at`, always pass a `match` when
  updating/deleting.

Because the small `qwen3:1.7b` model has limited reasoning, this prompt is kept
tight and explicit — it is doing a lot of the "thinking" for the model.

---

## 6. Safety guards

Letting an LLM write to a database is risky, so main.py has several guardrails
that run **before** any database call:

1. **Table + column whitelist (`ALLOWED_TABLES`).** `_validate()` rejects any
   table or column the model invents. If the model tries `db_find("teachers")`
   or an unknown column, it gets an error string back instead of a query.
2. **No blind mass writes.** `db_update` and `db_delete` **refuse to run with an
   empty `match`** — this prevents an accidental "update/delete every row".
3. **Human confirmation on every write.** `confirm()` prints
   `[confirm] Allow ... ? [y/N]` and only proceeds on `y`/`yes`. Reads
   (`db_find`) never prompt.
4. **Errors are caught, not fatal.** `dispatch_tool()` wraps each call in
   try/except; a DB error becomes a message the model can read and react to.

You can see the guards fire (they run without any DB connection):

```
db_find('teachers')            -> Unknown table 'teachers'. Allowed tables: ...
db_insert('students', {'nickname':'x'}) -> Unknown column(s) ['nickname'] ...
db_update('students', {}, {...})        -> Refusing to update without a match ...
db_delete('students', {})               -> Refusing to delete without a match ...
```

---

## 7. The database schema

Defined in [schema.sql](schema.sql). Three tables:

```sql
programs(id, name, description, created_at)
students(id, name, password, program_id -> programs.id, created_at)
events(id, title, starts_at, program_id -> programs.id, created_at)
```

- `id` columns are auto-incrementing (`generated always as identity`).
- `students.program_id` and `events.program_id` are optional foreign keys to
  `programs` (set to `NULL` if the program is deleted).
- `events.starts_at` is a `timestamptz` (a full date **and** time).

> **Note:** `students.password` is stored in **plain text** for simplicity. Do
> not use this as-is for anything real — see [Security notes](#14-security-notes).

---

## 8. Setup

**Prerequisites:** Python 3.12, [Ollama](https://ollama.com) installed, and a
Supabase project.

**1. Create the tables.** In the Supabase dashboard → **SQL Editor** → New query
→ paste [schema.sql](schema.sql) → **Run**.

**2. Configure [.env](.env).** From Supabase → **Project Settings → API**:

```
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_SERVICE_KEY=<your service_role secret key>
```

Use the **service_role** key (not the `anon` key) so the agent can write.

**3. Install dependencies** (already done in your `venv`):

```powershell
venv\Scripts\pip.exe install -r requirements.txt
```

**4. Pull the model and start Ollama:**

```powershell
ollama pull qwen3:1.7b
ollama serve          # if it isn't already running
```

**5. Run it** — either in the terminal:

```powershell
venv\Scripts\python.exe main.py
```

…or in the browser (see [Use it in the browser](#11-use-it-in-the-browser-web-ui)):

```powershell
venv\Scripts\python.exe -m uvicorn web:app --reload
# then open http://localhost:8000
```

---

## 9. Usage & worked examples

Once running, just type requests. Every write pauses for your `y/N` confirmation
and the tool call + result are printed so you can watch what happens.

### Create — "add a student named hari with password Pass@123"
```
[tool] db_insert({'table': 'students', 'values': {'name': 'hari', 'password': 'Pass@123'}})
[confirm] Allow insert into students: {...}? [y/N] y
[result] Inserted: [{"id": 1, "name": "hari", ...}]
AI: Added student hari.
```

### Update — "rename student hari to harikrishnan"
The model updates by matching on the name in a single call:
```
[tool] db_update({'table': 'students', 'match': {'name': 'hari'},
                  'values': {'name': 'harikrishnan'}})
```

### Delete — "delete student harikrishnan"
```
[tool] db_delete({'table': 'students', 'match': {'name': 'harikrishnan'}})
```

### Create an event — "add an event on 10 Oct 10 AM"
The system prompt gave the model today's date, so it resolves the datetime:
```
[tool] db_insert({'table': 'events', 'values': {'starts_at': '2026-10-10T10:00:00'}})
```

### The "read then write" case — "move the 10 Oct event to 1 AM"
This is a **two-step** flow because the model must first learn the event's `id`:
```
[tool] db_find({'table': 'events'})                       # step 1: find the row
[result] [{"id": 4, "starts_at": "2026-10-10T10:00:00", ...}]
[tool] db_update({'table': 'events', 'match': {'id': 4},  # step 2: update by id
                  'values': {'starts_at': '2026-10-10T01:00:00'}})
```

---

## 10. Chat commands

Anything starting with `/` is a local command, handled by `handle_command()`
without calling the model:

| Command | Effect |
|---|---|
| `/help` | Show help and example requests |
| `/tools` | List the 4 database tools |
| `/tables` | Show tables and their columns (`ALLOWED_TABLES`) |
| `/model [name]` | Show or switch the Ollama model |
| `/reset` | Clear the conversation (keeps the system prompt) |
| `/history` | Show how many messages are in context |
| `/clear` | Clear the terminal |
| `/exit`, `/quit` | Leave chat-db |

---

## 11. Use it in the browser (web UI)

Prefer a browser to the terminal? [web.py](web.py) puts the **same agent** behind
a small web page. Start it with:

```powershell
venv\Scripts\python.exe -m uvicorn web:app --reload
```

then open **http://localhost:8000**. (Ollama must still be running —
`ollama serve` — exactly as for the terminal version.)

### It reuses the agent — it does not reimplement it

[web.py](web.py) does `import main` and calls `main`'s own `SYSTEM_PROMPT`,
`TOOLS`, `chat_once()`, and `dispatch_tool()`. Nothing about the agent changes.
web.py only adds a *different front end* (a "harness") around it:

```
   ┌───────────────┐     HTTP      ┌──────────────────────┐        same
   │  Browser UI   │  <-------->   │   web.py (FastAPI)   │  <-->  Ollama +
   │  static/      │   /api/...    │  - one session / tab │        Supabase
   │  index.html   │               │  - reuses main.py    │        as the CLI
   └───────────────┘               └──────────────────────┘
```

Compare the two front ends: [main.py](main.py)'s `run_agent()` reads with
`input()` and prints to the screen; [web.py](web.py)'s `run_turn()` runs the
identical loop but sends each step to the browser as JSON. Same brain, same
tools, different skin.

### The confirm step becomes an Approve / Decline button

In the terminal a write pauses on `[confirm] ... [y/N]`. In the browser it
pauses on a button instead. web.py makes this swap with a single line at the
bottom of the file:

```python
main.confirm = web_confirm   # reroute every write's approval to the browser
```

Because the tools look up `confirm` on the `main` module **at call time**, this
reroutes approvals to the browser without editing a single tool.

### Why a turn runs in the background

The agent can stop mid-turn to ask for approval, which does not fit a plain
request → response. So each turn runs in its own thread and the browser polls
for what the agent is doing:

```
  You type     ──►  POST /api/message     (starts a background thread)
                         │
  Browser polls ◄─────── GET /api/events  returns: tool call, result,
     every 0.5s                            a "confirm" prompt, the answer
                         │
  On a "confirm":   the thread BLOCKS until you click Approve / Decline
  Your click   ──►  POST /api/confirm      (unblocks the thread; loop resumes)
```

Sessions live in memory, one per browser tab, so restarting the server clears
history — fine for exploring, not meant for production.

### Endpoints

| Route | Purpose |
|---|---|
| `GET /` | The chat page ([static/index.html](static/index.html)) |
| `GET /api/info` | Model name, tables/columns, tools, and whether Supabase is configured |
| `POST /api/message` | Start an agent turn for a session (creates one if needed) |
| `GET /api/events` | Poll the events produced so far (tool calls, results, confirms, answer) |
| `POST /api/confirm` | Approve or decline a pending write |
| `POST /api/reset` | Clear a session's history (keeps the system prompt) |

---

## 12. Configuration

All at the top of [main.py](main.py):

| Setting | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Where Ollama listens |
| `DEFAULT_MODEL` | `qwen3:1.7b` | Which local model to use |
| `ALLOWED_TABLES` | programs/students/events | Whitelisted tables + columns |
| `FIND_LIMIT` | `50` | Max rows a `db_find` returns |
| `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` | from `.env` | Supabase credentials |

To add a table or column, edit `ALLOWED_TABLES` and mention it in
`SYSTEM_PROMPT` so the model knows about it.

---

## 13. Troubleshooting

- **"Could not reach Ollama…"** — start it with `ollama serve` and confirm the
  model is pulled (`ollama list`).
- **"Supabase is not configured"** — `SUPABASE_URL` / `SUPABASE_SERVICE_KEY`
  are missing/blank in [.env](.env).
- **Auth / permission errors from Supabase** — make sure you used the
  **service_role** key, and that you ran [schema.sql](schema.sql).
- **The model skips the `db_find` step, or malforms JSON** — `qwen3:1.7b` is
  very small and struggles with multi-step tool use. Try a bigger model:
  `/model qwen2.5:7b` (after `ollama pull qwen2.5:7b`).
- **Nothing inserts** — remember every write needs you to answer `y` at the
  `[confirm]` prompt.

---

## 14. Security notes

- **Plain-text passwords.** `students.password` is stored as-is. For real use,
  hash passwords (e.g. bcrypt) and never store the raw value.
- **The service_role key bypasses Row Level Security.** It is a powerful secret.
  Keep [.env](.env) out of version control (add it to `.gitignore`) and never
  share it.
- **Confirmation is your last line of defense.** The agent cannot write without
  your `y`. Read each `[confirm]` line before approving.

---

## 15. File map

| File | Role |
|---|---|
| [main.py](main.py) | The whole agent: config, tools, the Ollama loop, chat commands |
| [web.py](web.py) | FastAPI browser front end — reuses `main.py`, adds the web UI + confirm buttons |
| [static/index.html](static/index.html) | The single-page chat UI served by `web.py` |
| [schema.sql](schema.sql) | SQL to create the `programs`/`students`/`events` tables |
| [.env](.env) | Supabase URL + service key (secret — do not commit) |
| [requirements.txt](requirements.txt) | Pinned Python dependencies |
| [README.md](README.md) | This document |
