# Forge CLI — User Manual

**Version 0.2.1** · Local-first autonomous coding agent

---

## What is Forge?

Forge is a command-line coding agent that runs on your machine. It combines a **local LLM** (via LM Studio) for code writing with a **cloud brain** (Gemini) for planning and review. It can autonomously build entire features, remember decisions across sessions, and hand off context to other AI agents.

---

## Quick-Start

```bash
# 1. First-time setup (API keys, model selection)
forge setup

# 2. Index your project
forge init

# 3. Build something autonomously
forge auto "build a FastAPI REST API with JWT auth and SQLite"

# 4. Or run a single task
forge run "add input validation to the signup endpoint"

# 5. Check what Forge knows
forge status
```

---

## All Commands

### `forge setup`

**What it does:** First-time wizard that configures Forge — detects your local LM Studio instance, stores your Gemini API key, and writes a `.forge/.env` file in the current project.

**When to use:** Once per machine (for global settings) and once per project (for `.forge/.env`). If you get an "API key missing" error, run `forge setup --key`.

**Options:**

| Flag | Description |
|------|-------------|
| `--reset` | Wipe stored config and start over |
| `--key` | Update only the Gemini API key (skip everything else) |

**Examples:**
```bash
forge setup           # full first-time wizard
forge setup --key     # just update the API key
forge setup --reset   # start from scratch
```

---

### `forge init`

**What it does:** Scans every `.py`, `.js`, `.ts`, `.go`, `.rs`, `.md`, and `.txt` file in the current directory, splits them into 512-character chunks, embeds each chunk with `all-MiniLM-L6-v2`, and saves a FAISS vector index to `.forge/vectors/faiss.index`. Also builds a NetworkX dependency graph of imports/functions/classes.

**When to use:**
- Once when you start working on a project
- After adding many new files (re-run to update the index)
- If `forge status` shows 0 vector chunks

**Output:** Shows files indexed, chunks created, graph nodes/edges.

**Examples:**
```bash
forge init            # index current directory
forge init --verbose  # show each file being indexed
```

> [!NOTE]
> Excluded directories: `.git`, `__pycache__`, `node_modules`, `.venv`, `dist`, `build`, `.forge`

---

### `forge run "<goal>"`

**What it does:** Runs the Forge agent on a **single focused goal**. The Gemini brain breaks your goal into sub-tasks, the local LM Studio model executes each one, and Gemini reviews the output. Supports up to 3 retry iterations per sub-task.

**When to use:** For targeted tasks where you know exactly what you want — "add a rate limiter to the login route", "fix the bug in user serialisation", "write docstrings for auth.py".

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--file`, `-f` | none | Focus on a specific file for context |
| `--iterations`, `-i` | `3` | Max retry iterations per sub-task |
| `--dry-run` | off | Show the plan without executing it |

**Examples:**
```bash
forge run "add email validation to the signup form"
forge run "refactor the database connection to use connection pooling" -i 5
forge run "add auth middleware" --file src/middleware.py
forge run "build a user API" --dry-run    # preview the plan first
```

**After running:** Forge shows a diff of changes and asks: **apply / skip / rollback**.

---

### `forge chat`

**What it does:** Opens an interactive REPL where you can have a conversation with the Forge agent. Each message you type is treated as a goal and executed by the full plan→execute→review loop.

**When to use:** Exploratory sessions — when you're not sure exactly what you want, or want to iterate quickly without typing `forge run` each time.

**REPL commands (type inside the chat):**

| Command | Description |
|---------|-------------|
| `/status` | Show token usage and session statistics |
| `/summarise` | Summarise what was built this session |
| `/clear` | Reset session memory |
| `/help` | Show all commands |
| `/quit` | Exit the REPL |

**Example:**
```bash
forge chat
forge> add a health check endpoint
forge> now add rate limiting to it
forge> /status
forge> /quit
```

---

### `forge auto "<goal>"`

**What it does:** **Fully autonomous mode.** The Gemini brain generates a complete project roadmap (15–40 tasks), then executes every task one by one without you doing anything. Pauses every N tasks for a human checkpoint where you can: continue, edit the plan, rollback, or stop.

**When to use:** Building a new feature from scratch, or building an entire application. This is the most powerful command — give it a high-level goal and let it run.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint-every`, `-c` | `5` | Pause for human review every N tasks |
| `--max-tasks`, `-m` | `50` | Hard stop after N tasks (safety limit) |
| `--yes`, `-y` | off | Skip the initial confirmation prompt |
| `--from-plan` | none | Resume from a saved `plan.json` file |

**Examples:**
```bash
forge auto "build a FastAPI todo app with SQLite and JWT auth"
forge auto "add a payment module with Stripe" -c 3 --yes
forge auto           # resume an in-progress plan
forge auto --from-plan .forge/plan.json   # load a specific saved plan
```

**Checkpoint actions (when Forge pauses):**

| Key | Action |
|-----|--------|
| `c` | Continue executing the next batch |
| `e` | Open `plan.json` in your editor to modify tasks |
| `r` | Rollback all changes since the last checkpoint |
| `s` | Stop (plan is saved, resume later with `forge auto`) |

> [!TIP]
> Sessions are automatically recorded. Run `forge context` after a session to see what was built. Use `forge handoff` to package context for another agent.

---

### `forge status`

**What it does:** Shows a health dashboard — Forge init status, vector store size, graph stats, LM Studio connectivity, and Gemini API key status.

**When to use:** To check if everything is configured before running a task. If a component shows ❌, follow the hint to fix it.

**Example:**
```bash
forge status
```

**Sample output:**
```
🏗️  Forge              ✅ Initialized
   Last Init           2026-05-13T10:00:00
📦 Vector Store       ✅ 24 files, 312 chunks
🔗 Project Graph      ✅ 24 files, 180 functions, 42 classes
🤖 LM Studio          ✅ Connected — qwen3.5-9b-instruct
🧠 Master Brain       ✅ gemini/gemini-2.0-flash
```

---

### `forge summarise`

**What it does:** Asks the Gemini brain to summarise the most recent changes and saves a checkpoint JSON to `.forge/summaries/`. Useful for creating a human-readable record of what was built.

**When to use:** After completing a significant batch of work, before switching tasks, or before handing off to another developer.

**Options:**

| Flag | Description |
|------|-------------|
| `--force` | Create a checkpoint even if no recent changes are detected |

**Examples:**
```bash
forge summarise
forge summarise --force    # always generate, even with no changes
```

---

### `forge rollback`

**What it does:** Lists all `.forge_backup` files created by Forge during file edits and lets you restore any or all of them to their pre-Forge state.

**When to use:** If Forge made changes you don't want, or if a run went wrong and you want to undo everything.

**Interactive flow:**
1. Shows a table of all backup files with file names and sizes
2. You enter numbers (`1,3`) or `all`
3. Forge restores the originals and deletes the backups

**Examples:**
```bash
forge rollback   # interactive — pick which files to restore
```

---

### `forge remember "<text>"`

**What it does:** Stores a piece of information in **persistent memory** — a cross-session store that survives closing the terminal, switching models, or coming back days later. The text is embedded into a vector index and also stored in SQLite with decay tracking.

**When to use:**
- To save architectural decisions so Forge never forgets them
- To note requirements that must be respected across sessions
- To record errors you've already solved so you don't repeat them

**Options:**

| Flag | Choices | Default | Description |
|------|---------|---------|-------------|
| `--category`, `-c` | `requirement`, `decision`, `error`, `code`, `note` | `note` | Controls decay rate and retrieval priority |

**Decay rates** (how fast the memory loses priority over time):

| Category | Rate/day | Use for |
|----------|----------|---------|
| `requirement` | 2% | Business rules, non-negotiables |
| `decision` | 3% | Architecture choices, tech stack |
| `error` | 10% | Bug fixes, known pitfalls |
| `code` | 15% | Implementation patterns |
| `note` | 20% | General observations |

**Examples:**
```bash
forge remember "Use PostgreSQL, not SQLite — concurrent writes required" --category decision
forge remember "users table has composite PK on (user_id, tenant_id)" --category requirement
forge remember "bcrypt rounds must be >= 12 for security compliance" --category requirement
forge remember "ImportError on faiss-cpu when using Python 3.12 on Windows — pin to 1.7.4" --category error
```

---

### `forge recall "<query>"`

**What it does:** Performs a **semantic search** over everything stored with `forge remember`. Returns the most relevant memories sorted by a combined vector similarity + priority + recency score.

**When to use:** When starting a new session and you want to remind yourself what decisions were made. Or before implementing something to check if there's a stored constraint.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k`, `-k` | `5` | Number of results to return |
| `--category`, `-c` | all | Filter by a specific category |

**Examples:**
```bash
forge recall "database decisions"
forge recall "authentication approach" --top-k 3
forge recall "errors and pitfalls" --category error
forge recall "what tech stack are we using" --category decision
```

**Sample output:**
```
┏━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│ Category   │ Score   │ Text                                                    │
│ decision   │ 0.847   │ Use PostgreSQL, not SQLite — concurrent writes required │
│ requirement│ 0.721   │ users table has composite PK on (user_id, tenant_id)   │
└────────────┴─────────┴─────────────────────────────────────────────────────────┘
```

---

### `forge memory-status`

**What it does:** Shows statistics about the persistent memory store — how many memories are stored by category, how many sessions have been recorded, and where the memory files live on disk.

**When to use:** To check how much context has accumulated, or to verify that sessions are being recorded correctly.

**Example:**
```bash
forge memory-status
```

**Sample output:**
```
🧠 Persistent Memory Status
Memory entries          12
  (FAISS vectors)       12
Sessions recorded        4
Handoff packets          1
By category:
  decision               5
  requirement            3
  error                  2
  note                   2
Last session      today: add JWT authentication
Storage path      D:\myproject\.forge\memory
```

---

### `forge handoff`

**What it does:** Generates a compressed **context packet** that captures the entire project state — session history, key decisions, top memories, pending tasks, warnings — and saves it to `.forge/memory/handoff.gz`. This packet can be injected into any AI agent's system prompt to give it full project context instantly.

**When to use:**
- Switching from Forge to Claude Code, Cursor, or another agent
- Sharing project context with a teammate's AI assistant
- Starting a new session and wanting to inject full history automatically

**Options:**

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--target`, `-t` | `any` | `any`, `forge`, `cursor`, `claude-code`, `anti-gravity` | Target agent (informational, affects packet labelling) |
| `--print-prefix` | off | | Print the LLM-ready system prompt string instead of just the save confirmation |

**Examples:**
```bash
forge handoff                          # save handoff.gz
forge handoff --target claude-code     # label for Claude Code
forge handoff --print-prefix           # print the system prompt prefix to copy/paste
forge handoff --target cursor --print-prefix  # ready to paste into Cursor's system prompt
```

**The generated prompt prefix looks like:**
```
=== FORGE PROJECT CONTEXT ===
Project: /path/to/project (Python/FastAPI, 24 files)
Generated: 2026-05-13T10:00:00 UTC

WHAT WAS BUILT:
3 session(s). Built auth module with JWT tokens and refresh mechanism.

KEY DECISIONS:
- PostgreSQL chosen over SQLite (concurrent write support)
- JWT with RS256, refresh tokens stored in Redis

TASKS: 18 completed  4 pending  1 blocked

NEXT RECOMMENDED: Write unit tests for the auth module
...
=== END FORGE CONTEXT ===
```

---

### `forge context`

**What it does:** Prints the session history from persistent memory as a formatted string. Shows what was built in the last N sessions, which files changed, and any summaries.

**When to use:** At the start of a new work session to quickly remind yourself (and Forge) what was done before. The output is also suitable for copy-pasting into any LLM's chat.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--sessions`, `-n` | `3` | Number of past sessions to include |

**Examples:**
```bash
forge context              # last 3 sessions
forge context --sessions 5 # last 5 sessions
```

**Sample output:**
```
=== PREVIOUS SESSION CONTEXT ===

[1 day ago | goal: "add auth module"]
Completed: 12 tasks. Blocked: 0.
Changed: src/auth.py, src/models.py, src/routes/user.py
Model: gemini-2.0-flash
Summary: Built JWT authentication with RS256. Added user model and login route.

[today | goal: "add rate limiting"]
Completed: 4 tasks. Blocked: 1.
Changed: src/middleware.py
Summary: Added rate limiting middleware. Blocked on Redis config.

=== END SESSION CONTEXT ===
```

---

## File Structure Reference

```
.forge/
├── .env                    # project-specific API keys and config
├── metadata.json           # forge init statistics
├── plan.json               # active forge auto plan (in-progress)
├── plans/                  # archived completed/cancelled plans
├── summaries/              # forge summarise checkpoints
├── vectors/
│   └── faiss.index         # primary vector store (+ faiss.index.json metadata)
├── project_graph.json      # dependency graph (NetworkX node-link format)
└── memory/                 # persistent cross-session memory
    ├── hot.faiss            # binary FAISS index for memories
    ├── hot.meta.msgpack     # chunk metadata (compressed binary)
    ├── graph.pkl            # project graph binary cache (pickle 5)
    ├── memory.db            # SQLite: memories, sessions, handoffs tables
    ├── handoff.gz           # latest handoff packet (gzipped JSON)
    └── sessions/
        └── <timestamp>.session.gz   # per-session gzipped event log
```

> [!IMPORTANT]
> The entire `.forge/` directory is in `.gitignore` by default. Your API keys and memory are never committed to Git.

---

## Workflow Recipes

### Starting a brand new project
```bash
mkdir my-app && cd my-app
forge setup         # configure keys
forge init          # index (empty dir is fine)
forge auto "build a FastAPI REST API with user auth, SQLite, and pytest tests" --yes
```

### Resuming after a week away
```bash
cd my-app
forge context       # read what was done before
forge recall "what's left to do"
forge auto          # resume the in-progress plan
```

### Handing off to Claude Code / Cursor
```bash
forge handoff --print-prefix
# Copy the printed text → paste into Claude Code's system prompt
```

### Saving an important decision
```bash
forge remember "We chose Redis for session storage because SQLite doesn't support pub/sub" --category decision
```

### Checking what the agent knows before a big task
```bash
forge recall "database schema decisions"
forge recall "security requirements" --category requirement
forge memory-status
```

### Undoing a bad run
```bash
forge rollback     # lists all .forge_backup files, pick what to restore
```
