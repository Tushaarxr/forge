# Forge: Local Coding Agent CLI

Forge is a powerful, local-first coding assistant CLI designed to act as an autonomous software engineer directly in your terminal. It leverages a "Master Brain" (Gemini/Anthropic) for high-level reasoning and planning, while delegating the actual code generation to a fast, locally hosted worker model (via LM Studio).

Forge understands your entire project context using a hybrid retrieval engine that combines:
1. **Semantic Vector Search** (FAISS)
2. **Project Dependency Graphs** (NetworkX)

## Prerequisites

Before starting, ensure you have:
1. Python 3.10+ installed.
2. **LM Studio** installed and running locally on port `1234` with a model loaded (e.g., `qwen3.5-9b-instruct` or `qwen2.5-coder`).
3. A **Gemini API Key** (for the Master Brain).

## Installation

1. Clone or download the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install the dependencies in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Configuration

Forge uses environment variables for configuration. You can set these globally, or place a `.env` file inside your project's `.forge/` directory.

Copy the provided example to get started:
```bash
mkdir .forge
cp config/.env.example .forge/.env
```

**Key Environment Variables:**
- `GEMINI_API_KEY`: Your Google Gemini API key.
- `MASTER_MODEL`: The models to use for planning (supports comma-separated fallbacks, e.g., `gemini-2.5-flash,gemini-2.0-flash`).
- `LM_STUDIO_BASE_URL`: URL to your local LM Studio instance (default: `http://localhost:1234`).
- `LOCAL_MODEL`: The name of the model loaded in LM Studio.

## User Manual: CLI Commands

Once installed, you can use the `forge` command from anywhere within an initialized project.

### 1. Initialize a Project
```bash
forge init
```
**What it does:** Scans your project directory, parses your code into a dependency graph, and creates a FAISS vector index. This gives Forge its context.
*Note: You must run this command first in any new project directory.*

### 2. Check Status
```bash
forge status
```
**What it does:** Displays a dashboard showing the health of your vector store, project graph, LM Studio connection, and the Master Brain API.

### 3. Run a Task
```bash
forge run "add a divide(a, b) function to utils.py"
```
**Options:**
- `--file`, `-f`: Specify an active file to prioritize context for.
- `--dry-run`: View the execution plan without actually modifying any code.

**What it does:** 
1. **Plans:** The Master Brain creates a step-by-step plan based on your goal.
2. **Executes:** The local LM Studio worker writes the code for each step.
3. **Reviews:** The Master Brain reviews the generated code for correctness.
4. **Applies:** Before making changes, Forge creates a `.forge_backup` file, then prompts you to keep or reject the changes.

### 4. Interactive Chat
```bash
forge chat
```
**What it does:** Opens an interactive REPL session where you can converse with Forge continuously. 
Inside the chat, you can type goals, or use special slash commands:
- `/status` — Show session statistics.
- `/summarise` — View learnings from the current session.
- `/clear` — Wipe the current session memory.
- `/quit` — Exit the chat.

### 5. Rollback Changes
```bash
forge rollback
```
**What it does:** If you realize a change made by Forge broke something, run this command. It finds all `.forge_backup` files and lets you safely revert the affected files back to their original state.

### 6. Create Checkpoints
```bash
forge summarise
```
**What it does:** Summarizes recent changes made by the agent and creates a human-readable entry in `.forge/CHANGELOG.md`. This is useful for keeping track of what the AI has accomplished over multiple runs.

## Typical Workflow

1. **Start LM Studio**: Ensure your local worker model is running.
2. **Navigate to your code**: `cd my_project/`
3. **Initialize**: `forge init`
4. **Plan a change**: `forge run "refactor the database layer" --dry-run`
5. **Execute**: `forge run "refactor the database layer"`
6. **Review**: Check the file diffs presented in the terminal and hit `y` to apply, or `r` to rollback!
