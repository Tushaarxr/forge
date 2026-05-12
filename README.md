# forge — Local Coding Agent

Local-first autonomous coding agent CLI.

## Install

**One line (Linux/macOS)**
```bash
curl -fsSL https://raw.githubusercontent.com/Tushaarxr/forge/main/install.sh | bash
```

**One line (Windows PowerShell)**
```powershell
irm https://raw.githubusercontent.com/Tushaarxr/forge/main/install.ps1 | iex
```

**Or with pipx**
```bash
pipx install forge-agent
```

**Or with Docker**
```bash
docker build -t forge-agent .
docker run -it -e GEMINI_API_KEY=AIza... -v $(pwd):/workspace forge-agent auto "build a todo app"

# With docker-compose
GEMINI_API_KEY=AIza... docker-compose run forge auto "build a todo app"
```

## Quickstart (3 steps)
```bash
forge setup         # one-time wizard: API keys + LM Studio check
cd my-project/
forge auto "build a FastAPI todo app with SQLite and JWT auth"
```

## Commands

| Command | Description |
|---|---|
| `forge setup` | One-time setup: Configure API keys and create environment files. |
| `forge init` | Initialize forge: index files into vector store and build dependency graph. |
| `forge run` | Run the forge coding agent on a goal. |
| `forge auto` | Autonomous end-to-end build mode. |
| `forge chat` | Interactive REPL: chat with the coding agent. |
| `forge status` | Show current project status and metrics. |
| `forge summarise` | Create a checkpoint summary of recent changes. |
| `forge rollback` | Rollback files to their `.forge_backup` versions. |

## Requirements
- Python 3.10+
- LM Studio (free): https://lmstudio.ai — runs Qwen3.5-9B locally
- Gemini API key (free): https://aistudio.google.com

## Architecture
Forge leverages a "Master Brain" (Gemini) for high-level project planning and a "Local Worker" (LM Studio / Qwen3.5) for fast, iterative code execution. It uses FAISS for vector search and NetworkX to map your project's dependency graph.

## Contributing / Development setup

```bash
git clone https://github.com/Tushaarxr/forge.git
cd forge
pip install -e ".[dev]"
pytest tests/
```
