# forge — Local Coding Agent

<p align="center">
  <a href="https://pypi.org/project/forge-coder/">
    <img src="https://img.shields.io/pypi/v/forge-coder?color=blue&label=PyPI" alt="PyPI Version">
  </a>
  <a href="https://github.com/Tushaarxr/forge/actions/workflows/ci.yml">
    <img src="https://github.com/Tushaarxr/forge/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
  <a href="https://pypi.org/project/forge-coder/">
    <img src="https://img.shields.io/pypi/dm/forge-coder?color=green" alt="PyPI Downloads">
  </a>
  <a href="https://github.com/Tushaarxr/forge/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Tushaarxr/forge?color=orange" alt="License">
  </a>
  <a href="https://discord.gg/forge">
    <img src="https://img.shields.io/discord/123456789?color=purple" alt="Discord">
  </a>
</p>

> **Autonomous coding agent that runs locally** — Your own AI developer powered by Gemini (planning) + Qwen3.5 (coding)

## Why Forge?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FORGE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │   USER      │────▶│   MASTER    │────▶│   AUTO      │                  │
│   │   INPUT     │     │   BRAIN     │     │   RUNNER    │                  │
│   └─────────────┘     │  (Gemini)   │     └──────┬──────┘                  │
│                       └──────┬──────┘            │                          │
│                              │                   ▼                          │
│                              │            ┌─────────────┐                  │
│                              │            │   WORKER    │                  │
│                              │            │ (Qwen3.5)   │                  │
│                              │            └──────┬──────┘                  │
│                              │                   │                          │
│                              ▼            ┌────────▼────────┐              │
│                       ┌─────────────┐     │   FEEDBACK     │              │
│                       │  CONTEXT    │     │   LOOP         │              │
│                       │  ENGINE     │◀────│                │              │
│                       └──────┬──────┘     └────────────────┘              │
│                              │                                                 │
│         ┌────────────────────┼────────────────────┐                        │
│         │                    │                    │                        │
│         ▼                    ▼                    ▼                        │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                 │
│  │  VECTOR     │     │   PROJECT    │     │  PERSISTENT │                 │
│  │  STORE      │     │   GRAPH      │     │  MEMORY    │                 │
│  │  (FAISS)    │     │  (NetworkX)  │     │             │                 │
│  └─────────────┘     └─────────────┘     └─────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dual-LLM Architecture

| Component | Model | Purpose |
|-----------|-------|---------|
| **Master Brain** | Gemini 2.5 Flash | High-level planning, code review, task batching |
| **Local Worker** | Qwen 3.5 (via LM Studio) | Fast iterative code execution |

## Features

### 🚀 Performance
- **Query Caching**: LRU cache (128 entries) for repeated semantic searches
- **HTTP Connection Pooling**: Reused connections in Brain and Worker for reduced latency
- **Parallel Indexing**: ThreadPoolExecutor for indexing large files (>1KB) in parallel

### 🛡️ Reliability
- **Custom Exceptions**: Specific error types (`BrainError`, `WorkerError`, etc.) for better debugging
- **Retry with Backoff**: Exponential backoff for transient network failures
- **Graceful Degradation**: Components work independently when optional dependencies fail
- **Checkpoint System**: Pause every N tasks for human review and rollback

### ⚙️ Configuration
- **Pydantic Validation**: Type-safe configuration with validation
- **Environment Validation**: Clear error messages for missing required settings

### 🧠 Memory System
- **Cross-Session Memory**: Persistent memory that survives between sessions
- **Session Logger**: Track all operations with compression
- **Handoff Packets**: Generate context packets for handoff to other agents

## Install

```bash
# Quick install (Linux/macOS)
curl -fsSL https://raw.githubusercontent.com/Tushaarxr/forge/main/install.sh | bash

# Quick install (Windows PowerShell)
irm https://raw.githubusercontent.com/Tushaarxr/forge/main/install.ps1 | iex

# Or with pipx
pipx install forge-coder

# Or from source
git clone https://github.com/Tushaarxr/forge.git
cd forge
pip install -e ".[dev]"
```

### With Docker

```bash
# Build and run
docker build -t forge-agent .
docker run -it -e GEMINI_API_KEY=your_key -v $(pwd):/workspace forge-agent auto "build a todo app"

# Or with docker-compose
GEMINI_API_KEY=your_key docker-compose run forge auto "build a todo app"
```

## Quickstart (3 steps)

```bash
# 1. One-time setup
forge setup

# 2. Navigate to your project
cd my-project/

# 3. Let Forge build it!
forge auto "build a FastAPI todo app with SQLite and JWT auth"
```

## Commands

| Command | Description |
|---------|-------------|
| `forge setup` | One-time wizard: API keys + LM Studio check |
| `forge init` | Index files into vector store + build dependency graph |
| `forge run` | Interactive mode with feedback loop |
| `forge auto` | Autonomous end-to-end build with checkpoints |
| `forge chat` | Interactive REPL for incremental changes |
| `forge status` | Show project status and metrics |
| `forge init` | Initialize project context |
| `forge remember` | Save important info to memory |
| `forge recall` | Query from persistent memory |
| `forge handoff` | Generate cross-agent context packet |

## Requirements

- **Python** 3.10+
- **LM Studio** (free): https://lmstudio.ai — runs Qwen3.5-9B locally
- **Gemini API Key** (free): https://aistudio.google.com

## Architecture Deep Dive

```
User Goal
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│                      MASTER BRAIN (Gemini)                     │
│  • Breaks goal into ordered task list                          │
│  • Batch reviews completed work                                │
│  • Provides high-level planning and review                    │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│                      AUTO RUNNER                               │
│  • Iterates through task queue                                │
│  • Pauses every N tasks for checkpoint                        │
│  • Handles retry logic and dependency tracking                │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│                      LOCAL WORKER (Qwen3.5)                    │
│  • Executes code in real-time                                  │
│  • Parses <<<FILE:>>> and <<<PATCH:>>> blocks                  │
│  • Returns file changes + patch modifications                 │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│                      FEEDBACK LOOP                             │
│  • Applies changes with .forge_backup                         │
│  • Handles rollback on failure                                │
│  • Integrates with vector store + project graph              │
└────────────────────────────────────────────────────────────────┘
```

### Key Technologies

| Technology | Purpose |
|------------|---------|
| **FAISS** | Vector similarity search for semantic code lookup |
| **NetworkX** | Project dependency graph (imports, functions, classes) |
| **SentenceTransformers** | Code embeddings for semantic search |
| **Pydantic** | Type-safe configuration validation |
| **httpx** | Async HTTP client with connection pooling |

## Documentation

- **[Prerequisites](PREREQUISITES.md)** — Before you start (LM Studio, API keys)
- **[User Manual](forge_user_manual.md)** — Command reference and workflows
- **[Changelog](CHANGELOG.md)** — Version history and release notes
- **[CLAUDE.md](CLAUDE.md)** — For AI assistants working on this repo

## Contributing

```bash
# Clone and setup
git clone https://github.com/Tushaarxr/forge.git
cd forge

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with debug logging
FORGE_LOG_LEVEL=DEBUG forge auto "your goal"
```

## License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with 💻 by developers, for developers</strong>
</p>