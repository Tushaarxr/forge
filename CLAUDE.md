# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Forge is a local-first autonomous coding agent CLI that uses a dual-LLM architecture:
- **Master Brain** (Gemini 2.5 Flash): High-level project planning, code review, and task batch evaluation
- **Local Worker** (Qwen 3.5 via LM Studio): Fast, iterative code execution and file manipulation

The system includes persistent memory for cross-session context, FAISS vector search for semantic code lookup, and NetworkX-based dependency graph tracking.

## Common Commands

```bash
# Development setup
pip install -e ".[dev]"
pytest tests/

# Run a single test
pytest tests/test_core.py::test_function_name -v

# Project workflow
forge setup         # First-time configuration (API keys, model selection)
forge init          # Index project files into vector store + build dependency graph
forge auto "goal"  # Autonomous build mode with checkpoint pauses
forge run "goal"   # Interactive mode with feedback loop
forge chat         # Interactive REPL for incremental changes

# Memory & context
forge remember "text" --category note|requirement|decision|error|code
forge recall "query" --top-k 5
forge memory-status
forge handoff       # Generate cross-agent context packet
forge context       # Print session context for LLM injection
```

## Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| CLI Entry | `cli.py` | Click-based command interface, all `forge` subcommands |
| Master Brain | `brain.py` | Gemini/Anthropic API calls for planning, review, summarization |
| Local Worker | `worker.py` | LM Studio API client, parses `<<<FILE:>>>` and `<<<PATCH:>>>` blocks |
| Auto Runner | `auto_runner.py` | Autonomous plan-execute-review loop with checkpoints, batch review |
| Context Engine | `context_engine.py` | Retrieves relevant code context from vector store + graph |
| Vector Store | `vector_store.py` | FAISS index for semantic code search |
| Project Graph | `project_graph.py` | NetworkX dependency graph (imports, functions, classes) |
| Persistent Memory | `persistent_memory.py` | Cross-session memory with FAISS + decay scoring |
| Feedback Loop | `feedback.py` | Applies file changes, handles rollback |

### Data Flow

1. **Planning**: `brain.plan_full_project()` generates ordered task list
2. **Execution**: `auto_runner` iterates tasks, calls `worker.execute()` for each
3. **Output Parsing**: Worker returns `file_changes` (new files) and `patch_changes` (surgical edits)
4. **Review**: Every N tasks, `brain.batch_review()` evaluates work
5. **Checkpoint**: User reviews progress, can rollback or continue
6. **Persistence**: State saved to `.forge/plan.json` for resume capability

### Output Formats (Worker)

The worker expects these formats in LLM responses:

```
<<<FILE: path/to/file>>>
<complete file content>
<<<END FILE>>>

<<<PATCH: path/to/file>>>
<<<FIND>>
<exact lines to replace>
<<<REPLACE>>
<new lines>
<<<END PATCH>>>
```

### Configuration

- Global config: `~/.config/forge/config.json` (or `%APPDATA%/forge/config.json` on Windows)
- Project config: `.forge/.env` - API keys, model names, LM Studio URL
- Plan state: `.forge/plan.json` - persisted execution state

Environment variables:
- `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` - Master brain
- `MASTER_MODEL` - Defaults to "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-3.1-flash-lite"
- `LOCAL_MODEL` - Local model name (e.g., "qwen3.5-9b-instruct")
- `LM_STUDIO_BASE_URL` - Defaults to http://localhost:1234

## Key Design Patterns

- **Async throughout**: All LLM calls use `httpx.AsyncClient`
- **Lazy initialization**: Vector store, brain, worker created on-demand
- **Checkpoint-based autonomy**: User pauses every N tasks for review
- **Batch review**: Multiple tasks reviewed in single API call to save quota
- **Fallback models**: If one Gemini model hits quota, tries next in comma-separated list

## Testing

Tests use pytest with `pytest-asyncio`. Run with:
```bash
pytest tests/ -v
```

Key test files: `test_core.py`, `test_auto_runner.py`, `test_plan_parser.py`, `test_memory_system.py`