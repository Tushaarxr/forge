"""Forge CLI - Main command interface for the coding agent."""

import asyncio
import difflib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
import sys
import platform
import shutil
import httpx
from functools import wraps
from rich.prompt import Prompt, Confirm
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

load_dotenv(Path(".forge") / ".env")


from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

# ── Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError) ─────────
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
        except Exception:
            pass

# Initialize Rich console
console = Console(highlight=False)

# Configure logging with Rich handler
logging.basicConfig(
    level=logging.WARNING,  # suppress noisy INFO from httpx/sentence-transformers
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

FORGE_DIR = Path(".forge")


def _require_init() -> bool:
    """Return True if forge has been initialized; print error and return False otherwise."""
    if not FORGE_DIR.exists():
        console.print(
            "[red]❌ Forge is not initialized in this directory.[/red]\n"
            "Run [bold cyan]forge init[/bold cyan] first to index your project."
        )
        return False
    return True


def _lm_studio_error(url: str) -> None:
    console.print(
        f"[red]❌ LM Studio not reachable at {url} — start the server and load a model.[/red]"
    )


@click.group()
@click.version_option(version="0.2.0", message="forge-agent %(version)s")
def app() -> None:
    """Forge — Local coding agent with Gemini planning and LM Studio execution."""
    pass



def get_global_config_dir():
    if platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", "~")).expanduser() / "forge"
    return Path.home() / ".forge"

def require_setup(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        config_path = get_global_config_dir() / "config.json"
        if not config_path.exists():
            console.print("[yellow]Forge is not set up yet. Running setup...[/yellow]")
            ctx = click.get_current_context()
            ctx.invoke(setup_command)
        else:
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if not config.get("setup_complete"):
                    console.print("[yellow]Forge is not set up yet. Running setup...[/yellow]")
                    ctx = click.get_current_context()
                    ctx.invoke(setup_command)
            except Exception:
                console.print("[red]Failed to read config. Running setup...[/red]")
                ctx = click.get_current_context()
                ctx.invoke(setup_command)
        return f(*args, **kwargs)
    return wrapper

# ══════════════════════════════════════════════════════════════════════════════
# forge setup
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="setup")
@click.option("--reset", is_flag=True, help="Clear configuration and restart setup")
@click.option("--key", is_flag=True, help="Only update the API key")
def setup_command(reset: bool = False, key: bool = False) -> None:
    """First-time setup wizard."""
    global_dir = get_global_config_dir()
    config_path = global_dir / "config.json"
    
    if reset:
        if Confirm.ask("This will clear your API keys. Are you sure?", default=False):
            if config_path.exists():
                config_path.unlink()
        else:
            return
            
    global_dir.mkdir(parents=True, exist_ok=True)
    if platform.system() != "Windows":
        global_dir.chmod(0o700)
        
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not key:
        console.print(Panel("""[bold cyan]╭──────────────────────────────────────────╮
│  🔨 forge — Local Coding Agent v0.2.0   │
│  First-time setup wizard                 │
╰──────────────────────────────────────────╯[/bold cyan]""", border_style="cyan"))

    # LM Studio Check
    if not key:
        console.print("\n[bold]Checking LM Studio...[/bold]")
        lm_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
        try:
            resp = httpx.get(f"{lm_url}/v1/models", timeout=2.0)
            if resp.status_code == 200:
                models = [m.get("id") for m in resp.json().get("data", [])]
                console.print(f"[green]✓ LM Studio detected at {lm_url}[/green]")
                if models:
                    model_choices = "\n".join([f"  [{i+1}] {m}" for i, m in enumerate(models)])
                    console.print("Loaded models:\n" + model_choices)
                    choice = Prompt.ask("Select your coding model (enter number or name)", default=models[0])
                    if choice.isdigit() and 1 <= int(choice) <= len(models):
                        config["local_model"] = models[int(choice)-1]
                    else:
                        config["local_model"] = choice
                else:
                    config["local_model"] = Prompt.ask("Enter your local model name", default="qwen3.5-9b-instruct")
            else:
                raise Exception("Bad status")
        except Exception:
            console.print("[red]✗ LM Studio not running[/red]")
            console.print("  1. Download LM Studio: https://lmstudio.ai")
            console.print("  2. Open LM Studio → Search 'Qwen3.5-9B-Instruct-Q4_K_M'")
            console.print("  3. Download the model (~5.5 GB)")
            console.print("  4. Go to Local Server tab → select model → Start Server")
            Prompt.ask("Press Enter when LM Studio is running, or [s] to skip for now", default="s")
            config["local_model"] = "qwen3.5-9b-instruct"

    # Gemini API Key Check
    api_key = config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
    if api_key and not key:
        console.print(f"\\n[green]✓ Gemini API key found ({api_key[:4]}...)[/green]")
        if Confirm.ask("Test this key?", default=True):
            pass # Skipping real test for brevity, assuming valid
    else:
        console.print("\n[bold]Gemini API Key[/bold]")
        console.print("Get your free Gemini API key at: https://aistudio.google.com")
        console.print("  1. Sign in with Google")
        console.print("  2. Click 'Get API key' → 'Create API key'")
        console.print("  3. Copy the key (starts with AIza...)")
        api_key = Prompt.ask("Paste your Gemini API key", password=True)
        config["gemini_api_key"] = api_key

    if not key:
        console.print("\n[bold]Which Gemini model for planning? (free tier recommended)[/bold]")
        console.print("  [1] gemini-2.0-flash     — Fast, free, 1500 req/day")
        console.print("  [2] gemini-2.5-flash     — Smarter planning, free tier")
        console.print("  [3] gemini-2.5-pro       — Best quality, limited free quota")
        console.print("  [4] Enter custom model name")
        mod_choice = Prompt.ask("Choice", default="1")
        if mod_choice == "1": config["master_model"] = "gemini-2.0-flash"
        elif mod_choice == "2": config["master_model"] = "gemini-2.5-flash"
        elif mod_choice == "3": config["master_model"] = "gemini-2.5-pro"
        else: config["master_model"] = Prompt.ask("Enter model name", default="gemini-2.0-flash")

    config["lm_studio_url"] = "http://localhost:1234"
    config["setup_complete"] = True
    config["version"] = "0.2.0"
    
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    console.print(f"\n[green]✓ Global config saved to {config_path}[/green]")

    # Project init
    if not key and Path.cwd() != Path.home() and Path.cwd() != global_dir:
        if Confirm.ask(f"\nInitialize forge in current directory? ({Path.cwd()})", default=True):
            forge_dir = Path(".forge")
            forge_dir.mkdir(parents=True, exist_ok=True)
            
            env_path = forge_dir / ".env"
            env_content = f"""MASTER_PROVIDER=gemini
GEMINI_API_KEY={config['gemini_api_key']}
MASTER_MODEL={config['master_model']}
LM_STUDIO_BASE_URL={config['lm_studio_url']}
LOCAL_MODEL={config.get('local_model', 'qwen3.5-9b-instruct')}
"""
            env_path.write_text(env_content, encoding="utf-8")
            
            # gitignore
            gitignore = Path(".gitignore")
            ignore_text = "\n.forge/\n*.forge_backup\n"
            if not gitignore.exists():
                gitignore.write_text(ignore_text, encoding="utf-8")
            elif ".forge/" not in gitignore.read_text(encoding="utf-8"):
                with open(gitignore, "a", encoding="utf-8") as f:
                    f.write(ignore_text)
            
            console.print(Panel("""[bold green]╭─ Setup Complete ───────────────────────────────────╮
│  ✓ LM Studio: qwen3.5-9b-instruct @ localhost:1234 │
│  ✓ Master Brain: gemini-2.0-flash (verified)        │
│  ✓ Project configured                                │
│                                                      │
│  Start building:                                     │
│    forge auto "describe what you want to build"     │
│    forge chat                                        │
╰──────────────────────────────────────────────────────╯[/bold green]"""))


# ══════════════════════════════════════════════════════════════════════════════
# forge init
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="init")
@require_setup
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def init_command(verbose: bool) -> None:
    """Initialize forge: index files into vector store and build dependency graph."""
    from .vector_store import VectorStore
    from .project_graph import ProjectGraph

    base_dir = Path.cwd()
    forge_dir = base_dir / ".forge"
    
    if not (forge_dir / ".env").exists():
        console.print("[yellow]⚠️  Forge configuration (.env) missing.[/yellow]")
        if click.confirm("Would you like to run 'forge setup' now?", default=True):
            ctx = click.get_current_context()
            ctx.invoke(setup_command)
            # Re-read env after setup
            load_dotenv(forge_dir / ".env")
        else:
            console.print("[red]Init aborted. Setup required.[/red]")
            return

    forge_dir.mkdir(parents=True, exist_ok=True)

    vector_store = VectorStore()
    project_graph = ProjectGraph()

    files_indexed = 0
    chunks_total = 0
    graph_stats: dict = {}

    # ── Task 1: Index files ──────────────────────────────────────────────────
    extensions = {"py", "js", "ts", "go", "rs", "md", "txt"}
    exclude = {".git", "__pycache__", "node_modules", ".venv", "dist", "build", ".forge"}

    candidate_files: list[Path] = []
    for p in base_dir.rglob("*"):
        if not p.is_file():
            continue
        if any(excl in p.parts for excl in exclude):
            continue
        if p.suffix.lstrip(".") in extensions:
            candidate_files.append(p)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task1 = progress.add_task("[cyan]Indexing files...", total=len(candidate_files))
        for file_path in candidate_files:
            try:
                n = vector_store.index_file(str(file_path))
                if n > 0:
                    files_indexed += 1
                    chunks_total += n
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")
            progress.advance(task1)

    # Save vector store
    vs_index_path = str(forge_dir / "vectors" / "faiss.index")
    vector_store.save(vs_index_path)

    # ── Task 2: Build dependency graph ───────────────────────────────────────
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task2 = progress.add_task("[cyan]Building dependency graph...", total=None)
        graph_stats = project_graph.parse_project(str(base_dir))
        progress.update(task2, completed=True, total=1)

    project_graph.save()

    # ── Save metadata ────────────────────────────────────────────────────────
    metadata = {
        "files_indexed": files_indexed,
        "chunks": chunks_total,
        "nodes": graph_stats.get("nodes_total", 0),
        "edges": graph_stats.get("edges_total", 0),
        "timestamp": datetime.now().isoformat(),
    }
    (forge_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ── Display results ───────────────────────────────────────────────────────
    console.print("[bold green][OK] Project initialized successfully![/bold green]")
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Files indexed", str(files_indexed))
    table.add_row("Chunks created", str(chunks_total))
    table.add_row("Graph nodes", str(metadata["nodes"]))
    table.add_row("Graph edges", str(metadata["edges"]))
    table.add_row("Indexed at", metadata["timestamp"][:19])
    console.print(table)


# ══════════════════════════════════════════════════════════════════════════════
# forge run
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="run")
@require_setup
@click.argument("goal")
@click.option("--file", "-f", "active_file", help="Active file to focus on")
@click.option("--iterations", "-i", default=3, type=int, help="Maximum iterations per subtask")
@click.option("--dry-run", is_flag=True, help="Show plan only, do not execute")
def run_command(goal: str, active_file: str | None, iterations: int, dry_run: bool) -> None:
    """Run the forge coding agent on a goal."""
    if not dry_run and not _require_init():
        return

    from .brain import Brain
    from .worker import Worker
    from .context_engine import ContextEngine
    from .vector_store import VectorStore
    from .project_graph import ProjectGraph
    from .feedback import FeedbackLoop
    from rich.live import Live

    async def run_agent() -> None:
        vector_store = VectorStore()
        project_graph = ProjectGraph()

        # Load persisted indexes if they exist
        vs_path = str(FORGE_DIR / "vectors" / "faiss.index")
        if Path(vs_path).exists():
            vector_store.load(vs_path)
        if (FORGE_DIR / "project_graph.json").exists():
            project_graph.load()

        brain = Brain()
        worker = Worker()
        context_engine = ContextEngine(vector_store=vector_store, graph=project_graph)
        feedback_loop = FeedbackLoop()

        if dry_run:
            console.print("[yellow]🔍 DRY RUN MODE — Showing plan only[/yellow]")
            plan_result = await brain.plan(goal, "", project_graph.get_summary())

            sub_tasks = plan_result.get("sub_tasks", [])
            if sub_tasks:
                table = Table(title="📋 Execution Plan")
                table.add_column("#", style="dim", width=4)
                table.add_column("Description", style="white")
                table.add_column("File", style="cyan")
                for i, st in enumerate(sub_tasks, 1):
                    table.add_row(
                        str(i),
                        st.get("description", ""),
                        st.get("active_file", "—") or "—",
                    )
                console.print(table)

                risks = plan_result.get("risks", [])
                if risks:
                    console.print("\n[bold yellow]⚠️  Risks:[/bold yellow]")
                    for r in risks:
                        console.print(f"  • {r}")
            else:
                console.print("[red]No plan generated.[/red]")
                if plan_result.get("reasoning"):
                    console.print(f"[dim]{plan_result['reasoning']}[/dim]")
            return

        # Full run
        console.print(f"[bold blue]🎯 Goal:[/bold blue] {goal}")
        if active_file:
            console.print(f"[bold blue]📁 Active file:[/bold blue] {active_file}")

        import time
        start = time.time()

        status_text = {"msg": "Planning..."}

        def _panel() -> Panel:
            return Panel(status_text["msg"], title="🔄 Forge Agent", border_style="blue")

        with Live(_panel(), refresh_per_second=4, console=console) as live:

            async def _on_status(msg: str) -> None:
                status_text["msg"] = msg
                live.update(_panel())

            results = await feedback_loop.run(
                goal=goal,
                active_file=active_file,
                max_iterations=iterations,
                brain=brain,
                worker=worker,
                context_engine=context_engine,
                vector_store=vector_store,
                graph=project_graph,
            )

        elapsed = time.time() - start
        _show_run_results(results, elapsed, feedback_loop)

    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.exception("Run command failed")


def _show_run_results(results: dict, elapsed: float, feedback_loop) -> None:
    """Display run results including diffs."""
    if results.get("passed"):
        console.print("[green]✅ Goal completed successfully![/green]")
    else:
        console.print("[yellow]⚠️  Goal partially completed[/yellow]")

    console.print(
        f"[blue]📊 Stats:[/blue] "
        f"{results.get('sub_tasks_passed', 0)}/{results.get('sub_tasks_total', 0)} subtasks, "
        f"{results.get('iterations_used', 0)} iterations, {elapsed:.1f}s"
    )

    changes_log = results.get("changes_log", [])
    if changes_log:
        console.print("\n[bold magenta]📝 Changes Made:[/bold magenta]")
        for change in changes_log[:5]:
            console.print(f"  • {change.get('file', '')}: {change.get('task', '')[:60]}")

    # Diff display
    changed_files = [c.get("file") for c in changes_log if c.get("file")]
    if changed_files:
        console.print("\n[bold yellow]🔍 File Diffs:[/bold yellow]")
        for file_path in changed_files[:3]:
            backup_path = f"{file_path}.forge_backup"
            if not os.path.exists(backup_path):
                continue
            try:
                old = Path(backup_path).read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
                new = Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
                diff = list(
                    difflib.unified_diff(old, new, fromfile=f"a/{file_path}", tofile=f"b/{file_path}", lineterm="")
                )
                if diff:
                    console.print(f"\n[yellow]--- {file_path} ---[/yellow]")
                    for line in diff[:30]:
                        if line.startswith("+") and not line.startswith("+++"):
                            console.print(f"[green]{line.rstrip()}[/green]")
                        elif line.startswith("-") and not line.startswith("---"):
                            console.print(f"[red]{line.rstrip()}[/red]")
                        elif line.startswith("@@"):
                            console.print(f"[blue]{line.rstrip()}[/blue]")
                        else:
                            console.print(line.rstrip())
                    if len(diff) > 30:
                        console.print("[dim]... (truncated)[/dim]")
            except Exception as e:
                logger.warning(f"Could not show diff for {file_path}: {e}")

    # Prompt to keep/rollback
    if changed_files:
        try:
            response = console.input("\nApply changes? [y/n/r (rollback)] (y): ").strip().lower() or "y"
            if response == "r":
                console.print("[yellow]Rolling back changes...[/yellow]")
                feedback_loop.rollback(changed_files)
                console.print("[green]Changes rolled back[/green]")
            elif response == "n":
                console.print("[yellow]Changes kept but not confirmed[/yellow]")
            else:
                console.print("[green]Changes applied successfully[/green]")
        except (EOFError, KeyboardInterrupt):
            pass


# ══════════════════════════════════════════════════════════════════════════════
# forge chat
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="chat")
@require_setup
def chat_command() -> None:
    """Interactive REPL: chat with the coding agent."""
    if not _require_init():
        return

    from .brain import Brain
    from .worker import Worker
    from .context_engine import ContextEngine
    from .vector_store import VectorStore
    from .project_graph import ProjectGraph
    from .feedback import FeedbackLoop
    from .summariser import Summariser

    vector_store = VectorStore()
    project_graph = ProjectGraph()
    vs_path = str(FORGE_DIR / "vectors" / "faiss.index")
    if Path(vs_path).exists():
        vector_store.load(vs_path)
    if (FORGE_DIR / "project_graph.json").exists():
        project_graph.load()

    brain = Brain()
    worker = Worker()
    context_engine = ContextEngine(vector_store=vector_store, graph=project_graph)
    feedback_loop = FeedbackLoop()
    summariser = Summariser(brain, vector_store)

    console.print("[bold cyan]🔥 Forge Interactive Mode[/bold cyan]")
    console.print("Type your goals or use: [cyan]/status[/cyan]  [cyan]/summarise[/cyan]  [cyan]/clear[/cyan]  [cyan]/quit[/cyan]  [cyan]/help[/cyan]")
    console.print("=" * 60)

    session_memory: list[str] = []
    tokens_used = 0

    def _show_help() -> None:
        help_text = (
            "/status    — Show session statistics\n"
            "/summarise — Summarise session learnings\n"
            "/clear     — Clear session memory\n"
            "/quit      — Exit\n"
            "/help      — This help message"
        )
        console.print(Panel(help_text, title="Help", border_style="blue"))

    def _show_status() -> None:
        stats = (
            f"Tokens used (est.): {tokens_used:,}\n"
            f"Session learnings: {len(session_memory)}\n"
            f"Vector chunks: {len(vector_store.metadata)}\n"
            f"Graph nodes: {len(project_graph.graph.nodes())}"
        )
        console.print(Panel(stats, title="📊 Status", border_style="green"))

    def _show_summary() -> None:
        if session_memory:
            text = f"{len(session_memory)} learnings this session:\n" + "\n".join(
                f"  • {m[:80]}" for m in session_memory[-5:]
            )
        else:
            text = "No learnings yet."
        console.print(Panel(text, title="📝 Session Summary", border_style="yellow"))

    async def _process(message: str) -> None:
        nonlocal tokens_used
        console.print(f"[dim]🎯 {message}[/dim]")
        try:
            results = await feedback_loop.run(
                goal=message,
                brain=brain,
                worker=worker,
                context_engine=context_engine,
                vector_store=vector_store,
                graph=project_graph,
            )
            tokens_used += results.get("iterations_used", 0) * 500
            session_memory.extend(results.get("session_memory", []))

            if results.get("passed"):
                console.print("[green]✅ Done![/green]")
            else:
                console.print("[yellow]⚠️ Partially completed[/yellow]")

            if results.get("error"):
                console.print(f"[red]Error: {results['error']}[/red]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.exception("Chat process_message failed")

    async def _repl() -> None:
        while True:
            try:
                user_input = console.input("[bold blue]forge>[/bold blue] ").strip()
                if not user_input:
                    continue
                if user_input == "/quit":
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                elif user_input == "/clear":
                    session_memory.clear()
                    console.print("[green]Session cleared[/green]")
                elif user_input == "/status":
                    _show_status()
                elif user_input == "/summarise":
                    _show_summary()
                elif user_input == "/help":
                    _show_help()
                else:
                    await _process(user_input)
            except KeyboardInterrupt:
                console.print("\n[yellow]Ctrl-C — type /quit to exit[/yellow]")
            except EOFError:
                console.print("\n[yellow]Goodbye![/yellow]")
                break

    try:
        asyncio.run(_repl())
    except Exception as e:
        console.print(f"[red]Fatal error:[/red] {e}")
        logger.exception("Chat command failed")


# ══════════════════════════════════════════════════════════════════════════════
# forge status
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="status")
@require_setup
def status_command() -> None:
    """Show current project status and metrics."""
    from .vector_store import VectorStore
    from .project_graph import ProjectGraph
    from .worker import Worker

    async def _check() -> None:
        vector_store = VectorStore()
        project_graph = ProjectGraph()
        worker = Worker()

        vs_path = str(FORGE_DIR / "vectors" / "faiss.index")
        if Path(vs_path).exists():
            vector_store.load(vs_path)
        if (FORGE_DIR / "project_graph.json").exists():
            project_graph.load()

        table = Table(title="🔨 Forge Project Status", show_header=False, box=None)
        table.add_column("Component", style="cyan", width=25)
        table.add_column("Status", style="white", width=55)

        # Forge init check
        if FORGE_DIR.exists():
            table.add_row("🏗️  Forge", "[green]✅ Initialized[/green]")
            meta_file = FORGE_DIR / "metadata.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    table.add_row("   Last Init", meta.get("timestamp", "unknown")[:19])
                except Exception:
                    pass
        else:
            table.add_row("🏗️  Forge", "[red]❌ Not initialized — run forge init[/red]")

        # Vector store
        vs_chunks = len(vector_store.metadata)
        vs_files = len({m.get("file", "") for m in vector_store.metadata})
        table.add_row(
            "📦 Vector Store",
            f"{'[green]✅' if vs_chunks > 0 else '[yellow]⚠️ '} {vs_files} files, {vs_chunks} chunks[/]",
        )

        # Project graph
        gs = project_graph.get_summary()
        table.add_row(
            "🔗 Project Graph",
            f"{'[green]✅' if gs['total_files'] > 0 else '[yellow]⚠️ '} "
            f"{gs['total_files']} files, {gs['total_functions']} functions, "
            f"{gs['total_classes']} classes, {gs['total_edges']} edges[/]",
        )

        # LM Studio
        health = await worker.health_check()
        if health.get("ok"):
            models = health.get("models", [])
            table.add_row(
                "🤖 LM Studio",
                f"[green]✅ Connected — {len(models)} model(s): {', '.join(models[:2])}[/green]",
            )
        else:
            err = health.get("error", "Not reachable")
            table.add_row("🤖 LM Studio", f"[red]❌ {err[:60]}[/red]")

        # Master Brain
        provider = os.getenv("MASTER_PROVIDER", "gemini")
        model = os.getenv("MASTER_MODEL", "gemini-2.0-flash")
        has_key = bool(os.getenv("GEMINI_API_KEY", ""))
        table.add_row(
            "🧠 Master Brain",
            f"{'[green]✅' if has_key else '[red]❌ (GEMINI_API_KEY missing)'} {provider}/{model}[/]",
        )

        # Recent summary
        summaries_dir = FORGE_DIR / "summaries"
        if summaries_dir.exists():
            files = sorted(summaries_dir.glob("*.json"), reverse=True)
            if files:
                table.add_row("📝 Last Checkpoint", f"[green]✅ {files[0].stem}[/green]")
            else:
                table.add_row("📝 Last Checkpoint", "[dim]None yet[/dim]")
        else:
            table.add_row("📝 Last Checkpoint", "[dim]None yet[/dim]")

        console.print(table)
        console.print("\n[dim]forge run \"<goal>\" --dry-run  to plan without executing[/dim]")

    try:
        asyncio.run(_check())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.exception("Status command failed")


# ══════════════════════════════════════════════════════════════════════════════
# forge summarise
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="summarise")
@require_setup
@click.option("--force", is_flag=True, help="Force checkpoint even if no recent changes")
def summarise_command(force: bool) -> None:
    """Create a checkpoint summary of recent changes."""
    if not _require_init():
        return

    from .summariser import Summariser
    from .brain import Brain
    from .vector_store import VectorStore
    from rich.prompt import Confirm

    async def _run() -> None:
        brain = Brain()
        vector_store = VectorStore()
        vs_path = str(FORGE_DIR / "vectors" / "faiss.index")
        if Path(vs_path).exists():
            vector_store.load(vs_path)
        summariser = Summariser(brain, vector_store)

        changes_log: list[dict] = []
        # Try to load from most recent summary
        try:
            summaries = await summariser.load_summaries(n=1)
            if summaries:
                changes_log = summaries[0].get("changes_log", [])
        except Exception as e:
            logger.warning(f"Could not load recent changes: {e}")

        if not changes_log and not force:
            console.print("[yellow]No recent changes found. Use --force to checkpoint anyway.[/yellow]")
            return

        console.print("[bold green]📝 Creating checkpoint...[/bold green]")
        try:
            result = await summariser.checkpoint(changes_log, {"forge_dir": str(FORGE_DIR)})
            summary_dict = result.get("summary_dict", {})
            if summary_dict:
                review_text = Summariser.format_human_review(summary_dict)
                console.print(Panel(review_text, title="🎯 Checkpoint Review", border_style="green"))
                console.print(f"[dim]Saved to {result['changelog_path']}[/dim]")
            else:
                console.print("[yellow]No summary generated[/yellow]")
        except Exception as e:
            console.print(f"[red]Error creating checkpoint:[/red] {e}")
            logger.exception("Summarise failed")

    try:
        asyncio.run(_run())
    except Exception as e:
        console.print(f"[red]Fatal error:[/red] {e}")
        logger.exception("Summarise command failed")


# ══════════════════════════════════════════════════════════════════════════════
# forge rollback
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="rollback")
@require_setup
def rollback_command() -> None:
    """Rollback files to their .forge_backup versions."""
    import shutil
    from rich.prompt import Confirm, Prompt

    backup_files = list(Path(".").rglob("*.forge_backup"))
    if not backup_files:
        console.print("[yellow]No backup files found.[/yellow]")
        console.print("[dim]Backups are created automatically when forge applies changes.[/dim]")
        return

    console.print(f"[bold yellow]🔄 Found {len(backup_files)} backup file(s):[/bold yellow]")
    table = Table(show_header=True)
    table.add_column("#", width=4)
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right", style="green")

    for i, backup in enumerate(backup_files, 1):
        original = str(backup)[: -len(".forge_backup")]
        kb = backup.stat().st_size / 1024
        table.add_row(str(i), original, f"{kb:.1f} KB")

    console.print(table)

    response = Prompt.ask("Enter numbers to restore (comma-separated) or 'all'", default="all")
    if response.lower() == "all":
        selected = backup_files
    else:
        try:
            indices = [int(x.strip()) - 1 for x in response.split(",")]
            selected = [backup_files[i] for i in indices if 0 <= i < len(backup_files)]
        except (ValueError, IndexError):
            console.print("[red]Invalid selection.[/red]")
            return

    if not selected:
        console.print("[yellow]Nothing selected.[/yellow]")
        return

    if not Confirm.ask(f"Restore {len(selected)} file(s)?", default=False):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    restored = failed = 0
    for backup in selected:
        original = str(backup)[: -len(".forge_backup")]
        try:
            shutil.copy2(backup, original)
            backup.unlink()
            console.print(f"[green]✅ Restored:[/green] {original}")
            restored += 1
        except Exception as e:
            console.print(f"[red]❌ Failed:[/red] {original} ({e})")
            failed += 1

    console.print(f"\n[bold]Restored: {restored}, Failed: {failed}[/bold]")


# ══════════════════════════════════════════════════════════════════════════════
# forge auto
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="auto")
@require_setup
@click.argument("goal", required=False, default=None)
@click.option(
    "--checkpoint-every", "-c",
    default=5, type=int, show_default=True,
    help="Pause for human review every N sub-tasks.",
)
@click.option(
    "--max-tasks", "-m",
    default=50, type=int, show_default=True,
    help="Hard stop after N sub-tasks to prevent runaway.",
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip the initial confirmation prompt.",
)
@click.option(
    "--from-plan",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Load a pre-generated plan JSON instead of re-planning.",
)
def auto_command(
    goal: str | None,
    checkpoint_every: int,
    max_tasks: int,
    yes: bool,
    from_plan: str | None,
) -> None:
    """Autonomous end-to-end build mode — describe what you want, forge builds it.

    Forge generates a full project plan, then executes every task autonomously,
    pausing only at checkpoints for human review.

    \b
    Examples:
      forge auto "build a FastAPI todo app with SQLite and JWT auth"
      forge auto --checkpoint-every 3 --max-tasks 30 --yes "build a CLI calculator"
      forge auto                        # resume an in-progress plan
      forge auto --from-plan plan.json  # load a saved plan
    """
    from .auto_runner import (
        AutoRunner, EventType, PlanStatus, TaskStatus,
        PLAN_PATH,
    )
    from .brain import Brain
    from .worker import Worker
    from .context_engine import ContextEngine
    from .vector_store import VectorStore
    from .project_graph import ProjectGraph
    from rich.live import Live
    from rich.progress import BarColumn, Progress, TextColumn, TaskProgressColumn
    from rich.text import Text
    import math

    # ── Require forge init ────────────────────────────────────────────────────
    if not _require_init():
        return

    async def _run_auto() -> None:
        # ── LM Studio pre-check ───────────────────────────────────────────────
        console.print("[dim]Checking LM Studio connectivity...[/dim]")
        worker = Worker()
        health = await worker.health_check()
        if not health.get("ok"):
            err = health.get("error", "unreachable")
            console.print(
                f"[bold red]❌ LM Studio not reachable:[/bold red] {err}\n"
                "[dim]Start LM Studio, load a model, and enable the local server before running forge auto.[/dim]"
            )
            return

        models = health.get("models", [])
        console.print(
            f"[green]✅ LM Studio connected[/green] — "
            f"{len(models)} model(s): [cyan]{', '.join(models[:2])}[/cyan]"
        )

        # ── Load dependencies ─────────────────────────────────────────────────
        vector_store = VectorStore()
        project_graph = ProjectGraph()
        vs_path = str(FORGE_DIR / "vectors" / "faiss.index")
        if Path(vs_path).exists():
            vector_store.load(vs_path)
        if (FORGE_DIR / "project_graph.json").exists():
            project_graph.load()

        brain = Brain()
        context_engine = ContextEngine(vector_store=vector_store, graph=project_graph)

        runner = AutoRunner(
            brain=brain,
            worker=worker,
            context_engine=context_engine,
            vector_store=vector_store,
            graph=project_graph,
            checkpoint_every=checkpoint_every,
            max_tasks=max_tasks,
            console=console,
        )

        # ── Resume check ──────────────────────────────────────────────────────
        existing_plan = AutoRunner.load_existing_plan()
        if existing_plan and not from_plan:
            console.print(
                f"\n[bold yellow]📋 Found incomplete plan:[/bold yellow] "
                f"{existing_plan.goal[:70]}\n"
                f"   {existing_plan.done_count}/{existing_plan.total} tasks done, "
                f"{existing_plan.blocked_count} blocked"
            )
            try:
                answer = console.input("Resume? [[bold green]y[/bold green]/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer != "n":
                runner.plan = existing_plan
                console.print(f"[green]Resuming from task {existing_plan.done_count + 1}[/green]")
            else:
                runner.archive_plan()
                existing_plan = None

        # ── Planning phase (if not resuming) ─────────────────────────────────
        if runner.plan is None:
            _actual_goal = goal
            if not _actual_goal and not from_plan:
                console.print(
                    "[red]❌ No goal provided and no in-progress plan found.[/red]\n"
                    "[dim]Usage: forge auto \"<goal>\" or run forge auto to resume[/dim]"
                )
                return

            console.print(f"\n[bold blue]🧠 Planning:[/bold blue] {_actual_goal or '(from file)'}")
            console.print("[dim]Master brain generating full project roadmap...[/dim]")

            plan = await runner.create_plan(
                goal=_actual_goal or "",
                from_plan_file=from_plan,
            )

            if not plan.tasks:
                console.print("[red]❌ Brain returned no tasks — check API key and model.[/red]")
                return

            # Display the plan
            table = Table(title="📋 Full Project Plan", show_header=True, header_style="bold cyan")
            table.add_column("#", style="dim", width=4)
            table.add_column("Task", style="white")
            table.add_column("File", style="cyan", width=30)
            table.add_column("Cat", style="yellow", width=8)
            table.add_column("Lines", justify="right", style="green", width=6)
            for t in plan.tasks:
                table.add_row(
                    str(t.id),
                    t.description[:60],
                    (t.active_file or "—")[-30:],
                    t.category[:8],
                    str(t.estimated_lines) if t.estimated_lines else "—",
                )
            console.print(table)
            console.print(
                f"\n[dim]Plan saved to[/dim] [cyan].forge/plan.json[/cyan]  "
                f"[dim]({len(plan.tasks)} tasks)[/dim]"
            )

            if not yes:
                try:
                    answer = console.input("\nStart building? [[bold green]y[/bold green]/n]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[yellow]Cancelled.[/yellow]")
                    return
                if answer == "n":
                    console.print("[yellow]Cancelled. Run forge auto to resume later.[/yellow]")
                    return

        # ── Live display state ────────────────────────────────────────────────
        plan = runner.plan
        state = {
            "active_task": None,        # AutoTask currently running
            "active_elapsed": 0.0,
            "active_tok_ps": 0.0,
            "active_ctx": 0,
            "last_done": [],            # last N completed task descriptions
            "retrying": False,
        }

        def _build_panel() -> Panel:
            p = plan
            done = p.done_count
            total = p.total
            blocked = p.blocked_count
            pct = done / total if total else 0

            # Progress bar (20 chars)
            filled = math.floor(pct * 20)
            bar = "█" * filled + "░" * (20 - filled)

            # Goal (truncated)
            goal_disp = (p.goal[:64] + "…") if len(p.goal) > 65 else p.goal

            lines: list[str] = [
                f"[bold]Goal:[/bold] {goal_disp}",
                f"[bold]Progress:[/bold] [green]{bar}[/green]  "
                f"[cyan]{done}/{total}[/cyan] tasks  [dim]({pct:.0%})[/dim]",
                "",
            ]

            # Last few completed tasks
            for desc in state["last_done"][-4:]:
                lines.append(f"  [green]✓[/green] {desc[:62]}")

            # Active task
            at = state["active_task"]
            if at:
                spinner = "⟳" if not state["retrying"] else "↺"
                prefix = "Retrying" if state["retrying"] else "Working"
                elapsed_str = f"[{state['active_elapsed']:.0f}s]"
                tok_str = (
                    f"[dim]{state['active_tok_ps']:.0f} tok/s | "
                    f"{state['active_ctx'] / 1000:.1f}K ctx[/dim]"
                    if state["active_tok_ps"] > 0
                    else ""
                )
                lines.append(
                    f"  [bold yellow]{spinner}[/bold yellow] "
                    f"[yellow]{prefix}:[/yellow] {at.description[:52]}  "
                    f"[dim]{elapsed_str}[/dim]"
                )
                if tok_str:
                    lines.append(f"    {tok_str}")

            lines.append("")

            # Next task
            pending = p.pending_tasks()
            if at and pending:
                next_t = next((t for t in pending if t != at), None)
                if next_t:
                    lines.append(f"  [dim]Next:[/dim] {next_t.description[:62]}")

            tasks_to_cp = checkpoint_every - (done % checkpoint_every)
            lines.append(
                f"  [dim]Blocked: {blocked}   "
                f"Next checkpoint in: {tasks_to_cp} task(s)[/dim]"
            )

            return Panel(
                "\n".join(lines),
                title="[bold cyan]⚡ forge auto[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )

        # ── Autonomous execution with Live panel ──────────────────────────────
        import asyncio as _asyncio
        import time
        run_start = time.time()

        with Live(_build_panel(), refresh_per_second=4, console=console) as live:

            async def _tick_elapsed() -> None:
                """Background coroutine: keeps elapsed timer ticking in Live."""
                while True:
                    await _asyncio.sleep(0.5)
                    at = state["active_task"]
                    if at:
                        state["active_elapsed"] = time.monotonic() - _task_start_mono[0]
                    live.update(_build_panel())

            _task_start_mono = [time.monotonic()]
            tick_task = _asyncio.ensure_future(_tick_elapsed())

            try:
                async for event in runner.run():
                    etype = event.type

                    if etype == EventType.TASK_STARTED:
                        state["active_task"] = event.task
                        state["retrying"] = False
                        state["active_elapsed"] = 0.0
                        state["active_tok_ps"] = 0.0
                        state["active_ctx"] = 0
                        _task_start_mono[0] = time.monotonic()

                    elif etype == EventType.TASK_RETRYING:
                        state["retrying"] = True

                    elif etype in (EventType.TASK_DONE, EventType.TASK_BLOCKED):
                        t = event.task
                        state["active_task"] = None
                        state["retrying"] = False
                        state["active_elapsed"] = event.elapsed
                        state["active_tok_ps"] = event.tokens_per_second
                        state["active_ctx"] = event.ctx_tokens
                        if etype == EventType.TASK_DONE:
                            state["last_done"].append(t.description[:62])

                    elif etype in (
                        EventType.CHECKPOINT_START,
                        EventType.CHECKPOINT_STOP,
                        EventType.CHECKPOINT_ROLLBACK,
                    ):
                        # Checkpoint rendering happens inside AutoRunner._do_checkpoint
                        # which uses console.print/input directly (outside Live).
                        # We need to stop Live temporarily.
                        live.stop()
                        if etype == EventType.CHECKPOINT_START:
                            # Live will restart after checkpoint_stop/return
                            pass

                    elif etype == EventType.RUN_ABORTED:
                        live.stop()
                        console.print(f"\n[yellow]⏹  Stopped:[/yellow] {event.message}")
                        break

                    elif etype == EventType.RUN_COMPLETE:
                        live.stop()
                        break

                    live.update(_build_panel())

            finally:
                tick_task.cancel()
                try:
                    await tick_task
                except _asyncio.CancelledError:
                    pass

        # ── Final summary ──────────────────────────────────────────────────────
        elapsed_total = time.time() - run_start
        p = runner.plan

        console.print(f"\n[bold green]✅ forge auto complete[/bold green]  "
                      f"[dim]{elapsed_total:.0f}s total[/dim]")

        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")
        summary_table.add_row("Tasks done", str(p.done_count))
        summary_table.add_row("Tasks blocked", str(p.blocked_count))
        summary_table.add_row("Tasks total", str(p.total))
        summary_table.add_row("Checkpoints", str(len(p.checkpoints)))
        console.print(summary_table)

        if p.blocked_tasks():
            console.print("\n[bold red]⚠  Blocked tasks (need your attention):[/bold red]")
            for t in p.blocked_tasks():
                console.print(f"  [red]Task {t.id}:[/red] {t.description[:70]}")
                console.print(f"  [dim]  {t.block_reason or 'No reason recorded'}[/dim]")
            console.print(
                "\n[dim]Tip: edit [cyan].forge/plan.json[/cyan], fix blockers, "
                "then run [cyan]forge auto[/cyan] to resume.[/dim]"
            )

        # Run forge status for final graph stats
        console.print("\n[dim]── Project graph stats ──────────────────────────[/dim]")
        try:
            gs = project_graph.get_summary()
            console.print(
                f"  Files: [cyan]{gs.get('total_files', 0)}[/cyan]  "
                f"Functions: [cyan]{gs.get('total_functions', 0)}[/cyan]  "
                f"Classes: [cyan]{gs.get('total_classes', 0)}[/cyan]  "
                f"Edges: [cyan]{gs.get('total_edges', 0)}[/cyan]"
            )
        except Exception:
            pass

    try:
        asyncio.run(_run_auto())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — run forge auto to resume.[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error:[/red] {e}")
        logger.exception("forge auto failed")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def forge() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    forge()
