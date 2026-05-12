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
def app() -> None:
    """Forge — Local coding agent with Gemini planning and LM Studio execution."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# forge init
# ══════════════════════════════════════════════════════════════════════════════

@app.command(name="init")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def init_command(verbose: bool) -> None:
    """Initialize forge: index files into vector store and build dependency graph."""
    from src.vector_store import VectorStore
    from src.project_graph import ProjectGraph

    base_dir = Path.cwd()
    forge_dir = base_dir / ".forge"
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
@click.argument("goal")
@click.option("--file", "-f", "active_file", help="Active file to focus on")
@click.option("--iterations", "-i", default=3, type=int, help="Maximum iterations per subtask")
@click.option("--dry-run", is_flag=True, help="Show plan only, do not execute")
def run_command(goal: str, active_file: str | None, iterations: int, dry_run: bool) -> None:
    """Run the forge coding agent on a goal."""
    if not dry_run and not _require_init():
        return

    from src.brain import Brain
    from src.worker import Worker
    from src.context_engine import ContextEngine
    from src.vector_store import VectorStore
    from src.project_graph import ProjectGraph
    from src.feedback import FeedbackLoop
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
def chat_command() -> None:
    """Interactive REPL: chat with the coding agent."""
    if not _require_init():
        return

    from src.brain import Brain
    from src.worker import Worker
    from src.context_engine import ContextEngine
    from src.vector_store import VectorStore
    from src.project_graph import ProjectGraph
    from src.feedback import FeedbackLoop
    from src.summariser import Summariser

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
def status_command() -> None:
    """Show current project status and metrics."""
    from src.vector_store import VectorStore
    from src.project_graph import ProjectGraph
    from src.worker import Worker

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
@click.option("--force", is_flag=True, help="Force checkpoint even if no recent changes")
def summarise_command(force: bool) -> None:
    """Create a checkpoint summary of recent changes."""
    if not _require_init():
        return

    from src.summariser import Summariser
    from src.brain import Brain
    from src.vector_store import VectorStore
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
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
