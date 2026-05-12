"""AutoRunner — autonomous plan-execute-review loop for `forge auto`."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

FORGE_DIR = Path(".forge")
PLAN_PATH = FORGE_DIR / "plan.json"
PLANS_ARCHIVE_DIR = FORGE_DIR / "plans"


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class PlanStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


@dataclass
class AutoTask:
    id: int
    description: str
    active_file: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    review_score: int = 0
    files_changed: list[str] = field(default_factory=list)
    block_reason: str | None = None
    category: str = "logic"
    estimated_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "active_file": self.active_file,
            "status": self.status.value,
            "review_score": self.review_score,
            "files_changed": self.files_changed,
            "block_reason": self.block_reason,
            "category": self.category,
            "estimated_lines": self.estimated_lines,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AutoTask":
        return cls(
            id=d.get("id", 0),
            description=d.get("description", ""),
            active_file=d.get("active_file"),
            status=TaskStatus(d.get("status", "pending")),
            review_score=d.get("review_score", 0),
            files_changed=d.get("files_changed", []),
            block_reason=d.get("block_reason"),
            category=d.get("category", "logic"),
            estimated_lines=d.get("estimated_lines", 0),
        )


@dataclass
class CheckpointRecord:
    after_task: int
    timestamp: str
    summary: str
    # Files changed since last checkpoint (for rollback)
    files_backed_up: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "after_task": self.after_task,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "files_backed_up": self.files_backed_up,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointRecord":
        return cls(
            after_task=d.get("after_task", 0),
            timestamp=d.get("timestamp", ""),
            summary=d.get("summary", ""),
            files_backed_up=d.get("files_backed_up", []),
        )


@dataclass
class AutoPlan:
    goal: str
    created_at: str
    status: PlanStatus = PlanStatus.PENDING
    tasks: list[AutoTask] = field(default_factory=list)
    checkpoints: list[CheckpointRecord] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def done_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.DONE)

    @property
    def blocked_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.BLOCKED)

    def pending_tasks(self) -> list[AutoTask]:
        """Return tasks that still need execution (pending + blocked on prior run)."""
        return [t for t in self.tasks if t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)]

    def blocked_tasks(self) -> list[AutoTask]:
        return [t for t in self.tasks if t.status == TaskStatus.BLOCKED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "created_at": self.created_at,
            "status": self.status.value,
            "tasks": [t.to_dict() for t in self.tasks],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AutoPlan":
        plan = cls(
            goal=d.get("goal", ""),
            created_at=d.get("created_at", datetime.now().isoformat()),
            status=PlanStatus(d.get("status", "pending")),
        )
        plan.tasks = [AutoTask.from_dict(t) for t in d.get("tasks", [])]
        plan.checkpoints = [CheckpointRecord.from_dict(c) for c in d.get("checkpoints", [])]
        return plan


# ══════════════════════════════════════════════════════════════════════════════
# Events (yielded to CLI for Live display updates)
# ══════════════════════════════════════════════════════════════════════════════

class EventType(str, Enum):
    TASK_STARTED = "task_started"
    TASK_DONE = "task_done"
    TASK_BLOCKED = "task_blocked"
    TASK_RETRYING = "task_retrying"
    CHECKPOINT_START = "checkpoint_start"
    CHECKPOINT_STOP = "checkpoint_stop"         # user chose [s]
    CHECKPOINT_ROLLBACK = "checkpoint_rollback"
    RUN_COMPLETE = "run_complete"
    RUN_ABORTED = "run_aborted"


@dataclass
class RunEvent:
    type: EventType
    task: AutoTask | None = None
    plan: AutoPlan | None = None
    message: str = ""
    elapsed: float = 0.0
    tokens_per_second: float = 0.0
    ctx_tokens: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# AutoRunner
# ══════════════════════════════════════════════════════════════════════════════

class AutoRunner:
    """Autonomous execution engine for `forge auto`.

    Usage::
        runner = AutoRunner(brain, worker, context_engine, vector_store, graph,
                            checkpoint_every=5, max_tasks=50, console=console)
        async for event in runner.run():
            update_live_display(event)
    """

    def __init__(
        self,
        brain,
        worker,
        context_engine,
        vector_store,
        graph,
        checkpoint_every: int = 5,
        max_tasks: int = 50,
        console=None,
    ) -> None:
        self.brain = brain
        self.worker = worker
        self.context_engine = context_engine
        self.vector_store = vector_store
        self.graph = graph
        self.checkpoint_every = checkpoint_every
        self.max_tasks = max_tasks
        self.console = console

        self.plan: AutoPlan | None = None
        # Track .forge_backup files created in the current checkpoint window
        self._checkpoint_backups: list[str] = []
        # Reuse FeedbackLoop's apply/rollback helpers
        from .feedback import FeedbackLoop
        self._feedback = FeedbackLoop()

    # ── Plan I/O ──────────────────────────────────────────────────────────────

    def save_plan(self) -> None:
        """Persist current plan state to .forge/plan.json after every status change."""
        if self.plan is None:
            return
        FORGE_DIR.mkdir(parents=True, exist_ok=True)
        PLAN_PATH.write_text(
            json.dumps(self.plan.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def plan_exists() -> bool:
        return PLAN_PATH.exists()

    @staticmethod
    def load_existing_plan() -> AutoPlan | None:
        """Load plan.json if it exists and status is in_progress."""
        if not PLAN_PATH.exists():
            return None
        try:
            data = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
            plan = AutoPlan.from_dict(data)
            if plan.status == PlanStatus.IN_PROGRESS:
                return plan
        except Exception as e:
            logger.warning(f"Could not load plan.json: {e}")
        return None

    def archive_plan(self) -> None:
        """Move current plan.json to .forge/plans/<timestamp>.json."""
        if not PLAN_PATH.exists():
            return
        PLANS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = PLANS_ARCHIVE_DIR / f"{ts}.json"
        shutil.move(str(PLAN_PATH), str(dest))
        logger.info(f"Archived old plan to {dest}")

    async def create_plan(
        self,
        goal: str,
        from_plan_file: str | None = None,
    ) -> AutoPlan:
        """Create a fresh plan from goal or load from a file path."""
        if from_plan_file:
            data = json.loads(Path(from_plan_file).read_text(encoding="utf-8"))
            plan = AutoPlan.from_dict(data)
            plan.goal = goal or plan.goal
            self.plan = plan
            self.save_plan()
            return plan

        project_summary: dict[str, Any] = {}
        if self.graph:
            project_summary = self.graph.get_summary()

        tasks_raw = await self.brain.plan_full_project(goal, project_summary)

        plan = AutoPlan(
            goal=goal,
            created_at=datetime.now().isoformat(),
            status=PlanStatus.IN_PROGRESS,
        )
        plan.tasks = [
            AutoTask(
                id=t.get("id", i + 1),
                description=t.get("description", ""),
                active_file=t.get("active_file"),
                category=t.get("category", "logic"),
                estimated_lines=t.get("estimated_lines", 0),
            )
            for i, t in enumerate(tasks_raw)
        ]
        self.plan = plan
        self.save_plan()
        return plan

    # ── Main autonomous loop ──────────────────────────────────────────────────

    async def run(self) -> AsyncIterator[RunEvent]:
        """Async generator yielding RunEvent objects throughout execution.

        Drives:
          - task pickup (pending → in_progress)
          - execute with one auto-retry
          - silent review
          - checkpoint pause every N tasks
          - hard stop at max_tasks
        """
        assert self.plan is not None, "Call create_plan() before run()"

        plan = self.plan
        plan.status = PlanStatus.IN_PROGRESS
        self.save_plan()

        tasks_since_checkpoint = 0
        total_executed = 0

        pending = plan.pending_tasks()

        for task in pending:
            if total_executed >= self.max_tasks:
                break

            # ── Mark in_progress ────────────────────────────────────────────
            task.status = TaskStatus.IN_PROGRESS
            self.save_plan()
            t_start = time.monotonic()
            yield RunEvent(type=EventType.TASK_STARTED, task=task, plan=plan)

            # ── Execute (with one retry) ─────────────────────────────────────
            success, tok_ps, ctx_tok = await self._execute_task(task, attempt=0)

            if not success:
                yield RunEvent(
                    type=EventType.TASK_RETRYING,
                    task=task,
                    plan=plan,
                    message="First attempt failed — auto-retrying with improved prompt",
                )
                success, tok_ps, ctx_tok = await self._execute_task(task, attempt=1)

            elapsed = time.monotonic() - t_start

            if success:
                task.status = TaskStatus.DONE
            else:
                task.status = TaskStatus.BLOCKED
                self.save_plan()
                yield RunEvent(
                    type=EventType.TASK_BLOCKED,
                    task=task,
                    plan=plan,
                    elapsed=elapsed,
                    tokens_per_second=tok_ps,
                    ctx_tokens=ctx_tok,
                )
                tasks_since_checkpoint += 1
                total_executed += 1
                if tasks_since_checkpoint >= self.checkpoint_every:
                    stop = False
                    async for ev in self._do_checkpoint(tasks_since_checkpoint):
                        yield ev
                        if ev.type == EventType.CHECKPOINT_STOP:
                            stop = True
                    if stop:
                        yield RunEvent(type=EventType.RUN_ABORTED, plan=plan,
                                       message="User stopped at checkpoint")
                        return
                    tasks_since_checkpoint = 0
                continue

            self.save_plan()
            yield RunEvent(
                type=EventType.TASK_DONE,
                task=task,
                plan=plan,
                elapsed=elapsed,
                tokens_per_second=tok_ps,
                ctx_tokens=ctx_tok,
            )

            tasks_since_checkpoint += 1
            total_executed += 1

            # ── Checkpoint? ─────────────────────────────────────────────────
            if tasks_since_checkpoint >= self.checkpoint_every:
                stop = False
                async for ev in self._do_checkpoint(tasks_since_checkpoint):
                    yield ev
                    if ev.type == EventType.CHECKPOINT_STOP:
                        stop = True
                if stop:
                    yield RunEvent(type=EventType.RUN_ABORTED, plan=plan,
                                   message="User stopped at checkpoint")
                    return
                tasks_since_checkpoint = 0

        # ── Final state ───────────────────────────────────────────────────────
        plan.status = PlanStatus.DONE if plan.blocked_count == 0 else PlanStatus.IN_PROGRESS
        self.save_plan()

        # Final checkpoint / summary
        async for ev in self._do_checkpoint(tasks_since_checkpoint, is_final=True):
            yield ev
            if ev.type == EventType.CHECKPOINT_STOP:
                break

        yield RunEvent(type=EventType.RUN_COMPLETE, plan=plan)

    # ── Single task execution ─────────────────────────────────────────────────

    async def _execute_task(
        self,
        task: AutoTask,
        attempt: int = 0,
    ) -> tuple[bool, float, int]:
        """Execute one task. Returns (success, tokens_per_second, ctx_tokens).

        On attempt=1 the task.description should already have been replaced
        with brain's retry_prompt.
        """
        # Retrieve context
        task_context = ""
        if self.context_engine:
            try:
                task_context = self.context_engine.get_context(
                    task.description, task.active_file
                )
            except Exception as e:
                logger.warning(f"Context retrieval failed for task {task.id}: {e}")

        ctx_tokens = len(task_context) // 4

        # Execute via local worker
        output: dict[str, Any] = {"raw_response": "", "file_changes": []}
        tok_ps = 0.0
        if self.worker:
            try:
                output = await self.worker.execute(
                    task=task.description,
                    context=task_context,
                    active_file=task.active_file,
                )
                tok_ps = output.get("tokens_per_second", 0.0)
            except Exception as e:
                logger.error(f"Worker execution failed (task {task.id}, attempt {attempt}): {e}")
                output["error"] = str(e)

        if output.get("error") and not output.get("raw_response"):
            return False, tok_ps, ctx_tokens

        # Apply file changes (with backup — tracked for checkpoint rollback)
        changed = self._feedback.apply_changes(output.get("file_changes", []))
        self._checkpoint_backups.extend(changed)
        task.files_changed = changed

        # Update indexes silently
        self._update_indexes(changed)

        # Review output silently
        review: dict[str, Any] = {"passed": True, "score": 7, "learnings": []}
        if self.brain and output.get("raw_response"):
            try:
                review = await self.brain.review(
                    task=task.description,
                    output=output.get("raw_response", ""),
                    changed_files=changed,
                )
            except Exception as e:
                logger.error(f"Review failed for task {task.id}: {e}")

        task.review_score = review.get("score", 0)

        if not review.get("passed", True):
            if attempt == 0:
                # Replace description with brain's improved retry prompt for next attempt
                task.description = review.get("retry_prompt") or task.description
            else:
                task.block_reason = (
                    "Worker failed twice: "
                    + "; ".join(str(i) for i in review.get("issues", ["unknown error"]))
                )
            return False, tok_ps, ctx_tokens

        return True, tok_ps, ctx_tokens

    def _update_indexes(self, changed_files: list[str]) -> None:
        """Silently update vector store and project graph for changed files."""
        for file_path in changed_files:
            if self.vector_store:
                try:
                    self.vector_store.update_file(file_path)
                except Exception as e:
                    logger.warning(f"Vector store update failed for {file_path}: {e}")
            if self.graph:
                try:
                    self.graph.parse_file(file_path)
                except Exception as e:
                    logger.warning(f"Graph parse failed for {file_path}: {e}")

    # ── Checkpoint ────────────────────────────────────────────────────────────

    async def _do_checkpoint(
        self,
        tasks_since_last: int,
        is_final: bool = False,
    ) -> AsyncIterator[RunEvent]:
        """Render checkpoint panel and handle user input.

        Yields CheckpointStart event, then CheckpointStop/Rollback depending on input.
        """
        assert self.plan is not None

        yield RunEvent(
            type=EventType.CHECKPOINT_START,
            plan=self.plan,
            message="final" if is_final else f"every {self.checkpoint_every}",
        )

        if self.console is None:
            return

        from rich.panel import Panel
        from rich.table import Table
        from rich import box as rich_box

        plan = self.plan
        blocked = plan.blocked_tasks()

        # Build a summary via brain
        summary_text = ""
        if self.brain:
            try:
                recent_changes: list[dict] = []
                for t in plan.tasks:
                    if t.status == TaskStatus.DONE and t.files_changed:
                        recent_changes.append({"task": t.description, "files": t.files_changed})
                summary_result = await self.brain.summarise(recent_changes, plan.to_dict())
                summary_text = summary_result.get("summary", "")
            except Exception as e:
                logger.warning(f"Checkpoint summary failed: {e}")
                summary_text = "(summary unavailable)"

        # Record checkpoint
        cp = CheckpointRecord(
            after_task=plan.done_count,
            timestamp=datetime.now().isoformat(),
            summary=summary_text,
            files_backed_up=list(self._checkpoint_backups),
        )
        plan.checkpoints.append(cp)
        self.save_plan()

        # ── Render checkpoint panel ───────────────────────────────────────────
        lines: list[str] = []
        header = "✅ FINAL SUMMARY" if is_final else f"⏸  CHECKPOINT — {tasks_since_last} tasks completed"
        lines.append(f"[bold cyan]{header}[/bold cyan]\n")

        # Recent task results
        recent_done = [t for t in plan.tasks if t.status == TaskStatus.DONE][-tasks_since_last:]
        if recent_done:
            lines.append("[bold green]Completed:[/bold green]")
            for t in recent_done:
                files_str = ", ".join(t.files_changed[:3]) or "—"
                lines.append(f"  [green]✓[/green] {t.description[:55]:<55} → [dim]{files_str}[/dim]")

        # Blocked tasks
        if blocked:
            lines.append("\n[bold red]⚠  BLOCKED (needs your attention):[/bold red]")
            for t in blocked:
                lines.append(f"  [red]Task {t.id}:[/red] {t.description[:60]}")
                lines.append(f"  [dim]  Reason: {(t.block_reason or 'unknown')[:80]}[/dim]")

        # Brain summary
        if summary_text:
            lines.append(f"\n[bold yellow]📋 What was built:[/bold yellow]")
            lines.append(f"  [dim]{summary_text[:200]}[/dim]")

        # Upcoming tasks
        upcoming = plan.pending_tasks()[:5]
        if upcoming and not is_final:
            lines.append(f"\n[bold]Next {len(upcoming)} tasks:[/bold]")
            for t in upcoming:
                lines.append(f"  [dim]{t.id}.[/dim] {t.description[:65]}")

        content = "\n".join(lines)
        title = "🔨 forge auto — checkpoint" if not is_final else "🔨 forge auto — complete"
        self.console.print(Panel(content, title=title, border_style="cyan", padding=(1, 2)))

        if is_final:
            return

        # ── Prompt for action ─────────────────────────────────────────────────
        while True:
            try:
                raw = self.console.input(
                    "\n  [[bold cyan]c[/bold cyan]] continue  "
                    "[[bold yellow]e[/bold yellow]] edit plan  "
                    "[[bold red]s[/bold red]] stop  "
                    "[[bold magenta]r[/bold magenta]] rollback  : "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                raw = "s"

            if raw == "c":
                # Clear the checkpoint backup list (new window starts)
                self._checkpoint_backups = []
                return
            elif raw == "e":
                self._open_plan_in_editor()
                # Re-read modified plan
                try:
                    data = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
                    updated = AutoPlan.from_dict(data)
                    self.plan.tasks = updated.tasks
                    self.console.print("[green]Plan reloaded.[/green]")
                except Exception as ex:
                    self.console.print(f"[red]Could not reload plan: {ex}[/red]")
                self._checkpoint_backups = []
                return
            elif raw == "s":
                yield RunEvent(type=EventType.CHECKPOINT_STOP, plan=plan,
                               message="User requested stop")
                return
            elif raw == "r":
                self._rollback_checkpoint(cp)
                yield RunEvent(type=EventType.CHECKPOINT_ROLLBACK, plan=plan,
                               message="Rolled back checkpoint changes")
                self._checkpoint_backups = []
                return
            else:
                self.console.print("[dim]Type c, e, s, or r[/dim]")

    def _open_plan_in_editor(self) -> None:
        """Open plan.json in $EDITOR (falls back to notepad on Windows, nano on Linux)."""
        import subprocess
        import sys
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if not editor:
            editor = "notepad" if sys.platform == "win32" else "nano"
        try:
            subprocess.call([editor, str(PLAN_PATH)])
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Could not open editor: {e}[/red]")
                self.console.print(f"[dim]Manually edit: {PLAN_PATH.absolute()}[/dim]")

    def _rollback_checkpoint(self, cp: CheckpointRecord) -> None:
        """Restore all .forge_backup files created since the last checkpoint."""
        self._feedback.rollback(cp.files_backed_up)
        # Reset task statuses for rolled-back tasks
        if self.plan:
            for task in self.plan.tasks:
                any_match = any(f in cp.files_backed_up for f in task.files_changed)
                if any_match and task.status in (TaskStatus.DONE, TaskStatus.BLOCKED):
                    task.status = TaskStatus.PENDING
                    task.files_changed = []
                    task.block_reason = None
                    task.review_score = 0
            self.save_plan()
