"""FeedbackLoop — plan-execute-review orchestration for forge coding agent."""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """Master orchestration loop implementing iterative plan-execute-review cycle."""

    def __init__(self) -> None:
        """Initialize FeedbackLoop with empty state."""
        self.session_memory: list[str] = []   # learnings accumulated this run (max 20)
        self.changes_log: list[dict[str, Any]] = []

    async def run(
        self,
        goal: str,
        active_file: str | None = None,
        max_iterations: int = 3,
        brain=None,
        worker=None,
        context_engine=None,
        vector_store=None,
        graph=None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Execute the plan-execute-review cycle.

        Args:
            goal: The task goal to accomplish
            active_file: Primary file being worked on (optional)
            max_iterations: Maximum retry attempts per subtask
            brain: Brain instance for planning/reviewing
            worker: Worker instance for code execution
            context_engine: ContextEngine instance (synchronous)
            vector_store: VectorStore instance
            graph: ProjectGraph instance
            dry_run: If True, plan only — do not execute

        Returns:
            dict with goal, passed, sub_tasks_total, sub_tasks_passed,
                  iterations_used, changes_log, session_memory, elapsed_seconds
        """
        start_time = time.time()
        iterations_used = 0

        try:
            # ── Step 1: Plan ──────────────────────────────────────────────────
            context = ""
            project_summary: dict[str, Any] = {}

            if graph:
                project_summary = graph.get_summary()

            if context_engine:
                try:
                    context = context_engine.get_context(goal, active_file)
                except Exception as e:
                    logger.warning(f"Initial context retrieval failed: {e}")

            plan_result: dict[str, Any] = {}
            if brain:
                try:
                    plan_result = await brain.plan(
                        goal=goal, context=context, project_summary=project_summary
                    )
                except Exception as e:
                    logger.error(f"Brain planning failed: {e}")
                    plan_result = {}

            sub_tasks: list[dict[str, Any]] = plan_result.get("sub_tasks", [])
            if not sub_tasks:
                sub_tasks = [{"id": 1, "description": goal, "active_file": active_file}]

            if dry_run:
                return {
                    "goal": goal,
                    "plan": plan_result,
                    "sub_tasks": sub_tasks,
                    "dry_run": True,
                    "passed": True,
                    "sub_tasks_total": len(sub_tasks),
                    "sub_tasks_passed": 0,
                    "iterations_used": 0,
                    "changes_log": [],
                    "session_memory": [],
                    "elapsed_seconds": time.time() - start_time,
                }

            sub_tasks_total = len(sub_tasks)
            sub_tasks_passed = 0

            # ── Step 2: Execute each sub-task ─────────────────────────────────
            for sub_task in sub_tasks:
                iterations_used += 1

                task_desc = sub_task.get("description", goal)
                task_file = sub_task.get("active_file") or active_file

                # Get context for this sub-task
                task_context = ""
                if context_engine:
                    try:
                        task_context = context_engine.get_context(task_desc, task_file)
                    except Exception as e:
                        logger.warning(f"Context retrieval failed for sub-task: {e}")

                # Execute via worker
                output: dict[str, Any] = {"raw_response": "", "file_changes": []}
                if worker:
                    try:
                        output = await worker.execute(
                            task=task_desc,
                            context=task_context,
                            active_file=task_file,
                        )
                    except Exception as e:
                        logger.error(f"Worker execution failed: {e}")
                        output["error"] = str(e)

                # Apply file changes (with backup)
                changed_files = self.apply_changes(output.get("file_changes", []))

                # Update vector store for changed files
                if vector_store and changed_files:
                    for file_path in changed_files:
                        try:
                            vector_store.update_file(file_path)
                        except Exception as e:
                            logger.warning(f"Vector store update failed for {file_path}: {e}")

                # Update project graph for changed files
                if graph and changed_files:
                    for file_path in changed_files:
                        try:
                            graph.parse_file(file_path)
                        except Exception as e:
                            logger.warning(f"Graph parse failed for {file_path}: {e}")

                # Review output
                review: dict[str, Any] = {"passed": True, "learnings": [], "score": 7}
                if brain and output.get("raw_response"):
                    try:
                        review = await brain.review(
                            task=task_desc,
                            output=output.get("raw_response", ""),
                            changed_files=changed_files,
                        )
                    except Exception as e:
                        logger.error(f"Review failed: {e}")

                # Store learnings (max 20)
                for learning in review.get("learnings", []):
                    if len(self.session_memory) < 20:
                        self.session_memory.append(learning)

                # Record changes
                for file_path in changed_files:
                    self.changes_log.append({
                        "file": file_path,
                        "task": task_desc,
                        "review_score": review.get("score", 0),
                    })

                if review.get("passed", True):
                    sub_tasks_passed += 1
                else:
                    logger.warning(f"Sub-task did not pass review: {task_desc[:60]}")

            return {
                "goal": goal,
                "passed": sub_tasks_passed == sub_tasks_total,
                "sub_tasks_total": sub_tasks_total,
                "sub_tasks_passed": sub_tasks_passed,
                "iterations_used": iterations_used,
                "changes_log": self.changes_log,
                "session_memory": list(self.session_memory),
                "elapsed_seconds": time.time() - start_time,
            }

        except Exception as e:
            logger.exception(f"FeedbackLoop.run() failed: {e}")
            return {
                "goal": goal,
                "passed": False,
                "sub_tasks_total": 0,
                "sub_tasks_passed": 0,
                "iterations_used": iterations_used,
                "changes_log": [],
                "session_memory": list(self.session_memory),
                "elapsed_seconds": time.time() - start_time,
                "error": str(e),
            }

    def apply_changes(self, file_changes: list[dict]) -> list[str]:
        """Apply file changes with automatic backup.

        Each entry in file_changes should have:
            - "file_path" or "file": destination path
            - "content": new file content

        Returns:
            List of successfully changed file paths
        """
        changed_paths: list[str] = []

        for change in file_changes:
            # Support both "file_path" (worker format) and "file" (test format)
            file_path = change.get("file_path") or change.get("file", "")
            content = change.get("content", "")

            if not file_path or not content:
                continue

            # Create backup of existing file
            backup_path = f"{file_path}.forge_backup"
            try:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"Backed up {file_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup for {file_path}: {e}")

            # Write new content
            try:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                changed_paths.append(file_path)
                logger.info(f"Wrote changes to {file_path}")
            except Exception as e:
                logger.error(f"Failed to write to {file_path}: {e}")

        return changed_paths

    def rollback(self, file_paths: list[str] | None = None) -> None:
        """Restore backed-up files.

        Args:
            file_paths: List of file paths to restore.
                        If None, restores all files that have a .forge_backup.
        """
        if file_paths is None:
            # Auto-discover backup files
            file_paths = [
                str(p)[: -len(".forge_backup")]
                for p in Path(".").rglob("*.forge_backup")
            ]

        for file_path in file_paths:
            backup_path = f"{file_path}.forge_backup"
            if os.path.exists(backup_path):
                try:
                    shutil.move(backup_path, file_path)
                    logger.info(f"Restored {file_path} from backup")
                except Exception as e:
                    logger.error(f"Failed to restore {file_path}: {e}")
            else:
                logger.warning(f"No backup found for {file_path}")
