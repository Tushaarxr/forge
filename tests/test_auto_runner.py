"""Tests for AutoRunner — autonomous build mode engine."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_runner(tmp_path: Path, checkpoint_every: int = 5, max_tasks: int = 50):
    """Return a minimally configured AutoRunner with mocked dependencies."""
    # Patch FORGE_DIR so plan.json goes to tmp_path
    import forge.auto_runner as ar_module
    ar_module.FORGE_DIR = tmp_path / ".forge"
    ar_module.PLAN_PATH = tmp_path / ".forge" / "plan.json"
    ar_module.PLANS_ARCHIVE_DIR = tmp_path / ".forge" / "plans"

    from forge.auto_runner import AutoRunner

    brain = MagicMock()
    brain.plan_full_project = AsyncMock(return_value=[
        {"id": 1, "description": "Scaffold project", "active_file": "main.py",
         "category": "scaffold", "estimated_lines": 20},
        {"id": 2, "description": "Write models", "active_file": "models.py",
         "category": "model", "estimated_lines": 40},
        {"id": 3, "description": "Add routes", "active_file": "routes.py",
         "category": "route", "estimated_lines": 60},
    ])
    brain.review = AsyncMock(return_value={
        "passed": True, "score": 8, "issues": [], "retry_prompt": "", "learnings": []
    })
    brain.summarise = AsyncMock(return_value={
        "summary": "Built scaffold and models", "key_decisions": [], "next_suggested": [], "risk_flags": []
    })

    worker = MagicMock()
    worker.execute = AsyncMock(return_value={
        "raw_response": "<<<FILE: main.py>>>\nprint('hello')\n<<<END FILE>>>",
        "file_changes": [],
        "tokens_used": 100,
        "elapsed_seconds": 1.0,
        "tokens_per_second": 100.0,
    })

    feedback = MagicMock()
    feedback.apply_changes = MagicMock(return_value=[])
    feedback.rollback = MagicMock()

    runner = AutoRunner(
        brain=brain,
        worker=worker,
        context_engine=None,
        vector_store=None,
        graph=None,
        checkpoint_every=checkpoint_every,
        max_tasks=max_tasks,
        console=None,
    )
    # Inject mock feedback loop
    runner._feedback = feedback
    return runner


# ── AutoTask / AutoPlan data models ───────────────────────────────────────────

class TestAutoTask:
    def test_to_dict_roundtrip(self):
        from forge.auto_runner import AutoTask, TaskStatus
        task = AutoTask(id=1, description="Write tests", active_file="test_main.py",
                        status=TaskStatus.DONE, review_score=9, files_changed=["test_main.py"])
        d = task.to_dict()
        restored = AutoTask.from_dict(d)
        assert restored.id == 1
        assert restored.description == "Write tests"
        assert restored.status == TaskStatus.DONE
        assert restored.review_score == 9
        assert restored.files_changed == ["test_main.py"]

    def test_default_status_is_pending(self):
        from forge.auto_runner import AutoTask, TaskStatus
        t = AutoTask(id=1, description="x")
        assert t.status == TaskStatus.PENDING


class TestAutoPlan:
    def test_counts(self):
        from forge.auto_runner import AutoPlan, AutoTask, TaskStatus
        plan = AutoPlan(goal="test", created_at="2025-01-01T00:00:00")
        plan.tasks = [
            AutoTask(id=1, description="a", status=TaskStatus.DONE),
            AutoTask(id=2, description="b", status=TaskStatus.BLOCKED),
            AutoTask(id=3, description="c", status=TaskStatus.PENDING),
        ]
        assert plan.done_count == 1
        assert plan.blocked_count == 1
        assert plan.total == 3

    def test_pending_tasks_excludes_done_and_blocked(self):
        from forge.auto_runner import AutoPlan, AutoTask, TaskStatus
        plan = AutoPlan(goal="test", created_at="2025-01-01T00:00:00")
        plan.tasks = [
            AutoTask(id=1, description="done", status=TaskStatus.DONE),
            AutoTask(id=2, description="blocked", status=TaskStatus.BLOCKED),
            AutoTask(id=3, description="pending", status=TaskStatus.PENDING),
        ]
        pending = plan.pending_tasks()
        assert len(pending) == 1
        assert pending[0].description == "pending"

    def test_roundtrip_serialisation(self):
        from forge.auto_runner import AutoPlan, AutoTask, TaskStatus, PlanStatus
        plan = AutoPlan(goal="build something", created_at=datetime.now().isoformat(),
                        status=PlanStatus.IN_PROGRESS)
        plan.tasks = [AutoTask(id=1, description="task 1")]
        d = plan.to_dict()
        restored = AutoPlan.from_dict(d)
        assert restored.goal == "build something"
        assert restored.status == PlanStatus.IN_PROGRESS
        assert len(restored.tasks) == 1
        assert restored.tasks[0].description == "task 1"


# ── Plan I/O ──────────────────────────────────────────────────────────────────

class TestPlanPersistence:
    def test_save_and_load(self, tmp_path):
        runner = _make_runner(tmp_path)
        import forge.auto_runner as ar_module
        from forge.auto_runner import AutoPlan, AutoTask, PlanStatus

        plan = AutoPlan(goal="my goal", created_at=datetime.now().isoformat(),
                        status=PlanStatus.IN_PROGRESS)
        plan.tasks = [AutoTask(id=1, description="do something")]
        runner.plan = plan
        runner.save_plan()

        plan_file = tmp_path / ".forge" / "plan.json"
        assert plan_file.exists()
        data = json.loads(plan_file.read_text())
        assert data["goal"] == "my goal"
        assert data["status"] == "in_progress"
        assert len(data["tasks"]) == 1

    def test_load_existing_plan_returns_none_if_missing(self, tmp_path):
        import forge.auto_runner as ar_module
        ar_module.PLAN_PATH = tmp_path / ".forge" / "plan.json"
        from forge.auto_runner import AutoRunner
        result = AutoRunner.load_existing_plan()
        assert result is None

    def test_load_existing_plan_returns_none_if_done(self, tmp_path):
        import forge.auto_runner as ar_module
        forge_dir = tmp_path / ".forge"
        forge_dir.mkdir()
        plan_file = forge_dir / "plan.json"
        ar_module.PLAN_PATH = plan_file
        plan_data = {
            "goal": "test", "created_at": "2025-01-01T00:00:00",
            "status": "done", "tasks": [], "checkpoints": []
        }
        plan_file.write_text(json.dumps(plan_data))
        from forge.auto_runner import AutoRunner
        result = AutoRunner.load_existing_plan()
        assert result is None  # only returns in_progress plans

    def test_archive_plan(self, tmp_path):
        import forge.auto_runner as ar_module
        forge_dir = tmp_path / ".forge"
        forge_dir.mkdir()
        plan_file = forge_dir / "plan.json"
        ar_module.PLAN_PATH = plan_file
        ar_module.PLANS_ARCHIVE_DIR = forge_dir / "plans"
        plan_file.write_text(json.dumps({"goal": "old", "status": "in_progress",
                                          "created_at": "2025-01-01", "tasks": [], "checkpoints": []}))

        from forge.auto_runner import AutoRunner
        runner = AutoRunner(None, None, None, None, None)
        runner.archive_plan()

        assert not plan_file.exists()
        archived = list((forge_dir / "plans").glob("*.json"))
        assert len(archived) == 1


# ── Autonomous execution loop ─────────────────────────────────────────────────

class TestAutonomousLoop:
    @pytest.mark.asyncio
    async def test_run_marks_tasks_done(self, tmp_path):
        runner = _make_runner(tmp_path, checkpoint_every=10)
        plan = await runner.create_plan(goal="build a calculator")

        events = []
        async for event in runner.run():
            events.append(event)

        from forge.auto_runner import EventType, TaskStatus
        done_events = [e for e in events if e.type == EventType.TASK_DONE]
        assert len(done_events) == 3

        for task in runner.plan.tasks:
            assert task.status == TaskStatus.DONE

    @pytest.mark.asyncio
    async def test_failed_review_marks_blocked_after_two_attempts(self, tmp_path):
        runner = _make_runner(tmp_path, checkpoint_every=10)

        # Make review always fail
        runner.brain.review = AsyncMock(return_value={
            "passed": False, "score": 2,
            "issues": ["import error"], "retry_prompt": "try again", "learnings": []
        })

        plan = await runner.create_plan(goal="build something broken")

        events = []
        async for event in runner.run():
            events.append(event)

        from forge.auto_runner import EventType, TaskStatus
        blocked_events = [e for e in events if e.type == EventType.TASK_BLOCKED]
        assert len(blocked_events) == 3

        for task in runner.plan.tasks:
            assert task.status == TaskStatus.BLOCKED
            assert task.block_reason is not None
            assert "import error" in task.block_reason

    @pytest.mark.asyncio
    async def test_max_tasks_hard_stop(self, tmp_path):
        runner = _make_runner(tmp_path, checkpoint_every=10, max_tasks=2)
        plan = await runner.create_plan(goal="build something")

        events = []
        async for event in runner.run():
            events.append(event)

        from forge.auto_runner import EventType, TaskStatus
        done_events = [e for e in events if e.type == EventType.TASK_DONE]
        # Should only process 2 out of 3 tasks
        assert len(done_events) == 2

    @pytest.mark.asyncio
    async def test_retry_uses_improved_prompt(self, tmp_path):
        runner = _make_runner(tmp_path, checkpoint_every=10)

        call_count = 0

        async def _failing_then_passing(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"passed": False, "score": 3, "issues": ["bad code"],
                        "retry_prompt": "IMPROVED: do it better", "learnings": []}
            return {"passed": True, "score": 8, "issues": [], "retry_prompt": "", "learnings": []}

        runner.brain.review = AsyncMock(side_effect=_failing_then_passing)

        plan = await runner.create_plan(goal="build something")

        from forge.auto_runner import EventType
        events = []
        async for event in runner.run():
            events.append(event)

        retry_events = [e for e in events if e.type == EventType.TASK_RETRYING]
        assert len(retry_events) >= 1


# ── plan_full_project Brain method ────────────────────────────────────────────

class TestBrainPlanFullProject:
    @pytest.mark.asyncio
    async def test_returns_task_list(self):
        from forge.brain import Brain
        brain = Brain()
        mock_response = json.dumps({
            "tasks": [
                {"id": 1, "description": "Scaffold", "active_file": "main.py",
                 "estimated_lines": 20, "category": "scaffold"},
                {"id": 2, "description": "Write models", "active_file": "models.py",
                 "estimated_lines": 40, "category": "model"},
            ]
        })
        with patch.object(brain, "_call", AsyncMock(return_value=mock_response)):
            tasks = await brain.plan_full_project("build a todo app", {})

        assert len(tasks) == 2
        assert tasks[0]["id"] == 1
        assert tasks[1]["description"] == "Write models"

    @pytest.mark.asyncio
    async def test_reassigns_sequential_ids(self):
        from forge.brain import Brain
        brain = Brain()
        # IDs are not sequential in raw response
        mock_response = json.dumps({
            "tasks": [
                {"id": 99, "description": "A", "active_file": None,
                 "estimated_lines": 10, "category": "scaffold"},
                {"id": 5, "description": "B", "active_file": None,
                 "estimated_lines": 20, "category": "model"},
            ]
        })
        with patch.object(brain, "_call", AsyncMock(return_value=mock_response)):
            tasks = await brain.plan_full_project("goal", {})

        # IDs should be re-assigned to 1, 2
        assert tasks[0]["id"] == 1
        assert tasks[1]["id"] == 2

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_api_error(self):
        from forge.brain import Brain
        import httpx
        brain = Brain()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "quota exceeded"
        with patch.object(brain, "_call",
                          AsyncMock(side_effect=httpx.HTTPStatusError("quota", request=MagicMock(), response=mock_response))):
            tasks = await brain.plan_full_project("goal", {})

        assert tasks == []


# ── Checkpoint rollback ────────────────────────────────────────────────────────

class TestCheckpointRollback:
    def test_rollback_resets_task_statuses(self, tmp_path):
        import forge.auto_runner as ar_module
        ar_module.FORGE_DIR = tmp_path / ".forge"
        ar_module.PLAN_PATH = tmp_path / ".forge" / "plan.json"
        ar_module.PLANS_ARCHIVE_DIR = tmp_path / ".forge" / "plans"

        from forge.auto_runner import (
            AutoRunner, AutoPlan, AutoTask, CheckpointRecord, TaskStatus
        )

        runner = AutoRunner(None, None, None, None, None)
        runner._feedback = MagicMock()
        runner._feedback.rollback = MagicMock()

        plan = AutoPlan(goal="test", created_at="2025-01-01T00:00:00")
        plan.tasks = [
            AutoTask(id=1, description="t1", status=TaskStatus.DONE,
                     files_changed=["main.py"]),
            AutoTask(id=2, description="t2", status=TaskStatus.BLOCKED,
                     files_changed=["models.py"]),
        ]
        runner.plan = plan
        runner.save_plan()

        cp = CheckpointRecord(
            after_task=2,
            timestamp="2025-01-01T00:00:00",
            summary="",
            files_backed_up=["main.py", "models.py"],
        )

        runner._rollback_checkpoint(cp)

        # All tasks should be reset to pending
        for t in plan.tasks:
            assert t.status == TaskStatus.PENDING
            assert t.files_changed == []
