"""tests/test_core.py — Unit tests for Forge core components."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.vector_store import VectorStore
from forge.project_graph import ProjectGraph
from forge.context_engine import ContextEngine
from forge.worker import Worker
from forge.feedback import FeedbackLoop


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test artifacts."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# TestVectorStore
# ──────────────────────────────────────────────────────────────────────────────

class TestVectorStore:
    """Test suite for VectorStore functionality."""

    def test_vector_store_index_and_search(self, temp_dir: Path) -> None:
        """VectorStore indexes files and returns matching results."""
        file1 = temp_dir / "script1.py"
        file1.write_text('def hello():\n    print("Hello from script 1")\n')

        file2 = temp_dir / "script2.py"
        file2.write_text('def calculate(x, y):\n    """Add two numbers."""\n    return x + y\n')

        file3 = temp_dir / "utils.py"
        file3.write_text('def helper():\n    pass\n\ndef format_text(text):\n    return text.upper()\n')

        store = VectorStore()
        store.index_files([str(file1), str(file2), str(file3)])

        results = store.search("calculate numbers", top_k=3)

        assert len(results) > 0, "Search should return at least one result"

        # The top result should come from the file containing 'calculate'
        top_file = results[0]["file"]
        assert str(file2) in top_file or "calculate" in results[0]["text"], (
            f"Expected result from script2.py (calculate), got {top_file}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestProjectGraph
# ──────────────────────────────────────────────────────────────────────────────

class TestProjectGraph:
    """Test suite for ProjectGraph functionality."""

    def test_project_graph_parse(self, temp_dir: Path) -> None:
        """ProjectGraph correctly parses import relationships."""
        # utils.py must be parsed first so main.py can resolve the import edge
        utils_file = temp_dir / "utils.py"
        utils_file.write_text("def helper():\n    pass\n\ndef format_text(text):\n    return text.upper()\n")

        main_file = temp_dir / "main.py"
        main_file.write_text("import utils\n\ndef main():\n    print('Starting main')\n")

        graph = ProjectGraph()
        graph.parse_project(str(temp_dir))

        edges = graph.get_edges()
        assert len(edges) > 0, "At least one edge should exist"

        # Verify main.py imports utils (import edge: main_file → utils_file)
        import_edges = [e for e in edges if e["type"] == "imports"]
        assert len(import_edges) > 0, "Should have at least one import edge"

        found = any(
            str(main_file) in e["source"] and (str(utils_file) in e["target"] or "utils" in e["target"])
            for e in import_edges
        )
        assert found, (
            f"Expected import edge from main.py to utils.py. Edges: {import_edges}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestContextEngine
# ──────────────────────────────────────────────────────────────────────────────

class TestContextEngine:
    """Test suite for ContextEngine budget constraints."""

    def test_context_engine_budget(self) -> None:
        """ContextEngine respects the character budget."""
        # Use a real VectorStore with no data and a real (empty) ProjectGraph
        vs = VectorStore()
        pg = ProjectGraph()
        engine = ContextEngine(vector_store=vs, graph=pg)

        # With no indexed data the result is the empty-context header
        context = engine.get_context("test query", budget_chars=500)

        assert len(context) <= 500, (
            f"Context length {len(context)} exceeds budget 500"
        )

    def test_context_engine_returns_string(self) -> None:
        """ContextEngine.get_context always returns a string."""
        vs = VectorStore()
        pg = ProjectGraph()
        engine = ContextEngine(vector_store=vs, graph=pg)
        result = engine.get_context("hello world")
        assert isinstance(result, str)


# ──────────────────────────────────────────────────────────────────────────────
# TestWorker
# ──────────────────────────────────────────────────────────────────────────────

class TestWorker:
    """Test suite for Worker file block parsing."""

    def test_worker_parse_file_blocks(self) -> None:
        """Worker correctly parses <<<FILE:>>> blocks from LLM response."""
        sample_response = (
            "Here's what I'll change:\n\n"
            "<<<FILE: src/main.py>>>\n"
            "def main():\n"
            "    pass\n"
            "<<<END FILE>>>\n\n"
            "<<<FILE: src/utils.py>>>\n"
            "def helper():\n"
            "    return True\n"
            "<<<END FILE>>>\n"
        )

        worker = Worker()
        blocks = worker.parse_file_blocks(sample_response)

        assert len(blocks) == 2, f"Expected 2 file blocks, got {len(blocks)}"
        assert blocks[0]["file_path"] == "src/main.py"
        assert "def main():" in blocks[0]["content"]
        assert blocks[1]["file_path"] == "src/utils.py"
        assert "def helper():" in blocks[1]["content"]

    def test_worker_no_blocks(self) -> None:
        """Worker returns empty list when no FILE blocks are present."""
        worker = Worker()
        blocks = worker.parse_file_blocks("Some plain text response with no file blocks.")
        assert blocks == []


# ──────────────────────────────────────────────────────────────────────────────
# TestFeedbackLoop
# ──────────────────────────────────────────────────────────────────────────────

class TestFeedbackLoop:
    """Test suite for FeedbackLoop apply_changes and rollback."""

    def test_feedback_apply_and_rollback(self, temp_dir: Path) -> None:
        """FeedbackLoop applies changes with backup, then rolls back correctly."""
        test_file = temp_dir / "test.txt"
        original_content = "Original content"
        test_file.write_text(original_content, encoding="utf-8")

        feedback = FeedbackLoop()

        changes = [
            {"file_path": str(test_file), "content": "Modified content"}
        ]

        changed = feedback.apply_changes(changes)

        # File should be changed
        assert test_file.read_text(encoding="utf-8") == "Modified content"
        # Backup should exist
        backup = Path(f"{test_file}.forge_backup")
        assert backup.exists(), "Backup file should have been created"
        assert backup.read_text(encoding="utf-8") == original_content
        # apply_changes returns list of changed paths
        assert str(test_file) in changed

        # Rollback
        feedback.rollback([str(test_file)])

        # Original content restored
        assert test_file.read_text(encoding="utf-8") == original_content
        # Backup should be gone after rollback
        assert not backup.exists(), "Backup should be removed after rollback"