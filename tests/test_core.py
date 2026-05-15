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


class TestPatchApplyRoundtrip:
    def test_patch_apply_roundtrip(self, temp_dir):
        from forge.worker import Worker
        worker = Worker()
        
        test_file = temp_dir / "target.py"
        test_file.write_text("def hello():\n    print('world')\n", encoding="utf-8")
        
        # Test successful patch
        success = worker.apply_patch(str(test_file), "print('world')", "print('hello world')")
        assert success is True
        assert test_file.read_text(encoding="utf-8") == "def hello():\n    print('hello world')\n"
        
        # Test failing patch (find string not present)
        success = worker.apply_patch(str(test_file), "print('missing')", "print('found')")
        assert success is False
        assert test_file.read_text(encoding="utf-8") == "def hello():\n    print('hello world')\n"


# ──────────────────────────────────────────────────────────────────────────────
# Edge Cases for VectorStore
# ──────────────────────────────────────────────────────────────────────────────

class TestVectorStoreEdgeCases:
    """Edge case tests for VectorStore."""

    def test_search_empty_query(self, temp_dir: Path) -> None:
        """Search with empty query returns empty list."""
        store = VectorStore()
        results = store.search("")
        assert results == []

    def test_search_nonexistent_query(self, temp_dir: Path) -> None:
        """Search returns results but with low score for unrelated query."""
        file1 = temp_dir / "test.py"
        file1.write_text("def foo():\n    pass\n")
        store = VectorStore()
        store.index_file(str(file1))
        results = store.search("nonexistent term xyz 12345")
        # FAISS always returns results but with low score for unrelated query
        assert len(results) >= 1
        # The score should be low since the query is unrelated
        assert results[0]["score"] < 0.5

    def test_remove_and_reindex(self, temp_dir: Path) -> None:
        """Remove file and re-index works correctly."""
        file1 = temp_dir / "test.py"
        file1.write_text("def original():\n    pass\n")

        store = VectorStore()
        store.index_file(str(file1))
        results1 = store.search("original")
        assert len(results1) > 0

        # Remove and update with new content
        store.remove_file(str(file1))
        file1.write_text("def modified():\n    pass\n")
        store.index_file(str(file1))

        results2 = store.search("modified")
        assert len(results2) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Edge Cases for ProjectGraph
# ──────────────────────────────────────────────────────────────────────────────

class TestProjectGraphEdgeCases:
    """Edge case tests for ProjectGraph."""

    def test_empty_directory(self, temp_dir: Path) -> None:
        """Graph handles empty directory without errors."""
        graph = ProjectGraph()
        graph.parse_project(str(temp_dir))
        # Use graph.graph.nodes to get nodes
        nodes = list(graph.graph.nodes)
        edges = graph.get_edges()
        assert nodes == []
        assert edges == []

    def test_circular_imports(self, temp_dir: Path) -> None:
        """Graph handles circular imports (a imports b, b imports a)."""
        file_a = temp_dir / "a.py"
        file_a.write_text("import b\n\ndef func_a():\n    pass\n")

        file_b = temp_dir / "b.py"
        file_b.write_text("import a\n\ndef func_b():\n    pass\n")

        graph = ProjectGraph()
        graph.parse_project(str(temp_dir))

        edges = graph.get_edges()
        import_edges = [e for e in edges if e["type"] == "imports"]
        # Should have both directions
        assert len(import_edges) >= 2


# ──────────────────────────────────────────────────────────────────────────────
# Edge Cases for Worker
# ──────────────────────────────────────────────────────────────────────────────

class TestWorkerEdgeCases:
    """Edge case tests for Worker."""

    def test_parse_multiple_file_blocks(self) -> None:
        """Worker correctly parses multiple FILE blocks."""
        response = (
            "<<<FILE: file1.py>>>\ncontent1\n<<<END FILE>>>\n"
            "<<<FILE: file2.py>>>\ncontent2\n<<<END FILE>>>\n"
            "<<<FILE: file3.py>>>\ncontent3\n<<<END FILE>>>"
        )
        worker = Worker()
        blocks = worker.parse_file_blocks(response)
        assert len(blocks) == 3
        assert blocks[0]["file_path"] == "file1.py"
        assert blocks[1]["file_path"] == "file2.py"
        assert blocks[2]["file_path"] == "file3.py"

    def test_parse_mixed_blocks(self) -> None:
        """Worker handles FILE and PATCH blocks mixed."""
        response = (
            "First some text\n"
            "<<<FILE: new.py>>>\nnew content\n<<<END FILE>>>\n"
            "Then more text\n"
            "<<<PATCH: existing.py>>>\n<<<FIND>>>\nold\n<<<REPLACE>>>\nnew\n<<<END PATCH>>>"
        )
        worker = Worker()
        file_blocks = worker.parse_file_blocks(response)
        patch_blocks = worker.parse_patch_blocks(response)
        assert len(file_blocks) == 1
        assert len(patch_blocks) == 1

    def test_whitespace_normalization_in_patch(self, temp_dir: Path) -> None:
        """Worker normalizes whitespace when applying patches."""
        worker = Worker()
        test_file = temp_dir / "ws_test.py"
        test_file.write_text("def   foo():\n    pass\n", encoding="utf-8")

        # Try to patch with different whitespace
        success = worker.apply_patch(str(test_file), "def   foo()", "def bar()")
        # Should succeed since whitespace is normalized
        assert success is True


# ──────────────────────────────────────────────────────────────────────────────
# Edge Cases for FeedbackLoop
# ──────────────────────────────────────────────────────────────────────────────

class TestFeedbackLoopEdgeCases:
    """Edge case tests for FeedbackLoop."""

    def test_rollback_nonexistent_file(self, temp_dir: Path) -> None:
        """Rollback of non-existent file does not raise."""
        feedback = FeedbackLoop()
        # Should not raise
        feedback.rollback(["nonexistent_file.py"])

    def test_apply_changes_multiple_files(self, temp_dir: Path) -> None:
        """Apply changes to multiple files works correctly."""
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("original1", encoding="utf-8")
        file2.write_text("original2", encoding="utf-8")

        feedback = FeedbackLoop()
        changes = [
            {"file_path": str(file1), "content": "modified1"},
            {"file_path": str(file2), "content": "modified2"},
        ]
        changed = feedback.apply_changes(changes)

        assert file1.read_text(encoding="utf-8") == "modified1"
        assert file2.read_text(encoding="utf-8") == "modified2"
        assert len(changed) == 2

    def test_rollback_multiple_files(self, temp_dir: Path) -> None:
        """Rollback of multiple files restores original content."""
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        original1 = "original1"
        original2 = "original2"
        file1.write_text(original1, encoding="utf-8")
        file2.write_text(original2, encoding="utf-8")

        feedback = FeedbackLoop()
        changes = [
            {"file_path": str(file1), "content": "modified1"},
            {"file_path": str(file2), "content": "modified2"},
        ]
        feedback.apply_changes(changes)

        feedback.rollback([str(file1), str(file2)])

        assert file1.read_text(encoding="utf-8") == original1
        assert file2.read_text(encoding="utf-8") == original2
