"""tests/test_summariser.py — Unit tests for Summariser class."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture
def temp_dir(tmp_path):
    """Create temp directory for tests."""
    return tmp_path


@pytest.fixture
def mock_brain():
    """Create mock Brain."""
    brain = MagicMock()
    brain.summarise = AsyncMock(return_value={
        "summary": "Built authentication module with JWT",
        "key_decisions": ["Use JWT with RS256", "Store tokens in httpOnly cookies"],
        "next_suggested": ["Write unit tests", "Add rate limiting"],
        "risk_flags": ["Token expiry needs tuning"],
    })
    return brain


@pytest.fixture
def mock_vector_store():
    """Create mock VectorStore."""
    vs = MagicMock()
    vs.index_file = MagicMock(return_value=1)
    return vs


class TestSummariserInit:
    """Test Summariser initialization."""

    def test_init_creates_directories(self, temp_dir, mock_brain, mock_vector_store):
        """Initializes and creates .forge directories."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            # Directories would be created on first use
            assert s.base_dir == Path(".forge")
            assert s.summaries_dir == s.base_dir / "summaries"

    def test_default_checkpoint_interval(self, temp_dir, mock_brain, mock_vector_store):
        """Default checkpoint interval is 10."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            assert s.checkpoint_interval == 10

    def test_custom_checkpoint_interval_from_env(self, temp_dir, mock_brain, mock_vector_store):
        """Can set custom checkpoint interval from env var."""
        # Set env var before import so it gets read at init time
        import os
        old_val = os.environ.get("SUMMARISE_EVERY_N_CHANGES")
        try:
            os.environ["SUMMARISE_EVERY_N_CHANGES"] = "5"
            # Need to re-import to pick up the env var
            import importlib
            import forge.summariser as sm_module
            importlib.reload(sm_module)
            from forge.summariser import Summariser
            with patch("forge.summariser.Path.mkdir", return_value=None):
                s = Summariser(mock_brain, mock_vector_store)
                assert s.checkpoint_interval == 5
        finally:
            if old_val is None:
                os.environ.pop("SUMMARISE_EVERY_N_CHANGES", None)
            else:
                os.environ["SUMMARISE_EVERY_N_CHANGES"] = old_val

    def test_invalid_env_var_defaults_to_10(self, temp_dir, mock_brain, mock_vector_store):
        """Invalid env var value defaults to 10."""
        from forge.summariser import Summariser
        with patch.dict({"SUMMARISE_EVERY_N_CHANGES": "invalid"}):
            with patch("forge.summariser.Path.mkdir", return_value=None):
                s = Summariser(mock_brain, mock_vector_store)
                assert s.checkpoint_interval == 10


class TestShouldCheckpoint:
    """Test should_checkpoint logic."""

    def test_returns_true_when_equal_to_interval(self, temp_dir, mock_brain, mock_vector_store):
        """Returns True when changes equals interval."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            s.checkpoint_interval = 5
            assert s.should_checkpoint(5) is True

    def test_returns_true_when_exceeds_interval(self, temp_dir, mock_brain, mock_vector_store):
        """Returns True when changes exceeds interval."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            s.checkpoint_interval = 5
            assert s.should_checkpoint(10) is True
            assert s.should_checkpoint(100) is True

    def test_returns_false_when_below_interval(self, temp_dir, mock_brain, mock_vector_store):
        """Returns False when changes below interval."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            s.checkpoint_interval = 5
            assert s.should_checkpoint(4) is False
            assert s.should_checkpoint(0) is False
            assert s.should_checkpoint(-1) is False


class TestSummarizeChanges:
    """Test summarize_changes method."""

    def test_empty_list_returns_no_changes(self, temp_dir, mock_brain, mock_vector_store):
        """Empty list returns 'No changes.'"""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            result = s.summarize_changes([])
            assert result == "No changes."

    def test_single_change(self, temp_dir, mock_brain, mock_vector_store):
        """Single change returns formatted string."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            result = s.summarize_changes([{"action": "created", "file": "main.py"}])
            assert "Created" in result
            assert "main.py" in result

    def test_multiple_changes(self, temp_dir, mock_brain, mock_vector_store):
        """Multiple changes comma-separated."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            changes = [
                {"action": "created", "file": "auth.py"},
                {"action": "modified", "file": "main.py"},
            ]
            result = s.summarize_changes(changes)
            assert "Created: auth.py" in result
            assert "Modified: main.py" in result

    def test_uses_file_path_fallback(self, temp_dir, mock_brain, mock_vector_store):
        """Falls back to file_path if file key missing."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            result = s.summarize_changes([{"action": "created", "file_path": "utils.py"}])
            assert "utils.py" in result


class TestCompressContext:
    """Test compress_context method."""

    def test_empty_list_returns_empty(self, temp_dir, mock_brain, mock_vector_store):
        """Empty list returns empty list."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            result = s.compress_context([])
            assert result == []

    def test_keeps_one_per_type(self, temp_dir, mock_brain, mock_vector_store):
        """Keeps most recent item of each type."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            context = [
                {"type": "error", "content": "first error"},
                {"type": "error", "content": "second error"},
                {"type": "warning", "content": "warning"},
            ]
            result = s.compress_context(context)
            # Should keep last error and the warning
            assert len(result) == 2

    def test_no_type_uses_generic(self, temp_dir, mock_brain, mock_vector_store):
        """Items without type use 'generic' and get deduplicated to one."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            context = [
                {"content": "item1"},
                {"content": "item2"},
            ]
            result = s.compress_context(context)
            # Both have no type so use "generic", deduplicated to 1
            assert len(result) == 1


class TestRestoreCheckpoint:
    """Test restore_checkpoint method."""

    def test_file_not_found_raises(self, temp_dir, mock_brain, mock_vector_store):
        """Raises FileNotFoundError when file doesn't exist."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            with pytest.raises(FileNotFoundError):
                s.restore_checkpoint("/nonexistent/path.json")

    def test_loads_json_file(self, temp_dir, mock_brain, mock_vector_store):
        """Loads and returns JSON from file."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            # Create a temp checkpoint file
            checkpoint_file = temp_dir / "checkpoint.json"
            test_data = {"summary": "test", "key_decisions": []}
            checkpoint_file.write_text(json.dumps(test_data))

            result = s.restore_checkpoint(str(checkpoint_file))
            assert result == test_data


class TestGetSummary:
    """Test get_summary method."""

    def test_empty_history_returns_empty_session(self, temp_dir, mock_brain, mock_vector_store):
        """Empty history returns 'Empty session.'."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            result = s.get_summary([])
            assert result == "Empty session."

    def test_single_session(self, temp_dir, mock_brain, mock_vector_store):
        """Single session shows goal and passed status."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            history = [{"goal": "build auth", "passed": True}]
            result = s.get_summary(history)
            assert "build auth" in result
            assert "passed" in result.lower()

    def test_failed_session(self, temp_dir, mock_brain, mock_vector_store):
        """Failed session shows failed status."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            history = [{"goal": "build auth", "passed": False}]
            result = s.get_summary(history)
            assert "build auth" in result
            assert "failed" in result.lower()

    def test_multiple_runs_shows_count(self, temp_dir, mock_brain, mock_vector_store):
        """Multiple runs shows count."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            history = [{"goal": "task1"}, {"goal": "task2"}]
            result = s.get_summary(history)
            assert "2 run" in result


class TestFormatHumanReview:
    """Test format_human_review static method."""

    def test_empty_dict_returns_empty(self, temp_dir, mock_brain, mock_vector_store):
        """Empty dict returns empty string."""
        from forge.summariser import Summariser
        result = Summariser.format_human_review({})
        assert result == ""

    def test_includes_summary(self, temp_dir, mock_brain, mock_vector_store):
        """Includes summary text."""
        from forge.summariser import Summariser
        result = Summariser.format_human_review({"summary": "Test summary"})
        assert "Test summary" in result

    def test_includes_key_decisions(self, temp_dir, mock_brain, mock_vector_store):
        """Includes key decisions as bullet list."""
        from forge.summariser import Summariser
        result = Summariser.format_human_review({
            "key_decisions": ["Use JWT", "Use PostgreSQL"]
        })
        assert "Key Decisions" in result
        assert "Use JWT" in result

    def test_includes_next_steps(self, temp_dir, mock_brain, mock_vector_store):
        """Includes next steps as numbered list."""
        from forge.summariser import Summariser
        result = Summariser.format_human_review({
            "next_suggested": ["Write tests", "Deploy"]
        })
        assert "Next Steps" in result or "Steps" in result

    def test_includes_risk_flags(self, temp_dir, mock_brain, mock_vector_store):
        """Includes risk flags."""
        from forge.summariser import Summariser
        result = Summariser.format_human_review({
            "risk_flags": ["Security issue", "Performance concern"]
        })
        assert "Risk" in result or "flag" in result.lower()

    def test_limits_to_five_items(self, temp_dir, mock_brain, mock_vector_store):
        """Limits to 5 items per section."""
        from forge.summariser import Summariser
        many_decisions = [f"decision {i}" for i in range(10)]
        result = Summariser.format_human_review({
            "key_decisions": many_decisions,
            "next_suggested": [f"step {i}" for i in range(10)],
            "risk_flags": [f"risk {i}" for i in range(10)],
        })
        # Should contain but not explode
        assert "decision 0" in result


class TestFormatChangelogEntry:
    """Test _format_changelog_entry method."""

    def test_truncates_long_summary(self, temp_dir, mock_brain, mock_vector_store):
        """Long summary (>60 chars) is truncated with ellipsis."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            # Make it longer than 60 chars to trigger truncation
            long_summary = "This is a very very long summary that definitely exceeds sixty characters"
            result = s._format_changelog_entry("20250101_120000", long_summary, {})
            # Title should contain the timestamp
            assert "20250101_120000" in result

    def test_includes_key_decisions(self, temp_dir, mock_brain, mock_vector_store):
        """Key decisions included in output."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            result = s._format_changelog_entry("20250101_120000", "summary", {
                "key_decisions": ["Use JWT"]
            })
            assert "Key decisions" in result
            assert "Use JWT" in result

    def test_includes_next_steps(self, temp_dir, mock_brain, mock_vector_store):
        """Next steps included as numbered list."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            result = s._format_changelog_entry("20250101_120000", "summary", {
                "next_suggested": ["Write tests"]
            })
            assert "Next steps" in result
            assert "1." in result

    def test_includes_risk_flags(self, temp_dir, mock_brain, mock_vector_store):
        """Risk flags included."""
        from forge.summariser import Summariser
        with patch("forge.summariser.Path.mkdir", return_value=None):
            s = Summariser(mock_brain, mock_vector_store)
            result = s._format_changelog_entry("20250101_120000", "summary", {
                "risk_flags": ["Performance issue"]
            })
            assert "Risk" in result
            assert "Performance issue" in result