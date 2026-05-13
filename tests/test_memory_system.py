"""tests/test_memory_system.py

Unit tests for the persistent cross-session memory system:
  - PersistentMemory  (persistent_memory.py)
  - SessionLogger     (session_logger.py)
  - HandoffPacket     (handoff.py)

Strategy
--------
• FAISS and SentenceTransformer are mocked so tests run without GPU/model
  download overhead and remain fully offline.
• tmp_path (built-in pytest fixture) is used for every file I/O test so
  there are zero leftover files on disk.
• Each test class is isolated — a fresh PersistentMemory is created per test.
"""

from __future__ import annotations

import gzip
import json
import pickle
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared mock factories
# ---------------------------------------------------------------------------

def _make_faiss_index(ntotal: int = 0):
    """Return a MagicMock that behaves enough like faiss.IndexFlatIP."""
    idx = MagicMock()
    idx.ntotal = ntotal
    idx.d = 384

    def _add(vec):
        idx.ntotal += 1

    idx.add.side_effect = _add
    # search returns (scores, ids) arrays shaped (1, k)
    idx.search.return_value = (
        np.array([[0.9, 0.8]], dtype="float32"),
        np.array([[0, 1]], dtype="int64"),
    )
    return idx


def _make_st_model():
    """Return a MagicMock SentenceTransformer that returns a fixed 384-d vector."""
    m = MagicMock()
    m.encode.return_value = np.ones(384, dtype="float32")
    return m


# ---------------------------------------------------------------------------
# Autouse patch: prevent real FAISS / sentence-transformers from loading
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_heavyweights(monkeypatch):
    """Patch FAISS I/O and SentenceTransformer for the entire test session."""
    import forge.persistent_memory as pm_mod

    fake_idx = _make_faiss_index()
    monkeypatch.setattr(pm_mod.faiss, "IndexFlatIP", lambda dim: _make_faiss_index())
    monkeypatch.setattr(pm_mod.faiss, "write_index", lambda idx, path: None)
    monkeypatch.setattr(pm_mod.faiss, "read_index", lambda path: _make_faiss_index(ntotal=1))
    monkeypatch.setattr(pm_mod, "SentenceTransformer", lambda name: _make_st_model())


# ===========================================================================
# TestPersistentMemory
# ===========================================================================

class TestPersistentMemory:
    """Tests for PersistentMemory class."""

    # ── helpers ──────────────────────────────────────────────────────────────

    @pytest.fixture()
    def pm(self, tmp_path):
        from forge.persistent_memory import PersistentMemory
        return PersistentMemory(str(tmp_path))

    # ── init ─────────────────────────────────────────────────────────────────

    def test_init_creates_memory_directory(self, tmp_path):
        from forge.persistent_memory import PersistentMemory
        PersistentMemory(str(tmp_path))
        assert (tmp_path / ".forge" / "memory").is_dir()
        assert (tmp_path / ".forge" / "memory" / "sessions").is_dir()

    def test_init_creates_sqlite_tables(self, pm):
        cur = pm.db.cursor()
        tables = {
            r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"memories", "sessions", "handoffs"} <= tables

    # ── remember ─────────────────────────────────────────────────────────────

    def test_remember_returns_uuid_string(self, pm):
        cid = pm.remember("hello world", category="note")
        assert isinstance(cid, str)
        assert len(cid) == 36  # standard UUID4

    def test_remember_inserts_sqlite_row(self, pm):
        pm.remember("decision text", category="decision", source="user")
        row = pm.db.execute("SELECT category, source FROM memories").fetchone()
        assert row == ("decision", "user")

    def test_remember_stores_correct_decay_rate_for_decision(self, pm):
        pm.remember("arch decision", category="decision")
        row = pm.db.execute("SELECT decay_rate FROM memories").fetchone()
        assert row[0] == pytest.approx(0.03)

    def test_remember_stores_correct_decay_rate_for_note(self, pm):
        pm.remember("quick note", category="note")
        row = pm.db.execute("SELECT decay_rate FROM memories").fetchone()
        assert row[0] == pytest.approx(0.20)

    @pytest.mark.parametrize("category,expected_rate", [
        ("requirement", 0.02),
        ("decision",    0.03),
        ("error",       0.10),
        ("code",        0.15),
        ("note",        0.20),
    ])
    def test_decay_rates_by_category(self, pm, category, expected_rate):
        pm.remember("text", category=category)
        row = pm.db.execute("SELECT decay_rate FROM memories").fetchone()
        assert row[0] == pytest.approx(expected_rate)

    def test_remember_text_is_compressed_not_raw(self, pm):
        """Raw text must NEVER appear uncompressed in meta."""
        secret = "JWT secret key: super-private"
        pm.remember(secret, category="code")
        # meta stores text_gz (base64+gzip), not the plain string
        for record in pm.meta.values():
            assert secret not in record.get("text_gz", "")

    def test_remember_multiple_items_increments_index(self, pm):
        pm.remember("one", category="note")
        pm.remember("two", category="note")
        assert pm.index.ntotal == 2

    # ── recall ────────────────────────────────────────────────────────────────

    def test_recall_empty_store_returns_empty_list(self, pm):
        results = pm.recall("anything")
        assert results == []

    def test_recall_returns_list_of_dicts(self, pm):
        pm.remember("JWT auth with RS256", category="decision")
        pm.remember("PostgreSQL schema", category="requirement")
        results = pm.recall("authentication", top_k=5)
        assert isinstance(results, list)
        for r in results:
            assert {"id", "category", "text", "priority", "score", "source"} <= r.keys()

    def test_recall_category_filter(self, pm):
        """Results filtered by category only contain that category."""
        pm.remember("decision content", category="decision")
        pm.remember("note content", category="note")
        # patch search to return both indices
        pm.index.search.return_value = (
            np.array([[0.9, 0.8]], dtype="float32"),
            np.array([[0, 1]], dtype="int64"),
        )
        results = pm.recall("query", top_k=10, category="decision")
        assert all(r["category"] == "decision" for r in results)

    def test_recall_updates_access_count(self, pm):
        pm.remember("important fact", category="note")
        pm.recall("important fact", top_k=5)
        row = pm.db.execute("SELECT access_count FROM memories").fetchone()
        assert row[0] >= 1

    # ── forget ────────────────────────────────────────────────────────────────

    def test_forget_removes_from_sqlite(self, pm):
        cid = pm.remember("to delete", category="note")
        pm.forget(cid)
        row = pm.db.execute("SELECT id FROM memories WHERE id=?", (cid,)).fetchone()
        assert row is None

    def test_forget_removes_from_meta(self, pm):
        cid = pm.remember("to delete", category="note")
        pm.forget(cid)
        assert cid not in pm.meta

    def test_forget_unknown_id_does_not_raise(self, pm):
        pm.forget("non-existent-id-123")  # must not raise

    # ── decay_scores ──────────────────────────────────────────────────────────

    def test_decay_scores_reduces_priority(self, pm):
        pm.remember("old memory", category="note")
        # Backdate last_accessed by 10 days
        pm.db.execute(
            "UPDATE memories SET last_accessed=?",
            (time.time() - 10 * 86400,),
        )
        pm.db.commit()

        pm.decay_scores()

        row = pm.db.execute("SELECT priority FROM memories").fetchone()
        assert row[0] < 1.0  # priority should have dropped

    def test_decay_scores_floor_is_0_05(self, pm):
        pm.remember("ancient memory", category="note")
        # Backdate 10000 days — should hit floor
        pm.db.execute(
            "UPDATE memories SET last_accessed=?, priority=0.06",
            (time.time() - 10000 * 86400,),
        )
        pm.db.commit()

        pm.decay_scores()

        row = pm.db.execute("SELECT priority FROM memories").fetchone()
        assert row[0] >= 0.05

    def test_decay_scores_decision_decays_slower_than_note(self, pm):
        """decision (0.03/day) should retain more priority than note (0.20/day) after 5 days."""
        pm.remember("arch decision", category="decision")
        pm.remember("random note", category="note")
        backdate = time.time() - 5 * 86400
        pm.db.execute("UPDATE memories SET last_accessed=?", (backdate,))
        pm.db.commit()

        pm.decay_scores()

        rows = pm.db.execute(
            "SELECT category, priority FROM memories ORDER BY category"
        ).fetchall()
        by_cat = {r[0]: r[1] for r in rows}
        assert by_cat["decision"] > by_cat["note"]

    # ── get_session_context ───────────────────────────────────────────────────

    def test_get_session_context_empty(self, pm):
        ctx = pm.get_session_context()
        assert ctx == ""

    def test_get_session_context_returns_formatted_string(self, pm):
        import time as _t
        pm.write_session_record(
            session_id="s1",
            started_at=_t.time() - 86400,
            ended_at=_t.time(),
            goal="add auth",
            tasks_completed=5,
            tasks_blocked=0,
            files_changed=["auth.py"],
            model_used="gemini-2.0-flash",
            summary="Built JWT auth module",
        )
        ctx = pm.get_session_context(last_n_sessions=3)
        assert "=== PREVIOUS SESSION CONTEXT ===" in ctx
        assert "add auth" in ctx
        assert "auth.py" in ctx
        assert "=== END SESSION CONTEXT ===" in ctx

    # ── write_session_record ──────────────────────────────────────────────────

    def test_write_session_record_inserts_row(self, pm):
        pm.write_session_record(
            session_id="abc",
            started_at=1000.0,
            ended_at=2000.0,
            goal="build thing",
            tasks_completed=3,
            tasks_blocked=1,
            files_changed=["a.py", "b.py"],
            model_used="gemini",
            summary="done",
        )
        row = pm.db.execute("SELECT goal, tasks_completed FROM sessions WHERE id='abc'").fetchone()
        assert row == ("build thing", 3)

    def test_write_session_record_compresses_summary(self, pm):
        pm.write_session_record(
            session_id="xyz",
            started_at=0.0,
            ended_at=1.0,
            goal="g",
            tasks_completed=0,
            tasks_blocked=0,
            files_changed=[],
            model_used="m",
            summary="This is the summary text",
        )
        row = pm.db.execute("SELECT summary_gz FROM sessions WHERE id='xyz'").fetchone()
        # summary_gz must be decompressible and yield the original text
        recovered = pm._decompress(row[0])
        assert recovered == "This is the summary text"

    # ── save_state / load_state ───────────────────────────────────────────────

    def test_save_state_does_not_raise_on_mock_objects(self, pm):
        vs_mock = MagicMock()
        vs_mock.index = None          # no FAISS index to save
        graph_mock = MagicMock()
        graph_mock.graph = None
        pm.save_state(vs_mock, graph_mock)   # must not raise

    def test_load_state_returns_false_when_no_files(self, pm):
        ok, idx, graph = pm.load_state()
        assert ok is False
        assert idx is None
        assert graph is None

    def test_load_state_returns_true_when_graph_exists(self, tmp_path):
        from forge.persistent_memory import PersistentMemory
        import networkx as nx
        pm = PersistentMemory(str(tmp_path))
        # Manually write a valid pickled graph
        g = nx.DiGraph()
        g.add_node("a")
        with open(pm.root / "graph.pkl", "wb") as f:
            pickle.dump(g, f, protocol=5)

        ok, idx, loaded_graph = pm.load_state()
        assert ok is True
        assert loaded_graph.number_of_nodes() == 1

    # ── stats ─────────────────────────────────────────────────────────────────

    def test_stats_keys(self, pm):
        st = pm.stats()
        assert {"memories", "sessions", "handoffs", "categories", "index_vectors"} <= st.keys()

    def test_stats_counts_correctly(self, pm):
        pm.remember("a", category="decision")
        pm.remember("b", category="note")
        st = pm.stats()
        assert st["memories"] == 2
        assert st["categories"]["decision"] == 1
        assert st["categories"]["note"] == 1


# ===========================================================================
# TestSessionLogger
# ===========================================================================

class TestSessionLogger:
    """Tests for SessionLogger class."""

    @pytest.fixture()
    def mem_root(self, tmp_path) -> Path:
        root = tmp_path / ".forge" / "memory"
        root.mkdir(parents=True)
        return root

    @pytest.fixture()
    def sl(self, mem_root):
        from forge.session_logger import SessionLogger
        return SessionLogger(mem_root, goal="test goal", model="gemini-2.0-flash")

    # ── init ─────────────────────────────────────────────────────────────────

    def test_init_creates_sessions_dir(self, mem_root):
        from forge.session_logger import SessionLogger
        SessionLogger(mem_root, goal="g", model="m")
        assert (mem_root / "sessions").is_dir()

    def test_init_has_valid_session_id(self, sl):
        assert isinstance(sl.session_id, str)
        assert len(sl.session_id) > 0

    # ── log ───────────────────────────────────────────────────────────────────

    def test_log_buffers_events(self, sl):
        sl.log("task_started", {"task_id": 1, "description": "do thing"})
        assert len(sl._buf) >= 1  # model_used is logged in __init__ too

    def test_log_flushes_on_10_events(self, sl):
        # Already has 1 event (model_used from __init__), add 9 more
        for i in range(9):
            sl.log("user_note", {"text": f"note {i}"})
        # After 10th event the buffer should have been flushed
        assert sl._jsonl_path.exists()

    def test_log_tracks_completed_tasks(self, sl):
        sl.log("task_completed", {"task_id": 1, "score": 9, "files_changed": ["a.py"]})
        assert sl._tasks_completed == 1
        assert "a.py" in sl._files_changed

    def test_log_tracks_blocked_tasks(self, sl):
        sl.log("task_blocked", {"task_id": 2, "reason": "LM Studio offline"})
        assert sl._tasks_blocked == 1

    def test_log_tracks_file_changes(self, sl):
        sl.log("file_changed", {"path": "src/auth.py"})
        assert "src/auth.py" in sl._files_changed

    def test_log_deduplicates_files(self, sl):
        sl.log("file_changed", {"path": "src/auth.py"})
        sl.log("file_changed", {"path": "src/auth.py"})
        assert sl._files_changed.count("src/auth.py") == 1

    def test_log_after_close_is_no_op(self, sl):
        sl.close(summary="done")
        sl.log("user_note", {"text": "post-close note"})  # must not raise or write

    # ── close ─────────────────────────────────────────────────────────────────

    def test_close_creates_gz_file(self, sl):
        gz = sl.close(summary="all done")
        assert Path(gz).exists()
        assert gz.endswith(".gz")

    def test_close_gz_is_valid_gzip(self, sl):
        gz = sl.close(summary="done")
        raw = gzip.decompress(Path(gz).read_bytes())
        assert isinstance(raw, bytes)

    def test_close_deletes_raw_jsonl(self, sl):
        # Flush something to disk first
        sl.log("user_note", {"text": "hi"})
        sl._flush()
        raw_path = sl._jsonl_path
        sl.close(summary="done")
        assert not raw_path.exists()

    def test_close_idempotent(self, sl):
        gz1 = sl.close(summary="first")
        gz2 = sl.close(summary="second")  # must not raise
        assert gz1 == gz2

    def test_close_writes_sqlite_record_via_memory(self, tmp_path, mem_root):
        from forge.persistent_memory import PersistentMemory
        from forge.session_logger import SessionLogger
        pm = PersistentMemory(str(tmp_path))
        sl = SessionLogger(mem_root, goal="goal", model="m", memory=pm)
        sl.log("task_completed", {"task_id": 1, "score": 8, "files_changed": []})
        sl.close(summary="done")

        row = pm.db.execute("SELECT goal, tasks_completed FROM sessions").fetchone()
        assert row is not None
        assert row[0] == "goal"
        assert row[1] == 1

    # ── context manager ───────────────────────────────────────────────────────

    def test_context_manager_closes_on_exit(self, mem_root):
        from forge.session_logger import SessionLogger
        with SessionLogger(mem_root, goal="ctx test", model="m") as sl:
            sl.log("user_note", {"text": "inside context"})
        assert sl._closed is True

    # ── provider inference ────────────────────────────────────────────────────

    def test_provider_inference_gemini(self):
        from forge.session_logger import _infer_provider
        assert _infer_provider("gemini-2.0-flash") == "gemini"

    def test_provider_inference_anthropic(self):
        from forge.session_logger import _infer_provider
        assert _infer_provider("claude-3-sonnet") == "anthropic"

    def test_provider_inference_local(self):
        from forge.session_logger import _infer_provider
        assert _infer_provider("qwen3.5-9b-instruct") == "local"


# ===========================================================================
# TestHandoffPacket
# ===========================================================================

class TestHandoffPacket:
    """Tests for HandoffPacket class."""

    @pytest.fixture()
    def memory(self, tmp_path):
        from forge.persistent_memory import PersistentMemory
        return PersistentMemory(str(tmp_path))

    @pytest.fixture()
    def brain(self):
        b = MagicMock()
        b.summarise = MagicMock(return_value={
            "summary": "Summarised sessions",
            "next_suggested": ["Write tests"],
            "risk_flags": [],
        })
        # Make it awaitable
        import asyncio

        async def _summarise(*a, **kw):
            return b.summarise.return_value

        b.summarise = _summarise
        return b

    @pytest.fixture()
    def hp(self, memory, brain):
        from forge.handoff import HandoffPacket
        return HandoffPacket(memory=memory, brain=brain)

    # ── to_prompt_prefix ─────────────────────────────────────────────────────

    def test_to_prompt_prefix_has_header_and_footer(self, hp):
        packet = {
            "project": {"root": "/proj", "language": "python", "framework": "fastapi", "files_count": 10},
            "sessions_summary": "Built auth.",
            "completed_tasks": ["Task 1"],
            "pending_tasks": [],
            "blocked_tasks": [],
            "key_decisions": ["Use JWT"],
            "top_memories": [],
            "next_recommended": "Write tests",
            "warnings": [],
            "generated_at": "2026-01-01T00:00:00",
        }
        prefix = hp.to_prompt_prefix(packet)
        assert "=== FORGE PROJECT CONTEXT ===" in prefix
        assert "=== END FORGE CONTEXT ===" in prefix

    def test_to_prompt_prefix_contains_project_info(self, hp):
        packet = {
            "project": {"root": "/my/proj", "language": "python", "framework": "cli", "files_count": 42},
            "sessions_summary": "3 sessions.",
            "completed_tasks": [],
            "pending_tasks": [],
            "blocked_tasks": [],
            "key_decisions": [],
            "top_memories": [],
            "next_recommended": "",
            "warnings": [],
            "generated_at": "2026-01-01T00:00:00",
        }
        prefix = hp.to_prompt_prefix(packet)
        assert "/my/proj" in prefix
        assert "42 files" in prefix

    def test_to_prompt_prefix_shows_key_decisions(self, hp):
        packet = {
            "project": {"root": "/p", "language": "python", "framework": "", "files_count": 5},
            "sessions_summary": "",
            "completed_tasks": [],
            "pending_tasks": [],
            "blocked_tasks": [],
            "key_decisions": ["PostgreSQL over SQLite", "JWT RS256"],
            "top_memories": [],
            "next_recommended": "",
            "warnings": [],
            "generated_at": "2026-01-01T00:00:00",
        }
        prefix = hp.to_prompt_prefix(packet)
        assert "PostgreSQL over SQLite" in prefix
        assert "JWT RS256" in prefix

    def test_to_prompt_prefix_shows_next_recommended(self, hp):
        packet = {
            "project": {"root": "/p", "language": "python", "framework": "", "files_count": 0},
            "sessions_summary": "",
            "completed_tasks": [], "pending_tasks": [], "blocked_tasks": [],
            "key_decisions": [], "top_memories": [],
            "next_recommended": "Write unit tests for auth module",
            "warnings": [],
            "generated_at": "2026-01-01T00:00:00",
        }
        prefix = hp.to_prompt_prefix(packet)
        assert "Write unit tests for auth module" in prefix

    # ── save / load ───────────────────────────────────────────────────────────

    def test_save_creates_handoff_gz(self, hp, tmp_path):
        packet = {"format_version": "1.0", "target_agent": "any", "generated_at": "now"}
        path = hp.save(packet)
        assert path.exists()
        assert path.name == "handoff.gz"

    def test_save_gz_is_valid_gzip(self, hp):
        packet = {"format_version": "1.0", "target_agent": "any", "generated_at": "now"}
        path = hp.save(packet)
        raw = gzip.decompress(path.read_bytes())
        data = json.loads(raw)
        assert data["format_version"] == "1.0"

    def test_save_writes_handoffs_table(self, hp, memory):
        packet = {"format_version": "1.0", "target_agent": "cursor", "generated_at": "t"}
        hp.save(packet)
        row = memory.db.execute("SELECT target_agent FROM handoffs").fetchone()
        assert row[0] == "cursor"

    def test_save_marks_previous_handoffs_used(self, hp, memory):
        packet = {"format_version": "1.0", "target_agent": "any", "generated_at": "t"}
        hp.save(packet)
        hp.save(packet)   # second save → first should be marked used
        rows = memory.db.execute("SELECT used FROM handoffs ORDER BY id").fetchall()
        assert rows[0][0] == 1   # first handoff marked used

    def test_load_returns_none_when_no_file(self, hp):
        result = hp.load()
        assert result is None

    def test_load_returns_dict_after_save(self, hp):
        packet = {"format_version": "1.0", "target_agent": "forge", "generated_at": "now"}
        hp.save(packet)
        loaded = hp.load()
        assert loaded is not None
        assert loaded["format_version"] == "1.0"
        assert loaded["target_agent"] == "forge"

    # ── _detect_project_info ──────────────────────────────────────────────────

    def test_detect_project_info_detects_python(self, tmp_path):
        from forge.handoff import _detect_project_info
        (tmp_path / "main.py").write_text("# hello")
        info = _detect_project_info(str(tmp_path))
        assert info["language"] == "python"

    def test_detect_project_info_detects_fastapi(self, tmp_path):
        from forge.handoff import _detect_project_info
        (tmp_path / "main.py").write_text("# hello")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='app'\n\n[tool.fastapi]\nfoo=1")
        info = _detect_project_info(str(tmp_path))
        assert info["framework"] == "fastapi"

    def test_detect_project_info_has_required_keys(self, tmp_path):
        from forge.handoff import _detect_project_info
        info = _detect_project_info(str(tmp_path))
        assert {"root", "language", "framework", "files_count", "last_modified"} <= info.keys()


# ===========================================================================
# TestCompressionPrimitives
# ===========================================================================

class TestCompressionPrimitives:
    """Round-trip tests for gzip/base64 compress / decompress."""

    @pytest.fixture()
    def pm(self, tmp_path):
        from forge.persistent_memory import PersistentMemory
        return PersistentMemory(str(tmp_path))

    def test_compress_decompress_round_trip(self, pm):
        original = "Hello, this is a test string with unicode: 日本語"
        assert pm._decompress(pm._compress(original)) == original

    def test_compress_output_is_string(self, pm):
        assert isinstance(pm._compress("text"), str)

    def test_compress_reduces_repetitive_text(self, pm):
        repetitive = "abcdefg " * 200
        compressed = pm._compress(repetitive)
        # base64 of gzip should be much smaller than raw
        assert len(compressed) < len(repetitive)

    def test_empty_string_round_trip(self, pm):
        assert pm._decompress(pm._compress("")) == ""
