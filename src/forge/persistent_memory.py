"""PersistentMemory — cross-session memory store for forge.

Persists a FAISS vector index, NetworkX project graph, and per-session logs
to .forge/memory/.  All text payloads are stored gzip-compressed; metadata
uses msgpack binary format; the graph uses pickle protocol 5.
"""

from __future__ import annotations

import base64
import gzip
import logging
import pickle
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import faiss
import msgpack
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# Per-day decay rates by memory category
_DECAY_RATES: dict[str, float] = {
    "requirement": 0.02,
    "decision": 0.03,
    "error": 0.10,
    "code": 0.15,
    "note": 0.20,
}
_DECAY_FLOOR = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# PersistentMemory
# ──────────────────────────────────────────────────────────────────────────────

class PersistentMemory:
    """Cross-session memory store.

    Persists FAISS index, graph, and session history to .forge/memory/.
    Uses compressed binary formats throughout — no uncompressed text on disk.

    Usage::
        pm = PersistentMemory("/path/to/project")
        chunk_id = pm.remember("JWT with RS256", category="decision")
        results  = pm.recall("authentication approach", top_k=5)
    """

    def __init__(self, project_root: str) -> None:
        self.root = Path(project_root) / ".forge" / "memory"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sessions").mkdir(exist_ok=True)

        self.db: sqlite3.Connection = sqlite3.connect(
            str(self.root / "memory.db"), check_same_thread=False
        )
        self._init_db()

        self._model: SentenceTransformer | None = None  # lazy
        self.index: faiss.IndexFlatIP | None = None     # lazy
        self.meta: dict[str, dict] = {}                  # chunk_id → record
        self.graph: nx.DiGraph | None = None             # lazy
        # FIX: Reverse lookup for O(1) FAISS ID to chunk_id mapping
        self._faiss_idx_to_chunk: dict[int, str] = {}

    # ── Model ─────────────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def _embed(self, text: str) -> np.ndarray:
        vec = self._get_model().encode(text, convert_to_numpy=True).astype("float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ── Compression primitives ─────────────────────────────────────────────────

    def _compress(self, text: str) -> str:
        """gzip text → base64 string.  Used for all text payloads."""
        compressed = gzip.compress(text.encode("utf-8"), compresslevel=6)
        return base64.b64encode(compressed).decode("ascii")

    def _decompress(self, b64: str) -> str:
        """Reverse of _compress."""
        return gzip.decompress(base64.b64decode(b64)).decode("utf-8")

    # ── Metadata (msgpack) ─────────────────────────────────────────────────────

    def _save_meta(self) -> None:
        """Write metadata dict to msgpack binary file."""
        path = self.root / "hot.meta.msgpack"
        with open(path, "wb") as f:
            f.write(msgpack.packb(self.meta, use_bin_type=True))

    def _load_meta(self) -> None:
        """Load metadata dict from msgpack binary file."""
        path = self.root / "hot.meta.msgpack"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self.meta = msgpack.unpackb(f.read(), raw=False)
            except Exception as e:
                logger.warning(f"Failed to load meta: {e} — starting fresh")
                self.meta = {}
        else:
            self.meta = {}

    # ── FAISS index ────────────────────────────────────────────────────────────

    def _save_index(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.root / "hot.faiss"))

    def _load_index(self) -> None:
        path = self.root / "hot.faiss"
        if path.exists():
            try:
                self.index = faiss.read_index(str(path))
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e} — starting fresh")
                self.index = None

    def _ensure_index(self) -> faiss.IndexFlatIP:
        if self.index is None:
            self.index = faiss.IndexFlatIP(EMBED_DIM)
        return self.index

    # ── Graph (pickle) ─────────────────────────────────────────────────────────

    def _save_graph(self) -> None:
        """Pickle the NetworkX graph with protocol 5."""
        if self.graph is None:
            return
        with open(self.root / "graph.pkl", "wb") as f:
            pickle.dump(self.graph, f, protocol=5)

    def _load_graph(self) -> bool:
        """Load pickled graph.  Returns False if unpickling fails."""
        path = self.root / "graph.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                self.graph = pickle.load(f)
            return True
        except Exception as e:
            logger.warning(f"Failed to unpickle graph: {e} — caller should re-parse")
            return False

    # ── SQLite schema ──────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        cur = self.db.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                file_path   TEXT,
                category    TEXT,
                priority    REAL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                created_at  REAL,
                decay_rate  REAL,
                source      TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                started_at  REAL,
                ended_at    REAL,
                goal        TEXT,
                tasks_completed INTEGER DEFAULT 0,
                tasks_blocked   INTEGER DEFAULT 0,
                files_changed   TEXT,
                model_used  TEXT,
                summary_gz  TEXT
            );

            CREATE TABLE IF NOT EXISTS handoffs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  REAL,
                target_agent TEXT,
                packet_gz   TEXT,
                used        INTEGER DEFAULT 0
            );
        """)
        self.db.commit()

    # ── Lazy initialisation ────────────────────────────────────────────────────

    def _lazy_load(self) -> None:
        """Load index + metadata from disk if not yet loaded."""
        if self.index is None:
            self._load_index()
        if not self.meta:
            self._load_meta()

    # ── Public API ─────────────────────────────────────────────────────────────

    def remember(
        self,
        text: str,
        category: str = "note",
        source: str = "auto",
        tags: list[str] | None = None,  # noqa: F841 — reserved for future tag search
        file_path: str = "",
    ) -> str:
        """Store a piece of information in persistent memory.

        Text is embedded, compressed, and stored. Raw text is NEVER written to disk.
        Returns the chunk id.
        """
        self._lazy_load()

        chunk_id = str(uuid.uuid4())
        vec = self._embed(text)

        idx = self._ensure_index()
        current_idx = idx.ntotal
        idx.add(vec.reshape(1, -1))

        text_gz = self._compress(text)
        now = time.time()
        decay_rate = _DECAY_RATES.get(category, 0.15)

        self.meta[chunk_id] = {
            "id": chunk_id,
            "file_path": file_path,
            "chunk_id": chunk_id,
            "text_gz": text_gz,
            "indexed_at": now,
            "access_count": 0,
            "faiss_idx": current_idx,
        }

        # FIX: Update reverse lookup
        self._faiss_idx_to_chunk[current_idx] = chunk_id

        self._save_index()
        self._save_meta()

        cur = self.db.cursor()
        cur.execute(
            """INSERT INTO memories
               (id, file_path, category, priority, access_count,
                last_accessed, created_at, decay_rate, source)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (chunk_id, file_path, category, 1.0, 0, now, now, decay_rate, source),
        )
        self.db.commit()

        logger.info(f"Remembered chunk {chunk_id} [{category}]")
        return chunk_id

    def recall(
        self,
        query: str,
        top_k: int = 8,
        category: str | None = None,
    ) -> list[dict]:
        """Semantic search over persistent memory.

        Returns list of {id, category, text, priority, score, source}.
        Text is decompressed on retrieval — never stored uncompressed.
        """
        self._lazy_load()

        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self._embed(query).reshape(1, -1)
        k = min(top_k * 3, self.index.ntotal)  # over-fetch for category filter
        scores, ids = self.index.search(query_vec, k)

        now = time.time()
        results: list[dict] = []

        for raw_score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id < 0:
                continue

            # FIX: O(1) lookup using reverse index instead of O(n) iteration
            chunk_id = self._faiss_idx_to_chunk.get(int(faiss_id))

            if chunk_id is None:
                continue

            # Fetch SQLite row
            cur = self.db.cursor()
            row = cur.execute(
                "SELECT id, category, priority, access_count, last_accessed, decay_rate, source "
                "FROM memories WHERE id=?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                continue

            rid, cat, priority, access_count, last_accessed, decay_rate, source = row

            if category and cat != category:
                continue

            # Apply decay
            days_since = (now - last_accessed) / 86400 if last_accessed else 0
            effective_score = float(raw_score) * max(
                _DECAY_FLOOR,
                priority * (1.0 - decay_rate * days_since),
            )

            # Update access stats
            cur.execute(
                "UPDATE memories SET access_count=?, last_accessed=? WHERE id=?",
                (access_count + 1, now, chunk_id),
            )

            # Decompress text
            text_gz = self.meta[chunk_id].get("text_gz", "")
            try:
                text = self._decompress(text_gz) if text_gz else ""
            except Exception:
                text = ""

            results.append({
                "id": chunk_id,
                "category": cat,
                "text": text,
                "priority": priority,
                "score": effective_score,
                "source": source,
            })

            if len(results) >= top_k:
                break

        self.db.commit()
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def forget(self, chunk_id: str) -> None:
        """Remove a memory entry.  Rebuilds FAISS index without this entry."""
        self._lazy_load()

        # Remove from SQLite
        self.db.execute("DELETE FROM memories WHERE id=?", (chunk_id,))
        self.db.commit()

        if chunk_id not in self.meta:
            return

        del self.meta[chunk_id]
        self._save_meta()

        # Rebuild FAISS index from remaining metadata
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Re-embed all kept chunks and rebuild the FAISS index."""
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        model = self._get_model()
        new_meta: dict[str, dict] = {}
        # FIX: Rebuild reverse lookup
        self._faiss_idx_to_chunk.clear()

        for chunk_id, record in self.meta.items():
            text_gz = record.get("text_gz", "")
            try:
                text = self._decompress(text_gz)
            except Exception:
                continue
            vec = model.encode(text, convert_to_numpy=True).astype("float32")
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            current_idx = self.index.ntotal
            self.index.add(vec.reshape(1, -1))
            record["faiss_idx"] = current_idx
            new_meta[chunk_id] = record
            # FIX: Update reverse lookup
            self._faiss_idx_to_chunk[current_idx] = chunk_id

        self.meta = new_meta
        self._save_index()  # FIX: Actually save the rebuilt index
        self._save_meta()

    def decay_scores(self) -> None:
        """Apply time-based decay to all memory entries.

        Called at session start to age out stale memories.
        Floor at 0.05 — never fully forget, just deprioritise.
        """
        now = time.time()
        cur = self.db.cursor()
        rows = cur.execute(
            "SELECT id, priority, last_accessed, decay_rate FROM memories"
        ).fetchall()

        for rid, priority, last_accessed, decay_rate in rows:
            days_since = (now - last_accessed) / 86400 if last_accessed else 0
            new_priority = max(
                _DECAY_FLOOR,
                priority * (1.0 - decay_rate * days_since),
            )
            cur.execute(
                "UPDATE memories SET priority=? WHERE id=?",
                (new_priority, rid),
            )

        self.db.commit()
        logger.info(f"Decayed {len(rows)} memory entries")

    def get_session_context(self, last_n_sessions: int = 3) -> str:
        """Reconstruct a compressed context string from the last N sessions.

        Returns a string suitable for inclusion in the brain's system prompt.
        """
        cur = self.db.cursor()
        rows = cur.execute(
            "SELECT id, started_at, ended_at, goal, tasks_completed, tasks_blocked, "
            "files_changed, model_used, summary_gz "
            "FROM sessions ORDER BY started_at DESC LIMIT ?",
            (last_n_sessions,),
        ).fetchall()

        if not rows:
            return ""

        lines: list[str] = ["=== PREVIOUS SESSION CONTEXT ==="]
        now = time.time()

        for row in reversed(rows):  # oldest first
            (sid, started_at, ended_at, goal, tasks_done, tasks_blocked,
             files_changed_json, model_used, summary_gz) = row

            days_ago = int((now - (started_at or now)) / 86400)
            age_str = f"{days_ago} day{'s' if days_ago != 1 else ''} ago" if days_ago > 0 else "today"
            lines.append(f"\n[{age_str} | goal: \"{goal or 'unknown'}\"]")
            lines.append(f"Completed: {tasks_done or 0} tasks. Blocked: {tasks_blocked or 0}.")

            if files_changed_json:
                try:
                    import json
                    files = json.loads(files_changed_json)
                    if files:
                        lines.append(f"Changed: {', '.join(files[:8])}")
                except Exception:
                    pass

            if model_used:
                lines.append(f"Model: {model_used}")

            if summary_gz:
                try:
                    summary = self._decompress(summary_gz)
                    lines.append(f"Summary: {summary[:300]}")
                except Exception:
                    pass

        lines.append("=== END SESSION CONTEXT ===")
        return "\n".join(lines)

    def write_session_record(
        self,
        session_id: str,
        started_at: float,
        ended_at: float,
        goal: str,
        tasks_completed: int,
        tasks_blocked: int,
        files_changed: list[str],
        model_used: str,
        summary: str,
    ) -> None:
        """Write a completed session record to SQLite.

        Called by SessionLogger.close() after the session ends.
        """
        import json
        summary_gz = self._compress(summary) if summary else ""
        files_json = json.dumps(files_changed)

        self.db.execute(
            """INSERT OR REPLACE INTO sessions
               (id, started_at, ended_at, goal, tasks_completed, tasks_blocked,
                files_changed, model_used, summary_gz)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (session_id, started_at, ended_at, goal, tasks_completed, tasks_blocked,
             files_json, model_used, summary_gz),
        )
        self.db.commit()

    def save_state(self, vector_store, graph) -> None:
        """Persist the current VectorStore and ProjectGraph into binary memory.

        Called at session end and at every checkpoint in forge auto.
        The existing .forge/vectors/ JSON-based saves are NOT replaced —
        this writes a secondary faster-loading binary copy.
        """
        try:
            # Save FAISS index from VectorStore
            vs_index = getattr(vector_store, "index", None)
            if vs_index is not None:
                faiss.write_index(vs_index, str(self.root / "hot.faiss"))
                logger.info("Saved VectorStore FAISS index to memory")
        except Exception as e:
            logger.warning(f"save_state: failed to save FAISS index: {e}")

        try:
            # Save graph
            g = getattr(graph, "graph", None)
            if g is not None:
                self.graph = g
                self._save_graph()
                logger.info("Saved ProjectGraph to memory")
        except Exception as e:
            logger.warning(f"save_state: failed to save graph: {e}")

    def load_state(self) -> tuple[bool, Any, Any]:
        """Restore VectorStore and ProjectGraph from binary memory.

        Returns (loaded: bool, vector_store_index, graph).
        If load fails, returns (False, None, None) — caller should re-index.
        """
        faiss_path = self.root / "hot.faiss"
        graph_path = self.root / "graph.pkl"

        if not faiss_path.exists() and not graph_path.exists():
            return False, None, None

        loaded_index = None
        loaded_graph = None

        try:
            if faiss_path.exists():
                loaded_index = faiss.read_index(str(faiss_path))
                logger.info(f"Loaded FAISS index from memory ({loaded_index.ntotal} vectors)")
        except Exception as e:
            logger.warning(f"load_state: failed to load FAISS index: {e}")

        try:
            if graph_path.exists():
                with open(graph_path, "rb") as f:
                    loaded_graph = pickle.load(f)
                logger.info(f"Loaded graph from memory ({loaded_graph.number_of_nodes()} nodes)")
        except Exception as e:
            logger.warning(f"load_state: failed to load graph: {e}")

        if loaded_index is None and loaded_graph is None:
            return False, None, None

        return True, loaded_index, loaded_graph

    def stats(self) -> dict:
        """Return memory statistics for display in forge memory-status."""
        self._lazy_load()
        cur = self.db.cursor()
        memories_count = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        sessions_count = cur.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        handoffs_count = cur.execute("SELECT COUNT(*) FROM handoffs").fetchone()[0]
        last_session = cur.execute(
            "SELECT goal, started_at FROM sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        categories = cur.execute(
            "SELECT category, COUNT(*) FROM memories GROUP BY category"
        ).fetchall()

        return {
            "memories": memories_count,
            "sessions": sessions_count,
            "handoffs": handoffs_count,
            "last_session": last_session,
            "categories": dict(categories),
            "index_vectors": self.index.ntotal if self.index else 0,
        }
