"""SessionLogger — per-session event recorder for forge.

Records everything that happens in a session to a gzipped JSONL file.
Lightweight: only appends events, does not read during the session.
On close(), the raw JSONL is gzipped and a record is written to SQLite.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _infer_provider(model: str) -> str:
    m = model.lower()
    if "gemini" in m:
        return "gemini"
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "openai" in m:
        return "openai"
    return "local"


class SessionLogger:
    """Records all events in the current session to a gzipped JSONL file."""

    EVENT_TYPES = frozenset({
        "task_started", "task_completed", "task_blocked",
        "file_changed", "brain_decision", "user_note",
        "model_used", "checkpoint",
    })

    def __init__(self, memory_root: Path, goal: str, model: str, memory=None) -> None:
        self.session_id = datetime.now(timezone.utc).isoformat()
        sessions_dir = memory_root / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        safe_ts = self.session_id.replace(":", "-").replace("+", "p")
        self._jsonl_path = sessions_dir / f"{safe_ts}.session.jsonl"

        self.goal = goal
        self.model = model
        self.memory = memory
        self.started_at = time.time()

        self._buf: list[dict] = []
        self._tasks_completed = 0
        self._tasks_blocked = 0
        self._files_changed: list[str] = []
        self._closed = False

        self.log("model_used", {"model_name": model, "provider": _infer_provider(model)})

    def log(self, event_type: str, data: dict) -> None:
        """Append an event to the session log buffer."""
        if self._closed:
            return

        if event_type == "task_completed":
            self._tasks_completed += 1
            for fp in data.get("files_changed", []):
                if fp not in self._files_changed:
                    self._files_changed.append(fp)
        elif event_type == "task_blocked":
            self._tasks_blocked += 1
        elif event_type == "file_changed":
            fp = data.get("path", "")
            if fp and fp not in self._files_changed:
                self._files_changed.append(fp)

        self._buf.append({"ts": time.time(), "type": event_type, "data": data})
        if len(self._buf) >= 10:
            self._flush()

    def close(self, summary: str = "") -> str:
        """Flush, gzip the JSONL, write SQLite session record. Returns gz path."""
        if self._closed:
            return str(self._jsonl_path.with_suffix(".gz"))

        self._flush()
        self._closed = True
        ended_at = time.time()

        gz_path = self._jsonl_path.with_suffix(".gz")
        try:
            if self._jsonl_path.exists():
                raw = self._jsonl_path.read_bytes()
                gz_path.write_bytes(gzip.compress(raw, compresslevel=6))
                self._jsonl_path.unlink()
            else:
                gz_path.write_bytes(gzip.compress(b"", compresslevel=6))
        except Exception as e:
            logger.warning(f"SessionLogger.close: gzip failed: {e}")

        if self.memory is not None:
            try:
                self.memory.write_session_record(
                    session_id=self.session_id,
                    started_at=self.started_at,
                    ended_at=ended_at,
                    goal=self.goal,
                    tasks_completed=self._tasks_completed,
                    tasks_blocked=self._tasks_blocked,
                    files_changed=self._files_changed,
                    model_used=self.model,
                    summary=summary,
                )
            except Exception as e:
                logger.warning(f"SessionLogger.close: SQLite write failed: {e}")

        logger.info(f"Session closed → {gz_path}")
        return str(gz_path)

    def _flush(self) -> None:
        if not self._buf:
            return
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                for event in self._buf:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._buf.clear()
        except Exception as e:
            logger.warning(f"SessionLogger._flush: {e}")

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._closed:
            self.close(summary="Session ended via context manager.")
