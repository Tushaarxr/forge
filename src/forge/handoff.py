"""HandoffPacket — cross-agent context packet generator for forge.

Generates a compressed JSON context object that any AI agent (forge, cursor,
claude-code, anti-gravity) can consume as a system prompt prefix to restore
full project context without re-reading the entire codebase.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HandoffPacket:
    """Generates a compressed context packet for cross-agent context transfer.

    Usage::
        hp = HandoffPacket(memory, brain)
        packet = await hp.generate(target="forge")
        prefix = hp.to_prompt_prefix(packet)
        path   = hp.save(packet)
    """

    FORMAT_VERSION = "1.0"

    def __init__(self, memory, brain) -> None:
        self.memory = memory
        self.brain = brain

    # ── Generate ───────────────────────────────────────────────────────────────

    async def generate(self, target: str = "any") -> dict[str, Any]:
        """Build a handoff packet. Calls brain to summarise + recall top memories.

        target: 'any' | 'forge' | 'cursor' | 'claude-code' | 'anti-gravity'
        """
        now = datetime.now(timezone.utc).isoformat()
        project_root = str(Path.cwd())

        # ── Project metadata ───────────────────────────────────────────────────
        project_info = _detect_project_info(project_root)

        # ── Session history ────────────────────────────────────────────────────
        sessions_summary = ""
        completed_tasks: list[str] = []
        pending_tasks: list[str] = []
        blocked_tasks: list[dict] = []
        key_decisions: list[str] = []
        files_changed: list[str] = []

        try:
            db = self.memory.db
            cur = db.cursor()

            # Last 5 sessions
            rows = cur.execute(
                "SELECT goal, tasks_completed, tasks_blocked, files_changed, summary_gz, started_at "
                "FROM sessions ORDER BY started_at DESC LIMIT 5"
            ).fetchall()

            total_sessions = cur.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            all_files: set[str] = set()

            session_summaries: list[str] = []
            for goal, tc, tb, fc_json, summary_gz, started_at in rows:
                if summary_gz:
                    try:
                        summary_text = self.memory._decompress(summary_gz)
                        session_summaries.append(summary_text[:200])
                    except Exception:
                        pass
                if fc_json:
                    try:
                        all_files.update(json.loads(fc_json))
                    except Exception:
                        pass

            files_changed = sorted(all_files)[:30]
            sessions_summary = (
                f"{total_sessions} session(s). " +
                " ".join(session_summaries[:2])
            ).strip()

        except Exception as e:
            logger.warning(f"HandoffPacket: failed to read session history: {e}")

        # ── Top memories ───────────────────────────────────────────────────────
        top_memories: list[dict] = []
        try:
            memories = self.memory.recall(
                query="important decisions requirements errors architecture",
                top_k=10,
            )
            top_memories = [
                {
                    "category": m["category"],
                    "text": m["text"][:500],
                    "priority": round(m["priority"], 3),
                    "score": round(m["score"], 3),
                }
                for m in memories
            ]
            # Extract decisions
            key_decisions = [
                m["text"][:200]
                for m in memories
                if m["category"] == "decision"
            ][:6]
        except Exception as e:
            logger.warning(f"HandoffPacket: failed to recall memories: {e}")

        # ── Brain: next recommended ────────────────────────────────────────────
        next_recommended = ""
        warnings: list[str] = []
        try:
            brain_result = await self.brain.summarise(
                changes=[{"summary": sessions_summary}],
                project_summary={
                    "files_changed": files_changed,
                    "key_decisions": key_decisions,
                },
            )
            next_steps = brain_result.get("next_suggested", [])
            if isinstance(next_steps, list) and next_steps:
                next_recommended = str(next_steps[0])[:200]
            elif isinstance(next_steps, str):
                next_recommended = next_steps[:200]
            warnings = [str(w) for w in brain_result.get("risk_flags", [])][:5]
        except Exception as e:
            logger.warning(f"HandoffPacket: brain summarise failed: {e}")

        return {
            "format_version": self.FORMAT_VERSION,
            "generated_at": now,
            "target_agent": target,
            "project": project_info,
            "sessions_summary": sessions_summary,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "blocked_tasks": blocked_tasks,
            "key_decisions": key_decisions,
            "files_changed": files_changed,
            "top_memories": top_memories,
            "next_recommended": next_recommended,
            "warnings": warnings,
        }

    # ── Prompt prefix ──────────────────────────────────────────────────────────

    def to_prompt_prefix(self, packet: dict) -> str:
        """Convert packet to a string for injection into any LLM system prompt."""
        proj = packet.get("project", {})
        lang = proj.get("language", "unknown")
        framework = proj.get("framework", "")
        root = proj.get("root", "")
        files_count = proj.get("files_count", 0)

        lang_str = f"{lang}/{framework}" if framework else lang
        lines = [
            "=== FORGE PROJECT CONTEXT ===",
            f"Project: {root} ({lang_str}, {files_count} files)",
            f"Generated: {packet.get('generated_at', '')[:19]} UTC",
            "",
            "WHAT WAS BUILT:",
            packet.get("sessions_summary", "No session history available."),
            "",
        ]

        decisions = packet.get("key_decisions", [])
        if decisions:
            lines.append("KEY DECISIONS:")
            for d in decisions:
                lines.append(f"- {d}")
            lines.append("")

        completed = packet.get("completed_tasks", [])
        pending = packet.get("pending_tasks", [])
        blocked = packet.get("blocked_tasks", [])
        lines.append(
            f"TASKS: {len(completed)} completed  "
            f"{len(pending)} pending  "
            f"{len(blocked)} blocked"
        )

        next_rec = packet.get("next_recommended", "")
        if next_rec:
            lines.append(f"\nNEXT RECOMMENDED: {next_rec}")

        warnings = packet.get("warnings", [])
        if warnings:
            lines.append("\nWARNINGS:")
            for w in warnings:
                lines.append(f"- {w}")

        memories = packet.get("top_memories", [])
        if memories:
            lines.append("\nRELEVANT CONTEXT:")
            for m in memories[:5]:
                cat = m.get("category", "note")
                text = m.get("text", "")[:300]
                lines.append(f"[{cat}] {text}")

        lines.append("=== END FORGE CONTEXT ===")
        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, packet: dict) -> Path:
        """Compress packet with gzip, save to .forge/memory/handoff.gz.
        Also writes to handoffs table and marks previous handoffs as used.
        """
        gz_path = self.memory.root / "handoff.gz"
        packet_json = json.dumps(packet, ensure_ascii=False)
        import base64
        packet_gz = base64.b64encode(
            gzip.compress(packet_json.encode("utf-8"), compresslevel=6)
        ).decode("ascii")

        gz_path.write_bytes(gzip.compress(packet_json.encode("utf-8"), compresslevel=6))

        try:
            db = self.memory.db
            db.execute("UPDATE handoffs SET used=1")
            db.execute(
                "INSERT INTO handoffs (created_at, target_agent, packet_gz, used) VALUES (?,?,?,0)",
                (time.time(), packet.get("target_agent", "any"), packet_gz),
            )
            db.commit()
        except Exception as e:
            logger.warning(f"HandoffPacket.save: SQLite write failed: {e}")

        logger.info(f"Handoff packet saved → {gz_path}")
        return gz_path

    def load(self) -> dict | None:
        """Load and decompress the latest handoff.gz. Returns None if not found."""
        gz_path = self.memory.root / "handoff.gz"
        if not gz_path.exists():
            return None
        try:
            raw = gzip.decompress(gz_path.read_bytes()).decode("utf-8")
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"HandoffPacket.load: failed: {e}")
            return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_project_info(project_root: str) -> dict[str, Any]:
    """Detect language and framework from project files."""
    root = Path(project_root)
    files = list(root.rglob("*"))
    extensions: dict[str, int] = {}
    for f in files:
        if f.is_file() and f.suffix:
            extensions[f.suffix] = extensions.get(f.suffix, 0) + 1

    # Detect primary language
    lang_map = {".py": "python", ".js": "javascript", ".ts": "typescript",
                ".go": "go", ".rs": "rust", ".java": "java"}
    language = "unknown"
    best = 0
    for ext, cnt in extensions.items():
        if ext in lang_map and cnt > best:
            language = lang_map[ext]
            best = cnt

    # Detect framework
    framework = ""
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(errors="ignore")
        if "fastapi" in text.lower():
            framework = "fastapi"
        elif "flask" in text.lower():
            framework = "flask"
        elif "django" in text.lower():
            framework = "django"
        elif "click" in text.lower():
            framework = "cli"

    package_json = root / "package.json"
    if package_json.exists() and not framework:
        text = package_json.read_text(errors="ignore")
        if "next" in text.lower():
            framework = "nextjs"
        elif "react" in text.lower():
            framework = "react"
        elif "vue" in text.lower():
            framework = "vue"

    py_files = [f for f in files if f.suffix == ".py" and not any(
        ex in f.parts for ex in {".venv", "__pycache__", "dist", "build", ".forge"}
    )]

    # Last modified
    try:
        mtimes = [f.stat().st_mtime for f in py_files if f.is_file()]
        last_mod = datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat() if mtimes else ""
    except Exception:
        last_mod = ""

    return {
        "root": str(root),
        "language": language,
        "framework": framework,
        "files_count": len([f for f in files if f.is_file()]),
        "last_modified": last_mod,
    }
