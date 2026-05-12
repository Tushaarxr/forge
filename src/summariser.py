"""Summariser — checkpoint management and human-review formatting for forge."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.brain import Brain
    from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Summariser:
    """Creates checkpoints every N changes for AI and human review."""

    def __init__(self, brain: "Brain", vector_store: "VectorStore") -> None:
        """Initialize Summariser.

        Args:
            brain: Brain instance for AI summarisation
            vector_store: VectorStore instance (used for indexing summaries)
        """
        self.brain = brain
        self.vector_store = vector_store
        self.base_dir = Path(".forge")
        self.summaries_dir = self.base_dir / "summaries"
        self.changelog_path = self.base_dir / "CHANGELOG.md"

        env_value = os.getenv("SUMMARISE_EVERY_N_CHANGES", "10")
        try:
            self.checkpoint_interval = int(env_value)
        except ValueError:
            logger.warning(f"Invalid SUMMARISE_EVERY_N_CHANGES='{env_value}', defaulting to 10")
            self.checkpoint_interval = 10

    def _ensure_dirs(self) -> None:
        """Ensure .forge directory structure exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

    async def checkpoint(
        self, changes_log: list[dict], project_summary: dict
    ) -> dict[str, Any]:
        """Create a new checkpoint with AI summarisation.

        Writes:
          - .forge/summaries/<timestamp>.json  — machine-readable summary
          - .forge/CHANGELOG.md               — human-readable append entry

        Returns:
            dict with summary_path, changelog_path, summary_dict
        """
        self._ensure_dirs()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.summaries_dir / f"{timestamp}.json"

        logger.info(f"Creating checkpoint at {summary_path}")

        try:
            summary_dict = await self.brain.summarise(changes_log, project_summary)
        except Exception as e:
            logger.error(f"Brain summarisation failed: {e}")
            summary_dict = {
                "summary": f"Checkpoint at {timestamp} (summarisation failed: {e})",
                "key_decisions": [],
                "next_suggested": [],
                "risk_flags": [],
            }

        # ── Write JSON summary ─────────────────────────────────────────────────
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to write summary to {summary_path}: {e}")
            raise

        # ── Append to CHANGELOG.md ────────────────────────────────────────────
        changelog_entry = self._format_changelog_entry(
            timestamp, summary_dict.get("summary", ""), summary_dict
        )
        try:
            with open(self.changelog_path, "a", encoding="utf-8") as f:
                f.write(changelog_entry)
        except IOError as e:
            logger.error(f"Failed to append to changelog: {e}")
            raise

        # ── Index summary text in vector store for future lookups ─────────────
        if self.vector_store:
            summary_text = summary_dict.get("summary", "")
            if summary_text:
                try:
                    # Write a temporary text file and index it
                    tmp_path = self.summaries_dir / f"{timestamp}_summary.txt"
                    tmp_path.write_text(summary_text, encoding="utf-8")
                    self.vector_store.index_file(str(tmp_path))
                except Exception as e:
                    logger.warning(f"Failed to index summary in vector store: {e}")

        logger.info(f"Checkpoint created: {summary_path}")
        return {
            "summary_path": str(summary_path),
            "changelog_path": str(self.changelog_path),
            "summary_dict": summary_dict,
        }

    def _format_changelog_entry(
        self, timestamp: str, summary_text: str, summary_dict: dict[str, Any]
    ) -> str:
        """Format markdown changelog entry matching the spec:

        ## <timestamp> — <first 60 chars of summary>
        <full summary paragraph>
        ### Key decisions
        - decision 1
        ### Next steps
        1. step 1
        """
        short = (summary_text[:60] + "…") if len(summary_text) > 60 else summary_text
        lines = [f"\n## {timestamp} — {short}\n\n{summary_text}\n"]

        decisions = summary_dict.get("key_decisions", [])
        if decisions:
            lines.append("### Key decisions")
            for d in decisions:
                lines.append(f"- {d}")
            lines.append("")

        next_steps = summary_dict.get("next_suggested", [])
        if next_steps:
            lines.append("### Next steps")
            for i, s in enumerate(next_steps, 1):
                lines.append(f"{i}. {s}")
            lines.append("")

        risk_flags = summary_dict.get("risk_flags", [])
        if risk_flags:
            lines.append("### Risk flags")
            for r in risk_flags:
                lines.append(f"- ⚠️ {r}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_human_review(summary_dict: dict[str, Any]) -> str:
        """Render summary for terminal display using Rich markup.

        Uses Rich Panel, Table layout concepts encoded as markup strings.
        risk_flags rendered in bold red, next_suggested as numbered list,
        key_decisions as bullet list.
        """
        parts: list[str] = []

        summary_text = summary_dict.get("summary", "")
        if summary_text:
            parts.append(f"[bold cyan]📝 Summary[/bold cyan]\n{summary_text}")

        key_decisions = summary_dict.get("key_decisions", [])
        if key_decisions:
            parts.append("[bold yellow]🔑 Key Decisions[/bold yellow]")
            for d in key_decisions[:5]:
                parts.append(f"  • {d}")

        next_suggested = summary_dict.get("next_suggested", [])
        if next_suggested:
            parts.append("[bold green]📌 Next Steps[/bold green]")
            for i, s in enumerate(next_suggested[:5], 1):
                parts.append(f"  {i}. {s}")

        risk_flags = summary_dict.get("risk_flags", [])
        if risk_flags:
            parts.append("[bold red]⚠️  Risk Flags[/bold red]")
            for r in risk_flags[:5]:
                parts.append(f"  [bold red]• {r}[/bold red]")

        return "\n".join(parts)

    async def load_summaries(self, n: int = 5) -> list[dict[str, Any]]:
        """Load the last N summaries from .forge/summaries/."""
        self._ensure_dirs()

        if not self.summaries_dir.exists():
            return []

        summary_files = sorted(
            [p for p in self.summaries_dir.glob("*.json") if not p.name.endswith("_summary.txt")],
            key=lambda p: p.name,
            reverse=True,
        )[:n]

        summaries = []
        for path in summary_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    summaries.append(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load summary from {path}: {e}")

        return summaries

    def should_checkpoint(self, changes_since_last: int) -> bool:
        """Return True if a checkpoint should be created."""
        return changes_since_last >= self.checkpoint_interval

    def summarize_changes(self, changes: list[dict[str, Any]]) -> str:
        """Produce a human-readable one-liner of a change list."""
        if not changes:
            return "No changes."
        parts = []
        for c in changes:
            action = c.get("action", "modified")
            file_name = Path(c.get("file", c.get("file_path", "unknown"))).name
            parts.append(f"{action.capitalize()}: {file_name}")
        return ", ".join(parts)

    def compress_context(self, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove redundant or outdated context items (keep most recent unique types)."""
        seen_types: set[str] = set()
        compressed: list[dict[str, Any]] = []
        for item in reversed(context):
            item_type = item.get("type", "generic")
            if item_type not in seen_types:
                seen_types.add(item_type)
                compressed.append(item)
        return list(reversed(compressed))

    def restore_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """Restore session state from a saved checkpoint JSON file."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_summary(self, session_history: list[dict[str, Any]]) -> str:
        """Generate a brief text summary of a session history list."""
        if not session_history:
            return "Empty session."
        n = len(session_history)
        last = session_history[-1]
        goal = last.get("goal", "unknown goal")
        passed = last.get("passed", False)
        return (
            f"Session with {n} run(s). Last goal: '{goal}'. "
            f"Status: {'✅ passed' if passed else '❌ failed'}."
        )
