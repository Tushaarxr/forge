"""Context Engine for hybrid retrieval combining vector search and graph traversal."""

import logging
import time
from typing import Any

from src.vector_store import VectorStore
from src.project_graph import ProjectGraph


logger = logging.getLogger(__name__)


class ContextEngine:
    """Hybrid context retrieval engine using FAISS vectors and NetworkX graphs."""

    def __init__(self, vector_store: VectorStore, graph: ProjectGraph) -> None:
        """Initialize the ContextEngine with vector store and project graph.

        Args:
            vector_store: VectorStore instance for semantic similarity search
            graph: ProjectGraph instance for dependency-aware context retrieval
        """
        self.vector_store = vector_store
        self.graph = graph
        logger.info("ContextEngine initialized with hybrid retrieval")

    def get_context(
        self, task: str, active_file: str | None = None, budget_chars: int = 14000
    ) -> str:
        """Retrieve context for a task using hybrid vector + graph search.

        Ranking formula:
            final_score = 0.5 * vector_score + 0.3 * graph_proximity_score + 0.2 * recency_score

        Args:
            task: The user task or query
            active_file: Optional active file to prioritize its neighbours
            budget_chars: Maximum character budget for context (default 14000)

        Returns:
            Formatted context string under budget
        """
        all_candidates: list[dict[str, Any]] = []

        # ── Step 1: Vector search ──────────────────────────────────────────────
        if self.vector_store and len(self.vector_store.metadata) > 0:
            try:
                vec_results = self.vector_store.search(task, top_k=8)
                for chunk in vec_results:
                    all_candidates.append({"chunk": chunk, "raw_vec_score": chunk["score"], "source": "vector"})
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # ── Step 2: Graph neighbours of active_file ────────────────────────────
        if active_file and self.graph:
            try:
                neighbour_files = self.graph.get_context_files(active_file, depth=1)
                for file_path in neighbour_files[:3]:
                    try:
                        file_chunks = self.vector_store.search(task, top_k=4, file_filter=file_path)
                        for chunk in file_chunks:
                            all_candidates.append({"chunk": chunk, "raw_vec_score": chunk["score"], "source": "graph"})
                    except Exception as e:
                        logger.warning(f"Failed to search {file_path}: {e}")
            except Exception as e:
                logger.warning(f"Graph neighbours retrieval failed: {e}")

        # ── Step 3: Recent files ───────────────────────────────────────────────
        if self.graph:
            try:
                file_nodes = [
                    (n, self.graph.graph.nodes[n])
                    for n in self.graph.graph.nodes()
                    if self.graph._is_file_node(n)
                ]
                recent_files = sorted(file_nodes, key=lambda x: x[1].get("last_modified", 0), reverse=True)[:5]
                for file_path, _ in recent_files:
                    try:
                        file_chunks = self.vector_store.search(task, top_k=4, file_filter=file_path)
                        for chunk in file_chunks:
                            all_candidates.append({"chunk": chunk, "raw_vec_score": chunk["score"], "source": "recency"})
                    except Exception as e:
                        logger.warning(f"Failed to search recent file {file_path}: {e}")
            except Exception as e:
                logger.warning(f"Recent edits retrieval failed: {e}")

        # ── De-duplicate by chunk_id ───────────────────────────────────────────
        seen_ids: set[str] = set()
        unique_candidates: list[dict[str, Any]] = []
        for cand in all_candidates:
            cid = cand["chunk"].get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_candidates.append(cand)

        # ── Rank with composite score ──────────────────────────────────────────
        ranked: list[dict[str, Any]] = []
        for cand in unique_candidates:
            chunk = cand["chunk"]
            file_path = chunk.get("file_path", "")
            vec_score = min(cand["raw_vec_score"], 1.0)
            graph_prox = self._get_graph_proximity(file_path, active_file)
            recency = self._get_recency_score(chunk)
            composite = 0.5 * vec_score + 0.3 * graph_prox + 0.2 * recency
            ranked.append({**cand, "score": composite})

        ranked.sort(key=lambda x: x["score"], reverse=True)

        # ── Fill budget greedily ───────────────────────────────────────────────
        context_chars = 0
        selected: list[dict[str, Any]] = []
        for entry in ranked:
            chunk_text = entry["chunk"].get("text", "")
            file_info = entry["chunk"].get("file_path", "unknown")
            header = f"[FILE: {file_info} | relevance: {entry['score']:.2f} | via: {entry['source']}]\n"
            total_len = len(header) + len(chunk_text)
            if context_chars + total_len <= budget_chars:
                context_chars += total_len
                selected.append(entry)
            else:
                break

        formatted = self.format_context(selected)
        logger.info(f"Retrieved {len(selected)} chunks, used {context_chars}/{budget_chars} chars")
        return formatted

    def format_context(self, entries: list[dict]) -> str:
        """Format context entries into a structured string."""
        if not entries:
            return "=== CONTEXT [0 entries] ===\n\n=== END CONTEXT ==="

        texts = [e["chunk"].get("text", "") for e in entries]
        est_tokens = self.estimate_tokens("".join(texts))
        header = f"=== CONTEXT [{len(entries)} entries, ~{est_tokens} tokens] ===\n\n"

        parts = []
        for entry in entries:
            chunk = entry["chunk"]
            file_path = chunk.get("file_path", "unknown")
            content = chunk.get("text", "")
            relevance = f"{entry.get('score', 1.0):.2f}"
            source = entry.get("source", "vector")
            parts.append(f"[FILE: {file_path} | relevance: {relevance} | via: {source}]\n{content}")

        return header + "\n\n".join(parts) + "\n=== END CONTEXT ==="

    def get_affected_warning(self, changed_files: list[str]) -> str:
        """Generate warning about files that will be affected by changes."""
        if not self.graph or not changed_files:
            return ""

        all_affected: set[str] = set()
        for file_path in changed_files:
            try:
                affected = self.graph.get_affected(file_path)
                all_affected.update(affected)
            except Exception as e:
                logger.warning(f"Failed to get affected files for {file_path}: {e}")

        other_affected = [f for f in all_affected if f not in changed_files]
        if other_affected:
            return (
                f"⚠ Changing {len(changed_files)} files may affect: "
                + ", ".join(sorted(other_affected))[:200]
                + "..."
            )
        return ""

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count (chars // 4)."""
        return len(text) // 4

    def _get_graph_proximity(self, file_path: str, active_file: str | None = None) -> float:
        """Calculate graph proximity score.

        Returns:
            1.0  — same file or direct neighbour (1 hop)
            0.5  — two hops away
            0.0  — otherwise
        """
        if not self.graph or not active_file or not file_path:
            return 0.0

        if file_path == active_file:
            return 1.0

        try:
            # Direct edge in either direction (1 hop)
            if self.graph.graph.has_edge(active_file, file_path) or self.graph.graph.has_edge(file_path, active_file):
                return 1.0

            # 2 hops
            neighbours = (
                set(self.graph.graph.successors(active_file))
                | set(self.graph.graph.predecessors(active_file))
            )
            for neighbour in neighbours:
                if self.graph.graph.has_edge(neighbour, file_path) or self.graph.graph.has_edge(file_path, neighbour):
                    return 0.5
        except Exception:
            pass

        return 0.0

    def _get_recency_score(self, chunk: dict) -> float:
        """Calculate recency score.

        1.0 for files modified in last 5 minutes, decays linearly to 0 at 1 hour.
        """
        if not self.graph or "file_path" not in chunk:
            return 0.0

        try:
            file_path = chunk["file_path"]
            if file_path in self.graph.graph.nodes():
                node_data = self.graph.graph.nodes[file_path]
                mtime = node_data.get("last_modified", 0)
                if mtime > 0:
                    age_seconds = time.time() - mtime
                    five_minutes = 5 * 60
                    one_hour = 3600
                    if age_seconds <= five_minutes:
                        return 1.0
                    elif age_seconds >= one_hour:
                        return 0.0
                    else:
                        # Linear decay from 1.0 at 5 min to 0.0 at 1 hour
                        return 1.0 - (age_seconds - five_minutes) / (one_hour - five_minutes)
        except Exception:
            pass
        return 0.0
