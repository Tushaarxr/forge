"""VectorStore class for forge coding agent with FAISS indexing and persistence."""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", ".forge/vectors/faiss.index")


class VectorStore:
    """Vector store with FAISS for semantic search and persistence."""

    def __init__(self) -> None:
        self.model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict[str, Any]] = []
        # FIX: Use IndexIDMap to support efficient ID-based removal
        self._use_id_map: bool = False
        print("Loading embedding model (this may take a moment on first run)...")
        self._load_model()

    def _load_model(self) -> None:
        """Load sentence-transformers model lazily."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(EMBED_MODEL)
                logger.info(f"Loaded embedding model: {EMBED_MODEL}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    def _create_index(self, dim: int) -> None:
        """Create FAISS index with given dimension."""
        self.index = faiss.IndexFlatIP(dim)
        logger.info(f"Created FAISS index with {dim} dimensions")

    def _get_dim(self) -> int:
        """Get embedding dimension from model."""
        if self.model is None:
            return 384
        # New API (sentence-transformers >= 3.x)
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def index_file(self, file_path: str) -> int:
        """Index a single file by chunks. Returns number of chunks added."""
        path = Path(file_path).expanduser()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return 0
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return 0

        if not text.strip():
            logger.info(f"Empty file, skipping: {file_path}")
            return 0

        return self._add_text_chunks(file_path, text)

    def _add_text_chunks(self, file_path: str, text: str) -> int:
        """Embed and add text chunks for a file."""
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        embeddings = np.array(
            [self.model.encode(c, convert_to_numpy=True) for c in chunks],
            dtype="float32",
        )
        embeddings = self._normalize(embeddings)

        if self.index is None:
            dim = self._get_dim()
            # Use IndexIDMap to support efficient ID-based removal
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            self._use_id_map = True

        # Generate unique IDs for tracking
        start_id = len(self.metadata)
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        path = Path(file_path)
        new_meta = [
            {
                "file": file_path,          # primary key used by tests
                "file_path": file_path,     # legacy key kept for compat
                "chunk_id": f"{path.stem}_{i}_{start_id}",  # FIX: include index ID to avoid collisions
                "text": c,
                "indexed_at": 0,
                "faiss_id": start_id + i,   # Track FAISS ID for efficient removal
            }
            for i, c in enumerate(chunks)
        ]
        self.metadata.extend(new_meta)

        logger.info(f"Indexed {len(chunks)} chunks from {file_path}")
        return len(chunks)

    def index_files(self, file_paths: list[str]) -> int:
        """Index multiple files. Returns total chunks added."""
        total = 0
        for fp in file_paths:
            total += self.index_file(fp)
        return total

    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = 150) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i: i + chunk_size])
        return chunks

    @lru_cache(maxsize=128)
    def _encode_cached(self, text: str) -> np.ndarray:
        """Cached encoding for queries to avoid re-encoding similar queries.

        Uses LRU cache with 128 entries for query encoding.
        """
        if self.model is None:
            self._load_model()
        vec = self.model.encode(text, convert_to_numpy=True).astype("float32")
        return vec

    def _index_single_file(self, file_path: str) -> tuple[str, int, bool]:
        """Index a single file. Used for parallel indexing.

        Returns:
            Tuple of (file_path, chunks_count, success)
        """
        try:
            chunks = self.index_file(file_path)
            return file_path, chunks, True
        except Exception as e:
            logger.warning(f"Failed to index {file_path}: {e}")
            return file_path, 0, False

    def index_project(self, root_dir: str, extensions: list[str]) -> dict:
        """Index all files in directory matching extensions.

        Uses parallel indexing for files > 1KB for improved performance.
        """
        path = Path(root_dir).expanduser()
        stats = {"files_indexed": 0, "chunks_total": 0, "skipped": 0}

        # Collect all files to index
        files_to_index = []
        for ext in extensions:
            pattern = re.compile(f".*\\.{ext}$")
            for p in path.rglob("*"):
                if not p.is_file():
                    continue
                if not pattern.match(p.name):
                    stats["skipped"] += 1
                    continue
                if any(
                    excl in p.parts
                    for excl in {".git", "__pycache__", "node_modules", ".venv", "dist", "build", ".forge"}
                ):
                    stats["skipped"] += 1
                    continue

                # Skip empty or very small files
                try:
                    if p.stat().st_size < 100:
                        stats["skipped"] += 1
                        continue
                except OSError:
                    pass

                files_to_index.append(str(p))

        # Separate small files (sequential) from larger files (parallel)
        small_files = []
        large_files = []

        for fp in files_to_index:
            try:
                size = Path(fp).stat().st_size
                if size > 1024:  # > 1KB
                    large_files.append(fp)
                else:
                    small_files.append(fp)
            except OSError:
                small_files.append(fp)

        # Process small files sequentially
        for p in small_files:
            chunks = self.index_file(p)
            stats["files_indexed"] += 1
            stats["chunks_total"] += chunks

        # Process large files in parallel (4 at a time)
        if large_files:
            logger.info(f"Parallel indexing {len(large_files)} large files...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._index_single_file, fp): fp
                    for fp in large_files
                }
                for future in as_completed(futures):
                    fp, chunks, success = future.result()
                    if success:
                        stats["files_indexed"] += 1
                        stats["chunks_total"] += chunks

        logger.info(f"Indexed project: {stats}")
        return stats

    def search(self, query: str, top_k: int = 8, file_filter: str | None = None) -> list[dict]:
        """Search for similar chunks. Returns top-k results with metadata.

        Uses cached encoding for improved performance on repeated queries.
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        try:
            # Use cached encoding for better performance
            query_vec = self._encode_cached(query).astype("float32")
            query_vec = self._normalize(query_vec.reshape(1, -1))[0]
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            return []

        if self.index is None or self.index.ntotal == 0:
            logger.warning("No index available")
            return []

        try:
            k = min(top_k, self.index.ntotal)
            scores, ids = self.index.search(query_vec.reshape(1, -1), k)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        results = []
        for i in range(min(top_k, len(ids[0]))):
            idx = int(ids[0][i])
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if file_filter and meta["file_path"] != file_filter:
                continue

            results.append(
                {
                    "file": meta["file"],
                    "file_path": meta["file_path"],
                    "chunk_id": meta["chunk_id"],
                    "score": float(scores[0][i]),
                    "text": meta["text"],
                    "metadata": meta,  # expose full metadata for test compatibility
                }
            )

        logger.info(f"Found {len(results)} results for query")
        return results

    def update_file(self, file_path: str) -> int:
        """Remove existing chunks and re-index a file."""
        self._remove_file_chunks(file_path)
        path = Path(file_path).expanduser()
        if not path.exists():
            logger.warning(f"File not found for update: {file_path}")
            return 0
        return self.index_file(file_path)

    def _remove_file_chunks(self, file_path: str) -> None:
        """Remove chunks for a specific file from index and metadata.

        FIX: Uses efficient ID-based removal instead of rebuilding entire index.
        """
        chunks_to_remove = [m for m in self.metadata if m["file_path"] == file_path]
        if not chunks_to_remove:
            return

        # Get FAISS IDs to remove
        ids_to_remove = np.array([m.get("faiss_id", i) for i, m in enumerate(chunks_to_remove)
                                  if "faiss_id" in m], dtype=np.int64)

        if len(ids_to_remove) == len(self.metadata):
            self.index = None
            self.metadata = []
            self._use_id_map = False
            return

        # Remove from metadata
        self.metadata = [m for m in self.metadata if m["file_path"] != file_path]

        # If using IndexIDMap, remove by IDs (efficient)
        if self._use_id_map and self.index is not None:
            try:
                # Create selector for IDs to remove
                selector = faiss.IDSelectorBatch(ids_to_remove)
                self.index.remove_ids(selector)
            except Exception as e:
                logger.warning(f"ID-based removal failed, falling back to rebuild: {e}")
                self._rebuild_index_slow()
        elif self.metadata:
            self._rebuild_index_slow()
        else:
            self.index = None
            self._use_id_map = False

    def remove_file(self, file_path: str) -> None:
        """Remove all chunks for a file (does not require file to exist on disk)."""
        self._remove_file_chunks(file_path)
        logger.info(f"Removed all chunks for {file_path}")

    def _rebuild_index_slow(self) -> None:
        """Fallback: rebuild entire index by re-encoding all chunks.

        Used when ID-based removal fails.
        """
        if not self.metadata:
            self.index = None
            self._use_id_map = False
            return

        dim = self._get_dim()
        embeddings = np.array(
            [self.model.encode(m["text"], convert_to_numpy=True) for m in self.metadata],
            dtype="float32",
        )
        embeddings = self._normalize(embeddings)

        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._use_id_map = True
        ids = np.arange(len(self.metadata), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        # Update faiss_id in metadata
        for i, m in enumerate(self.metadata):
            m["faiss_id"] = i

    def save(self, index_path: str | None = None) -> None:
        """Persist index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        path = index_path or FAISS_INDEX_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        try:
            faiss.write_index(self.index, path)
            # Save metadata with extra info for reconstruction
            meta_with_info = {
                "metadata": self.metadata,
                "use_id_map": self._use_id_map,
            }
            with open(f"{path}.json", "w", encoding="utf-8") as f:
                json.dump(meta_with_info, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved index and metadata to {path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def load(self, index_path: str | None = None) -> None:
        """Restore index and metadata from disk."""
        path = index_path or FAISS_INDEX_PATH
        if not os.path.exists(path):
            logger.warning(f"Index file not found at {path}, starting fresh")
            return

        try:
            self.index = faiss.read_index(path)
            dim = self._get_dim()
            if self.index.d != dim:
                logger.error(f"Dimension mismatch: expected {dim}, got {self.index.d}")
                self.index = None
                return

            # Check if it's an IndexIDMap
            self._use_id_map = hasattr(self.index, 'id_map')

            meta_path = f"{path}.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    # Handle both old format (just list) and new format (dict)
                    if isinstance(loaded, dict):
                        self.metadata = loaded.get("metadata", [])
                        self._use_id_map = loaded.get("use_id_map", self._use_id_map)
                    else:
                        self.metadata = loaded
            else:
                logger.warning("Metadata file not found, index loaded without metadata")
                self.metadata = []

            logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
