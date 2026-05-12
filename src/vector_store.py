"""VectorStore class for forge coding agent with FAISS indexing and persistence."""

import json
import logging
import os
import re
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
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)

        path = Path(file_path)
        new_meta = [
            {
                "file": file_path,          # primary key used by tests
                "file_path": file_path,     # legacy key kept for compat
                "chunk_id": f"{path.name}_{i}",
                "text": c,
                "indexed_at": 0,
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

    def index_project(self, root_dir: str, extensions: list[str]) -> dict:
        """Index all files in directory matching extensions."""
        path = Path(root_dir).expanduser()
        stats = {"files_indexed": 0, "chunks_total": 0, "skipped": 0}

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

                chunks = self.index_file(str(p))
                stats["files_indexed"] += 1
                stats["chunks_total"] += chunks

        logger.info(f"Indexed project: {stats}")
        return stats

    def search(self, query: str, top_k: int = 8, file_filter: str | None = None) -> list[dict]:
        """Search for similar chunks. Returns top-k results with metadata."""
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        try:
            query_vec = self.model.encode(query, convert_to_numpy=True).astype("float32")
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
        """Remove chunks for a specific file from index and metadata."""
        ids_to_remove = [i for i, m in enumerate(self.metadata) if m["file_path"] == file_path]
        if not ids_to_remove:
            return

        if len(ids_to_remove) == len(self.metadata):
            self.index = None
            self.metadata = []
            return

        self.metadata = [m for i, m in enumerate(self.metadata) if i not in ids_to_remove]

        if self.metadata:
            embeddings = np.array(
                [self.model.encode(m["text"], convert_to_numpy=True) for m in self.metadata],
                dtype="float32",
            )
            embeddings = self._normalize(embeddings)
            dim = self._get_dim()
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
        else:
            self.index = None

    def remove_file(self, file_path: str) -> None:
        """Remove all chunks for a file (does not require file to exist on disk)."""
        self._remove_file_chunks(file_path)
        logger.info(f"Removed all chunks for {file_path}")

    def save(self, index_path: str | None = None) -> None:
        """Persist index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        path = index_path or FAISS_INDEX_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        try:
            faiss.write_index(self.index, path)
            with open(f"{path}.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
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

            meta_path = f"{path}.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                logger.warning("Metadata file not found, index loaded without metadata")
                self.metadata = []

            logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
