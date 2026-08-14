"""Vector store interface with FAISS default."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from document_intelligence.config.settings import get_settings
from document_intelligence.stage2.schema import Chunk
from document_intelligence.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorRecord:
    chunk_id: str
    document_id: str
    document_type: str
    page_start: int
    page_end: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: np.ndarray | None = None


class VectorStore(ABC):
    @abstractmethod
    def add(self, records: list[VectorRecord], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_n: int) -> list[tuple[VectorRecord, float]]: ...

    @abstractmethod
    def save(self, path: Path | str) -> None: ...

    @abstractmethod
    def load(self, path: Path | str) -> None: ...

    @abstractmethod
    def size(self) -> int: ...


class FAISSVectorStore(VectorStore):
    """Local FAISS flat index with metadata sidecar."""

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.records: list[VectorRecord] = []

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.dimension = dim
            self.index = faiss.IndexFlatIP(dim)

    def add(self, records: list[VectorRecord], vectors: np.ndarray) -> None:
        if len(records) == 0:
            return
        vectors = vectors.astype(np.float32)
        self._ensure_index(vectors.shape[1])
        assert self.index is not None
        self.index.add(vectors)
        for rec, vec in zip(records, vectors):
            rec.vector = vec
            self.records.append(rec)
        logger.info("Added vectors to index", count=len(records), total=self.size())

    def search(self, query_vector: np.ndarray, top_n: int) -> list[tuple[VectorRecord, float]]:
        if self.index is None or self.size() == 0:
            return []
        q = query_vector.astype(np.float32).reshape(1, -1)
        k = min(top_n, self.size())
        scores, indices = self.index.search(q, k)
        results: list[tuple[VectorRecord, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.records[idx], float(score)))
        return results

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
        meta = [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "document_type": r.document_type,
                "page_start": r.page_start,
                "page_end": r.page_end,
                "text": r.text,
                "metadata": r.metadata,
            }
            for r in self.records
        ]
        with (path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump({"dimension": self.dimension, "records": meta}, f, indent=2)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        index_file = path / "index.faiss"
        meta_file = path / "metadata.json"
        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Index not found at {path}")
        self.index = faiss.read_index(str(index_file))
        with meta_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.dimension = data["dimension"]
        self.records = [VectorRecord(**r) for r in data["records"]]

    def size(self) -> int:
        return len(self.records)


def create_vector_store(backend: str | None = None) -> VectorStore:
    settings = get_settings()
    backend = backend or settings.vector_store
    if backend == "faiss":
        return FAISSVectorStore()
    raise ValueError(f"Unsupported vector store backend: {backend}")


def index_chunks(chunks: list[Chunk], embedder: Any, store: VectorStore | None = None) -> VectorStore:
    from document_intelligence.stage3.embeddings import EmbeddingModel

    embedder = embedder or EmbeddingModel()
    store = store or create_vector_store()
    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts)
    records = [
        VectorRecord(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_type=c.document_type,
            page_start=c.page_start,
            page_end=c.page_end,
            text=c.text,
            metadata=c.metadata,
        )
        for c in chunks
    ]
    store.add(records, vectors)
    return store
