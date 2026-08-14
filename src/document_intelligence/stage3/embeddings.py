"""Pluggable embedding interface."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from document_intelligence.config.settings import get_settings
from document_intelligence.utils.logging import get_logger

logger = get_logger(__name__)

EMBEDDING_DIM = 384


def _hash_embed(texts: list[str], dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic hash-based embeddings for offline/testing fallback."""
    vectors = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vectors.append(vec)
    return np.vstack(vectors)


class EmbeddingModel:
    """Hugging Face Sentence Transformers embedding model with caching."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
        device: str | None = None,
        use_hash_fallback: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.normalize = normalize if normalize is not None else settings.normalize_embeddings
        self.device = device or settings.device
        self.use_hash_fallback = (
            use_hash_fallback
            if use_hash_fallback is not None
            else os.getenv("USE_HASH_EMBEDDINGS", "").lower() in ("1", "true", "yes")
        )
        self._model: Any = None
        self._cache: dict[str, np.ndarray] = {}
        self._fallback = False

    def _load_model(self) -> None:
        if self._model is not None or self._fallback:
            return
        if self.use_hash_fallback:
            self._fallback = True
            logger.info("Using hash embedding fallback", mode="hash")
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            logger.warning("SentenceTransformer unavailable, using hash fallback", error=str(exc))
            self._fallback = True

    def encode(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        self._load_model()
        if self._fallback:
            return _hash_embed(texts)

        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        results: list[np.ndarray | None] = [None] * len(texts)

        for i, text in enumerate(texts):
            key = text[:500]
            if use_cache and key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            embs = self._model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            for idx, emb, text in zip(uncached_indices, embs, uncached_texts):
                arr = np.array(emb, dtype=np.float32)
                results[idx] = arr
                if use_cache:
                    self._cache[text[:500]] = arr

        return np.vstack([r for r in results if r is not None])

    @property
    def version(self) -> str:
        return "hash-fallback" if self._fallback else self.model_name
