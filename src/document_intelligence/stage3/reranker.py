"""Optional cross-encoder reranking."""

from __future__ import annotations

import os
from typing import Any

from document_intelligence.config.settings import get_settings
from document_intelligence.stage3.scoring import normalize_scores
from document_intelligence.stage3.vector_store import VectorRecord
from document_intelligence.utils.logging import get_logger

logger = get_logger(__name__)
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


class Reranker:
    """Hugging Face cross-encoder reranker for top-N candidates."""

    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool | None = None,
        batch_size: int = 16,
        local_files_only: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.enabled = enabled if enabled is not None else settings.use_reranker
        self.batch_size = batch_size
        self.local_files_only = (
            local_files_only
            if local_files_only is not None
            else _env_flag("RERANKER_LOCAL_FILES_ONLY", True)
        )
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None or not self.enabled:
            return
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker", model=self.model_name, local_files_only=self.local_files_only)
        self._model = CrossEncoder(self.model_name, local_files_only=self.local_files_only)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[VectorRecord, float]],
    ) -> list[tuple[VectorRecord, float, float]]:
        if not self.enabled or not candidates:
            return [(rec, vec_score, vec_score) for rec, vec_score in candidates]

        self._load()
        if self._model is None:
            return [(rec, vec_score, vec_score) for rec, vec_score in candidates]

        pairs = [(query, rec.text) for rec, _ in candidates]
        raw_scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        norm = normalize_scores([float(s) for s in raw_scores])

        reranked: list[tuple[VectorRecord, float, float]] = []
        for (rec, vec_score), rr_score in zip(candidates, norm):
            reranked.append((rec, vec_score, rr_score))
        reranked.sort(key=lambda x: x[2], reverse=True)
        return reranked
