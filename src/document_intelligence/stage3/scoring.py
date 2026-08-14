"""Normalized retrieval scoring."""

from __future__ import annotations

import numpy as np


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float64)
    min_s, max_s = arr.min(), arr.max()
    if max_s == min_s:
        return [1.0] * len(scores)
    return ((arr - min_s) / (max_s - min_s)).tolist()


def combine_scores(
    vector_score: float,
    rerank_score: float | None = None,
    vector_weight: float = 0.4,
) -> float:
    if rerank_score is None:
        return vector_score
    rerank_weight = 1.0 - vector_weight
    return vector_weight * vector_score + rerank_weight * rerank_score
