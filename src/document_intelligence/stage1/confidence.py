"""Confidence computation utilities."""

from __future__ import annotations

import numpy as np


def classifier_confidence(probability: float) -> float:
    """Map classifier probability to confidence in [0, 1]."""
    return float(np.clip(probability, 0.0, 1.0))


def similarity_margin_confidence(score: float, threshold: float) -> float:
    """Confidence from distance to decision threshold."""
    margin = abs(score - threshold)
    return float(np.clip(margin / max(threshold, 1 - threshold, 0.01), 0.0, 1.0))


def combined_confidence(probabilities: list[float], weights: list[float] | None = None) -> float:
    if not probabilities:
        return 0.0
    if weights is None:
        weights = [1.0] * len(probabilities)
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return float(sum(p * w for p, w in zip(probabilities, weights)) / total_w)
