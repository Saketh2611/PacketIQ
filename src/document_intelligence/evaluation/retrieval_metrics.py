"""Retrieval evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

import numpy as np


@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    precision_at_1: float
    precision_at_5: float
    mrr: float
    ndcg: float
    avg_latency_seconds: float
    indexing_time_seconds: float


def _dcg(relevances: list[float]) -> float:
    return sum(rel / log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    relevances = relevances[:k]
    dcg = _dcg(relevances)
    ideal = _dcg(sorted(relevances, reverse=True))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(relevant & top) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r in relevant)
    return hits / k


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(
    queries: list[dict],
    indexing_time: float = 0.0,
) -> RetrievalMetrics:
    """
    Each query dict: {retrieved_ids: [...], relevant_ids: set/list, latency: float}
    """
    r1, r3, r5, p1, p5, mrrs, ndcgs, lats = [], [], [], [], [], [], [], []
    for q in queries:
        rel = set(q.get("relevant_ids", []))
        ret = q.get("retrieved_ids", [])
        lats.append(q.get("latency", 0.0))
        r1.append(recall_at_k(rel, ret, 1))
        r3.append(recall_at_k(rel, ret, 3))
        r5.append(recall_at_k(rel, ret, 5))
        p1.append(precision_at_k(rel, ret, 1))
        p5.append(precision_at_k(rel, ret, 5))
        mrrs.append(mrr(rel, ret))
        rel_vec = [1.0 if rid in rel else 0.0 for rid in ret]
        ndcgs.append(ndcg_at_k(rel_vec, 5))

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return RetrievalMetrics(
        recall_at_1=mean(r1),
        recall_at_3=mean(r3),
        recall_at_5=mean(r5),
        precision_at_1=mean(p1),
        precision_at_5=mean(p5),
        mrr=mean(mrrs),
        ndcg=mean(ndcgs),
        avg_latency_seconds=mean(lats),
        indexing_time_seconds=indexing_time,
    )
