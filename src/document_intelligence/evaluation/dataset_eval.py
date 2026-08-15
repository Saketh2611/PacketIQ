"""Evaluate Stage 1 on page-stream datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from document_intelligence.dataset.adapter import DocSplitAdapter, PacketAnnotation
from document_intelligence.evaluation.stage1_metrics import evaluate_stage1
from document_intelligence.ingestion.page_extractor import PageRepresentation
from document_intelligence.stage1.boundary_baseline import BoundaryBaseline, BoundaryDecision
from document_intelligence.stage1.boundary_classifier import BoundaryClassifier
from document_intelligence.stage1.grouping import decisions_to_groups
from document_intelligence.stage1.page_features import PageFeatureBuilder


@dataclass
class StreamEvalSample:
    stream_id: str
    pages: list[PageRepresentation]
    annotation: PacketAnnotation
    true_boundary_pairs: list[tuple[int, int, bool]]
    true_groups: list[list[int]]


def group_rows_by_stream(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stream_id"])].append(row)
    return grouped


def build_stream_sample(adapter: DocSplitAdapter, stream_id: str, rows: list[dict[str, Any]]) -> StreamEvalSample:
    annotation = adapter.stream_rows_to_annotation(rows)
    pages = adapter.stream_rows_to_pages(rows)
    page_to_group: dict[int, int] = {}
    for gi, group in enumerate(annotation.groups):
        for p in group:
            page_to_group[p] = gi

    true_pairs: list[tuple[int, int, bool]] = []
    for i in range(len(pages) - 1):
        a, b = pages[i].page_number, pages[i + 1].page_number
        is_boundary = page_to_group.get(a) != page_to_group.get(b)
        true_pairs.append((a, b, is_boundary))

    return StreamEvalSample(
        stream_id=stream_id,
        pages=pages,
        annotation=annotation,
        true_boundary_pairs=true_pairs,
        true_groups=annotation.groups,
    )


def iter_stream_samples(adapter: DocSplitAdapter, dataset_rows: list[dict[str, Any]]) -> Iterator[StreamEvalSample]:
    for stream_id, rows in group_rows_by_stream(dataset_rows).items():
        if len(rows) < 2:
            continue
        yield build_stream_sample(adapter, stream_id, rows)


def _pred_pairs_from_decisions(decisions: list[BoundaryDecision]) -> list[tuple[int, int, bool]]:
    return [(d.page_a, d.page_b, d.is_boundary) for d in decisions]


def _pred_groups(page_count: int, decisions: list[BoundaryDecision], packet_id: str) -> list[list[int]]:
    return [g.page_numbers for g in decisions_to_groups(page_count, decisions, packet_id)]


def evaluate_boundary_method(
    samples: list[StreamEvalSample],
    method: str,
    classifier: BoundaryClassifier | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    baseline = BoundaryBaseline(threshold=threshold) if method in {"baseline_rule", "baseline_embedding"} else None
    feature_builder = PageFeatureBuilder() if method == "learned" else None

    boundary_scores: list[tuple[float, float, float]] = []
    grouping_scores: list[float] = []
    page_pairs = 0

    for sample in samples:
        if method == "baseline_rule":
            assert baseline is not None
            baseline.mode = "weighted"
            decisions = baseline.predict_pairs(sample.pages)
        elif method == "baseline_embedding":
            assert baseline is not None
            decisions = baseline.predict_embedding_only(sample.pages)
        elif method == "learned":
            assert classifier is not None and feature_builder is not None
            pairs = feature_builder.build_all_pairs(sample.pages)
            decisions = classifier.predict_decisions(pairs)
        else:
            raise ValueError(f"Unknown method: {method}")

        pred_pairs = _pred_pairs_from_decisions(decisions)
        pred_groups = _pred_groups(len(sample.pages), decisions, sample.stream_id)
        metrics = evaluate_stage1(
            sample.true_boundary_pairs,
            pred_pairs,
            sample.true_groups,
            pred_groups,
            [],
            [],
        )
        boundary_scores.append(
            (metrics.boundary_precision, metrics.boundary_recall, metrics.boundary_f1)
        )
        grouping_scores.append(metrics.page_grouping_accuracy)
        page_pairs += len(sample.true_boundary_pairs)

    if not boundary_scores:
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "page_grouping_accuracy": 0.0,
            "streams_evaluated": 0,
            "page_pairs_evaluated": 0,
        }

    return {
        "boundary_precision": float(np.mean([s[0] for s in boundary_scores])),
        "boundary_recall": float(np.mean([s[1] for s in boundary_scores])),
        "boundary_f1": float(np.mean([s[2] for s in boundary_scores])),
        "page_grouping_accuracy": float(np.mean(grouping_scores)),
        "streams_evaluated": len(samples),
        "page_pairs_evaluated": page_pairs,
    }


def build_training_matrix(
    adapter: DocSplitAdapter,
    dataset_rows: list[dict[str, Any]],
    feature_builder: PageFeatureBuilder | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_builder = feature_builder or PageFeatureBuilder()
    features: list[list[float]] = []
    labels: list[int] = []
    for sample in iter_stream_samples(adapter, dataset_rows):
        pairs = feature_builder.build_all_pairs(sample.pages)
        page_to_group: dict[int, int] = {}
        for gi, group in enumerate(sample.annotation.groups):
            for p in group:
                page_to_group[p] = gi
        for pair in pairs:
            same_doc = page_to_group.get(pair.page_a_number) == page_to_group.get(pair.page_b_number)
            features.append(pair.features.tolist())
            labels.append(1 if same_doc else 0)
    if not features:
        raise ValueError("No training examples produced from dataset rows")
    return np.array(features, dtype=np.float64), np.array(labels, dtype=np.int64)
