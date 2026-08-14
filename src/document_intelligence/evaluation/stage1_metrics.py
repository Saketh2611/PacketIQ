"""Stage 1 evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class Stage1Metrics:
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    page_grouping_accuracy: float
    classification_accuracy: float
    confusion: list[list[int]] | None = None


def _boundaries_from_groups(groups: list[list[int]]) -> set[int]:
    """Return set of page numbers where a new document starts (after first page)."""
    boundaries: set[int] = set()
    for group in groups:
        if len(group) > 1:
            for p in group[1:]:
                boundaries.add(p)
    return boundaries


def _groups_to_boundary_set(page_count: int, groups: list[list[int]]) -> set[tuple[int, int]]:
    """Represent boundaries as page pairs (a, b) where b starts new document."""
    all_pages = sorted({p for g in groups for p in g})
    boundary_pairs: set[tuple[int, int]] = set()
    for group in groups:
        for i in range(1, len(group)):
            boundary_pairs.add((group[i - 1], group[i]))
    return boundary_pairs


def evaluate_boundaries(
    predicted_pairs: list[tuple[int, int, bool]],
    true_pairs: list[tuple[int, int, bool]],
) -> tuple[float, float, float]:
    """
    Evaluate boundary detection at page-pair level.

    Convention: is_boundary=True means pages belong to different documents.
    """
    pred_map = {(a, b): v for a, b, v in predicted_pairs}
    y_true, y_pred = [], []
    for a, b, true_val in true_pairs:
        y_true.append(int(true_val))
        y_pred.append(int(pred_map.get((a, b), False)))
    if not y_true:
        return 0.0, 0.0, 0.0
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return float(p), float(r), float(f1)


def page_grouping_accuracy(true_groups: list[list[int]], pred_groups: list[list[int]]) -> float:
    if not true_groups:
        return 0.0
    correct = 0
    total = 0
    true_page_to_group = {}
    for gi, group in enumerate(true_groups):
        for p in group:
            true_page_to_group[p] = gi
    pred_page_to_group = {}
    for gi, group in enumerate(pred_groups):
        for p in group:
            pred_page_to_group[p] = gi
    for page, true_g in true_page_to_group.items():
        if page in pred_page_to_group:
            total += 1
            if pred_page_to_group[page] == true_g:
                correct += 1
    return correct / total if total else 0.0


def evaluate_stage1(
    true_boundary_pairs: list[tuple[int, int, bool]],
    pred_boundary_pairs: list[tuple[int, int, bool]],
    true_groups: list[list[int]],
    pred_groups: list[list[int]],
    true_types: list[str],
    pred_types: list[str],
) -> Stage1Metrics:
    p, r, f1 = evaluate_boundaries(pred_boundary_pairs, true_boundary_pairs)
    pga = page_grouping_accuracy(true_groups, pred_groups)
    cls_acc = accuracy_score(true_types, pred_types) if true_types and pred_types else 0.0
    cm = confusion_matrix(true_types, pred_types).tolist() if true_types else None
    return Stage1Metrics(
        boundary_precision=p,
        boundary_recall=r,
        boundary_f1=f1,
        page_grouping_accuracy=pga,
        classification_accuracy=cls_acc,
        confusion=cm,
    )
