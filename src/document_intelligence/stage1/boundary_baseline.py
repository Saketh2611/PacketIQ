"""Rule/threshold baseline for page-pair boundary detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from document_intelligence.config.settings import get_settings
from document_intelligence.ingestion.page_extractor import PageRepresentation
from document_intelligence.stage1.page_features import PageFeatureBuilder, PagePairFeatures
from document_intelligence.stage1.similarity import cosine_similarity


@dataclass
class BoundaryDecision:
    page_a: int
    page_b: int
    score: float
    is_boundary: bool
    method: str = "baseline_rule"


class BoundaryBaseline:
    """
    Baseline A: weighted rule/threshold using engineered signals.
    Baseline B variant: embedding-only cosine similarity threshold.
    """

    def __init__(
        self,
        threshold: float | None = None,
        mode: str = "weighted",
    ) -> None:
        settings = get_settings()
        self.threshold = threshold if threshold is not None else settings.boundary_threshold
        self.mode = mode
        self.feature_builder = PageFeatureBuilder()
        self.weights = np.array(
            [
                settings.semantic_weight,
                0.1,
                settings.structural_weight * 0.5,
                settings.layout_weight,
                settings.layout_weight * 0.5,
                0.05,
                0.05,
                settings.structural_weight * 0.5,
                settings.type_agreement_weight,
            ]
        )
        total = self.weights.sum()
        if total > 0:
            self.weights = self.weights / total

    def score_pair(self, pair: PagePairFeatures) -> float:
        if self.mode == "embedding_only":
            return float(pair.features[0])
        return float(np.dot(pair.features, self.weights))

    def predict_pairs(self, pages: list[PageRepresentation]) -> list[BoundaryDecision]:
        pairs = self.feature_builder.build_all_pairs(pages)
        decisions: list[BoundaryDecision] = []
        for pair in pairs:
            score = self.score_pair(pair)
            # High score => same document => NOT a boundary
            is_boundary = score < self.threshold
            decisions.append(
                BoundaryDecision(
                    page_a=pair.page_a_number,
                    page_b=pair.page_b_number,
                    score=score,
                    is_boundary=is_boundary,
                    method=f"baseline_{self.mode}",
                )
            )
        return decisions

    def predict_embedding_only(self, pages: list[PageRepresentation], threshold: float | None = None) -> list[BoundaryDecision]:
        old_mode = self.mode
        old_threshold = self.threshold
        self.mode = "embedding_only"
        if threshold is not None:
            self.threshold = threshold
        result = self.predict_pairs(pages)
        self.mode = old_mode
        self.threshold = old_threshold
        return result
