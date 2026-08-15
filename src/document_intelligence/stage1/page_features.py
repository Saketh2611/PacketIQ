"""Build numerical features for adjacent page pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from document_intelligence.ingestion.page_extractor import PageRepresentation
from document_intelligence.stage1.similarity import (
    block_position_similarity,
    cosine_similarity,
    heading_style_similarity,
    jaccard_entities,
    layout_similarity,
    normalized_overlap,
    structural_similarity,
    text_length_ratio,
    tokenize,
)


FEATURE_NAMES = [
    "semantic_cosine",
    "token_jaccard",
    "text_length_ratio",
    "layout_similarity",
    "block_position_similarity",
    "heading_style_similarity",
    "entity_overlap",
    "structural_similarity",
    "type_agreement",
]


@dataclass
class PagePairFeatures:
    page_a_number: int
    page_b_number: int
    features: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: FEATURE_NAMES.copy())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.features.tolist()))


class PageFeatureBuilder:
    """Engineer deterministic page-pair features."""

    def __init__(self, embedder: Any | None = None) -> None:
        self._embedder = embedder
        self._embedding_cache: dict[str, np.ndarray] = {}

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from document_intelligence.stage3.embeddings import EmbeddingModel

            self._embedder = EmbeddingModel()
        return self._embedder

    def _get_embedding(self, page: PageRepresentation) -> np.ndarray:
        if page.page_id in self._embedding_cache:
            return self._embedding_cache[page.page_id]
        text = page.text.strip() or " "
        emb = self.embedder.encode([text])[0]
        self._embedding_cache[page.page_id] = emb
        return emb

    def _prefetch_embeddings(self, pages: list[PageRepresentation]) -> None:
        missing = [page for page in pages if page.page_id not in self._embedding_cache]
        if not missing:
            return
        texts = [page.text.strip() or " " for page in missing]
        embeddings = self.embedder.encode(texts)
        for page, emb in zip(missing, embeddings):
            self._embedding_cache[page.page_id] = emb

    def _predict_page_type_hint(self, page: PageRepresentation) -> str:
        from document_intelligence.stage1.document_classifier import HeuristicDocumentClassifier

        clf = HeuristicDocumentClassifier()
        result = clf.classify_pages([page])
        return result.document_type

    def build_pair(
        self,
        page_a: PageRepresentation,
        page_b: PageRepresentation,
        use_type_hint: bool = True,
    ) -> PagePairFeatures:
        emb_a = self._get_embedding(page_a)
        emb_b = self._get_embedding(page_b)
        sem = cosine_similarity(emb_a, emb_b)
        tok_a, tok_b = tokenize(page_a.text), tokenize(page_b.text)
        token_j = normalized_overlap(tok_a, tok_b)
        tlr = text_length_ratio(page_a.text_length, page_b.text_length)
        layout = layout_similarity(page_a, page_b)
        block_pos = block_position_similarity(page_a.blocks, page_b.blocks)
        heading = heading_style_similarity(page_a, page_b)
        entity = jaccard_entities(page_a.text, page_b.text)
        structural = structural_similarity(page_a, page_b)

        type_agreement = 0.0
        type_a: str | None = None
        type_b: str | None = None
        if use_type_hint:
            type_a = self._predict_page_type_hint(page_a)
            type_b = self._predict_page_type_hint(page_b)
            type_agreement = 1.0 if type_a == type_b and type_a != "unknown" else 0.0

        features = np.array(
            [sem, token_j, tlr, layout, block_pos, heading, entity, structural, type_agreement],
            dtype=np.float64,
        )
        return PagePairFeatures(
            page_a_number=page_a.page_number,
            page_b_number=page_b.page_number,
            features=features,
            metadata={"type_a": type_a, "type_b": type_b},
        )

    def build_all_pairs(self, pages: list[PageRepresentation]) -> list[PagePairFeatures]:
        self._prefetch_embeddings(pages)
        pairs: list[PagePairFeatures] = []
        for i in range(len(pages) - 1):
            pairs.append(self.build_pair(pages[i], pages[i + 1]))
        return pairs
