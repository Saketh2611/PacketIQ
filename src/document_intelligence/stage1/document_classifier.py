"""Document type classification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from document_intelligence.ingestion.page_extractor import PageRepresentation
from document_intelligence.stage1.confidence import classifier_confidence


@dataclass
class ClassificationResult:
    document_type: str
    confidence: float
    candidates: list[tuple[str, float]] = field(default_factory=list)
    method: str = "heuristic"


# Heuristic keyword patterns (baseline, not exclusive)
TYPE_PATTERNS: dict[str, list[str]] = {
    "invoice": [r"\binvoice\b", r"\bbill to\b", r"\btotal amount\b", r"\bsubtotal\b"],
    "resume": [r"\bresume\b", r"\bcurriculum vitae\b", r"\bexperience\b", r"\beducation\b", r"\bskills\b"],
    "passport": [r"\bpassport\b", r"\bnationality\b", r"\bdate of birth\b", r"\bplace of birth\b"],
    "contract": [r"\bagreement\b", r"\bcontract\b", r"\bterms and conditions\b", r"\bparty\b"],
    "receipt": [r"\breceipt\b", r"\bpaid\b", r"\btransaction\b"],
    "letter": [r"\bdear\b", r"\bsincerely\b", r"\bto whom it may concern\b"],
    "report": [r"\bexecutive summary\b", r"\bintroduction\b", r"\bconclusion\b", r"\babstract\b"],
    "form": [r"\bapplication form\b", r"\bplease fill\b", r"\bcheckbox\b"],
}


class HeuristicDocumentClassifier:
    """Rule/keyword baseline classifier."""

    def classify_text(self, text: str) -> ClassificationResult:
        text_lower = text.lower()
        scores: dict[str, float] = {}
        for doc_type, patterns in TYPE_PATTERNS.items():
            matches = sum(1 for p in patterns if re.search(p, text_lower))
            if matches:
                scores[doc_type] = matches / len(patterns)

        if not scores:
            return ClassificationResult(document_type="unknown", confidence=0.0, method="heuristic")

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = ranked[0]
        return ClassificationResult(
            document_type=best_type,
            confidence=classifier_confidence(best_score),
            candidates=[(t, classifier_confidence(s)) for t, s in ranked[:3]],
            method="heuristic",
        )

    def classify_pages(self, pages: list[PageRepresentation]) -> ClassificationResult:
        combined = "\n".join(p.text for p in pages)
        return self.classify_text(combined)


class EmbeddingDocumentClassifier:
    """Embedding-based nearest-prototype classifier."""

    def __init__(self) -> None:
        self._embedder = None
        self._prototypes: dict[str, np.ndarray] = {}

    @property
    def embedder(self):
        if self._embedder is None:
            from document_intelligence.stage3.embeddings import EmbeddingModel

            self._embedder = EmbeddingModel()
        return self._embedder

    def fit_prototypes(self, labeled_texts: dict[str, list[str]]) -> None:
        for doc_type, texts in labeled_texts.items():
            if not texts:
                continue
            embs = self.embedder.encode(texts)
            self._prototypes[doc_type] = np.mean(embs, axis=0)

    def classify_pages(self, pages: list[PageRepresentation]) -> ClassificationResult:
        text = "\n".join(p.text for p in pages)
        if not text.strip():
            return ClassificationResult(document_type="unknown", confidence=0.0, method="embedding")

        if not self._prototypes:
            return HeuristicDocumentClassifier().classify_pages(pages)

        query = self.embedder.encode([text])[0]
        scores: dict[str, float] = {}
        for doc_type, proto in self._prototypes.items():
            norm_q = np.linalg.norm(query)
            norm_p = np.linalg.norm(proto)
            if norm_q == 0 or norm_p == 0:
                sim = 0.0
            else:
                sim = float(np.dot(query, proto) / (norm_q * norm_p))
            scores[doc_type] = sim

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = ranked[0]
        confidence = classifier_confidence(best_score)
        settings_threshold = 0.3
        if confidence < settings_threshold:
            return ClassificationResult(
                document_type="unknown",
                confidence=confidence,
                candidates=[(t, classifier_confidence(s)) for t, s in ranked[:3]],
                method="embedding",
            )
        return ClassificationResult(
            document_type=best_type,
            confidence=confidence,
            candidates=[(t, classifier_confidence(s)) for t, s in ranked[:3]],
            method="embedding",
        )


class DocumentClassifier:
    """Combined classifier: heuristic baseline with optional embedding upgrade."""

    def __init__(self, use_embedding: bool = True) -> None:
        self.heuristic = HeuristicDocumentClassifier()
        self.embedding = EmbeddingDocumentClassifier() if use_embedding else None

    def classify_pages(self, pages: list[PageRepresentation]) -> ClassificationResult:
        heuristic_result = self.heuristic.classify_pages(pages)
        if self.embedding is None:
            return heuristic_result

        emb_result = self.embedding.classify_pages(pages)
        if emb_result.confidence > heuristic_result.confidence:
            return emb_result
        return heuristic_result
