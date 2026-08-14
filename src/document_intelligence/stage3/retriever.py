"""Evidence retrieval pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from document_intelligence.config.settings import get_settings
from document_intelligence.stage2.schema import Evidence
from document_intelligence.stage3.embeddings import EmbeddingModel
from document_intelligence.stage3.reranker import Reranker
from document_intelligence.stage3.scoring import combine_scores, normalize_scores
from document_intelligence.stage3.vector_store import VectorStore
from document_intelligence.utils.logging import get_logger
from document_intelligence.utils.timing import TimerResult, timer

logger = get_logger(__name__)


@dataclass
class RetrievalResponse:
    query: str
    results: list[Evidence]
    latency_seconds: float = 0.0
    warnings: list[str] | None = None


class EvidenceRetriever:
    """Retrieve ranked evidence from vector store."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingModel | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder or EmbeddingModel()
        self.reranker = reranker or Reranker()
        settings = get_settings()
        self.top_k = settings.top_k
        self.retrieval_top_n = settings.retrieval_top_n
        self.rerank_top_n = settings.rerank_top_n

    @staticmethod
    def normalize_query(query: str) -> str:
        q = query.strip()
        q = re.sub(r"\s+", " ", q)
        return q

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_type: str | None = None,
        use_reranker: bool | None = None,
    ) -> RetrievalResponse:
        top_k = top_k or self.top_k
        query = self.normalize_query(query)
        warnings: list[str] = []

        with timer("retrieval") as t:
            query_vec = self.embedder.encode([query])[0]
            candidates = self.store.search(query_vec, self.retrieval_top_n)

            if document_type:
                candidates = [(r, s) for r, s in candidates if r.document_type == document_type]

            if not candidates:
                warnings.append("no_results")
                return RetrievalResponse(query=query, results=[], latency_seconds=t.elapsed_seconds, warnings=warnings)

            if use_reranker is None:
                use_reranker = self.reranker.enabled

            if use_reranker:
                rerank_candidates = candidates[: self.rerank_top_n]
                reranked = self.reranker.rerank(query, rerank_candidates)
                norm_vec = normalize_scores([s for _, s, _ in reranked])
                evidence_list: list[Evidence] = []
                for i, (rec, vec_score, rr_score) in enumerate(reranked[:top_k]):
                    final = combine_scores(norm_vec[i], rr_score)
                    evidence_list.append(
                        Evidence(
                            document_id=rec.document_id,
                            document_type=rec.document_type,
                            page=rec.page_start,
                            chunk_id=rec.chunk_id,
                            evidence=rec.text,
                            score=final,
                            vector_score=vec_score,
                            rerank_score=rr_score,
                        )
                    )
            else:
                norm_scores = normalize_scores([s for _, s in candidates])
                evidence_list = []
                for (rec, vec_score), norm_score in zip(candidates[:top_k], norm_scores[:top_k]):
                    evidence_list.append(
                        Evidence(
                            document_id=rec.document_id,
                            document_type=rec.document_type,
                            page=rec.page_start,
                            chunk_id=rec.chunk_id,
                            evidence=rec.text,
                            score=norm_score,
                            vector_score=vec_score,
                        )
                    )

        logger.info(
            "Retrieval complete",
            query=query[:100],
            results=len(evidence_list),
            latency=t.elapsed_seconds,
        )
        return RetrievalResponse(query=query, results=evidence_list, latency_seconds=t.elapsed_seconds, warnings=warnings)
