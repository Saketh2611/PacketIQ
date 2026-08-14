"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import fitz

from document_intelligence.config.settings import get_settings
from document_intelligence.ingestion.ocr import enrich_page_with_ocr
from document_intelligence.ingestion.page_extractor import PageExtractor, PageRepresentation
from document_intelligence.ingestion.pdf_loader import PDFLoader
from document_intelligence.stage1.boundary_baseline import BoundaryBaseline
from document_intelligence.stage1.boundary_classifier import BoundaryClassifier
from document_intelligence.stage1.document_classifier import DocumentClassifier
from document_intelligence.stage1.grouping import DocumentGroup, decisions_to_groups
from document_intelligence.stage1.page_features import PageFeatureBuilder
from document_intelligence.stage2.chunking import DocumentChunker
from document_intelligence.stage2.schema import StructuredDocument
from document_intelligence.stage2.structure import StructureExtractor
from document_intelligence.stage3.embeddings import EmbeddingModel
from document_intelligence.stage3.retriever import EvidenceRetriever
from document_intelligence.stage3.vector_store import create_vector_store, index_chunks
from document_intelligence.utils.io import write_json
from document_intelligence.utils.logging import get_logger
from document_intelligence.utils.timing import timer

logger = get_logger(__name__)


class DocumentIntelligencePipeline:
    """Orchestrate the 3-stage document intelligence pipeline."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.page_extractor = PageExtractor()
        self.boundary_baseline = BoundaryBaseline()
        self.boundary_classifier: BoundaryClassifier | None = None
        self.doc_classifier = DocumentClassifier(use_embedding=False)
        self.structure_extractor = StructureExtractor()
        self.chunker = DocumentChunker()
        self.embedder = EmbeddingModel()
        self.vector_store = create_vector_store()
        self.retriever: EvidenceRetriever | None = None

    def _load_classifier(self) -> BoundaryClassifier | None:
        model_path = self.settings.models_dir / "boundary_classifier.joblib"
        if model_path.exists():
            clf = BoundaryClassifier()
            clf.load(model_path)
            return clf
        return None

    def extract_pages(self, pdf_path: Path | str, packet_id: str | None = None) -> tuple[list[PageRepresentation], dict[str, Any]]:
        loader = PDFLoader(pdf_path, packet_id)
        doc, packet = loader.load()
        pages: list[PageRepresentation] = []
        try:
            for page_ref in packet.pages:
                page = self.page_extractor.extract(doc, page_ref.page_number, page_ref.page_id)
                page = enrich_page_with_ocr(doc, page)
                pages.append(page)
        finally:
            doc.close()
        meta = {
            "packet_id": packet.packet_id,
            "source_file": packet.source_file,
            "page_count": packet.page_count,
            "file_hash": packet.file_hash,
        }
        return pages, meta

    def run_stage1(
        self,
        pdf_path: Path | str,
        packet_id: str | None = None,
        method: str = "baseline",
    ) -> dict[str, Any]:
        with timer("stage1") as t:
            pages, meta = self.extract_pages(pdf_path, packet_id)
            feature_builder = PageFeatureBuilder(embedder=self.embedder)
            pairs = feature_builder.build_all_pairs(pages)

            if method == "learned":
                clf = self._load_classifier()
                if clf is None:
                    logger.warning("Learned classifier not found, falling back to baseline")
                    decisions = self.boundary_baseline.predict_pairs(pages)
                else:
                    decisions = clf.predict_decisions(pairs)
            elif method == "embedding":
                decisions = self.boundary_baseline.predict_embedding_only(pages)
            else:
                decisions = self.boundary_baseline.predict_pairs(pages)

            groups = decisions_to_groups(len(pages), decisions, meta["packet_id"])
            documents: list[dict[str, Any]] = []
            for group in groups:
                group_pages = [p for p in pages if p.page_number in group.page_numbers]
                classification = self.doc_classifier.classify_pages(group_pages)
                documents.append(
                    {
                        "document_id": group.document_id,
                        "page_start": group.page_start,
                        "page_end": group.page_end,
                        "page_numbers": group.page_numbers,
                        "document_type": classification.document_type,
                        "confidence": classification.confidence,
                        "candidates": classification.candidates,
                        "group_confidence": group.group_confidence,
                    }
                )

        result = {
            **meta,
            "method": method,
            "processing_time_seconds": t.elapsed_seconds,
            "boundaries": [
                {"page_a": d.page_a, "page_b": d.page_b, "score": d.score, "is_boundary": d.is_boundary}
                for d in decisions
            ],
            "documents": documents,
            "warnings": [w for p in pages for w in p.warnings],
        }
        return result

    def run_stage2(self, stage1_output: dict[str, Any], pdf_path: Path | str) -> list[StructuredDocument]:
        pages, meta = self.extract_pages(pdf_path, stage1_output.get("packet_id"))
        page_map = {p.page_number: p for p in pages}
        structured: list[StructuredDocument] = []

        for doc_info in stage1_output.get("documents", []):
            from document_intelligence.stage1.grouping import DocumentGroup
            from document_intelligence.stage1.document_classifier import ClassificationResult

            group = DocumentGroup(
                document_id=doc_info["document_id"],
                page_numbers=doc_info["page_numbers"],
                page_start=doc_info["page_start"],
                page_end=doc_info["page_end"],
                group_confidence=doc_info.get("group_confidence", 1.0),
            )
            group_pages = [page_map[p] for p in doc_info["page_numbers"] if p in page_map]
            classification = ClassificationResult(
                document_type=doc_info["document_type"],
                confidence=doc_info["confidence"],
                candidates=doc_info.get("candidates", []),
            )
            doc = self.structure_extractor.extract(
                meta["packet_id"],
                meta["source_file"],
                group,
                group_pages,
                classification,
            )
            structured.append(doc)
        return structured

    def build_index(self, structured_docs: list[StructuredDocument]) -> dict[str, Any]:
        with timer("indexing") as t:
            all_chunks = []
            for doc in structured_docs:
                all_chunks.extend(self.chunker.chunk_document(doc))
            self.vector_store = index_chunks(all_chunks, self.embedder, self.vector_store)
            index_path = self.settings.indexes_dir / "default"
            self.vector_store.save(index_path)
            self.retriever = EvidenceRetriever(self.vector_store, self.embedder)
        return {
            "chunks_indexed": len(all_chunks),
            "index_path": str(index_path),
            "indexing_time_seconds": t.elapsed_seconds,
        }

    def load_index(self, index_path: Path | str | None = None) -> None:
        path = Path(index_path or self.settings.indexes_dir / "default")
        self.vector_store = create_vector_store()
        self.vector_store.load(path)
        self.retriever = EvidenceRetriever(self.vector_store, self.embedder)

    def query(self, query_text: str, top_k: int | None = None) -> dict[str, Any]:
        if self.retriever is None:
            self.load_index()
        assert self.retriever is not None
        response = self.retriever.retrieve(query_text, top_k=top_k)
        return {
            "query": response.query,
            "results": [r.model_dump() for r in response.results],
            "latency_seconds": response.latency_seconds,
            "warnings": response.warnings,
        }
