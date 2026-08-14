"""Tests for retrieval."""

import numpy as np

from document_intelligence.stage2.schema import Chunk
from document_intelligence.stage3.embeddings import EmbeddingModel
from document_intelligence.stage3.retriever import EvidenceRetriever
from document_intelligence.stage3.vector_store import FAISSVectorStore, VectorRecord, index_chunks


def test_vector_index_search():
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            document_type="invoice",
            page_start=1,
            page_end=1,
            text="Total Amount: 52340",
        ),
        Chunk(
            chunk_id="c2",
            document_id="d2",
            document_type="resume",
            page_start=2,
            page_end=2,
            text="Experience: Software Engineer",
        ),
    ]
    embedder = EmbeddingModel()
    store = index_chunks(chunks, embedder, FAISSVectorStore())
    assert store.size() == 2
    q = embedder.encode(["invoice total amount"])[0]
    results = store.search(q, 2)
    assert len(results) >= 1


def test_retrieval_ranking():
    chunks = [
        Chunk(chunk_id="c1", document_id="d1", document_type="invoice", page_start=1, page_end=1, text="Total Amount: 52340"),
        Chunk(chunk_id="c2", document_id="d2", document_type="resume", page_start=2, page_end=2, text="Skills: Python"),
    ]
    embedder = EmbeddingModel()
    store = index_chunks(chunks, embedder, FAISSVectorStore())
    retriever = EvidenceRetriever(store, embedder)
    response = retriever.retrieve("What is the total amount on the invoice?", top_k=1)
    assert len(response.results) >= 1
    assert response.results[0].document_type == "invoice"
