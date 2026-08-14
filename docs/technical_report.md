# Technical Report

## 1. Problem Understanding

Document packets merge multiple logical documents (invoices, resumes, passports) into a single PDF without explicit separators. The system must detect boundaries, classify types, structure content, and retrieve evidence—not generate answers.

## 2. Overall Approach

Hybrid architecture using:
- Engineered page-pair features (not LLM boundary detection)
- Lightweight classifiers (Logistic Regression baseline)
- General-purpose embeddings (Sentence Transformers)
- Local FAISS vector store
- Optional cross-encoder reranking

## 3. Architecture

Three-stage pipeline: Ingestion → Boundary/Classification → Structure/Chunk → Embed/Retrieve. See [architecture.md](architecture.md).

## 4. Three Most Important Technical Decisions

1. **Page-pair features over LLM boundary detection** — measurable, deterministic, resource-efficient
2. **PyMuPDF native extraction before OCR** — avoids unnecessary rendering and compute
3. **Evidence-centric retrieval** — returns provenance-linked text chunks, not generated responses

## 5. Technology Selection and Alternatives

| Choice | Alternative Considered | Rationale |
|--------|----------------------|-----------|
| PyMuPDF | pdfplumber | Faster, block-level extraction |
| FAISS | Chroma, Qdrant | Zero infrastructure, local default |
| all-MiniLM-L6-v2 | Larger embedding models | CPU-compatible, good quality/size trade-off |
| Logistic Regression | XGBoost, LightGBM | Fast baseline, interpretable features |
| pytesseract | PaddleOCR | Simpler install; PaddleOCR configurable later |

## 6. Stage 1 Method

Adjacent page pairs receive 9 engineered features: semantic cosine, token Jaccard, text length ratio, layout similarity, block position, heading style, entity overlap, structural similarity, type agreement.

Three methods compared:
- **Baseline A**: Weighted rule/threshold
- **Baseline B**: Embedding cosine only
- **Final**: Logistic Regression on full feature vector

## 7. Stage 2 Method

Native PDF blocks grouped into sections using font-size heading heuristic. Tables preserved as logical chunks. Metadata includes content hashes and extraction method per document.

## 8. Stage 3 Method

Structural chunks embedded with Sentence Transformers, stored in FAISS (inner product on normalized vectors). Query → embed → top-50 search → optional rerank top-20 → return top-k evidence.

## 9. Evaluation Methodology

- **Stage 1**: Boundary precision/recall/F1 at page-pair level; page grouping accuracy; classification accuracy
- **Stage 2**: Provenance correctness, extraction completeness (no fake accuracy without ground truth)
- **Stage 3**: Recall@k, Precision@k, MRR, nDCG on query/chunk relevance pairs
- **Resources**: Wall-clock latency, RSS memory, index size

Train/val/test split by packet ID to prevent leakage.

## 10. Benchmark Results

[RUN BENCHMARK TO FILL]

Execute: `python scripts/run_benchmarks.py` and `python scripts/generate_sample_outputs.py`

## 11. Failure Analysis

| Failure | Handling |
|---------|----------|
| Empty PDF | PDFLoadError with clear message |
| Scanned pages | OCR fallback with confidence |
| Unknown doc type | Explicit `unknown` label |
| Low boundary confidence | Score retained in output |
| No retrieval match | Empty results + warning |

## 12. Resource & Performance Analysis

[RUN BENCHMARK TO FILL]

Defaults target CPU execution. Embedding model ~80MB. Reranker disabled by default. Batch encoding for throughput.

## 13. Trade-offs

- Heuristic classification is fast but less accurate than trained embedding classifier
- Flat FAISS index is simple but doesn't scale to millions of chunks
- Basic table extraction vs. dedicated table models

## 14. Future Improvements

- Full OpenPSS mirror training for boundary classifier
- PaddleOCR for production OCR
- Confidence calibration (Platt scaling)
- Qdrant adapter for distributed deployment

## 15. AI Usage Declaration

- **AI-assisted code generation**: Implementation scaffolded and refined with Cursor AI coding assistant
- **Human engineering decisions**: Architecture (3-stage hybrid), feature selection, model choices, evaluation methodology
- **Experiments performed**: Synthetic benchmark runs, end-to-end sample pipeline on generated PDF
- **Validation performed**: pytest unit/integration tests, manual sample output verification

*Benchmark numbers on OpenPSS mirror and doc-split-benchmark: [RUN BENCHMARK TO FILL]*
