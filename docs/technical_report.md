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

- **Stage 1**: Macro-averaged boundary precision/recall/F1 and page grouping accuracy **per stream**, evaluated on `nutrientdocs/doc-split-benchmark` (config `our200`, 200 test streams). Classifier trained on `nutrientdocs/openpss-mirror` (config `SHORT`, train split, 40,715 rows).
- **Stage 2**: Provenance correctness on sample PDF only (datasets lack structured-output ground truth)
- **Stage 3**: Recall@k, Precision@k, MRR, nDCG on a small smoke-test query set (no labeled queries in page-stream datasets)
- **Resources**: Wall-clock latency, RSS memory, model size

Train/eval separation: classifier trained on OpenPSS mirror train; never evaluated on OpenPSS mirror test during benchmark reporting. `doc-split-v2` was referenced in the assignment but was not accessible on HuggingFace Hub; `openpss-mirror` is the verified public training mirror per dataset card.

## 10. Benchmark Results

| Method | Boundary F1 | Page Grouping Accuracy |
|--------|-------------|------------------------|
| baseline_rule | 0.799 | 0.715 |
| baseline_embedding | 0.768 | 0.715 |
| learned_classifier | 0.774 | 0.723 |

Run date: 2026-08-15. Hash embedding fallback used. See [benchmark_report.md](benchmark_report.md) for full tables.

## 11. Failure Analysis

| Failure | Handling |
|---------|----------|
| Empty PDF | PDFLoadError with clear message |
| Scanned pages | OCR fallback with confidence |
| Unknown doc type | Explicit `unknown` label |
| Low boundary confidence | Score retained in output |
| No retrieval match | Empty results + warning |

## 12. Resource & Performance Analysis

| Metric | Value |
|--------|-------|
| Classifier training time | 342.7 s |
| Stage 1 eval (3 methods) | ~18 s total |
| Peak RSS | 7,987 MB (full train feature build) |
| Saved classifier size | ~1.5 KB |

Defaults target CPU execution. Hash embedding fallback used when Sentence Transformers unavailable. Reranker disabled by default.

## 13. Trade-offs

- Heuristic classification is fast but less accurate than trained embedding classifier
- Flat FAISS index is simple but doesn't scale to millions of chunks
- Basic table extraction vs. dedicated table models

## 14. Future Improvements

- Re-run benchmarks with Sentence Transformers embeddings (disable hash fallback)
- PaddleOCR for production OCR
- Confidence calibration (Platt scaling)
- Qdrant adapter for distributed deployment
- Labeled retrieval query set for Stage 3 evaluation

## 15. AI Usage Declaration

- **AI-assisted code generation**: Implementation scaffolded and refined with Cursor AI coding assistant
- **Human engineering decisions**: Architecture (3-stage hybrid), feature selection, model choices, evaluation methodology, dataset role mapping (openpss-mirror vs doc-split-benchmark vs doc-split-v2)
- **Experiments performed**: Dataset-backed Stage 1 benchmark on 200 test streams; classifier trained on 40,715 OpenPSS mirror rows; end-to-end sample PDF pipeline
- **Validation performed**: 34 pytest tests (including dataset adapter and eval tests), dataset schema inspection, benchmark JSON artifacts
