# Technical Report

## 1. Problem Understanding

Document packets merge multiple logical documents (invoices, resumes, passports) into a single PDF without
explicit separators. The system must detect boundaries, classify types, structure content, and retrieve
evidence — not generate answers. The problem breaks cleanly into three sequential stages, each with its own
input/output contract and its own measurable success criteria:

1. **Boundary detection & grouping** — given N pages, decide which adjacent pairs are same-document vs.
   different-document, then collapse that into contiguous page groups with a predicted type.
2. **Structuring** — given a page group, produce a schema-validated representation (title, sections,
   content blocks, page provenance) suitable for chunking.
3. **Retrieval** — given a query, return ranked evidence chunks with document ID, page, and score — not a
   generated answer.

Each stage's output is the next stage's input, so errors compound downstream; this is reflected in how the
pipeline is benchmarked (each stage evaluated on its own metrics, and Stage 3's evaluation additionally
depends on Stage 1/2 already being correct on the same sample packet).

## 2. Overall Approach

Hybrid architecture using:
- Engineered page-pair features (not LLM boundary detection)
- Lightweight classifiers (Logistic Regression baseline)
- General-purpose embeddings (Sentence Transformers, `all-MiniLM-L6-v2`)
- Local FAISS vector store
- Optional cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

## 3. Architecture

Three-stage pipeline: Ingestion → Boundary/Classification → Structure/Chunk → Embed/Retrieve. See
[architecture.md](architecture.md).

## 4. Three Most Important Technical Decisions

1. **Page-pair features over LLM boundary detection** — measurable, deterministic, resource-efficient, and
   avoids depending on an LLM's context window scaling with packet length.
2. **PyMuPDF native extraction before OCR** — avoids unnecessary rendering and compute; OCR only triggers
   when native text extraction falls below a length threshold.
3. **Evidence-centric retrieval, with an explicit reranking on/off switch surfaced end-to-end** — returns
   provenance-linked text chunks rather than generated responses, and lets reranking be toggled per-request
   (via `use_reranker` on the pipeline, API, and CLI) rather than only at the process/config level, which
   made the Stage 3 before/after comparison in this report possible without restarting the service.

## 5. Technology Selection and Alternatives

| Choice | Alternative Considered | Rationale |
|--------|----------------------|-----------|
| PyMuPDF | pdfplumber | Faster, block-level extraction |
| FAISS | Chroma, Qdrant | Zero infrastructure, local default |
| all-MiniLM-L6-v2 | Larger embedding models | CPU-compatible, good quality/size trade-off |
| Logistic Regression | XGBoost, LightGBM | Fast baseline, interpretable feature weights |
| pytesseract | PaddleOCR | Simpler install; PaddleOCR configurable later |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Larger cross-encoders | Small, CPU-feasible, standard MS MARCO reranker |

## 6. Stage 1 Method

Adjacent page pairs receive 9 engineered features: semantic cosine, token Jaccard, text length ratio,
layout similarity, block position, heading style, entity overlap, structural similarity, type agreement.

Three methods compared:
- **Baseline A (rule)**: weighted rule/threshold over all 9 features
- **Baseline B (embedding)**: cosine similarity threshold only
- **Final (learned)**: Logistic Regression on the full 9-feature vector, trained on 32,408 page-pair
  examples from `openpss-mirror`

## 7. Stage 2 Method

Native PDF blocks are grouped into sections using a font-size heading heuristic. Tables are preserved as
logical chunks. Metadata includes content hashes and extraction method per document. Each structured
document tracks `page_start`/`page_end` provenance back to the source PDF.

## 8. Stage 3 Method

Structural chunks are embedded with Sentence Transformers and stored in FAISS (inner product on normalized
vectors). Query → embed → top-50 vector search → optional cross-encoder rerank of top-20 → return top-k
evidence. When reranking is enabled, the final score is a weighted combination of the normalized vector
score and the normalized cross-encoder score (`combine_scores`, 0.4/0.6 weighting).

## 9. Evaluation Methodology

- **Stage 1**: Macro-averaged boundary precision/recall/F1 and page grouping accuracy **per stream**,
  evaluated on `nutrientdocs/doc-split-benchmark` (config `our200`, 200 test streams, 694 page pairs).
  Classifier trained on `nutrientdocs/openpss-mirror` (config `SHORT`, train split, 40,715 rows / 204
  streams). Train/test are always disjoint datasets — the classifier never sees `doc-split-benchmark`
  during training, only during evaluation.
- **Stage 2**: Documents processed, avg sections/blocks per document, and provenance correctness on the
  sample packet (invoice + resume + passport, 6 pages, 3 documents) via `evaluate_stage2()`. No
  structured-output ground-truth dataset exists publicly for this task, so this is measured against the
  bundled sample packet rather than at dataset scale.
- **Stage 3**: Recall@k, Precision@k, MRR, nDCG on 6 hand-labeled queries (2 per sample document type),
  each with a `relevant_ids` ground truth verified by inspecting the real extracted chunk text. Run twice —
  once with the reranker forced off, once forced on — to produce a controlled before/after comparison, since
  the page-stream datasets used for Stage 1 provide no retrieval query/evidence pairs.
- **Resources**: wall-clock latency, peak RSS memory, model size, per the fields in
  `evaluation/resource_metrics.py`.

Dataset roles: `doc-split-v2`, named in the assignment as the development dataset
(`https://huggingface.co/nutrientdocs/doc-split-v2`), is **not a HuggingFace dataset** — it is a commercial
model card (Nutrient's proprietary page-stream segmentation model, `nutrient-commercial` license, weights
not downloadable). Its own model card lists `doc-split-benchmark` as its training/eval data, so there is
nothing loadable at that URL via `datasets.load_dataset()`. Using it would also conflict with the
assignment's explicit restriction against models "specifically trained for the DocSplit or DocSplit v2
benchmark." In its place, `openpss-mirror` (train/dev) and `doc-split-benchmark` (test/eval only, never
trained on) are used — both real, loadable page-stream segmentation datasets from the same publisher. Full
explanation: [docs/datasets.md](datasets.md).

## 10. Benchmark Results

### Stage 1 — Boundary Detection

| Method | Boundary F1 | Page Grouping Accuracy |
|--------|-------------|------------------------|
| baseline_rule | 0.533 | 0.573 |
| baseline_embedding | 0.517 | 0.582 |
| learned_classifier | **0.784** | **0.722** |

Measured 2026-08-16 with real `all-MiniLM-L6-v2` embeddings (not the hash fallback) and the leakage-fixed
classifier (scaler fit only on training features, after the train/val split). Reproduced independently on a
second run the same day with matching results to three decimal places.

### Stage 2 — Structuring

| Metric | Value |
|--------|-------|
| Documents processed | 3 |
| Avg sections / document | 1.0 |
| Avg content blocks / document | 11.0 |
| Provenance correctness | 3 / 3 |
| Processing time | 0.015 s |

### Stage 3 — Retrieval

| Metric | Vector Only | Vector + Reranker |
|--------|-------------|-------------------|
| Top-1 accuracy | 3 / 6 | **4 / 6** |
| Recall@1 | 0.500 | **0.667** |
| MRR | 0.589 | **0.708** |
| nDCG | 0.648 | **0.738** |

Full tables and per-query breakdown: [benchmark_report.md](benchmark_report.md).

## 11. Failure Analysis

### System-level handling

| Failure | Handling |
|---------|----------|
| Empty PDF | `PDFLoadError` with clear message |
| Scanned pages | OCR fallback with confidence |
| Unknown doc type | Explicit `unknown` label |
| Low boundary confidence | Score retained in output, not discarded |
| No retrieval match | Empty results + warning |

### Stage 3 retrieval failure cases (from the 6-query benchmark)

- **Fixed by reranking:** "What is the total amount on the invoice?" — vector-only search ranked the bare
  heading `"INVOICE"` above the actual line `"Total Amount: ₹52,340"`. Cause: `all-MiniLM-L6-v2`'s
  bi-encoder embeds short, low-information chunks close to many queries in cosine space when the indexed
  corpus is small (33 chunks here). The cross-encoder reranker, which scores query+chunk pairs jointly
  rather than comparing independent embeddings, correctly promoted the true answer to rank 1.
- **Not fixed by reranking:** "What skills does the candidate have?" — both modes return the chunk
  containing only the heading `"Skills:"` instead of the chunk containing the actual list
  (`"Python, Machine Learning, FastAPI"`). This is a chunking problem, not a ranking problem: the heading and
  its content are separate chunks with no strong semantic link, so reranking the same candidate set cannot
  recover a chunk that was never a close match. Likely fix: merge short headings with their immediately
  following content block during Stage 2 chunking, rather than one-line-per-block.
- **Reranking makes one case worse:** "What is the candidate's most recent job title?" — reranking replaces
  one wrong answer with a different wrong answer (`"Software Engineer"`, a generic title on an unrelated
  line) that the cross-encoder scored as a closer lexical match to "job title" than the correct chunk
  (`"Senior Developer at TechCo (2020-2024)"`). This demonstrates reranking is not a strict per-query
  improvement even though it improves aggregate metrics — worth stating plainly rather than only reporting
  the aggregate win.

## 12. Resource & Performance Analysis

| Metric | Value |
|--------|-------|
| Stage 1 classifier training time | 2,073.6 s (~34.6 min) |
| Stage 1 total wall clock | 2,128.8 s (~35.5 min) |
| Stage 1 peak RSS | 6,248 MB |
| Saved classifier size | ~1.5 KB |
| Stage 2 processing time (3 docs) | 0.015 s |
| Stage 3 indexing time (33 chunks) | ~13 s |
| Stage 3 query latency, warm, no reranker | ~0.022 s |
| Stage 3 query latency, warm, with reranker | ~0.13–0.26 s |

All defaults target CPU execution. Real Sentence Transformer embeddings used throughout (not the hash
fallback). Reranker disabled by default (`USE_RERANKER=false`), enabled per-request via `use_reranker`.

**Measurement caveat:** both the embedding model and the cross-encoder reranker load lazily on first use.
The first query in a fresh process therefore reports inflated `latency_seconds` that reflects model loading,
not retrieval cost — the reranked run's first query showed 9.6s while subsequent queries in the same run
were 0.13–0.26s. The "warm" figures above exclude that first-query cost. This is a one-time process-startup
cost, not a per-query cost, and would not recur in a long-lived server process (e.g. the FastAPI app, which
constructs the pipeline once at startup).

## 13. Trade-offs

- Heuristic/rule-based boundary detection is fast but meaningfully less accurate than the trained classifier
  (F1 0.533 vs. 0.784) — the accuracy gain from training justified the added complexity of a train/eval
  split and model artifact.
- Flat FAISS index is simple and sufficient at this scale (33 chunks in the sample packet) but doesn't
  scale to millions of chunks without an ANN index or a managed vector database.
- Reranking improves aggregate retrieval quality (Top-1 3/6 → 4/6, MRR 0.589 → 0.708) but roughly 6–12×
  increases warm per-query latency (0.022s → 0.13–0.26s) and does not fix chunking-rooted failures — a
  deliberate trade of latency for accuracy, exposed as an opt-in flag rather than baked in, so callers can
  choose per request.
- Basic table extraction (native PDF blocks) vs. a dedicated table-structure model — kept simple since the
  target document types (invoices, resumes, passports) have limited tabular complexity in the sample data.

## 14. Future Improvements

- Fix the resume "Skills:" chunking failure by merging short headings with their following content block
  during Stage 2 chunking, rather than treating every block as an independent chunk.
- Expand the Stage 3 labeled query set beyond 6 queries (more per document type, plus adversarial/no-answer
  queries) for a more statistically stable Recall@k/MRR estimate.
- Wire `--stage stage2` into `scripts/run_benchmarks.py`'s CLI so all three stages run from one command
  instead of three separate scripts.
- Confidence calibration (Platt scaling) for the boundary classifier's probability outputs.
- Qdrant or pgvector adapter for distributed deployment beyond the local FAISS flat index.
- PaddleOCR as an alternative OCR backend for production scanned-document handling.

## 15. AI Usage Declaration

- **AI-assisted code generation and debugging**: Implementation scaffolded and refined with an AI coding
  assistant (Claude), used interactively across the project — including diagnosing and fixing two real
  bugs found during benchmarking: (1) `Reranker.enabled` was not being overridden by the per-request
  `use_reranker` flag, so requesting reranking silently had no effect until `EvidenceRetriever.retrieve()`
  was patched to temporarily flip `Reranker.enabled` before calling `rerank()`; (2) `pipeline.query()` did
  not forward `use_reranker`/`document_type` to the retriever at all, and the `/retrieve` API endpoint
  accepted but silently dropped both fields from its request schema.
- **Human engineering decisions**: Architecture (3-stage hybrid), feature selection, model choices,
  evaluation methodology, dataset role mapping (`openpss-mirror` vs. `doc-split-benchmark` vs.
  `doc-split-v2`), the decision to hand-label a real Stage 3 query set rather than continue reporting a
  fabricated smoke test, and the interpretation of the reranker before/after results (including the
  decision to report the case where reranking made a result worse, rather than only the aggregate
  improvement).
- **Experiments performed**: Dataset-backed Stage 1 benchmark on 200 test streams (reproduced twice with
  matching results); classifier trained on 40,715 OpenPSS mirror rows; real Stage 2 structuring benchmark
  on the sample packet; real Stage 3 retrieval benchmark on 6 hand-labeled queries, run both with and
  without reranking for a controlled comparison.
- **Validation performed**: 36 pytest tests (including dataset adapter and eval tests), dataset schema
  inspection, benchmark JSON artifacts, manual verification of the `doc-split-v2` HuggingFace page content
  to correct an earlier inaccurate claim that it "was not accessible" (it loads fine — it's a commercial
  model card, not a dataset), and hand-verification of Stage 3 ground-truth chunk IDs against the actual
  extracted text in `outputs/samples/structured/*.json`.