# Technical Report

## 1. Problem Understanding

Document packets merge multiple logical documents (invoices, resumes, passports) into a single PDF without
explicit separators. The system must detect boundaries, classify types, structure content, and retrieve
evidence — not generate answers. The problem breaks cleanly into three sequential stages, each with its own
input/output contract and its own measurable success criteria:

1. **Boundary detection & grouping** — given N pages, decide which adjacent pairs are same-document vs.
   different-document, then collapse that into contiguous page groups with a predicted type.
2. **Structuring** — given a page group, produce a schema-validated representation (title, sections, typed
   content blocks including tables/figures/lists, page provenance) suitable for chunking.
3. **Retrieval** — given a query, return ranked evidence chunks with document ID, page, and score — not a
   generated answer.

Each stage's output is the next stage's input, so errors compound downstream, and fixes at an earlier stage
can resolve problems that look like later-stage failures. This showed up directly during development: a
retrieval failure initially attributed to ranking was actually a Stage 2 chunking problem, and fixing
chunking resolved it without touching retrieval logic at all (see §11).

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
3. **Chunking that preserves label-content relationships, not just token windows** — short field labels
   (e.g. `"Skills:"`) are merged with their immediately following content block during chunking, and
   multi-row tables/multi-item lists are detected and preserved as single structured chunks rather than
   split into disconnected paragraph chunks. This was added after benchmarking showed a class of retrieval
   failures that reranking could not fix because the correct information was never in the same chunk as
   what the query would naturally match against — see §11.

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

Native PDF blocks are grouped into sections using a font-size heading heuristic. Consecutive
column-structured lines are detected and merged into `table` blocks (with parsed headers/rows) rather than
left as independent paragraph blocks; consecutive bullet/numbered lines are merged into `list` blocks;
image blocks are captured as `figure` blocks. Each block carries a `metadata` dict with its source bounding
box for traceability back to the original PDF layout. Each structured document tracks
`page_start`/`page_end` provenance back to the source PDF.

## 8. Stage 3 Method

Structural chunks are embedded with Sentence Transformers and stored in FAISS (inner product on normalized
vectors). During chunking, a short label block immediately followed by its content (e.g. `"Skills:"` →
list of skills) is merged into a single chunk rather than kept as two disconnected chunks, so retrieval can
match the label semantically while returning the actual content. Query → normalize (empty/whitespace-only
queries short-circuit with a validation warning, skipping the embed/search entirely) → embed → top-50 vector
search → optional cross-encoder rerank of top-20 → return top-k evidence. When reranking is enabled, the
final score is a weighted combination of the normalized vector score and the normalized cross-encoder score.

## 9. Evaluation Methodology

- **Stage 1**: Macro-averaged boundary precision/recall/F1 and page grouping accuracy **per stream**,
  evaluated on `nutrientdocs/doc-split-benchmark` (200 test streams, 694 page pairs). Classifier trained on
  `nutrientdocs/openpss-mirror` (train split, 40,715 rows). Train/test are always disjoint datasets.
- **Stage 2**: Documents processed, avg sections/blocks per document, table/figure block counts, and
  provenance correctness on the sample packet via `evaluate_stage2()`. No structured-output ground-truth
  dataset exists publicly for this task.
- **Stage 3**: Recall@k, Precision@k, MRR, nDCG on 6 hand-labeled queries (2 per sample document type).
  Ground truth (`relevant_ids`) is resolved dynamically by matching expected answer text against the
  current chunker's output — not hardcoded chunk IDs — so the evaluation stays valid as chunking logic
  changes, which mattered directly when the chunking fix below changed chunk boundaries and counts (33 →
  27 total chunks). Run separately with the reranker forced off and forced on for a controlled before/after
  comparison.
- **Resources**: wall-clock latency, peak RSS memory, model size.

Dataset roles: `doc-split-v2`, named in the assignment as the development dataset, is **not a HuggingFace
dataset** — it is a commercial model card (Nutrient's proprietary page-stream segmentation model,
`nutrient-commercial` license, weights not downloadable). Its own model card lists `doc-split-benchmark` as
its training/eval data. Using it would also conflict with the assignment's explicit restriction against
models "specifically trained for the DocSplit or DocSplit v2 benchmark." In its place, `openpss-mirror`
(train/dev) and `doc-split-benchmark` (test/eval only, never trained on) are used. Full explanation:
[docs/datasets.md](datasets.md).

## 10. Benchmark Results

### Stage 1 — Boundary Detection

| Method | Boundary F1 | Page Grouping Accuracy |
|--------|-------------|------------------------|
| baseline_rule | 0.533 | 0.573 |
| baseline_embedding | 0.517 | 0.582 |
| learned_classifier | **0.784** | **0.722** |

Independently reproduced twice on 2026-08-16, matching to four decimal places both times.

### Stage 2 — Structuring

| Metric | Value |
|--------|-------|
| Documents processed | 3 |
| Avg blocks / document | 10.0 |
| Table blocks detected | 1 |
| Provenance correctness | 3 / 3 |
| Processing time | 0.018 s |

### Stage 3 — Retrieval, before vs. after the chunking fix

| Metric | Before fix (vector) | After fix (vector) | After fix (+ reranker) |
|--------|----------------------|----------------------|--------------------------|
| Top-1 accuracy | 3 / 6 | **5 / 6** | 5 / 6 |
| Recall@1 | 0.500 | **0.833** | 0.833 |
| MRR | 0.589 | **0.867** | 0.833 |
| nDCG | 0.648 | **0.898** | 0.833 |

The chunking fix produced a larger accuracy gain than the reranker did in the previous benchmark round, and
resolved a failure the reranker could not. Full tables and per-query breakdown:
[benchmark_report.md](benchmark_report.md).

## 11. Failure Analysis

### System-level handling

| Failure | Handling |
|---------|----------|
| Empty PDF | `PDFLoadError` with clear message |
| Scanned pages | OCR fallback with confidence |
| Unknown doc type | Explicit `unknown` label |
| Low boundary confidence | Score retained in output, not discarded |
| Empty/whitespace query | Short-circuits with `empty_query` warning, no search performed |
| No retrieval match | Empty results + warning |

### Stage 3 retrieval failure cases — root cause and resolution history

- **Resolved: "Skills:" chunking failure.** Originally, "What skills does the candidate have?" retrieved
  only the bare heading chunk `"Skills:"` in both vector-only and reranked modes, because the heading and
  its content list were separate, disconnected chunks — reranking a candidate set that never contained the
  right chunk cannot fix a missing candidate. The fix was at Stage 2/chunking, not Stage 3: short label
  blocks are now merged with their following content block before chunking. After the fix, this query
  passes in both modes. This is a concrete example of the compounding-errors point in §1 — a stage-1/2
  problem masquerading as a stage-3 ranking problem.
- **New/ongoing: reranking regression on "most recent job title."** Vector-only search correctly ranks the
  true answer (`"Senior Developer at TechCo"`) first. With reranking enabled, that chunk is pushed out of
  the top 5 entirely, replaced by chunks the cross-encoder judged more lexically similar to "job title"
  (e.g. a generic "Software Engineer" mention). This is a repeat, more severe instance of a pattern also
  seen in the previous benchmark round (reranking swapping one wrong answer for another on the same query
  type) — reranking is not a strict improvement per-query, and specifically underperforms vector-only search
  on this "most recent role" query type across two independent benchmark rounds. This is a genuine
  limitation, not a one-off artifact, and is treated as such rather than averaged away in the aggregate
  metrics.
- **Not yet tested at scale:** all Stage 3 failure analysis above is on a 6-query hand-labeled set against a
  3-document sample packet. The chunking fix's effect on Stage 1-scale data (200+ streams) is untested,
  since Stage 1's dataset provides boundary labels only, not structuring or retrieval ground truth.

## 12. Resource & Performance Analysis

| Metric | Value |
|--------|-------|
| Stage 1 classifier training time | ~2,074–2,123 s (~35 min), two independent runs |
| Stage 1 peak RSS | 3,644–6,248 MB across runs |
| Saved classifier size | ~1.5 KB |
| Stage 2 processing time (3 docs) | 0.018 s |
| Stage 3 indexing time (27 chunks) | 13.5–14.7 s |
| Stage 3 query latency, warm, no reranker | ~0.02–0.03 s |
| Stage 3 query latency, warm, with reranker | ~0.15–0.31 s |

All defaults target CPU execution. Real Sentence Transformer embeddings used throughout. Reranker disabled
by default (`USE_RERANKER=false`), enabled per-request via `use_reranker`.

**Measurement caveat, still applicable:** both the embedding model and the cross-encoder reranker load
lazily on first use in a fresh process, inflating the first query's reported latency (reranked run's first
query: ~1.0s vs. ~0.15–0.31s for subsequent queries in the same run). Figures above exclude first-query
cold-start cost. This is a one-time process-startup cost that would not recur in a long-lived server
process.

## 13. Trade-offs

- Heuristic/rule-based boundary detection is fast but meaningfully less accurate than the trained classifier
  (F1 0.533 vs. 0.784).
- Flat FAISS index is simple and sufficient at this scale but doesn't scale to millions of chunks without
  an ANN index or managed vector database.
- **Chunking that merges labels with content trades chunk granularity for retrievability** — a merged chunk
  is less atomic and slightly larger, but is far more likely to actually contain the answer to a natural
  question about that field. This traded some chunk-size uniformity for a measurable accuracy gain (Top-1
  3/6 → 5/6).
- **Reranking is not applied unconditionally** — given the demonstrated regression on "most recent role"
  style queries across two benchmark rounds, defaulting reranking to off and exposing it as an explicit
  per-request flag (rather than always-on) was the safer choice; a production system might want reranking
  enabled only for query types empirically shown to benefit (numeric/entity lookups), which the two rounds
  of benchmarking here start to distinguish.
- Basic table extraction (structural pattern-matching on column-like spacing) vs. a dedicated
  table-structure model — kept simple since the target document types have limited tabular complexity in
  the sample data, though this is now schema-aware (`table` blocks with real `headers`/`rows`) rather than
  flattened to paragraph text.

## 14. Future Improvements

- Investigate why reranking specifically hurts "most recent role/title"-style queries across two benchmark
  rounds — likely a cross-encoder lexical-similarity bias toward any job-title-shaped text regardless of
  recency, which the model has no explicit signal for; consider a recency-aware re-ranking feature or
  excluding this query archetype from the reranking path.
- Expand the Stage 3 labeled query set beyond 6 queries for a more statistically stable Recall@k/MRR
  estimate, and to determine whether the reranking regression generalizes beyond this one query pattern.
- Extend the chunking fix's label-merging logic to a broader class of "heading + content" patterns beyond
  the current short-label heuristic, and validate it doesn't over-merge on documents with legitimately
  short but complete field values.
- Confidence calibration (Platt scaling) for the boundary classifier's probability outputs.
- Qdrant or pgvector adapter for distributed deployment beyond the local FAISS flat index.
- PaddleOCR as an alternative OCR backend for production scanned-document handling.

## 15. AI Usage Declaration

- **AI-assisted code generation and debugging**: Implementation scaffolded and refined with an AI coding
  assistant (Claude), used interactively throughout — including diagnosing two retrieval-wiring bugs
  (`Reranker.enabled` not respecting the per-request `use_reranker` override; `pipeline.query()` and the
  `/retrieve` API endpoint silently dropping `use_reranker`/`document_type` instead of forwarding them), and
  correctly diagnosing that an initial "Skills:" retrieval failure was a Stage 2 chunking problem rather
  than a Stage 3 ranking problem — which directed the fix to the right layer instead of over-tuning
  reranking to compensate for a missing chunk.
- **Human engineering decisions**: architecture (3-stage hybrid), feature selection, model choices,
  evaluation methodology, dataset role mapping, the decision to hand-label a real Stage 3 query set instead
  of a fabricated smoke test, the decision to resolve ground truth dynamically against chunker output rather
  than hardcoding chunk IDs (so evaluation survives chunking changes), the specific chunking fix (merge
  short labels with following content; detect tables/lists structurally), and the interpretation of the
  reranking regression as a genuine, reportable limitation rather than noise to average away.
- **Experiments performed**: Stage 1 dataset benchmark on 200 test streams, reproduced twice with matching
  results; Stage 2 structuring benchmark on the sample packet, before and after the chunking fix; Stage 3
  retrieval benchmark on 6 hand-labeled queries, run with and without reranking, both before and after the
  chunking fix — four total retrieval configurations compared.
- **Validation performed**: dataset schema inspection, benchmark JSON artifacts, manual verification of the
  `doc-split-v2` HuggingFace page content (it is a commercial model card, not a dataset — corrected from an
  earlier inaccurate claim that it was inaccessible), hand-verification of Stage 3 answer-text-based ground
  truth against real chunk content, and independent re-derivation of the reported MRR/Recall@1 figures from
  raw per-query results to confirm the evaluation code computed them correctly.