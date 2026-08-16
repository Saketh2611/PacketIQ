# Benchmark Report

> **Canonical run.** Stage 1 measured **2026-08-16** via `python scripts/run_benchmarks.py --stage stage1`.
> Stage 2 measured the same day via `python scripts/run_stage2_benchmark.py`.
> Stage 3 measured the same day via `python scripts/run_stage3_benchmark.py --no-reranker` and `--use-reranker`.
> Embeddings: `sentence-transformers/all-MiniLM-L6-v2` with `USE_HASH_EMBEDDINGS=false` (real transformer
> embeddings, not the hash fallback). Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
> Leakage fix applied: the Stage 1 classifier's `StandardScaler` is fit only on the training split, after
> the train/validation split, not before.

## Dataset Roles

| Dataset | Config | Split | Role | Rows | Streams |
|---------|--------|-------|------|------|---------|
| `nutrientdocs/openpss-mirror` | SHORT | train | Train/dev (OpenPSS community mirror) | 40,715 | 204 |
| `nutrientdocs/doc-split-benchmark` | our200 | test | Official evaluation slice (test-only, never trained on) | 894 | 200 |
| `nutrientdocs/doc-split-v2` | — | — | Not a dataset — a commercial model card. Excluded per assignment restrictions. | — | — |

**Relationship:** `openpss-mirror` is a public redistribution of the OpenPSS page-stream segmentation
benchmark for training/experimentation. `doc-split-benchmark` is the official evaluation slice behind the
leaderboard — used here only as a held-out test set, never for fitting the classifier. They share the same
page-stream schema (`stream_id`, `position`, boundary label, text, image) but use different field names
(`label`/`text` vs `boundary`/`page_text`).

`doc-split-v2` (`https://huggingface.co/nutrientdocs/doc-split-v2`) is **not a HuggingFace dataset** — it's
a commercial, non-downloadable model (Nutrient's proprietary page-stream segmenter, license
`nutrient-commercial`). Its own model card lists `doc-split-benchmark` as its training/eval data, and there
is nothing to load from that URL via `datasets.load_dataset()`. Using it would also directly conflict with
the assignment's restriction against "models or checkpoints specifically trained for the DocSplit or
DocSplit v2 benchmark" — so excluding it satisfies the rules, not just a practical necessity. Full
explanation: [docs/datasets.md](datasets.md).

## Stage 1 — Boundary Detection (doc-split-benchmark test, 200 streams, 694 page pairs)

Metrics are macro-averaged per stream on the test set. The classifier was trained on the full
`openpss-mirror` SHORT train split (32,408 train / 8,103 val pairs) and validated with the scaler fit on
training features only.

| Method | Boundary Precision | Boundary Recall | Boundary F1 | Page Grouping Accuracy |
|--------|-------------------|-----------------|-------------|------------------------|
| baseline_rule | 0.532 | 0.583 | 0.533 | 0.573 |
| baseline_embedding | 0.535 | 0.536 | 0.517 | 0.582 |
| learned_classifier | 0.753 | 0.890 | **0.784** | **0.722** |

| Metric | Value |
|--------|-------|
| Classification Accuracy | N/A (datasets provide boundary labels only) |
| Classifier train pairs | 32,408 train / 8,103 val |
| Classifier training time (s) | 2,073.6 |
| Total Stage 1 wall clock (s) | 2,128.8 |
| Peak memory (MB) | 6,248.0 |
| Model size (MB) | 0.0015 |

### Top Feature Importances

| Feature | Weight |
|---------|--------|
| token_jaccard | 0.686 |
| type_agreement | 0.309 |
| text_length_ratio | 0.112 |
| structural_similarity | 0.112 |
| entity_overlap | -0.107 |
| semantic_cosine | -0.090 |

**Reproducibility:** re-run on 2026-08-16 independently of the run this table was originally built from;
F1 and page-grouping accuracy matched to three decimal places (0.784 / 0.722 both times), confirming the
result is stable given the fixed `random_seed`.

## Stage 2 — Document Structuring (sample packet, 3 documents)

Run via `python scripts/run_stage2_benchmark.py`, which calls the previously-unused `evaluate_stage2()`
function against the real pipeline output on the bundled sample packet (invoice + resume + passport, 6
pages).

| Metric | Value |
|--------|-------|
| Documents Processed | 3 |
| Avg Sections per Document | 1.0 |
| Avg Content Blocks per Document | 11.0 |
| Provenance Correctness | 3 / 3 |
| Processing Time (s) | 0.015 |

*The page-stream datasets do not provide Stage 2 structured-output ground truth (they're page streams with
boundary labels only, not annotated document structure), so this is measured against the bundled sample
packet rather than a labeled dataset — same scope as Stage 1's page-grouping evaluation would need if a
structuring-labeled dataset existed.*

## Stage 3 — Evidence Retrieval (sample packet, 6 hand-labeled queries)

Run via `python scripts/run_stage3_benchmark.py`, which replaces the earlier fabricated 2-query smoke test
with 6 real queries against the real pipeline output — two per sample document type (invoice, resume,
passport). Ground truth (`relevant_ids`) was hand-labeled by inspecting the actual extracted chunk text in
`outputs/samples/structured/*.json`.

| Metric | Vector Only | Vector + Reranker |
|--------|-------------|-------------------|
| Top-1 Accuracy | 3 / 6 | **4 / 6** |
| Recall@1 | 0.500 | **0.667** |
| Recall@3 | 0.667 | 0.667 |
| Recall@5 | 0.833 | 0.833 |
| Precision@1 | 0.500 | **0.667** |
| Precision@5 | 0.167 | 0.167 |
| MRR | 0.589 | **0.708** |
| nDCG | 0.648 | **0.738** |
| Avg latency (post-warmup, s) | ~0.022 | ~0.18 |
| Indexing time (s) | 13.05 | 13.52 |

*Latency note: the reported `avg_latency_seconds` for the reranked run (1.75s) is skewed by one-time
cross-encoder model loading on the first query (9.62s) — every subsequent query in the same run was
0.13–0.26s. The "post-warmup" figures above use queries 2–6 only, which is the honest per-query cost once
the model is resident in memory. The same caveat previously applied to embedding-model load time inside
`latency_seconds`; both are one-time process-startup costs, not steady-state query cost.*

### Per-query results

| Query | Doc type | No reranker | With reranker |
|---|---|---|---|
| "What is the total amount on the invoice?" | invoice | ❌ MISS (`chunk_001`, "INVOICE") | ✅ OK (`chunk_013`, correct) |
| "What is the invoice number?" | invoice | ✅ OK | ✅ OK |
| "What skills does the candidate have?" | resume | ❌ MISS (`chunk_010`, "Skills:") | ❌ MISS (`chunk_010`, "Skills:") |
| "What is the candidate's most recent job title?" | resume | ❌ MISS (`chunk_007`) | ❌ MISS (`chunk_003`, "Software Engineer") |
| "What is the passport holder's date of birth?" | passport | ✅ OK | ✅ OK |
| "What is the passport number?" | passport | ✅ OK | ✅ OK |

### Failure analysis

- **Fixed by reranking:** "total amount on the invoice" — vector search ranked the bare heading `"INVOICE"`
  above the actual total `"Total Amount: ₹52,340"`, because `all-MiniLM-L6-v2`'s bi-encoder embeds short,
  low-information chunks close to many queries in cosine space when the corpus is small. The cross-encoder
  reranker scores query+chunk pairs jointly instead of comparing independent embeddings, and correctly
  promoted the true answer to rank 1.
- **Not fixed by reranking — resume "Skills:" query:** both modes return `chunk_010` (the bare heading
  `"Skills:"`) instead of `chunk_011` (`"Python, Machine Learning, FastAPI"`, the actual list). Root cause is
  structural, not a ranking failure: the heading and its content ended up as separate chunks with no strong
  semantic link between them, so no amount of re-ranking the same candidate set recovers the missing chunk if
  it isn't a strong enough match to begin with. A structural fix — merging short headings with their
  immediately following content block, or chunking by section instead of one-line-per-block for
  resume-style content — would likely help; reranking alone cannot fix a bad chunk boundary.
- **Reranker makes one case worse:** for "most recent job title," reranking replaces one wrong answer
  (`chunk_007`) with a different wrong answer (`chunk_003`, "Software Engineer" — a generic title on an
  unrelated line) that the cross-encoder judged a closer lexical match to "job title." This shows reranking
  is not a strict per-query improvement even though it improves the aggregate metrics above — it's a
  re-ranking of already-retrieved candidates, so it can't recover a missing correct chunk, and it can
  occasionally promote a more superficially relevant wrong one.

## Comparison Summary

- **Stage 1:** `learned_classifier` is the strongest boundary-detection method, F1 0.784, reproducible across
  independent runs with the fixed seed.
- **Stage 2:** structuring completes correctly on all 3 sample documents with full page-range provenance;
  no structured-output ground-truth dataset exists to benchmark against at scale.
- **Stage 3:** reranking improved Top-1 accuracy from 3/6 to 4/6 and MRR from 0.589 to 0.708, but did not
  fix two resume-related failures rooted in chunk boundaries rather than ranking — see failure analysis
  above. This is a more defensible result than a single "reranking helps" claim, since it also documents
  where reranking specifically does not help.

Raw results: `outputs/benchmarks/stage1_dataset.json`, `outputs/benchmarks/stage2.json`,
`outputs/benchmarks/retrieval_real.json`

Dataset relationship report: `data/processed/dataset_relationship_report.json`