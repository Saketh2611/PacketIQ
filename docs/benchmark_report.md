# Benchmark Report

> **Canonical run.** Stage 1 measured **2026-08-16** via `python scripts/run_benchmarks.py --stage stage1`
> (reproduced twice independently, matching to 3 decimal places both times).
> Stage 2/3 measured the same day via `run_stage2_benchmark.py` / `run_stage3_benchmark.py --no-reranker` /
> `--use-reranker`, **after** a chunking fix (see "Chunking fix" below) that merges short field labels
> (e.g. `"Skills:"`) with their following content block, and adds table/figure detection to Stage 2
> structuring. All Stage 2/3 numbers below reflect the fixed chunker, not the earlier run.
> Embeddings: `sentence-transformers/all-MiniLM-L6-v2` with `USE_HASH_EMBEDDINGS=false` (real transformer
> embeddings). Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
> Leakage fix (Stage 1): the classifier's `StandardScaler` is fit only on the training split, after the
> train/validation split, not before.

## Dataset Roles

| Dataset | Config | Split | Role | Rows | Streams |
|---------|--------|-------|------|------|---------|
| `nutrientdocs/openpss-mirror` | SHORT | train | Train/dev (OpenPSS community mirror) | 40,715 | 204 |
| `nutrientdocs/doc-split-benchmark` | our200 | test | Official evaluation slice (test-only, never trained on) | 894 | 200 |
| `nutrientdocs/doc-split-v2` | — | — | Not a dataset — a commercial model card. Excluded per assignment restrictions. | — | — |

**Relationship:** `openpss-mirror` is a public redistribution of the OpenPSS page-stream segmentation
benchmark for training/experimentation. `doc-split-benchmark` is the official evaluation slice behind the
leaderboard — used here only as a held-out test set, never for fitting the classifier. They share the same
page-stream schema (`stream_id`, `position`, boundary label, text, image) but use different field names.

`doc-split-v2` (`https://huggingface.co/nutrientdocs/doc-split-v2`) is **not a HuggingFace dataset** — it's
a commercial, non-downloadable model (Nutrient's proprietary page-stream segmenter, license
`nutrient-commercial`). Its own model card lists `doc-split-benchmark` as its training/eval data, and there
is nothing to load from that URL via `datasets.load_dataset()`. Using it would also directly conflict with
the assignment's restriction against models "specifically trained for the DocSplit or DocSplit v2
benchmark." Full explanation: [docs/datasets.md](datasets.md).

## Stage 1 — Boundary Detection (doc-split-benchmark test, 200 streams, 694 page pairs)

Metrics are macro-averaged per stream on the test set. The classifier was trained on the full
`openpss-mirror` SHORT train split (32,408 train / 8,103 val pairs) with `class_weight="balanced"` (page
streams are naturally imbalanced toward same-document pairs, so this reweights the minority boundary class
during training) and validated with the scaler fit on training features only.

| Method | Boundary Precision | Boundary Recall | Boundary F1 | Page Grouping Accuracy |
|--------|-------------------|-----------------|-------------|------------------------|
| baseline_rule | 0.532 | 0.583 | 0.533 | 0.573 |
| baseline_embedding | 0.535 | 0.536 | 0.517 | 0.582 |
| learned_classifier | 0.753 | 0.890 | **0.784** | **0.722** |

| Metric | Value |
|--------|-------|
| Classifier train pairs | 32,408 train / 8,103 val |
| Classifier training time (s) | ~2,074–2,123 (two runs) |
| Total Stage 1 wall clock (s) | ~2,129–2,179 (two runs) |
| Model size (MB) | 0.0015 |

**Reproducibility:** re-run independently twice on 2026-08-16 with matching F1 (0.7838, 0.7838) and
page-grouping accuracy (0.7223, 0.7223) to four decimal places, confirming the result is stable.

### Top Feature Importances

| Feature | Weight |
|---------|--------|
| token_jaccard | 0.683–0.686 |
| type_agreement | 0.307–0.309 |
| text_length_ratio | 0.112 |
| structural_similarity | 0.112 |
| entity_overlap | -0.107 |
| semantic_cosine | -0.090 |

## Stage 2 — Document Structuring (sample packet, 3 documents)

Run via `python scripts/run_stage2_benchmark.py`. Post-chunking-fix, Stage 2 now also detects and reports
tables and figures as distinct block types instead of flattening everything to paragraphs.

| Metric | Before chunking fix | After chunking fix |
|--------|---------------------|---------------------|
| Documents Processed | 3 | 3 |
| Avg Blocks per Document | 11.0 | 10.0 |
| Table Blocks | *(not tracked)* | 1 |
| Figure Blocks | *(not tracked)* | 0 |
| Provenance Correctness | 3 / 3 | 3 / 3 |
| Processing Time (s) | 0.015 | 0.018 |

*The page-stream datasets do not provide Stage 2 structured-output ground truth, so this is measured
against the bundled sample packet rather than a labeled dataset.*

## Stage 3 — Evidence Retrieval (sample packet, 6 hand-labeled queries)

Run via `run_stage3_benchmark.py --no-reranker` / `--use-reranker`. Ground truth (`relevant_ids`) is now
resolved dynamically by matching answer text patterns against the current chunker's output, so it stays
correct even as chunk boundaries change — rather than hardcoded chunk IDs that would silently go stale.

### Before vs. after the chunking fix

| Metric | Before fix (vector only) | After fix (vector only) | After fix (+ reranker) |
|--------|---------------------------|--------------------------|--------------------------|
| Top-1 Accuracy | 3 / 6 | **5 / 6** | 5 / 6 |
| Recall@1 | 0.500 | **0.833** | 0.833 |
| Recall@5 | 0.833 | **1.000** | 0.833 |
| MRR | 0.589 | **0.867** | 0.833 |
| nDCG | 0.648 | **0.898** | 0.833 |
| Indexing time (s) | 13.05 | 14.65 | 13.54 |

**The chunking fix — not the reranker — is what fixed the "Skills:" failure.** Merging short field labels
with their following content block at chunk-build time means the query "What skills does the candidate
have?" now matches a chunk that actually contains `"Python, Machine Learning, FastAPI"` instead of only the
bare heading `"Skills:"`. This confirms the diagnosis from the previous benchmark round: that failure was a
chunk-boundary problem, and no amount of reranking a bad candidate set could have fixed it — only fixing the
chunking itself could, and it did.

**One remaining failure, present in both modes:** "What is the candidate's most recent job title?" still
misses in both vector-only and reranked runs — vector-only ranks `chunk_004` ("Software Engineer") above the
correct `chunk_004`/`Senior Developer` chunk at position 1 incorrectly; **with reranking, the correct chunk
does not appear in the top 5 at all**, a regression from the vector-only run where it was at least present.
See failure analysis below.

### Per-query results (vector only vs. reranker)

| Query | Doc type | Vector only | + Reranker |
|---|---|---|---|
| "What is the total amount on the invoice?" | invoice | ❌ MISS (rank 5) | ✅ OK |
| "What is the invoice number?" | invoice | ✅ OK | ✅ OK |
| "What skills does the candidate have?" | resume | ✅ OK | ✅ OK |
| "What is the candidate's most recent job title?" | resume | ✅ OK | ❌ MISS (not in top 5) |
| "What is the passport holder's date of birth?" | passport | ✅ OK | ✅ OK |
| "What is the passport number?" | passport | ✅ OK | ✅ OK |

### Failure analysis

- **Fixed by the chunking change:** "Skills:" query — see above. This was previously documented as a
  reranking-cannot-fix-this case; it's now resolved at the source.
- **Reranking regression on "most recent job title":** vector-only search correctly ranks the true answer
  (`"Senior Developer at TechCo"`) at position 1. With reranking enabled, that chunk drops out of the top 5
  entirely, replaced by generic-title chunks the cross-encoder scored as closer lexical matches to "job
  title" (e.g. "Software Engineer"). This is a clearer, more concerning version of the same pattern seen in
  the previous benchmark round: the reranker can demote a correct, already-well-ranked result in favor of a
  more superficially relevant one. Given this and the previous round's similar finding, reranking should be
  treated as a targeted improvement for specific query types (e.g. numeric/entity lookups like "total
  amount," "invoice number," "passport number" — all pass in both modes) rather than a universal win, and
  this query type ("most recent role/title") is a case where it currently hurts more than it helps.
- **Indexing time increased slightly** (13.05s → 14.65s) post-fix, consistent with slightly more complex
  per-block metadata (bounding boxes, merged-label tracking) being computed during structuring/chunking.

### Operational note

The sample packet used for this Stage 2/3 run was generated by an earlier successful
`generate_sample_outputs.py` run; a later regeneration attempt in the same session failed with a file
permission error (the output PDF was likely still open in another process/viewer). The benchmark scripts
ran successfully against the existing on-disk sample packet regardless, so the numbers above are valid, but
worth closing any PDF viewer before regenerating samples to avoid the same error.

## Comparison Summary

- **Stage 1:** `learned_classifier` remains the strongest boundary-detection method, F1 0.784, independently
  reproduced twice with matching results.
- **Stage 2:** structuring now correctly detects tables and figures as distinct block types (1 table
  detected in the sample invoice) in addition to full provenance correctness (3/3).
- **Stage 3:** the chunking fix (merging short labels with following content) resolved the previously
  documented "Skills:" failure and improved Top-1 accuracy from 3/6 to 5/6 on vector search alone — a larger
  and more durable gain than reranking provided in isolation. Reranking on top of the fixed chunker holds
  Top-1 at 5/6 but trades one correct answer for a different one, net neutral on this query set and actively
  worse on the "job title" query specifically. This suggests prioritizing chunking-quality fixes over
  reranking tuning going forward, and treating reranking as query-type-dependent rather than a blanket
  improvement.

Raw results: `outputs/benchmarks/stage1_dataset.json`, `outputs/benchmarks/stage2.json`,
`outputs/benchmarks/stage3_vector.json`, `outputs/benchmarks/stage3_reranker.json`

Dataset relationship report: `data/processed/dataset_relationship_report.json`