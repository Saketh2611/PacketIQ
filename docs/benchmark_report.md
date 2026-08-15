# Benchmark Report

> Measured on **2026-08-15** using `python scripts/run_benchmarks.py --stage stage1`.
> Embeddings: hash fallback (`USE_HASH_EMBEDDINGS=1`) due to local Sentence Transformers/torch compatibility.

## Dataset Roles

| Dataset | Config | Split | Role | Rows | Streams |
|---------|--------|-------|------|------|---------|
| `nutrientdocs/openpss-mirror` | SHORT | train | Train/dev (OpenPSS community mirror) | 40,715 | 204 |
| `nutrientdocs/doc-split-benchmark` | our200 | test | Official evaluation slice | 894 | 200 |
| `nutrientdocs/doc-split-v2` | — | — | Referenced in assignment; **not accessible** on HuggingFace Hub | — | — |

**Relationship:** `openpss-mirror` is a public redistribution of the OpenPSS page-stream segmentation benchmark for training/experimentation. `doc-split-benchmark` is the official evaluation slice behind the leaderboard. They share the same page-stream schema (`stream_id`, `position`, boundary label, text, image) but use different field names (`label`/`text` vs `boundary`/`page_text`). They are **not** synonymous with `doc-split-v2`, which could not be loaded from the Hub.

## Stage 1 (doc-split-benchmark test, 200 streams, 694 page pairs)

Metrics are **macro-averaged per stream** on the test set. Classifier trained on full `openpss-mirror` SHORT train split (40,508 page pairs).

| Method | Boundary Precision | Boundary Recall | Boundary F1 | Page Grouping Accuracy |
|--------|-------------------|-----------------|-------------|------------------------|
| baseline_rule | 0.761 | 0.921 | 0.799 | 0.715 |
| baseline_embedding | 0.729 | 0.882 | 0.768 | 0.715 |
| learned_classifier | 0.743 | 0.877 | 0.774 | 0.723 |

| Metric | Value |
|--------|-------|
| Classification Accuracy | N/A (datasets provide boundary labels only) |
| Classifier train pairs | 32,408 train / 8,103 val |
| Eval latency — baseline_rule (s) | 6.22 |
| Eval latency — baseline_embedding (s) | 5.64 |
| Eval latency — learned_classifier (s) | 6.67 |
| Classifier training time (s) | 342.72 |
| Peak memory (MB) | 7987.3 |
| Model size (MB) | 0.001 |

### Top feature importances (learned classifier)

| Feature | Weight |
|---------|--------|
| semantic_cosine | 0.428 |
| token_jaccard | 0.354 |
| type_agreement | 0.287 |
| entity_overlap | −0.084 |

## Stage 2

| Metric | Value |
|--------|-------|
| Documents Processed | 3 (sample PDF pipeline only) |
| Provenance Correctness | 3/3 |
| Processing Time (s) | < 0.1 |

*The page-stream datasets do not provide Stage 2 structured-output ground truth. Only sample-PDF provenance metrics are reported.*

## Stage 3

| Metric | Vector Only | Vector + Reranker |
|--------|-------------|-------------------|
| Recall@1 | 0.500 (smoke test) | [RUN WITH USE_RERANKER=true] |
| Recall@3 | 1.000 | — |
| Recall@5 | 1.000 | — |
| Precision@1 | 0.500 | — |
| Precision@5 | 0.200 | — |
| MRR | 0.750 | — |
| nDCG | 0.815 | — |

*Retrieval metrics are from a 2-query smoke test. The page-stream datasets do not include labeled query/evidence pairs.*

## Comparison Summary

- **baseline_rule** achieves the highest boundary F1 (0.799) on this run with hash embeddings
- **learned_classifier** slightly improves page grouping accuracy (0.723 vs 0.715)
- Re-run with Sentence Transformers (`USE_HASH_EMBEDDINGS=false`) for production-quality embedding signals

Raw results: `outputs/benchmarks/stage1_dataset.json`, `outputs/benchmarks/retrieval.json`

Dataset relationship report: `data/processed/dataset_relationship_report.json`
