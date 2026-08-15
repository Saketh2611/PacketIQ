# Benchmark Report

> Measured on **2026-08-16** using `.venv` with `python scripts/run_benchmarks.py --stage all`.
> Embeddings: `sentence-transformers/all-MiniLM-L6-v2` with `USE_HASH_EMBEDDINGS=false` and local cached model files.
> Leakage fix: the learned classifier now performs the train/validation split before fitting `StandardScaler`; the scaler is fit only on the training split.

## Dataset Roles

| Dataset | Config | Split | Role | Rows | Streams |
|---------|--------|-------|------|------|---------|
| `nutrientdocs/openpss-mirror` | SHORT | train | Train/dev (OpenPSS community mirror) | 40,715 | 204 |
| `nutrientdocs/doc-split-benchmark` | our200 | test | Official evaluation slice | 894 | 200 |
| `nutrientdocs/doc-split-v2` | - | - | Referenced in assignment; not accessible on HuggingFace Hub | - | - |

**Relationship:** `openpss-mirror` is a public redistribution of the OpenPSS page-stream segmentation benchmark for training/experimentation. `doc-split-benchmark` is the official evaluation slice behind the leaderboard. They share the same page-stream schema (`stream_id`, `position`, boundary label, text, image) but use different field names (`label`/`text` vs `boundary`/`page_text`). They are not synonymous with `doc-split-v2`, which could not be loaded from the Hub.

## Stage 1 (doc-split-benchmark test, 200 streams, 694 page pairs)

Metrics are macro-averaged per stream on the test set. The classifier was trained on the full `openpss-mirror` SHORT train split and validated with the scaler fit on training features only.

| Method | Boundary Precision | Boundary Recall | Boundary F1 | Page Grouping Accuracy |
|--------|-------------------|-----------------|-------------|------------------------|
| baseline_rule | 0.532 | 0.583 | 0.533 | 0.573 |
| baseline_embedding | 0.535 | 0.536 | 0.517 | 0.582 |
| learned_classifier | 0.753 | 0.890 | 0.784 | 0.722 |

| Metric | Value |
|--------|-------|
| Classification Accuracy | N/A (datasets provide boundary labels only) |
| Classifier train pairs | 32,408 train / 8,103 val |
| Eval latency - baseline_rule (s) | 42.98 |
| Eval latency - baseline_embedding (s) | 23.56 |
| Eval latency - learned_classifier (s) | 65.62 |
| Classifier training time (s) | 2164.37 |
| Total Stage 1 wall clock (s) | 2296.55 |
| Peak memory (MB) | 482.3 |
| Model size (MB) | 0.001 |

### Top Feature Importances

| Feature | Weight |
|---------|--------|
| token_jaccard | 0.686 |
| type_agreement | 0.307 |
| text_length_ratio | 0.112 |
| structural_similarity | 0.112 |
| entity_overlap | -0.107 |
| semantic_cosine | -0.091 |

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
| Recall@3 | 1.000 | - |
| Recall@5 | 1.000 | - |
| Precision@1 | 0.500 | - |
| Precision@5 | 0.200 | - |
| MRR | 0.750 | - |
| nDCG | 0.815 | - |

*Retrieval metrics are from a 2-query smoke test. The page-stream datasets do not include labeled query/evidence pairs.*

## Comparison Summary

- **learned_classifier** is the strongest Stage 1 method after the leakage fix, with boundary F1 0.784.
- The rule and embedding baselines dropped with real transformer embeddings because their fixed thresholds were tuned against the prior hash-embedding behavior.
- Runtime is dominated by full CPU embedding computation over the 40,715-row training split; dataset loading is secondary.

Raw results: `outputs/benchmarks/stage1_dataset.json`, `outputs/benchmarks/retrieval.json`

Dataset relationship report: `data/processed/dataset_relationship_report.json`
