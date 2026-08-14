# Benchmark Report

> Results from `python scripts/run_benchmarks.py` (synthetic evaluation data, hash embedding fallback).
> Re-run with Sentence Transformers, OpenPSS mirror, and doc-split-benchmark for production metrics.

## Stage 1

| Method | Boundary Precision | Boundary Recall | Boundary F1 | Page Grouping Accuracy | Classification Accuracy |
|--------|-------------------|-----------------|-------------|------------------------|------------------------|
| baseline_rule | 0.667 | 1.000 | 0.800 | 0.333 | 1.000 |
| baseline_embedding | 0.400 | 1.000 | 0.571 | 0.167 | 1.000 |
| learned_classifier | [RUN AFTER TRAINING ON DOCSPLIT] | — | — | — | — |

| Metric | Value |
|--------|-------|
| Latency (s) | 0.033 |
| Memory (MB) | 179.3 |
| Model Size (MB) | N/A (hash fallback) |

## Stage 2

| Metric | Value |
|--------|-------|
| Documents Processed | 3 (sample pipeline) |
| Provenance Correctness | 3/3 |
| Processing Time (s) | < 0.1 |

*Note: Dataset-backed Stage 1 metrics should be run on `nutrientdocs/openpss-mirror` for training/development and `nutrientdocs/doc-split-benchmark` for test/evaluation. Stage 2 metrics measure extraction completeness and provenance correctness on the sample PDF only.*

## Stage 3

| Metric | Vector Only | Vector + Reranker |
|--------|-------------|-------------------|
| Recall@1 | 0.500 | [RUN WITH USE_RERANKER=true] |
| Recall@3 | 1.000 | — |
| Recall@5 | 1.000 | — |
| Precision@1 | 0.500 | — |
| Precision@5 | 0.200 | — |
| MRR | 0.750 | — |
| nDCG | 0.815 | — |
| Avg Retrieval Latency (s) | 0.040 | — |
| Indexing Time (s) | 1.200 | — |

## Comparison Summary

- **Rule baseline** outperforms **embedding-only baseline** on synthetic data (F1: 0.80 vs 0.57)
- **Vector-only retrieval** tested on 2 synthetic queries; reranker comparison pending
- End-to-end sample pipeline indexed **33 chunks** from 6-page PDF (3 documents)

Raw results: `outputs/benchmarks/*.json`
