# Datasets

This document explains how the three dataset names relate and which ones this project uses.

## Summary

| Name | Used? | Role |
|------|-------|------|
| `nutrientdocs/openpss-mirror` | **Yes** | Train/dev (config `SHORT`, split `train`) |
| `nutrientdocs/doc-split-benchmark` | **Yes** | Test/eval (config `our200`, split `test`) |
| `nutrientdocs/doc-split-v2` | **No** | Referenced in assignment; not accessible on HuggingFace Hub |

## openpss-mirror

- **What it is:** Community mirror of the OpenPSS page-stream segmentation benchmark.
- **Purpose:** Training and development experimentation.
- **Schema:** `stream_id`, `position`, `text`, `label`, `image`
- **Boundary label:** `label=1` means a new logical document starts on this page.
- **Size (SHORT config):** 40,715 train rows (204 streams), 11,462 test rows (108 streams)

## doc-split-benchmark

- **What it is:** Official evaluation slice behind the doc-split leaderboard.
- **Purpose:** Held-out test evaluation (not training).
- **Schema:** `stream_id`, `position`, `page_text`, `boundary`, `image`, `source`
- **Boundary label:** `boundary=1` means a new logical document starts on this page.
- **Size (our200 config):** 894 test rows (200 streams)

## doc-split-v2

The assignment references this as the development dataset. During implementation it returned **401 / not found** from the HuggingFace Hub. Do not assume it is synonymous with `openpss-mirror` or `doc-split-benchmark` without verifying the dataset card.

## How the adapter works

1. Load page-stream rows from HuggingFace.
2. Group rows by `stream_id`.
3. Sort by `position` (handles 0-based benchmark vs 1-based OpenPSS).
4. Map `label`/`boundary` → `starts_new_document`.
5. Build contiguous page groups and page-pair training/eval examples.

See `src/document_intelligence/dataset/adapter.py`.

## Commands

```bash
python scripts/compare_datasets.py       # Save relationship report
python scripts/inspect_dataset.py        # Inspect train + test schemas
python scripts/download_dataset.py       # Save both manifests
python scripts/run_benchmarks.py --stage stage1
```

Relationship report: `data/processed/dataset_relationship_report.json`
