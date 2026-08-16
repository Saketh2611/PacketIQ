# Datasets

This document explains how the three dataset names relate and which ones this project uses.

## Summary

| Name | Used? | Role |
|------|-------|------|
| `nutrientdocs/openpss-mirror` | **Yes** | Train/dev (config `SHORT`, split `train`) |
| `nutrientdocs/doc-split-benchmark` | **Yes, test-only** | Test/eval (config `our200`, split `test`) |
| `nutrientdocs/doc-split-v2` | **No** | Not a dataset — a commercial model. Excluded per assignment restrictions. |

## openpss-mirror

- **What it is:** Community mirror of the OpenPSS page-stream segmentation benchmark.
- **Purpose:** Training and development experimentation.
- **Schema:** `stream_id`, `position`, `text`, `label`, `image`
- **Boundary label:** `label=1` means a new logical document starts on this page.
- **Size (SHORT config):** 40,715 train rows (204 streams), 11,462 test rows (108 streams)

## doc-split-benchmark

- **What it is:** Official evaluation slice behind the doc-split leaderboard.
- **Purpose:** Held-out test evaluation only. **Not used for training** in this project — the boundary
  classifier is trained exclusively on `openpss-mirror`, and `doc-split-benchmark` is only ever scored
  against, never fit on.
- **Schema:** `stream_id`, `position`, `page_text`, `boundary`, `image`, `source`
- **Boundary label:** `boundary=1` means a new logical document starts on this page.
- **Size (our200 config):** 894 test rows (200 streams)
- **Note:** this same slice is listed on the `doc-split-v2` model card as the dataset that commercial
  model was trained/evaluated on. Using it here as a held-out test set for *our own* classifier is not the
  same as using or fine-tuning `doc-split-v2` itself, and the assignment's restriction is on benchmark-specific
  *models/checkpoints*, not on evaluating against a public benchmark slice. We call this out explicitly
  rather than leaving it unaddressed, since it's a reasonable question to raise in review.

## doc-split-v2

The assignment names this as the development dataset, linking to
`https://huggingface.co/nutrientdocs/doc-split-v2`. That page is **not a dataset repository** — it is a
**commercial model card** for Nutrient's proprietary page-stream segmentation model ("doc-split-v2, flagship").
Key facts from the page itself:

- Pipeline tag is `image-text-to-text` (a model), not a dataset listing.
- License is `nutrient-commercial`; weights are explicitly **not downloadable** ("contact Nutrient sales" for
  on-prem access).
- Its own model card lists `nutrientdocs/doc-split-benchmark` as the dataset it was trained/evaluated on —
  i.e. `doc-split-v2` is a consumer *of* a dataset we do use, not itself loadable via `datasets.load_dataset()`.

There is no train/validation split to load from this URL because it isn't a HuggingFace `datasets` repo at
all. This also means using it would directly conflict with the assignment's restriction against "models or
checkpoints specifically trained for the DocSplit or DocSplit v2 benchmark" — so excluding it is not just
a practical necessity but the correct call under the stated rules.

In place of it, we use `openpss-mirror` (train/dev) and `doc-split-benchmark` (test/eval), which are real,
loadable page-stream segmentation datasets from the same publisher and share a compatible schema (see
below).

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