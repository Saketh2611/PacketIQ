# Document Packet Intelligence & Evidence Retrieval (PacketIQ)

A hybrid, measurable, resource-aware system for analyzing **PDF packets** that contain multiple independent logical documents. The pipeline detects page boundaries, groups pages into documents, classifies document types, converts each document into structured JSON, and retrieves **evidence** (not generated answers) for user queries.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [What This System Does](#what-this-system-does)
3. [Architecture Overview](#architecture-overview)
4. [Three-Stage Pipeline](#three-stage-pipeline)
5. [Datasets and Evaluation](#datasets-and-evaluation)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Running the Pipeline](#running-the-pipeline)
9. [API](#api)
10. [Benchmarks and Reports](#benchmarks-and-reports)
11. [Project Structure](#project-structure)
12. [Testing](#testing)
13. [Docker](#docker)
14. [Limitations and Future Work](#limitations-and-future-work)

---

## Problem Statement

Real-world document packets—loan applications, compliance bundles, onboarding packages—often merge several unrelated documents into one PDF:

```
packet.pdf
  Pages 1–3  → Invoice
  Pages 4–5  → Resume
  Pages 6–9  → Passport
```

There are no explicit separators. A retrieval system that treats the whole PDF as one document will return wrong evidence (e.g., mixing invoice totals with resume skills). This project solves that by:

1. **Detecting document boundaries** between pages
2. **Grouping** contiguous pages into logical documents
3. **Classifying** each document type with confidence scores
4. **Structuring** each document as JSON with page provenance
5. **Retrieving** ranked evidence chunks for a query—never fabricating answers

---

## What This System Does

| Stage | Input | Output |
|-------|-------|--------|
| **Ingestion** | PDF packet | Per-page text, layout blocks, OCR fallback |
| **Stage 1** | Page representations | Document groups, types, boundary scores |
| **Stage 2** | Grouped pages | Structured JSON + retrieval-ready chunks |
| **Stage 3** | Chunks + query | Ranked evidence with document ID, page, score |

This is **not a chatbot**. The `/retrieve` endpoint returns supporting text spans with provenance—it does not generate conversational answers.

---

## Architecture Overview

```
                    PDF Packet
                        │
                        ▼
              ┌─────────────────┐
              │   Ingestion     │  PyMuPDF native text + OCR fallback
              └────────┬────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                             │
   Stage 1: Boundaries                 │
   • Page-pair features                │
   • Rule / embedding baselines        │
   • Learned classifier (LR/XGB)     │
   • Page grouping + doc classification│
         │                             │
         ▼                             │
   Stage 2: Structure                 │
   • Sections, headings, paragraphs   │
   • Metadata + content hashes         │
   • Semantic/structural chunking      │
         │                             │
         ▼                             │
   Stage 3: Retrieval                  │
   • Sentence Transformers embeddings   │
   • FAISS vector index                │
   • Optional cross-encoder reranker   │
         │                             │
         ▼                             │
   Evidence results                   │
   (doc_id, page, chunk, score)       │
```

Detailed diagrams: [docs/architecture.md](docs/architecture.md)

### Design Principles

- **No LLM for boundary detection** — engineered features + small classifiers
- **Native PDF extraction before OCR** — resource-efficient
- **Configurable models** — embeddings, reranker, OCR, vector store via env/config
- **Provenance everywhere** — every chunk retains document ID and page references
- **Reproducible benchmarks** — real dataset evaluation, not synthetic placeholders
- **Stage-aware debugging** — downstream retrieval failures are traced back to earlier structuring/chunking problems when applicable

---

## Three-Stage Pipeline

### Stage 1: Page-Pair Boundary Detection

For each adjacent page pair `(i, i+1)`, the system computes **9 engineered features**:

| Feature | Description |
|---------|-------------|
| `semantic_cosine` | Embedding similarity between page texts |
| `token_jaccard` | Normalized token overlap |
| `text_length_ratio` | Length ratio between pages |
| `layout_similarity` | Block count and dimension similarity |
| `block_position_similarity` | Spatial layout profile similarity |
| `heading_style_similarity` | Heading count similarity |
| `entity_overlap` | Named entity / number overlap |
| `structural_similarity` | Combined structural signals |
| `type_agreement` | Heuristic document-type agreement |

Three methods are compared:

| Method | Description |
|--------|-------------|
| `baseline` | Weighted rule/threshold on all features |
| `embedding` | Cosine similarity threshold only |
| `learned` | Logistic Regression (or XGBoost) on feature vector |

> **`learned` requires a trained model first.** It loads `models/boundary_classifier.joblib`, which is **not** included in the repo (it's gitignored and only produced by training). If that file is missing, the pipeline logs a warning and **silently falls back to `baseline`** — you won't get an error, just baseline-quality results. Train it with:
> ```bash
> python scripts/run_benchmarks.py --stage stage1
> ```
> This downloads the configured HuggingFace datasets and trains on the full OpenPSS mirror train split (~40k rows), so it needs internet access and a few minutes. Use `--max-train-streams N` to train on a smaller subset for a quick smoke test.

Boundary decisions are converted to contiguous page groups. Each group is classified using heuristic classification; an embedding-prototype classifier is also implemented as an alternative but is not currently wired into the main pipeline because the available page-stream datasets do not provide document-type labels.

### Stage 2: Structured Document Representation

Each grouped document becomes JSON:

```json
{
  "document_id": "packet_doc_001",
  "document_type": "invoice",
  "confidence": 0.94,
  "source": { "page_start": 1, "page_end": 3 },
  "content": {
    "title": "INVOICE",
    "sections": [{ "heading": "...", "blocks": [...] }]
  }
}
```

Stage 2:

- groups native PDF blocks into sections using heading/font-size heuristics
- detects table-like column structures and preserves them as `table` blocks
- merges consecutive bullet/numbered lines into `list` blocks
- captures image blocks as `figure` blocks
- records source bounding boxes in block metadata
- preserves document/page provenance
- performs retrieval-aware chunking

A key chunking rule merges short field labels with their immediately following content. For example:

```text
Skills:
Python, Machine Learning, FastAPI
```

becomes one retrieval chunk rather than two disconnected chunks.

### Stage 3: Evidence Retrieval

```text
Structured JSON → chunks → embeddings → FAISS index

Query → embed → top-N search → optional rerank → top-k evidence
```

Example result:

```json
{
  "query": "What is the total amount on the invoice?",
  "results": [{
    "document_id": "sample_packet_doc_001",
    "document_type": "invoice",
    "page": 3,
    "chunk_id": "sample_packet_doc_001_chunk_013",
    "evidence": "Total Amount: ₹52,340",
    "score": 0.82
  }]
}
```

Retrieval uses normalized Sentence Transformer embeddings with a local FAISS index. When enabled, the cross-encoder reranks the top vector candidates. Empty or whitespace-only queries short-circuit without embedding/search.

---

## Datasets and Evaluation

### Dataset roles

| Dataset | Config | Split | Role | Rows | Streams |
|---------|--------|-------|------|------|---------|
| [`nutrientdocs/openpss-mirror`](https://huggingface.co/datasets/nutrientdocs/openpss-mirror) | `SHORT` | train | **Train/dev** — OpenPSS community mirror | 40,715 | 204 |
| [`nutrientdocs/doc-split-benchmark`](https://huggingface.co/datasets/nutrientdocs/doc-split-benchmark) | `our200` | test | **Official evaluation slice** — held-out test | 894 | 200 |
| `nutrientdocs/doc-split-v2` | — | — | **Commercial model card, not a dataset** | — | — |

### Dataset relationship

`openpss-mirror` is a public redistribution of the OpenPSS page-stream segmentation benchmark for training and experimentation. `doc-split-benchmark` is the official evaluation slice used here only for held-out testing.

The two usable datasets share the same page-stream concept and schema, but use different field names. `DocSplitAdapter` normalizes them.

`doc-split-v2` is **not a downloadable HuggingFace dataset**. The URL points to Nutrient's proprietary page-stream segmenter model card with a `nutrient-commercial` license. Its model card references `doc-split-benchmark`; there is no dataset to load with `datasets.load_dataset()`. It is excluded from the implementation both because it is not a dataset and because the assignment restricts use of models specifically trained for the DocSplit/DocSplit v2 benchmark.

### Schema mapping

| Field | openpss-mirror | doc-split-benchmark |
|-------|----------------|---------------------|
| Stream ID | `stream_id` | `stream_id` |
| Page order | `position` (1-based) | `position` (0-based) |
| Boundary label | `label` (1=new doc) | `boundary` (1=new doc) |
| Text | `text` | `page_text` |
| Image | `image` | `image` |

Rows are grouped by `stream_id`, sorted by `position`, and converted to page groups using boundary labels.

### Inspect and download datasets

```bash
# Inspect both configured datasets
python scripts/inspect_dataset.py

# Compare dataset cards and save relationship report
python scripts/compare_datasets.py

# Download both manifests
python scripts/download_dataset.py

# Download only the train/dev manifest
python scripts/download_dataset.py --train-only

# Download only the test/eval manifest
python scripts/download_dataset.py --test-only
```

`download_dataset.py` uses the dataset names/configs from `.env` / settings and writes:

```text
data/processed/train_manifest.json
data/processed/test_manifest.json
```

It does not take `--dataset`, `--config`, or `--output` flags; change the dataset variables in `.env` if you need different sources.

---

## Installation

**Requirements:** Python 3.11+, ~8 GB RAM for full benchmark training, optional Tesseract for OCR.

```bash
git clone <repo-url>
cd PacketIQ

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
pip install -e .
```

> `pip install -e .` installs the package in editable mode so `document_intelligence` is importable from anywhere.
>
> The `pyproject.toml` declares a `doc-intel` console-script entry point, but `document_intelligence.cli` is not currently implemented. Use the scripts in `scripts/` (or `python main.py`) instead.

Copy environment template:

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/macOS
```

---

## Configuration

Settings load from `.env`, `configs/config.yaml`, and `configs/models.yaml`.

| Variable | Description | Default |
|----------|-------------|---------|
| `DATASET_NAME` | Train/dev dataset | `nutrientdocs/openpss-mirror` |
| `DATASET_CONFIG` | Train config | `SHORT` |
| `TEST_DATASET_NAME` | Test/eval dataset | `nutrientdocs/doc-split-benchmark` |
| `TEST_DATASET_CONFIG` | Test config | `our200` |
| `EMBEDDING_MODEL` | Sentence Transformers model | `all-MiniLM-L6-v2` |
| `RERANKER_MODEL` | Cross-encoder reranker | `ms-marco-MiniLM-L-6-v2` |
| `VECTOR_STORE` | Vector backend | `faiss` |
| `OCR_ENGINE` | OCR engine | `pytesseract` |
| `USE_HASH_EMBEDDINGS` | Deterministic hash fallback | `false` |
| `USE_RERANKER` | Enable cross-encoder reranking | `false` |
| `BOUNDARY_THRESHOLD` | Same-document score threshold | `0.5` |
| `GEMINI_API_KEY` | Optional LLM fallback | (empty) |
| `HF_TOKEN` | HuggingFace token | (empty) |

Production benchmark results below were obtained with:

```text
USE_HASH_EMBEDDINGS=false
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
USE_RERANKER=false (default)
```

---

## Running the Pipeline

### Quick end-to-end demo

```bash
python scripts/generate_sample_outputs.py
```

This generates a synthetic 6-page packet containing invoice + resume + passport documents and runs all three stages. Outputs include:

```text
outputs/samples/sample_packet.pdf
outputs/samples/stage1_output.json
outputs/samples/structured/
outputs/samples/retrieval_output.json
outputs/samples/failure_case_empty_query.json
```

If the PDF is open in another application, close it before regenerating the sample to avoid file-permission errors on Windows.

### Stage 1 — Analyze a PDF packet

```bash
python scripts/run_stage1.py \
  --input outputs/samples/sample_packet.pdf \
  --output outputs/stage1.json \
  --method baseline
```

Methods:

```text
baseline | embedding | learned
```

For `learned`, train the classifier first:

```bash
python scripts/run_benchmarks.py --stage stage1
```

### Stage 2 — Structure documents

```bash
python scripts/run_stage2.py \
  --stage1 outputs/stage1.json \
  --pdf outputs/samples/sample_packet.pdf \
  --output outputs/structured/
```

### Stage 3 — Index and query

```bash
python scripts/build_index.py --structured outputs/structured/
python scripts/query.py --query "What is the total amount on the invoice?"
```

### Dataset-backed benchmarks

```bash
# Stage 1
python scripts/run_benchmarks.py --stage stage1

# All available benchmark stages
python scripts/run_benchmarks.py
```

Stage-specific benchmark scripts used for the canonical Stage 2/3 measurements are:

```bash
python scripts/run_stage2_benchmark.py
python scripts/run_stage3_benchmark.py --no-reranker
python scripts/run_stage3_benchmark.py --use-reranker
```

---

## API

Start the FastAPI server:

```bash
uvicorn document_intelligence.api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/analyze` | POST | Upload PDF → Stage 1 results |
| `/index` | POST | Build retrieval index from structured docs |
| `/retrieve` | POST | Query evidence |

Example:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the invoice total?", "top_k": 5}'
```

The retrieval API can expose the reranker as an explicit per-request option. Reranking is disabled by default because the benchmark showed a query-specific regression on "most recent job title" style queries.

---

# Benchmarks and Reports

## Canonical benchmark run

The canonical benchmark was measured on **2026-08-16**.

Stage 1 was run with:

```bash
python scripts/run_benchmarks.py --stage stage1
```

The Stage 1 experiment was independently reproduced twice, with matching Boundary F1 and page-grouping accuracy to four decimal places.

Stage 2 and Stage 3 were measured using the fixed chunker:

```bash
python scripts/run_stage2_benchmark.py
python scripts/run_stage3_benchmark.py --no-reranker
python scripts/run_stage3_benchmark.py --use-reranker
```

Embeddings were real transformer embeddings, not hash fallbacks:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The Stage 1 classifier scaler is fit only on the training split after the train/validation split, preventing validation leakage.

---

## Stage 1 — Boundary Detection

Evaluation: `doc-split-benchmark`, `our200`, 200 test streams and 694 page pairs.

Metrics are macro-averaged per stream.

| Method | Boundary Precision | Boundary Recall | Boundary F1 | Page Grouping Accuracy |
|--------|---------------------|-----------------|-------------|------------------------|
| `baseline_rule` | 0.532 | 0.583 | 0.533 | 0.573 |
| `baseline_embedding` | 0.535 | 0.536 | 0.517 | 0.582 |
| `learned_classifier` | **0.753** | **0.890** | **0.784** | **0.722** |

### Training and resource measurements

| Metric | Value |
|--------|-------|
| Classifier train pairs | 32,408 train / 8,103 validation |
| Training time | ~2,074–2,123 s |
| Total Stage 1 wall clock | ~2,129–2,179 s |
| Peak RSS | ~3,644–6,248 MB |
| Saved classifier size | ~1.5 KB |

The classifier was trained on the full `openpss-mirror` SHORT train split using `class_weight="balanced"` because same-document page pairs are much more common than boundary pairs.

### Reproducibility

Two independent runs on 2026-08-16 produced:

```text
Boundary F1:              0.7838 / 0.7838
Page grouping accuracy:   0.7223 / 0.7223
```

This confirms that the reported Stage 1 result is stable.

### Top feature weights

| Feature | Weight |
|---------|--------|
| `token_jaccard` | 0.683–0.686 |
| `type_agreement` | 0.307–0.309 |
| `text_length_ratio` | 0.112 |
| `structural_similarity` | 0.112 |
| `entity_overlap` | -0.107 |
| `semantic_cosine` | -0.090 |

---

## Stage 2 — Document Structuring

The benchmark uses the bundled 6-page sample packet containing 3 logical documents.

| Metric | Before chunking fix | After chunking fix |
|--------|---------------------|---------------------|
| Documents processed | 3 | **3** |
| Avg blocks / document | 11.0 | **10.0** |
| Table blocks | Not tracked | **1** |
| Figure blocks | Not tracked | **0** |
| Provenance correctness | 3 / 3 | **3 / 3** |
| Processing time | 0.015 s | **0.018 s** |

The fixed structurer now recognizes:

- tables as `table` blocks
- figures/images as `figure` blocks
- lists as structured list blocks
- short labels merged with following content during chunk construction
- bounding-box metadata for layout traceability

There is no public labeled Stage 2 structured-output dataset, so this measurement is intentionally reported as a sample-packet benchmark rather than as a dataset accuracy claim.

---

## Stage 3 — Evidence Retrieval

Evaluation uses **6 hand-labeled queries**, two for each sample document type.

Ground truth is resolved dynamically by matching expected answer text against the current chunk output. This avoids hardcoded chunk IDs becoming invalid when chunk boundaries change.

### Before vs. after chunking fix

| Metric | Before fix (vector only) | After fix (vector only) | After fix (+ reranker) |
|--------|---------------------------|--------------------------|--------------------------|
| Top-1 Accuracy | 3 / 6 | **5 / 6** | **5 / 6** |
| Recall@1 | 0.500 | **0.833** | **0.833** |
| Recall@5 | 0.833 | **1.000** | 0.833 |
| MRR | 0.589 | **0.867** | 0.833 |
| nDCG | 0.648 | **0.898** | 0.833 |
| Indexing time | 13.05 s | 14.65 s | 13.54 s |

### Per-query results

| Query | Document | Vector only | + Reranker |
|---|---|---|---|
| "What is the total amount on the invoice?" | Invoice | ❌ MISS (rank 5) | ✅ OK |
| "What is the invoice number?" | Invoice | ✅ OK | ✅ OK |
| "What skills does the candidate have?" | Resume | ✅ OK | ✅ OK |
| "What is the candidate's most recent job title?" | Resume | ✅ OK | ❌ MISS |
| "What is the passport holder's date of birth?" | Passport | ✅ OK | ✅ OK |
| "What is the passport number?" | Passport | ✅ OK | ✅ OK |

### Key finding: chunking fixed the "Skills" failure

The query:

```text
What skills does the candidate have?
```

previously retrieved only:

```text
Skills:
```

because the label and its content were separate chunks.

The fix merges:

```text
Skills:
Python, Machine Learning, FastAPI
```

into a single retrieval unit.

This improved vector-only Top-1 accuracy from **3/6 to 5/6** and demonstrates that the failure was caused by Stage 2 chunk construction, not by Stage 3 ranking.

### Reranking regression

The query:

```text
What is the candidate's most recent job title?
```

is the remaining notable failure.

Vector-only retrieval correctly ranked:

```text
Senior Developer at TechCo
```

first.

With the cross-encoder reranker enabled, that chunk falls outside the top 5 and generic title mentions such as:

```text
Software Engineer
```

are preferred because they are lexically closer to "job title."

This regression was observed across two benchmark rounds. Therefore reranking is **not treated as a universal improvement**. It is exposed as an optional feature rather than enabled by default.

Reranking performed well on numeric/entity lookup queries such as:

- invoice total
- invoice number
- passport number

but currently hurts the "most recent role/title" query type.

---

## Resource & Performance Analysis

| Metric | Value |
|--------|-------|
| Stage 1 classifier training | ~2,074–2,123 s |
| Stage 1 peak RSS | ~3,644–6,248 MB |
| Classifier size | ~1.5 KB |
| Stage 2 processing | 0.018 s for 3 documents |
| Stage 3 indexing | 13.5–14.7 s for 27 chunks |
| Stage 3 warm query latency, no reranker | ~0.02–0.03 s |
| Stage 3 warm query latency, reranker | ~0.15–0.31 s |

All defaults target CPU execution.

### Measurement caveats

The embedding model and cross-encoder load lazily on first use. Therefore first-query latency is higher than warm-query latency. The reranked benchmark's first query was approximately 1 second, while subsequent queries were approximately 0.15–0.31 seconds.

The reported Stage 1 memory values are RSS snapshots collected after the measured work. They are useful resource indicators but are not equivalent to a continuously sampled OS-level maximum. A future benchmark can use `resource.getrusage(RUSAGE_SELF).ru_maxrss` for a more precise peak measurement.

---

## Failure Analysis

| Failure | Handling |
|---------|----------|
| Empty PDF | `PDFLoadError` with clear message |
| Scanned pages | OCR fallback with confidence |
| Unknown document type | Explicit `unknown` label |
| Low boundary confidence | Score retained in output |
| Empty/whitespace query | `empty_query` warning; no search performed |
| No retrieval match | Empty results + warning |

### System-level lesson

The pipeline's stages are sequential, so errors can compound:

```text
Boundary/grouping error
        ↓
Wrong document structure
        ↓
Bad chunk
        ↓
Missing retrieval candidate
        ↓
Reranker cannot recover it
```

The "Skills:" failure was a concrete example. Fixing Stage 2 chunking solved the Stage 3 failure without changing the retrieval model.

---

## Trade-offs

- **Rule-based boundary detection** is fast but substantially less accurate than the learned classifier: F1 0.533 vs. 0.784.
- **Logistic Regression** is small, interpretable, and resource-efficient, though larger tree-based models remain possible.
- **Flat FAISS** is simple and infrastructure-free, but a larger production corpus may require an ANN index or managed vector database.
- **Label-content merging** slightly reduces chunk atomicity but significantly improves retrievability for natural field-level questions.
- **Reranking is optional**, because it improves some entity/numeric lookups while currently hurting "most recent role/title" queries.
- **Basic table extraction** is intentionally lightweight and uses structural PDF patterns rather than a dedicated table model.
- **Native extraction before OCR** reduces compute for text-based PDFs.

---

## Limitations and Future Work

| Limitation | Notes |
|------------|-------|
| `doc-split-v2` unavailable as a dataset | It is a commercial model card, not a downloadable dataset; excluded from evaluation |
| No document-type labels in page-stream datasets | Classification is therefore evaluated on the sample packet rather than a large labeled corpus |
| Stage 3 labeled evaluation is small | Only 6 hand-labeled queries on the sample packet |
| Stage 3 has not been tested at Stage 1 dataset scale | Public page-stream datasets provide boundary labels, not structured/retrieval ground truth |
| Hash embedding fallback | Useful for offline tests; production benchmarks use real Sentence Transformer embeddings |
| Basic table extraction | Dedicated table models could improve complex layouts |
| Resource usage | Full Stage 1 training peaks around 6 GB RSS in the measured runs |

### Future improvements

1. Re-benchmark larger Sentence Transformer models.
2. Investigate the reranker regression on recency/title queries.
3. Add recency-aware signals to reranking.
4. Expand the Stage 3 labeled query set beyond six queries.
5. Extend label-content merging beyond the current short-label heuristic.
6. Add confidence calibration, e.g. Platt scaling.
7. Add Qdrant or pgvector adapters for distributed deployments.
8. Add PaddleOCR as an alternative OCR backend.
9. Activate the implemented `EmbeddingDocumentClassifier` using labeled document-type prototypes.
10. Build a larger labeled retrieval evaluation set for statistically stable Recall@k/MRR/nDCG measurements.

---

## Model Choices Summary

| Component | Baseline | Final Candidate | Optional |
|-----------|----------|-----------------|----------|
| Boundary detection | Rule/threshold | Logistic Regression | XGBoost, visual features |
| Classification | Keyword heuristics | Heuristic currently wired; embedding prototype implemented | LLM fallback (Gemini) |
| Embeddings | Hash fallback (offline) | `all-MiniLM-L6-v2` | Larger Sentence Transformer models |
| Retrieval | FAISS flat index | FAISS + structural chunks | Qdrant, pgvector |
| Reranking | Disabled | Optional cross-encoder | Query-type-specific routing |
| OCR | PyMuPDF native | pytesseract fallback | PaddleOCR |

---

## Testing

```bash
pytest tests/ -v
```

By default, tests use the real `sentence-transformers/all-MiniLM-L6-v2` model, which is downloaded and cached on first use. For offline or deterministic tests:

```bash
set USE_HASH_EMBEDDINGS=true       # Windows
# export USE_HASH_EMBEDDINGS=true  # Linux/macOS

pytest tests/ -v
```

Test coverage includes:

- PDF ingestion and OCR fallback
- page-pair similarity and boundary baselines
- classifier train/save/load
- page grouping and document classification
- dataset adapter for OpenPSS and benchmark schemas
- dataset-backed Stage 1 evaluation
- structured JSON validation and chunking provenance
- table/list/figure structuring
- vector indexing and retrieval ranking
- empty-query handling
- API health and error handling

---

## Docker

```bash
docker-compose up --build
curl http://localhost:8000/health
```

Volumes mount:

```text
data/
outputs/
models/
```

No secrets are baked into the image; configure credentials and API keys through `.env`.

---

## Project Structure

```text
PacketIQ/
├── src/document_intelligence/
│   ├── config/           # Settings (env + YAML)
│   ├── ingestion/        # PDF load, extract, OCR
│   ├── stage1/           # Boundary detection, grouping, classification
│   ├── stage2/           # Structure, chunking, metadata, schemas
│   ├── stage3/           # Embeddings, FAISS, retrieval, reranker
│   ├── evaluation/       # Metrics, benchmarks, dataset eval
│   ├── dataset/          # HuggingFace dataset adapter
│   ├── api/              # FastAPI endpoints
│   └── pipeline.py       # End-to-end orchestrator
├── scripts/              # CLI entry points and benchmark runners
├── tests/                # pytest suite
├── configs/              # config.yaml, models.yaml
├── docs/                 # architecture, benchmark, technical reports
├── data/                 # raw, processed, indexes
├── outputs/              # pipeline outputs, benchmarks, samples
├── models/               # runtime-trained boundary classifier
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Benchmarks and Raw Artifacts

Canonical raw outputs:

```text
outputs/benchmarks/stage1_dataset.json
outputs/benchmarks/stage2.json
outputs/benchmarks/stage3_vector.json
outputs/benchmarks/stage3_reranker.json
data/processed/dataset_relationship_report.json
```

Supporting reports:

```text
docs/architecture.md
docs/datasets.md
docs/benchmark_report.md
docs/technical_report.md
```

The benchmark report contains the detailed per-stage measurements and failure analysis. The technical report documents methodology, technical decisions, resource measurements, trade-offs, and future work.

---

## AI Usage Declaration

Implementation was scaffolded and refined with an AI coding assistant (Claude), used interactively for code generation, debugging, and diagnosis.

Human engineering decisions included:

- the three-stage hybrid architecture
- engineered boundary feature selection
- model and infrastructure choices
- dataset-role mapping
- evaluation methodology
- use of a real hand-labeled Stage 3 query set
- dynamic ground-truth resolution against current chunks
- label-content chunking fix
- table/list/figure structuring
- interpretation of reranking regression as a genuine limitation

Experiments included:

- Stage 1 dataset benchmark on 200 held-out test streams
- two independent Stage 1 reproductions
- Stage 2 benchmark before and after the chunking fix
- Stage 3 vector-only and reranked benchmarks before and after the chunking fix

Validation included dataset schema inspection, benchmark artifact inspection, verification of the `doc-split-v2` model-card status, manual verification of Stage 3 answer-text ground truth, and independent re-derivation of retrieval metrics from raw per-query results.

---

## License and Attribution

- **OpenPSS mirror:** redistribution of the OpenPSS page-stream segmentation benchmark ([University of Amsterdam / HuggingFace](https://huggingface.co/datasets/nutrientdocs/openpss-mirror))
- **Doc-split benchmark:** official evaluation slice ([HuggingFace](https://huggingface.co/datasets/nutrientdocs/doc-split-benchmark))
- **Doc-split-v2:** proprietary/commercial model card; not used as a dataset or model in this implementation.

