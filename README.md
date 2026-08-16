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

Boundary decisions are converted to contiguous page groups. Each group is classified using heuristic + optional embedding-based classifiers.

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

Chunking prefers **section boundaries** and preserves tables as logical units. Every chunk carries `chunk_id`, `document_id`, and `page_start`/`page_end`.

### Stage 3: Evidence Retrieval

```
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

---

## Datasets and Evaluation

### Dataset roles (verified from HuggingFace dataset cards)

| Dataset | Config | Split | Purpose |
|---------|--------|-------|---------|
| [`nutrientdocs/openpss-mirror`](https://huggingface.co/datasets/nutrientdocs/openpss-mirror) | `SHORT` | train | **Train/dev** — OpenPSS community mirror |
| [`nutrientdocs/doc-split-benchmark`](https://huggingface.co/datasets/nutrientdocs/doc-split-benchmark) | `our200` | test | **Test/eval** — official leaderboard slice |
| `nutrientdocs/doc-split-v2` | — | — | Referenced in assignment; **not accessible** on Hub |

**Important:** These three names are **not interchangeable**. Full explanation: [docs/datasets.md](docs/datasets.md)

- **`openpss-mirror`** — public redistribution of the OpenPSS page-stream segmentation benchmark for training and experimentation (40,715 train rows, 204 streams).
- **`doc-split-benchmark`** — the official evaluation slice (894 rows, 200 streams) used for leaderboard scoring.
- **`doc-split-v2`** — named in the assignment as the development dataset, but the Hub repo was not found/accessible during implementation. Use `openpss-mirror` for development and `doc-split-benchmark` for test evaluation.

### Schema mapping (handled by `DocSplitAdapter`)

Both usable datasets store **page streams** (one row per page), not pre-grouped packets:

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

# Download both manifests (train -> data/processed/train_manifest.json,
# test -> data/processed/test_manifest.json), using the datasets configured
# in .env / configs/config.yaml
python scripts/download_dataset.py

# Download only the train/dev manifest
python scripts/download_dataset.py --train-only

# Download only the test/eval manifest
python scripts/download_dataset.py --test-only
```

`download_dataset.py` doesn't take `--dataset`/`--config`/`--output` flags — it always downloads the pair of datasets defined by `DATASET_NAME`/`DATASET_CONFIG` (train) and `TEST_DATASET_NAME`/`TEST_DATASET_CONFIG` (test) in your settings, and always writes to `data/processed/{train,test}_manifest.json`. To point it at different datasets, change those variables in `.env` first.

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

> **Note:** `pip install -e .` installs the package in editable mode so `document_intelligence` is importable from anywhere (needed for the API and for `import document_intelligence...` in your own scripts). `pyproject.toml` also declares a `doc-intel` console-script entry point, but the module it points to (`document_intelligence.cli`) doesn't exist yet — use the scripts in `scripts/` (or `python main.py` for a one-line pointer) instead of `doc-intel` for now.

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
| `USE_HASH_EMBEDDINGS` | Deterministic hash fallback (testing/offline) | `false` |
| `USE_RERANKER` | Enable cross-encoder reranking | `false` |
| `BOUNDARY_THRESHOLD` | Same-document score threshold | `0.5` |
| `GEMINI_API_KEY` | Optional LLM fallback | (empty) |
| `HF_TOKEN` | HuggingFace token (rate limits) | (empty) |

---

## Running the Pipeline

### Quick end-to-end demo (sample PDF)

```bash
python scripts/generate_sample_outputs.py
```

This generates a synthetic 6-page packet (invoice + resume + passport), runs all three stages on it, and writes to `outputs/samples/`: a fresh `sample_packet.pdf`, `stage1_output.json`, per-document JSON under `structured/`, `retrieval_output.json`, and `failure_case_empty_query.json`. The repo already ships pre-generated versions of these files as a reference — running this script regenerates and overwrites them, which is a good first command to confirm your install works end to end.

### Stage 1 — Analyze a PDF packet

```bash
python scripts/run_stage1.py \
  --input outputs/samples/sample_packet.pdf \
  --output outputs/stage1.json \
  --method baseline
```

Methods: `baseline` | `embedding` | `learned` (run `python scripts/run_benchmarks.py --stage stage1` once beforehand to train the model that `--method learned` needs — see the note in [Stage 1](#stage-1-page-pair-boundary-detection))

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

---

## Benchmarks and Reports

### Run dataset-backed benchmarks

```bash
# Full Stage 1 benchmark (trains on openpss-mirror, evaluates on doc-split-benchmark)
python scripts/run_benchmarks.py --stage stage1

# All stages (Stage 3 retrieval uses smoke-test queries)
python scripts/run_benchmarks.py
```

Results: `outputs/benchmarks/stage1_dataset.json`

### Measured results (2026-08-15, doc-split-benchmark test, 200 streams)

| Method | Boundary F1 | Page Grouping Accuracy |
|--------|-------------|------------------------|
| baseline_rule | **0.799** | 0.715 |
| baseline_embedding | 0.768 | 0.715 |
| learned_classifier | 0.774 | **0.723** |

Classifier trained on 40,715 OpenPSS mirror rows. Full tables: [docs/benchmark_report.md](docs/benchmark_report.md)

Technical write-up: [docs/technical_report.md](docs/technical_report.md)

---

## Project Structure

```
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
├── scripts/              # CLI entry points
├── tests/                # pytest suite (39 tests)
├── configs/              # config.yaml, models.yaml
├── docs/                 # architecture, benchmark, technical reports
├── data/                 # raw, processed, indexes
├── outputs/              # pipeline outputs, benchmarks, samples
├── models/               # saved boundary classifier (created at runtime, gitignored — see below)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Testing

```bash
pytest tests/ -v
```

> **First run needs internet.** By default tests use the real `sentence-transformers/all-MiniLM-L6-v2` embedding model, which is downloaded and cached (~90 MB) the first time it's loaded. If you're offline or want faster/deterministic tests, set `USE_HASH_EMBEDDINGS=true` in your environment before running pytest to use a hash-based embedding fallback instead.

Test coverage includes:

- PDF ingestion and OCR fallback
- Page-pair similarity and boundary baselines
- Classifier train/save/load
- Page grouping and document classification
- Dataset adapter (OpenPSS + benchmark schema mapping)
- Dataset-backed Stage 1 evaluation
- Structured JSON validation and chunking provenance
- Vector indexing and retrieval ranking
- API health and error handling

---

## Docker

```bash
docker-compose up --build
curl http://localhost:8000/health
```

Volumes mount `data/`, `outputs/`, and `models/` for persistence. No secrets are baked into the image—configure via `.env`.

---

## Limitations and Future Work

| Limitation | Notes |
|------------|-------|
| `doc-split-v2` unavailable | Use `openpss-mirror` + `doc-split-benchmark` instead |
| No document-type labels in page-stream datasets | Classification evaluated on sample PDFs only |
| Stage 3 retrieval metrics | Smoke-test queries only; no labeled query set in datasets |
| Hash embedding fallback | Used when Sentence Transformers unavailable; re-run with `USE_HASH_EMBEDDINGS=false` for production signals |
| Basic table extraction | Native PDF blocks only; no dedicated table model |
| Resource usage | Full benchmark training peaks ~8 GB RAM |

**Future improvements:** Sentence Transformers re-benchmark, PaddleOCR, confidence calibration, Qdrant/pgvector adapters, labeled retrieval eval set.

---

## Model Choices Summary

| Component | Baseline | Final Candidate | Optional |
|-----------|----------|-----------------|----------|
| Boundary detection | Rule/threshold | Logistic Regression | XGBoost, visual features |
| Classification | Keyword heuristics | Embedding prototypes | LLM fallback (Gemini) |
| Embeddings | Hash fallback (offline) | `all-MiniLM-L6-v2` | Larger ST models |
| Retrieval | FAISS flat index | + cross-encoder rerank | Qdrant, pgvector |
| OCR | PyMuPDF native | pytesseract fallback | PaddleOCR |

---

## License and Attribution

- **OpenPSS mirror:** redistribution of the OpenPSS page-stream segmentation benchmark ([University of Amsterdam](https://huggingface.co/datasets/nutrientdocs/openpss-mirror))
- **Doc-split benchmark:** official evaluation slice ([nutrientdocs/doc-split-benchmark](https://huggingface.co/datasets/nutrientdocs/doc-split-benchmark))