# Document Packet Intelligence & Evidence Retrieval

A hybrid, measurable, resource-aware system for analyzing PDF packets containing multiple logical documents, detecting page boundaries, classifying document types, structuring content, and retrieving evidence for user queries.

## Problem Statement

Real-world document packets (loan applications, compliance bundles, onboarding packages) often contain multiple independent documents merged into a single PDF. This system:

1. Detects document boundaries between pages
2. Groups pages into logical documents
3. Classifies each document type with confidence scores
4. Converts documents to structured JSON for retrieval
5. Retrieves ranked evidence (not generated answers) for user queries

## Architecture

```
PDF Packet → Ingestion → Stage 1 (Boundaries) → Stage 2 (Structure) → Stage 3 (Retrieval)
```

See [docs/architecture.md](docs/architecture.md) for detailed component diagrams.

## Why Page-Boundary Detection?

Merged PDFs lack explicit document separators. Without boundary detection, retrieval mixes evidence across unrelated documents (e.g., invoice totals with resume skills). Page-pair boundary classification uses semantic, layout, and structural signals—not LLM guessing—to split packets deterministically.

## Dataset Setup

```bash
python scripts/download_dataset.py
python scripts/inspect_dataset.py
```

Uses Hugging Face `nutrientdocs/openpss-mirror` as the training/development dataset and `nutrientdocs/doc-split-benchmark` as the test/evaluation dataset. The adapter inspects the actual schema programmatically.

Default commands use `openpss-mirror` config `SHORT`. Test inspection can be run explicitly:

```bash
python scripts/inspect_dataset.py --dataset nutrientdocs/doc-split-benchmark --config our200 --split test
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -e .
```

## Environment Variables

Copy `.env.example` to `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Sentence Transformers model | `all-MiniLM-L6-v2` |
| `RERANKER_MODEL` | Cross-encoder reranker | `ms-marco-MiniLM-L-6-v2` |
| `VECTOR_STORE` | Backend (`faiss`) | `faiss` |
| `OCR_ENGINE` | OCR engine (`pytesseract`, `none`) | `pytesseract` |
| `USE_RERANKER` | Enable reranking | `false` |
| `GEMINI_API_KEY` | Optional LLM fallback | (empty) |

## Local Execution

### Generate sample outputs (end-to-end)

```bash
python scripts/generate_sample_outputs.py
```

### Stage 1 — Boundary detection & classification

```bash
python scripts/run_stage1.py --input outputs/samples/sample_packet.pdf --output outputs/stage1.json
```

Methods: `baseline` (rule/threshold), `embedding`, `learned`

### Stage 2 — Structured extraction

```bash
python scripts/run_stage2.py --stage1 outputs/stage1.json --pdf outputs/samples/sample_packet.pdf --output outputs/structured/
```

### Stage 3 — Index & retrieve

```bash
python scripts/build_index.py --structured outputs/structured/
python scripts/query.py --query "What is the total amount on the invoice?"
```

## Docker Execution

```bash
docker-compose up --build
curl http://localhost:8000/health
```

## API Usage

```bash
uvicorn document_intelligence.api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health |
| `/analyze` | POST | Upload PDF, run Stage 1 |
| `/index` | POST | Build retrieval index |
| `/retrieve` | POST | Query evidence |

Example retrieve:

```json
POST /retrieve
{"query": "What is the invoice total?", "top_k": 5}
```

## Evaluation / Benchmarks

```bash
python scripts/run_benchmarks.py
```

Results saved to `outputs/benchmarks/`. See [docs/benchmark_report.md](docs/benchmark_report.md).

## Example Output

**Stage 1** — 3 documents detected from 6-page packet (invoice, resume, passport)

**Retrieval** — Query: "What is the total amount on the invoice?"

```json
{
  "query": "What is the total amount on the invoice?",
  "results": [{
    "document_id": "sample_packet_doc_001",
    "document_type": "invoice",
    "page": 3,
    "evidence": "Total Amount: ₹52,340",
    "score": 0.96
  }]
}
```

## Model Choices

| Component | Baseline | Final Candidate | Optional |
|-----------|----------|-----------------|----------|
| Boundary | Rule/threshold | Logistic Regression / XGBoost | Visual features |
| Classification | Keyword heuristics | Embedding prototypes | LLM fallback |
| Embeddings | — | `all-MiniLM-L6-v2` | Larger models |
| Retrieval | FAISS vector search | + Cross-encoder rerank | Qdrant/pgvector |
| OCR | PyMuPDF native | pytesseract fallback | PaddleOCR |

## Resource Requirements

- **CPU**: Works on CPU-only (default)
- **RAM**: ~2 GB minimum (embedding model loaded)
- **Disk**: ~500 MB for models (cached on first run)
- **GPU**: Optional, configurable via `device` in config

## Failure Cases

- Empty/corrupted PDFs → structured errors
- Image-only pages → OCR fallback with warnings
- Unknown document types → explicit `unknown` label
- Low-confidence boundaries → retained in output with scores
- No retrieval results → empty results with warning

## Limitations

- Heuristic classification is a baseline; embedding classifier requires training data
- OpenPSS and benchmark schemas may differ; adapter handles multiple field conventions
- Table extraction is basic (native text blocks only)
- Reranker disabled by default for resource efficiency

## Future Improvements

- Train boundary classifier on full OpenPSS mirror split
- PaddleOCR integration for better scanned document support
- Layout-aware table extraction
- Qdrant/pgvector adapters for production scale
- Calibration for confidence scores

## Tests

```bash
pytest tests/ -v
```

## Project Structure

```
src/document_intelligence/   # Core package
scripts/                     # CLI tools
tests/                       # Unit & integration tests
configs/                     # YAML configuration
docs/                        # Architecture & reports
outputs/samples/             # Sample pipeline outputs
```
