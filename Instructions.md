<ROLE>

You are a senior AI/ML engineer responsible for implementing the complete engineering challenge described below.

You must produce a **fully working, reproducible, documented implementation**, not pseudocode.

You must:
- inspect the repository first;
- create/update all required source files;
- implement the complete 3-stage pipeline;
- provide tests;
- provide benchmark/evaluation scripts;
- provide Docker/Docker Compose;
- provide a README;
- provide an architecture diagram;
- provide sample outputs;
- provide a technical report template/content based only on measured results;
- make every major engineering choice configurable;
- never hard-code API keys, model credentials, absolute local paths, or machine-specific settings.

</ROLE>

<ASSIGNMENT_CONTEXT>

The challenge is a **Document Packet Intelligence & Evidence Retrieval** system.

The system receives a **single PDF packet containing multiple independent logical documents**.

Example input:

    packet.pdf
      Page 1 ┐
      Page 2 ├── Document A: Invoice
      Page 3 ┘

      Page 4 ┐
      Page 5 ├── Document B: Resume
      Page 6 ┘

      Page 7 ┐
      Page 8 ├── Document C: Passport
      Page 9 ┘

The system must:

1. Detect document boundaries.
2. Group pages belonging to the same logical document.
3. Classify each document type.
4. Produce confidence scores.
5. Convert each identified document into a structured representation suitable for downstream retrieval.
6. Retrieve the most relevant supporting evidence for a user query.
7. Return evidence with document identifier, page reference, and relevance/confidence score.

This is **not a chatbot-generation task**. The retrieval system should return evidence, not fabricate an answer.

Development/experimentation dataset:

- OpenPSS mirror
- Hugging Face dataset: `nutrientdocs/openpss-mirror`
- Test/evaluation dataset: `nutrientdocs/doc-split-benchmark`

Final evaluation may use a hidden dataset.

Do NOT use:
- DocSplit/DocSplit-v2-specific pretrained models/checkpoints;
- benchmark-specific end-to-end solutions that directly solve the complete task;
- commercial Document AI APIs that already perform the complete required solution.

General-purpose pretrained:
- OCR models,
- embedding models,
- vision models,
- language models,
- detection models,
- retrieval/reranking models

are allowed.

</ASSIGNMENT_CONTEXT>

<PRIMARY_ENGINEERING_APPROACH>

Implement a **hybrid, measurable, resource-aware architecture**.

Do NOT make an LLM the primary page-boundary detector.

The recommended design is:

    PDF Packet
        |
        v
    PDF Ingestion / Rendering
        |
        v
    Page-level extraction
        |
        +--> Native PDF text extraction with PyMuPDF
        |
        +--> OCR fallback for scanned/image pages
        |
        +--> Layout / block features
        |
        v
    Stage 1: Page-Pair Boundary Detection
        |
        +--> text semantic similarity
        +--> layout similarity
        +--> structural similarity
        +--> document-type agreement
        +--> optional entity overlap
        |
        v
    Boundary Classifier
        |
        +--> baseline: rule/threshold method
        +--> learned model: Logistic Regression / XGBoost / LightGBM
        |
        v
    Sequence-aware page grouping
        |
        v
    Document Type Classification
        |
        v
    Stage 1 Output
        |
        v
    Stage 2: Structured Document Representation
        |
        +--> headings
        +--> paragraphs
        +--> lists
        +--> tables
        +--> figures/images when detectable
        +--> page references
        +--> metadata
        +--> logical hierarchy
        |
        v
    Semantic/structural chunking
        |
        v
    Embedding model
        |
        v
    Vector DB + metadata store
        |
        v
    Stage 3: Evidence Retrieval
        |
        +--> query embedding
        +--> vector retrieval
        +--> optional metadata filtering
        +--> optional reranking
        |
        v
    Evidence results
        - document_id
        - document_type
        - page
        - chunk_id
        - evidence text
        - relevance score

Use heavier AI models only where experiments show they are necessary.

</PRIMARY_ENGINEERING_APPROACH>

<MODEL_AND_PROVIDER_POLICY>

The implementation must support configurable models.

Recommended open-source defaults:

1. Embeddings:
   - choose a strong general-purpose Hugging Face/Sentence Transformers embedding model;
   - make the exact model configurable through an environment variable/config file;
   - do not use a DocSplit-specific model.

2. OCR:
   - preferred: PaddleOCR or another reliable open-source OCR engine;
   - use PyMuPDF native extraction first;
   - invoke OCR when native text is empty/insufficient or explicitly requested.

3. Boundary classifier:
   - baseline: thresholded similarity;
   - final candidate: Logistic Regression, XGBoost, or LightGBM using engineered page-pair features.

4. Document type classifier:
   - implement a lightweight baseline first;
   - support a general embedding-based classifier and/or general-purpose model;
   - do not depend on hard-coded document types only.

5. Retrieval:
   - vector database must be pluggable;
   - local default should work without external infrastructure;
   - support FAISS or Chroma for simple local execution;
   - optionally support Qdrant or pgvector through configuration/Docker.

6. Reranker:
   - optional but recommended;
   - use a general-purpose Hugging Face reranker/cross-encoder;
   - run only on top-N candidates.

7. Large language model:
   - LLM use is OPTIONAL and must not be required for the basic pipeline;
   - if used, make it a fallback for ambiguous cases, structure normalization, or difficult extraction;
   - support a Gemini API through `GEMINI_API_KEY` if the user configures it;
   - NEVER hard-code credentials;
   - include a provider interface so other providers can be substituted;
   - do not use an LLM to replace the entire required pipeline.

The implementation may also support other external providers such as Scale AI or other compatible services, but all such integrations must be optional adapters and must not be required for local reproducibility.

</MODEL_AND_PROVIDER_POLICY>

<PROJECT_STRUCTURE>

Create a clean Python project with the following structure unless the repository already has an equivalent structure worth preserving:

    document-intelligence/
    |
    +-- src/
    |   +-- document_intelligence/
    |       |
    |       +-- config/
    |       |   +-- settings.py
    |       |
    |       +-- ingestion/
    |       |   +-- pdf_loader.py
    |       |   +-- page_renderer.py
    |       |   +-- page_extractor.py
    |       |   +-- ocr.py
    |       |
    |       +-- stage1/
    |       |   +-- page_features.py
    |       |   +-- similarity.py
    |       |   +-- boundary_baseline.py
    |       |   +-- boundary_classifier.py
    |       |   +-- grouping.py
    |       |   +-- document_classifier.py
    |       |   +-- confidence.py
    |       |
    |       +-- stage2/
    |       |   +-- structure.py
    |       |   +-- schema.py
    |       |   +-- chunking.py
    |       |   +-- metadata.py
    |       |
    |       +-- stage3/
    |       |   +-- embeddings.py
    |       |   +-- vector_store.py
    |       |   +-- retriever.py
    |       |   +-- reranker.py
    |       |   +-- scoring.py
    |       |
    |       +-- evaluation/
    |       |   +-- stage1_metrics.py
    |       |   +-- stage2_metrics.py
    |       |   +-- retrieval_metrics.py
    |       |   +-- benchmark.py
    |       |   +-- resource_metrics.py
    |       |
    |       +-- api/
    |       |   +-- models.py
    |       |   +-- main.py
    |       |
    |       +-- utils/
    |           +-- logging.py
    |           +-- io.py
    |           +-- hashing.py
    |           +-- timing.py
    |
    +-- scripts/
    |   +-- download_dataset.py
    |   +-- inspect_dataset.py
    |   +-- run_stage1.py
    |   +-- run_stage2.py
    |   +-- build_index.py
    |   +-- query.py
    |   +-- run_benchmarks.py
    |   +-- generate_sample_outputs.py
    |
    +-- tests/
    |   +-- test_ingestion.py
    |   +-- test_boundary_detection.py
    |   +-- test_grouping.py
    |   +-- test_classification.py
    |   +-- test_structuring.py
    |   +-- test_chunking.py
    |   +-- test_retrieval.py
    |   +-- test_api.py
    |
    +-- configs/
    |   +-- config.yaml
    |   +-- models.yaml
    |
    +-- data/
    |   +-- raw/
    |   +-- processed/
    |   +-- indexes/
    |   +-- samples/
    |
    +-- experiments/
    |   +-- boundary/
    |   +-- classification/
    |   +-- retrieval/
    |
    +-- benchmarks/
    |
    +-- outputs/
    |
    +-- docs/
    |   +-- architecture.md
    |   +-- technical_report.md
    |   +-- benchmark_report.md
    |
    +-- Dockerfile
    +-- docker-compose.yml
    +-- requirements.txt
    +-- pyproject.toml
    +-- .env.example
    +-- .gitignore
    +-- README.md

Keep modules small and testable.

</PROJECT_STRUCTURE>

<FILE_RESPONSIBILITIES>

<FILE path="src/document_intelligence/config/settings.py">
Centralized typed configuration using environment variables and/or YAML.

Include:
- dataset path
- input/output paths
- embedding model
- reranker model
- OCR configuration
- vector DB configuration
- thresholds
- top-k values
- API host/port
- optional LLM provider settings

Never store secrets in source code.
</FILE>

<FILE path="src/document_intelligence/ingestion/pdf_loader.py">
Load PDF packets safely.

Requirements:
- open PDF;
- count pages;
- expose page iteration;
- detect page dimensions;
- retain original page number;
- handle corrupted/empty pages gracefully;
- produce deterministic page IDs.
</FILE>

<FILE path="src/document_intelligence/ingestion/page_renderer.py">
Render pages to images when visual/OCR processing is needed.

Requirements:
- configurable DPI;
- deterministic output;
- avoid unnecessary rendering if native text extraction is sufficient.
</FILE>

<FILE path="src/document_intelligence/ingestion/page_extractor.py">
Extract page-level native content using PyMuPDF.

Extract:
- text;
- blocks;
- bounding boxes;
- images where relevant;
- document/page metadata;
- text length/statistics.

Return a stable internal `PageRepresentation`.
</FILE>

<FILE path="src/document_intelligence/ingestion/ocr.py">
OCR fallback.

Requirements:
- detect scanned/image-only pages;
- run OCR only when needed;
- return text and confidence where available;
- preserve page provenance;
- make OCR engine configurable.
</FILE>

<FILE path="src/document_intelligence/stage1/page_features.py">
Build numerical/features for adjacent page pairs.

Include as many reliable signals as practical:
- semantic embedding cosine similarity;
- text length ratio;
- token overlap;
- common entities/terms when available;
- layout/block-count similarity;
- text block position similarity;
- heading/style similarity when detectable;
- predicted page/document-type agreement;
- page image similarity only if computationally justified.

The function must be deterministic for a fixed model/config.
</FILE>

<FILE path="src/document_intelligence/stage1/similarity.py">
Implement reusable similarity functions.

Include:
- cosine similarity;
- normalized overlap;
- safe numerical handling;
- layout similarity.
</FILE>

<FILE path="src/document_intelligence/stage1/boundary_baseline.py">
Implement the simple baseline.

Input:
- ordered page representations.

Output:
- boundary probability/score for each pair;
- boundary decision.

Provide configurable thresholding.

This baseline is required because later experiments must compare it against the learned classifier.
</FILE>

<FILE path="src/document_intelligence/stage1/boundary_classifier.py">
Implement learned page-pair boundary classification.

Recommended:
- Logistic Regression and/or XGBoost.

Training example:
- adjacent page pair;
- engineered features;
- target:
  - 1 = same logical document;
  - 0 = document boundary
  OR the equivalent convention, but document it clearly.

Requirements:
- train;
- validate;
- save model;
- load model;
- predict probability;
- expose feature importance where supported;
- handle class imbalance.

Do not leak test information into training.
</FILE>

<FILE path="src/document_intelligence/stage1/grouping.py">
Convert pairwise boundary decisions into contiguous document groups.

Example:

pages 1-2: same
pages 2-3: same
pages 3-4: boundary
pages 4-5: same

=> groups:
[1,2,3], [4,5]

Ensure:
- all pages belong to exactly one group;
- groups preserve page order;
- no empty groups;
- confidence for each boundary/group is retained.
</FILE>

<FILE path="src/document_intelligence/stage1/document_classifier.py">
Classify each grouped document.

Implement:
1. simple baseline, preferably rules/keyword heuristics where appropriate;
2. learned/general-purpose classifier.

Do not rely exclusively on hard-coded labels.

Use the entire grouped document representation when possible.

Return:
- document_type
- confidence
- optional top-k type candidates.

Document unknown/low-confidence classification explicitly.
</FILE>

<FILE path="src/document_intelligence/stage1/confidence.py">
Define confidence computation consistently.

Confidence must not be an arbitrary random number.

Document whether confidence comes from:
- classifier probability;
- normalized similarity margin;
- combined calibrated probability;
- or another defensible method.

Where feasible, provide calibration support.
</FILE>

<FILE path="src/document_intelligence/stage2/schema.py">
Define typed Pydantic/dataclass schemas for:

Packet
Page
Document
Section
Table
Figure
Chunk
Metadata
Evidence

Every structured element must preserve page/document provenance.
</FILE>

<FILE path="src/document_intelligence/stage2/structure.py">
Convert each identified document to structured JSON.

Preserve, where available:
- title;
- headings;
- subheadings;
- paragraphs;
- lists;
- tables;
- figures/images;
- metadata;
- page references;
- logical hierarchy.

Use native PDF extraction first.

Use OCR/layout parsing only when needed.

Do not invent content.
</FILE>

<FILE path="src/document_intelligence/stage2/metadata.py">
Generate consistent metadata.

At minimum:
- packet_id;
- document_id;
- document_type;
- source_file;
- page_start;
- page_end;
- page_count;
- extraction_method;
- processing timestamp;
- model versions where relevant.

Use content hashes for reproducibility where helpful.
</FILE>

<FILE path="src/document_intelligence/stage2/chunking.py">
Implement structural/semantic chunking.

Do NOT simply split every N characters without preserving meaning.

Recommended strategy:
- prefer section boundaries;
- preserve tables as logical chunks;
- preserve page references;
- use token/character limits only as secondary constraints;
- include controlled overlap only where useful.

Every chunk must retain:
- chunk_id
- document_id
- page_start/page_end
- section
- chunk text
- metadata
</FILE>

<FILE path="src/document_intelligence/stage3/embeddings.py">
Provide a pluggable embedding interface.

Default:
- Hugging Face/Sentence Transformers model.

Requirements:
- batch encoding;
- normalization support;
- caching;
- model name/version recorded;
- CPU-compatible default;
- optional GPU usage.
</FILE>

<FILE path="src/document_intelligence/stage3/vector_store.py">
Implement a vector store interface.

Must support a local zero-infrastructure option.

Recommended:
- FAISS or Chroma as default.

Optional:
- Qdrant / pgvector adapters.

Store:
- vector;
- chunk_id;
- document_id;
- document_type;
- page_start/page_end;
- text;
- metadata.

The system must preserve exact provenance.
</FILE>

<FILE path="src/document_intelligence/stage3/retriever.py">
Implement retrieval.

Pipeline:
1. normalize query;
2. embed query;
3. retrieve top-N candidates;
4. optionally apply metadata filters;
5. optionally rerank;
6. return top-k evidence.

Output must contain:
- evidence;
- document_id;
- document_type;
- page reference;
- chunk_id;
- relevance score.

Do not generate a conversational answer.
</FILE>

<FILE path="src/document_intelligence/stage3/reranker.py">
Optional reranking layer.

Use a general-purpose Hugging Face cross-encoder/reranker.

Requirements:
- top-N only;
- configurable;
- can be disabled;
- benchmark retrieval with and without reranking.
</FILE>

<FILE path="src/document_intelligence/stage3/scoring.py">
Implement normalized/consistent retrieval scoring.

Clearly distinguish:
- vector similarity;
- reranker score;
- final ranking score.

Do not call an arbitrary number a probability unless calibrated.
</FILE>

<FILE path="src/document_intelligence/evaluation/stage1_metrics.py">
Implement:
- Boundary Precision
- Boundary Recall
- Boundary F1
- Page Grouping Accuracy
- Classification Accuracy
- confusion matrix where relevant

Define boundary representation carefully and document it.
</FILE>

<FILE path="src/document_intelligence/evaluation/stage2_metrics.py">
Implement reasonable structured-output metrics.

At minimum measure:
- extraction completeness/coverage;
- field/element preservation where ground truth exists;
- table preservation where evaluable;
- page provenance correctness;
- processing time.

If the dataset does not provide a reliable Stage 2 ground truth, do not invent a fake accuracy metric. Clearly report what can and cannot be objectively measured.
</FILE>

<FILE path="src/document_intelligence/evaluation/retrieval_metrics.py">
Implement:
- Recall@1
- Recall@3
- Recall@5
- Precision@1
- Precision@5
- MRR
- nDCG
- average retrieval latency
- indexing time

Use a clearly defined query/evidence evaluation dataset.
</FILE>

<FILE path="src/document_intelligence/evaluation/resource_metrics.py">
Measure:
- wall-clock latency;
- CPU usage where practical;
- RAM;
- model size;
- index size;
- optional GPU memory;
- indexing throughput.

Keep measurements reproducible.
</FILE>

<FILE path="src/document_intelligence/evaluation/benchmark.py">
Run complete benchmarks for:
- Stage 1 baseline;
- Stage 1 improved model;
- Stage 2 processing;
- Stage 3 retrieval;
- resource usage.

Produce machine-readable JSON/CSV and human-readable summaries.
</FILE>

<FILE path="src/document_intelligence/api/models.py">
Define API request/response schemas using Pydantic.

Required endpoints should support:
- ingest/build index;
- inspect packet;
- run packet intelligence;
- retrieve evidence.
</FILE>

<FILE path="src/document_intelligence/api/main.py">
Implement a FastAPI API.

Recommended endpoints:

POST /analyze
POST /index
POST /retrieve
GET /health

The API should return structured JSON and preserve provenance.
</FILE>

</FILE_RESPONSIBILITIES>

<DATASET_PIPELINE>

Implement scripts to fetch and inspect the OpenPSS mirror and doc-split-benchmark datasets.

The agent must first determine the actual dataset schema programmatically.

Do not assume field names.

`download_dataset.py` should:
- use Hugging Face datasets when practical;
- download/copy the development data;
- save a local metadata manifest;
- avoid downloading duplicate data unnecessarily.

`inspect_dataset.py` should print:
- split names;
- number of samples;
- available fields;
- example sample structure;
- page/document labels if present;
- useful statistics.

Then implement dataset adapters that transform the real dataset format into the internal training representation.

The agent must not fabricate labels.

If the dataset schema differs from expectations, adapt to the actual schema and document the mapping.

</DATASET_PIPELINE>

<STAGE1_TRAINING_DATA>

Build page-pair examples from the dataset.

For each packet:
- order pages exactly as provided;
- generate adjacent page pairs;
- derive the boundary/same-document target from the true annotations;
- generate feature vectors;
- store a reproducible training artifact.

Avoid data leakage:
- packets, not pages, should define train/validation/test separation;
- never put pages from the same packet into multiple splits.

If the development dataset already provides explicit document IDs/boundaries, use those as ground truth.

</STAGE1_TRAINING_DATA>

<STAGE1_ALGORITHM>

Implement and compare the following:

### Baseline A — Rule/threshold

Use semantic similarity and simple structural signals.

### Baseline B — Text embedding similarity

Use adjacent-page cosine similarity.

### Final candidate — Learned classifier

Use engineered pair features with Logistic Regression and/or XGBoost.

Optional:
- add a visual/layout feature set if the benchmark shows an improvement worth the compute.

Boundary decisions must be contiguous and deterministic.

Document all thresholds.

</STAGE1_ALGORITHM>

<DOCUMENT_CLASSIFICATION>

Implement a simple-to-strong progression:

1. heuristic baseline;
2. embedding-based classifier;
3. optional general-purpose model for ambiguous cases.

The final classification output must include:
- predicted type;
- confidence;
- optional candidate classes.

Do not overfit to the development dataset's exact examples.

Avoid brittle hard-coded keyword-only classification as the final method.

</DOCUMENT_CLASSIFICATION>

<STAGE2_STRUCTURED_FORMAT>

Use JSON as the primary persisted structured representation.

Recommended document JSON shape:

```json
{
  "document_id": "doc_001",
  "document_type": "invoice",
  "confidence": 0.94,
  "source": {
    "packet_id": "packet_001",
    "source_file": "packet.pdf",
    "page_start": 1,
    "page_end": 3
  },
  "content": {
    "title": "...",
    "sections": [
      {
        "heading": "Invoice Details",
        "page": 1,
        "blocks": [
          {
            "type": "paragraph",
            "text": "...",
            "page": 1
          },
          {
            "type": "table",
            "page": 2,
            "headers": ["Item", "Qty", "Amount"],
            "rows": [
              ["Laptop", "2", "50000"]
            ]
          }
        ]
      }
    ]
  }
}
```

This is an example shape, not a reason to fabricate unavailable information.

</STAGE2_STRUCTURED_FORMAT>

<RETRIEVAL_DESIGN>

Implement a RAG-style ingestion/retrieval flow, but the system must remain evidence-centric.

Ingestion:

    structured JSON
        |
        v
    semantic/structural chunks
        |
        v
    embeddings
        |
        v
    vector store

Query:

    user query
        |
        v
    query embedding
        |
        v
    top-N vector search
        |
        +--> optional metadata filtering
        |
        +--> optional reranker
        |
        v
    top-k evidence

Example:

Query:
"What is the total amount on the invoice?"

Expected result shape:

```json
{
  "query": "What is the total amount on the invoice?",
  "results": [
    {
      "document_id": "doc_001",
      "document_type": "invoice",
      "page": 3,
      "chunk_id": "doc_001_chunk_004",
      "evidence": "Total Amount: ₹52,340",
      "score": 0.96
    }
  ]
}
```

Never return unsupported generated facts as evidence.

</RETRIEVAL_DESIGN>

<CLI_REQUIREMENTS>

Provide commands such as:

    python scripts/download_dataset.py
    python scripts/inspect_dataset.py

    python scripts/run_stage1.py \
        --input /path/to/packet.pdf \
        --output outputs/stage1.json

    python scripts/run_stage2.py \
        --stage1 outputs/stage1.json \
        --output outputs/structured/

    python scripts/build_index.py \
        --structured outputs/structured/

    python scripts/query.py \
        --query "What is the total amount on the invoice?"

    python scripts/run_benchmarks.py

Make commands work both locally and inside Docker.

</CLI_REQUIREMENTS>

<API_REQUIREMENTS>

Provide FastAPI endpoints:

### POST /analyze

Input:
- PDF upload or configured file path.

Return:
- packet metadata;
- document groups;
- page ranges;
- predicted document types;
- confidence scores.

### POST /index

Input:
- structured documents or analyzed packet.

Action:
- chunk;
- embed;
- store.

Return:
- index statistics.

### POST /retrieve

Input:

```json
{
  "query": "What is the invoice total?",
  "top_k": 5
}
```

Return:
- ranked evidence;
- document IDs;
- page references;
- scores.

### GET /health

Return service health/status.

</API_REQUIREMENTS>

<TESTING_REQUIREMENTS>

Write unit tests and integration tests.

At minimum test:

1. PDF page counting.
2. Native text extraction.
3. OCR fallback behavior.
4. Similarity functions.
5. Boundary thresholding.
6. Boundary classifier training/loading.
7. Page grouping.
8. Document classification.
9. Structured JSON schema validation.
10. Chunking preserves provenance.
11. Vector indexing/search.
12. Retrieval ranking.
13. API endpoints.
14. Empty/malformed inputs.
15. Multi-page documents.
16. Single-page documents.
17. Boundary at first/last transition.
18. Scanned document.
19. Repeated headers/footers.
20. Low-confidence classification.

Tests must not require external API keys.

</TESTING_REQUIREMENTS>

<FAILURE_CASES>

Explicitly handle:
- empty PDF;
- corrupted PDF;
- encrypted PDF where possible;
- image-only pages;
- OCR failure;
- pages with almost no text;
- repeated headers/footers;
- pages from similar document types;
- pages with high visual similarity but different logical documents;
- pages with low semantic similarity but same document;
- tables;
- mixed OCR quality;
- unknown document type;
- low-confidence boundary decisions;
- duplicate chunks;
- retrieval with no relevant result.

Return clear errors and structured warnings.

Do not silently discard problematic pages.

</FAILURE_CASES>

<LOGGING_AND_OBSERVABILITY>

Implement structured logging.

Log:
- packet ID;
- page ID;
- document ID;
- stage;
- processing time;
- OCR usage;
- model name;
- vector index operation;
- retrieval latency;
- failures.

Do not log secret keys or sensitive credentials.

</LOGGING_AND_OBSERVABILITY>

<RESOURCE_EFFICIENCY>

This assignment explicitly evaluates resource efficiency.

Therefore:
- use PyMuPDF native extraction before OCR;
- batch embeddings;
- cache embeddings where useful;
- avoid rendering pages unnecessarily;
- only rerank top-N candidates;
- keep models configurable;
- default to CPU-compatible execution;
- allow optional GPU;
- report model sizes and runtime;
- avoid using an LLM for every page.

Benchmark:
- CPU;
- memory;
- latency;
- model size;
- index size;
- throughput where practical.

</RESOURCE_EFFICIENCY>

<DOCKER>

Create:
- Dockerfile;
- docker-compose.yml.

The default local stack should work without paid external services.

Recommended local services:
- application;
- optional Qdrant if selected;
- optional other infrastructure only when justified.

Mount:
- data;
- outputs;
- model/cache directories when appropriate.

Use environment variables for configuration.

Do not bake secrets into images.

</DOCKER>

<ENVIRONMENT>

Create `.env.example` containing placeholders such as:

    HF_TOKEN=
    GEMINI_API_KEY=
    EMBEDDING_MODEL=
    RERANKER_MODEL=
    VECTOR_STORE=
    OCR_ENGINE=
    LOG_LEVEL=INFO

The application must work without `GEMINI_API_KEY` if the LLM fallback is disabled.

Do not commit `.env`.

</ENVIRONMENT>

<README_REQUIREMENTS>

Write a complete README covering:

1. Problem statement.
2. What the system does.
3. Architecture.
4. Why page-boundary detection is required.
5. Dataset setup.
6. Installation.
7. Environment variables.
8. Local execution.
9. Docker execution.
10. Stage 1 usage.
11. Stage 2 usage.
12. Stage 3 usage.
13. API usage.
14. Evaluation/benchmark commands.
15. Example input/output.
16. Model choices.
17. Resource requirements.
18. Failure cases.
19. Limitations.
20. Future improvements.

Clearly distinguish:
- baseline;
- final candidate;
- optional components.

</README_REQUIREMENTS>

<ARCHITECTURE_DOCUMENT>

Create `docs/architecture.md` with:
- component diagram;
- end-to-end data flow;
- Stage 1 flow;
- Stage 2 flow;
- Stage 3 flow;
- data schemas;
- storage design;
- model boundaries;
- API boundaries.

Also include a Mermaid diagram in the Markdown.

</ARCHITECTURE_DOCUMENT>

<BENCHMARK_REPORT>

Create `docs/benchmark_report.md`.

Include tables for:

### Stage 1
- Boundary Precision
- Boundary Recall
- Boundary F1
- Page Grouping Accuracy
- Classification Accuracy
- Latency
- Memory
- Model Size

### Stage 2
- structure/extraction quality metrics actually supported by the dataset;
- provenance correctness;
- processing time;
- throughput/resource metrics.

### Stage 3
- Recall@1
- Recall@3
- Recall@5
- Precision@1
- Precision@5
- MRR
- nDCG
- Average Retrieval Latency
- Indexing Time

Also compare:
- rule baseline;
- embedding baseline;
- learned boundary model;
- vector-only retrieval;
- vector + reranker if implemented.

Do not fabricate benchmark numbers. The files should contain generated results from actual runs.

</BENCHMARK_REPORT>

<TECHNICAL_REPORT>

Create `docs/technical_report.md`, within the assignment's maximum 10-page target when rendered.

Cover:
1. Problem Understanding
2. Overall Approach
3. Architecture
4. Three Most Important Technical Decisions
5. Technology Selection and Alternatives
6. Stage 1 Method
7. Stage 2 Method
8. Stage 3 Method
9. Evaluation Methodology
10. Benchmark Results
11. Failure Analysis
12. Resource & Performance Analysis
13. Trade-offs
14. Future Improvements
15. AI Usage Declaration

Do not claim experiments that were not actually run.

Where benchmark values are unknown, use placeholders like:

    [RUN BENCHMARK TO FILL]

and clearly mark them.

</TECHNICAL_REPORT>

<AI_USAGE_DECLARATION>

Add an AI usage section that honestly describes use of coding assistants/model assistants.

Do not falsely claim that all engineering decisions were human-written.

Separate:
- AI-assisted code generation;
- human engineering decisions;
- experiments performed;
- validation performed.

</AI_USAGE_DECLARATION>

<SAMPLE_OUTPUTS>

Generate representative outputs under:

    outputs/samples/

At minimum:
- `stage1_output.json`
- `structured_document.json`
- `retrieval_output.json`

Also include one failure-case example if possible.

Do not invent sample values when real dataset/sample execution is available.

</SAMPLE_OUTPUTS>

<SECURITY_AND_CREDENTIALS>

Never:
- hard-code API keys;
- hard-code Hugging Face tokens;
- commit credentials;
- print credentials;
- store secrets in benchmark outputs.

Use:
- environment variables;
- `.env.example`;
- secret-safe logging.

</SECURITY_AND_CREDENTIALS>

<CODE_QUALITY>

Use:
- Python 3.11+ unless the repository requires otherwise;
- type hints;
- docstrings for public classes/functions;
- clear naming;
- small modules;
- error handling;
- deterministic seeds where relevant;
- configuration-driven behavior;
- reproducible CLI commands.

Prefer:
- Pydantic for schemas;
- pytest for tests;
- FastAPI for API;
- Hugging Face/Sentence Transformers for embeddings;
- PyMuPDF for PDF extraction;
- open-source OCR;
- FAISS/Chroma/Qdrant/pgvector for vector storage.

Do not add unnecessary dependencies.

</CODE_QUALITY>

<IMPLEMENTATION_ORDER>

Implement in this exact order:

1. Inspect repository.
2. Inspect actual OpenPSS mirror and doc-split-benchmark schemas.
3. Build dataset adapter.
4. Build PDF ingestion.
5. Build PageRepresentation.
6. Build Stage 1 baseline.
7. Build Stage 1 learned boundary classifier.
8. Build page grouping.
9. Build document classification.
10. Benchmark Stage 1.
11. Build Stage 2 schema.
12. Build structure extraction.
13. Build chunking.
14. Build embeddings.
15. Build vector store.
16. Build Stage 3 retrieval.
17. Add optional reranking.
18. Benchmark Stage 3.
19. Add API.
20. Add tests.
21. Add Docker.
22. Generate sample outputs.
23. Generate architecture docs.
24. Generate benchmark report.
25. Generate technical report.
26. Run the complete pipeline end-to-end.
27. Fix all errors found during the final run.
28. Update README with commands that actually work.

</IMPLEMENTATION_ORDER>

<IMPORTANT_REASONING_RULES>

1. Do not over-engineer before establishing a baseline.
2. Do not use an LLM just because the task contains the word "AI".
3. Prefer measurable signals and small models where they work.
4. Use general-purpose models only.
5. Keep the whole pipeline reproducible.
6. Preserve page/document provenance from ingestion to retrieval.
7. Never fabricate structure or evidence.
8. Do not hide failure cases.
9. Benchmark before and after major changes.
10. Choose the final architecture based on accuracy/resource trade-offs.
11. Keep optional LLM functionality isolated behind an adapter.
12. Do not assume the dataset schema; inspect it first.
13. Do not assume document types from a README alone; inspect actual labels.
14. Do not leak test data into training.
15. Do not hard-code thresholds without documenting how they were selected.

</IMPORTANT_REASONING_RULES>

<FINAL_ACCEPTANCE_CRITERIA>

The implementation is complete only when all of the following work:

### Input
A PDF packet containing multiple logical documents.

### Stage 1
The system:
- reads all pages;
- detects likely document boundaries;
- groups contiguous pages;
- classifies document types;
- produces confidence scores;
- reports Stage 1 metrics.

### Stage 2
The system:
- converts each grouped document to structured JSON;
- preserves page references;
- preserves useful structure;
- creates retrieval-ready chunks;
- stores metadata.

### Stage 3
The system:
- embeds chunks;
- indexes them;
- accepts a user query;
- retrieves relevant evidence;
- optionally reranks;
- returns document ID, page, evidence, chunk ID, and score;
- reports retrieval metrics.

### Engineering
The project:
- runs locally;
- runs in Docker;
- has tests;
- has CLI commands;
- has API endpoints;
- has benchmark scripts;
- has sample outputs;
- has architecture documentation;
- has README;
- has technical report;
- has no hard-coded secrets.

</FINAL_ACCEPTANCE_CRITERIA>

<FINAL_INSTRUCTION_TO_AGENT>

Start by inspecting the repository and the actual OpenPSS mirror and doc-split-benchmark dataset schemas.

Then implement the complete system described above.

Do not stop after creating a plan.

Actually create/update all required files and code.

After implementation:
1. run tests;
2. run dataset inspection;
3. run a small end-to-end sample;
4. run benchmark scripts where feasible;
5. fix errors;
6. verify Docker configuration;
7. verify README commands;
8. verify outputs;
9. report exactly what was implemented and which benchmarks were actually executed.

If something in the dataset or repository differs from the assumptions in this prompt, adapt to the real data rather than inventing a schema.

The final implementation must be a working engineering project, not a collection of placeholders or pseudocode.

</FINAL_INSTRUCTION_TO_AGENT>
