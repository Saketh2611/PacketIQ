FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY data/ data/
COPY outputs/ outputs/

ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=INFO
ENV EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV VECTOR_STORE=faiss
ENV OCR_ENGINE=pytesseract

EXPOSE 8000

CMD ["uvicorn", "document_intelligence.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
