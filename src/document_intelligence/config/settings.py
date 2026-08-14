"""Centralized typed configuration using environment variables and YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """Application settings loaded from env vars and YAML config files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Field(default_factory=_project_root)
    data_dir: Path = Field(default_factory=lambda: _project_root() / "data")
    raw_dir: Path = Field(default_factory=lambda: _project_root() / "data" / "raw")
    processed_dir: Path = Field(default_factory=lambda: _project_root() / "data" / "processed")
    indexes_dir: Path = Field(default_factory=lambda: _project_root() / "data" / "indexes")
    outputs_dir: Path = Field(default_factory=lambda: _project_root() / "outputs")
    models_dir: Path = Field(default_factory=lambda: _project_root() / "models")

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_batch_size: int = 32
    normalize_embeddings: bool = True
    device: str = "cpu"

    # Vector store
    vector_store: Literal["faiss", "chroma", "qdrant"] = "faiss"

    # OCR
    ocr_engine: Literal["pytesseract", "none"] = "pytesseract"
    ocr_min_text_length: int = 50
    render_dpi: int = 150

    # Stage 1
    boundary_threshold: float = 0.5
    semantic_weight: float = 0.4
    layout_weight: float = 0.2
    structural_weight: float = 0.2
    type_agreement_weight: float = 0.2
    boundary_classifier_type: str = "logistic_regression"
    random_seed: int = 42

    # Stage 2
    max_chunk_chars: int = 1000
    chunk_overlap: int = 100

    # Stage 3
    top_k: int = 5
    retrieval_top_n: int = 50
    rerank_top_n: int = 20
    use_reranker: bool = False

    # LLM (optional)
    gemini_api_key: str | None = None
    use_llm_fallback: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    # HuggingFace
    hf_token: str | None = None

    # Dataset
    dataset_name: str = "nutrientdocs/openpss-mirror"
    dataset_config: str | None = "SHORT"
    test_dataset_name: str = "nutrientdocs/doc-split-benchmark"
    test_dataset_config: str | None = "our200"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._apply_yaml_overrides()

    def _apply_yaml_overrides(self) -> None:
        config_path = self.project_root / "configs" / "config.yaml"
        models_path = self.project_root / "configs" / "models.yaml"
        config = _load_yaml(config_path)
        models = _load_yaml(models_path)

        paths = config.get("paths", {})
        for key, val in paths.items():
            attr = key.replace("_dir", "_dir") if key.endswith("_dir") else key
            if hasattr(self, attr):
                setattr(self, attr, self.project_root / val if isinstance(val, str) else val)

        stage1 = config.get("stage1", {})
        for k, v in stage1.items():
            mapped = {
                "classifier": "boundary_classifier_type",
                "boundary_threshold": "boundary_threshold",
                "semantic_weight": "semantic_weight",
                "layout_weight": "layout_weight",
                "structural_weight": "structural_weight",
                "type_agreement_weight": "type_agreement_weight",
                "random_seed": "random_seed",
            }.get(k, k)
            if hasattr(self, mapped):
                setattr(self, mapped, v)

        stage2 = config.get("stage2", {})
        if "max_chunk_chars" in stage2:
            self.max_chunk_chars = stage2["max_chunk_chars"]
        if "chunk_overlap" in stage2:
            self.chunk_overlap = stage2["chunk_overlap"]

        stage3 = config.get("stage3", {})
        for k in ("top_k", "retrieval_top_n", "rerank_top_n", "use_reranker", "normalize_embeddings"):
            if k in stage3:
                setattr(self, k, stage3[k])

        dataset = config.get("dataset", {})
        if "name" in dataset:
            self.dataset_name = os.getenv("DATASET_NAME", dataset["name"])
        if "config" in dataset:
            self.dataset_config = os.getenv("DATASET_CONFIG", dataset["config"])
        if "test_name" in dataset:
            self.test_dataset_name = os.getenv("TEST_DATASET_NAME", dataset["test_name"])
        if "test_config" in dataset:
            self.test_dataset_config = os.getenv("TEST_DATASET_CONFIG", dataset["test_config"])

        ocr = config.get("ocr", {}) | models.get("ocr", {})
        if "engine" in ocr:
            self.ocr_engine = ocr["engine"]
        if "min_text_length" in ocr:
            self.ocr_min_text_length = ocr["min_text_length"]
        if "render_dpi" in ocr:
            self.render_dpi = ocr["render_dpi"]

        emb = models.get("embedding", {})
        if "model_name" in emb:
            self.embedding_model = os.getenv("EMBEDDING_MODEL", emb["model_name"])
        if "batch_size" in emb:
            self.embedding_batch_size = emb["batch_size"]
        if "device" in emb:
            self.device = emb["device"]

        rerank = models.get("reranker", {})
        if "model_name" in rerank:
            self.reranker_model = os.getenv("RERANKER_MODEL", rerank["model_name"])
        if "enabled" in rerank:
            self.use_reranker = rerank["enabled"]

        vs = models.get("vector_store", {})
        if "backend" in vs:
            self.vector_store = os.getenv("VECTOR_STORE", vs["backend"])

        api = config.get("api", {})
        if "host" in api:
            self.api_host = api["host"]
        if "port" in api:
            self.api_port = api["port"]

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.indexes_dir,
            self.outputs_dir,
            self.models_dir,
            self.outputs_dir / "samples",
            self.outputs_dir / "structured",
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
