"""Dataset adapter for page-stream segmentation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_intelligence.config.settings import get_settings
from document_intelligence.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PagePairTrainingExample:
    packet_id: str
    page_a: int
    page_b: int
    features: list[float]
    label: int  # 1=same document, 0=boundary
    document_type_a: str | None = None
    document_type_b: str | None = None


@dataclass
class PacketAnnotation:
    packet_id: str
    page_count: int
    groups: list[list[int]]
    document_types: list[str]
    pdf_path: str | None = None


@dataclass
class NormalizedStreamPage:
    stream_id: str
    page_number: int
    original_position: int
    text: str
    starts_new_document: bool
    image: Any | None = None
    source: str | None = None


class DocSplitAdapter:
    """Adapt HuggingFace page-stream datasets to internal training format."""

    def __init__(self, dataset_name: str | None = None, dataset_config: str | None = None) -> None:
        settings = get_settings()
        self.dataset_name = dataset_name or settings.dataset_name
        self.dataset_config = dataset_config if dataset_config is not None else settings.dataset_config
        self._dataset = None
        self._schema: dict[str, Any] = {}

    def load_dataset(self, split: str = "train") -> Any:
        from datasets import load_dataset

        logger.info("Loading dataset", name=self.dataset_name, config=self.dataset_config, split=split)
        if self.dataset_config:
            self._dataset = load_dataset(self.dataset_name, self.dataset_config, split=split)
        else:
            self._dataset = load_dataset(self.dataset_name, split=split)
        return self._dataset

    def inspect_schema(self) -> dict[str, Any]:
        if self._dataset is None:
            self.load_dataset()
        assert self._dataset is not None
        sample = self._dataset[0]
        self._schema = {
            "num_samples": len(self._dataset),
            "fields": list(sample.keys()),
            "example_keys": {k: type(v).__name__ for k, v in sample.items()},
        }
        return self._schema

    def _extract_groups_from_sample(self, sample: dict[str, Any]) -> tuple[list[list[int]], list[str]]:
        """Extract page groups and document types from dataset sample."""
        groups: list[list[int]] = []
        types: list[str] = []

        if "documents" in sample:
            for doc in sample["documents"]:
                pages = doc.get("pages", doc.get("page_numbers", []))
                if isinstance(pages, int):
                    pages = [pages]
                groups.append(sorted(pages) if pages else [])
                types.append(doc.get("document_type", doc.get("type", "unknown")))
            return groups, types

        if "page_labels" in sample and "document_ids" in sample:
            page_labels = sample["page_labels"]
            doc_ids = sample["document_ids"]
            doc_map: dict[Any, list[int]] = {}
            type_map: dict[Any, str] = {}
            for i, (did, label) in enumerate(zip(doc_ids, page_labels)):
                page_num = i + 1
                doc_map.setdefault(did, []).append(page_num)
                if "document_types" in sample:
                    type_map[did] = sample["document_types"][i]
                elif label:
                    type_map[did] = str(label)
            for did, pages in doc_map.items():
                groups.append(sorted(pages))
                types.append(type_map.get(did, "unknown"))
            return groups, types

        if "boundaries" in sample:
            boundaries = sorted(sample["boundaries"])
            page_count = sample.get("num_pages", sample.get("page_count", len(sample.get("pages", []))))
            start = 1
            for b in boundaries:
                groups.append(list(range(start, b + 1)))
                start = b + 1
            if start <= page_count:
                groups.append(list(range(start, page_count + 1)))
            types = sample.get("document_types", ["unknown"] * len(groups))
            return groups, types

        if "labels" in sample:
            labels = sample["labels"]
            current_label = labels[0]
            current_group = [1]
            for i in range(1, len(labels)):
                if labels[i] == current_label:
                    current_group.append(i + 1)
                else:
                    groups.append(current_group)
                    types.append(str(current_label))
                    current_group = [i + 1]
                    current_label = labels[i]
            groups.append(current_group)
            types.append(str(current_label))
            return groups, types

        logger.warning("Could not parse groups from sample", keys=list(sample.keys()))
        return [[1]], ["unknown"]

    def sample_to_annotation(self, sample: dict[str, Any], idx: int) -> PacketAnnotation:
        packet_id = sample.get("id", sample.get("packet_id", f"packet_{idx}"))
        groups, types = self._extract_groups_from_sample(sample)
        page_count = max((p for g in groups for p in g), default=0)
        if "num_pages" in sample:
            page_count = max(page_count, sample["num_pages"])
        return PacketAnnotation(
            packet_id=str(packet_id),
            page_count=page_count,
            groups=groups,
            document_types=types,
            pdf_path=sample.get("pdf_path", sample.get("file_name")),
        )

    @staticmethod
    def _stream_boundary_field(row: dict[str, Any]) -> str:
        if "boundary" in row:
            return "boundary"
        if "label" in row:
            return "label"
        raise KeyError("Expected stream row to contain 'boundary' or 'label'")

    @staticmethod
    def _stream_text_field(row: dict[str, Any]) -> str:
        if "page_text" in row:
            return "page_text"
        if "text" in row:
            return "text"
        raise KeyError("Expected stream row to contain 'page_text' or 'text'")

    def normalize_stream_rows(self, rows: list[dict[str, Any]]) -> list[NormalizedStreamPage]:
        """Normalize OpenPSS and doc-split-benchmark page-stream row schemas."""
        if not rows:
            raise ValueError("Cannot normalize an empty stream")

        boundary_field = self._stream_boundary_field(rows[0])
        text_field = self._stream_text_field(rows[0])
        ordered = sorted(rows, key=lambda row: row["position"])
        stream_id = str(ordered[0].get("stream_id", "stream_0"))

        return [
            NormalizedStreamPage(
                stream_id=stream_id,
                page_number=idx,
                original_position=int(row["position"]),
                text=str(row.get(text_field) or ""),
                starts_new_document=bool(row.get(boundary_field) == 1),
                image=row.get("image"),
                source=row.get("source"),
            )
            for idx, row in enumerate(ordered, start=1)
        ]

    def stream_rows_to_annotation(self, rows: list[dict[str, Any]]) -> PacketAnnotation:
        """Convert normalized page-stream rows into one packet annotation.

        OpenPSS uses ``label`` and ``text`` with 1-based positions. The benchmark
        uses ``boundary`` and ``page_text`` with 0-based positions. Both are
        converted to 1-based page order before grouping.
        """
        pages = self.normalize_stream_rows(rows)
        groups: list[list[int]] = []
        current: list[int] = []

        for page in pages:
            if page.starts_new_document and current:
                groups.append(current)
                current = []
            current.append(page.page_number)

        if current:
            groups.append(current)

        return PacketAnnotation(
            packet_id=pages[0].stream_id,
            page_count=len(pages),
            groups=groups,
            document_types=["unknown"] * len(groups),
        )

    def stream_rows_to_pages(self, rows: list[dict[str, Any]]) -> list[Any]:
        """Convert stream rows into internal PageRepresentation objects."""
        from document_intelligence.ingestion.page_extractor import PageRepresentation

        pages = []
        for page in self.normalize_stream_rows(rows):
            pages.append(
                PageRepresentation(
                    page_id=f"{page.stream_id}_page_{page.page_number:04d}",
                    page_number=page.page_number,
                    text=page.text,
                    metadata={
                        "stream_id": page.stream_id,
                        "original_position": page.original_position,
                        "starts_new_document": page.starts_new_document,
                        "source": page.source,
                    },
                )
            )
        return pages

    def build_page_pair_examples(
        self,
        annotation: PacketAnnotation,
        feature_builder: Any,
        pages: list[Any] | None = None,
    ) -> list[PagePairTrainingExample]:
        examples: list[PagePairTrainingExample] = []
        page_to_doc = {}
        for gi, group in enumerate(annotation.groups):
            for p in group:
                page_to_doc[p] = gi

        if pages is None:
            from document_intelligence.ingestion.page_extractor import PageRepresentation

            pages = [
                PageRepresentation(
                    page_id=f"{annotation.packet_id}_page_{i:04d}",
                    page_number=i,
                    text=f"page {i} content",
                )
                for i in range(1, annotation.page_count + 1)
            ]

        for i in range(len(pages) - 1):
            pa, pb = pages[i], pages[i + 1]
            pair = feature_builder.build_pair(pa, pb)
            same_doc = page_to_doc.get(pa.page_number) == page_to_doc.get(pb.page_number)
            label = 1 if same_doc else 0
            examples.append(
                PagePairTrainingExample(
                    packet_id=annotation.packet_id,
                    page_a=pa.page_number,
                    page_b=pb.page_number,
                    features=pair.features.tolist(),
                    label=label,
                )
            )
        return examples

    def save_manifest(self, path: Path | str) -> None:
        schema = self.inspect_schema()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"dataset": self.dataset_name, "config": self.dataset_config, "schema": schema}, f, indent=2)
