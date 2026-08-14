#!/usr/bin/env python3
"""Download the configured page-stream dataset and save local manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.config.settings import get_settings
from document_intelligence.dataset.adapter import DocSplitAdapter
from document_intelligence.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download configured page-stream dataset")
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset name")
    parser.add_argument("--config", default=None, help="Optional Hugging Face dataset config")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output", default=None, help="Manifest output path")
    args = parser.parse_args()

    settings = get_settings()
    adapter = DocSplitAdapter(dataset_name=args.dataset, dataset_config=args.config)
    try:
        adapter.load_dataset(args.split)
    except Exception as exc:
        logger.warning(f"Could not load dataset from HuggingFace: {exc}")
        logger.info("Creating placeholder manifest for offline development")
        manifest_path = Path(args.output or settings.processed_dir / "dataset_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with manifest_path.open("w") as f:
            json.dump(
                {
                    "dataset": adapter.dataset_name,
                    "config": adapter.dataset_config,
                    "status": "offline_placeholder",
                    "message": str(exc),
                },
                f,
                indent=2,
            )
        print(f"Saved placeholder manifest to {manifest_path}")
        return

    schema = adapter.inspect_schema()
    manifest_path = Path(args.output or settings.processed_dir / "dataset_manifest.json")
    adapter.save_manifest(manifest_path)
    print(f"Dataset loaded: {schema['num_samples']} samples")
    print(f"Fields: {schema['fields']}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
