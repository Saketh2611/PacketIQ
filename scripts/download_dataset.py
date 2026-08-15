#!/usr/bin/env python3
"""Download configured page-stream datasets and save local manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document_intelligence.config.settings import get_settings
from document_intelligence.dataset.adapter import DocSplitAdapter


def download_one(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    output: Path,
) -> bool:
    adapter = DocSplitAdapter(dataset_name=dataset_name, dataset_config=dataset_config)
    try:
        adapter.load_dataset(split)
        schema = adapter.inspect_schema()
        adapter.save_manifest(output)
        print(f"  {dataset_name} [{dataset_config or 'default'}] split={split}")
        print(f"    rows={schema['num_samples']} fields={schema['fields']}")
        print(f"    manifest -> {output}")
        return True
    except Exception as exc:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "config": dataset_config,
                    "split": split,
                    "status": "error",
                    "message": str(exc),
                },
                f,
                indent=2,
            )
        print(f"  FAILED {dataset_name}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download configured page-stream datasets")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    processed = settings.processed_dir
    ok = True

    if not args.test_only:
        print("Downloading train/dev dataset...")
        ok &= download_one(
            settings.dataset_name,
            settings.dataset_config,
            "train",
            processed / "train_manifest.json",
        )

    if not args.train_only:
        print("Downloading test/eval dataset...")
        ok &= download_one(
            settings.test_dataset_name,
            settings.test_dataset_config,
            "test",
            processed / "test_manifest.json",
        )

    if ok:
        print("\nAll manifests saved under", processed)
    else:
        print("\nSome downloads failed. Check manifests for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
