"""Resource usage metrics."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass
class ResourceMetrics:
    wall_clock_seconds: float
    peak_rss_mb: float
    cpu_percent: float
    model_size_mb: float | None = None
    index_size_mb: float | None = None


def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def file_size_mb(path: Path | str) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def measure_resources(
    elapsed: float,
    index_path: Path | str | None = None,
    model_cache_dir: Path | str | None = None,
) -> ResourceMetrics:
    return ResourceMetrics(
        wall_clock_seconds=elapsed,
        peak_rss_mb=get_process_memory_mb(),
        cpu_percent=psutil.cpu_percent(interval=0.1),
        model_size_mb=file_size_mb(model_cache_dir) if model_cache_dir else None,
        index_size_mb=file_size_mb(index_path) if index_path else None,
    )
