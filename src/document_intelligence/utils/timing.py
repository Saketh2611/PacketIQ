"""Timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class TimerResult:
    elapsed_seconds: float = 0.0
    name: str = ""


@contextmanager
def timer(name: str = "") -> Generator[TimerResult, None, None]:
    result = TimerResult(name=name)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start
