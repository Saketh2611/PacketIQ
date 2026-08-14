"""Structured logging utilities."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from document_intelligence.config.settings import get_settings


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **fields: Any,
) -> None:
    record = logger.makeRecord(logger.name, level, "", 0, message, (), None)
    record.extra_fields = fields
    logger.handle(record)


class StructuredLogger:
    """Logger wrapper accepting structured keyword fields."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(StructuredFormatter())
            self._logger.addHandler(handler)
            settings = get_settings()
            self._logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
            self._logger.propagate = False

    def debug(self, message: str, **fields: Any) -> None:
        log_event(self._logger, logging.DEBUG, message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        log_event(self._logger, logging.INFO, message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        log_event(self._logger, logging.WARNING, message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        log_event(self._logger, logging.ERROR, message, **fields)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)
