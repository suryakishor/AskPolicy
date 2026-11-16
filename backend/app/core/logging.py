"""Logging setup shared by the API and the ingestion CLI."""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once, replacing existing handlers."""
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )

    # Keep third-party libraries quiet unless debugging.
    for noisy_logger in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
