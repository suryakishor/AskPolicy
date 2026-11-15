"""Application logging setup.

A single ``configure_logging`` call gives the API and the ingestion CLI a
consistent format. Logs are intentionally written to stderr/stdout only; this
project does not ship log files or a log aggregation service.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once.

    ``force=True`` replaces handlers installed by Uvicorn (or by a previous
    call) so log lines do not appear twice during development reloads.
    """
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )

    # Third-party libraries are useful at DEBUG level only when explicitly
    # requested; otherwise they are noisy during normal operation.
    for noisy_logger in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
