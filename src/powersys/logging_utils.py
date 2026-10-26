"""Console + file logging shared by every stage of the pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "powersys"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s %(name)s  %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(suffix: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if suffix is None else f"{LOGGER_NAME}.{suffix}")
