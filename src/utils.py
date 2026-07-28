"""
utils.py
--------
Shared helper utilities: logging setup and directory helpers.
"""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Create (or fetch) a configured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)