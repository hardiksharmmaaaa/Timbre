"""Seed + logging helpers shared across the perception pipeline."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Deterministically seed python, numpy (and sklearn if imported)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import sklearn  # noqa: F401
        from sklearn.utils import check_random_state  # noqa: F401
    except ImportError:  # pragma: no cover
        pass


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def parse_ravdess_fields(filename: str) -> tuple[int, ...]:
    """Validate and return the seven encoded RAVDESS filename fields."""
    fields = Path(filename).stem.split("-")
    limits = (3, 2, 8, 2, 2, 2, 24)
    if len(fields) != len(limits) or any(
        len(field) != 2 or not field.isascii() or not field.isdigit() for field in fields
    ):
        raise ValueError(f"invalid RAVDESS filename: {filename!r}")
    values = tuple(int(field) for field in fields)
    if any(not 1 <= value <= limit for value, limit in zip(values, limits)):
        raise ValueError(f"invalid RAVDESS filename fields: {filename!r}")
    return values


def parse_ravdess_emotion(filename: str) -> int:
    """RAVDESS filenames: 03-01-06-... -> 3rd token (01..08) is the emotion.

    01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
    """
    return parse_ravdess_fields(filename)[2]


def parse_ravdess_actor(filename: str) -> int:
    """Return the actor ID (seventh RAVDESS filename field)."""
    return parse_ravdess_fields(filename)[6]
