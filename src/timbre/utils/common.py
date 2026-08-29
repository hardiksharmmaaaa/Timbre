"""Seed + logging helpers shared across the perception pipeline."""

from __future__ import annotations

import logging
import random

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


def parse_ravdess_emotion(filename: str) -> int:
    """RAVDESS filenames: 03-01-06-... -> 3rd token (01..08) is the emotion.

    01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
    """
    stem = filename.split(".")[0]
    return int(stem.split("-")[2])


def parse_ravdess_actor(filename: str) -> int:
    """Return the actor ID (seventh RAVDESS filename field)."""
    stem = filename.split(".")[0]
    fields = stem.split("-")
    if len(fields) != 7:
        raise ValueError(f"invalid RAVDESS filename: {filename!r}")
    actor = int(fields[6])
    if not 1 <= actor <= 24:
        raise ValueError(f"invalid RAVDESS actor ID: {actor}")
    return actor
