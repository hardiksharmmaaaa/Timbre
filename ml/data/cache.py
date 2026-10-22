"""Validated access to feature caches governed by the canonical manifest."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.data.manifest import read_manifest


@dataclass(frozen=True)
class CachedClip:
    clip_id: str
    dataset: str
    speaker_id: str
    label: int
    split: str
    audio_sha256: str
    features: np.ndarray


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_path(processed_dir: Path, row: dict) -> Path:
    return processed_dir / row["dataset"] / f"{row['clip_id']}.npz"


def load_cached_clips(
    manifest_path: Path,
    processed_dir: Path,
    splits: set[str],
    datasets: set[str] | None = None,
) -> list[CachedClip]:
    """Load selected cached clips and reject missing, stale, or malformed artifacts."""
    selected = [
        row
        for row in read_manifest(manifest_path)
        if row["split"] in splits and (datasets is None or row["dataset"] in datasets)
    ]
    if not selected:
        raise ValueError(f"manifest has no clips for splits {sorted(splits)}")
    clips = []
    missing = []
    for row in selected:
        path = cache_path(processed_dir, row)
        if not path.exists():
            missing.append(row["clip_id"])
            continue
        try:
            with np.load(path) as data:
                features = np.asarray(data["features"])
                cached_hash = str(data["audio_sha256"].item())
                cached_label = int(data["label"].item())
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"invalid feature cache: {path}") from exc
        if cached_hash != row["audio_sha256"]:
            raise ValueError(f"stale feature cache for {row['clip_id']}")
        if cached_label != row["label_id"]:
            raise ValueError(f"label mismatch in feature cache for {row['clip_id']}")
        if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
            raise ValueError(f"invalid feature matrix for {row['clip_id']}")
        clips.append(
            CachedClip(
                clip_id=row["clip_id"],
                dataset=row["dataset"],
                speaker_id=row["speaker_id"],
                label=row["label_id"],
                split=row["split"],
                audio_sha256=row["audio_sha256"],
                features=features,
            )
        )
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} feature caches are missing (first: {preview}); run `make data-extract`"
        )
    return clips


def validate_cache_coverage(
    manifest_path: Path, processed_dir: Path
) -> dict[str, int]:
    """Validate all manifest rows and return cache counts by split."""
    rows = read_manifest(manifest_path)
    splits = {row["split"] for row in rows}
    clips = load_cached_clips(manifest_path, processed_dir, splits)
    counts = {split: 0 for split in sorted(splits)}
    for clip in clips:
        counts[clip.split] += 1
    return counts


def load_augmented_training_clips(
    manifest_path: Path, augmentation_dir: Path
) -> list[CachedClip]:
    """Load exactly one current augmentation for every frozen training clip."""
    rows = [row for row in read_manifest(manifest_path) if row["split"] == "train"]
    clips = []
    missing = []
    for row in rows:
        path = augmentation_dir / row["dataset"] / f"{row['clip_id']}__reverb.npz"
        if not path.exists():
            missing.append(row["clip_id"])
            continue
        try:
            with np.load(path) as data:
                features = np.asarray(data["features"])
                cached_hash = str(data["audio_sha256"].item())
                cached_label = int(data["label"].item())
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"invalid augmented feature cache: {path}") from exc
        if cached_hash != row["audio_sha256"] or cached_label != row["label_id"]:
            raise ValueError(f"stale augmented feature cache for {row['clip_id']}")
        if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
            raise ValueError(f"invalid augmented features for {row['clip_id']}")
        clips.append(
            CachedClip(
                clip_id=f"{row['clip_id']}__reverb",
                dataset=row["dataset"],
                speaker_id=row["speaker_id"],
                label=row["label_id"],
                split="train",
                audio_sha256=row["audio_sha256"],
                features=features,
            )
        )
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} augmented caches are missing (first: {preview}); "
            "run `make data-augment`"
        )
    return clips
