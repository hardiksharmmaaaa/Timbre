"""Create a separate deterministic reverberated feature cache for training clips only."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np

from timbre.data.manifest import read_manifest
from timbre.features.augmentation import apply_reverb, select_rir
from timbre.features.extractor import extract_frame_features, feature_contract
from timbre.utils.common import get_logger
from timbre.utils.config import settings

log = get_logger("augment")
AUGMENTATION_VERSION = "rirs_reverb_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_path(row: dict) -> Path:
    return settings.augmentation_dir / row["dataset"] / f"{row['clip_id']}__reverb.npz"


def _extract_row(row: dict, rir_paths: list[Path], contract_json: str) -> str:
    output = _output_path(row)
    rir_path = select_rir(rir_paths, row["clip_id"], settings.seed)
    rir_relative = rir_path.relative_to(settings.raw_dir).as_posix()
    rir_digest = _sha256(rir_path)
    augmentation = json.dumps(
        {
            "version": AUGMENTATION_VERSION,
            "rir_path": rir_relative,
            "wet_ratio": 0.35,
            "seed": settings.seed,
        },
        sort_keys=True,
    )
    if output.exists():
        try:
            with np.load(output) as data:
                if (
                    str(data["audio_sha256"].item()) == row["audio_sha256"]
                    and str(data["rir_sha256"].item()) == rir_digest
                    and str(data["feature_contract"].item()) == contract_json
                    and str(data["augmentation"].item()) == augmentation
                ):
                    return "cached"
        except (OSError, KeyError, ValueError):
            pass

    audio_path = settings.raw_dir / row["relative_audio_path"]
    audio, _ = librosa.load(audio_path, sr=settings.sample_rate, mono=True)
    impulse, _ = librosa.load(rir_path, sr=settings.sample_rate, mono=True)
    augmented = apply_reverb(audio, impulse)
    features = extract_frame_features(augmented, settings.sample_rate)
    if features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError(f"invalid augmented features for {row['clip_id']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=features,
            label=row["label_id"],
            clip_id=row["clip_id"],
            dataset=row["dataset"],
            split="train",
            audio_sha256=row["audio_sha256"],
            rir_sha256=rir_digest,
            feature_contract=contract_json,
            augmentation=augmentation,
        )
    temporary.replace(output)
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    manifest_path = settings.manifest_dir / "all_clips.csv"
    rows = [row for row in read_manifest(manifest_path) if row["split"] == "train"]
    rir_root = settings.raw_dir / "augmentation" / "RIRS_NOISES"
    rir_paths = sorted(rir_root.rglob("simulated_rirs/**/*.wav"))
    if not rir_paths:
        raise FileNotFoundError("RIRS_NOISES simulated room responses are missing")
    contract_json = json.dumps(feature_contract(), sort_keys=True)
    counts = {"written": 0, "cached": 0}
    if args.workers <= 1:
        statuses = (_extract_row(row, rir_paths, contract_json) for row in rows)
        for index, status in enumerate(statuses, start=1):
            counts[status] += 1
            if index % 100 == 0:
                log.info("augmented %d/%d training clips", index, len(rows))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            statuses = pool.map(
                lambda row: _extract_row(row, rir_paths, contract_json), rows
            )
            for index, status in enumerate(statuses, start=1):
                counts[status] += 1
                if index % 100 == 0:
                    log.info("augmented %d/%d training clips", index, len(rows))
    log.info("augmentation cache complete: %s", {**counts, "total": len(rows)})


if __name__ == "__main__":
    main()
