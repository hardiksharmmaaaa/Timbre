"""Extract cached features for every accepted canonical-manifest clip.

Command: python -m ml.training.extract_features
Artifacts: data/processed/<dataset>/<clip_id>.npz
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ml.config import EmotionState, get_logger, parse_ravdess_fields, settings
from ml.core.features.extractor import feature_contract, load_and_extract
from ml.data.manifest import read_manifest

log = get_logger("extract")


def find_ravdess_clips(raw_dir: Path) -> list[tuple[Path, EmotionState]]:
    clips = []
    for wav in sorted(raw_dir.rglob("*.wav")):
        try:
            fields = parse_ravdess_fields(wav.name)
        except ValueError:
            continue
        if fields[:2] != (3, 1):
            continue  # Audio-only speech; select by metadata, independent of archive layout.
        clips.append((wav, EmotionState(fields[2])))
    return clips


def feature_path(row: dict) -> Path:
    return settings.processed_dir / row["dataset"] / f"{row['clip_id']}.npz"


def cache_matches(path: Path, row: dict, contract_json: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            return (
                str(data["audio_sha256"].item()) == row["audio_sha256"]
                and str(data["feature_contract"].item()) == contract_json
            )
    except (OSError, KeyError, ValueError):
        return False


def extract_row(row: dict, contract_json: str) -> tuple[str, str]:
    """Idempotently extract one manifest row and atomically publish its cache."""
    output = feature_path(row)
    if cache_matches(output, row, contract_json):
        return row["clip_id"], "cached"
    source = settings.raw_dir / row["relative_audio_path"]
    features, sample_rate = load_and_extract(str(source), sr=settings.sample_rate)
    if not np.isfinite(features).all() or features.shape[0] == 0:
        raise ValueError(f"non-finite or empty features for {row['clip_id']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=features,
            sr=sample_rate,
            label=row["label_id"],
            clip_id=row["clip_id"],
            dataset=row["dataset"],
            split=row["split"],
            audio_sha256=row["audio_sha256"],
            feature_contract=contract_json,
        )
    temporary.replace(output)
    return row["clip_id"], "written"


def process_manifest(workers: int = 1, selected_splits: set[str] | None = None) -> dict:
    manifest_path = settings.manifest_dir / "all_clips.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("canonical manifest missing; run `make data-prepare` first")
    rows = read_manifest(manifest_path)
    if selected_splits:
        rows = [row for row in rows if row["split"] in selected_splits]
    contract_json = json.dumps(feature_contract(), sort_keys=True)
    counts = {"written": 0, "cached": 0}
    if workers <= 1:
        results = (extract_row(row, contract_json) for row in rows)
        for index, (_, status) in enumerate(results, start=1):
            counts[status] += 1
            if index % 100 == 0:
                log.info("processed %d/%d clips", index, len(rows))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(lambda row: extract_row(row, contract_json), rows)
            for index, (_, status) in enumerate(results, start=1):
                counts[status] += 1
                if index % 100 == 0:
                    log.info("processed %d/%d clips", index, len(rows))
    counts["total"] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--splits",
        default="train,val,test,external_test",
        help="comma-separated manifest splits to extract",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    selected = {value.strip() for value in args.splits.split(",") if value.strip()}
    counts = process_manifest(args.workers, selected)
    log.info("feature cache complete: %s", counts)


if __name__ == "__main__":
    main()
