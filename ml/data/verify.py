"""Fail unless the canonical manifest, frozen splits, and all feature caches are ready."""

from __future__ import annotations

import argparse
import json

from ml.config import settings
from ml.data.cache import (
    load_augmented_training_clips,
    manifest_sha256,
    validate_cache_coverage,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augmented", action="store_true")
    args = parser.parse_args()
    manifest_path = settings.manifest_dir / "all_clips.csv"
    split_path = settings.splits_dir / "six_class_v1.json"
    if not manifest_path.exists() or not split_path.exists():
        raise FileNotFoundError("run `make data-prepare` before verifying training data")
    split_document = json.loads(split_path.read_text())
    digest = manifest_sha256(manifest_path)
    if split_document.get("manifest_sha256") != digest:
        raise ValueError("split file does not match the current canonical manifest")
    counts = validate_cache_coverage(manifest_path, settings.processed_dir)
    required = {"train", "val", "test", "external_test"}
    if not required <= counts.keys() or any(counts[name] == 0 for name in required):
        raise ValueError(f"required split is empty: {counts}")
    result = {"manifest_sha256": digest, "cached_clips": counts}
    if args.augmented:
        augmented = load_augmented_training_clips(manifest_path, settings.augmentation_dir)
        result["augmented_training_clips"] = len(augmented)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
