"""Build the quality-screened canonical manifest and frozen speaker-disjoint splits."""

from __future__ import annotations

import argparse
import json

from timbre.data.manifest import prepare_manifest
from timbre.utils.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    summary = prepare_manifest(
        raw_dir=settings.raw_dir,
        manifest_dir=settings.manifest_dir,
        splits_dir=settings.splits_dir,
        seed=settings.seed,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
