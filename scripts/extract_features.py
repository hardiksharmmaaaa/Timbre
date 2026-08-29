"""Extract framewise features from raw audio -> data/processed/<dataset>/<id>.npz.

Command: python -m scripts.extract_features
Skips clips already processed (cache); run once per dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from timbre.core.types import EmotionState
from timbre.features.extractor import load_and_extract
from timbre.utils.common import get_logger, parse_ravdess_emotion
from timbre.utils.config import settings

log = get_logger("extract")


def find_ravdess_clips(raw_dir: Path) -> list[tuple[Path, EmotionState]]:
    clips = []
    for wav in sorted(raw_dir.rglob("*.wav")):
        if "Audio_Speech" not in str(wav):
            continue  # only speech audio, skip song/general files
        try:
            label = parse_ravdess_emotion(wav.name)
        except ValueError:
            continue
        clips.append((wav, EmotionState(label)))
    return clips


def process(dataset: str, raw_dir: Path) -> None:
    out_dir = settings.processed_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = find_ravdess_clips(raw_dir)
    log.info("found %d %s clips", len(clips), dataset)
    for i, (wav, label) in enumerate(clips):
        out = out_dir / f"{wav.stem}.npz"
        if out.exists():
            continue
        feats, sr = load_and_extract(str(wav))
        np.savez(out, features=feats, sr=sr, label=int(label))
        if i % 100 == 0:
            log.info("processed %d/%d", i, len(clips))


def main() -> None:
    process("RAVDESS", settings.raw_dir / "RAVDESS")


if __name__ == "__main__":
    main()
