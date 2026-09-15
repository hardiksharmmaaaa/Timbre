"""Create deterministic (seeded) train/val/test splits at the CLIP level.

Command: python -m scripts.make_splits
Writes data/splits/<dataset>_<split>.json as lists of clip ids + labels.
Never split at frame level (leakage guard) and never let a clip appear in >1 split.
"""

from __future__ import annotations

import json

import numpy as np

from timbre.utils.common import get_logger, parse_ravdess_actor, set_seed
from timbre.utils.config import settings

log = get_logger("split")


def load_features(dataset: str) -> list[tuple[str, int, int]]:
    """Return ``(clip_id, label, actor_id)`` for one processed dataset."""
    dirp = settings.processed_dir / dataset
    clips = []
    for npz in sorted(dirp.glob("*.npz")):
        with np.load(npz) as data:
            clips.append((npz.stem, int(data["label"]), parse_ravdess_actor(npz.name)))
    return clips


def split_by_actor(
    clips: list[tuple[str, int, int]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """Build deterministic speaker-independent splits with no actor leakage."""
    if not clips:
        raise ValueError("cannot split an empty dataset")
    if val_frac <= 0 or test_frac <= 0 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be positive and sum to less than one")

    actors = np.array(sorted({actor for _, _, actor in clips}))
    if actors.size < 3:
        raise ValueError("at least three actors are required for train/val/test splits")
    rng = np.random.default_rng(seed)
    rng.shuffle(actors)
    # Rounded holdouts must leave at least one actor in every partition.
    n_test = min(max(1, round(actors.size * test_frac)), actors.size - 2)
    n_val = min(max(1, round(actors.size * val_frac)), actors.size - n_test - 1)
    test_actors = set(actors[:n_test].tolist())
    val_actors = set(actors[n_test:n_test + n_val].tolist())
    train_actors = set(actors[n_test + n_val:].tolist())

    actor_sets = {"train": train_actors, "val": val_actors, "test": test_actors}
    splits: dict[str, dict] = {
        "metadata": {
            "seed": seed,
            "strategy": "speaker-independent",
            "actors": {name: sorted(group) for name, group in actor_sets.items()},
        }
    }
    for name, actor_group in actor_sets.items():
        splits[name] = {
            clip_id: label
            for clip_id, label, actor in clips
            if actor in actor_group
        }
    return splits


def make_splits(dataset: str, val_frac: float = 0.15, test_frac: float = 0.15) -> None:
    set_seed(settings.seed)
    clips = load_features(dataset)
    splits = split_by_actor(clips, val_frac, test_frac, settings.seed)
    out = settings.splits_dir / f"{dataset.lower()}_splits.json"
    out.write_text(json.dumps(splits, indent=2))
    log.info(
        "wrote %s: %d/%d/%d clips across disjoint actors",
        out,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )


def main() -> None:
    make_splits("RAVDESS")


if __name__ == "__main__":
    main()
