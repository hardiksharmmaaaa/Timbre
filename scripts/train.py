"""Train the pooled six-class reduction + per-emotion HMM pipeline.

Command: python -m scripts.train
Artifacts: models/reducer.joblib, models/state_models.joblib, models/meta.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from timbre.data.cache import (
    load_augmented_training_clips,
    load_cached_clips,
    manifest_sha256,
)
from timbre.features.extractor import feature_contract
from timbre.reduction.reducer import fit_reduction, reduce_sequences
from timbre.utils.common import get_logger, set_seed
from timbre.utils.config import ROOT, settings
from timbre.utils.repro import environment_metadata, git_commit

log = get_logger("train")


def _atomic_joblib_dump(value, path: Path) -> None:
    import joblib

    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def train_pipeline(model_dir: Path, include_augmented: bool = False) -> None:
    set_seed(settings.seed)
    manifest_path = settings.manifest_dir / "all_clips.csv"
    train_clips = load_cached_clips(manifest_path, settings.processed_dir, {"train"})
    original_clip_count = len(train_clips)
    if include_augmented:
        train_clips.extend(
            load_augmented_training_clips(manifest_path, settings.augmentation_dir)
        )
    train = [(clip.label, clip.features) for clip in train_clips]

    # Concatenate ALL training frames to fit reduction (leakage-safe: train frames only).
    all_frames = np.concatenate([f for _, f in train], axis=0)
    log.info("fitting reduction (%s, n=%d) on %d frames",
             settings.reduction_method, settings.n_components, all_frames.shape[0])
    reducer = fit_reduction(all_frames, method=settings.reduction_method)
    reduced = reduce_sequences([f for _, f in train], reducer)

    # Train one HMM per emotion. Hidden states model acoustic regimes within that emotion.
    from timbre.hmm.decoder import train_hmm

    state_ids = sorted({lbl for lbl, _ in train})
    state_models = {}
    labels = [lb for lb, _ in train]
    for sid in state_ids:
        seqs = [r for (lb, r) in zip(labels, reduced) if lb == sid]
        if not seqs:
            continue
        log.info("training HMM for state %d on %d clips", sid, len(seqs))
        state_models[sid] = train_hmm(seqs)

    model_dir.mkdir(parents=True, exist_ok=True)
    _atomic_joblib_dump(reducer, model_dir / "reducer.joblib")
    _atomic_joblib_dump(state_models, model_dir / "state_models.joblib")
    label_counts = Counter(clip.label for clip in train_clips)
    dataset_counts = Counter(clip.dataset for clip in train_clips)
    metadata = {
        "schema_version": 2,
        "model_family": "per-emotion-diagonal-gaussian-hmm",
        "state_ids": state_ids,
        "reduction_method": settings.reduction_method,
        "n_components": settings.n_components,
        "hmm_n_states": settings.hmm_n_states,
        "hmm_n_iter": settings.hmm_n_iter,
        "class_score": "length_normalized_viterbi",
        "seed": settings.seed,
        "manifest_sha256": manifest_sha256(manifest_path),
        "feature_contract": feature_contract(),
        "training_clip_count": len(train_clips),
        "original_training_clip_count": original_clip_count,
        "augmented_training_clip_count": len(train_clips) - original_clip_count,
        "augmentation": "rirs_reverb_v1" if include_augmented else None,
        "training_counts_by_dataset": dict(sorted(dataset_counts.items())),
        "training_counts_by_label_id": {
            str(key): value for key, value in sorted(label_counts.items())
        },
        "git_commit": git_commit(ROOT),
        "environment": environment_metadata(),
    }
    (model_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    log.info("saved model artifacts to %s", model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-augmented", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=settings.model_dir)
    args = parser.parse_args()
    train_pipeline(args.model_dir, include_augmented=args.include_augmented)


if __name__ == "__main__":
    main()
