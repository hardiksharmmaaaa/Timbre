"""Train the perception pipeline: reduction + per-state HMMs, save to models/.

Command: python -m scripts.train
Artifacts: models/reducer.joblib, models/state_models.joblib, models/meta.json
"""

from __future__ import annotations

import json

import numpy as np

from timbre.reduction.reducer import fit_reduction, reduce_sequences
from timbre.utils.common import get_logger, set_seed
from timbre.utils.config import settings

log = get_logger("train")


def load_train_sequences(split_file) -> list[tuple[int, np.ndarray]]:
    """Load (label, reduced_features) for clips in the train split."""
    import json as _json
    from pathlib import Path

    splits = _json.loads(Path(split_file).read_text())
    out = []
    for clip_id, label in splits["train"].items():
        npz = settings.processed_dir / "RAVDESS" / f"{clip_id}.npz"
        with np.load(npz) as d:
            out.append((int(label), d["features"]))
    return out


def main() -> None:
    set_seed(settings.seed)
    train = load_train_sequences(settings.splits_dir / "ravdess_splits.json")

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

    import joblib

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer, settings.model_dir / "reducer.joblib")
    joblib.dump(state_models, settings.model_dir / "state_models.joblib")
    (settings.model_dir / "meta.json").write_text(
        json.dumps(
            {
                "state_ids": state_ids,
                "reduction_method": settings.reduction_method,
                "hmm_n_states": settings.hmm_n_states,
                "class_score": "length_normalized_viterbi",
            }
        )
    )
    log.info("saved model artifacts to %s", settings.model_dir)


if __name__ == "__main__":
    main()
