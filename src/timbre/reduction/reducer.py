"""Stage 2 — PCA / LLE dimensionality reduction. Built by us.

Compresses the 20-30 raw features per frame to `n_components` (default 8). PCA assumes a
linear latent manifold; LLE does not. Comparing their emotion-cluster separation is a
documented research point (see DESIGN.md §5.1).

IMPORTANT leakage guard: fit reduction on *training frames only*. Never fit on val/test.
"""

from __future__ import annotations

import numpy as np

from timbre.utils.config import settings

SUPPORTED = ("pca", "lle")


def fit_reduction(frames: np.ndarray, method: str = "pca",
                  n_components: int | None = None):
    """Fit a reducer on training frames.

    Args:
        frames: concatenated (N_train_frames, n_features) training matrix.
        method: "pca" or "lle".
        n_components: target dimensionality.

    Returns:
        A fitted object exposing `.transform(X_2d) -> reduced`.
    """
    method = (method or settings.reduction_method).lower()
    n_components = n_components or settings.n_components
    if method not in SUPPORTED:
        raise ValueError(f"method must be one of {SUPPORTED}, got {method!r}")

    if method == "pca":
        # Library-backed linear reduction.
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, random_state=settings.seed)
    else:
        # Library-backed non-linear reduction.
        from sklearn.manifold import LocallyLinearEmbedding

        reducer = LocallyLinearEmbedding(
            n_components=n_components, n_neighbors=8, random_state=settings.seed
        )

    # Acoustic features have incompatible units (Hz, energy, MFCCs, and ratios). Fit scaling
    # on the same training-only frames as the reducer to prevent both scale bias and leakage.
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([("scaler", StandardScaler()), (method, reducer)])
    pipeline.fit(frames)
    return pipeline


def reduce_sequences(sequences: list[np.ndarray], reducer) -> list[np.ndarray]:
    """Transform a list of per-clip feature matrices through a fitted reducer.

    Each input is (nframes_i, n_features); returns matching (nframes_i, n_components).
    (At inference a single clip is passed as a one-element list.)
    """
    return [np.asarray(reducer.transform(seq)) for seq in sequences]
