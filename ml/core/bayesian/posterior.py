"""Stage 4 — Bayesian uncertainty layer. Built by us.

Instead of the HMM returning a single argmax, we estimate a *posterior distribution* over
emotional states with uncertainty, producing e.g. "72% stressed, 20% tired, ±8%".

These are uncertainty surrogates, not inference over HMM parameter uncertainty:
  - "hybrid": tempered softmax with heuristic interval widths.
  - "vi":     iterative responsibility surrogate with heuristic interval widths.
  - "mcmc":   compatibility name for independent samples from a Dirichlet surrogate,
              using normalized scores as pseudo-observation weights.

The sampling path is neither a Markov chain nor Gibbs inference. Its intervals describe
only the specified surrogate; no method here establishes calibration on held-out speech.

At inference the API uses whichever `settings.bayesian_method` selects.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np

from ml.config import EmotionPosterior, EmotionState, settings


def _normalise(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def posterior_from_log_likelihoods(
    log_likelihoods: dict[EmotionState, float],
    n_frames: int,
    method: str | None = None,
) -> EmotionPosterior:
    """Hand-implemented uncertainty surrogates over per-state log evidence.

    Args:
        log_likelihoods: mapping state -> log P(obs | state).
        n_frames: positive integer number of frames, used as pseudo-observation count.
        method: "hybrid" | "vi" | "mcmc" (independent Dirichlet sampling alias).

    Returns:
        EmotionPosterior with probabilities and surrogate uncertainty half-widths.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, Integral) or n_frames <= 0:
        raise ValueError("n_frames must be a positive integer")
    method = (method or settings.bayesian_method).lower()
    states = list(log_likelihoods.keys())
    logits = np.array([log_likelihoods[s] for s in states], dtype=float)
    if (
        logits.size == 0
        or np.isnan(logits).any()
        or np.isposinf(logits).any()
        or not np.isfinite(logits).any()
    ):
        raise ValueError("log likelihoods must include finite evidence and no NaN or +inf")

    if method == "hybrid":
        return _hybrid(states, logits, n_frames)
    if method == "vi":
        return _vi(states, logits, n_frames)
    if method == "mcmc":
        return _dirichlet_surrogate(states, logits, n_frames)
    raise ValueError(f"unknown bayesian_method {method!r}")


def _hybrid(states, logits, n_frames) -> EmotionPosterior:
    """Softmax over log-likelihoods with a confidence scaling by evidence size.

    Its sharpness is tempered when fewer frames are available. This does not establish
    calibration, and the returned widths are heuristic rather than empirical intervals.
    """
    # Temperature rises (flatter posterior) when evidence is thin.
    temp = 1.0 + 0.5 * np.exp(-n_frames / 40.0)
    probs = _normalise(logits / temp)
    return _build_posterior(states, probs, n_frames)


def _vi(states, logits, n_frames) -> EmotionPosterior:
    """Mean-field variational approximation.

    Treat responsibilities as per-state latent variables; a few coordinate-ascent updates over
    the softmax keep it light while yielding intervals via a surrogate beta-scale dispersion.
    """
    probs = _normalise(logits)
    for _ in range(3):  # few mean-field sweeps; converges fast for a categorical
        probs = _normalise(logits + np.log(probs + 1e-300) * 0.1)
    return _build_posterior(states, probs, n_frames)


def _dirichlet_surrogate(states, logits, n_frames, n_samples: int = 4000) -> EmotionPosterior:
    """Library-backed independent sampling from a documented Dirichlet surrogate.

    A uniform Dirichlet prior receives ``n_frames * softmax(logits)`` pseudo-counts.
    These weights are derived from class scores; frames are not observed emotion labels
    or assumed to be independently labelled data. Concentration is K + n_frames.

    Draw probability vectors directly: no Markov chain or burn-in is required. Summarize
    the 2.5th/97.5th percentiles with a symmetric half-width that encloses both quantiles
    about the sample mean. These are surrogate intervals, not calibrated coverage claims.
    """
    rng = np.random.default_rng(settings.seed)
    alpha = 1.0 + n_frames * _normalise(logits)
    draws = rng.dirichlet(alpha, size=n_samples)
    probs = draws.mean(axis=0)
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    interval = np.maximum(probs - lower, upper - probs)
    return EmotionPosterior(
        probs={s: float(p) for s, p in zip(states, probs)},
        intervals={s: float(i) for s, i in zip(states, interval)},
    )


def _build_posterior(states: Iterable[EmotionState], probs: np.ndarray,
                     n_frames: int) -> EmotionPosterior:
    # Credible half-width: grows with uncertainty, shrinks with evidence (simple heuristic
    # consistent with posterior concentration for a categorical).
    order = np.sort(probs)
    dispersion = 1.0 - (order[-1] - order[-2] if len(order) > 1 else order[-1])
    half = dispersion * 0.5 * np.sqrt(1.0 / max(n_frames, 1))
    return EmotionPosterior(
        probs={s: float(p) for s, p in zip(states, probs)},
        intervals={s: half for s in states},
    )
