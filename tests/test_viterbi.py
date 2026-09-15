"""Correctness tests for the hand-written Viterbi decoder against a tiny known HMM.

We construct a 2-state / 2-observation HMM by hand, compute the best path by brute force,
and assert our hand-written decoder matches.
"""

import itertools

import numpy as np

from timbre.hmm.decoder import score_viterbi_sequences, viterbi_decode


class _TinyModel:
    """Minimal HMM interface exposing exactly what viterbi_decode needs."""

    def __init__(self):
        self.n_components = 2
        self.startprob_ = np.array([0.99, 0.01])
        self.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
        # A funky but valid per-state log-likelihood table: (T, n_states).
        self._ll = np.array([
            [np.log(0.9), np.log(0.1)],  # t=0
            [np.log(0.2), np.log(0.8)],  # t=1
            [np.log(0.6), np.log(0.4)],  # t=2
        ])

    def _compute_log_likelihood(self, obs):
        return self._ll


def test_viterbi_matches_bruteforce():
    m = _TinyModel()
    obs = np.zeros((3, 1))
    states, ll = viterbi_decode(m, obs)

    # Brute force over all 2^3 paths.
    best_ll, best_path = -np.inf, None
    for path in itertools.product(range(2), repeat=3):
        p = np.log(m.startprob_[path[0]])
        p += m._ll[0, path[0]]
        for t in range(1, 3):
            p += np.log(m.transmat_[path[t - 1], path[t]])
            p += m._ll[t, path[t]]
        if p > best_ll:
            best_ll, best_path = p, path

    assert list(states) == list(best_path)
    assert np.isclose(ll, best_ll)


def test_viterbi_single_frame():
    m = _TinyModel()
    obs = np.zeros((1, 1))
    states, ll = viterbi_decode(m, obs)
    # Single frame: argmax over startprob * first likelihood.
    expected = int(np.argmax(np.log(m.startprob_) + m._ll[0]))
    assert states[0] == expected
    assert np.isclose(ll, np.log(m.startprob_[expected]) + m._ll[0, expected])


def test_viterbi_rejects_empty_observations():
    import pytest

    with pytest.raises(ValueError, match="non-empty 2D"):
        viterbi_decode(_TinyModel(), np.empty((0, 1)))


def test_viterbi_class_score_is_length_normalized():
    model = _TinyModel()
    _, raw = viterbi_decode(model, np.zeros((3, 1)))
    assert score_viterbi_sequences(model, [np.zeros((3, 1))]) == [raw / 3]


def test_viterbi_never_starts_in_an_impossible_state():
    model = _TinyModel()
    model.startprob_ = np.array([1.0, 0.0])
    model._ll = np.array([[-1000.0, 0.0]])
    states, score = viterbi_decode(model, np.zeros((1, 1)))
    assert states.tolist() == [0]
    assert score == -1000.0


def test_viterbi_never_takes_an_impossible_transition():
    model = _TinyModel()
    model.startprob_ = np.array([1.0, 0.0])
    model.transmat_ = np.eye(2)
    model._ll = np.array([[0.0, -1000.0], [-1000.0, 0.0]])
    states, score = viterbi_decode(model, np.zeros((2, 1)))
    assert states.tolist() == [0, 0]
    assert score == -1000.0
