"""Tests for deterministic, speaker-independent RAVDESS splitting."""

from scripts.make_splits import split_by_actor
from timbre.utils.common import parse_ravdess_actor


def _clips() -> list[tuple[str, int, int]]:
    return [
        (f"03-01-{emotion:02d}-01-01-01-{actor:02d}", emotion, actor)
        for actor in range(1, 9)
        for emotion in range(1, 9)
    ]


def test_parse_ravdess_actor():
    assert parse_ravdess_actor("03-01-06-01-02-01-17.wav") == 17


def test_actor_splits_are_disjoint_and_complete():
    clips = _clips()
    splits = split_by_actor(clips, val_frac=0.25, test_frac=0.25, seed=42)
    actors = {name: set(splits["metadata"]["actors"][name]) for name in ("train", "val", "test")}

    assert actors["train"].isdisjoint(actors["val"])
    assert actors["train"].isdisjoint(actors["test"])
    assert actors["val"].isdisjoint(actors["test"])
    assert set().union(*actors.values()) == set(range(1, 9))
    assert sum(len(splits[name]) for name in actors) == len(clips)


def test_actor_split_is_deterministic():
    assert split_by_actor(_clips(), seed=9) == split_by_actor(_clips(), seed=9)
