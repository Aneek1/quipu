"""The split is by repo (spec §3.1): commits of one repo share code, so a repo split
across train and test would leak test answers into training."""
from collections import Counter

import pytest

from stepbuild.dataset.filters import Commit, FileChange
from stepbuild.dataset.format import format_example
from stepbuild.dataset.split import assign_split

NAMES = [f"owner{i % 97}/project-{i}" for i in range(2000)]


def test_deterministic():
    assert [assign_split(n) for n in NAMES] == [assign_split(n) for n in NAMES]
    # Pinned values: a change of hash would silently reshuffle a built dataset.
    pinned = {
        "owner/name": "train",
        "owner17/project-17": "test",
        "owner26/project-26": "validation",
        "owner53/project-53": "test",
    }
    assert {n: assign_split(n) for n in pinned} == pinned


def test_values_are_the_three_splits():
    assert {assign_split(n) for n in NAMES} == {"train", "validation", "test"}


def test_ratios_within_two_percent():
    counts = Counter(assign_split(n) for n in NAMES)
    for name, want in (("train", 0.90), ("validation", 0.05), ("test", 0.05)):
        assert abs(counts[name] / len(NAMES) - want) <= 0.02, counts


def test_custom_ratios():
    assert {assign_split(n, ratios=(0.0, 0.0, 1.0)) for n in NAMES[:200]} == {"test"}
    counts = Counter(assign_split(n, ratios=(0.5, 0.5, 0.0)) for n in NAMES)
    assert counts["test"] == 0 and abs(counts["train"] / len(NAMES) - 0.5) <= 0.03


def test_repo_names_ignore_case():
    # GitHub names are case-insensitive: one repo must not land in two splits.
    assert all(assign_split(n) == assign_split(n.upper()) for n in NAMES[:200])


@pytest.mark.parametrize("ratios", [(0.9, 0.05), (0.9, 0.05, 0.1), (1.1, -0.05, -0.05), "abc"])
def test_bad_ratios(ratios):
    with pytest.raises((ValueError, TypeError)):
        assign_split("a/b", ratios=ratios)


def test_bad_repo():
    with pytest.raises((ValueError, TypeError)):
        assign_split("")


def test_all_commits_of_a_repo_share_a_split():
    for repo in NAMES[:300]:
        splits = set()
        for i in range(5):
            c = Commit(
                sha=f"{i:040x}",
                message="Add the item store with ids",
                parents=1,
                changes=(FileChange("backend/models.py", None, "x = 1\n", 1, 0),),
            )
            splits.add(format_example(repo, "MIT", "flask", c, {}, [])["split"])
        assert len(splits) == 1, repo
