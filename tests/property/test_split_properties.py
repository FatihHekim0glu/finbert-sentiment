"""Hypothesis property tests for the group split's leakage invariant.

For ANY valid labelled corpus with enough groups per class, the seeded stratified
group split must: (1) place no normalized-sentence hash in more than one fold,
(2) cover every row exactly once with disjoint folds, (3) keep every class in
every fold, and (4) be deterministic under a fixed seed.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from finbert_sentiment._constants import N_CLASSES
from finbert_sentiment.data.dedup import sentence_hash
from finbert_sentiment.data.split import assert_no_group_overlap, stratified_group_split

pytestmark = pytest.mark.property


@st.composite
def _corpus(draw: st.DrawFn) -> tuple[list[int], tuple[str, ...]]:
    """Draw a corpus with >= 4 distinct groups per class (so a 3-fold split fits).

    Distinct sentences (hence distinct group hashes) are synthesized per class so
    stratification always has enough groups to populate every fold.
    """
    per_class = draw(st.integers(min_value=4, max_value=12))
    labels: list[int] = []
    texts: list[str] = []
    for cls in range(N_CLASSES):
        for k in range(per_class):
            labels.append(cls)
            texts.append(f"sentence class {cls} item {k}")
    hashes = tuple(sentence_hash(t) for t in texts)
    return labels, hashes


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(corpus=_corpus(), seed=st.integers(min_value=0, max_value=10_000))
def test_split_never_leaks_a_group_across_folds(
    corpus: tuple[list[int], tuple[str, ...]], seed: int
) -> None:
    """No group hash straddles two folds, for any corpus and seed (the leak guard)."""
    labels, hashes = corpus
    split = stratified_group_split(labels, hashes, seed=seed)
    assert_no_group_overlap(split, hashes)
    train_h = {hashes[i] for i in split.train}
    test_h = {hashes[i] for i in split.test}
    val_h = {hashes[i] for i in split.val}
    assert train_h.isdisjoint(test_h)
    assert train_h.isdisjoint(val_h)
    assert val_h.isdisjoint(test_h)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(corpus=_corpus(), seed=st.integers(min_value=0, max_value=10_000))
def test_split_covers_every_row_once_and_keeps_classes(
    corpus: tuple[list[int], tuple[str, ...]], seed: int
) -> None:
    """Folds partition the rows and every class is present in every fold."""
    labels, hashes = corpus
    split = stratified_group_split(labels, hashes, seed=seed)
    covered = sorted([*split.train, *split.val, *split.test])
    assert covered == list(range(len(labels)))
    for fold in (split.train, split.val, split.test):
        assert {labels[i] for i in fold} == set(range(N_CLASSES))


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(corpus=_corpus(), seed=st.integers(min_value=0, max_value=10_000))
def test_split_is_deterministic(corpus: tuple[list[int], tuple[str, ...]], seed: int) -> None:
    """The same inputs and seed reproduce the same split."""
    labels, hashes = corpus
    a = stratified_group_split(labels, hashes, seed=seed)
    b = stratified_group_split(labels, hashes, seed=seed)
    assert a.to_dict() == b.to_dict()
