"""Unit tests for the seeded stratified group split (the locked-test-set leakage guard).

Covers: disjoint folds covering every row once; NO train/val/test
sentence-hash overlap (the real PhraseBank leak); stratification preserves every
class in every fold; determinism under seed; and the validation / insufficient-data
error paths.
"""

from __future__ import annotations

import pytest

from finbert_sentiment._constants import N_CLASSES
from finbert_sentiment._exceptions import InsufficientDataError, ValidationError
from finbert_sentiment.data.dedup import dedup_sentences
from finbert_sentiment.data.load import LabelledDataset
from finbert_sentiment.data.split import (
    SplitIndices,
    assert_no_group_overlap,
    stratified_group_split,
)

pytestmark = pytest.mark.unit

#: A larger synthetic corpus (distinct group hashes) for split-ratio assertions.
_HASHES = tuple(f"{i:032x}" for i in range(60))
_LABELS = tuple(i % N_CLASSES for i in range(60))  # 20 of each class


def _deduped(phrasebank_sample: LabelledDataset) -> tuple[list[int], tuple[str, ...]]:
    result = dedup_sentences(phrasebank_sample)
    return list(result.dataset.labels), result.group_hashes


def test_split_folds_are_disjoint_and_cover_all_rows(
    phrasebank_sample: LabelledDataset,
) -> None:
    """Train/val/test are disjoint and together cover every deduped row once."""
    labels, hashes = _deduped(phrasebank_sample)
    split = stratified_group_split(labels, hashes)
    all_idx = sorted([*split.train, *split.val, *split.test])
    assert all_idx == list(range(len(labels)))
    assert set(split.train).isdisjoint(split.val)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.val).isdisjoint(split.test)


def test_split_has_no_train_test_group_overlap(
    phrasebank_sample: LabelledDataset,
) -> None:
    """The headline leakage guard: no sentence-hash straddles two folds."""
    labels, hashes = _deduped(phrasebank_sample)
    split = stratified_group_split(labels, hashes)
    # Explicit hash-set overlap check across all fold pairs...
    train_h = {hashes[i] for i in split.train}
    val_h = {hashes[i] for i in split.val}
    test_h = {hashes[i] for i in split.test}
    assert train_h.isdisjoint(test_h)
    assert train_h.isdisjoint(val_h)
    assert val_h.isdisjoint(test_h)
    # ...and via the library's own executable guarantee.
    assert_no_group_overlap(split, hashes)


def test_split_preserves_every_class_in_every_fold(
    phrasebank_sample: LabelledDataset,
) -> None:
    """Stratification places at least one of each class into each fold."""
    labels, hashes = _deduped(phrasebank_sample)
    split = stratified_group_split(labels, hashes)
    for fold in (split.train, split.val, split.test):
        present = {labels[i] for i in fold}
        assert present == set(range(N_CLASSES))


def test_split_ratios_track_targets_on_larger_corpus() -> None:
    """Test/val fractions are roughly honoured on a 60-row balanced corpus."""
    split = stratified_group_split(
        list(_LABELS), _HASHES, test_fraction=0.2, val_fraction=0.1, seed=7
    )
    n = len(_LABELS)
    assert split.sizes["train"] + split.sizes["val"] + split.sizes["test"] == n
    # Within a reasonable tolerance of the targets (whole-group granularity).
    assert 0.10 <= split.sizes["test"] / n <= 0.35
    assert 0.03 <= split.sizes["val"] / n <= 0.25
    assert_no_group_overlap(split, _HASHES)


def test_split_is_deterministic_under_seed() -> None:
    """The same ``(labels, hashes, seed)`` yields a byte-identical split."""
    a = stratified_group_split(list(_LABELS), _HASHES, seed=123)
    b = stratified_group_split(list(_LABELS), _HASHES, seed=123)
    assert a.to_dict() == b.to_dict()


def test_split_changes_with_seed() -> None:
    """A different seed generally produces a different assignment."""
    a = stratified_group_split(list(_LABELS), _HASHES, seed=1)
    b = stratified_group_split(list(_LABELS), _HASHES, seed=2)
    assert a.to_dict() != b.to_dict()


def test_split_to_dict_is_json_serializable() -> None:
    """The split serializes to plain JSON."""
    import json

    payload = stratified_group_split(list(_LABELS), _HASHES, seed=1).to_dict()
    assert set(payload) == {"train", "val", "test", "seed"}
    json.loads(json.dumps(payload))


def test_split_rejects_misaligned_inputs() -> None:
    """Mismatched ``labels``/``group_hashes`` lengths are rejected."""
    with pytest.raises(ValidationError):
        stratified_group_split([0, 1, 2], ("a", "b"))


def test_split_rejects_empty_inputs() -> None:
    """Empty labels are rejected."""
    with pytest.raises(ValidationError):
        stratified_group_split([], [])


@pytest.mark.parametrize(
    ("test_frac", "val_frac"),
    [(0.0, 0.1), (1.0, 0.1), (0.2, 0.0), (0.2, 1.0), (0.6, 0.5)],
)
def test_split_rejects_bad_fractions(test_frac: float, val_frac: float) -> None:
    """Out-of-range or summing-to->=1 fractions are rejected."""
    with pytest.raises(ValidationError):
        stratified_group_split(
            list(_LABELS), _HASHES, test_fraction=test_frac, val_fraction=val_frac
        )


def test_split_rejects_out_of_range_labels() -> None:
    """Labels outside ``[0, N_CLASSES)`` are rejected."""
    with pytest.raises(ValidationError):
        stratified_group_split([0, 1, 9, 2], ("a", "b", "c", "d"))


def test_split_raises_insufficient_data_when_a_class_is_too_small() -> None:
    """A class with fewer than 3 groups cannot be split into three folds."""
    # 'positive' (index 2) appears in only two groups -> cannot fill 3 folds.
    labels = [0] * 6 + [1] * 6 + [2, 2]
    hashes = tuple(f"{i:032x}" for i in range(len(labels)))
    with pytest.raises(InsufficientDataError):
        stratified_group_split(labels, hashes)


def test_assert_no_group_overlap_raises_on_leak() -> None:
    """A hand-built leaking split is caught by the guard."""
    # Same hash 'aa' in both train (idx 0) and test (idx 1) -> leakage.
    hashes = ("aa", "aa", "bb", "cc")
    leaking = SplitIndices(train=(0, 2), val=(3,), test=(1,), seed=0)
    with pytest.raises(ValidationError):
        assert_no_group_overlap(leaking, hashes)


def test_assert_no_group_overlap_rejects_out_of_range_index() -> None:
    """An index beyond the hash list is rejected."""
    with pytest.raises(ValidationError):
        assert_no_group_overlap(SplitIndices(train=(0,), val=(1,), test=(9,), seed=0), ("a", "b"))


def test_split_indices_rejects_overlapping_construction() -> None:
    """Directly constructing overlapping folds is rejected."""
    with pytest.raises(ValidationError):
        SplitIndices(train=(0, 1), val=(1,), test=(2,), seed=0)


def test_split_indices_rejects_duplicate_indices() -> None:
    """A fold with duplicate indices is rejected."""
    with pytest.raises(ValidationError):
        SplitIndices(train=(0, 0), val=(1,), test=(2,), seed=0)


def test_split_indices_rejects_val_duplicate() -> None:
    """A val fold with duplicate indices is rejected."""
    with pytest.raises(ValidationError):
        SplitIndices(train=(0,), val=(1, 1), test=(2,), seed=0)


def test_split_indices_rejects_test_duplicate() -> None:
    """A test fold with duplicate indices is rejected."""
    with pytest.raises(ValidationError):
        SplitIndices(train=(0,), val=(1,), test=(2, 2), seed=0)


def test_split_indices_rejects_train_test_overlap() -> None:
    """Train/test overlap is rejected."""
    with pytest.raises(ValidationError):
        SplitIndices(train=(0, 2), val=(1,), test=(2,), seed=0)


def test_split_indices_rejects_val_test_overlap() -> None:
    """Val/test overlap is rejected."""
    with pytest.raises(ValidationError):
        SplitIndices(train=(0,), val=(2,), test=(2,), seed=0)


def test_split_rejects_non_1d_labels() -> None:
    """A 2-D ``labels`` input is rejected before any grouping."""
    with pytest.raises(ValidationError):
        stratified_group_split([[0, 1], [1, 2]], ("a", "b"))  # type: ignore[list-item]
