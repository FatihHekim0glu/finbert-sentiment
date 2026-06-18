"""Seeded, stratified, group-aware train/val/test split (the locked test set).

The split is the project's central leakage guard. It is:

* **grouped** by normalized-sentence hash, so near-duplicate sentences never
  straddle two folds (the real PhraseBank leakage risk);
* **stratified** by class, so each fold preserves the (neutral-heavy) prior;
* **seeded**, so the same ``(dataset, seed)`` always yields the same split and
  the test set is locked once and never re-shuffled.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from finbert_sentiment._constants import DEFAULT_SEED, DEFAULT_TEST_FRACTION, DEFAULT_VAL_FRACTION

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """Immutable train/val/test index partition over a deduplicated dataset.

    The three index tuples are disjoint and together cover every row exactly
    once. Indices reference positions in the deduplicated
    :class:`~finbert_sentiment.data.load.LabelledDataset`.

    Attributes
    ----------
    train:
        Row indices assigned to the training fold.
    val:
        Row indices assigned to the validation fold.
    test:
        Row indices assigned to the locked test fold.
    seed:
        The seed that produced this split (for provenance).
    """

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        """Validate that the three folds are disjoint and non-overlapping."""
        raise NotImplementedError

    @property
    def sizes(self) -> dict[str, int]:
        """Return ``{"train": n, "val": n, "test": n}`` fold sizes."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this split."""
        return asdict(self)


def stratified_group_split(
    labels: Sequence[int],
    group_hashes: Sequence[str],
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> SplitIndices:
    """Produce a seeded, class-stratified, group-disjoint train/val/test split.

    Whole groups (all rows sharing a normalized-sentence hash) are assigned to a
    single fold; the assignment is stratified so each fold approximately
    preserves the per-class prior, and seeded so the result is reproducible. The
    test set produced here is treated as LOCKED.

    Parameters
    ----------
    labels:
        Integer class indices, one per row.
    group_hashes:
        The normalized-sentence hash for each row (the group key), parallel to
        ``labels``.
    test_fraction:
        Target fraction of rows in the test fold.
    val_fraction:
        Target fraction of rows in the validation fold (of the whole dataset).
    seed:
        Master seed for the assignment.

    Returns
    -------
    SplitIndices
        The disjoint, group-aware, stratified partition.

    Raises
    ------
    ValidationError
        If ``labels`` and ``group_hashes`` are misaligned or the fractions are
        not in ``(0, 1)`` with ``test_fraction + val_fraction < 1``.
    InsufficientDataError
        If there are too few groups/examples to place every class in every fold.
    """
    raise NotImplementedError


def assert_no_group_overlap(split: SplitIndices, group_hashes: Sequence[str]) -> None:
    """Assert that no normalized-sentence hash appears in more than one fold.

    The executable form of the headline leakage guarantee. Raises if any group
    hash is shared across train/val/test (which would let a near-duplicate
    sentence leak from train into test).

    Parameters
    ----------
    split:
        The split to check.
    group_hashes:
        The per-row group hashes the split was built from.

    Raises
    ------
    ValidationError
        If any group hash straddles two or more folds.
    """
    raise NotImplementedError
