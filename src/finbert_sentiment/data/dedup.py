"""Sentence-level deduplication and the normalized-sentence hash.

Near-duplicate sentences are the real leakage risk in the Financial PhraseBank:
the same headline appears in slightly different surface forms. We normalize each
sentence (lowercase, collapse whitespace, strip punctuation noise) and hash it;
exact-normalized duplicates collapse to one example, and the hash later groups
near-duplicates so they never straddle the train/val/test split.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finbert_sentiment.data.load import LabelledDataset


def normalize_sentence(text: str) -> str:
    """Return the canonical normalized form of ``text`` for hashing/dedup.

    The normalization is deterministic and idempotent: lowercase, Unicode-NFKC,
    collapse internal whitespace to single spaces, and strip leading/trailing
    whitespace. Two sentences that differ only in case or spacing map to the same
    normalized string (and therefore the same hash).

    Parameters
    ----------
    text:
        A raw sentence.

    Returns
    -------
    str
        The normalized sentence.
    """
    raise NotImplementedError


def sentence_hash(text: str) -> str:
    """Return a stable BLAKE2b-16 hex digest of ``normalize_sentence(text)``.

    The digest is the *group key* for the leakage-safe split: all sentences with
    the same hash are placed in the same fold.

    Parameters
    ----------
    text:
        A raw sentence.

    Returns
    -------
    str
        A 32-character hex digest.
    """
    raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DedupResult:
    """Immutable outcome of a dedup pass.

    Attributes
    ----------
    dataset:
        The deduplicated :class:`~finbert_sentiment.data.load.LabelledDataset`
        (one example per unique normalized sentence).
    group_hashes:
        The normalized-sentence hash for each surviving example, parallel to
        ``dataset.texts``. Reused as the group key by the split.
    n_dropped:
        How many exact-normalized duplicate rows were removed.
    """

    dataset: LabelledDataset
    group_hashes: tuple[str, ...]
    n_dropped: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (dataset flattened)."""
        out: dict[str, Any] = asdict(self)
        out["dataset"] = self.dataset.to_dict()
        return out


def dedup_sentences(dataset: LabelledDataset) -> DedupResult:
    """Collapse exact-normalized duplicate sentences, keeping the first occurrence.

    A label conflict among duplicates (the same normalized sentence carrying two
    different labels) is resolved by keeping the first occurrence and is counted
    in the result; the ``allagree`` PhraseBank config makes this rare.

    Parameters
    ----------
    dataset:
        The (pre-dedup) labelled corpus.

    Returns
    -------
    DedupResult
        The deduplicated dataset, its per-example group hashes, and the drop count.

    Raises
    ------
    ValidationError
        If ``dataset`` is empty.
    """
    raise NotImplementedError
