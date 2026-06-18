"""Unit tests for sentence-level dedup + the normalized-sentence hash.

Covers: normalization (idempotent, case/whitespace/punctuation-insensitive), the
group hash (stable, 32-hex, equal for near-duplicates), and ``dedup_sentences``
(drops exact-normalized dups, keeps first occurrence, counts drops).
"""

from __future__ import annotations

import pytest

from finbert_sentiment._exceptions import ValidationError
from finbert_sentiment.data.dedup import (
    DedupResult,
    dedup_sentences,
    normalize_sentence,
    sentence_hash,
)
from finbert_sentiment.data.load import LabelledDataset, sample_dataset

pytestmark = pytest.mark.unit


def test_normalize_is_idempotent() -> None:
    """Normalizing an already-normalized string is a no-op."""
    raw = "  Quarterly  Profit   ROSE!! "
    once = normalize_sentence(raw)
    assert normalize_sentence(once) == once


def test_normalize_collapses_case_and_whitespace() -> None:
    """Case and internal whitespace differences vanish after normalization."""
    a = "Shares Rose After   the firm beat estimates."
    b = "shares rose after the firm beat estimates."
    assert normalize_sentence(a) == normalize_sentence(b)


def test_normalize_strips_terminal_punctuation() -> None:
    """A trailing full stop does not change the normalized form."""
    assert normalize_sentence("Profit rose.") == normalize_sentence("Profit rose")


def test_normalize_rejects_non_string() -> None:
    """A non-string input is rejected."""
    with pytest.raises(ValidationError):
        normalize_sentence(123)  # type: ignore[arg-type]


def test_sentence_hash_is_32_hex_chars() -> None:
    """The BLAKE2b-16 digest is exactly 32 hex characters."""
    digest = sentence_hash("Profit rose sharply.")
    assert len(digest) == 32
    assert all(c in "0123456789abcdef" for c in digest)


def test_sentence_hash_equal_for_near_duplicates() -> None:
    """Two sentences differing only in case/space/period share a hash."""
    a = "Shares rose after the firm beat estimates and upgraded its outlook."
    b = "shares rose after the firm  beat estimates and upgraded its outlook"
    assert sentence_hash(a) == sentence_hash(b)


def test_sentence_hash_differs_for_distinct_sentences() -> None:
    """Genuinely different sentences hash differently."""
    assert sentence_hash("Profit rose.") != sentence_hash("Profit fell.")


def test_dedup_drops_exact_normalized_duplicates(
    phrasebank_sample: LabelledDataset,
) -> None:
    """The fixture's two near-duplicate pairs collapse to one each."""
    result = dedup_sentences(phrasebank_sample)
    assert isinstance(result, DedupResult)
    # 18 rows in -> 2 near-dup pairs collapse -> 16 unique survivors.
    assert result.n_dropped == 2
    assert result.dataset.n == 16
    assert len(result.group_hashes) == result.dataset.n


def test_dedup_group_hashes_are_unique_after_dedup(
    phrasebank_sample: LabelledDataset,
) -> None:
    """After dedup every surviving example has a distinct group hash."""
    result = dedup_sentences(phrasebank_sample)
    assert len(set(result.group_hashes)) == len(result.group_hashes)


def test_dedup_keeps_first_occurrence() -> None:
    """The first surface form of a duplicate is the one retained."""
    ds = sample_dataset(
        ["Profit rose.", "profit rose", "Sales fell."],
        ["positive", "positive", "negative"],
    )
    result = dedup_sentences(ds)
    assert result.n_dropped == 1
    assert result.dataset.texts == ("Profit rose.", "Sales fell.")


def test_dedup_result_to_dict_flattens_dataset(
    phrasebank_sample: LabelledDataset,
) -> None:
    """``DedupResult.to_dict`` is JSON-serializable with a flattened dataset."""
    import json

    payload = dedup_sentences(phrasebank_sample).to_dict()
    assert isinstance(payload["dataset"], dict)
    json.loads(json.dumps(payload))  # must not raise


def test_dedup_rejects_empty_via_construction() -> None:
    """An empty dataset cannot even be constructed, so dedup never sees one."""
    with pytest.raises(ValidationError):
        LabelledDataset(texts=(), labels=())
