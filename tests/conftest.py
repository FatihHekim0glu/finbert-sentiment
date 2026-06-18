"""Shared, seeded, OFFLINE test fixtures.

Every fixture here is deterministic and **needs no network and no torch**: the
labelled sentence set is a small set baked directly into this module, so the
load -> dedup -> split -> (lexicon) eval pipeline can be exercised end-to-end in
CI without ever contacting HuggingFace or importing a deep-learning framework.

Fixtures
--------
``phrasebank_sample``
    A small, offline-cached labelled corpus of clearly negative / neutral /
    positive financial sentences (an offline stand-in for the Financial
    PhraseBank ``sentences_allagree`` config), as a
    :class:`finbert_sentiment.data.load.LabelledDataset`. The sentences are
    deliberately lexicon-separable so the lexicon baseline is sign-correct on the
    clear +/- examples.
``shuffled_label_control``
    The same texts with their labels deterministically shuffled (seeded), so any
    classifier scores no better than the class prior — the negative control that
    proves the evaluation harness is not leaking signal.

Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from finbert_sentiment._rng import make_rng

if TYPE_CHECKING:
    import numpy as np

    from finbert_sentiment.data.load import LabelledDataset

#: Master seed shared by the fixtures for byte-identical synthetic data.
SEED = 20260618

#: Offline-cached labelled sentences: ``(sentence, label)`` with label in
#: ``{"negative", "neutral", "positive"}``. Hand-written to be lexicon-separable
#: on the clear +/- cases and genuinely neutral on the neutral cases. A couple of
#: near-duplicate pairs are included so the dedup + group-split guards have
#: something to collapse / keep together.
PHRASEBANK_SAMPLE: tuple[tuple[str, str], ...] = (
    # --- positive ---------------------------------------------------------- #
    ("Quarterly profit rose sharply as revenue gains beat analyst estimates.", "positive"),
    ("The company reported record growth and raised its full-year guidance.", "positive"),
    ("Operating margins improved and the stock surged to a new high.", "positive"),
    ("Strong demand boosted earnings well above expectations.", "positive"),
    ("Shares rose after the firm beat estimates and upgraded its outlook.", "positive"),
    # a near-duplicate of the line above (differs only in case/spacing/punctuation)
    ("Shares rose after the firm beat estimates and upgraded its outlook", "positive"),
    # --- negative ---------------------------------------------------------- #
    ("Quarterly loss widened as revenue declined and margins fell.", "negative"),
    ("The company cut its guidance after a sharp drop in demand.", "negative"),
    ("Profit plunged and the stock dropped to a multi-year low.", "negative"),
    ("Weak sales and a downgrade dragged the shares lower.", "negative"),
    ("The firm warned of further losses amid a lawsuit and bankruptcy risk.", "negative"),
    # a near-duplicate of the line above
    ("The firm warned of further losses amid a lawsuit and bankruptcy risk", "negative"),
    # --- neutral ----------------------------------------------------------- #
    ("The company will hold its annual general meeting in May.", "neutral"),
    ("The board appointed a new chief financial officer effective next month.", "neutral"),
    ("The headquarters are located in Helsinki, Finland.", "neutral"),
    ("The report covers the fiscal period ending in December.", "neutral"),
    ("The press release was distributed to shareholders on Tuesday.", "neutral"),
    ("The agreement is subject to customary regulatory approvals.", "neutral"),
)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(SEED)


@pytest.fixture
def phrasebank_sample() -> LabelledDataset:
    """A small, offline-cached labelled corpus (no network, no torch).

    Built from :data:`PHRASEBANK_SAMPLE` via
    :func:`finbert_sentiment.data.load.sample_dataset`, so it is a genuine
    :class:`~finbert_sentiment.data.load.LabelledDataset` the whole pipeline
    accepts. The set is lexicon-separable on the clear +/- examples and includes
    near-duplicate pairs for the dedup / group-split guards.
    """
    from finbert_sentiment.data.load import sample_dataset

    texts = [t for t, _ in PHRASEBANK_SAMPLE]
    labels = [lab for _, lab in PHRASEBANK_SAMPLE]
    return sample_dataset(texts, labels, source="offline-cached-sample")


@pytest.fixture
def shuffled_label_control(phrasebank_sample: LabelledDataset) -> LabelledDataset:
    """The sample with labels deterministically shuffled (the negative control).

    Permuting the labels with a fixed seed destroys the text->label relationship,
    so no classifier should beat the class prior on this control. Used to prove
    the eval harness is not leaking signal and that the class-prior baseline is
    genuinely the floor.
    """
    from finbert_sentiment.data.load import sample_dataset

    gen = make_rng(SEED)
    texts = list(phrasebank_sample.texts)
    labels = list(phrasebank_sample.labels)
    perm = gen.permutation(len(labels))
    shuffled = [labels[i] for i in perm]
    return sample_dataset(texts, shuffled, source="shuffled-label-control")
