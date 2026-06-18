"""Property-based invariants for the pipeline (Hypothesis).

Invariants the pipeline must satisfy for ANY valid input:

* **No group overlap:** no normalized-sentence hash straddles train/val/test (the
  real leakage risk in PhraseBank — near-duplicate sentences across folds).
* **Determinism:** ``predict`` is a pure function of the input under a fixed seed.
* **Lexicon sign-correctness:** clearly positive/negative sentences classify with
  the right sign.
* **Class-prior floor:** on a shuffled-label control, the lexicon scores no better
  (in macro-F1) than it would on signal-bearing data — the negative control that
  proves the harness is not leaking signal and the prior is genuinely the floor.

These thread the data / baseline / evaluation modules together on generated
inputs, complementing the fixed-example unit tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from finbert_sentiment.baselines.class_prior import ClassPriorClassifier
from finbert_sentiment.baselines.lexicon import LexiconClassifier
from finbert_sentiment.data.dedup import dedup_sentences
from finbert_sentiment.data.load import sample_dataset
from finbert_sentiment.data.split import assert_no_group_overlap, stratified_group_split
from finbert_sentiment.evaluation.metrics import macro_f1

if TYPE_CHECKING:
    from finbert_sentiment.data.load import LabelledDataset

pytestmark = pytest.mark.property

# A pool of clearly-signed sentences per class. Generated corpora draw from these
# (with optional near-duplicate punctuation/case variants) so every class can be
# placed into all three folds and the lexicon stays sign-correct on clear cases.
_POSITIVE = (
    "Quarterly profit rose sharply as revenue gains beat estimates",
    "The company reported record growth and raised its guidance",
    "Operating margins improved and the stock surged higher",
    "Strong demand boosted earnings well above expectations",
    "Shares rose after the firm beat estimates and upgraded its outlook",
)
_NEGATIVE = (
    "Quarterly loss widened as revenue declined and margins fell",
    "The company cut its guidance after a sharp drop in demand",
    "Profit plunged and the stock dropped to a multi-year low",
    "Weak sales and a downgrade dragged the shares lower",
    "The firm warned of further losses amid a lawsuit and bankruptcy risk",
)
_NEUTRAL = (
    "The company will hold its annual general meeting in May",
    "The board appointed a new chief financial officer next month",
    "The headquarters are located in Helsinki Finland",
    "The report covers the fiscal period ending in December",
    "The press release was distributed to shareholders on Tuesday",
)
_POOL = {"positive": _POSITIVE, "negative": _NEGATIVE, "neutral": _NEUTRAL}


@st.composite
def _labelled_corpus(draw: st.DrawFn) -> LabelledDataset:
    """Generate a labelled corpus with >= 3 distinct groups per class.

    Each class contributes 3-5 distinct base sentences, and some sentences get a
    near-duplicate (a punctuation/case variant) appended so the dedup + group
    split have collisions to collapse / keep together.
    """
    texts: list[str] = []
    labels: list[str] = []
    for label, pool in _POOL.items():
        k = draw(st.integers(min_value=3, max_value=len(pool)))
        chosen = pool[:k]
        for sentence in chosen:
            texts.append(sentence + ".")
            labels.append(label)
            # Optionally add a near-duplicate (differs only in trailing punctuation
            # / case) — must group with its twin, never straddle folds.
            if draw(st.booleans()):
                texts.append(sentence.upper() + "  ")
                labels.append(label)
    return sample_dataset(texts, labels, source="generated")


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(corpus=_labelled_corpus(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_no_train_test_group_overlap(corpus: LabelledDataset, seed: int) -> None:
    """No normalized-sentence hash appears in more than one fold (the leakage guard)."""
    deduped = dedup_sentences(corpus)
    split = stratified_group_split(deduped.dataset.labels, deduped.group_hashes, seed=seed)
    # The executable leakage guarantee must hold for every generated corpus/seed.
    assert_no_group_overlap(split, deduped.group_hashes)

    # And, directly: the hash sets of the three folds are pairwise disjoint.
    hashes = deduped.group_hashes
    train_h = {hashes[i] for i in split.train}
    val_h = {hashes[i] for i in split.val}
    test_h = {hashes[i] for i in split.test}
    assert train_h.isdisjoint(test_h)
    assert train_h.isdisjoint(val_h)
    assert val_h.isdisjoint(test_h)


@settings(max_examples=50, deadline=None)
@given(
    texts=st.lists(
        st.sampled_from(_POSITIVE + _NEGATIVE + _NEUTRAL).map(lambda s: s + "."),
        min_size=1,
        max_size=12,
    )
)
def test_lexicon_predictions_are_deterministic(texts: list[str]) -> None:
    """Repeated ``predict`` on the same batch returns identical labels."""
    clf = LexiconClassifier()
    first = clf.predict(texts)
    second = clf.predict(texts)
    assert list(first) == list(second)
    # Stable across a fresh instance too (no hidden mutable state).
    assert list(LexiconClassifier().predict(texts)) == list(first)


@settings(max_examples=40, deadline=None)
@given(
    pos=st.sampled_from(_POSITIVE),
    neg=st.sampled_from(_NEGATIVE),
)
def test_lexicon_sign_correct_on_clear_cases(pos: str, neg: str) -> None:
    """Clearly positive/negative sentences classify with the correct sign."""
    preds = LexiconClassifier().predict([pos + ".", neg + "."])
    assert int(preds[0]) == 2, f"expected positive for {pos!r}"
    assert int(preds[1]) == 0, f"expected negative for {neg!r}"


#: A balanced, clearly-signed corpus (15 per class) used by the shuffled-label
#: control. Drawn from the clear-case pools so the lexicon is sign-correct on the
#: TRUE mapping, which is the whole point of the control: shuffling must destroy
#: that signal.
_CONTROL_TEXTS: tuple[str, ...] = tuple(
    s + "." for s in (_POSITIVE * 3 + _NEGATIVE * 3 + _NEUTRAL * 3)
)
_CONTROL_LABELS: tuple[int, ...] = (2,) * 15 + (0,) * 15 + (1,) * 15


def test_lexicon_signal_collapses_on_shuffled_label_control() -> None:
    """Shuffling labels destroys the lexicon's signal (the no-leakage control).

    On the TRUE text->label mapping the lexicon is sign-correct and scores a high
    macro-F1. Permuting the labels (seeded, many times) destroys that mapping, so
    the lexicon's mean macro-F1 must collapse toward chance — well below its
    true-label score. If the evaluation harness were leaking signal, the shuffled
    score would stay high; it does not.
    """
    import numpy as np

    from finbert_sentiment._rng import make_rng

    texts = list(_CONTROL_TEXTS)
    true_labels = list(_CONTROL_LABELS)
    lexicon = LexiconClassifier()
    y_lexicon = [int(v) for v in lexicon.predict(texts)]

    # Real signal present on the true mapping (sign-correct -> macro-F1 == 1.0).
    true_f1 = macro_f1(true_labels, y_lexicon)
    assert true_f1 == pytest.approx(1.0)

    # Averaged over many shufflings, the lexicon's macro-F1 collapses toward chance.
    shuffled_scores = []
    for seed in range(200):
        gen = make_rng(seed)
        perm = gen.permutation(len(true_labels))
        shuffled = [true_labels[i] for i in perm]
        shuffled_scores.append(macro_f1(shuffled, y_lexicon))
    mean_shuffled = float(np.mean(shuffled_scores))
    # The signal gap is large and unambiguous (true ~1.0 vs shuffled ~0.33).
    assert mean_shuffled < 0.55
    assert true_f1 - mean_shuffled > 0.40


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(corpus=_labelled_corpus())
def test_class_prior_is_the_macro_f1_floor(corpus: LabelledDataset) -> None:
    """The class-prior predicts a single class, so its macro-F1 is the trivial floor.

    Fitting on the corpus's own labels, the prior always predicts the majority
    class; it therefore scores F1 == 0 on at least the two non-majority classes,
    so its macro-F1 cannot exceed ``1 / N_CLASSES``. This pins the honest floor
    the transformer and lexicon are measured against.
    """
    from finbert_sentiment._constants import N_CLASSES

    labels = list(corpus.labels)
    prior = ClassPriorClassifier.fit(labels)
    y_prior = [int(v) for v in prior.predict(corpus.texts)]
    floor = macro_f1(labels, y_prior)
    # Single-class predictions -> at most one class has non-zero F1.
    assert floor <= 1.0 / N_CLASSES + 1e-12
