"""Unit tests for the torch-free baselines (class-prior + lexicon).

These tests are self-contained: they build their inputs from in-memory text and
label lists rather than the ``phrasebank_sample`` fixture, so they exercise the
baseline kernels without depending on the data group's ``LabelledDataset``.

Coverage of the brief's baseline requirements:

* ``ClassPriorClassifier`` predicts the TRAIN majority class and scores the train
  prior exactly; it is deterministic and the macro-F1 floor.
* ``LexiconClassifier`` is sign-correct on clearly positive / negative finance
  sentences, defaults to neutral on no-cue / tied sentences, is deterministic,
  and its ``predict_proba`` argmax is consistent with ``predict``.
"""

from __future__ import annotations

import numpy as np
import pytest

from finbert_sentiment._constants import INDEX_TO_LABEL, LABEL_TO_INDEX, N_CLASSES
from finbert_sentiment._exceptions import ValidationError
from finbert_sentiment.baselines import ClassPriorClassifier, LexiconClassifier

pytestmark = pytest.mark.unit

# --------------------------------------------------------------------------- #
# Shared, lexicon-separable sample sentences.                                  #
# --------------------------------------------------------------------------- #
_POSITIVE_SENTENCES = (
    "Quarterly profit rose sharply as revenue gains beat analyst estimates.",
    "The company reported record growth and raised its full-year guidance.",
    "Strong demand boosted earnings well above expectations.",
)
_NEGATIVE_SENTENCES = (
    "Quarterly loss widened as revenue declined and margins fell.",
    "Profit plunged and the stock dropped to a multi-year low.",
    "Weak sales and a downgrade dragged the shares lower.",
)
_NEUTRAL_SENTENCES = (
    "The company will hold its annual general meeting in May.",
    "The headquarters are located in Helsinki, Finland.",
    "The report covers the fiscal period ending in December.",
)

_NEG = LABEL_TO_INDEX["negative"]
_NEU = LABEL_TO_INDEX["neutral"]
_POS = LABEL_TO_INDEX["positive"]


# --------------------------------------------------------------------------- #
# ClassPriorClassifier                                                         #
# --------------------------------------------------------------------------- #
def test_class_prior_predicts_train_majority() -> None:
    """The classifier always predicts the most frequent TRAIN label index."""
    # neutral (index 1) is the strict majority: 5 vs 2 vs 1.
    labels = [_NEU] * 5 + [_POS] * 2 + [_NEG] * 1
    clf = ClassPriorClassifier.fit(labels)
    assert clf.majority_index == _NEU
    preds = clf.predict(["any text", "another", "third"])
    assert preds.tolist() == [_NEU, _NEU, _NEU]
    assert preds.dtype == np.int64


def test_class_prior_scores_equal_train_prior() -> None:
    """``predict_proba`` broadcasts the exact train class-frequency vector."""
    labels = [_NEG, _NEG, _NEU, _NEU, _NEU, _NEU, _POS, _POS]  # 2/4/2 of 8
    clf = ClassPriorClassifier.fit(labels)
    expected = np.array([2 / 8, 4 / 8, 2 / 8])
    np.testing.assert_allclose(clf.prior, expected, atol=1e-12)
    proba = clf.predict_proba(["a", "b"])
    assert proba.shape == (2, N_CLASSES)
    np.testing.assert_allclose(proba[0], expected, atol=1e-12)
    np.testing.assert_allclose(proba[1], expected, atol=1e-12)
    # Prior is a proper distribution.
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(2), atol=1e-12)


def test_class_prior_accepts_string_labels() -> None:
    """String class names are normalized to indices via the shared validator."""
    clf = ClassPriorClassifier.fit(["neutral", "neutral", "positive"])
    assert INDEX_TO_LABEL[clf.majority_index] == "neutral"


def test_class_prior_is_deterministic() -> None:
    """Fitting and predicting twice yields byte-identical results."""
    labels = [_POS, _NEU, _NEU, _NEG, _NEU]
    a = ClassPriorClassifier.fit(labels)
    b = ClassPriorClassifier.fit(labels)
    assert a.prior == b.prior
    assert a.majority_index == b.majority_index
    np.testing.assert_array_equal(a.predict(["x"] * 4), b.predict(["x"] * 4))


def test_class_prior_tie_breaks_to_lowest_index() -> None:
    """A perfectly balanced prior breaks the majority tie toward index 0."""
    clf = ClassPriorClassifier.fit([_NEG, _NEU, _POS])
    assert clf.majority_index == 0
    np.testing.assert_allclose(clf.prior, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)


def test_class_prior_rejects_empty_labels() -> None:
    """An empty training label set is a validation error."""
    with pytest.raises(ValidationError):
        ClassPriorClassifier.fit([])


def test_class_prior_rejects_out_of_range_labels() -> None:
    """A label outside the 3-way space is rejected by the shared validator."""
    with pytest.raises(ValidationError):
        ClassPriorClassifier.fit([0, 1, 7])


def test_class_prior_predict_rejects_empty_batch() -> None:
    """Predicting on an empty batch raises (the per-batch validator runs)."""
    clf = ClassPriorClassifier.fit([_NEU, _NEU])
    with pytest.raises(ValidationError):
        clf.predict([])


def test_class_prior_to_dict_round_trips() -> None:
    """``to_dict`` is JSON-shaped and reconstructs an equivalent classifier."""
    clf = ClassPriorClassifier.fit([_NEU, _NEU, _POS])
    d = clf.to_dict()
    assert set(d) == {"prior", "majority_index"}
    rebuilt = ClassPriorClassifier(prior=tuple(d["prior"]), majority_index=d["majority_index"])
    assert rebuilt == clf


# --------------------------------------------------------------------------- #
# LexiconClassifier                                                            #
# --------------------------------------------------------------------------- #
def test_lexicon_sign_correct_on_clear_positive() -> None:
    """Clearly positive finance sentences classify as positive."""
    clf = LexiconClassifier()
    preds = clf.predict(list(_POSITIVE_SENTENCES))
    assert preds.tolist() == [_POS] * len(_POSITIVE_SENTENCES)


def test_lexicon_sign_correct_on_clear_negative() -> None:
    """Clearly negative finance sentences classify as negative."""
    clf = LexiconClassifier()
    preds = clf.predict(list(_NEGATIVE_SENTENCES))
    assert preds.tolist() == [_NEG] * len(_NEGATIVE_SENTENCES)


def test_lexicon_defaults_to_neutral_on_no_cue_sentences() -> None:
    """Sentences with no cue words fall through to neutral."""
    clf = LexiconClassifier()
    preds = clf.predict(list(_NEUTRAL_SENTENCES))
    assert preds.tolist() == [_NEU] * len(_NEUTRAL_SENTENCES)


def test_lexicon_neutral_on_balanced_cues() -> None:
    """Equal positive and negative cue counts net to neutral."""
    clf = LexiconClassifier()
    # one positive cue ("gains") and one negative cue ("loss").
    preds = clf.predict(["The gains were offset by an equal loss."])
    assert preds.tolist() == [_NEU]


def test_lexicon_is_case_and_punctuation_insensitive() -> None:
    """Matching ignores case and surrounding punctuation."""
    clf = LexiconClassifier()
    a = clf.predict(["Profit ROSE, and gains SURGED!"])
    b = clf.predict(["profit rose and gains surged"])
    np.testing.assert_array_equal(a, b)
    assert a.tolist() == [_POS]


def test_lexicon_predictions_are_deterministic() -> None:
    """Repeated prediction over the same batch is byte-identical."""
    clf = LexiconClassifier()
    batch = list(_POSITIVE_SENTENCES + _NEGATIVE_SENTENCES + _NEUTRAL_SENTENCES)
    np.testing.assert_array_equal(clf.predict(batch), clf.predict(batch))


def test_lexicon_predict_proba_is_row_stochastic() -> None:
    """``predict_proba`` rows are non-negative and sum to one."""
    clf = LexiconClassifier()
    batch = list(_POSITIVE_SENTENCES + _NEGATIVE_SENTENCES + _NEUTRAL_SENTENCES)
    proba = clf.predict_proba(batch)
    assert proba.shape == (len(batch), N_CLASSES)
    assert bool((proba >= 0.0).all())
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(batch)), atol=1e-12)


def test_lexicon_predict_proba_argmax_matches_predict() -> None:
    """The score argmax agrees with the hard label everywhere (incl. ties)."""
    clf = LexiconClassifier()
    batch = [
        *_POSITIVE_SENTENCES,
        *_NEGATIVE_SENTENCES,
        *_NEUTRAL_SENTENCES,
        "The gains were offset by an equal loss.",  # tie -> neutral
    ]
    preds = clf.predict(batch)
    argmax = clf.predict_proba(batch).argmax(axis=1)
    np.testing.assert_array_equal(preds, argmax)


def test_lexicon_proba_favours_neutral_when_no_cues() -> None:
    """On a no-cue sentence neutral carries the most mass."""
    clf = LexiconClassifier()
    proba = clf.predict_proba(["The headquarters are located in Helsinki."])
    assert int(proba[0].argmax()) == _NEU


def test_lexicon_rejects_bare_string() -> None:
    """A bare ``str`` (not wrapped in a list) is a validation error."""
    clf = LexiconClassifier()
    with pytest.raises(ValidationError):
        clf.predict("a single string is a caller mistake")  # type: ignore[arg-type]


def test_lexicon_rejects_blank_element() -> None:
    """A blank element fails the per-text batch validation."""
    clf = LexiconClassifier()
    with pytest.raises(ValidationError):
        clf.predict(["valid sentence", "   "])


def test_lexicon_to_dict_serializes_word_sets_as_sorted_lists() -> None:
    """``to_dict`` emits the cue sets as sorted, JSON-serializable lists."""
    clf = LexiconClassifier()
    d = clf.to_dict()
    assert d["positive"] == sorted(d["positive"])
    assert d["negative"] == sorted(d["negative"])
    assert isinstance(d["positive"], list)
    assert "profit" in d["positive"]
    assert "loss" in d["negative"]


def test_lexicon_score_one_counts_cues() -> None:
    """The private per-sentence scorer counts positive and negative cues."""
    clf = LexiconClassifier()
    pos, neg = clf._score_one("profit and gains rose but loss and decline fell")
    assert pos == 3  # profit, gains, rose
    assert neg == 3  # loss, decline, fell


def test_lexicon_custom_word_sets_are_honoured() -> None:
    """A custom cue set overrides the defaults (the dataclass is configurable)."""
    clf = LexiconClassifier(
        positive=frozenset({"moon"}),
        negative=frozenset({"crash"}),
    )
    assert clf.predict(["to the moon"]).tolist() == [_POS]
    assert clf.predict(["a market crash"]).tolist() == [_NEG]
    # default cue words no longer fire.
    assert clf.predict(["record profit growth"]).tolist() == [_NEU]
