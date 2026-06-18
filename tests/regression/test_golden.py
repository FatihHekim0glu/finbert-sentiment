"""Regression / locked-behaviour tests.

* The lexicon baseline's confusion matrix on ``phrasebank_sample`` is pinned to a
  golden snapshot, so a change in the word lists or the decision rule is caught.
* The class-prior baseline's prediction equals the majority class and its score
  equals the train prior exactly.
* The ``derive_verdict`` truth table is pinned (the honest ``beats_lexicon``
  decision rule) across all four corners.

These pin cross-module behaviour on the offline fixture; a drift in any kernel
(lexicon word lists, prior fit, verdict rule) trips a snapshot here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from finbert_sentiment.baselines.class_prior import ClassPriorClassifier
from finbert_sentiment.baselines.lexicon import LexiconClassifier
from finbert_sentiment.evaluation.metrics import confusion_matrix, macro_f1
from finbert_sentiment.evaluation.verdict import Verdict, derive_verdict

if TYPE_CHECKING:
    from finbert_sentiment.data.load import LabelledDataset

pytestmark = pytest.mark.regression

#: Golden confusion matrix for the lexicon on the offline ``phrasebank_sample``
#: (rows = true negative/neutral/positive, cols = predicted). The sample is
#: deliberately lexicon-separable, so the lexicon classifies every clear example
#: correctly and the matrix is block-diagonal. Any drift in the word lists or the
#: net-tone decision rule perturbs this snapshot.
_GOLDEN_LEXICON_CONFUSION = np.array(
    [
        [6, 0, 0],  # 6 negative sentences, all predicted negative
        [0, 6, 0],  # 6 neutral sentences, all predicted neutral
        [0, 0, 6],  # 6 positive sentences, all predicted positive
    ],
    dtype=np.int64,
)


def test_lexicon_confusion_matrix_golden(phrasebank_sample: LabelledDataset) -> None:
    """Pin the lexicon confusion matrix on the offline sample to a golden snapshot."""
    y_true = list(phrasebank_sample.labels)
    y_pred = [int(v) for v in LexiconClassifier().predict(list(phrasebank_sample.texts))]
    cm = confusion_matrix(y_true, y_pred)
    np.testing.assert_array_equal(cm, _GOLDEN_LEXICON_CONFUSION)
    # The sample is separable by construction, so macro-F1 is exactly 1.0 here.
    assert macro_f1(y_true, y_pred) == pytest.approx(1.0)


def test_class_prior_equals_prior() -> None:
    """The class-prior classifier predicts the majority class and scores the prior."""
    # 5 neutral, 3 positive, 2 negative -> majority is neutral (index 1).
    train_labels = [1, 1, 1, 1, 1, 2, 2, 2, 0, 0]
    clf = ClassPriorClassifier.fit(train_labels)
    assert clf.majority_index == 1
    # Every prediction is the majority class, regardless of input content.
    preds = clf.predict(["whatever text", "another sentence", "third"])
    assert list(preds) == [1, 1, 1]
    # The score vector equals the exact train class frequencies.
    proba = clf.predict_proba(["x"])
    np.testing.assert_allclose(proba[0], [0.2, 0.5, 0.3], atol=1e-12)


@pytest.mark.parametrize(
    "model_f1,lexicon_f1,p_value,expected_verdict,expected_beats",
    [
        # margin cleared AND significant -> model beats lexicon
        (0.88, 0.70, 0.001, Verdict.MODEL_BEATS_LEXICON, True),
        # margin too small (significant) -> no significant difference
        (0.705, 0.70, 0.001, Verdict.NO_SIGNIFICANT_DIFFERENCE, False),
        # margin cleared but NOT significant -> no significant difference
        (0.90, 0.70, 0.20, Verdict.NO_SIGNIFICANT_DIFFERENCE, False),
        # neither margin nor significance -> no significant difference
        (0.705, 0.70, 0.30, Verdict.NO_SIGNIFICANT_DIFFERENCE, False),
        # no transformer in this build -> lexicon-only, beats is None
        (None, 0.70, None, Verdict.LEXICON_ONLY, None),
    ],
)
def test_derive_verdict_truth_table(
    model_f1: float | None,
    lexicon_f1: float,
    p_value: float | None,
    expected_verdict: Verdict,
    expected_beats: bool | None,
) -> None:
    """Pin the ``beats_lexicon`` truth table (margin AND McNemar significance)."""
    res = derive_verdict(model_f1, lexicon_f1, p_value, alpha=0.05, min_margin=0.02)
    assert res.verdict is expected_verdict
    assert res.beats_lexicon is expected_beats
