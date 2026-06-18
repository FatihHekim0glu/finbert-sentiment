"""Regression / locked-behaviour tests.

* The lexicon baseline's confusion matrix on ``phrasebank_sample`` is pinned to a
  golden snapshot, so a change in the word lists or the decision rule is caught.
* The class-prior baseline's prediction equals the majority class and its score
  equals the train prior exactly.
* The ``derive_verdict`` truth table is pinned (the honest ``beats_lexicon``
  decision rule).

These are skipped until the corresponding kernels are implemented.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

_PENDING_LEXICON = "pending finbert_sentiment.baselines.lexicon.LexiconClassifier"
_PENDING_PRIOR = "pending finbert_sentiment.baselines.class_prior.ClassPriorClassifier"
_PENDING_VERDICT = "pending finbert_sentiment.evaluation.verdict.derive_verdict"


@pytest.mark.skip(reason=_PENDING_LEXICON)
def test_lexicon_confusion_matrix_golden() -> None:
    """Pin the lexicon confusion matrix on the offline sample to a golden snapshot."""
    raise AssertionError("snapshot once LexiconClassifier lands")


@pytest.mark.skip(reason=_PENDING_PRIOR)
def test_class_prior_equals_prior() -> None:
    """The class-prior classifier predicts the majority class and scores the prior."""
    raise AssertionError("implement once ClassPriorClassifier lands")


@pytest.mark.skip(reason=_PENDING_VERDICT)
def test_derive_verdict_truth_table() -> None:
    """Pin the ``beats_lexicon`` truth table (margin AND McNemar significance)."""
    raise AssertionError("implement once derive_verdict lands")
