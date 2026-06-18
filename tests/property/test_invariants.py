"""Property-based invariants for the pipeline (Hypothesis).

Invariants the pipeline must satisfy for ANY valid input:

* **No group overlap:** no normalized-sentence hash straddles train/val/test.
* **Determinism:** ``predict`` is a pure function of the input under a fixed seed.
* **Tokenization stability:** repeated tokenization of the same text is identical.
* **Lexicon sign-correctness:** clearly positive/negative sentences classify with
  the right sign.
* **Class-prior floor:** on a shuffled-label control, no classifier beats the
  prior.

These are skipped until the corresponding kernels are implemented.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.property

_PENDING_SPLIT = "pending finbert_sentiment.data.split.stratified_group_split"
_PENDING_LEXICON = "pending finbert_sentiment.baselines.lexicon.LexiconClassifier"


@pytest.mark.skip(reason=_PENDING_SPLIT)
def test_no_train_test_group_overlap() -> None:
    """No normalized-sentence hash appears in more than one fold (the leakage guard)."""
    raise AssertionError("implement with Hypothesis once split lands")


@pytest.mark.skip(reason=_PENDING_LEXICON)
def test_lexicon_predictions_are_deterministic() -> None:
    """Repeated ``predict`` on the same batch returns identical labels."""
    raise AssertionError("implement once LexiconClassifier.predict lands")


@pytest.mark.skip(reason=_PENDING_LEXICON)
def test_lexicon_sign_correct_on_clear_cases() -> None:
    """Clearly positive/negative sentences classify with the correct sign."""
    raise AssertionError("implement once LexiconClassifier.predict lands")
