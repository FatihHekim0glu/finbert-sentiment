"""Loughran-McDonald-style 3-way lexicon classifier (the torch-free LIVE model).

This classifier counts domain-specific positive/negative cue words in each
sentence and decides ``positive`` / ``negative`` by the signed count, defaulting
to ``neutral`` when the net tone is zero. It is the honest lexical floor the
transformer is measured against AND the always-available served model the FastAPI
tool falls back to when no ONNX artifact is present (it needs only Python — no
torch, no onnxruntime, no network).

The word lists below are a compact, finance-oriented subset in the spirit of the
Loughran-McDonald master dictionary; they are illustrative, not the full LM list.
Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

#: Positive finance cue words (lowercase stems). Compact LM-style subset.
LEXICON_POSITIVE: frozenset[str] = frozenset(
    {
        "gain",
        "gains",
        "profit",
        "profits",
        "growth",
        "rose",
        "rise",
        "rises",
        "rising",
        "up",
        "surge",
        "surged",
        "beat",
        "beats",
        "strong",
        "improved",
        "improvement",
        "record",
        "higher",
        "positive",
        "upgrade",
        "outperform",
        "boost",
        "boosted",
        "exceeded",
    }
)

#: Negative finance cue words (lowercase stems). Compact LM-style subset.
LEXICON_NEGATIVE: frozenset[str] = frozenset(
    {
        "loss",
        "losses",
        "decline",
        "declined",
        "fell",
        "fall",
        "falls",
        "drop",
        "dropped",
        "down",
        "plunge",
        "plunged",
        "miss",
        "missed",
        "weak",
        "weaker",
        "cut",
        "cuts",
        "lower",
        "negative",
        "downgrade",
        "underperform",
        "warning",
        "lawsuit",
        "bankruptcy",
    }
)


@dataclass(frozen=True, slots=True)
class LexiconClassifier:
    """A stateless, deterministic lexicon sentiment classifier.

    The classifier holds no fitted parameters (the word lists are fixed), so it
    is constructible directly and is safe to share across requests. ``predict``
    is a pure function of the input text and the two frozen word sets.

    Attributes
    ----------
    positive:
        Positive cue-word set (defaults to :data:`LEXICON_POSITIVE`).
    negative:
        Negative cue-word set (defaults to :data:`LEXICON_NEGATIVE`).
    """

    positive: frozenset[str] = field(default_factory=lambda: LEXICON_POSITIVE)
    negative: frozenset[str] = field(default_factory=lambda: LEXICON_NEGATIVE)

    def _score_one(self, text: str) -> tuple[int, int]:
        """Return the ``(pos_count, neg_count)`` cue-word counts for one sentence."""
        raise NotImplementedError

    def predict(self, texts: Sequence[str]) -> NDArray[np.int64]:
        """Classify each text as negative/neutral/positive by net cue tone.

        Decision rule: ``positive`` if ``pos_count > neg_count``, ``negative`` if
        ``neg_count > pos_count``, else ``neutral``. Deterministic and content-only.

        Parameters
        ----------
        texts:
            The batch to classify.

        Returns
        -------
        numpy.ndarray
            A length-``len(texts)`` ``int64`` vector of class indices.

        Raises
        ------
        ValidationError
            If ``texts`` fails the batch validation.
        """
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return pseudo-probabilities derived from normalized cue counts.

        The scores are a softmax-like normalization of ``(neg, neutral, pos)``
        pseudo-logits; they are NOT calibrated probabilities (the lexicon has no
        likelihood model) and are documented as such wherever surfaced.

        Parameters
        ----------
        texts:
            The batch to classify.

        Returns
        -------
        numpy.ndarray
            A ``(len(texts), N_CLASSES)`` row-stochastic score matrix.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (word sets as sorted lists)."""
        out: dict[str, Any] = asdict(self)
        out["positive"] = sorted(self.positive)
        out["negative"] = sorted(self.negative)
        return out
