"""Pure-function ``beats_lexicon`` verdict derivation (the honest headline).

The headline ``beats_lexicon`` boolean is a PURE FUNCTION of the measured
numbers: the served model's macro-F1, the lexicon baseline's macro-F1, and the
McNemar p-value. It cannot read ``True`` while the model's macro-F1 fails to
exceed the lexicon's by a meaningful margin, or while McNemar fails to reject
equal error rates. This keeps the README honest — the verdict is derived, not
narrated — and it correctly returns ``None``/``False`` when no transformer was
trained in this build (the lexicon-only fallback path).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    """Possible headline verdicts for the model-vs-lexicon comparison.

    The values are stable string identifiers safe to serialize across the API
    boundary and render in the frontend.
    """

    #: The served transformer's macro-F1 exceeds the lexicon's by the required
    #: margin AND McNemar rejects equal error rates — the model beats the lexicon.
    MODEL_BEATS_LEXICON = "model_beats_lexicon"

    #: The model does not clear both bars — no significant improvement over the
    #: lexical floor on this test set.
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"

    #: No transformer was trained/served in this build, so there is nothing to
    #: compare — the lexicon IS the served model (the fallback path).
    LEXICON_ONLY = "lexicon_only"


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Immutable verdict bundle: the enum, the boolean (nullable), and rationale.

    Attributes
    ----------
    verdict:
        The derived :class:`Verdict`.
    beats_lexicon:
        ``True`` only for :attr:`Verdict.MODEL_BEATS_LEXICON`; ``False`` for a
        non-significant difference; ``None`` for the lexicon-only build (nothing
        to compare).
    rationale:
        Human-readable reason string (which condition failed, if any).
    """

    verdict: Verdict
    beats_lexicon: bool | None
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the verdict bundle."""
        out = asdict(self)
        out["verdict"] = self.verdict.value
        return out


def derive_verdict(
    model_macro_f1: float | None,
    lexicon_macro_f1: float,
    mcnemar_p_value: float | None,
    *,
    alpha: float = 0.05,
    min_margin: float = 0.02,
) -> VerdictResult:
    r"""Derive the ``beats_lexicon`` verdict from measured numbers (pure function).

    Decision rule (truth-table unit-tested):

    * If ``model_macro_f1 is None`` (no transformer in this build) the verdict is
      :attr:`Verdict.LEXICON_ONLY` and ``beats_lexicon`` is ``None``.
    * Otherwise ``beats_lexicon`` is ``True`` if and only if BOTH:

      1. ``model_macro_f1 >= lexicon_macro_f1 + min_margin`` (a real margin), and
      2. ``mcnemar_p_value < alpha`` (McNemar rejects equal error rates).

      If either fails, the verdict is
      :attr:`Verdict.NO_SIGNIFICANT_DIFFERENCE` and ``beats_lexicon`` is ``False``.

    HONESTY REQUIREMENT: this function MUST NOT return ``True`` unless both the
    margin and the significance conditions hold — regardless of any other
    consideration.

    Parameters
    ----------
    model_macro_f1:
        The served transformer's macro-F1, or ``None`` if no transformer was
        trained/served in this build.
    lexicon_macro_f1:
        The lexicon baseline's macro-F1 on the same locked test set.
    mcnemar_p_value:
        The McNemar two-sided p-value (model vs. lexicon); may be ``None`` only
        when ``model_macro_f1`` is ``None``.
    alpha:
        Significance level for the McNemar test (default ``0.05``).
    min_margin:
        Minimum macro-F1 margin the model must clear (default ``0.02``).

    Returns
    -------
    VerdictResult
        The derived verdict, the (nullable) ``beats_lexicon`` boolean, and a
        rationale.

    Raises
    ------
    ValidationError
        If ``lexicon_macro_f1`` is outside ``[0, 1]``, or (when a transformer is
        present) ``model_macro_f1`` / ``mcnemar_p_value`` are out of range.
    """
    raise NotImplementedError
