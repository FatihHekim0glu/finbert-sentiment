"""McNemar's test: is the model's error pattern significantly different from the lexicon?

Macro-F1 alone cannot say whether a gap is statistically real. McNemar's test
compares two classifiers on the SAME test set by looking only at the discordant
pairs — examples one model gets right and the other gets wrong — and asks whether
that discordance is lopsided beyond chance. We use the exact binomial form for
small discordant counts and the continuity-corrected chi-square form otherwise.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Immutable outcome of a McNemar model-vs-lexicon comparison.

    Attributes
    ----------
    n01:
        Count of examples the lexicon got right but the model got wrong.
    n10:
        Count of examples the model got right but the lexicon got wrong.
    statistic:
        The test statistic (chi-square with continuity correction, or ``nan``
        when the exact binomial branch was used).
    p_value:
        The two-sided p-value.
    exact:
        Whether the exact binomial test was used (small discordant counts).
    """

    n01: int
    n10: int
    statistic: float
    p_value: float
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the result."""
        return asdict(self)


def mcnemar_test(
    y_true: Sequence[int],
    y_pred_model: Sequence[int],
    y_pred_lexicon: Sequence[int],
    *,
    exact_threshold: int = 25,
) -> McNemarResult:
    """Run McNemar's test comparing the model against the lexicon on one test set.

    Both prediction vectors must be aligned to the same ``y_true``. When the
    number of discordant pairs (``n01 + n10``) is below ``exact_threshold`` the
    exact binomial test is used; otherwise the continuity-corrected chi-square
    approximation is used.

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred_model:
        The model's predicted class indices.
    y_pred_lexicon:
        The lexicon baseline's predicted class indices.
    exact_threshold:
        Discordant-pair count below which the exact binomial branch is taken.

    Returns
    -------
    McNemarResult
        The discordant counts, statistic, p-value, and which branch was used.

    Raises
    ------
    ValidationError
        If the three vectors are misaligned, empty, or contain invalid indices.
    """
    raise NotImplementedError
