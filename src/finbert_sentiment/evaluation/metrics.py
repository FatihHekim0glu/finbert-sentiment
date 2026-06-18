"""Honest classification metrics: macro-F1, per-class P/R/F1, confusion, bootstrap CIs.

The headline is **macro-F1** — the unweighted mean of per-class F1 — because the
PhraseBank neutral class dominates (~60%) and accuracy alone would flatter a
majority-class predictor. Every metric here is checked against ``sklearn.metrics``
to 1e-10 in the parity tests, and the macro-F1 confidence interval is computed by
SEEDED bootstrap resampling of the test set.

Importing this module has no side effects (numpy is the only heavy dependency and
is imported at module top-level; sklearn is NOT imported here).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from finbert_sentiment._constants import DEFAULT_BOOTSTRAP_RESAMPLES, DEFAULT_SEED

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> NDArray[np.int64]:
    """Return the ``(N_CLASSES, N_CLASSES)`` confusion matrix (rows=true, cols=pred).

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices (same length as ``y_true``).

    Returns
    -------
    numpy.ndarray
        An ``int64`` matrix; ``C[i, j]`` counts true-class-``i`` examples
        predicted as class ``j``.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, empty, or contain out-of-range indices.
    """
    raise NotImplementedError


def per_class_precision_recall_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return per-class ``(precision, recall, f1)`` vectors of length ``N_CLASSES``.

    Zero-division (a class never predicted, or never present) yields ``0.0`` for
    that class, matching ``sklearn``'s ``zero_division=0`` convention.

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The per-class precision, recall, and F1 vectors.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or invalid.
    """
    raise NotImplementedError


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return the macro-averaged F1 (unweighted mean of per-class F1).

    Matches ``sklearn.metrics.f1_score(..., average="macro", zero_division=0)`` to
    1e-10 (parity-tested). This is the project's headline metric.

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices.

    Returns
    -------
    float
        The macro-F1 in ``[0, 1]``.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or invalid.
    """
    raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Immutable bundle of the full honest metric set for one model on one test set.

    Attributes
    ----------
    macro_f1:
        The headline macro-averaged F1.
    accuracy:
        Plain accuracy (reported alongside, never instead of, macro-F1).
    per_class_precision:
        Per-class precision, length ``N_CLASSES``.
    per_class_recall:
        Per-class recall.
    per_class_f1:
        Per-class F1.
    confusion:
        The ``N_CLASSES x N_CLASSES`` confusion matrix as nested lists.
    macro_f1_ci:
        The ``(low, high)`` bootstrap CI for macro-F1 (``None`` if not computed).
    n:
        The number of test examples the report was computed on.
    """

    macro_f1: float
    accuracy: float
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    per_class_f1: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]
    n: int
    macro_f1_ci: tuple[float, float] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the report."""
        return asdict(self)


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    bootstrap_ci: bool = True,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> ClassificationReport:
    """Compute the full honest metric bundle for a model's test-set predictions.

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices.
    bootstrap_ci:
        Whether to attach a bootstrap macro-F1 confidence interval.
    n_resamples:
        Number of bootstrap resamples (when ``bootstrap_ci``).
    seed:
        Seed for the bootstrap resampling.

    Returns
    -------
    ClassificationReport
        The assembled metric bundle.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or invalid.
    """
    raise NotImplementedError


def bootstrap_macro_f1_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Return a SEEDED bootstrap ``(low, high)`` confidence interval for macro-F1.

    The test set is resampled with replacement ``n_resamples`` times (via
    :func:`finbert_sentiment._rng.make_rng`), macro-F1 is recomputed on each
    resample, and the ``confidence`` percentile interval is returned.

    Parameters
    ----------
    y_true:
        Integer true class indices.
    y_pred:
        Integer predicted class indices.
    n_resamples:
        Number of bootstrap resamples.
    confidence:
        Central interval mass (e.g. ``0.95`` for a 95% CI).
    seed:
        Master seed for reproducible resampling.

    Returns
    -------
    tuple[float, float]
        The lower and upper macro-F1 bounds.

    Raises
    ------
    ValidationError
        If the inputs are invalid or ``confidence`` is not in ``(0, 1)``.
    """
    raise NotImplementedError
