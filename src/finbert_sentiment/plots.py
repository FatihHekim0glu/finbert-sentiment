"""Plotly figure builders (no Plotly dependency at import; ``viz`` extra renders).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same Plotly-schema JSON the FastAPI layer serializes and the Next.js
``PlotlyChart`` component renders — built directly from native Python types, so no
Plotly object ever crosses the API boundary and the builders do not import Plotly
at all. Plotly/kaleido (the OPTIONAL ``viz`` extra) are only needed downstream to
*render* these dicts to an image; importing this module has no side effects.

The two figures back the honest story: the test-set confusion matrix (where the
neutral-heavy errors live) and the per-class F1 bar (which class the macro-F1
average is being dragged down by).

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

# quantcore-candidate: mirrors hrp / lstm-forecast plots.py ({data, layout} shape).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def confusion_matrix_figure(
    confusion: Sequence[Sequence[int]] | NDArray[np.int64],
    *,
    normalize: bool = False,
    title: str = "Confusion matrix (test set)",
) -> FigureDict:
    """Build a confusion-matrix heatmap figure over the 3-way label space.

    The heatmap axes are the canonical
    :data:`finbert_sentiment._constants.LABELS` (rows=true, cols=pred). When
    ``normalize`` is set, each row is divided by its sum (row-normalized recall
    view); otherwise raw counts are shown.

    Parameters
    ----------
    confusion:
        A ``N_CLASSES x N_CLASSES`` matrix of counts (rows=true, cols=pred).
    normalize:
        Whether to row-normalize the displayed values.
    title:
        Figure title.

    Returns
    -------
    FigureDict
        A ``{data, layout}`` mapping (a single ``heatmap`` trace).

    Raises
    ------
    ValidationError
        If ``confusion`` is not a square ``N_CLASSES``-sized matrix.
    """
    raise NotImplementedError


def per_class_f1_figure(
    per_class_f1: Sequence[float] | NDArray[np.float64],
    *,
    title: str = "Per-class F1 (test set)",
) -> FigureDict:
    """Build a per-class F1 bar-chart figure over the 3-way label space.

    One bar per class (negative/neutral/positive); the chart makes visible which
    class the macro-F1 average is being held down by.

    Parameters
    ----------
    per_class_f1:
        Per-class F1 values, length ``N_CLASSES``, in
        :data:`finbert_sentiment._constants.LABELS` order.
    title:
        Figure title.

    Returns
    -------
    FigureDict
        A ``{data, layout}`` mapping (a single ``bar`` trace).

    Raises
    ------
    ValidationError
        If ``per_class_f1`` does not have length ``N_CLASSES`` or contains
        out-of-range values.
    """
    raise NotImplementedError
