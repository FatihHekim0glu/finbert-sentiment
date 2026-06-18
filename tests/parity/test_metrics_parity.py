"""Parity tests: our metrics vs. an independent reference (sklearn), and ONNX vs. torch.

These pin our hand-rolled metric kernels to ``sklearn.metrics`` to 1e-10 and (when
the ``[train]`` extra produced a model) the ONNX logits to the torch model to
1e-3. The metric-parity tests run today against sklearn; the ONNX-vs-torch test
stays skipped until the ``[train]`` DistilBERT fine-tune + export lands.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)
from sklearn.metrics import (
    f1_score as sk_f1_score,
)
from sklearn.metrics import (
    precision_recall_fscore_support as sk_prf,
)

from finbert_sentiment._constants import N_CLASSES
from finbert_sentiment._rng import make_rng
from finbert_sentiment.evaluation.metrics import (
    confusion_matrix,
    macro_f1,
    per_class_precision_recall_f1,
)

pytestmark = pytest.mark.parity

_LABELS = list(range(N_CLASSES))


def _random_pair(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A reproducible random ``(y_true, y_pred)`` pair over the 3-way label space."""
    rng = make_rng(seed)
    y_true = rng.integers(0, N_CLASSES, size=n)
    y_pred = rng.integers(0, N_CLASSES, size=n)
    return y_true, y_pred


# A spread of deterministic scenarios, including ones where a class is never
# predicted / never present (exercising the zero_division=0 branch) and a
# perfect-prediction case.
_SCENARIOS: tuple[tuple[list[int], list[int]], ...] = (
    # perfect predictions
    ([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]),
    # class 2 never predicted (zero-division in precision for class 2)
    ([0, 1, 2, 2, 1, 0], [0, 1, 0, 1, 1, 0]),
    # class 0 never present in y_true (zero-division in recall for class 0)
    ([1, 1, 2, 2, 1, 2], [0, 1, 2, 1, 1, 0]),
    # everything predicted as the majority neutral class
    ([0, 1, 2, 0, 1, 2, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
    # all wrong
    ([0, 0, 0], [1, 1, 1]),
)


@pytest.mark.parametrize("y_true,y_pred", _SCENARIOS)
def test_macro_f1_matches_sklearn(y_true: list[int], y_pred: list[int]) -> None:
    """``macro_f1`` matches ``sklearn.metrics.f1_score(average='macro')`` to 1e-10."""
    ours = macro_f1(y_true, y_pred)
    ref = float(sk_f1_score(y_true, y_pred, labels=_LABELS, average="macro", zero_division=0))
    assert abs(ours - ref) < 1e-10


@pytest.mark.parametrize("seed", [1, 7, 42, 20260618])
def test_macro_f1_matches_sklearn_random(seed: int) -> None:
    """Random batches: macro-F1 matches sklearn to 1e-10."""
    y_true, y_pred = _random_pair(seed, n=200)
    ours = macro_f1(y_true, y_pred)
    ref = float(sk_f1_score(y_true, y_pred, labels=_LABELS, average="macro", zero_division=0))
    assert abs(ours - ref) < 1e-10


@pytest.mark.parametrize("y_true,y_pred", _SCENARIOS)
def test_per_class_prf_matches_sklearn(y_true: list[int], y_pred: list[int]) -> None:
    """Per-class precision/recall/F1 match sklearn's per-class report to 1e-10."""
    p, r, f = per_class_precision_recall_f1(y_true, y_pred)
    ref_p, ref_r, ref_f, _ = sk_prf(y_true, y_pred, labels=_LABELS, average=None, zero_division=0)
    np.testing.assert_allclose(p, ref_p, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(r, ref_r, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(f, ref_f, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("seed", [3, 11, 99])
def test_per_class_prf_matches_sklearn_random(seed: int) -> None:
    """Random batches: per-class P/R/F1 match sklearn to 1e-10."""
    y_true, y_pred = _random_pair(seed, n=150)
    p, r, f = per_class_precision_recall_f1(y_true, y_pred)
    ref_p, ref_r, ref_f, _ = sk_prf(y_true, y_pred, labels=_LABELS, average=None, zero_division=0)
    np.testing.assert_allclose(p, ref_p, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(r, ref_r, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(f, ref_f, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("y_true,y_pred", _SCENARIOS)
def test_confusion_matrix_matches_sklearn(y_true: list[int], y_pred: list[int]) -> None:
    """``confusion_matrix`` matches ``sklearn.metrics.confusion_matrix`` exactly."""
    ours = confusion_matrix(y_true, y_pred)
    ref = sk_confusion_matrix(y_true, y_pred, labels=_LABELS)
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.parametrize("seed", [5, 13, 21])
def test_confusion_matrix_matches_sklearn_random(seed: int) -> None:
    """Random batches: confusion matrix matches sklearn exactly."""
    y_true, y_pred = _random_pair(seed, n=300)
    ours = confusion_matrix(y_true, y_pred)
    ref = sk_confusion_matrix(y_true, y_pred, labels=_LABELS)
    np.testing.assert_array_equal(ours, ref)


def test_confusion_matrix_row_col_semantics() -> None:
    """Rows index the true class, columns the predicted class (sklearn convention)."""
    # two true-0 examples both predicted as class 1 -> C[0, 1] == 2
    cm = confusion_matrix([0, 0], [1, 1])
    assert int(cm[0, 1]) == 2
    assert int(cm[0, 0]) == 0


@pytest.mark.parametrize("y_true,y_pred", _SCENARIOS)
def test_macro_f1_is_mean_of_per_class_f1(y_true: list[int], y_pred: list[int]) -> None:
    """macro-F1 equals the unweighted mean of the per-class F1 vector."""
    _, _, f = per_class_precision_recall_f1(y_true, y_pred)
    assert abs(macro_f1(y_true, y_pred) - float(np.mean(f))) < 1e-12


def test_string_labels_match_int_labels() -> None:
    """String class names and integer indices yield identical metrics."""
    str_true = ["negative", "neutral", "positive", "neutral"]
    str_pred = ["negative", "positive", "positive", "neutral"]
    int_true = [0, 1, 2, 1]
    int_pred = [0, 2, 2, 1]
    assert abs(macro_f1(str_true, str_pred) - macro_f1(int_true, int_pred)) < 1e-12
    np.testing.assert_array_equal(
        confusion_matrix(str_true, str_pred), confusion_matrix(int_true, int_pred)
    )


def _committed_torch_model_dir() -> str | None:
    """Return the committed serve-artifact dir IFF it holds a saved torch model.

    The fine-tuned ``model.safetensors`` (+ ``config.json``) lives next to the
    shipped ONNX/tokenizer when ``[train]`` ran locally; it is gitignored (only
    the ONNX + tokenizer + metrics ship). When absent, the ONNX-vs-torch parity
    test self-skips — it cannot re-derive torch logits without the saved weights.
    """
    import os

    artifacts = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "src",
        "finbert_sentiment",
        "artifacts",
    )
    has_weights = os.path.isfile(os.path.join(artifacts, "model.safetensors"))
    has_config = os.path.isfile(os.path.join(artifacts, "config.json"))
    return artifacts if (has_weights and has_config) else None


@pytest.mark.train
@pytest.mark.slow
@pytest.mark.skipif(
    _committed_torch_model_dir() is None,
    reason="requires the saved torch model from a local [train] run (gitignored)",
)
def test_onnx_logits_match_torch(tmp_path: object) -> None:  # pragma: no cover - [train]-only
    """The fp32 ONNX export matches the torch model's logits to 1e-3.

    Re-exports the committed saved torch model to a fresh fp32 ONNX graph and
    compares onnxruntime logits against the torch forward pass on a fixed probe
    batch. The export helper measures this max-abs logit diff itself; this test
    pins it under the 1e-3 tolerance the brief requires (the shipped int8 graph
    carries extra quantization error and is validated separately by the serve
    path's score-level checks).
    """
    from finbert_sentiment.model.export import export_to_onnx

    model_dir = _committed_torch_model_dir()
    assert model_dir is not None  # guarded by skipif
    result = export_to_onnx(
        model_dir,
        output_dir=str(tmp_path),  # type: ignore[arg-type]
        int8=False,
        opset=14,
    )
    assert result.max_logit_abs_diff is not None
    assert result.max_logit_abs_diff < 1e-3
