"""Parity tests: our metrics vs. an independent reference (sklearn), and ONNX vs. torch.

These pin our hand-rolled metric kernels to ``sklearn.metrics`` to 1e-10 and (when
the ``[train]`` extra produced a model) the ONNX logits to the torch model to
1e-3. They are skipped until the corresponding kernels are implemented by the
``evaluation`` / ``model`` modules.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.parity

_PENDING = "pending implementation of finbert_sentiment.evaluation.metrics"


@pytest.mark.skip(reason=_PENDING)
def test_macro_f1_matches_sklearn() -> None:
    """``macro_f1`` matches ``sklearn.metrics.f1_score(average='macro')`` to 1e-10."""
    raise AssertionError("implement against sklearn once metrics.macro_f1 lands")


@pytest.mark.skip(reason=_PENDING)
def test_per_class_prf_matches_sklearn() -> None:
    """Per-class precision/recall/F1 match sklearn's per-class report to 1e-10."""
    raise AssertionError("implement against sklearn once metrics land")


@pytest.mark.skip(reason=_PENDING)
def test_confusion_matrix_matches_sklearn() -> None:
    """``confusion_matrix`` matches ``sklearn.metrics.confusion_matrix`` exactly."""
    raise AssertionError("implement against sklearn once metrics land")


@pytest.mark.train
@pytest.mark.skip(reason="pending the [train] DistilBERT fine-tune + ONNX export")
def test_onnx_logits_match_torch() -> None:
    """Exported ONNX logits match the torch model to 1e-3 (only when [train] ran)."""
    raise AssertionError("implement once model.train + model.export land")
