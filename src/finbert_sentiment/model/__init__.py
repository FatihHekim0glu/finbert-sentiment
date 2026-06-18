"""Offline training and ONNX export (the ``[train]`` path — never in the container).

Everything in this subpackage is gated behind the heavy ``[train]`` extra
(torch + transformers + onnx) and is imported LAZILY inside functions, so
importing this subpackage stays free of any deep-learning framework. The serve
container installs only ``[serve]`` (onnxruntime + tokenizers) and never touches
these modules. Importing this subpackage has no side effects.
"""

from __future__ import annotations

from finbert_sentiment.model.export import ExportResult, export_to_onnx
from finbert_sentiment.model.train import TrainConfig, TrainResult, train_distilbert

__all__ = [
    "ExportResult",
    "TrainConfig",
    "TrainResult",
    "export_to_onnx",
    "train_distilbert",
]
