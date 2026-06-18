"""Export a fine-tuned DistilBERT to ONNX + dynamic int8 (``[train]`` only).

This converts a saved torch model into the lean serve artifact pair the
container actually ships: an ONNX graph (dynamic int8-quantized) plus the
``tokenizer.json``. torch / transformers / onnx are imported LAZILY inside
:func:`export_to_onnx`; the serve container never installs ``[train]`` and never
imports this module. The exported artifacts are validated for logit parity
against the torch model by the parity tests (tolerance 1e-3) when ``[train]`` ran.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Immutable record of an ONNX export.

    Attributes
    ----------
    onnx_path:
        Path to the exported (int8-quantized) ONNX graph.
    tokenizer_path:
        Path to the exported ``tokenizer.json``.
    int8:
        Whether dynamic int8 quantization was applied.
    opset:
        The ONNX opset version used.
    max_logit_abs_diff:
        Max absolute logit difference vs. the torch model over a probe batch
        (``None`` if parity was not checked at export time).
    """

    onnx_path: str
    tokenizer_path: str
    int8: bool
    opset: int
    max_logit_abs_diff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this export."""
        return asdict(self)


def export_to_onnx(
    model_dir: str,
    *,
    output_dir: str,
    int8: bool = True,
    opset: int = 14,
) -> ExportResult:
    """Export a saved torch DistilBERT to ONNX (optionally int8) + tokenizer.json.

    LAZY IMPORT: ``torch``, ``transformers``, and ``onnx`` are imported inside
    this function. The exported ONNX graph takes ``input_ids`` + ``attention_mask``
    and emits 3-way logits in the canonical
    :data:`finbert_sentiment._constants.LABELS` order.

    Parameters
    ----------
    model_dir:
        Directory holding the saved torch model + tokenizer (from
        :func:`finbert_sentiment.model.train.train_distilbert`).
    output_dir:
        Where to write ``model.onnx`` (or ``model.int8.onnx``) and
        ``tokenizer.json``.
    int8:
        If ``True``, apply dynamic int8 quantization to the exported graph.
    opset:
        ONNX opset version.

    Returns
    -------
    ExportResult
        Paths to the exported artifacts and the quantization/opset metadata.

    Raises
    ------
    FinbertSentimentError
        If the ``[train]`` dependencies are missing or export fails.
    """
    raise NotImplementedError
