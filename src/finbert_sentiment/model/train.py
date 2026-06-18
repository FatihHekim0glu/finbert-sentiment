"""DistilBERT 3-way fine-tune on the Financial PhraseBank (``[train]`` only).

This module fine-tunes ``distilbert-base-uncased`` for 3-way sentiment using the
HuggingFace ``Trainer``, seeded and early-stopped on validation macro-F1. torch
and transformers are imported LAZILY inside :func:`train_distilbert` so that
``import finbert_sentiment`` never pulls in a deep-learning framework; the serve
container never installs ``[train]`` and never imports this module.

The label encoder and any class-weight statistics are computed on TRAIN ONLY
(the pretrained tokenizer is frozen, so it touching all text is fine). Importing
this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from finbert_sentiment._constants import DEFAULT_SEED

if TYPE_CHECKING:
    from finbert_sentiment.data.load import LabelledDataset
    from finbert_sentiment.data.split import SplitIndices


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Immutable hyper-parameters for the DistilBERT fine-tune.

    Attributes
    ----------
    model_name:
        HuggingFace base checkpoint to fine-tune.
    epochs:
        Maximum training epochs (early-stopping may stop sooner).
    batch_size:
        Per-device train/eval batch size.
    learning_rate:
        AdamW peak learning rate.
    max_length:
        Tokenizer truncation length.
    early_stopping_patience:
        Epochs without val-macro-F1 improvement before stopping.
    seed:
        Master seed (set on torch, numpy, and the Trainer).
    """

    model_name: str = "distilbert-base-uncased"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 128
    early_stopping_patience: int = 1
    seed: int = DEFAULT_SEED

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable outcome of a fine-tune run.

    Attributes
    ----------
    output_dir:
        Directory holding the saved torch model + tokenizer.
    best_val_macro_f1:
        Best validation macro-F1 reached during training.
    epochs_run:
        Number of epochs actually executed (<= ``config.epochs``).
    config:
        The :class:`TrainConfig` used.
    label_order:
        The class-name order the model's logits correspond to.
    """

    output_dir: str
    best_val_macro_f1: float
    epochs_run: int
    config: TrainConfig
    label_order: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (config flattened)."""
        out: dict[str, Any] = asdict(self)
        out["config"] = self.config.to_dict()
        return out


def train_distilbert(
    dataset: LabelledDataset,
    split: SplitIndices,
    config: TrainConfig | None = None,
    *,
    output_dir: str,
) -> TrainResult:
    """Fine-tune DistilBERT on the PhraseBank train fold, early-stopped on val macro-F1.

    LAZY IMPORT: ``torch`` and ``transformers`` are imported inside this function.
    The val fold drives early stopping; the locked test fold is NOT touched here.

    Parameters
    ----------
    dataset:
        The deduplicated labelled corpus.
    split:
        The locked train/val/test partition (only train+val are used here).
    config:
        Hyper-parameters (defaults to :class:`TrainConfig`).
    output_dir:
        Where to save the best model + tokenizer.

    Returns
    -------
    TrainResult
        The training outcome (best val macro-F1, epochs run, saved-model dir).

    Raises
    ------
    FinbertSentimentError
        If the ``[train]`` dependencies are missing or training fails.
    """
    raise NotImplementedError
