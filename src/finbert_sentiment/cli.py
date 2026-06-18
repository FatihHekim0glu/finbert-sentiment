"""Command-line interface (Typer): train / evaluate / predict.

A thin orchestration layer over the compute library. Typer (and, on the train
path, torch/transformers) are imported LAZILY inside :func:`build_app` / the
command bodies, so importing :mod:`finbert_sentiment.cli` registers no commands
and does no I/O. The module-level entry point :func:`main` builds the app lazily
and runs it, backing the ``finbert-sentiment`` console script.

The three commands map to the honest workflow:

- ``train``    — run the offline DistilBERT fine-tune on the Financial PhraseBank
  (load -> dedup -> seeded group split -> fine-tune -> ONNX+int8 export). This is
  the ONLY command that may touch torch/transformers, and only via the lazy
  ``[train]`` path inside :mod:`finbert_sentiment.model`.
- ``evaluate`` — compute the honest metric bundle (macro-F1 + per-class P/R/F1 +
  confusion + bootstrap CIs) and the McNemar test vs. the lexicon on the LOCKED
  test set, for the lexicon and/or the served transformer. NEVER accuracy alone.
- ``predict``  — classify input sentences WITHOUT torch: it serves the committed
  ONNX artifact via onnxruntime when present, otherwise falls back to the
  torch-free lexicon. Either way no training engine is imported.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers ``train``, ``evaluate``, and ``predict`` on a fresh ``typer.Typer``.
    Typer is imported lazily inside this function so importing
    :mod:`finbert_sentiment.cli` does not import Typer or register any commands. A
    fresh instance is returned on every call (no shared mutable state).

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    raise NotImplementedError


def main() -> None:
    """Entry point for the ``finbert-sentiment`` console script.

    Builds the Typer app lazily via :func:`build_app` and invokes it. Kept tiny so
    the console-script import path stays free of Typer until the command actually
    runs.
    """
    raise NotImplementedError
