"""ONNX inference session — the transformer SERVE path (onnxruntime + tokenizers).

The container and the FastAPI router run the fine-tuned DistilBERT through this
module ONLY. It loads the committed ``artifacts/model*.onnx`` graph with
onnxruntime and the ``artifacts/tokenizer.json`` with the ``tokenizers`` library;
torch and transformers are NEVER imported here. Both heavy imports happen LAZILY
inside :meth:`OnnxSentimentSession.load`, so ``import finbert_sentiment`` stays
free of any inference engine.

Importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

#: Directory holding the committed, shipped serve artifacts (ONNX + tokenizer).
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: Filenames of the committed serve artifacts.
ONNX_ARTIFACT_NAME: str = "model.int8.onnx"
TOKENIZER_ARTIFACT_NAME: str = "tokenizer.json"


def default_artifact_dir() -> Path:
    """Return the package's ``artifacts/`` directory (pure path arithmetic).

    Does NOT check existence and imports nothing heavy, so it is safe to call at
    a caller's import time.

    Returns
    -------
    pathlib.Path
        ``<package>/artifacts``.
    """
    return ARTIFACTS_DIR


def onnx_artifacts_present(artifact_dir: str | Path | None = None) -> bool:
    """Return ``True`` iff BOTH the ONNX graph and the tokenizer.json exist.

    Used by the predictor to decide whether the transformer backend is available
    or it must fall back to the lexicon. Pure filesystem check — imports nothing.

    Parameters
    ----------
    artifact_dir:
        Directory to probe (defaults to :func:`default_artifact_dir`).

    Returns
    -------
    bool
        Whether the transformer serve artifacts are both present.
    """
    raise NotImplementedError


class OnnxSentimentSession:
    """A thin onnxruntime + tokenizers wrapper that serves the committed model.

    The onnxruntime session and the tokenizer are created LAZILY on first
    :meth:`predict_proba` (or via :meth:`load`), so constructing this object is
    cheap and import-pure.
    """

    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        """Record the artifact directory; defer session/tokenizer creation to :meth:`load`.

        Parameters
        ----------
        artifact_dir:
            Directory holding ``model*.onnx`` + ``tokenizer.json``. Defaults to
            the package's shipped artifacts (:func:`default_artifact_dir`).
        """
        self._artifact_dir = Path(artifact_dir) if artifact_dir else default_artifact_dir()
        self._session: object | None = None
        self._tokenizer: object | None = None

    @property
    def artifact_dir(self) -> Path:
        """Return the resolved artifact directory this session will load from."""
        return self._artifact_dir

    def load(self) -> OnnxSentimentSession:
        """Create the onnxruntime session and load the tokenizer (lazy, idempotent).

        LAZY IMPORT: ``onnxruntime`` and ``tokenizers`` are imported inside this
        method. NO torch / transformers import occurs anywhere on this path.

        Returns
        -------
        OnnxSentimentSession
            ``self``, with an initialized session and tokenizer.

        Raises
        ------
        ArtifactError
            If either artifact is missing or initialization fails.
        """
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Tokenize, run the ONNX forward pass, and softmax to class scores.

        Loads the session on first use. Returns one row of
        ``(neg, neutral, pos)`` softmax probabilities per input text, in the
        canonical :data:`finbert_sentiment._constants.LABELS` order.

        Parameters
        ----------
        texts:
            The batch to classify.

        Returns
        -------
        numpy.ndarray
            A ``(len(texts), N_CLASSES)`` row-stochastic score matrix.

        Raises
        ------
        ArtifactError
            If the session/tokenizer cannot be loaded or the forward pass fails.
        ValidationError
            If ``texts`` fails the batch validation.
        """
        raise NotImplementedError

    def predict(self, texts: Sequence[str]) -> NDArray[np.int64]:
        """Return the argmax class index per text (thin wrapper over scores).

        Parameters
        ----------
        texts:
            The batch to classify.

        Returns
        -------
        numpy.ndarray
            A length-``len(texts)`` ``int64`` vector of class indices.
        """
        raise NotImplementedError
