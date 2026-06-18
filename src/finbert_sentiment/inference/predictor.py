"""Unified predictor over the transformer-ONNX OR lexicon backend.

:class:`Predictor` exposes a single ``predict(texts) -> list[Prediction]``
interface and dispatches to whichever backend is available: the transformer-ONNX
session when the committed artifacts are present, else the torch-free lexicon
classifier. The selected backend is recorded so the API can report which model it
actually served. Nothing here imports a heavy dependency at module load.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The backend identifiers surfaced to callers / the API ``served_model`` field.
BackendName = Literal["distilbert-onnx", "lexicon"]

#: The model the caller *requests* (the API may fall back to the lexicon).
RequestedModel = Literal["distilbert", "lexicon"]


@dataclass(frozen=True, slots=True)
class Prediction:
    """An immutable per-text prediction: label + the 3-way score vector.

    Attributes
    ----------
    text:
        The input sentence (echoed back for alignment in the UI table).
    label:
        The predicted class name (one of
        :data:`finbert_sentiment._constants.LABELS`).
    scores:
        ``{"negative": p, "neutral": p, "positive": p}`` scores summing to one.
    """

    text: str
    label: str
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this prediction."""
        return asdict(self)


class Predictor:
    """Backend-agnostic sentiment predictor (transformer-ONNX or lexicon).

    Construct via :func:`load_predictor` (which picks the backend by artifact
    availability) rather than directly. The chosen backend is fixed for the
    lifetime of the instance and exposed via :attr:`backend`.
    """

    def __init__(self, backend: BackendName, *, session: object | None = None) -> None:
        """Wire up the predictor around a concrete backend.

        Parameters
        ----------
        backend:
            Which backend this predictor serves.
        session:
            The backend object (an ``OnnxSentimentSession`` or a
            ``LexiconClassifier``); kept loosely typed so this module imports
            nothing heavy at load time.
        """
        self._backend: BackendName = backend
        self._session = session

    @property
    def backend(self) -> BackendName:
        """Return the backend identifier this predictor serves."""
        return self._backend

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify a batch and return one :class:`Prediction` per input text.

        Parameters
        ----------
        texts:
            The batch to classify (1..64 non-empty sentences).

        Returns
        -------
        list[Prediction]
            One prediction per input, in input order.

        Raises
        ------
        ValidationError
            If ``texts`` fails the batch validation.
        ArtifactError
            If the transformer backend is selected but its artifacts fail to load.
        """
        raise NotImplementedError


def load_predictor(
    requested: RequestedModel = "distilbert",
    *,
    artifact_dir: str | None = None,
) -> Predictor:
    """Build a :class:`Predictor`, selecting the backend by availability.

    If ``requested == "distilbert"`` AND the ONNX + tokenizer artifacts are
    present, the transformer backend is used; otherwise the predictor
    transparently falls back to the torch-free lexicon backend. ``requested ==
    "lexicon"`` always uses the lexicon.

    Parameters
    ----------
    requested:
        The model the caller would prefer.
    artifact_dir:
        Override for the serve-artifact directory (defaults to the package's
        ``artifacts/``).

    Returns
    -------
    Predictor
        A predictor bound to the resolved backend.
    """
    raise NotImplementedError
