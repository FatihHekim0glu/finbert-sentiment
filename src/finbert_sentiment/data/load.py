"""Financial PhraseBank loader and the in-memory labelled-dataset container.

The loader pulls the HuggingFace ``financial_phrasebank`` dataset (config
``sentences_allagree``) and normalizes it into a :class:`LabelledDataset` of
``(texts, labels)`` with the canonical 3-way label space. ``datasets`` is
imported LAZILY inside :func:`load_phrasebank` so that ``import
finbert_sentiment`` never triggers a network call or a heavy import. Tests use an
offline-cached sample baked into the conftest fixture, never the network.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class LabelledDataset:
    """An immutable, aligned ``(texts, labels)`` corpus with provenance.

    Attributes
    ----------
    texts:
        The raw sentences, one per example (parallel to ``labels``).
    labels:
        Integer class indices in ``{0, 1, 2}`` (see
        :data:`finbert_sentiment._constants.LABELS`), one per text.
    source:
        Human-readable provenance string (e.g.
        ``"financial_phrasebank/sentences_allagree"`` or
        ``"offline-cached-sample"``).
    """

    texts: tuple[str, ...]
    labels: tuple[int, ...]
    source: str = "unknown"

    def __post_init__(self) -> None:
        """Validate that ``texts`` and ``labels`` are aligned and non-empty."""
        raise NotImplementedError

    @property
    def n(self) -> int:
        """Return the number of examples."""
        raise NotImplementedError

    def class_counts(self) -> dict[str, int]:
        """Return a ``{class_name: count}`` mapping over the 3-way label space."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this dataset."""
        return asdict(self)


def load_phrasebank(
    *,
    config: str | None = None,
    cache_dir: str | None = None,
) -> LabelledDataset:
    """Load the Financial PhraseBank into a :class:`LabelledDataset`.

    LAZY IMPORT: ``datasets`` is imported inside this function. The HuggingFace
    label ids (``0=negative, 1=neutral, 2=positive`` for PhraseBank) are mapped
    onto the canonical :data:`finbert_sentiment._constants.LABELS` order.

    Parameters
    ----------
    config:
        Dataset config name (defaults to
        :data:`finbert_sentiment._constants.PHRASEBANK_CONFIG`).
    cache_dir:
        Optional HuggingFace datasets cache directory.

    Returns
    -------
    LabelledDataset
        The full PhraseBank corpus (pre-dedup), tagged with its source.

    Raises
    ------
    FinbertSentimentError
        If the dataset cannot be downloaded or parsed.
    """
    raise NotImplementedError


def sample_dataset(
    texts: Sequence[str],
    labels: Sequence[int | str],
    *,
    source: str = "offline-cached-sample",
) -> LabelledDataset:
    """Build a :class:`LabelledDataset` from in-memory texts/labels (no network).

    The constructor used by the offline test fixture and by the FastAPI demo's
    bundled example headlines. String labels are normalized to integer indices.

    Parameters
    ----------
    texts:
        The sentences.
    labels:
        Integer indices or canonical string class names, one per text.
    source:
        Provenance tag stored on the returned dataset.

    Returns
    -------
    LabelledDataset
        The validated, aligned corpus.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, empty, or contain invalid labels.
    """
    raise NotImplementedError
