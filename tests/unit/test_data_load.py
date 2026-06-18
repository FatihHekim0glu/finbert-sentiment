"""Unit tests for the data loader + ``LabelledDataset`` container (offline, no network).

These exercise :func:`finbert_sentiment.data.load.sample_dataset` and the
:class:`~finbert_sentiment.data.load.LabelledDataset` validation/accessors. The
network loader :func:`load_phrasebank` is NOT called here (no network in tests);
its lazy-import contract is covered by the import-purity integration test.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from finbert_sentiment._exceptions import FinbertSentimentError, ValidationError
from finbert_sentiment.data.load import LabelledDataset, load_phrasebank, sample_dataset

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit


class _FakeHFDataset:
    """A minimal stand-in for a HuggingFace ``Dataset`` (column access only)."""

    def __init__(self, sentences: list[str], labels: list[int]) -> None:
        self._cols: dict[str, list[Any]] = {"sentence": sentences, "label": labels}

    def __getitem__(self, key: str) -> list[Any]:
        return self._cols[key]


@pytest.fixture
def fake_datasets(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Install a fake ``datasets`` module so ``load_phrasebank`` runs offline.

    The loader imports ``datasets`` lazily inside the function, so injecting a
    fake module into ``sys.modules`` lets us exercise the loader's mapping and
    error-handling logic with NO network and NO real ``datasets`` dependency.
    """
    state: dict[str, Any] = {"args": None, "raise": None, "return": None}

    def _load_dataset(*args: Any, **kwargs: Any) -> _FakeHFDataset:
        state["args"] = (args, kwargs)
        if state["raise"] is not None:
            raise state["raise"]
        result: _FakeHFDataset = state["return"]
        return result

    fake = types.ModuleType("datasets")
    fake.load_dataset = _load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    yield state


def test_labelled_dataset_basic_accessors(phrasebank_sample: LabelledDataset) -> None:
    """``n`` and ``class_counts`` agree with the fixture content."""
    assert phrasebank_sample.n == 18
    counts = phrasebank_sample.class_counts()
    assert set(counts) == {"negative", "neutral", "positive"}
    assert sum(counts.values()) == phrasebank_sample.n
    # The fixture is balanced 6/6/6 across the three classes.
    assert counts == {"negative": 6, "neutral": 6, "positive": 6}


def test_to_dict_is_json_serializable(phrasebank_sample: LabelledDataset) -> None:
    """``to_dict`` returns a plain, round-trippable mapping."""
    import json

    payload = phrasebank_sample.to_dict()
    assert set(payload) == {"texts", "labels", "source"}
    restored = json.loads(json.dumps(payload))
    assert list(restored["texts"]) == list(phrasebank_sample.texts)
    assert list(restored["labels"]) == list(phrasebank_sample.labels)


def test_sample_dataset_accepts_string_labels() -> None:
    """String class names are normalized to integer indices in canonical order."""
    ds = sample_dataset(
        ["a loss", "steady", "a gain"],
        ["negative", "neutral", "positive"],
    )
    assert ds.labels == (0, 1, 2)
    assert ds.source == "offline-cached-sample"


def test_sample_dataset_accepts_int_labels() -> None:
    """Integer indices pass through unchanged."""
    ds = sample_dataset(["x", "y"], [2, 0], source="custom")
    assert ds.labels == (2, 0)
    assert ds.source == "custom"


def test_sample_dataset_rejects_bare_string() -> None:
    """A bare string for ``texts`` is a caller mistake and is rejected."""
    with pytest.raises(ValidationError):
        sample_dataset("not a list", [0])


def test_sample_dataset_rejects_empty() -> None:
    """An empty corpus is rejected."""
    with pytest.raises(ValidationError):
        sample_dataset([], [])


def test_sample_dataset_rejects_misaligned() -> None:
    """A texts/labels length mismatch is rejected."""
    with pytest.raises(ValidationError):
        sample_dataset(["a", "b"], [0])


def test_sample_dataset_rejects_unknown_label() -> None:
    """An out-of-vocabulary string label is rejected."""
    with pytest.raises(ValidationError):
        sample_dataset(["a"], ["bullish"])


def test_labelled_dataset_rejects_out_of_range_label() -> None:
    """Direct construction with an out-of-range index is rejected."""
    with pytest.raises(ValidationError):
        LabelledDataset(texts=("a",), labels=(7,))


def test_labelled_dataset_rejects_blank_text() -> None:
    """A blank/whitespace-only sentence is rejected."""
    with pytest.raises(ValidationError):
        LabelledDataset(texts=("   ",), labels=(1,))


def test_labelled_dataset_rejects_misaligned_direct() -> None:
    """Direct construction with misaligned texts/labels is rejected."""
    with pytest.raises(ValidationError):
        LabelledDataset(texts=("a", "b"), labels=(1,))


def test_labelled_dataset_rejects_empty_direct() -> None:
    """Direct construction of an empty dataset is rejected."""
    with pytest.raises(ValidationError):
        LabelledDataset(texts=(), labels=())


def test_labelled_dataset_rejects_non_string_text() -> None:
    """A non-string text element is rejected by ``__post_init__``."""
    with pytest.raises(ValidationError):
        LabelledDataset(texts=(123,), labels=(1,))  # type: ignore[arg-type]


def test_labelled_dataset_rejects_non_int_label() -> None:
    """A non-int (e.g. bool) label element is rejected by ``__post_init__``."""
    with pytest.raises(ValidationError):
        LabelledDataset(texts=("a",), labels=(True,))


# --------------------------------------------------------------------------- #
# load_phrasebank (offline, via an injected fake `datasets` module)            #
# --------------------------------------------------------------------------- #
def test_load_phrasebank_maps_labels_and_source(fake_datasets: dict[str, Any]) -> None:
    """The loader maps HF label ids to canonical indices and tags the source."""
    fake_datasets["return"] = _FakeHFDataset(
        ["a loss widened", "steady meeting", "profit rose"],
        [0, 1, 2],
    )
    ds = load_phrasebank()
    assert isinstance(ds, LabelledDataset)
    assert ds.labels == (0, 1, 2)
    assert ds.source == "financial_phrasebank/sentences_allagree"
    # The default config was forwarded to datasets.load_dataset.
    args, _kwargs = fake_datasets["args"]
    assert args[0] == "financial_phrasebank"
    assert args[1] == "sentences_allagree"


def test_load_phrasebank_honours_explicit_config(fake_datasets: dict[str, Any]) -> None:
    """A caller-supplied config name flows through to the source tag."""
    fake_datasets["return"] = _FakeHFDataset(["x"], [1])
    ds = load_phrasebank(config="sentences_50agree")
    assert ds.source == "financial_phrasebank/sentences_50agree"


def test_load_phrasebank_wraps_loader_failure(fake_datasets: dict[str, Any]) -> None:
    """Any failure inside ``datasets.load_dataset`` becomes a library error."""
    fake_datasets["raise"] = RuntimeError("network down")
    with pytest.raises(FinbertSentimentError):
        load_phrasebank()


def test_load_phrasebank_rejects_unexpected_label_id(fake_datasets: dict[str, Any]) -> None:
    """An out-of-vocabulary HF label id is rejected with a library error."""
    fake_datasets["return"] = _FakeHFDataset(["weird"], [9])
    with pytest.raises(FinbertSentimentError):
        load_phrasebank()


def test_load_phrasebank_falls_back_to_parquet_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the canonical script loader fails, the parquet mirror is used (train+test).

    Simulates ``datasets >= 3`` rejecting the script-based canonical repo: the
    first repo id raises, and the mirror repo returns per-split fake data, which
    the loader concatenates into the full corpus tagged with the mirror source.
    """
    calls: list[tuple[str, str]] = []

    def _load_dataset(repo: str, config: str, *, split: str, cache_dir: Any = None) -> Any:
        calls.append((repo, split))
        if repo == "financial_phrasebank":
            raise RuntimeError("Dataset scripts are no longer supported")
        # The parquet mirror: distinct rows per split, concatenated by the loader.
        per_split = {
            "train": _FakeHFDataset(["won a contract"], [2]),
            "test": _FakeHFDataset(["profit down"], [0]),
        }
        return per_split[split]

    fake = types.ModuleType("datasets")
    fake.load_dataset = _load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)

    ds = load_phrasebank()
    # The canonical attempt was made first, then both mirror splits were read.
    assert calls[0][0] == "financial_phrasebank"
    assert ("gtfintechlab/financial_phrasebank_sentences_allagree", "train") in calls
    assert ("gtfintechlab/financial_phrasebank_sentences_allagree", "test") in calls
    # The full corpus is the concatenation of train + test, mirror-tagged.
    assert ds.n == 2
    assert ds.source.startswith("gtfintechlab/financial_phrasebank_sentences_allagree")
    assert ds.class_counts() == {"negative": 1, "neutral": 0, "positive": 1}


def test_load_phrasebank_wraps_when_both_canonical_and_mirror_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If BOTH the canonical loader and the parquet mirror fail, a library error is raised."""

    def _load_dataset(
        repo: str, config: str, *, split: str = "train", cache_dir: Any = None
    ) -> Any:
        raise RuntimeError(f"unreachable: {repo}")

    fake = types.ModuleType("datasets")
    fake.load_dataset = _load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)

    with pytest.raises(FinbertSentimentError, match="parquet mirror"):
        load_phrasebank()
