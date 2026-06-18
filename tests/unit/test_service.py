"""Unit tests for the public service entrypoint (:mod:`finbert_sentiment.service`).

Covers the backend ``run_sentiment`` contract WITHOUT torch or the network:

- The committed-metrics loader reads ``artifacts/metrics.json`` verbatim and
  raises :class:`ArtifactError` on a missing / malformed file.
- ``run_sentiment`` (lexicon path) returns the honest summary (eval scalars from
  the committed metrics, NOT recomputed), one prediction per input, and the two
  well-formed ``{data, layout}`` figures, all JSON-serializable.
- ``eval_macro_f1`` equals the committed value (proof it is loaded, not
  recomputed on the unlabeled request texts).
- ``_safe_float`` coerces NaN/inf/None/garbage to ``None``.
- The figure builder fails loudly when the committed metrics lack a confusion /
  per-class-F1 block.

The lexicon path is self-contained and torch-free; nothing here trains a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from finbert_sentiment._exceptions import ArtifactError, ValidationError
from finbert_sentiment.service import (
    SentimentResult,
    SentimentSummary,
    _ci_tuple,
    _safe_float,
    build_evaluation_figures,
    default_metrics_path,
    load_committed_metrics,
    run_sentiment,
)

pytestmark = pytest.mark.unit


# A minimal, self-consistent committed-metrics bundle (lexicon-only fallback).
_FAKE_METRICS: dict[str, Any] = {
    "schema_version": 1,
    "served_model": "lexicon",
    "data_source": "financial_phrasebank/sentences_allagree",
    "eval_macro_f1": 0.6525,
    "eval_macro_f1_ci": [0.5944, 0.7098],
    "eval_accuracy": 0.7616,
    "lexicon_macro_f1": 0.6525,
    "class_prior_macro_f1": 0.2535,
    "beats_lexicon": None,
    "verdict": "lexicon_only",
    "per_class_f1": [0.5053, 0.8444, 0.6077],
    "confusion": [[24, 33, 4], [4, 266, 8], [6, 53, 55]],
    "labels": ["negative", "neutral", "positive"],
    "transformer_published_macro_f1": {"measured_in_this_build": False},
    "notes": "Sentiment is a text label, not a tradable signal — no alpha is claimed.",
}


@pytest.fixture
def fake_artifact_dir(tmp_path: Path) -> Path:
    """Write a fake ``metrics.json`` into a temp artifact dir (no real artifacts)."""
    (tmp_path / "metrics.json").write_text(json.dumps(_FAKE_METRICS), encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# _safe_float                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        (1.5, 1.5),
        (0, 0.0),
        ("2.5", 2.5),
        (None, None),
        ("nan", None),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        ("not-a-number", None),
        ([], None),
    ],
)
def test_safe_float_coerces_or_nones(value: Any, expected: float | None) -> None:
    """``_safe_float`` returns a finite float or ``None`` (never NaN/inf)."""
    assert _safe_float(value) == expected


# --------------------------------------------------------------------------- #
# load_committed_metrics                                                       #
# --------------------------------------------------------------------------- #
def test_load_committed_metrics_reads_verbatim(fake_artifact_dir: Path) -> None:
    """The loader returns the committed bundle unchanged."""
    metrics = load_committed_metrics(fake_artifact_dir)
    assert metrics["eval_macro_f1"] == 0.6525
    assert metrics["beats_lexicon"] is None
    assert metrics["confusion"] == [[24, 33, 4], [4, 266, 8], [6, 53, 55]]


def test_load_committed_metrics_missing_file_raises(tmp_path: Path) -> None:
    """A missing metrics.json raises ArtifactError (no silent default)."""
    with pytest.raises(ArtifactError, match="metrics file not found"):
        load_committed_metrics(tmp_path)


def test_load_committed_metrics_bad_json_raises(tmp_path: Path) -> None:
    """A malformed metrics.json raises ArtifactError."""
    (tmp_path / "metrics.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="failed to read committed metrics"):
        load_committed_metrics(tmp_path)


def test_default_metrics_path_points_at_artifacts() -> None:
    """The default path resolves under the package's artifacts/ directory."""
    path = default_metrics_path()
    assert path.name == "metrics.json"
    assert path.parent.name == "artifacts"


def test_committed_metrics_artifact_exists_and_is_consistent() -> None:
    """The SHIPPED metrics.json loads and is internally consistent (honest bundle)."""
    metrics = load_committed_metrics()
    # The live served model is the torch-free lexicon in this build.
    assert metrics["served_model"] == "lexicon"
    # eval_macro_f1 equals the lexicon's macro-F1 (the lexicon IS the served model).
    assert metrics["eval_macro_f1"] == metrics["lexicon_macro_f1"]
    # The lexicon clearly beats the class-prior floor.
    assert metrics["lexicon_macro_f1"] > metrics["class_prior_macro_f1"]
    # Lexicon-only build: nothing to compare, so beats_lexicon is null.
    assert metrics["beats_lexicon"] is None
    # The transformer figure is published, NOT measured in this build.
    assert metrics["transformer_published_macro_f1"]["measured_in_this_build"] is False


# --------------------------------------------------------------------------- #
# build_evaluation_figures                                                     #
# --------------------------------------------------------------------------- #
def test_build_evaluation_figures_shapes() -> None:
    """Both figures are well-formed ``{data, layout}`` Plotly dicts."""
    confusion_fig, f1_fig = build_evaluation_figures(_FAKE_METRICS)
    for fig in (confusion_fig, f1_fig):
        assert set(fig) == {"data", "layout"}
        assert isinstance(fig["data"], list)
        json.dumps(fig)  # JSON-serializable (no numpy/Plotly leak)
    assert confusion_fig["data"][0]["type"] == "heatmap"
    assert f1_fig["data"][0]["type"] == "bar"


def test_build_evaluation_figures_missing_block_raises() -> None:
    """Missing confusion / per-class-F1 in the metrics raises ArtifactError."""
    incomplete = {k: v for k, v in _FAKE_METRICS.items() if k != "confusion"}
    with pytest.raises(ArtifactError, match="missing 'confusion'"):
        build_evaluation_figures(incomplete)


def test_build_evaluation_figures_malformed_confusion_raises() -> None:
    """A wrong-shape confusion matrix is normalized into an ArtifactError."""
    bad = {**_FAKE_METRICS, "confusion": [[1, 2], [3, 4]]}  # 2x2, not 3x3
    with pytest.raises(ArtifactError, match="failed to build evaluation figures"):
        build_evaluation_figures(bad)


# --------------------------------------------------------------------------- #
# _ci_tuple                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ([0.5, 0.7], (0.5, 0.7)),
        ((0.1, 0.9), (0.1, 0.9)),
        (None, None),
        ([0.5], None),  # wrong arity
        ([0.5, 0.6, 0.7], None),  # wrong arity
        (["bad", 0.7], None),  # non-numeric element
        ([0.5, float("nan")], None),  # non-finite element
    ],
)
def test_ci_tuple_coerces_or_nones(raw: Any, expected: tuple[float, float] | None) -> None:
    """``_ci_tuple`` returns a finite (low, high) pair or ``None``."""
    assert _ci_tuple(raw) == expected


def test_run_sentiment_handles_missing_ci(tmp_path: Path) -> None:
    """A committed bundle with no CI yields a ``None`` summary CI (no crash)."""
    metrics = {k: v for k, v in _FAKE_METRICS.items() if k != "eval_macro_f1_ci"}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    result = run_sentiment(["Profit rose."], model_pref="lexicon", artifact_dir=tmp_path)
    assert result.summary.eval_macro_f1_ci is None


# --------------------------------------------------------------------------- #
# run_sentiment (lexicon path, torch-free)                                     #
# --------------------------------------------------------------------------- #
def test_run_sentiment_lexicon_assembles_full_response(fake_artifact_dir: Path) -> None:
    """``run_sentiment`` returns summary + predictions + figures (no torch, no net)."""
    texts = [
        "Quarterly profit rose sharply and beat estimates.",
        "Losses widened as demand fell and the stock dropped.",
        "The annual general meeting will be held in May.",
    ]
    result = run_sentiment(texts, model_pref="lexicon", seed=7, artifact_dir=fake_artifact_dir)
    assert isinstance(result, SentimentResult)
    summary = result.summary
    assert isinstance(summary, SentimentSummary)

    # Served model is the torch-free lexicon (no ONNX artifact present).
    assert summary.served_model == "lexicon"
    # Eval scalars come from the committed metrics VERBATIM (not recomputed).
    assert summary.eval_macro_f1 == 0.6525
    assert summary.lexicon_macro_f1 == 0.6525
    assert summary.class_prior_macro_f1 == 0.2535
    assert summary.eval_macro_f1_ci == (0.5944, 0.7098)
    assert summary.beats_lexicon is None
    assert summary.n_texts == 3
    assert summary.transformer_measured is False

    # One prediction per input, in order, sign-correct on the clear cases.
    assert len(result.predictions) == 3
    assert result.predictions[0]["label"] == "positive"
    assert result.predictions[1]["label"] == "negative"
    assert result.predictions[2]["label"] == "neutral"
    for record in result.predictions:
        assert set(record["scores"]) == {"negative", "neutral", "positive"}

    # Figures are present and the whole response is JSON-serializable.
    assert set(result.confusion_figure) == {"data", "layout"}
    assert set(result.per_class_f1_figure) == {"data", "layout"}
    json.dumps(result.to_dict())


def test_run_sentiment_distilbert_falls_back_to_lexicon(fake_artifact_dir: Path) -> None:
    """Requesting distilbert with no ONNX artifact transparently serves the lexicon."""
    result = run_sentiment(
        ["Profit rose and revenue grew."],
        model_pref="distilbert",
        artifact_dir=fake_artifact_dir,
    )
    # No model.int8.onnx in the fake dir -> lexicon fallback.
    assert result.summary.served_model == "lexicon"
    assert result.summary.n_texts == 1


def test_run_sentiment_rejects_empty_batch(fake_artifact_dir: Path) -> None:
    """An empty text batch fails validation before any work."""
    with pytest.raises(ValidationError):
        run_sentiment([], model_pref="lexicon", artifact_dir=fake_artifact_dir)


def test_run_sentiment_rejects_oversized_batch(fake_artifact_dir: Path) -> None:
    """A batch over the 64-text cap is rejected by the boundary validator."""
    with pytest.raises(ValidationError):
        run_sentiment(["x"] * 65, model_pref="lexicon", artifact_dir=fake_artifact_dir)


def test_summary_to_dict_round_trips() -> None:
    """SentimentSummary.to_dict is JSON-native and preserves the CI pair."""
    summary = SentimentSummary(
        served_model="lexicon",
        eval_macro_f1=0.65,
        eval_macro_f1_ci=(0.59, 0.71),
        eval_accuracy=0.76,
        lexicon_macro_f1=0.65,
        class_prior_macro_f1=0.25,
        beats_lexicon=None,
        n_texts=2,
        data_source="src",
        transformer_measured=False,
        notes="note",
    )
    d = summary.to_dict()
    assert d["eval_macro_f1_ci"] == [0.59, 0.71]
    assert d["beats_lexicon"] is None
    json.dumps(d)


def test_summary_to_dict_null_ci() -> None:
    """A None CI serializes to null, not a tuple."""
    summary = SentimentSummary(
        served_model="lexicon",
        eval_macro_f1=None,
        eval_macro_f1_ci=None,
        eval_accuracy=None,
        lexicon_macro_f1=None,
        class_prior_macro_f1=None,
        beats_lexicon=None,
        n_texts=0,
        data_source="src",
        transformer_measured=False,
        notes="note",
    )
    assert summary.to_dict()["eval_macro_f1_ci"] is None
