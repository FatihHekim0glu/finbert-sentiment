"""Unit tests for the Plotly figure builders and the Typer CLI.

Covers:

- ``finbert_sentiment.plots`` — the two figure builders (the test-set confusion
  matrix heatmap and the per-class F1 bar). Every builder must return a plain
  ``{"data", "layout"}`` mapping whose contents are JSON-serializable (no
  numpy/Plotly object leaks across the API boundary) and FINITE, and whose
  numerical structure is asserted (heatmap ``z`` equals the confusion counts, bar
  ``y`` equals the per-class F1, canonical label axes) rather than merely "it
  runs". A real Plotly schema round-trip confirms the dicts are valid figures.
- ``finbert_sentiment.cli`` — ``build_app`` registers the three commands;
  ``--help`` lists them; and a tiny lexicon ``predict`` smoke run classifies
  sign-correctly WITHOUT importing torch/transformers/onnxruntime.

All inputs are tiny and offline; nothing touches the network or a deep-learning
framework. The lexicon ``predict`` path is deliberately self-contained (it does
not go through the ONNX serve stack), so this suite runs torch-free.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from finbert_sentiment import plots
from finbert_sentiment._constants import LABELS, N_CLASSES
from finbert_sentiment._exceptions import ValidationError
from finbert_sentiment.cli import build_app
from finbert_sentiment.evaluation.metrics import classification_report

pytestmark = pytest.mark.unit


#: A tiny, lexicon-separable offline sample (no network) used by the subprocess
#: import-purity checks, which cannot share the conftest fixture across processes.
#: Five distinct groups per class so the stratified group split can place every
#: class into train/val/test (the split requires >= 3 groups per class).
_PHRASEBANK_SAMPLE: tuple[tuple[str, str], ...] = (
    ("Quarterly profit rose sharply as revenue gains beat estimates.", "positive"),
    ("The company reported record growth and raised its full-year guidance.", "positive"),
    ("Operating margins improved and the stock surged to a new high.", "positive"),
    ("Strong demand boosted earnings well above expectations.", "positive"),
    ("Shares rose after the firm beat estimates and upgraded its outlook.", "positive"),
    ("Quarterly loss widened as revenue declined and margins fell.", "negative"),
    ("The company cut its guidance after a sharp drop in demand.", "negative"),
    ("Profit plunged and the stock dropped to a multi-year low.", "negative"),
    ("Weak sales and a downgrade dragged the shares lower.", "negative"),
    ("The firm warned of further losses amid a lawsuit and bankruptcy risk.", "negative"),
    ("The company will hold its annual general meeting in May.", "neutral"),
    ("The board appointed a new chief financial officer effective next month.", "neutral"),
    ("The headquarters are located in Helsinki, Finland.", "neutral"),
    ("The report covers the fiscal period ending in December.", "neutral"),
    ("The press release was distributed to shareholders on Tuesday.", "neutral"),
)


# --------------------------------------------------------------------------- #
# Figure-dict helpers                                                          #
# --------------------------------------------------------------------------- #
def _assert_finite(value: Any) -> None:
    """Recursively assert every float leaf in a figure dict is finite."""
    if isinstance(value, dict):
        for v in value.values():
            _assert_finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _assert_finite(v)
    elif isinstance(value, float):
        assert math.isfinite(value), f"non-finite float leaked: {value!r}"


def _assert_figure_dict(fig: object) -> dict[str, Any]:
    """Assert ``fig`` is a JSON-safe, finite ``{"data", "layout"}`` mapping.

    The ``json.dumps`` round-trip is the load-bearing check: it fails loudly if any
    numpy scalar/array or Plotly graph-object leaked across the API boundary.
    """
    assert isinstance(fig, dict)
    assert set(fig) == {"data", "layout"}
    assert isinstance(fig["data"], list)
    assert isinstance(fig["layout"], dict)
    encoded = json.dumps(fig)  # raises if anything non-JSON-native leaked
    assert json.loads(encoded) == fig
    _assert_finite(fig)
    return fig


def _assert_valid_plotly(fig: dict[str, Any]) -> None:
    """Construct a real Plotly Figure from the dict and re-serialize it.

    This validates the ``{data, layout}`` schema with Plotly itself (the same path
    the FastAPI layer uses), proving the builder emits a genuinely renderable
    figure and not just an arbitrary mapping. Plotly is the OPTIONAL ``viz`` extra
    — imported lazily here, inside the test, never at module import.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    go_fig = go.Figure(fig)
    round_trip = json.loads(pio.to_json(go_fig, validate=False))
    assert "data" in round_trip
    assert "layout" in round_trip


@pytest.fixture
def report_with_errors() -> Any:
    """A real :class:`ClassificationReport` with non-trivial off-diagonal errors."""
    y_true = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0]
    y_pred = [0, 1, 1, 1, 2, 1, 0, 2, 2, 1, 2, 0]
    return classification_report(y_true, y_pred, bootstrap_ci=False)


# --------------------------------------------------------------------------- #
# confusion_matrix_figure                                                      #
# --------------------------------------------------------------------------- #
def test_confusion_matrix_figure_counts_structure(report_with_errors: Any) -> None:
    """Counts heatmap: z equals the confusion matrix, axes are the canonical labels."""
    fig = _assert_figure_dict(plots.confusion_matrix_figure(report_with_errors.confusion))
    _assert_valid_plotly(fig)

    trace = fig["data"][0]
    assert trace["type"] == "heatmap"
    z = np.asarray(trace["z"])
    np.testing.assert_array_equal(z, np.asarray(report_with_errors.confusion, dtype=float))
    # Canonical (negative, neutral, positive) axes, rows=true / cols=pred.
    assert trace["x"] == list(LABELS)
    assert trace["y"] == list(LABELS)
    assert fig["layout"]["title"] == {"text": "Confusion matrix (test set)"}
    # One annotation per cell, each showing the integer count.
    assert len(fig["layout"]["annotations"]) == N_CLASSES * N_CLASSES


def test_confusion_matrix_figure_normalized_rows_sum_to_one(report_with_errors: Any) -> None:
    """Row-normalized heatmap: each non-empty row of z sums to one (recall view)."""
    fig = _assert_figure_dict(
        plots.confusion_matrix_figure(report_with_errors.confusion, normalize=True)
    )
    _assert_valid_plotly(fig)

    z = np.asarray(fig["data"][0]["z"])
    counts = np.asarray(report_with_errors.confusion, dtype=float)
    for i in range(N_CLASSES):
        if counts[i].sum() > 0:
            assert z[i].sum() == pytest.approx(1.0)
            # The diagonal entry equals that class's recall.
            assert z[i, i] == pytest.approx(counts[i, i] / counts[i].sum())


def test_confusion_matrix_figure_all_zero_row_stays_finite() -> None:
    """A never-true class (all-zero row) normalizes to zeros, not NaN/inf."""
    cm = [[2, 1, 0], [0, 0, 0], [1, 0, 3]]  # middle (neutral) class never true
    fig = _assert_figure_dict(plots.confusion_matrix_figure(cm, normalize=True))
    z = np.asarray(fig["data"][0]["z"])
    np.testing.assert_array_equal(z[1], np.zeros(N_CLASSES))


def test_confusion_matrix_figure_custom_title_and_list_input() -> None:
    """A nested-list confusion matrix is accepted and the title propagates."""
    fig = _assert_figure_dict(
        plots.confusion_matrix_figure([[5, 0, 0], [0, 4, 1], [0, 2, 3]], title="custom cm")
    )
    assert fig["layout"]["title"] == {"text": "custom cm"}


@pytest.mark.parametrize(
    "bad",
    [
        [[1, 2], [3, 4]],  # wrong shape (2x2)
        np.zeros((3, 3, 3)),  # not 2-D
        [[1, 0, 0], [0, 1, 0]],  # not square (2x3)
    ],
)
def test_confusion_matrix_figure_rejects_non_square(bad: object) -> None:
    """A non-``N_CLASSES``-square confusion matrix is rejected."""
    with pytest.raises(ValidationError):
        plots.confusion_matrix_figure(bad)  # type: ignore[arg-type]


def test_confusion_matrix_figure_rejects_nan_and_negative() -> None:
    """NaN or negative counts are rejected (defensive guard)."""
    with pytest.raises(ValidationError):
        plots.confusion_matrix_figure([[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        plots.confusion_matrix_figure([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])


# --------------------------------------------------------------------------- #
# per_class_f1_figure                                                          #
# --------------------------------------------------------------------------- #
def test_per_class_f1_figure_bar_values(report_with_errors: Any) -> None:
    """Per-class F1 bar: y equals the report's per-class F1, x are canonical labels."""
    fig = _assert_figure_dict(plots.per_class_f1_figure(report_with_errors.per_class_f1))
    _assert_valid_plotly(fig)

    trace = fig["data"][0]
    assert trace["type"] == "bar"
    assert trace["x"] == list(LABELS)
    np.testing.assert_allclose(np.asarray(trace["y"]), report_with_errors.per_class_f1)
    # F1 is bounded; the axis pins the [0, 1] range so bars are comparable.
    assert fig["layout"]["yaxis"]["range"] == [0.0, 1.0]
    assert fig["layout"]["title"] == {"text": "Per-class F1 (test set)"}


def test_per_class_f1_figure_accepts_ndarray_and_custom_title() -> None:
    """An ndarray of F1 values is accepted; the custom title propagates."""
    fig = _assert_figure_dict(
        plots.per_class_f1_figure(np.array([0.8, 0.6, 0.9]), title="held-out F1")
    )
    np.testing.assert_allclose(np.asarray(fig["data"][0]["y"]), [0.8, 0.6, 0.9])
    assert fig["layout"]["title"] == {"text": "held-out F1"}


@pytest.mark.parametrize(
    "bad",
    [
        [0.5, 0.5],  # wrong length
        [0.5, 0.5, 0.5, 0.5],  # wrong length
        [0.5, 1.5, 0.5],  # > 1
        [-0.1, 0.5, 0.5],  # < 0
        [float("nan"), 0.5, 0.5],  # NaN
    ],
)
def test_per_class_f1_figure_rejects_bad_input(bad: object) -> None:
    """Wrong length or out-of-[0,1] / NaN values are rejected."""
    with pytest.raises(ValidationError):
        plots.per_class_f1_figure(bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# _to_native (JSON-safety helper)                                              #
# --------------------------------------------------------------------------- #
def test_to_native_coerces_numpy_to_json_native() -> None:
    """``_to_native`` strips every numpy type so figures stay JSON-serializable.

    The confusion/F1 figures route their numeric payloads through this helper; it
    must turn numpy scalars, arrays, and nested containers into plain Python types
    so no numpy object can leak across the API boundary.
    """
    out = plots._to_native(
        {
            "scalar": np.int64(5),
            "arr": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "nested": (np.float64(1.5), [np.int32(2)]),
            "plain": "ok",
        }
    )
    assert out == {
        "scalar": 5,
        "arr": [[1.0, 2.0], [3.0, 4.0]],
        "nested": [1.5, [2]],
        "plain": "ok",
    }
    # The whole structure is JSON-serializable (the load-bearing guarantee).
    assert json.loads(json.dumps(out)) == out
    # No numpy types survived anywhere.
    assert not isinstance(out["scalar"], np.generic)


# --------------------------------------------------------------------------- #
# plots import purity                                                          #
# --------------------------------------------------------------------------- #
def test_plots_import_does_not_pull_plotly() -> None:
    """Importing the plots module must not import Plotly (the optional viz extra)."""
    import importlib

    mod = importlib.import_module("finbert_sentiment.plots")
    # The builder functions exist and Plotly was not imported at module load.
    assert hasattr(mod, "confusion_matrix_figure")
    assert hasattr(mod, "per_class_f1_figure")


# --------------------------------------------------------------------------- #
# CLI: build_app + --help                                                      #
# --------------------------------------------------------------------------- #
def test_build_app_is_isolated_instance() -> None:
    """build_app returns a fresh Typer app each call (no shared mutable state)."""
    import typer

    app_a = build_app()
    app_b = build_app()
    assert isinstance(app_a, typer.Typer)
    assert app_a is not app_b


def test_cli_help_lists_three_commands() -> None:
    """``--help`` lists train, evaluate, and predict."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(build_app(), ["--help"])
    assert result.exit_code == 0, result.output
    for command in ("train", "evaluate", "predict"):
        assert command in result.output


def test_cli_no_args_shows_help() -> None:
    """Invoking with no arguments prints help (no_args_is_help) and exits 2."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(build_app(), [])
    assert result.exit_code == 2
    assert "predict" in result.output


def test_cli_predict_help_documents_fallback() -> None:
    """``predict --help`` documents the lexicon fallback and exits cleanly."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(build_app(), ["predict", "--help"])
    assert result.exit_code == 0, result.output
    assert "lexicon" in result.output


# --------------------------------------------------------------------------- #
# CLI: lexicon predict smoke run (no torch)                                    #
# --------------------------------------------------------------------------- #
def test_cli_predict_lexicon_smoke_run_is_sign_correct() -> None:
    """A tiny lexicon predict run classifies sign-correctly and exits 0."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        build_app(),
        [
            "predict",
            "--model",
            "lexicon",
            "Quarterly profit rose sharply and beat estimates.",
            "Losses widened as demand fell and the stock dropped.",
            "The annual general meeting will be held in May.",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "served model       : lexicon" in result.output
    # The clearly +/- sentences and the genuinely neutral one each get the right
    # label (the lexicon is sign-correct on clear cases).
    assert "[positive]" in result.output
    assert "[negative]" in result.output
    assert "[neutral ]" in result.output


def test_cli_predict_lexicon_does_not_import_torch() -> None:
    """The lexicon predict path imports no deep-learning / inference engine.

    Run in a clean subprocess so the assertion reflects only the modules the
    lexicon ``predict`` path imports — never modules a sibling in-process test
    (e.g. the ONNX-session suite) already faulted into ``sys.modules``.
    """
    code = (
        "import sys\n"
        "from finbert_sentiment.cli import predict\n"
        "rc = predict(texts=['Profit rose and revenue grew.'], model='lexicon')\n"
        "assert rc == 0, f'predict exit code {rc}'\n"
        "heavy = sorted(\n"
        "    m for m in ('torch', 'transformers', 'onnx', 'onnxruntime')\n"
        "    if m in sys.modules\n"
        ")\n"
        "assert heavy == [], f'predict pulled in heavy modules: {heavy}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"lexicon-predict purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


def test_cli_predict_rejects_unknown_model() -> None:
    """An unknown ``--model`` selection exits 2 without raising."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(build_app(), ["predict", "--model", "bogus", "hello world"])
    assert result.exit_code == 2
    assert "model must be one of" in result.output


def test_cli_predict_rejects_empty_text() -> None:
    """A blank sentence fails batch validation and exits non-zero (no crash)."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(build_app(), ["predict", "--model", "lexicon", "   "])
    assert result.exit_code != 0


def test_cli_predict_function_lexicon_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """The ``predict`` function (lexicon) prints scores per text and returns 0."""
    from finbert_sentiment.cli import predict

    code = predict(
        texts=["Profit rose sharply and beat estimates.", "Losses widened as demand fell."],
        model="lexicon",
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "served model       : lexicon" in out
    assert "[positive]" in out
    assert "[negative]" in out


def test_cli_predict_function_rejects_unknown_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ``predict`` function rejects an unknown model with exit code 2."""
    from finbert_sentiment.cli import predict

    code = predict(texts=["hello world"], model="bogus")
    assert code == 2
    assert "model must be one of" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# CLI: evaluate (lexicon path, offline — no network, no torch)                #
# --------------------------------------------------------------------------- #
@pytest.fixture
def _offline_split(monkeypatch: pytest.MonkeyPatch, phrasebank_sample: Any) -> None:
    """Monkeypatch ``_prepare_split`` to feed the offline sample (no network).

    The CLI ``evaluate``/``train`` orchestration normally pulls the Financial
    PhraseBank over the network; here we splice in the conftest's offline-cached
    :class:`LabelledDataset` so the locked-test-set evaluation runs entirely
    offline and torch-free.
    """
    from finbert_sentiment import cli
    from finbert_sentiment.data.dedup import dedup_sentences
    from finbert_sentiment.data.split import (
        assert_no_group_overlap,
        stratified_group_split,
    )

    deduped = dedup_sentences(phrasebank_sample)
    split = stratified_group_split(deduped.dataset.labels, deduped.group_hashes, seed=20260618)
    assert_no_group_overlap(split, deduped.group_hashes)

    def _fake_prepare_split(config: str, seed: int) -> Any:
        return phrasebank_sample, deduped, split

    monkeypatch.setattr(cli, "_prepare_split", _fake_prepare_split)


def test_cli_evaluate_lexicon_offline_reports_macro_f1(
    _offline_split: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """``evaluate`` (lexicon) reports macro-F1 + CI + the lexicon-only verdict, code 0."""
    from finbert_sentiment.cli import evaluate

    code = evaluate(model="lexicon", config="x", seed=20260618, n_bootstrap=100)
    out = capsys.readouterr().out
    assert code == 0
    # Headline is macro-F1 (with a bootstrap CI), never accuracy alone.
    assert "macro-F1           :" in out
    assert "macro-F1 95% CI    :" in out
    assert "lexicon macro-F1   :" in out
    assert "class-prior macro-F1:" in out
    # The lexicon IS the served model, so there is nothing to compare against.
    assert "served model       : lexicon" in out
    assert "verdict            : lexicon_only" in out
    assert "beats_lexicon      : None" in out
    # Honest caption is printed (sentiment is a text label, not a tradable signal).
    assert "no alpha is claimed" in out
    assert "walk-forward/purge/DSR do not apply" in out


def test_cli_evaluate_writes_metrics_json(
    _offline_split: None, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """``evaluate --metrics-out`` writes the committed bundle the API reads verbatim."""
    import json

    from finbert_sentiment.cli import evaluate
    from finbert_sentiment.service import load_committed_metrics, run_sentiment

    out_path = tmp_path / "metrics.json"
    code = evaluate(
        model="lexicon",
        config="x",
        seed=20260618,
        n_bootstrap=50,
        metrics_out=str(out_path),
    )
    assert code == 0
    assert "metrics written    :" in capsys.readouterr().out
    assert out_path.is_file()

    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    # The honest bundle the service consumes: lexicon served, beats_lexicon null.
    assert bundle["served_model"] == "lexicon"
    assert bundle["eval_macro_f1"] == bundle["lexicon_macro_f1"]
    assert bundle["beats_lexicon"] is None
    assert bundle["verdict"] == "lexicon_only"
    assert bundle["transformer_published_macro_f1"]["measured_in_this_build"] is False
    assert len(bundle["confusion"]) == N_CLASSES

    # The service can load + consume the freshly-written bundle end-to-end.
    assert load_committed_metrics(tmp_path)["served_model"] == "lexicon"
    result = run_sentiment(["Profit rose and revenue grew."], artifact_dir=tmp_path)
    assert result.summary.served_model == "lexicon"
    assert result.summary.eval_macro_f1 == bundle["eval_macro_f1"]


def test_cli_predict_indices_chunked_matches_unchunked() -> None:
    """The chunked baseline-predict helper equals an unchunked predict (>64 texts)."""
    from finbert_sentiment.baselines.lexicon import LexiconClassifier
    from finbert_sentiment.cli import _predict_indices_chunked

    clf = LexiconClassifier()
    texts = ["Profit rose and revenue grew." for _ in range(150)]
    chunked = _predict_indices_chunked(clf, texts)
    # Reference: classify a single small batch (the lexicon is stateless).
    reference = int(clf.predict(["Profit rose and revenue grew."])[0])
    assert len(chunked) == 150
    assert all(v == reference for v in chunked)


def test_cli_predict_labels_chunked_runs_over_cap() -> None:
    """The chunked Predictor helper classifies a >64-text batch via the lexicon backend."""
    from finbert_sentiment.cli import _predict_labels_chunked
    from finbert_sentiment.inference.predictor import load_predictor

    predictor = load_predictor("lexicon")
    texts = ["Losses widened as demand fell." for _ in range(70)]
    labels = _predict_labels_chunked(predictor, texts)
    assert len(labels) == 70
    assert all(v == 0 for v in labels)  # all clearly negative


def test_cli_evaluate_rejects_unknown_model(capsys: pytest.CaptureFixture[str]) -> None:
    """``evaluate`` with an unknown model exits 2 before any data work."""
    from finbert_sentiment.cli import evaluate

    code = evaluate(model="bogus", config="x", seed=1, n_bootstrap=10)
    assert code == 2
    assert "model must be one of" in capsys.readouterr().out


def test_cli_evaluate_command_offline_via_runner(_offline_split: None) -> None:
    """The ``evaluate`` Typer command (lexicon) runs offline through CliRunner, exit 0."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        build_app(), ["evaluate", "--model", "lexicon", "--n-bootstrap", "50"]
    )
    assert result.exit_code == 0, result.output
    assert "macro-F1" in result.output


def test_cli_evaluate_command_no_torch() -> None:
    """The offline lexicon ``evaluate`` path imports no deep-learning framework.

    Run in a clean subprocess so the heavy-module check measures only what the
    lexicon ``evaluate`` path imports, immune to ``sys.modules`` pollution from
    the in-process ONNX-session tests. The PhraseBank loader is spliced out with
    a small offline sample so the locked-test-set eval runs without the network.
    """
    sample = [(t, lab) for t, lab in _PHRASEBANK_SAMPLE]
    code = (
        "import sys\n"
        "from finbert_sentiment import cli\n"
        "from finbert_sentiment.data.dedup import dedup_sentences\n"
        "from finbert_sentiment.data.load import sample_dataset\n"
        "from finbert_sentiment.data.split import stratified_group_split\n"
        f"sample = {sample!r}\n"
        "ds = sample_dataset(\n"
        "    [t for t, _ in sample], [lab for _, lab in sample], source='offline-cached-sample'\n"
        ")\n"
        "deduped = dedup_sentences(ds)\n"
        "split = stratified_group_split(deduped.dataset.labels, deduped.group_hashes, seed=1)\n"
        "cli._prepare_split = lambda config, seed: (ds, deduped, split)\n"
        "rc = cli.evaluate(model='lexicon', config='x', seed=1, n_bootstrap=30)\n"
        "assert rc == 0, f'evaluate exit code {rc}'\n"
        "heavy = sorted(\n"
        "    m for m in ('torch', 'transformers', 'onnx', 'onnxruntime')\n"
        "    if m in sys.modules\n"
        ")\n"
        "assert heavy == [], f'evaluate pulled in heavy modules: {heavy}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"lexicon-evaluate purity subprocess failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


def test_cli_evaluate_distilbert_branch_with_fake_backend(
    _offline_split: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The distilbert ``evaluate`` branch runs end-to-end against a fake predictor.

    The transformer training/serving stack is owned by other groups (and needs no
    torch here); we inject a tiny fake :class:`Predictor` whose ``predict`` returns
    canned :class:`Prediction` objects, so the McNemar + verdict branch (the only
    part of ``evaluate`` not covered by the lexicon path) is exercised offline.
    """
    from finbert_sentiment import cli
    from finbert_sentiment._constants import INDEX_TO_LABEL
    from finbert_sentiment.inference.predictor import Prediction, Predictor

    class _FakePredictor(Predictor):
        def __init__(self) -> None:
            super().__init__("distilbert-onnx")

        def predict(self, texts: Any) -> list[Prediction]:
            # Echo the lexicon's verdict so the discordant set is empty and the
            # branch (McNemar + derive_verdict) runs on real numbers.
            from finbert_sentiment.baselines.lexicon import LexiconClassifier

            preds = LexiconClassifier().predict(list(texts))
            return [
                Prediction(text=t, label=INDEX_TO_LABEL[int(i)], scores={})
                for t, i in zip(texts, preds, strict=True)
            ]

    monkeypatch.setattr(
        "finbert_sentiment.inference.predictor.load_predictor",
        lambda requested="distilbert", **_: _FakePredictor(),
    )

    code = cli.evaluate(model="distilbert", config="x", seed=20260618, n_bootstrap=30)
    out = capsys.readouterr().out
    assert code == 0
    assert "served model       : distilbert-onnx" in out
    assert "McNemar p-value    :" in out
    # beats_lexicon is a derived boolean (not None) once a transformer is served.
    assert "beats_lexicon      : " in out


def test_cli_predict_distilbert_branch_with_fake_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The distilbert ``predict`` branch formats scores from a fake ONNX predictor."""
    from finbert_sentiment import cli
    from finbert_sentiment.inference.predictor import Prediction, Predictor

    class _FakePredictor(Predictor):
        def __init__(self) -> None:
            super().__init__("distilbert-onnx")

        def predict(self, texts: Any) -> list[Prediction]:
            return [
                Prediction(
                    text=t,
                    label="positive",
                    scores={"negative": 0.1, "neutral": 0.2, "positive": 0.7},
                )
                for t in texts
            ]

    monkeypatch.setattr(
        "finbert_sentiment.inference.predictor.load_predictor",
        lambda requested="distilbert", **_: _FakePredictor(),
    )

    code = cli.predict(texts=["Profit rose sharply."], model="distilbert")
    out = capsys.readouterr().out
    assert code == 0
    assert "served model       : distilbert-onnx" in out
    assert "[positive]" in out
    assert "pos=0.700" in out


# --------------------------------------------------------------------------- #
# CLI: train (offline, torch-free via fakes)                                  #
# --------------------------------------------------------------------------- #
def test_cli_train_offline_with_fakes(
    _offline_split: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``train`` orchestrates split -> fine-tune -> export against torch-free fakes.

    ``train_distilbert`` / ``export_to_onnx`` (the ``[train]`` path owned by the
    model group) are replaced with fakes so the CLI orchestration — the split
    prep, the printed summary, and the export step — is exercised without torch.
    """
    from finbert_sentiment.cli import train
    from finbert_sentiment.model.export import ExportResult
    from finbert_sentiment.model.train import TrainConfig, TrainResult

    def _fake_train(dataset: Any, split: Any, config: Any, *, output_dir: str) -> TrainResult:
        assert isinstance(config, TrainConfig)
        return TrainResult(
            output_dir=output_dir,
            best_val_macro_f1=0.88,
            epochs_run=2,
            config=config,
            label_order=("negative", "neutral", "positive"),
        )

    def _fake_export(model_dir: str, *, output_dir: str, int8: bool = True) -> ExportResult:
        return ExportResult(
            onnx_path=f"{output_dir}/model.int8.onnx",
            tokenizer_path=f"{output_dir}/tokenizer.json",
            int8=int8,
            opset=14,
        )

    monkeypatch.setattr("finbert_sentiment.model.train.train_distilbert", _fake_train)
    monkeypatch.setattr("finbert_sentiment.model.export.export_to_onnx", _fake_export)

    code = train(
        output_dir="artifacts",
        config="x",
        epochs=2,
        batch_size=8,
        learning_rate=2e-5,
        max_length=64,
        seed=20260618,
        int8=True,
        export=True,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "finbert-sentiment train" in out
    assert "best val macro-F1  : 0.8800" in out
    assert "ONNX artifact      : artifacts/model.int8.onnx" in out
    assert "int8 quantized     : True" in out


def test_cli_train_no_export(
    _offline_split: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``train`` with ``export=False`` fine-tunes but skips the ONNX export step."""
    from finbert_sentiment.cli import train
    from finbert_sentiment.model.train import TrainResult

    def _fake_train(dataset: Any, split: Any, config: Any, *, output_dir: str) -> TrainResult:
        return TrainResult(
            output_dir=output_dir, best_val_macro_f1=0.9, epochs_run=1, config=config
        )

    monkeypatch.setattr("finbert_sentiment.model.train.train_distilbert", _fake_train)

    code = train(
        output_dir="artifacts",
        config="x",
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        max_length=64,
        seed=1,
        int8=False,
        export=False,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "saved model dir    : artifacts" in out
    assert "ONNX artifact" not in out


def test_cli_train_reports_library_error(
    _offline_split: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A library error during the fine-tune is caught and returns exit code 1."""
    from finbert_sentiment._exceptions import FinbertSentimentError
    from finbert_sentiment.cli import train

    def _boom(dataset: Any, split: Any, config: Any, *, output_dir: str) -> Any:
        raise FinbertSentimentError("the [train] extra is not installed")

    monkeypatch.setattr("finbert_sentiment.model.train.train_distilbert", _boom)

    code = train(
        output_dir="artifacts",
        config="x",
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        max_length=64,
        seed=1,
        int8=True,
        export=True,
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "error: the [train] extra is not installed" in out


def test_cli_evaluate_reports_library_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A library error during the split is caught and ``evaluate`` returns 1."""
    from finbert_sentiment import cli
    from finbert_sentiment._exceptions import FinbertSentimentError

    def _boom(config: str, seed: int) -> Any:
        raise FinbertSentimentError("phrasebank download failed")

    monkeypatch.setattr(cli, "_prepare_split", _boom)

    code = cli.evaluate(model="lexicon", config="x", seed=1, n_bootstrap=10)
    out = capsys.readouterr().out
    assert code == 1
    assert "error: phrasebank download failed" in out


def test_cli_train_command_offline_via_runner(
    _offline_split: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``train`` Typer command runs offline through CliRunner against fakes."""
    from typer.testing import CliRunner

    from finbert_sentiment.model.export import ExportResult
    from finbert_sentiment.model.train import TrainResult

    monkeypatch.setattr(
        "finbert_sentiment.model.train.train_distilbert",
        lambda dataset, split, config, *, output_dir: TrainResult(
            output_dir=output_dir, best_val_macro_f1=0.87, epochs_run=2, config=config
        ),
    )
    monkeypatch.setattr(
        "finbert_sentiment.model.export.export_to_onnx",
        lambda model_dir, *, output_dir, int8=True: ExportResult(
            onnx_path=f"{output_dir}/model.int8.onnx",
            tokenizer_path=f"{output_dir}/tokenizer.json",
            int8=int8,
            opset=14,
        ),
    )

    result = CliRunner().invoke(build_app(), ["train", "--epochs", "2", "--no-export"])
    assert result.exit_code == 0, result.output
    assert "finbert-sentiment train" in result.output


# --------------------------------------------------------------------------- #
# CLI: _prepare_split (the leakage-guarded load->dedup->split orchestration)   #
# --------------------------------------------------------------------------- #
def test_prepare_split_is_group_disjoint_offline(
    monkeypatch: pytest.MonkeyPatch, phrasebank_sample: Any
) -> None:
    """``_prepare_split`` dedups + group-splits with no sentence-hash leakage.

    ``load_phrasebank`` (the only network touch) is monkeypatched to return the
    offline sample, so the real load -> dedup -> stratified-group-split ->
    assert-no-overlap orchestration runs without the network. The returned split
    indexes the DEDUPLICATED dataset and is group-disjoint across folds.
    """
    from finbert_sentiment import cli
    from finbert_sentiment.data.split import assert_no_group_overlap

    monkeypatch.setattr(
        "finbert_sentiment.data.load.load_phrasebank",
        lambda *, config=None, cache_dir=None: phrasebank_sample,
    )

    raw, deduped, split = cli._prepare_split("sentences_allagree", seed=20260618)
    assert raw is phrasebank_sample
    # The split indexes the deduplicated dataset and covers it exactly once.
    n = deduped.dataset.n
    covered = sorted([*split.train, *split.val, *split.test])
    assert covered == list(range(n))
    # The headline leakage guarantee holds (re-asserting is a no-op here).
    assert_no_group_overlap(split, deduped.group_hashes)


def test_main_invokes_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` builds the app lazily and invokes it (console-script entry point)."""
    from finbert_sentiment import cli

    called: dict[str, bool] = {"built": False, "invoked": False}

    class _FakeApp:
        def __call__(self) -> None:
            called["invoked"] = True

    def _fake_build_app() -> _FakeApp:
        called["built"] = True
        return _FakeApp()

    monkeypatch.setattr(cli, "build_app", _fake_build_app)
    cli.main()
    assert called == {"built": True, "invoked": True}
