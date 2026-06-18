"""Unit tests for the offline ``[train]`` path (train + export), torch-free.

The whole point of this partition is that ``finbert_sentiment.model.train`` and
``finbert_sentiment.model.export`` are importable, signature-correct, and exercise
their pure (torch-free) logic WITHOUT torch / transformers / onnx installed:

* importing the modules pulls in no deep-learning framework (import purity);
* the config dataclasses validate their hyper-parameters;
* :func:`build_label_arrays` (the "dry-run config builder") slices train/val off a
  locked split and refuses to leak the test fold or train on a missing class;
* :func:`write_metrics_json` round-trips the offline metrics the router serves;
* calling the heavy entry points without ``[train]`` raises a typed error, never
  an ``ImportError``.

The ACTUAL DistilBERT fine-tune / ONNX export are covered by the ``train``-marked
tests below, which ``skip`` unless the heavy extra is present, so this suite stays
green torch-free.
"""

from __future__ import annotations

import sys

import pytest

from finbert_sentiment._constants import DEFAULT_SEED, LABELS, N_CLASSES
from finbert_sentiment._exceptions import FinbertSentimentError, InsufficientDataError
from finbert_sentiment.data.split import SplitIndices, stratified_group_split
from finbert_sentiment.model import (
    ExportResult,
    TrainConfig,
    TrainResult,
    build_label_arrays,
    export_available,
    export_to_onnx,
    train_available,
    train_distilbert,
    write_metrics_json,
)
from finbert_sentiment.model.export import (
    INPUT_NAMES,
    ONNX_INT8_NAME,
    OUTPUT_NAME,
    TOKENIZER_NAME,
    _probe_texts,
)
from finbert_sentiment.model.train import _compute_macro_f1
from tests.conftest import PHRASEBANK_SAMPLE  # reuse the offline corpus


# --------------------------------------------------------------------------- #
# Import purity: importing the train/export modules pulls in NO heavy framework #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_importing_model_does_not_import_torch_transformers_onnx() -> None:
    """``finbert_sentiment.model`` (already imported) brought in no torch/transformers/onnx."""
    for forbidden in ("torch", "transformers", "onnx"):
        assert forbidden not in sys.modules, (
            f"{forbidden} was imported at module load — must be lazy inside functions."
        )


@pytest.mark.unit
def test_train_export_availability_probes_are_torch_free() -> None:
    """The availability probes return a bool without importing the heavy deps."""
    assert isinstance(train_available(), bool)
    assert isinstance(export_available(), bool)
    # Probing must not have side-loaded torch/transformers/onnx.
    for forbidden in ("torch", "transformers", "onnx"):
        assert forbidden not in sys.modules


# --------------------------------------------------------------------------- #
# TrainConfig: a frozen, validated, JSON-serializable hyper-parameter bundle    #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_train_config_defaults_and_to_dict() -> None:
    cfg = TrainConfig()
    assert cfg.model_name == "distilbert-base-uncased"
    assert cfg.epochs == 3
    assert cfg.seed == DEFAULT_SEED
    d = cfg.to_dict()
    assert d["model_name"] == "distilbert-base-uncased"
    assert set(d) == {
        "model_name",
        "epochs",
        "batch_size",
        "learning_rate",
        "max_length",
        "early_stopping_patience",
        "seed",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": "  "},
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"learning_rate": -1.0},
        {"max_length": 0},
        {"early_stopping_patience": -1},
        {"seed": -1},
    ],
)
def test_train_config_rejects_bad_hyperparams(kwargs: dict[str, object]) -> None:
    with pytest.raises(FinbertSentimentError):
        TrainConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
def test_train_result_to_dict_flattens_config() -> None:
    res = TrainResult(
        output_dir="/tmp/out",
        best_val_macro_f1=0.87,
        epochs_run=2,
        config=TrainConfig(epochs=2),
        label_order=LABELS,
    )
    d = res.to_dict()
    assert d["best_val_macro_f1"] == 0.87
    assert d["epochs_run"] == 2
    assert isinstance(d["config"], dict)
    assert d["config"]["epochs"] == 2
    assert tuple(d["label_order"]) == LABELS


@pytest.mark.unit
def test_export_result_to_dict() -> None:
    res = ExportResult(
        onnx_path="/tmp/model.int8.onnx",
        tokenizer_path="/tmp/tokenizer.json",
        int8=True,
        opset=14,
        max_logit_abs_diff=1e-4,
    )
    d = res.to_dict()
    assert d == {
        "onnx_path": "/tmp/model.int8.onnx",
        "tokenizer_path": "/tmp/tokenizer.json",
        "int8": True,
        "opset": 14,
        "max_logit_abs_diff": 1e-4,
    }


@pytest.mark.unit
def test_export_artifact_names_match_serve_session() -> None:
    """Export must write the exact artifact names the serve session looks for."""
    from finbert_sentiment.inference.onnx_session import (
        ONNX_ARTIFACT_NAME,
        TOKENIZER_ARTIFACT_NAME,
    )

    assert ONNX_INT8_NAME == ONNX_ARTIFACT_NAME
    assert TOKENIZER_NAME == TOKENIZER_ARTIFACT_NAME
    assert INPUT_NAMES == ("input_ids", "attention_mask")
    assert OUTPUT_NAME == "logits"


@pytest.mark.unit
def test_probe_texts_cover_all_three_classes() -> None:
    """The export-time parity probe is a non-empty, lexicon-separable batch."""
    probe = _probe_texts()
    assert len(probe) == N_CLASSES
    assert all(isinstance(t, str) and t.strip() for t in probe)


# --------------------------------------------------------------------------- #
# build_label_arrays: the torch-free "dry-run config builder"                   #
# --------------------------------------------------------------------------- #
def _sample_split(dataset: object) -> SplitIndices:
    """Build a locked split over the offline sample using its dedup hashes."""
    from finbert_sentiment.data.dedup import normalize_sentence

    labels = list(dataset.labels)  # type: ignore[attr-defined]
    hashes = [normalize_sentence(t) for t in dataset.texts]  # type: ignore[attr-defined]
    return stratified_group_split(labels, hashes, seed=DEFAULT_SEED)


@pytest.mark.unit
def test_build_label_arrays_slices_train_and_val_only(phrasebank_sample: object) -> None:
    split = _sample_split(phrasebank_sample)
    tr_texts, tr_labels, va_texts, va_labels = build_label_arrays(
        phrasebank_sample,  # type: ignore[arg-type]
        split,
    )
    # Sizes match the split folds exactly.
    assert len(tr_texts) == len(tr_labels) == len(split.train)
    assert len(va_texts) == len(va_labels) == len(split.val)
    # The locked TEST fold is never returned.
    all_texts = set(tr_texts) | set(va_texts)
    test_texts = {phrasebank_sample.texts[i] for i in split.test}  # type: ignore[attr-defined]
    assert all_texts.isdisjoint(test_texts)
    # Every class appears in train (so the model can learn each).
    assert set(tr_labels) == set(range(N_CLASSES))


@pytest.mark.unit
def test_build_label_arrays_torch_free() -> None:
    """``build_label_arrays`` runs without importing torch/transformers/onnx."""
    from finbert_sentiment.data.load import sample_dataset

    texts = [t for t, _ in PHRASEBANK_SAMPLE]
    labels = [lab for _, lab in PHRASEBANK_SAMPLE]
    ds = sample_dataset(texts, labels)
    split = _sample_split(ds)
    build_label_arrays(ds, split)
    for forbidden in ("torch", "transformers", "onnx"):
        assert forbidden not in sys.modules


@pytest.mark.unit
def test_build_label_arrays_rejects_empty_train(phrasebank_sample: object) -> None:
    split = SplitIndices(train=(), val=(0, 1), test=(2,), seed=1)
    with pytest.raises(InsufficientDataError, match="train fold is empty"):
        build_label_arrays(phrasebank_sample, split)  # type: ignore[arg-type]


@pytest.mark.unit
def test_build_label_arrays_rejects_empty_val(phrasebank_sample: object) -> None:
    split = SplitIndices(train=(0, 1, 2), val=(), test=(3,), seed=1)
    with pytest.raises(InsufficientDataError):
        build_label_arrays(phrasebank_sample, split)  # type: ignore[arg-type]


@pytest.mark.unit
def test_build_label_arrays_rejects_train_missing_a_class(phrasebank_sample: object) -> None:
    # The first six sample rows are all "positive" (label 2) — a train fold of
    # only those is missing negative + neutral, which must be refused.
    pos_idx = tuple(i for i, lab in enumerate(phrasebank_sample.labels) if lab == 2)  # type: ignore[attr-defined]
    other = tuple(i for i in range(phrasebank_sample.n) if i not in pos_idx)  # type: ignore[attr-defined]
    split = SplitIndices(train=pos_idx, val=other[:1], test=other[1:], seed=1)
    with pytest.raises(InsufficientDataError):
        build_label_arrays(phrasebank_sample, split)  # type: ignore[arg-type]


@pytest.mark.unit
def test_build_label_arrays_rejects_out_of_range_index(phrasebank_sample: object) -> None:
    bad = phrasebank_sample.n + 5  # type: ignore[attr-defined]
    split = SplitIndices(train=(0, 1, bad), val=(2,), test=(3,), seed=1)
    with pytest.raises(FinbertSentimentError):
        build_label_arrays(phrasebank_sample, split)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# _compute_macro_f1: the torch-free metric callback used inside the Trainer     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_compute_macro_f1_perfect_and_matches_evaluation_metric() -> None:
    from finbert_sentiment.evaluation.metrics import macro_f1

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    assert _compute_macro_f1(y_true, y_pred) == pytest.approx(1.0)
    # Agrees with the project's own macro-F1 on an imperfect case.
    y_pred2 = [0, 2, 2, 0, 1, 1]
    assert _compute_macro_f1(y_true, y_pred2) == pytest.approx(macro_f1(y_true, y_pred2))


@pytest.mark.unit
def test_compute_macro_f1_handles_absent_predicted_class() -> None:
    # No example is predicted "positive" (2): that class's F1 is 0, not NaN.
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]
    val = _compute_macro_f1(y_true, y_pred)
    assert 0.0 <= val <= 1.0


# --------------------------------------------------------------------------- #
# write_metrics_json: the offline metrics the FastAPI router serves             #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_write_metrics_json_round_trips(tmp_path: object) -> None:
    import json

    out = write_metrics_json(
        {
            "served_model": "lexicon",
            "eval_macro_f1": None,
            "lexicon_macro_f1": 0.61,
            "class_prior_macro_f1": 0.28,
        },
        output_dir=str(tmp_path),  # type: ignore[arg-type]
    )
    assert out.endswith("metrics.json")
    with open(out, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["label_order"] == list(LABELS)
    assert loaded["served_model"] == "lexicon"
    assert loaded["lexicon_macro_f1"] == 0.61
    assert loaded["eval_macro_f1"] is None


@pytest.mark.unit
def test_write_metrics_json_rejects_unserializable(tmp_path: object) -> None:
    with pytest.raises(FinbertSentimentError):
        write_metrics_json({"bad": object()}, output_dir=str(tmp_path))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Heavy entry points: typed error (never ImportError) when [train] is absent    #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.skipif(train_available(), reason="[train] extra is installed; tested by train marker")
def test_train_distilbert_raises_typed_error_without_train_extra(
    phrasebank_sample: object,
) -> None:
    split = _sample_split(phrasebank_sample)
    with pytest.raises(FinbertSentimentError):
        train_distilbert(
            phrasebank_sample,  # type: ignore[arg-type]
            split,
            output_dir="/tmp/finbert_train_should_not_exist",
        )


@pytest.mark.unit
@pytest.mark.skipif(export_available(), reason="[train] extra is installed; tested by train marker")
def test_export_to_onnx_raises_typed_error_without_train_extra(tmp_path: object) -> None:
    # Use an existing dir so the missing-dir guard does not pre-empt the deps guard.
    with pytest.raises(FinbertSentimentError):
        export_to_onnx(str(tmp_path), output_dir=str(tmp_path))  # type: ignore[arg-type]


@pytest.mark.unit
def test_export_to_onnx_rejects_missing_model_dir() -> None:
    with pytest.raises(FinbertSentimentError):
        export_to_onnx("/no/such/model/dir", output_dir="/tmp/whatever")


@pytest.mark.unit
def test_export_to_onnx_rejects_bad_opset(tmp_path: object) -> None:
    with pytest.raises(FinbertSentimentError):
        export_to_onnx(str(tmp_path), output_dir=str(tmp_path), opset=9)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The ACTUAL fine-tune + export (skips unless the heavy [train] extra is here)  #
# --------------------------------------------------------------------------- #
@pytest.mark.train
@pytest.mark.slow
@pytest.mark.skipif(not train_available(), reason="requires the [train] extra (torch/transformers)")
def test_distilbert_fine_tune_and_export_end_to_end(
    phrasebank_sample: object, tmp_path: object
) -> None:  # pragma: no cover - heavy [train]-only path, skipped torch-free
    """A tiny seeded fine-tune trains, saves, and exports a parity-correct ONNX graph."""
    import os

    split = _sample_split(phrasebank_sample)
    model_dir = os.path.join(str(tmp_path), "model")  # type: ignore[arg-type]
    export_dir = os.path.join(str(tmp_path), "artifacts")  # type: ignore[arg-type]

    cfg = TrainConfig(epochs=1, batch_size=4, early_stopping_patience=0)
    result = train_distilbert(
        phrasebank_sample,  # type: ignore[arg-type]
        split,
        cfg,
        output_dir=model_dir,
    )
    assert isinstance(result, TrainResult)
    assert 0.0 <= result.best_val_macro_f1 <= 1.0
    assert result.label_order == LABELS
    assert os.path.isdir(model_dir)

    export = export_to_onnx(model_dir, output_dir=export_dir, int8=True, opset=14)
    assert isinstance(export, ExportResult)
    assert os.path.isfile(export.onnx_path)
    assert os.path.isfile(export.tokenizer_path)
    assert export.max_logit_abs_diff is not None
    assert export.max_logit_abs_diff < 1e-3
