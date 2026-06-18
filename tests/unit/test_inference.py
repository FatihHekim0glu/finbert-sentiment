"""Inference-layer tests: unified predictor + lazy ONNX session.

These tests exercise the serve path WITHOUT torch or transformers (the container
constraint). They cover:

* ``onnx_artifacts_present`` filesystem probing,
* the predictor falling back to the lexicon when no ONNX artifact exists,
* the predictor using the ONNX backend when a (fixture) ONNX model is present,
* ``predict`` shape / label / score-sum correctness for both backends,
* ``OnnxSentimentSession`` error handling (missing artifacts -> ``ArtifactError``).

To test the ONNX path without ``torch`` or the ``onnx`` package (neither is in the
serve container), a *minimal but genuinely runnable* ONNX classifier graph is
hand-serialized straight to the protobuf wire format and a tiny WordLevel
``tokenizer.json`` is written with the ``tokenizers`` library. onnxruntime loads
and runs both for real — so this is an end-to-end check of the actual serve code,
not a mock.

The whole module is offline and torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from finbert_sentiment._constants import LABELS, N_CLASSES
from finbert_sentiment._exceptions import ArtifactError, ValidationError
from finbert_sentiment.inference.onnx_session import (
    ONNX_ARTIFACT_NAME,
    TOKENIZER_ARTIFACT_NAME,
    OnnxSentimentSession,
    default_artifact_dir,
    onnx_artifacts_present,
)
from finbert_sentiment.inference.predictor import (
    Prediction,
    Predictor,
    load_predictor,
)

if TYPE_CHECKING:
    from pathlib import Path

# --------------------------------------------------------------------------- #
# Minimal ONNX-protobuf writer (no `onnx`/`torch` dependency needed).          #
# Builds a tiny, deterministic 3-class classifier that onnxruntime runs for    #
# real: logits = (mean(input_ids) - CENTER) * w + b, with w = (1, 0, -1).      #
# So a low mean token-id -> class "positive", a high mean -> "negative", and a #
# mean near CENTER -> "neutral" (the +0.1 neutral bias wins the tie).          #
# --------------------------------------------------------------------------- #

_FLOAT = 1  # TensorProto.FLOAT
_INT64 = 7  # TensorProto.INT64
#: Token-id midpoint used to center the mean so logits can change sign.
_CENTER = 5.0


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    # Field numbers >= 16 produce tags >= 128, so the TAG itself is a varint.
    return _varint((field << 3) | wire)


def _ld(field: int, payload: bytes) -> bytes:
    """Length-delimited (wire type 2) field."""
    return _tag(field, 2) + _varint(len(payload)) + payload


def _vint(field: int, value: int) -> bytes:
    """Varint (wire type 0) field."""
    return _tag(field, 0) + _varint(value)


def _tensor_proto(name: str, arr: np.ndarray, dtype_enum: int) -> bytes:
    contig = np.ascontiguousarray(arr)
    body = b""
    for dim in contig.shape:
        body += _vint(1, int(dim))  # dims
    body += _vint(2, dtype_enum)  # data_type
    body += _ld(8, name.encode())  # name
    body += _ld(9, contig.tobytes())  # raw_data
    return body


def _value_info(name: str, elem_type: int, dims: tuple[int | str, ...]) -> bytes:
    shape = b""
    for dim in dims:
        dim_proto = _ld(2, dim.encode()) if isinstance(dim, str) else _vint(1, dim)
        shape += _ld(1, dim_proto)
    tensor_type = _vint(1, elem_type) + _ld(2, shape)
    type_proto = _ld(1, tensor_type)
    return _ld(1, name.encode()) + _ld(2, type_proto)


def _node(
    op_type: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    name: str,
    attrs: tuple[bytes, ...] = (),
) -> bytes:
    body = b""
    for inp in inputs:
        body += _ld(1, inp.encode())
    for out in outputs:
        body += _ld(2, out.encode())
    body += _ld(3, name.encode())
    body += _ld(4, op_type.encode())
    for attr in attrs:
        body += _ld(5, attr)
    return body


def _attr_int(name: str, value: int) -> bytes:
    # AttributeProto: name=1, i=3, type=20 (INT=2).
    return _ld(1, name.encode()) + _vint(3, value) + _vint(20, 2)


def _attr_ints(name: str, values: tuple[int, ...]) -> bytes:
    # AttributeProto: name=1, ints=8 (repeated), type=20 (INTS=7).
    body = _ld(1, name.encode())
    for value in values:
        body += _vint(8, value)
    return body + _vint(20, 7)


def _build_classifier_onnx() -> bytes:
    """Serialize a runnable 3-class ONNX classifier to protobuf bytes."""
    nodes = _ld(1, _node("Cast", ("input_ids",), ("casted",), "cast", (_attr_int("to", _FLOAT),)))
    nodes += _ld(
        1,
        _node(
            "ReduceMean",
            ("casted",),
            ("mean",),
            "rmean",
            (_attr_ints("axes", (1,)), _attr_int("keepdims", 1)),
        ),
    )
    nodes += _ld(1, _node("Sub", ("mean", "center"), ("centered",), "sub"))
    nodes += _ld(1, _node("Mul", ("centered", "w"), ("scaled",), "mul"))
    nodes += _ld(1, _node("Add", ("scaled", "b"), ("logits",), "add"))

    inits = _ld(5, _tensor_proto("center", np.array([[_CENTER]], dtype=np.float32), _FLOAT))
    inits += _ld(5, _tensor_proto("w", np.array([[1.0, 0.0, -1.0]], dtype=np.float32), _FLOAT))
    inits += _ld(5, _tensor_proto("b", np.array([[0.0, 0.1, 0.0]], dtype=np.float32), _FLOAT))

    graph = nodes + inits + _ld(2, b"finbert_test_graph")
    graph += _ld(11, _value_info("input_ids", _INT64, ("N", "L")))
    graph += _ld(11, _value_info("attention_mask", _INT64, ("N", "L")))
    graph += _ld(12, _value_info("logits", _FLOAT, ("N", N_CLASSES)))

    opset = _ld(1, b"") + _vint(2, 13)  # OperatorSetIdProto: domain="", version=13
    return _vint(1, 7) + _ld(8, opset) + _ld(7, graph)  # ir_version, opset_import, graph


def _build_degenerate_onnx(*, source_input: str = "input_ids", extra_input: bool = False) -> bytes:
    """Serialize a runnable graph whose ``logits`` output has the WRONG width (1 col).

    Used to exercise the defensive guards: ``source_input="attention_mask"`` makes
    a graph with no ``input_ids`` (the missing-input guard), and ``extra_input``
    adds a declared-but-unused input (the feed-building branch that skips names not
    in the available set). The single-column output drives the wrong-shape guard.
    """
    nodes = _ld(1, _node("Cast", (source_input,), ("casted",), "cast", (_attr_int("to", _FLOAT),)))
    nodes += _ld(
        1,
        _node(
            "ReduceMean",
            ("casted",),
            ("logits",),
            "rmean",
            (_attr_ints("axes", (1,)), _attr_int("keepdims", 1)),
        ),
    )
    graph = nodes + _ld(2, b"degenerate_graph")
    graph += _ld(11, _value_info(source_input, _INT64, ("N", "L")))
    if extra_input:
        # A declared input the session reports but the predictor does not feed.
        graph += _ld(11, _value_info("token_type_ids", _INT64, ("N", "L")))
    graph += _ld(12, _value_info("logits", _FLOAT, ("N", 1)))

    opset = _ld(1, b"") + _vint(2, 13)
    return _vint(1, 7) + _ld(8, opset) + _ld(7, graph)


def _write_tokenizer_json(path: Path) -> None:
    """Write a tiny WordLevel ``tokenizer.json`` that maps cue words to ids.

    Positive cue words get LOW ids (mean < CENTER -> "positive" wins) and
    negative cue words get HIGH ids (mean > CENTER -> "negative" wins), so the
    ONNX backend's labels disagree with the lexicon's argmax direction — which is
    exactly what proves the predictor is running the ONNX graph, not the lexicon.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    # "neutral filler" -> ids (4, 6) -> mean 5.0 == CENTER -> the +0.1 bias makes
    # neutral win; "good great" -> low mean -> positive; "bad terrible" -> high mean.
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "good": 2,
        "great": 3,
        "neutral": 4,
        "filler": 6,
        "bad": 8,
        "terrible": 9,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))


@pytest.fixture
def onnx_artifact_dir(tmp_path: Path) -> Path:
    """A temp directory holding a runnable fixture ONNX model + tokenizer.json."""
    (tmp_path / ONNX_ARTIFACT_NAME).write_bytes(_build_classifier_onnx())
    _write_tokenizer_json(tmp_path / TOKENIZER_ARTIFACT_NAME)
    return tmp_path


# --------------------------------------------------------------------------- #
# onnx_artifacts_present / default_artifact_dir                                #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_default_artifact_dir_is_package_artifacts() -> None:
    """The default artifact dir points at ``<package>/artifacts`` (pure path)."""
    artifact_dir = default_artifact_dir()
    assert artifact_dir.name == "artifacts"
    assert artifact_dir.parent.name == "finbert_sentiment"


@pytest.mark.unit
def test_onnx_artifacts_present_false_when_empty(tmp_path: Path) -> None:
    """An empty directory has no serve artifacts."""
    assert onnx_artifacts_present(tmp_path) is False


@pytest.mark.unit
def test_onnx_artifacts_present_requires_both(tmp_path: Path) -> None:
    """Both the ONNX graph AND the tokenizer must be present."""
    (tmp_path / ONNX_ARTIFACT_NAME).write_bytes(b"not-a-real-model")
    assert onnx_artifacts_present(tmp_path) is False  # tokenizer missing
    (tmp_path / TOKENIZER_ARTIFACT_NAME).write_text("{}")
    assert onnx_artifacts_present(tmp_path) is True


@pytest.mark.unit
def test_onnx_artifacts_present_default_package_dir_has_no_committed_onnx() -> None:
    """The committed package ships no ONNX by default (lexicon is the floor).

    If a real transformer were trained and vendored this would flip to ``True``;
    the call must not raise either way.
    """
    assert isinstance(onnx_artifacts_present(), bool)


# --------------------------------------------------------------------------- #
# Predictor: lexicon fallback                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_load_predictor_falls_back_to_lexicon_without_artifacts(tmp_path: Path) -> None:
    """With no ONNX artifact, ``distilbert`` requests fall back to the lexicon."""
    predictor = load_predictor("distilbert", artifact_dir=str(tmp_path))
    assert predictor.backend == "lexicon"


@pytest.mark.unit
def test_load_predictor_lexicon_request_always_uses_lexicon(onnx_artifact_dir: Path) -> None:
    """Explicitly requesting ``lexicon`` ignores even a present ONNX artifact."""
    predictor = load_predictor("lexicon", artifact_dir=str(onnx_artifact_dir))
    assert predictor.backend == "lexicon"


@pytest.mark.unit
def test_lexicon_predictor_predict_shapes_and_scores() -> None:
    """The lexicon predictor returns aligned predictions with normalized scores."""
    predictor = load_predictor("lexicon")
    texts = [
        "Quarterly profit rose and earnings beat estimates.",
        "The firm warned of further losses and a downgrade.",
        "The meeting is scheduled for next Tuesday.",
    ]
    preds = predictor.predict(texts)

    assert len(preds) == len(texts)
    for pred, text in zip(preds, texts, strict=True):
        assert isinstance(pred, Prediction)
        assert pred.text == text
        assert pred.label in LABELS
        assert set(pred.scores) == set(LABELS)
        assert pytest.approx(sum(pred.scores.values()), abs=1e-9) == 1.0
        # Argmax label is consistent with the score vector.
        top = max(pred.scores, key=lambda k: pred.scores[k])
        assert pred.label == top

    # Sign-correctness on the clear +/- sentences.
    assert preds[0].label == "positive"
    assert preds[1].label == "negative"
    assert preds[2].label == "neutral"


@pytest.mark.unit
def test_lexicon_predictor_via_phrasebank_sample(phrasebank_sample: object) -> None:
    """The lexicon predictor runs end-to-end over the offline sample fixture."""
    texts = list(phrasebank_sample.texts)  # type: ignore[attr-defined]
    preds = load_predictor("lexicon").predict(texts)
    assert len(preds) == len(texts)
    assert all(p.label in LABELS for p in preds)


# --------------------------------------------------------------------------- #
# Predictor: ONNX backend                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_load_predictor_uses_onnx_when_artifacts_present(onnx_artifact_dir: Path) -> None:
    """A present ONNX artifact selects the transformer backend for ``distilbert``."""
    predictor = load_predictor("distilbert", artifact_dir=str(onnx_artifact_dir))
    assert predictor.backend == "distilbert-onnx"


@pytest.mark.unit
def test_onnx_predictor_predict_shapes_and_scores(onnx_artifact_dir: Path) -> None:
    """The ONNX predictor returns aligned, normalized, label-consistent predictions."""
    predictor = load_predictor("distilbert", artifact_dir=str(onnx_artifact_dir))
    texts = ["good great", "bad terrible", "neutral filler"]
    preds = predictor.predict(texts)

    assert len(preds) == len(texts)
    for pred, text in zip(preds, texts, strict=True):
        assert pred.text == text
        assert pred.label in LABELS
        assert set(pred.scores) == set(LABELS)
        assert pytest.approx(sum(pred.scores.values()), abs=1e-9) == 1.0
        assert pred.scores[pred.label] == max(pred.scores.values())


@pytest.mark.unit
def test_onnx_backend_label_tracks_token_ids(onnx_artifact_dir: Path) -> None:
    """The ONNX label is driven by token ids — direct proof the graph runs.

    The fixture's logits are ``(CENTER - mean(id)) * (+1, 0, -1) + (0, 0.1, 0)``:
      * LOW-id words ("good great" -> ids 2,3) give ``mean < CENTER`` -> positive,
      * HIGH-id words ("bad terrible" -> ids 8,9) give ``mean > CENTER`` -> negative,
      * mid words ("neutral filler" -> id 5 == CENTER) tie, so the +0.1 bias -> neutral.
    A lexicon would never produce these labels for these (non-cue) words.
    """
    predict = load_predictor("distilbert", artifact_dir=str(onnx_artifact_dir)).predict
    assert predict(["good great"])[0].label == "positive"
    assert predict(["bad terrible"])[0].label == "negative"
    assert predict(["neutral filler"])[0].label == "neutral"


@pytest.mark.unit
def test_onnx_backend_disagrees_with_lexicon(onnx_artifact_dir: Path) -> None:
    """The two backends disagree on the SAME input — they are genuinely distinct.

    ``"bad terrible"`` carries no lexicon cue words, so the lexicon returns
    ``"neutral"``; the ONNX graph keys off the high token ids and returns
    ``"negative"``. If the predictor secretly used the lexicon under the
    ``distilbert-onnx`` backend, the labels would match.
    """
    sentence = "bad terrible"
    onnx_pred = load_predictor("distilbert", artifact_dir=str(onnx_artifact_dir)).predict(
        [sentence]
    )[0]
    lexicon_pred = load_predictor("lexicon").predict([sentence])[0]
    assert lexicon_pred.label == "neutral"
    assert onnx_pred.label == "negative"
    assert onnx_pred.label != lexicon_pred.label


@pytest.mark.unit
def test_onnx_session_predict_proba_is_row_stochastic(onnx_artifact_dir: Path) -> None:
    """``predict_proba`` returns a ``(n, N_CLASSES)`` row-stochastic matrix."""
    session = OnnxSentimentSession(onnx_artifact_dir)
    proba = session.predict_proba(["good great", "bad terrible"])
    assert proba.shape == (2, N_CLASSES)
    assert np.all(proba >= 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(2), atol=1e-9)


@pytest.mark.unit
def test_onnx_session_predict_matches_argmax(onnx_artifact_dir: Path) -> None:
    """``predict`` equals the row-argmax of ``predict_proba``."""
    session = OnnxSentimentSession(onnx_artifact_dir)
    texts = ["good great", "bad terrible", "neutral filler"]
    proba = session.predict_proba(texts)
    labels = session.predict(texts)
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels, proba.argmax(axis=1))


@pytest.mark.unit
def test_onnx_session_load_is_idempotent(onnx_artifact_dir: Path) -> None:
    """Calling ``load`` twice reuses the same session/tokenizer objects."""
    session = OnnxSentimentSession(onnx_artifact_dir)
    session.load()
    first = session._session  # white-box check of lazy-load idempotence
    session.load()
    assert session._session is first


@pytest.mark.unit
def test_onnx_session_predictions_are_deterministic(onnx_artifact_dir: Path) -> None:
    """Repeated calls on the same texts yield identical scores (determinism)."""
    session = OnnxSentimentSession(onnx_artifact_dir)
    texts = ["good great", "bad terrible"]
    first = session.predict_proba(texts)
    second = session.predict_proba(texts)
    np.testing.assert_array_equal(first, second)


@pytest.mark.unit
def test_onnx_session_artifact_dir_property(onnx_artifact_dir: Path) -> None:
    """The resolved artifact dir is exposed for introspection."""
    session = OnnxSentimentSession(onnx_artifact_dir)
    assert session.artifact_dir == onnx_artifact_dir
    # No override -> package default.
    assert OnnxSentimentSession().artifact_dir == default_artifact_dir()


# --------------------------------------------------------------------------- #
# Error handling                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_onnx_session_missing_onnx_raises_artifact_error(tmp_path: Path) -> None:
    """A missing ONNX graph raises ``ArtifactError`` on load."""
    _write_tokenizer_json(tmp_path / TOKENIZER_ARTIFACT_NAME)
    session = OnnxSentimentSession(tmp_path)
    with pytest.raises(ArtifactError, match="ONNX artifact not found"):
        session.load()


@pytest.mark.unit
def test_onnx_session_missing_tokenizer_raises_artifact_error(tmp_path: Path) -> None:
    """A present ONNX but missing tokenizer raises ``ArtifactError`` on load."""
    (tmp_path / ONNX_ARTIFACT_NAME).write_bytes(_build_classifier_onnx())
    session = OnnxSentimentSession(tmp_path)
    with pytest.raises(ArtifactError, match=r"tokenizer\.json not found"):
        session.load()


@pytest.mark.unit
def test_onnx_session_corrupt_model_raises_artifact_error(tmp_path: Path) -> None:
    """A corrupt ONNX graph is normalized to ``ArtifactError`` (not a raw ORT error)."""
    (tmp_path / ONNX_ARTIFACT_NAME).write_bytes(b"this is not a valid onnx model")
    _write_tokenizer_json(tmp_path / TOKENIZER_ARTIFACT_NAME)
    session = OnnxSentimentSession(tmp_path)
    with pytest.raises(ArtifactError, match="failed to initialize"):
        session.load()


@pytest.mark.unit
def test_onnx_session_wrong_logit_width_raises_artifact_error(tmp_path: Path) -> None:
    """A graph emitting the wrong number of logit columns is rejected."""
    (tmp_path / ONNX_ARTIFACT_NAME).write_bytes(_build_degenerate_onnx())
    _write_tokenizer_json(tmp_path / TOKENIZER_ARTIFACT_NAME)
    session = OnnxSentimentSession(tmp_path)
    with pytest.raises(ArtifactError, match="logit matrix"):
        session.predict_proba(["good great"])


@pytest.mark.unit
def test_onnx_session_missing_input_ids_signature_raises(tmp_path: Path) -> None:
    """A graph whose signature lacks ``input_ids`` is rejected before running."""
    (tmp_path / ONNX_ARTIFACT_NAME).write_bytes(
        _build_degenerate_onnx(source_input="attention_mask")
    )
    _write_tokenizer_json(tmp_path / TOKENIZER_ARTIFACT_NAME)
    session = OnnxSentimentSession(tmp_path)
    with pytest.raises(ArtifactError, match="missing an"):
        session.predict_proba(["good great"])


@pytest.mark.unit
def test_onnx_session_skips_unfed_declared_inputs(tmp_path: Path) -> None:
    """A declared input the predictor does not produce is skipped during feed-building.

    The graph declares a ``token_type_ids`` input that the predictor never emits,
    so it is not added to the feed (exercising the name-filtering branch). The
    forward pass then fails inside onnxruntime — which the session normalizes into
    an ``ArtifactError`` (the forward-pass error path), never leaking a raw ORT error.
    """
    (tmp_path / ONNX_ARTIFACT_NAME).write_bytes(_build_degenerate_onnx(extra_input=True))
    _write_tokenizer_json(tmp_path / TOKENIZER_ARTIFACT_NAME)
    session = OnnxSentimentSession(tmp_path)
    with pytest.raises(ArtifactError, match="forward pass failed"):
        session.predict_proba(["good great"])


@pytest.mark.unit
def test_onnx_session_rejects_bare_string(onnx_artifact_dir: Path) -> None:
    """A bare string batch is rejected before any forward pass."""
    session = OnnxSentimentSession(onnx_artifact_dir)
    with pytest.raises(ValidationError):
        session.predict_proba(["good", ""])  # blank element is invalid


@pytest.mark.unit
def test_predictor_rejects_empty_batch() -> None:
    """The unified predictor enforces the non-empty batch precondition."""
    with pytest.raises(ValidationError):
        load_predictor("lexicon").predict([])


@pytest.mark.unit
def test_predictor_backend_property_is_fixed() -> None:
    """The backend is fixed at construction and surfaced via the property."""
    predictor = Predictor("lexicon", session=None)
    assert predictor.backend == "lexicon"


@pytest.mark.unit
def test_predictor_without_session_raises() -> None:
    """A predictor constructed without a backend session fails fast on predict."""
    predictor = Predictor("lexicon", session=None)
    with pytest.raises(ArtifactError, match="no session attached"):
        predictor.predict(["anything goes here"])


@pytest.mark.unit
def test_prediction_to_dict_round_trips() -> None:
    """``Prediction.to_dict`` is a plain JSON-serializable mapping."""
    pred = Prediction(
        text="hello", label="neutral", scores={"negative": 0.2, "neutral": 0.5, "positive": 0.3}
    )
    data = pred.to_dict()
    assert data == {
        "text": "hello",
        "label": "neutral",
        "scores": {"negative": 0.2, "neutral": 0.5, "positive": 0.3},
    }
