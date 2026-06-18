"""Unit tests for the shared, working infrastructure.

These cover the pieces that are fully implemented at scaffold time: the
constants/label space, the seeded RNG, the run manifest + config hash, and the
input-validation guardrails. The pipeline-module stubs (data/baselines/inference/
evaluation) are exercised by their own partition tests as they are filled in.
"""

from __future__ import annotations

import numpy as np
import pytest

from finbert_sentiment import (
    LABELS,
    N_CLASSES,
    RunManifest,
    config_hash,
    make_rng,
    spawn_substreams,
)
from finbert_sentiment._constants import INDEX_TO_LABEL, LABEL_TO_INDEX
from finbert_sentiment._exceptions import InsufficientDataError, ValidationError
from finbert_sentiment._validation import (
    ensure_labels,
    ensure_score_matrix,
    ensure_text_batch,
    validate_min_per_class,
)


@pytest.mark.unit
def test_label_space_is_consistent() -> None:
    assert LABELS == ("negative", "neutral", "positive")
    assert N_CLASSES == 3
    assert LABEL_TO_INDEX == {"negative": 0, "neutral": 1, "positive": 2}
    assert INDEX_TO_LABEL == {0: "negative", 1: "neutral", 2: "positive"}


@pytest.mark.unit
def test_make_rng_is_deterministic() -> None:
    a = make_rng(123).random(5)
    b = make_rng(123).random(5)
    assert np.array_equal(a, b)


@pytest.mark.unit
def test_make_rng_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        make_rng(-1)


@pytest.mark.unit
def test_spawn_substreams_are_independent_and_reproducible() -> None:
    s1 = spawn_substreams(7, 3)
    s2 = spawn_substreams(7, 3)
    assert len(s1) == 3
    draws1 = [g.random(4) for g in s1]
    draws2 = [g.random(4) for g in s2]
    for d1, d2 in zip(draws1, draws2, strict=True):
        assert np.array_equal(d1, d2)
    # distinct substreams should not be identical
    assert not np.array_equal(draws1[0], draws1[1])


@pytest.mark.unit
def test_config_hash_is_order_invariant() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


@pytest.mark.unit
def test_run_manifest_roundtrips_to_dict() -> None:
    manifest = RunManifest.capture({"model": "lexicon"}, seed=42)
    d = manifest.to_dict()
    assert d["seed"] == 42
    assert set(d) >= {"git_sha", "dirty", "config_hash", "seed"}


@pytest.mark.unit
def test_ensure_text_batch_happy_path() -> None:
    out = ensure_text_batch(["a", " b "])
    assert out == ["a", " b "]


@pytest.mark.unit
def test_ensure_text_batch_rejects_bare_string() -> None:
    with pytest.raises(ValidationError):
        ensure_text_batch("not a list")


@pytest.mark.unit
def test_ensure_text_batch_rejects_empty_and_blank() -> None:
    with pytest.raises(ValidationError):
        ensure_text_batch([])
    with pytest.raises(ValidationError):
        ensure_text_batch(["ok", "   "])


@pytest.mark.unit
def test_ensure_text_batch_enforces_caps() -> None:
    with pytest.raises(ValidationError):
        ensure_text_batch(["x"] * 5, max_batch=4)
    with pytest.raises(ValidationError):
        ensure_text_batch(["x" * 11], max_chars=10)


@pytest.mark.unit
def test_ensure_labels_accepts_ints_and_names() -> None:
    arr = ensure_labels([0, "neutral", 2])
    assert arr.tolist() == [0, 1, 2]
    assert arr.dtype == np.int64


@pytest.mark.unit
def test_ensure_labels_validates_range_and_length() -> None:
    with pytest.raises(ValidationError):
        ensure_labels([0, 3])
    with pytest.raises(ValidationError):
        ensure_labels(["bogus"])
    with pytest.raises(ValidationError):
        ensure_labels([0, 1], n_expected=3)


@pytest.mark.unit
def test_ensure_score_matrix_shape_and_domain() -> None:
    m = ensure_score_matrix([[0.1, 0.2, 0.7]])
    assert m.shape == (1, 3)
    with pytest.raises(ValidationError):
        ensure_score_matrix([0.1, 0.2, 0.7])  # 1-D
    with pytest.raises(ValidationError):
        ensure_score_matrix([[0.1, 0.9]])  # wrong width
    with pytest.raises(ValidationError):
        ensure_score_matrix([[-0.1, 0.5, 0.6]])  # negative


@pytest.mark.unit
def test_validate_min_per_class() -> None:
    # every class present at least once -> ok
    validate_min_per_class([0, 1, 2, 0, 1, 2], min_count=1)
    with pytest.raises(InsufficientDataError):
        validate_min_per_class([0, 0, 1], min_count=1)  # class 2 missing
