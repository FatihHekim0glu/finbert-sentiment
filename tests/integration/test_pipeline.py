"""End-to-end pipeline integration (offline, no network, no torch).

The headline integration guard: ``load (offline sample) -> dedup -> seeded group
split -> lexicon eval`` runs to completion on the ``phrasebank_sample`` fixture
WITHOUT any network call or deep-learning framework, and the resulting split has
no train/test sentence-hash overlap. This is the cross-group wiring test: it
threads the data, baseline, and evaluation modules together exactly as the
``finbert-sentiment evaluate`` (lexicon) path does, and asserts the honest
metric bundle comes out well-formed.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from finbert_sentiment._constants import N_CLASSES
from finbert_sentiment.baselines.class_prior import ClassPriorClassifier
from finbert_sentiment.baselines.lexicon import LexiconClassifier
from finbert_sentiment.data.dedup import dedup_sentences
from finbert_sentiment.data.split import assert_no_group_overlap, stratified_group_split
from finbert_sentiment.evaluation.metrics import classification_report
from finbert_sentiment.evaluation.verdict import Verdict, derive_verdict

if TYPE_CHECKING:
    from finbert_sentiment.data.load import LabelledDataset

pytestmark = pytest.mark.integration

_SEED = 20260618


def test_offline_pipeline_runs_end_to_end(phrasebank_sample: LabelledDataset) -> None:
    """load -> dedup -> group split -> lexicon eval on the offline sample (no network)."""
    # 1. Dedup the offline sample (collapses the near-duplicate pairs).
    deduped = dedup_sentences(phrasebank_sample)
    assert deduped.n_dropped >= 1, "the offline sample contains near-duplicate pairs to collapse"
    dataset = deduped.dataset

    # 2. Seeded stratified group split, leakage-guarded.
    split = stratified_group_split(dataset.labels, deduped.group_hashes, seed=_SEED)
    assert_no_group_overlap(split, deduped.group_hashes)
    # Every fold non-empty and disjoint cover of the deduped rows.
    assert split.train and split.val and split.test
    assert set(split.train) | set(split.val) | set(split.test) == set(range(dataset.n))

    # 3. Lexicon eval on the LOCKED test fold (the served-model floor).
    test_idx = list(split.test)
    train_idx = list(split.train)
    test_texts = [dataset.texts[i] for i in test_idx]
    y_true = [dataset.labels[i] for i in test_idx]

    lexicon = LexiconClassifier()
    y_lexicon = [int(v) for v in lexicon.predict(test_texts)]
    report = classification_report(y_true, y_lexicon, n_resamples=64, seed=_SEED)

    # The honest bundle is well-formed: macro-F1 in range, CI brackets the point.
    assert 0.0 <= report.macro_f1 <= 1.0
    assert report.macro_f1_ci is not None
    lo, hi = report.macro_f1_ci
    assert 0.0 <= lo <= report.macro_f1 <= hi <= 1.0
    assert len(report.confusion) == N_CLASSES
    assert all(len(row) == N_CLASSES for row in report.confusion)
    assert sum(int(c) for row in report.confusion for c in row) == len(y_true)

    # 4. Class-prior floor fit on TRAIN labels only (no test label leaks in).
    prior = ClassPriorClassifier.fit([dataset.labels[i] for i in train_idx])
    y_prior = [int(v) for v in prior.predict(test_texts)]
    prior_macro_f1 = classification_report(y_true, y_prior, bootstrap_ci=False).macro_f1
    assert 0.0 <= prior_macro_f1 <= 1.0

    # 5. The verdict is lexicon-only on the fallback path (no transformer served).
    verdict = derive_verdict(None, report.macro_f1, None)
    assert verdict.verdict is Verdict.LEXICON_ONLY
    assert verdict.beats_lexicon is None


def test_offline_pipeline_imports_nothing_heavy() -> None:
    """The whole offline load->dedup->split->lexicon-eval path imports no torch/onnx.

    Run in a clean subprocess so the assertion reflects only the modules the
    pipeline itself imports, immune to ``sys.modules`` pollution from sibling
    in-process suites (the ONNX-session tests fault ``onnxruntime`` in-process).
    """
    code = (
        "import sys\n"
        "from finbert_sentiment.data.load import sample_dataset\n"
        "from finbert_sentiment.data.dedup import dedup_sentences\n"
        "from finbert_sentiment.data.split import (\n"
        "    assert_no_group_overlap,\n"
        "    stratified_group_split,\n"
        ")\n"
        "from finbert_sentiment.baselines.lexicon import LexiconClassifier\n"
        "from finbert_sentiment.evaluation.metrics import classification_report\n"
        "texts = [\n"
        "    'Quarterly profit rose sharply as revenue gains beat estimates.',\n"
        "    'The company reported record growth and raised guidance.',\n"
        "    'Operating margins improved and the stock surged to a new high.',\n"
        "    'Quarterly loss widened as revenue declined and margins fell.',\n"
        "    'The company cut its guidance after a sharp drop in demand.',\n"
        "    'Profit plunged and the stock dropped to a multi-year low.',\n"
        "    'The company will hold its annual general meeting in May.',\n"
        "    'The headquarters are located in Helsinki, Finland.',\n"
        "    'The report covers the fiscal period ending in December.',\n"
        "]\n"
        "labels = ['positive', 'positive', 'positive', 'negative', 'negative',\n"
        "          'negative', 'neutral', 'neutral', 'neutral']\n"
        "ds = sample_dataset(texts, labels, source='offline')\n"
        "deduped = dedup_sentences(ds)\n"
        "split = stratified_group_split(deduped.dataset.labels, deduped.group_hashes, seed=1)\n"
        "assert_no_group_overlap(split, deduped.group_hashes)\n"
        "test_texts = [deduped.dataset.texts[i] for i in split.test]\n"
        "y_true = [deduped.dataset.labels[i] for i in split.test]\n"
        "y_pred = [int(v) for v in LexiconClassifier().predict(test_texts)]\n"
        "report = classification_report(y_true, y_pred, n_resamples=32, seed=1)\n"
        "assert 0.0 <= report.macro_f1 <= 1.0\n"
        "heavy = sorted(\n"
        "    m for m in ('torch', 'transformers', 'onnx', 'onnxruntime')\n"
        "    if m in sys.modules\n"
        ")\n"
        "assert heavy == [], f'pipeline pulled in heavy modules: {heavy}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"offline-pipeline purity subprocess failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


def test_run_sentiment_backend_entrypoint_offline() -> None:
    """The backend ``run_sentiment`` entrypoint serves live and reports committed eval.

    Uses the SHIPPED ``artifacts/metrics.json`` (no network). The served model is
    whichever build shipped — the transformer-ONNX graph when its artifacts are
    present, else the torch-free lexicon fallback. Asserts the honest contract that
    holds for BOTH: eval scalars come from the committed metrics verbatim, the
    served model is named, the baselines are ordered, the transformer
    ``measured``/``beats_lexicon`` flags are mutually consistent, live predictions
    are sign-correct on the clear cases, and the two figures are well-formed.
    """
    import json

    from finbert_sentiment.service import run_sentiment

    result = run_sentiment(
        [
            "Quarterly profit rose sharply and beat estimates.",
            "Losses widened as demand fell and the stock dropped.",
            "The annual general meeting will be held in May.",
        ],
        model_pref="distilbert",  # serves ONNX if present, else falls back to lexicon
        seed=20260618,
    )
    summary = result.summary
    assert summary.served_model in {"distilbert-onnx", "lexicon"}
    assert summary.n_texts == 3
    # eval_macro_f1 is the committed locked-test-set value (NOT recomputed here).
    assert summary.eval_macro_f1 is not None and 0.0 <= summary.eval_macro_f1 <= 1.0
    assert summary.class_prior_macro_f1 is not None
    # Lexicon clearly beats the trivial class-prior floor.
    assert summary.lexicon_macro_f1 is not None
    assert summary.lexicon_macro_f1 > summary.class_prior_macro_f1

    if summary.served_model == "lexicon":
        # Lexicon fallback: the lexicon IS the served model; nothing to compare.
        assert summary.lexicon_macro_f1 == summary.eval_macro_f1
        assert summary.beats_lexicon is None
        assert summary.transformer_measured is False
    else:
        # Transformer-ONNX build: a real, MEASURED verdict and figure.
        assert summary.transformer_measured is True
        assert isinstance(summary.beats_lexicon, bool)

    # Live predictions are sign-correct on the clear cases (both backends agree here).
    labels = [p["label"] for p in result.predictions]
    assert labels == ["positive", "negative", "neutral"]

    # Figures are well-formed and the whole response is JSON-serializable.
    assert set(result.confusion_figure) == {"data", "layout"}
    assert set(result.per_class_f1_figure) == {"data", "layout"}
    json.dumps(result.to_dict())


def test_run_sentiment_serve_path_imports_no_torch() -> None:
    """The ``run_sentiment`` serve path imports no torch/transformers (subprocess).

    The brief's headline serve constraint: the backend entrypoint serves through
    onnxruntime/lexicon ONLY. A clean subprocess proves that running it loads no
    deep-learning training framework, regardless of in-process pollution.
    """
    code = (
        "import sys\n"
        "from finbert_sentiment.service import run_sentiment\n"
        "res = run_sentiment(['Profit rose and revenue grew.'], model_pref='lexicon')\n"
        "assert res.summary.served_model == 'lexicon'\n"
        "heavy = sorted(\n"
        "    m for m in ('torch', 'transformers', 'onnx') if m in sys.modules\n"
        ")\n"
        "assert heavy == [], f'serve path imported heavy modules: {heavy}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"run_sentiment serve-purity subprocess failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
