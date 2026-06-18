"""Unit tests for the evaluation layer: bootstrap CI, McNemar, verdict, honesty guards.

These cover the evaluation-group kernels that are not pure sklearn-parity checks
(those live in ``tests/parity/test_metrics_parity.py``):

* the seeded bootstrap macro-F1 CI is sane (brackets the point estimate, is
  deterministic, narrows with more data, and validates its inputs);
* McNemar's test is correct on hand-worked cases and against ``statsmodels`` when
  available (exact and chi-square branches);
* the ``derive_verdict`` truth table is pinned (the honest ``beats_lexicon`` rule
  — True only on margin AND significance, None for the lexicon-only build);
* a CATEGORY-ERROR guard: the evaluation layer computes NO Sharpe / DSR /
  walk-forward / purge — sentiment is a text label, not a return series, so those
  finance metrics do not exist anywhere in this package.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil

import numpy as np
import pytest

import finbert_sentiment
from finbert_sentiment._exceptions import InsufficientDataError, ValidationError
from finbert_sentiment.evaluation.mcnemar import (
    McNemarResult,
    _binom_sf_inclusive,
    _chi2_sf_1df,
    mcnemar_test,
)
from finbert_sentiment.evaluation.metrics import (
    ClassificationReport,
    accuracy,
    bootstrap_macro_f1_ci,
    classification_report,
    macro_f1,
)
from finbert_sentiment.evaluation.verdict import Verdict, VerdictResult, derive_verdict

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# bootstrap macro-F1 CI                                                        #
# --------------------------------------------------------------------------- #
def _balanced_pair(n_per_class: int, n_wrong: int) -> tuple[list[int], list[int]]:
    """A balanced 3-class ``(y_true, y_pred)`` with exactly ``n_wrong`` errors."""
    y_true = [0] * n_per_class + [1] * n_per_class + [2] * n_per_class
    y_pred = list(y_true)
    for i in range(n_wrong):
        y_pred[i] = (y_pred[i] + 1) % 3
    return y_true, y_pred


def test_bootstrap_ci_brackets_point_estimate() -> None:
    """The bootstrap CI contains the point macro-F1 (with a tolerance band)."""
    y_true, y_pred = _balanced_pair(n_per_class=40, n_wrong=12)
    point = macro_f1(y_true, y_pred)
    low, high = bootstrap_macro_f1_ci(y_true, y_pred, n_resamples=500, seed=7)
    assert 0.0 <= low <= high <= 1.0
    # The 95% CI should comfortably surround the observed estimate.
    assert low - 0.1 <= point <= high + 0.1


def test_bootstrap_ci_is_deterministic_under_seed() -> None:
    """Same seed -> identical CI; different seed -> (generally) different CI."""
    y_true, y_pred = _balanced_pair(n_per_class=30, n_wrong=9)
    a = bootstrap_macro_f1_ci(y_true, y_pred, n_resamples=300, seed=123)
    b = bootstrap_macro_f1_ci(y_true, y_pred, n_resamples=300, seed=123)
    assert a == b
    c = bootstrap_macro_f1_ci(y_true, y_pred, n_resamples=300, seed=999)
    assert c != a  # overwhelmingly likely to differ


def test_bootstrap_ci_narrows_with_more_data() -> None:
    """A larger test set yields a tighter macro-F1 CI (lower sampling variance)."""

    def width(n_per_class: int) -> float:
        y_true, y_pred = _balanced_pair(n_per_class, n_wrong=n_per_class // 4)
        low, high = bootstrap_macro_f1_ci(y_true, y_pred, n_resamples=400, seed=3)
        return high - low

    assert width(120) < width(15)


def test_bootstrap_ci_zero_width_when_perfect() -> None:
    """Perfect predictions -> every resample scores 1.0 -> a degenerate CI at 1.0."""
    y_true = [0, 1, 2] * 20
    low, high = bootstrap_macro_f1_ci(y_true, y_true, n_resamples=200, seed=1)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_bootstrap_ci_rejects_bad_confidence() -> None:
    """Confidence outside (0, 1) is a ValidationError."""
    y_true, y_pred = _balanced_pair(10, 3)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValidationError):
            bootstrap_macro_f1_ci(y_true, y_pred, confidence=bad)


def test_bootstrap_ci_rejects_too_few_examples() -> None:
    """Fewer than two examples cannot support a bootstrap CI."""
    with pytest.raises(InsufficientDataError):
        bootstrap_macro_f1_ci([1], [1])


def test_bootstrap_ci_rejects_nonpositive_resamples() -> None:
    """n_resamples must be >= 1."""
    y_true, y_pred = _balanced_pair(10, 3)
    with pytest.raises(ValidationError):
        bootstrap_macro_f1_ci(y_true, y_pred, n_resamples=0)


# --------------------------------------------------------------------------- #
# classification_report bundle                                                 #
# --------------------------------------------------------------------------- #
def test_classification_report_assembles_full_bundle() -> None:
    """The report carries macro-F1, accuracy, per-class vectors, confusion, n, CI."""
    y_true, y_pred = _balanced_pair(n_per_class=20, n_wrong=6)
    rep = classification_report(y_true, y_pred, n_resamples=200, seed=5)
    assert isinstance(rep, ClassificationReport)
    assert rep.n == 60
    assert rep.macro_f1 == pytest.approx(macro_f1(y_true, y_pred))
    assert rep.accuracy == pytest.approx(accuracy(y_true, y_pred))
    assert len(rep.per_class_f1) == 3
    assert len(rep.confusion) == 3 and len(rep.confusion[0]) == 3
    assert rep.macro_f1_ci is not None
    low, high = rep.macro_f1_ci
    assert 0.0 <= low <= high <= 1.0
    # round-trips through JSON cleanly (tuples serialize to arrays)
    d = rep.to_dict()
    assert isinstance(d["macro_f1"], float)
    round_tripped = json.loads(json.dumps(d))
    assert round_tripped["n"] == 60
    assert len(round_tripped["confusion"]) == 3


def test_classification_report_without_ci() -> None:
    """``bootstrap_ci=False`` leaves the CI unset."""
    y_true, y_pred = _balanced_pair(10, 2)
    rep = classification_report(y_true, y_pred, bootstrap_ci=False)
    assert rep.macro_f1_ci is None


def test_accuracy_matches_manual() -> None:
    """``accuracy`` is the fraction of exactly-correct predictions."""
    assert accuracy([0, 1, 2, 0], [0, 1, 2, 1]) == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# McNemar's test                                                               #
# --------------------------------------------------------------------------- #
def test_mcnemar_counts_discordant_pairs() -> None:
    """n01 / n10 count exactly the right discordant outcomes."""
    # idx: 0 both correct, 1 model-only correct (n10), 2 lexicon-only (n01),
    #      3 both wrong, 4 model-only correct (n10)
    y_true = [0, 0, 0, 0, 0]
    y_pred_model = [0, 0, 1, 1, 0]  # correct at 0,1,4
    y_pred_lexicon = [0, 1, 0, 2, 1]  # correct at 0,2
    res = mcnemar_test(y_true, y_pred_model, y_pred_lexicon)
    assert res.n10 == 2  # idx 1, 4
    assert res.n01 == 1  # idx 2
    assert res.exact is True  # only 3 discordant pairs


def test_mcnemar_balanced_discordant_is_not_significant() -> None:
    """A perfectly balanced discordant split gives a p-value of exactly 1.0."""
    # 4 discordant in each direction, all else concordant-correct.
    y_true = [0] * 20
    model = [0] * 20
    lexicon = [0] * 20
    for i in range(4):  # model right, lexicon wrong (n10)
        lexicon[i] = 1
    for i in range(4, 8):  # lexicon right, model wrong (n01)
        model[i] = 1
    res = mcnemar_test(y_true, model, lexicon)
    assert res.n10 == 4 and res.n01 == 4
    assert res.p_value == pytest.approx(1.0)


def test_mcnemar_lopsided_small_counts_uses_exact_and_is_significant() -> None:
    """Strongly lopsided discordance (exact branch) yields a small p-value."""
    # model right & lexicon wrong on 12 items; the reverse on 0 -> p = 2 * 0.5**12
    n = 12
    y_true = [0] * n
    model = [0] * n  # all correct
    lexicon = [1] * n  # all wrong
    res = mcnemar_test(y_true, model, lexicon)
    assert res.exact is True
    assert res.n10 == n and res.n01 == 0
    assert res.p_value == pytest.approx(2.0 * 0.5**n)
    assert res.p_value < 0.05


def test_mcnemar_chi2_branch_above_threshold() -> None:
    """Above the discordant threshold the chi-square (non-exact) branch is taken."""
    # 60 discordant pairs, lopsided 45/15.
    y_true = [0] * 60
    model = [0] * 60
    lexicon = [0] * 60
    for i in range(45):  # n10
        lexicon[i] = 1
    for i in range(45, 60):  # n01
        model[i] = 1
    res = mcnemar_test(y_true, model, lexicon, exact_threshold=25)
    assert res.exact is False
    assert not np.isnan(res.statistic)
    # Edwards continuity-corrected chi-square: (|45-15|-1)^2 / 60.
    assert res.statistic == pytest.approx((abs(45 - 15) - 1) ** 2 / 60)
    assert res.p_value < 0.05


def test_mcnemar_no_discordance_is_pvalue_one() -> None:
    """Identical predictions -> zero discordant pairs -> p-value 1.0."""
    y_true = [0, 1, 2, 0, 1]
    res = mcnemar_test(y_true, y_true, y_true)
    assert res.n01 == 0 and res.n10 == 0
    assert res.p_value == pytest.approx(1.0)
    assert res.exact is True


def test_mcnemar_rejects_negative_threshold() -> None:
    """A negative exact_threshold is invalid."""
    with pytest.raises(ValidationError):
        mcnemar_test([0, 1], [0, 1], [1, 0], exact_threshold=-1)


def test_mcnemar_result_to_dict_is_json_friendly() -> None:
    """The result serializes to plain Python scalars."""
    res = mcnemar_test([0, 0, 0], [0, 0, 1], [0, 1, 1])
    d = res.to_dict()
    assert set(d) == {"n01", "n10", "statistic", "p_value", "exact"}
    assert isinstance(d["n01"], int) and isinstance(d["p_value"], float)


def test_binom_sf_inclusive_edges() -> None:
    """The internal binomial tail helper is correct at its boundaries."""
    assert _binom_sf_inclusive(0, 5) == pytest.approx(1.0)
    assert _binom_sf_inclusive(6, 5) == pytest.approx(0.0)
    assert _binom_sf_inclusive(5, 5) == pytest.approx(0.5**5)


def test_chi2_sf_1df_boundaries() -> None:
    """The 1-df chi-square survival helper: 1.0 at/below 0, decreasing above."""
    assert _chi2_sf_1df(0.0) == pytest.approx(1.0)
    assert _chi2_sf_1df(-3.0) == pytest.approx(1.0)
    # erfc(sqrt(x/2)) is strictly decreasing for x > 0.
    assert _chi2_sf_1df(3.841) == pytest.approx(0.05, abs=1e-3)  # 5% critical value
    assert _chi2_sf_1df(10.0) < _chi2_sf_1df(1.0) < 1.0


@pytest.mark.parametrize(
    "n10,n01,exact",
    [(2, 1, True), (0, 12, True), (45, 15, False), (4, 4, True)],
)
def test_mcnemar_matches_statsmodels_when_available(n10: int, n01: int, exact: bool) -> None:
    """Cross-check p-values against statsmodels' McNemar (skipped if absent)."""
    sm = pytest.importorskip("statsmodels.stats.contingency_tables")
    # Build a contingency table with n11 concordant-correct, n00 concordant-wrong
    # both zero so only the discordant cells matter.
    table = np.array([[0, n01], [n10, 0]], dtype=float)
    # statsmodels: exact uses binomial; correction=True is Edwards' chi-square.
    ref = sm.mcnemar(table, exact=exact, correction=not exact)

    # Reconstruct prediction vectors realizing (n10, n01).
    y_true = [0] * (n10 + n01)
    model = [0] * (n10 + n01)
    lexicon = [0] * (n10 + n01)
    for i in range(n10):  # model right, lexicon wrong
        lexicon[i] = 1
    for i in range(n10, n10 + n01):  # lexicon right, model wrong
        model[i] = 1
    threshold = 25
    ours = mcnemar_test(y_true, model, lexicon, exact_threshold=threshold)
    assert ours.p_value == pytest.approx(float(ref.pvalue), abs=1e-9)


# --------------------------------------------------------------------------- #
# derive_verdict truth table (the honest beats_lexicon rule)                   #
# --------------------------------------------------------------------------- #
def test_verdict_lexicon_only_when_no_transformer() -> None:
    """No model macro-F1 -> LEXICON_ONLY, beats_lexicon is None (nothing to compare)."""
    res = derive_verdict(None, 0.70, None)
    assert res.verdict is Verdict.LEXICON_ONLY
    assert res.beats_lexicon is None
    assert isinstance(res, VerdictResult)


def test_verdict_true_only_on_margin_and_significance() -> None:
    """beats_lexicon is True iff margin cleared AND McNemar significant."""
    res = derive_verdict(0.88, 0.70, 0.001, alpha=0.05, min_margin=0.02)
    assert res.verdict is Verdict.MODEL_BEATS_LEXICON
    assert res.beats_lexicon is True


def test_verdict_false_when_margin_insufficient() -> None:
    """Significant but the margin is too small -> NO_SIGNIFICANT_DIFFERENCE."""
    res = derive_verdict(0.705, 0.70, 0.001, alpha=0.05, min_margin=0.02)
    assert res.verdict is Verdict.NO_SIGNIFICANT_DIFFERENCE
    assert res.beats_lexicon is False


def test_verdict_false_when_not_significant() -> None:
    """A big margin but a non-significant McNemar p -> still does NOT beat the lexicon."""
    res = derive_verdict(0.90, 0.70, 0.20, alpha=0.05, min_margin=0.02)
    assert res.verdict is Verdict.NO_SIGNIFICANT_DIFFERENCE
    assert res.beats_lexicon is False


def test_verdict_false_when_both_fail() -> None:
    """Neither margin nor significance -> NO_SIGNIFICANT_DIFFERENCE."""
    res = derive_verdict(0.705, 0.70, 0.30)
    assert res.beats_lexicon is False


def test_verdict_margin_boundary_is_inclusive() -> None:
    """A margin of exactly ``min_margin`` clears the bar (>= is inclusive)."""
    res = derive_verdict(0.72, 0.70, 0.001, alpha=0.05, min_margin=0.02)
    assert res.beats_lexicon is True


def test_verdict_alpha_boundary_is_strict() -> None:
    """p exactly equal to alpha does NOT count as significant (strict <)."""
    res = derive_verdict(0.90, 0.70, 0.05, alpha=0.05, min_margin=0.02)
    assert res.beats_lexicon is False


@pytest.mark.parametrize(
    "model_f1,lex_f1,p,expected",
    [
        (None, 0.6, None, None),
        (0.95, 0.60, 0.0001, True),
        (0.61, 0.60, 0.0001, False),  # margin too small
        (0.95, 0.60, 0.50, False),  # not significant
        (0.30, 0.60, 0.0001, False),  # worse than lexicon
    ],
)
def test_verdict_truth_table(
    model_f1: float | None, lex_f1: float, p: float | None, expected: bool | None
) -> None:
    """Pin the full ``beats_lexicon`` truth table."""
    assert derive_verdict(model_f1, lex_f1, p).beats_lexicon is expected


def test_verdict_to_dict_serializes_enum() -> None:
    """``to_dict`` emits the stable string verdict value."""
    d = derive_verdict(0.95, 0.6, 0.0001).to_dict()
    assert d["verdict"] == "model_beats_lexicon"
    assert d["beats_lexicon"] is True
    assert isinstance(d["rationale"], str)


def test_verdict_rejects_out_of_range_inputs() -> None:
    """Out-of-range macro-F1 / p / alpha / margin are ValidationErrors."""
    with pytest.raises(ValidationError):
        derive_verdict(0.9, 1.5, 0.01)  # lexicon_macro_f1 > 1
    with pytest.raises(ValidationError):
        derive_verdict(1.5, 0.6, 0.01)  # model_macro_f1 > 1
    with pytest.raises(ValidationError):
        derive_verdict(0.9, 0.6, 1.5)  # p > 1
    with pytest.raises(ValidationError):
        derive_verdict(0.9, 0.6, 0.01, alpha=0.0)  # alpha not in (0, 1)
    with pytest.raises(ValidationError):
        derive_verdict(0.9, 0.6, 0.01, min_margin=-0.1)  # margin < 0


def test_verdict_requires_p_when_model_present() -> None:
    """A model macro-F1 without a McNemar p-value is a ValidationError."""
    with pytest.raises(ValidationError):
        derive_verdict(0.9, 0.6, None)


# --------------------------------------------------------------------------- #
# CATEGORY-ERROR guard: NO finance return-series metrics in this package       #
# --------------------------------------------------------------------------- #
# Sentiment is a TEXT LABEL, not a tradable signal — there is no return series,
# so Sharpe / Sortino / deflated-Sharpe (DSR) / walk-forward / purge / embargo
# DO NOT APPLY. Computing any of them here would be a category error. This guard
# fails loudly if such machinery is ever introduced into the source package.
_FORBIDDEN_TOKENS = (
    "sharpe",
    "sortino",
    "deflated_sharpe",
    "deflated-sharpe",
    "walk_forward",
    "walk-forward",
    "walkforward",
    "purged",
    "purge",
    "embargo",
    "dsr",
    "probabilistic_sharpe",
)


def _iter_source_modules() -> list[str]:
    """Importable module names under ``finbert_sentiment`` (skip heavy optional ones)."""
    names: list[str] = []
    skip = ("model.", "inference.", ".train", ".export", ".onnx_session", ".plots")
    for info in pkgutil.walk_packages(finbert_sentiment.__path__, prefix="finbert_sentiment."):
        if any(s in info.name for s in skip):
            continue
        names.append(info.name)
    return names


def test_evaluation_defines_no_return_series_metrics() -> None:
    """The evaluation package exposes NO Sharpe/DSR/walk-forward symbols (category error)."""
    import finbert_sentiment.evaluation as ev

    public = [name for name in dir(ev) if not name.startswith("_")]
    lowered = {name.lower() for name in public}
    for token in _FORBIDDEN_TOKENS:
        assert not any(token in name for name in lowered), (
            f"evaluation exposes a forbidden return-series metric matching {token!r}; "
            "sentiment is a text label, not a return series — Sharpe/DSR/walk-forward "
            "are a category error here."
        )


def test_source_has_no_sharpe_or_dsr_symbols() -> None:
    """No source module defines a Sharpe/DSR/walk-forward symbol or docstring claim."""
    for mod_name in _iter_source_modules():
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # pragma: no cover - optional deps absent
            continue
        for sym in dir(mod):
            if sym.startswith("__"):
                continue
            low = sym.lower()
            assert not any(tok in low for tok in _FORBIDDEN_TOKENS), (
                f"{mod_name}.{sym} looks like a return-series metric — category error."
            )


def test_mcnemar_result_is_not_a_finance_signal() -> None:
    """Sanity: McNemar output carries only classifier-comparison fields, no PnL/alpha."""
    fields = set(inspect.signature(McNemarResult).parameters)
    assert fields == {"n01", "n10", "statistic", "p_value", "exact"}
    assert not any(tok in f.lower() for f in fields for tok in ("sharpe", "alpha", "pnl", "return"))
