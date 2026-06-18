# Design

This document explains how `finbert-sentiment` is put together: the layering, the
data flow from raw PhraseBank sentences to a served prediction, the invariants the
compute core guarantees, and the testing strategy that keeps the honest headline
honest. For *why* individual contested choices were made, see the numbered ADRs in
[`docs/decisions/`](decisions/).

## Goals and non-goals

**Goals**

- A pure, typed (`mypy --strict`, `py.typed`), side-effect-free compute core that
  can be audited line by line and vendored into a backend without dragging torch,
  transformers, a UI, or a network dependency along.
- A faithful, from-scratch 3-way financial-sentence sentiment pipeline
  (load → dedup → leakage-guarded group split → baselines → fine-tune → ONNX →
  evaluate), parity-tested against `sklearn.metrics`.
- An **honest** measurement: macro-F1 (never accuracy alone) with per-class
  precision/recall, a confusion matrix, and bootstrap CIs, against two explicit
  baselines (class-prior and lexicon) that form the floor.
- A statistically defensible verdict (`beats_lexicon`) that is *mechanically*
  prevented from over-claiming, plus a dual serve path that always works.

**Non-goals**

- A tradable signal. **Sentiment is a text label, not alpha** — no alpha is
  claimed anywhere ([ADR-0002](decisions/0002-macro-f1-not-accuracy.md),
  [ADR-0004](decisions/0004-no-walkforward-category-error.md)).
- A live trading or backtesting system. There is no return series here, so
  walk-forward / purge / Deflated-Sharpe **do not apply** — using them would be a
  category error ([ADR-0004](decisions/0004-no-walkforward-category-error.md)).
- A general NLP toolkit. Everything exists to classify PhraseBank-style financial
  sentences into negative / neutral / positive.

## Layered architecture

The package is strictly layered; each layer imports only from the ones below it.
`src/finbert_sentiment/` has **zero import-time side effects** — no torch,
transformers, onnxruntime, model download, or I/O at import — guarded by a
subprocess import-purity test. The heavy engines are imported **lazily**, behind
functions, and split across optional extras
([ADR-0003](decisions/0003-onnx-or-lexicon-serve-no-torch.md)).

```
              cli.py (Typer)          plots.py (lazy Plotly)         service.run_sentiment
                   |                        |                              |
   ┌───────────────┴────────────────────────┴──────────────────────────────┘
   │                          evaluation/
   │        metrics.py · mcnemar.py · verdict.py
   │   (macro-F1 / per-class P-R-F1 / confusion + bootstrap CIs · McNemar · pure verdict)
   ├────────────────────────────────────────────────────────────────────────
   │                          inference/
   │              predictor.py · onnx_session.py
   │   (unified predict() over transformer-ONNX OR lexicon, by available artifact)
   ├────────────────────────────────────────────────────────────────────────
   │   model/  (`[train]` only, lazy torch)        baselines/
   │   train.py · export.py                  class_prior.py · lexicon.py
   │   (DistilBERT fine-tune · ONNX+int8)     (majority floor · LM-style lexical floor / LIVE fallback)
   ├────────────────────────────────────────────────────────────────────────
   │                          data/
   │          load.py · dedup.py · split.py
   │   (PhraseBank loader · sentence dedup · seeded stratified group split — locked test)
   ├────────────────────────────────────────────────────────────────────────
   │   foundation (no internal deps)
   │   _validation · _constants · _typing · _exceptions · _manifest · _rng · py.typed
   └────────────────────────────────────────────────────────────────────────
```

### Foundation (`_*.py`)

- `_constants.py` — `N_CLASSES = 3`, the canonical `negative/neutral/positive`
  label order, the default seed, and the default split fractions; one source of
  truth.
- `_validation.py` — input guards (non-empty text batches, length caps, shape and
  finiteness checks).
- `_typing.py` / `_exceptions.py` — shared aliases and the exception taxonomy
  (`ValidationError`, `InsufficientDataError`, …).
- `_manifest.py` / `_rng.py` — `RunManifest` plus seeded PCG64 substreams. The
  manifest makes a whole run reproducible; the same seed yields the same split,
  the same baseline predictions, and the same bootstrap CI.

### `data/`

`load.py` loads the Financial PhraseBank (`sentences_allagree`) via the HF
`datasets` loader, with an offline-cached fixture for tests (no network).
`dedup.py` removes exact and normalized-whitespace duplicate sentences.
`split.py` is the project's central leakage guard: a seeded, class-stratified
split that is **grouped by normalized-sentence hash** so near-duplicate sentences
never straddle two folds ([ADR-0001](decisions/0001-group-split-by-sentence-hash.md)).
The test set produced here is treated as **locked**, and `assert_no_group_overlap`
is the executable form of the no-leakage guarantee.

### `baselines/`

The honest floor. `class_prior.py` is a majority/prior classifier — the trivial
floor that exposes why accuracy alone is misleading on a neutral-heavy dataset.
`lexicon.py` is a Loughran-McDonald-style 3-way classifier that counts
domain-specific positive/negative cue words and decides by signed count: it is
both a real lexical floor **and** the always-available, torch-free served model
the tool falls back to when no ONNX artifact is present
([ADR-0003](decisions/0003-onnx-or-lexicon-serve-no-torch.md)).

### `model/` (`[train]` only)

`train.py` fine-tunes `distilbert-base-uncased` 3-way via the transformers
`Trainer` (seeded, early-stop on validation macro-F1) — torch and transformers are
imported **lazily**, inside the function, so the package stays import-pure.
`export.py` exports the trained graph to a dynamic-int8 **ONNX** artifact plus
`tokenizer.json`, committed under `src/finbert_sentiment/artifacts/`. This whole
layer is `[train]`-gated and never reachable from a plain import or the serve
container.

### `inference/`

`onnx_session.py` lazily builds an `onnxruntime` session over the committed ONNX
artifact and runs the matching `tokenizers` tokenizer — **never** torch.
`predictor.py` is the unified `predict(texts) -> Prediction` entrypoint: it serves
the transformer-ONNX backend when the artifact exists, and the lexicon backend
otherwise, behind a single interface
([ADR-0003](decisions/0003-onnx-or-lexicon-serve-no-torch.md)).

### `evaluation/`

`metrics.py` computes macro-F1, per-class precision/recall/F1, the confusion
matrix, and bootstrap confidence intervals — parity-tested against
`sklearn.metrics` to `1e-10` ([ADR-0002](decisions/0002-macro-f1-not-accuracy.md)).
`mcnemar.py` runs the paired McNemar model-vs-lexicon significance test.
`verdict.py` is a **pure function** mapping
`(model_macro_f1, lexicon_macro_f1, mcnemar_p_value)` to a fixed `Verdict` enum and
a nullable `beats_lexicon` boolean — it cannot read `True` without both a real
margin and McNemar significance ([ADR-0005](decisions/0005-no-fabricated-metrics.md)).

### Delivery seams

`plots.py` builds the confusion-matrix heatmap and per-class-F1 bar with a
**lazy** Plotly import. `cli.py` is a Typer app (`train` / `evaluate` / `predict`).
`service.run_sentiment()` is the single backend entrypoint: it serves live
per-sentence predictions and reports the offline-measured eval bundle **loaded
verbatim** from the committed `metrics.json`.

## Data flow: from a PhraseBank sentence to a served prediction

```
financial_phrasebank (sentences_allagree)
        │  load.py
        ▼
   sentence-level dedup (dedup.py)
        │
        ▼  split.py: seeded, stratified, GROUPED by normalized-sentence hash
   train / val / LOCKED test   ── assert_no_group_overlap (leakage tripwire)
        │
        ├─► baselines (TRAIN-fit only): class-prior, lexicon   ─┐
        │                                                       │
        ├─► [train]: DistilBERT fine-tune (val early-stop)      │  evaluated on the
        │       ─► ONNX + int8 export ─► artifacts/             │  LOCKED test fold
        │                                                       │
        ▼                                                       ▼
   inference.predict(texts)  ◄── transformer-ONNX if artifact, else lexicon
        │                                                       │
        ▼                                                       ▼
   live per-sentence {label, scores}        evaluation: macro-F1 + per-class P/R/F1
                                            + confusion + bootstrap CIs + McNemar
                                                                │
                                                                ▼
                                            verdict.py ──► beats_lexicon (pure-derived)
                                                                │
                                                                ▼
                                            committed metrics.json (served verbatim)
```

The label encoder and any class statistics are fit on **TRAIN only**; no val/test
text touches fitting. The tokenizer is pretrained/frozen, so it is leakage-free by
construction. Request texts at serve time are **unlabeled**, so the eval bundle is
the offline locked-test measurement — never recomputed per request.

## Key invariants

The compute core guarantees, and tests enforce:

1. **No group leakage.** No normalized-sentence hash appears in more than one fold;
   `assert_no_group_overlap` is asserted on the shipped split and on property-based
   inputs ([ADR-0001](decisions/0001-group-split-by-sentence-hash.md)).
2. **Train-only fitting.** The label encoder / class-prior is computed on TRAIN
   rows only; val/test text never participates in fitting.
3. **Stratification holds.** Every class is present in every fold; the per-class
   prior is approximately preserved.
4. **Metric parity.** macro-F1, precision, and recall match `sklearn.metrics` to
   `1e-10` on the same inputs ([ADR-0002](decisions/0002-macro-f1-not-accuracy.md)).
5. **Serve-path purity.** The served path imports only `onnxruntime` + `tokenizers`
   (ONNX backend) or nothing heavy at all (lexicon backend) — **never**
   torch/transformers ([ADR-0003](decisions/0003-onnx-or-lexicon-serve-no-torch.md)).
6. **Determinism.** Same seed → same split, same baseline predictions, same
   bootstrap CI; predictions are deterministic under a fixed artifact.
7. **Lexicon sign-correctness.** Clearly positive / negative sentences classify
   with the right sign; on a shuffled-label control the class-prior baseline is the
   floor.
8. **Verdict safety.** `beats_lexicon` cannot read `True` unless the macro-F1
   margin clears `min_margin` **and** McNemar rejects equal error rates
   (truth-table unit-tested); it is `None` on the lexicon-only build
   ([ADR-0005](decisions/0005-no-fabricated-metrics.md)).
9. **Import purity.** Importing any `finbert_sentiment` module triggers no torch /
   transformers / onnxruntime import, no model download, no network, no I/O
   (subprocess-tested).

## Testing strategy

Tests are partitioned by intent under `tests/` (markers in `pyproject.toml`):

- **`unit/`** — isolated kernels: the lexicon decision, class-prior, the metric
  functions, the verdict truth table, the CLI, the lazy plot builders.
- **`property/`** (Hypothesis) — the invariants above: no train/test sentence-hash
  overlap, stratification, deterministic predictions, lexicon sign-correctness.
- **`parity/`** — golden checks against independent references: macro-F1 /
  precision / recall vs `sklearn.metrics` to `1e-10`; and (only when the `[train]`
  extra produced a model) ONNX logits vs the torch model to `1e-3`.
- **`regression/`** — locked goldens: the lexicon confusion matrix on the cached
  PhraseBank sample, and `class-prior == prior`.
- **`integration/`** — end-to-end `load → dedup → split → (lexicon) eval` on the
  cached sample with **no network and no torch**, plus the import-purity subprocess
  test.

Seeded fixtures in `conftest.py` (`phrasebank_sample`, `shuffled_label_control`)
give every layer deterministic, offline inputs. Torch/transformers-dependent tests
are marked `slow` / `train` and skipped in the lean CI run, so the served path and
the evaluation both run **without** them.

## Backend & frontend boundary

The compute core is decoupled from delivery. The backend vendors
`finbert-sentiment[serve]` (onnxruntime + tokenizers — **not** torch/transformers)
under `api/lib/finbert_sentiment/`, byte-for-byte including the committed
`artifacts/`, and exposes `POST /tools/finbert-sentiment/run`. A module-level
`_MODEL = None` lazy-loads the ONNX model if present, else the lexicon. The
response carries summary scalars (`served_model`, `eval_macro_f1`,
`lexicon_macro_f1`, `class_prior_macro_f1`, `beats_lexicon`), per-sentence
predictions, and two Plotly `{data, layout}` figures (test-set confusion,
per-class F1). `eval_macro_f1` is the offline-measured value loaded from the
committed `metrics.json`, **not** recomputed per request. The frontend renders the
figures and surfaces the honest caption — *sentiment is a text label, not a
tradable signal* — alongside the served model and macro-F1.
