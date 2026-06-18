# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Shipped the measured DistilBERT fine-tune.** Fine-tuned
  `distilbert-base-uncased` 3-way on the Financial PhraseBank `sentences_allagree`
  fold (3 epochs, seed `20260618`, early-stop on val macro-F1; best val macro-F1
  0.933), exported to a dynamic-int8 ONNX graph + `tokenizer.json`, and committed
  both under `src/finbert_sentiment/artifacts/`. The live demo now serves
  `distilbert-onnx` through `onnxruntime` + `tokenizers` (no torch in the serve
  path), falling back to the lexicon automatically when no ONNX artifact exists.
- **Measured evaluation on the locked test set** (453 deduplicated sentences):
  served macro-F1 **0.960** (95% CI [0.932, 0.982]), per-class F1
  0.932 / 0.993 / 0.956, vs lexicon 0.653 and class-prior 0.254. McNemar vs the
  lexicon p ≈ 1.9e-22 → `beats_lexicon: true` (`measured_in_this_build: true`).
  `metrics.json` regenerated accordingly.
- Activated the `[train]`-gated tests (the end-to-end fine-tune + export and the
  ONNX-vs-torch logit parity check, max abs logit diff < 1e-3); made the
  import-purity assertions subprocess-isolated so they are order-independent.
- Forced the legacy TorchScript ONNX exporter (`dynamo=False`) for a stable
  export signature across torch versions; added `accelerate` to the `[train]`
  extra (required by the transformers Trainer).

- Project scaffold: import-pure, strictly-typed src-layout package
  `finbert_sentiment` with `py.typed`.
- Shared infrastructure (validation, constants, typing, exceptions, run
  manifest, seeded RNG) carried over from the house template.
- Typed module stubs for the full pipeline: `data` (PhraseBank load, sentence
  dedup, seeded stratified group split), `baselines` (class-prior + lexicon),
  `model` (DistilBERT fine-tune + ONNX/int8 export, `[train]`-only), `inference`
  (lazy ONNX session + unified predictor over transformer-ONNX OR lexicon),
  `evaluation` (macro-F1 + per-class P/R/F1 + confusion + bootstrap CIs,
  McNemar, honest `beats_lexicon` verdict), `plots`, and the Typer `cli`.
- Optional-dependency split: `[data]`, `[serve]` (onnxruntime + tokenizers,
  no torch/transformers), `[train]` (torch/transformers/evaluate/onnx),
  `[viz]`, `[dev]`.
- CI workflows: lint/type/test matrix (lean extras, excludes the `[train]`
  path) and the no-AI-attribution commit guard.
- Partitioned test suite (`unit`/`parity`/`property`/`regression`/`integration`)
  with seeded, offline fixtures (no network, no torch).
