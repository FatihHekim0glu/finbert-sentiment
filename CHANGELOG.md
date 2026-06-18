# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
