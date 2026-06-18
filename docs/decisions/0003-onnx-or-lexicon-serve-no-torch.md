# ADR-0003: Serve via ONNX or lexicon — torch/transformers are train-only

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** finbert-sentiment maintainers
- **Related:** [DESIGN.md](../DESIGN.md) (layering & import purity),
  [ADR-0005](0005-no-fabricated-metrics.md) (honest reporting of which model is served)

## Context

The model is a DistilBERT fine-tune built and trained with **torch** and
**transformers** — both heavy (hundreds of MB, slow cold start, large memory
footprint). But the hosted tool runs in a small, shared API container alongside the
other portfolio tools. Importing torch/transformers at request time, or at package
import time, would be unacceptable for a lean inference service. The package must
also stay **import-pure**: `import finbert_sentiment` must pull in no torch, no
transformers, no inference engine, no model download, and no I/O.

There is a second, environment-driven reason to decouple training from serving: in
this build environment torch could not be relied upon (it was not pre-installed and
a TF/Keras attempt hung). The design therefore had to guarantee a **real, served
model even when no transformer artifact exists.**

## Decision

**Training and serving use different engines, split across optional extras, behind
one interface:**

- **Train (`[train]` extra, offline, heavy):** `model/train.py` fine-tunes
  `distilbert-base-uncased` with the transformers `Trainer`; `model/export.py`
  exports the trained graph to a **dynamic-int8 ONNX** artifact plus
  `tokenizer.json`, committed inside the package under
  `src/finbert_sentiment/artifacts/`. torch and transformers are imported
  **lazily**, inside these functions.
- **Serve (`[serve]` extra, container, lean):** the served path imports **only**
  `onnxruntime` + `tokenizers`. The container **never imports torch or
  transformers.**

`inference/predictor.py` exposes a single `predict(texts) -> Prediction` interface
that selects the backend **by which artifact exists**:

1. **transformer-ONNX** when `artifacts/model.int8.onnx` + `tokenizer.json` are
   present (this build), run through `onnxruntime` + `tokenizers`; or
2. the torch-free **lexicon** classifier (`baselines/lexicon.py`) otherwise — a
   real, always-available served model that needs only Python.

A **parity test** asserts the exported ONNX logits match the torch model to
`1e-3` (marked `train`, so it runs only when the `[train]` extra produced a model).
Import purity is enforced: onnxruntime is imported lazily inside the inference
layer on first call, torch/transformers are never reachable from a plain import,
and a subprocess test verifies no import-time side effects.

## Consequences

- **Positive.** The serve container is tiny and fast; no torch/transformers at all.
- **Positive.** The tool *always* serves a real model: transformer-ONNX in this
  build, lexicon as the automatic fallback. There is no "model failed to load"
  dead end.
- **Positive.** `import finbert_sentiment` stays side-effect-free and vendorable
  byte-for-byte into `api/lib/finbert_sentiment/`.
- **Cost.** Two serving backends and a `tokenizers`-vs-`transformers` tokenizer to
  keep consistent, plus an ONNX/torch parity test gated behind `[train]` (marked
  `train` and skipped in the lean run).
- **Risk addressed.** "torch/transformers leaks into the inference container / the
  package is not import-pure / the tool has nothing to serve when the transformer
  is absent" — all three are structurally prevented.
