# ADR-0005: Never fabricate a transformer metric — report measured or cited-as-published

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** finbert-sentiment maintainers
- **Related:** [ADR-0002](0002-macro-f1-not-accuracy.md) (what the metric is),
  [ADR-0003](0003-onnx-or-lexicon-serve-no-torch.md) (dual serve path that makes a
  lexicon-only build legitimate)

## Context

The headline temptation in a "FinBERT" project is to quote a transformer macro-F1
of ~0.85–0.90 — the commonly cited FinBERT-class range — *whether or not a model
was actually trained and evaluated in this build*. Because this environment's torch
availability was uncertain ([ADR-0003](0003-onnx-or-lexicon-serve-no-torch.md)),
there was a live risk of shipping a published figure dressed up as a local
measurement. That would be fabrication.

## Decision

**A transformer macro-F1 is reported as a measured result ONLY if it was actually
measured in this build. Otherwise it is cited as published/expected, never
fabricated.** This is enforced in three places:

1. **A provenance flag in the data.** `metrics.json` carries
   `transformer_published_macro_f1.measured_in_this_build` (a boolean) and a
   `served_model` field. In this build the DistilBERT fine-tune *was* trained,
   exported to int8 ONNX, and evaluated on the locked test fold, so
   `measured_in_this_build: true`, `served_model: "distilbert-onnx"`, and the
   reported **0.960** is a real measurement. When no transformer is trained, the
   flag is `false`, `served_model` is `"lexicon"`, and the ~0.85–0.90 range is
   retained **only** as external context (the published ProsusAI/finbert figure),
   explicitly marked "expected, not measured here".

2. **A pure verdict that cannot over-claim.** `evaluation/verdict.py::derive_verdict`
   maps `(model_macro_f1, lexicon_macro_f1, mcnemar_p_value)` to the `Verdict`
   enum. It returns `MODEL_BEATS_LEXICON` (`beats_lexicon = True`) **only if** the
   model's macro-F1 clears the lexicon's by `min_margin` **and** McNemar rejects
   equal error rates (p < alpha). With no transformer (`model_macro_f1 is None`) it
   returns `LEXICON_ONLY` and `beats_lexicon = None` — never `True`. The truth
   table is unit-tested.

3. **The README states which model the live demo serves**, with the measured
   number and its CI when measured, or the lexicon's real macro-F1 plus the
   cited-as-published transformer range when not. The two cases are never blurred.

In this build the result is unambiguous: served macro-F1 **0.960** (95% CI
[0.932, 0.982]) vs lexicon **0.653**, McNemar **p ≈ 1.9e-22**, so
`beats_lexicon = true` (`verdict: model_beats_lexicon`) — and every one of those
numbers is measured on the locked test fold.

## Consequences

- **Positive.** The headline is *mechanically* honest: the verdict is derived from
  the inference, not narrated, and cannot read `True` while the evidence is absent.
- **Positive.** A lexicon-only build is a first-class, legitimate outcome — it
  serves a real model and reports a real macro-F1, with the transformer figure
  clearly external. No build is ever pressured into fabrication.
- **Positive.** The `measured_in_this_build` flag makes provenance machine-readable,
  so the frontend and backend cannot accidentally present a published figure as
  measured.
- **Cost.** The reporting code must branch on provenance and carry both the
  measured value and the published context, rather than printing one number.
- **Risk addressed.** "Quote a published FinBERT F1 as if it were measured here" —
  structurally impossible: it would require flipping `measured_in_this_build` and
  the verdict logic, both of which are tied to the actual evaluation.
