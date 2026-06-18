# ADR-0002: macro-F1 is the headline metric, never accuracy alone

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** finbert-sentiment maintainers
- **Related:** [ADR-0005](0005-no-fabricated-metrics.md) (the verdict that consumes
  these metrics), [DESIGN.md](../DESIGN.md)

## Context

The Financial PhraseBank `sentences_allagree` class distribution is heavily
imbalanced: the **neutral** class is ~61% of the data, with negative and positive
splitting the rest. On a dataset like this, **accuracy is a misleading headline.**
A do-nothing classifier that predicts "neutral" for every sentence scores ~0.76
*accuracy* while contributing nothing — it never identifies a single positive or
negative sentence, which is the entire point of the task.

The measured numbers make the trap concrete: the lexicon baseline scores **0.762
accuracy** but only **0.653 macro-F1**, and the class-prior (majority) floor scores
high accuracy yet **0.254 macro-F1**. Reporting accuracy alone would let a weak
model look strong simply by leaning on the majority class.

## Decision

The **headline metric is macro-F1** — the unweighted mean of the per-class F1
scores — reported **always** alongside:

- **Per-class precision, recall, and F1** (so the minority classes are visible);
- **The full confusion matrix** (so error structure is visible);
- **Bootstrap confidence intervals** on macro-F1 (so the margin is quantified, not
  asserted).

Accuracy may be reported as *context* but is **never** the headline and never the
basis of the `beats_lexicon` verdict. Macro-F1 weights every class equally, so a
model is rewarded only for actually distinguishing negative and positive from
neutral.

The metric implementations in `evaluation/metrics.py` are **parity-tested against
`sklearn.metrics` to `1e-10`** on the same inputs, so the numbers are not a
bespoke reimplementation that could quietly drift from the reference.

## Consequences

- **Positive.** The headline cannot be gamed by the majority class; the 0.960
  served macro-F1 means the model genuinely separates all three classes (per-class
  F1 0.932 / 0.993 / 0.956).
- **Positive.** The gap between accuracy and macro-F1 on the *lexicon* (0.762 vs
  0.653) is itself reported, demonstrating exactly why accuracy alone would be
  dishonest.
- **Positive.** Bootstrap CIs turn "0.960 beats 0.653" into "[0.932, 0.982] vs
  [0.594, 0.710]" — a quantified, non-overlapping margin.
- **Cost.** macro-F1 is harsher than accuracy and harder to move, so improvements
  on the minority classes matter disproportionately. That is the intended
  incentive.
- **Risk addressed.** "Quote a high accuracy that is really just the neutral
  prior" — impossible when the headline is macro-F1 with per-class breakdowns.
