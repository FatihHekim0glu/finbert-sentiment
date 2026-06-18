# ADR-0004: Walk-forward / purge / Deflated-Sharpe do not apply (category error)

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** finbert-sentiment maintainers
- **Related:** [ADR-0001](0001-group-split-by-sentence-hash.md) (the leakage guard
  that *does* apply here), [DESIGN.md](../DESIGN.md)

## Context

The other tools in this portfolio (`hrp-portfolio`, `lstm-forecast`,
`volforecast`, …) defend a **return time series** against look-ahead and
multiple-testing bias with walk-forward cross-validation, purge + embargo around
the test window, and a Deflated/Probabilistic Sharpe ratio that deflates the best
result by the number of trials. A reader familiar with that house style might
reasonably ask: *where are walk-forward and DSR in this project?*

They are deliberately **absent**, and that absence is correct.

## Decision

**Walk-forward, purge/embargo, and Deflated-Sharpe are not used here, because this
project has no return series.** It classifies **sentences** into negative /
neutral / positive. There is:

- **no temporal ordering** to walk forward through — the PhraseBank is a *set* of
  sentences, not a calendar of observations;
- **no test window adjacent to a train window** in time, so there is nothing to
  *purge* or *embargo* around;
- **no Sharpe ratio** — there are no returns, no P&L, no trial-grid of strategies —
  so there is nothing to *deflate*.

Applying those tools here would be a **category error**: importing machinery built
for financial return series into a text-classification task where its
preconditions do not hold. The README and `metrics.json` say so explicitly.

The leakage risk that *does* exist — near-duplicate sentences straddling the
train/test boundary — is handled by **grouping the split on a normalized-sentence
hash** with an asserted no-overlap guarantee
([ADR-0001](0001-group-split-by-sentence-hash.md)). The multiplicity risk that
*does* exist (model selection on the validation fold) is handled by **locking the
test set once** and reporting bootstrap CIs on the single locked measurement.

**Survivorship bias is N/A for the same reason:** we classify sentences, not a
stock universe, so the point-in-time universe builder is correctly unused. Any
live-news demo input is labelled "illustrative, as-published, no look-ahead
guarantee".

## Consequences

- **Positive.** The methodology matches the actual problem: a leakage-guarded group
  split and a locked test fold, not borrowed time-series machinery that would only
  add false rigor.
- **Positive.** The honesty is explicit and documented, so a reviewer is not left
  wondering whether walk-forward/DSR were *forgotten* — they were *ruled out*.
- **Cost.** None substantive; the project simply does not claim a temporal
  generalization guarantee it cannot make.
- **Risk addressed.** "Dress a text-label task in return-series validation
  machinery to look more rigorous" — explicitly rejected as a category error.
