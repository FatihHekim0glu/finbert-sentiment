# ADR-0001: Split is grouped by normalized-sentence hash (the leakage guard)

- **Status:** Accepted
- **Date:** 2026-06-18
- **Deciders:** finbert-sentiment maintainers
- **Related:** [DESIGN.md](../DESIGN.md) (data flow & invariants),
  [ADR-0004](0004-no-walkforward-category-error.md) (why the *temporal* leakage
  defenses do not apply here)

## Context

The Financial PhraseBank is a set of ~2,264 short financial sentences (the
`sentences_allagree` config), human-annotated negative / neutral / positive. It is
**not** a time series — there is no temporal ordering, no return to predict, and no
look-ahead in the calendar sense.

But it has a real, specific leakage risk: **near-duplicate sentences.** The corpus
was assembled from financial news, and the same or trivially-reworded sentence
("Operating profit rose to EUR 10 mn from EUR 8 mn") recurs with only an entity or
a number changed. A naive row-level random split places one copy in `train` and a
near-identical copy in `test`. The model (or even the lexicon) then "predicts" a
test sentence it has effectively already seen, and the reported macro-F1 is
optimistically biased. This is the dominant way a PhraseBank benchmark
accidentally inflates its headline.

## Decision

The split is **seeded, class-stratified, and grouped by a normalized-sentence
hash** (`data/split.py::stratified_group_split`):

1. **Dedup first.** `data/dedup.py` removes exact and whitespace-normalized
   duplicate sentences before splitting.
2. **Group key = normalized-sentence hash.** Every sentence is hashed after
   normalization; all rows sharing a hash form one *group* and are assigned to a
   **single fold** as a unit. Near-duplicates that survive dedup therefore cannot
   straddle `train` and `test`.
3. **Stratified.** Assignment is stratified by class so each fold approximately
   preserves the (neutral-heavy) prior, and a post-condition asserts every class
   is present in every fold.
4. **Seeded and locked.** The same `(dataset, seed)` always yields the same split;
   the **test set is locked once** and never re-shuffled.
5. **An executable guarantee.** `assert_no_group_overlap(split, group_hashes)`
   raises if any group hash appears in more than one fold. A property test runs it
   on the shipped split and on Hypothesis-generated inputs, so the no-leakage claim
   is *asserted*, not assumed.

## Consequences

- **Positive.** The headline macro-F1 (0.960 served, 0.653 lexicon) is measured on
  a test fold that shares no sentence with training — the number is honest.
- **Positive.** The leakage guard is a one-line, fast assertion that doubles as a
  regression tripwire: if a future refactor reintroduces row-level splitting, the
  property test fails.
- **Positive.** Stratification keeps the per-class prior stable across folds, which
  is what makes macro-F1 comparable between the baselines and the transformer.
- **Cost.** Group-grouping makes exact fraction targets (e.g. test = 20%) only
  approximate, because whole groups move together. We accept slightly uneven fold
  sizes in exchange for zero near-duplicate leakage.
- **Risk addressed.** "Near-duplicate sentence leaks from train into test and
  inflates macro-F1" — structurally prevented by the group key plus the
  no-overlap assertion.
