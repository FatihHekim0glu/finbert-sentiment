# finbert-sentiment

A from-scratch **DistilBERT fine-tune** for 3-way financial-sentence sentiment
(**negative / neutral / positive**) on the
[Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank)
(`sentences_allagree`), served via **ONNX + onnxruntime** and benchmarked
**honestly** against a class-prior baseline and a Loughran-McDonald-style lexicon
baseline.

> **Status:** live. The full pipeline (loader, dedup, leakage-guarded group
> split, baselines, evaluation, unified `predict`, the ONNX serve path, and the
> `run_sentiment` backend entrypoint) is implemented, typed, and tested. **The
> live demo serves the from-scratch DistilBERT fine-tune via ONNX** in this build
> — it was trained, exported to int8 ONNX, and evaluated here (see the measured
> numbers below). The torch-free lexicon baseline remains the always-available
> fallback (the predictor serves it automatically if the ONNX artifact is absent).

## What the live demo actually serves (measured here)

The served model in this build is the **DistilBERT fine-tune** (`distilbert-onnx`),
and these are the **real, measured** numbers on the locked Financial PhraseBank
`sentences_allagree` test fold (453 deduplicated sentences, seed `20260618`),
committed verbatim in `src/finbert_sentiment/artifacts/metrics.json`:

| Model (locked test set) | macro-F1 | note |
| --- | --- | --- |
| **DistilBERT fine-tune (served, ONNX int8)** | **0.960** (95% CI [0.932, 0.982]) | the live model, **measured here** |
| Lexicon baseline | 0.653 (95% CI [0.594, 0.710]) | honest lexical floor |
| Class-prior (majority) floor | 0.254 | trivial floor |

The transformer was fine-tuned from `distilbert-base-uncased` (3 epochs, seed
`20260618`, early-stop on val macro-F1; best val macro-F1 0.933), exported to a
dynamic-int8 ONNX graph, and served through `onnxruntime` + `tokenizers` — **no
torch in the serve path**. Per-class test F1 is negative 0.932 / neutral 0.993 /
positive 0.956. Against the lexicon, **McNemar's test gives p ≈ 1.9e-22**, so
`beats_lexicon` is **`True`** (verdict `model_beats_lexicon`): the gap is both
large (Δmacro-F1 ≈ 0.31) and statistically unambiguous.

Accuracy on the same fold is 0.976 — close to macro-F1 here because the fine-tune
also learns the minority classes well; on the lexicon baseline accuracy (0.762) is
*much higher* than its macro-F1 (0.653) precisely because the neutral class is
~61% of the data, which is exactly why **accuracy alone would be dishonest**.

> The measured 0.960 sits above the ~0.85–0.90 commonly cited for FinBERT-class
> models. That is consistent with the `sentences_allagree` config being the
> easiest, highest-annotator-agreement subset of the PhraseBank; it is a real
> measurement on the locked test fold, not a published figure. The published
> [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) range is retained in
> `metrics.json` only as external context.

## Validation

The numbers above are trusted because the machinery that produces them is checked
against independent references and the split is proven leakage-free. These checks
run in CI (the transformer-only parity check runs offline under the `[train]`
extra):

| Check | What it asserts | Tolerance / result |
| --- | --- | --- |
| Metric parity | macro-F1, per-class precision/recall vs `sklearn.metrics` on identical inputs | `1e-10` |
| Group-split no-overlap | no normalized-sentence hash appears in more than one fold (`assert_no_group_overlap`) | exact (zero overlap), asserted on the shipped split and Hypothesis inputs |
| ONNX parity | exported int8-ONNX logits vs the torch model | `1e-3` (runs only when `[train]` produced a model) |
| Determinism | same seed → same split, baselines, and bootstrap CI | byte-stable |
| Verdict safety | `beats_lexicon` cannot read `True` without margin **and** McNemar significance | truth-table unit-tested |
| Import purity | importing the package triggers no torch/transformers/onnxruntime/network/I/O | subprocess-tested |

The split is a **seeded, class-stratified split grouped by normalized-sentence
hash** (lossy near-duplicates that survive dedup move as a unit), so a sentence in
the locked test fold shares no near-duplicate with training — the dominant
PhraseBank leakage risk. See [`docs/DESIGN.md`](docs/DESIGN.md) and the ADRs in
[`docs/decisions/`](docs/decisions/) for the full rationale.

## Honest headline (read this first)

- **The headline metric is macro-F1**, reported with per-class precision/recall,
  a confusion matrix, and bootstrap confidence intervals — **never accuracy
  alone**. The PhraseBank neutral class is ~61% of the data, so a do-nothing
  majority predictor scores ~0.76 accuracy while contributing nothing; macro-F1
  exposes that (the class-prior floor is macro-F1 0.25).
- **The honest floor is two baselines:** a **class-prior** (majority) classifier
  and a torch-free **lexicon** classifier. The transformer is only interesting to
  the extent it beats both, and `beats_lexicon` is a **pure function** of the
  measured macro-F1 margin and a McNemar significance test.
- **Sentiment is a text label, not a tradable signal.** No alpha is claimed
  anywhere. A live-news demo input, if shown, is labelled "illustrative,
  as-published, no look-ahead guarantee".
- **A transformer macro-F1 is reported ONLY if it was actually measured in this
  build.** Here the DistilBERT fine-tune **was** trained, exported to ONNX, and
  evaluated on the locked test fold, so the reported **0.960** is a real
  measurement (`measured_in_this_build: true` in `metrics.json`). When the
  transformer is *not* trained in a given build, the predictor serves the lexicon
  baseline instead and the transformer figure is cited as **published / expected,
  not measured** — never fabricated.

## Why walk-forward / purge / Deflated-Sharpe do **not** apply

Those tools exist to defend a **return time series** against look-ahead and
multiple-testing bias. This project classifies **sentences** — there is no return
series, no temporal ordering to purge, and no Sharpe ratio to deflate. Applying
them here would be a **category error**. The real leakage risk in the PhraseBank
is *near-duplicate sentences* straddling the train/test boundary, which we defend
against by **grouping the split on a normalized-sentence hash** (and a test
asserts no train/test hash overlap). Survivorship bias is **N/A** for the same
reason: we classify sentences, not a stock universe, so the point-in-time
universe builder is correctly unused.

## Architecture

For the full layering, data flow, invariants, and testing strategy see
[`docs/DESIGN.md`](docs/DESIGN.md); for the contested decisions (group-split,
macro-F1-not-accuracy, ONNX-or-lexicon serve, no-walk-forward, no-fabricated-metrics)
see the ADRs in [`docs/decisions/`](docs/decisions/).

```
load (PhraseBank) -> dedup (sentence-level) -> seeded stratified GROUP split (locked test)
   -> baselines: class-prior + lexicon (torch-free, LIVE fallback)
   -> [train]: DistilBERT fine-tune -> ONNX + int8 export
   -> inference: unified predict() over transformer-ONNX OR lexicon
   -> evaluation: macro-F1 + per-class P/R/F1 + confusion + bootstrap CIs + McNemar
   -> honest verdict: beats_lexicon (pure function)
   -> service.run_sentiment(): the backend entrypoint (live predict + committed eval)
```

The backend calls **one** function. It serves live per-sentence predictions and
reports the offline-measured eval bundle **loaded verbatim** from the committed
`metrics.json` (never recomputed per request, since request texts are unlabeled):

```python
from finbert_sentiment import run_sentiment

result = run_sentiment(
    ["Quarterly profit rose sharply and beat estimates.",
     "Losses widened as demand fell and the stock dropped."],
    model_pref="distilbert",   # falls back to the lexicon if no ONNX artifact
    seed=20260618,
)
result.summary.served_model      # "distilbert-onnx" in this build (ONNX, no torch)
result.summary.eval_macro_f1     # 0.960 (committed locked-test-set value)
result.summary.beats_lexicon     # True (McNemar p ≈ 1.9e-22 vs the lexicon)
result.predictions               # one {text, label, scores} per sentence
result.confusion_figure          # Plotly {data, layout} (test-set confusion)
result.per_class_f1_figure       # Plotly {data, layout} (per-class F1 bar)
```

The serve path imports **only** `onnxruntime` + `tokenizers` (ONNX backend) or
nothing heavy at all (lexicon backend) — **never** torch/transformers. Reproduce
the fine-tune → ONNX export → committed eval bundle with:

```bash
uv pip install -e ".[train,serve]"             # heavy: torch + transformers (offline only)
finbert-sentiment train \
  --output-dir src/finbert_sentiment/artifacts  # fine-tune + export model.int8.onnx + tokenizer.json
finbert-sentiment evaluate --model distilbert \
  --metrics-out src/finbert_sentiment/artifacts/metrics.json   # measured macro-F1 + McNemar
```

(With no ONNX artifact present, `--model lexicon` regenerates the lexicon-only
bundle instead — the torch-free fallback.)

`src/finbert_sentiment/` is **import-pure**: importing it pulls in no torch /
transformers / onnxruntime / network call — those are imported lazily, behind
functions. The lean serve container installs only `[serve]`
(**onnxruntime + tokenizers, never torch/transformers**).

## Install

```bash
uv venv
uv pip install -e ".[data,viz,dev]"     # lean dev install (no torch)
# uv pip install -e ".[train]"           # only to fine-tune / export the ONNX model
```

## Limitations

- The Financial PhraseBank is **annotator-labelled**, **single-language**
  (English), and **neutral-majority**; macro-F1 is the right lens precisely
  because of that class imbalance.
- Sentiment is a **text label**, not alpha — see above.
- The measured **0.960** macro-F1 is on the `sentences_allagree` config — the
  highest-agreement, *easiest* PhraseBank subset. It is **not** a claim about
  noisier text (headlines, filings, social posts) or the lower-agreement
  `sentences_75agree`/`50agree` configs, where macro-F1 would be lower.
- Where the transformer is **not** fine-tuned in a given build, the predictor
  serves the lexicon baseline and the transformer figure is the **published**
  ProsusAI/finbert range, cited as expected — **never** a fabricated measurement.

## References

- Malo, Sinha, Korhonen, Wallenius, Takala (2014), *Good debt or bad debt:
  Detecting semantic orientations in economic texts* — the Financial PhraseBank.
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) — the published
  FinBERT model and its reported figures.
- McNemar (1947), *Note on the sampling error of the difference between
  correlated proportions or percentages* — the paired model-comparison test.

## License

MIT — see [LICENSE](LICENSE).
