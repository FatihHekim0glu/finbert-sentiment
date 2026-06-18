# finbert-sentiment

A from-scratch **DistilBERT fine-tune** for 3-way financial-sentence sentiment
(**negative / neutral / positive**) on the
[Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank)
(`sentences_allagree`), served via **ONNX + onnxruntime** and benchmarked
**honestly** against a class-prior baseline and a Loughran-McDonald-style lexicon
baseline.

> **Status:** scaffold. The package, public API, optional-dependency split, CI,
> and the partitioned test suite are in place; the compute kernels are typed
> stubs being filled in by the pipeline modules.

## Honest headline (read this first)

- **The headline metric is macro-F1**, reported with per-class precision/recall,
  a confusion matrix, and bootstrap confidence intervals — **never accuracy
  alone**. The PhraseBank neutral class is ~60% of the data, so a do-nothing
  majority predictor scores ~0.60 accuracy while contributing nothing; macro-F1
  exposes that.
- **The honest floor is two baselines:** a **class-prior** (majority) classifier
  and a torch-free **lexicon** classifier. The transformer is only interesting to
  the extent it beats both, and `beats_lexicon` is a **pure function** of the
  measured macro-F1 margin and a McNemar significance test.
- **Sentiment is a text label, not a tradable signal.** No alpha is claimed
  anywhere. A live-news demo input, if shown, is labelled "illustrative,
  as-published, no look-ahead guarantee".
- **A transformer macro-F1 is reported ONLY if it was actually measured in this
  build.** A from-scratch DistilBERT fine-tune is expected to reach macro-F1
  ~0.85–0.90 on the `allagree` split (on par with the published
  [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)); where the
  transformer was not trained here, that figure is cited as **published /
  expected, not measured in this build**, and the live demo serves the **lexicon
  baseline** (whose real macro-F1 is the one reported).

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

```
load (PhraseBank) -> dedup (sentence-level) -> seeded stratified GROUP split (locked test)
   -> baselines: class-prior + lexicon (torch-free, LIVE fallback)
   -> [train]: DistilBERT fine-tune -> ONNX + int8 export
   -> inference: unified predict() over transformer-ONNX OR lexicon
   -> evaluation: macro-F1 + per-class P/R/F1 + confusion + bootstrap CIs + McNemar
   -> honest verdict: beats_lexicon (pure function)
```

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
- Where the transformer was not fine-tuned in this build, its macro-F1 is the
  **published** ProsusAI/finbert figure, **not** a measurement from this repo.

## References

- Malo, Sinha, Korhonen, Wallenius, Takala (2014), *Good debt or bad debt:
  Detecting semantic orientations in economic texts* — the Financial PhraseBank.
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) — the published
  FinBERT model and its reported figures.
- McNemar (1947), *Note on the sampling error of the difference between
  correlated proportions or percentages* — the paired model-comparison test.

## License

MIT — see [LICENSE](LICENSE).
