# Contributing

Thanks for your interest in `finbert-sentiment`. This project uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.

## Dev setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the env and install the project with the LEAN extras + dev tooling.
#    NOTE: the [train] extra (torch + transformers) is heavy and is NOT needed to
#    run the test suite — install it only when fine-tuning/exporting the ONNX model.
uv venv
uv pip install -e ".[data,viz,dev]"
```

Prefix commands with `uv run` to use the env without activating it.

## Quality gates

These are exactly what CI runs (see `.github/workflows/ci.yml`). Run them locally
before opening a pull request:

```bash
uv run ruff check src tests                                                          # lint
uv run mypy src                                                                      # types (strict)
uv run pytest -q -m "not train and not slow" --cov=finbert_sentiment --cov-report=term --cov-fail-under=85
```

- **Lint** (`ruff`) must pass.
- **Types** (`mypy --strict`) is run on every PR. It is currently non-blocking in
  CI while residual strict-mode issues are burned down, but new code should not
  add type errors.
- **Tests** (`pytest`) must pass with **coverage ≥ 85%** (the gate also lives in
  `[tool.coverage.report] fail_under` in `pyproject.toml`). The torch/transformers
  fine-tune path is marked `train`/`slow` and excluded from the default run; the
  served path + eval + baselines run WITHOUT it.

CI runs the full matrix on Python 3.11, 3.12, and 3.13.

## The point of this project

This is an HONEST financial-sentiment classifier. When contributing, preserve the
non-negotiables:

- The headline metric is **macro-F1** (with per-class precision/recall, a
  confusion matrix, and bootstrap CIs) — **never accuracy alone** (the PhraseBank
  neutral class is ~60%, so accuracy flatters the majority predictor).
- Report a **class-prior baseline** and a **lexicon baseline** as the honest
  floor; `beats_lexicon` is a **pure function** of the measured numbers and must
  read `False` unless the model clears a real macro-F1 margin **and** McNemar is
  significant.
- **Sentiment is a text label, not a tradable signal** — make no alpha claim.
- **Walk-forward / purge / Deflated-Sharpe do not apply** (there is no return
  series) — using them would be a category error.
- The split is **grouped by normalized-sentence hash** so near-duplicate
  sentences never straddle train/val/test; the test set is **locked**.
- Never fabricate a transformer macro-F1: report it ONLY if measured in this
  build, else cite the published ProsusAI/finbert figure as "expected, not
  measured here".
- `src/finbert_sentiment/` is **import-pure**: no torch / transformers /
  onnxruntime / network / model-download at import time (heavy imports live
  behind functions).

## Commit hygiene

- Use clear, present-tense commit messages.
- **Do not** add AI-attribution trailers — no `Co-Authored-By: Claude`,
  no "Generated with Claude", no robot-emoji attribution lines. The
  `.github/workflows/no-ai-attribution.yml` guard fails any PR that contains them.

## Pull requests

- Branch off `main`; keep PRs focused.
- Make sure the three quality gates above are green locally.
- Update `CHANGELOG.md` (under `[Unreleased]`) when behaviour changes.
