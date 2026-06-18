"""Import-purity guard: ``import finbert_sentiment`` pulls in nothing heavy.

The package's headline constraint is that importing it (or any of its
import-pure submodules) triggers NO deep-learning framework, NO inference engine,
and NO network/model-download — those are imported lazily, behind functions. This
test imports the package in a clean subprocess and asserts that none of the heavy
modules ended up in ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Modules that must NOT be imported as a side effect of ``import finbert_sentiment``.
FORBIDDEN_AT_IMPORT = (
    "torch",
    "transformers",
    "onnxruntime",
    "tokenizers",
    "onnx",
    "datasets",
    "plotly",
    "sklearn",
)


@pytest.mark.integration
def test_import_finbert_sentiment_is_pure() -> None:
    """Importing the top-level package loads no torch/transformers/onnx/etc."""
    code = (
        "import sys\n"
        "import finbert_sentiment\n"
        f"forbidden = {FORBIDDEN_AT_IMPORT!r}\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "assert not leaked, f'heavy modules leaked at import: {leaked}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"import-purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


@pytest.mark.integration
def test_public_api_is_importable() -> None:
    """The curated public API names are all present on the package."""
    import finbert_sentiment

    assert finbert_sentiment.__version__
    for name in finbert_sentiment.__all__:
        assert hasattr(finbert_sentiment, name), f"missing public export: {name}"


@pytest.mark.integration
def test_submodules_import_pure() -> None:
    """Importing the pipeline submodules also loads nothing heavy."""
    code = (
        "import sys\n"
        "import finbert_sentiment.data\n"
        "import finbert_sentiment.baselines\n"
        "import finbert_sentiment.inference\n"
        "import finbert_sentiment.evaluation\n"
        "import finbert_sentiment.model  # lazy torch — must still be import-pure\n"
        "import finbert_sentiment.plots\n"
        "import finbert_sentiment.cli\n"
        "import finbert_sentiment.service  # the backend entrypoint — import-pure\n"
        f"forbidden = {FORBIDDEN_AT_IMPORT!r}\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "assert not leaked, f'heavy modules leaked at import: {leaked}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"submodule import-purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
