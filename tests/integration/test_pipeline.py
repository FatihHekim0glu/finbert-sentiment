"""End-to-end pipeline integration (offline, no network, no torch).

The headline integration guard: ``load (offline sample) -> dedup -> seeded group
split -> lexicon eval`` runs to completion on the ``phrasebank_sample`` fixture
WITHOUT any network call or deep-learning framework, and the resulting split has
no train/test sentence-hash overlap.

Skipped until the data + baseline + evaluation kernels are implemented; the
``test_import_purity`` integration tests already run today.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

_PENDING = "pending data.{dedup,split} + baselines.lexicon + evaluation.metrics"


@pytest.mark.skip(reason=_PENDING)
def test_offline_pipeline_runs_end_to_end() -> None:
    """load -> dedup -> group split -> lexicon eval on the offline sample (no network)."""
    raise AssertionError("wire up once the data/baseline/eval kernels land")
