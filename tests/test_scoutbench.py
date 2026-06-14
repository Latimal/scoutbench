"""ScoutBench reproducibility tests.

  1. `_qm` MAP/Hit metrics against a hand-computed example.
  2. the replacement-pair graph collapses into exactly 22 connected components
     (the block-bootstrap resampling unit asserted in the paper).
  3. the block bootstrap is deterministic under a fixed seed.

Tests 2-3 need the real benchmark inputs (gallery + join + pairs). They skip
cleanly if those research-only files are absent (see DATA.md), so the suite still
runs in a checkout without the StatsBomb-derived data.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from football_embed.evaluation.scoutbench import (
    DEF_GALLERY,
    DEF_JOIN,
    DEF_PAIRS,
    _qm,
    load_queries,
)

REPO = Path(__file__).resolve().parent.parent
_INPUTS = [REPO / DEF_GALLERY, REPO / DEF_JOIN, REPO / DEF_PAIRS]
_have_inputs = all(p.exists() for p in _INPUTS)
needs_data = pytest.mark.skipif(not _have_inputs, reason="benchmark inputs absent (research-only; see DATA.md)")


def test_qm_hand_computed():
    """query=0, sims=[10, .9, .3, .8, .5], relevant={2,4}.

    After masking the query, ranks (0-based) are: idx1=0, idx3=1, idx4=2, idx2=3.
    Relevant ranks sorted = [2 (idx4), 3 (idx2)], first relevant rank = 2.
      hit@1 = 0 (rank 2 not < 1); hit@5 = hit@10 = 1
      mrr   = 1 / (2 + 1) = 1/3
      AP    = mean(1/(2+1), 2/(3+1)) = mean(1/3, 1/2) = 5/12
      recall@10 = both relevant within top-10 -> 1.0
    """
    sims = np.array([10.0, 0.9, 0.3, 0.8, 0.5])
    r = _qm(sims, qi=0, rel={2, 4}, mask=None)
    assert r["hit@1"] == 0.0
    assert r["hit@5"] == 1.0
    assert r["hit@10"] == 1.0
    assert math.isclose(r["mrr"], 1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(r["map"], 5.0 / 12.0, rel_tol=1e-9)
    assert math.isclose(r["recall@10"], 1.0, rel_tol=1e-9)


def test_qm_perfect_top_ranks():
    """Both relevant at ranks 0 and 1 -> hit@1=1, mrr=1, AP=1."""
    # sims so that the two relevant items are the two most similar (after masking q=0)
    sims = np.array([10.0, 0.99, 0.98, 0.10, 0.05])
    r = _qm(sims, qi=0, rel={1, 2}, mask=None)
    assert r["hit@1"] == 1.0
    assert math.isclose(r["mrr"], 1.0, rel_tol=1e-9)
    # AP = mean(1/1, 2/2) = 1.0
    assert math.isclose(r["map"], 1.0, rel_tol=1e-9)


@needs_data
def test_connected_components_equals_22():
    """The replacement-pair graph collapses into exactly 22 components (the
    block-bootstrap resampling unit; paper Sec. 3)."""
    import pandas as pd

    from football_embed.evaluation.scoutbench_blockboot import _components, _tm2gi

    gallery = pd.read_parquet(REPO / DEF_GALLERY)
    join = pd.read_parquet(REPO / DEF_JOIN)
    pairs = pd.read_parquet(REPO / DEF_PAIRS)

    queries, _ = load_queries(gallery, join, pairs)
    tm2gi = _tm2gi(gallery, join)
    comp = _components(pairs, tm2gi)
    gi2comp = {tm2gi[tm]: root for tm, root in comp.items() if tm in tm2gi}
    cl = np.array([gi2comp.get(qi, -qi - 1) for qi, _ in queries.items()])

    assert len(queries) == 1363
    assert len(np.unique(cl)) == 22


@needs_data
def test_block_bootstrap_deterministic():
    """Same seed -> identical block-bootstrap p-value and CI (reproducibility)."""
    from football_embed.evaluation.scoutbench_blockboot import block_boot

    rng_data = np.random.default_rng(2024)
    a = rng_data.standard_normal(300)
    b = rng_data.standard_normal(300)
    cl = rng_data.integers(0, 10, size=300)

    r1 = block_boot(a, b, cl, B=500, rng=np.random.default_rng(99))
    r2 = block_boot(a, b, cl, B=500, rng=np.random.default_rng(99))
    assert r1 == r2
    # different seed should (almost surely) move the bootstrap CI
    r3 = block_boot(a, b, cl, B=500, rng=np.random.default_rng(100))
    assert r3["ci95_block"] != r1["ci95_block"]
