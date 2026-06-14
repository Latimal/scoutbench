#!/usr/bin/env python3
"""Paired-bootstrap significance test for ScoutBench Task B.

*** DEPRECATED FOR HEADLINE INFERENCE. *** This per-query bootstrap treats the 1363
queries as independent, but they cluster into ~22 position components (effective N ≈ 8),
so it UNDERSTATES variance and inflates significance (~8-10x), and it does not floor p
(can print p=0.0). All reported statistics use scoutbench_blockboot.py instead (block
bootstrap over components + a conservative component-t + Holm + floored p). This script
is retained ONLY to demonstrate the per-query-vs-clustered inflation, not as evidence.

The leaderboard reports same-position MAP of raw_card 0.0498, v11 0.0447,
random 0.0369 over 1363 query players. Mean gaps of ~0.005 over noisy per-query
metrics may not be statistically real. This resamples the 1363 queries with
replacement (paired) to put 95% CIs and bootstrap p-values on the gaps that
anchor the paper's claims:
  - is the learned model (v11) significantly BELOW a raw-stat kNN?  (the headline)
  - does ANY method significantly beat random within-position?      (is there signal?)
  - are the raw baselines distinguishable from each other?          (honesty check)

Usage:
    .venv/bin/python3 -m football_embed.evaluation.scoutbench_significance
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from football_embed.evaluation.scoutbench import (
    DEF_CKPT, DEF_GALLERY, DEF_JOIN, DEF_PAIRS,
    card_embeddings, load_queries, _qm,
)

METHODS = {
    "raw_card": "raw_card",
    "v11": "v11",
    "random": "random",
    "pca": "file:data/processed/benchmark/repr_pca.parquet",
    "fbref": "file:data/processed/benchmark/repr_fbref.parquet",
}

# (A, B, metric, scope) -- positive obs_diff => A better than B
COMPARISONS = [
    ("raw_card", "v11", "map", "sp"),      # headline: does the model lose to raw?
    ("raw_card", "v11", "hit@10", "sp"),
    ("raw_card", "v11", "map", "all"),
    ("fbref", "v11", "map", "sp"),         # best raw baseline vs model
    ("raw_card", "random", "map", "sp"),   # does the task carry signal at all?
    ("raw_card", "random", "map", "all"),
    ("pca", "raw_card", "map", "sp"),      # are raw baselines distinguishable?
]


def per_query_metrics(emb, qlist, subpos):
    """Per-query metric arrays for all-candidate and same-position scopes."""
    sims = emb.astype(np.float64) @ emb.astype(np.float64).T
    keys = ("map", "hit@10", "mrr")
    out = {"all": {k: [] for k in keys}, "sp": {k: [] for k in keys}}
    for qi, rel in qlist:
        a = _qm(sims[qi], qi, rel, None)
        m = (subpos == subpos[qi]); m[qi] = False
        p = _qm(sims[qi], qi, rel, m)
        for k in keys:
            out["all"][k].append(a[k]); out["sp"][k].append(p[k])
    return {scope: {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}
            for scope, d in out.items()}


def boot_diff(a, b, B, rng):
    """Paired bootstrap of mean(a - b) over queries."""
    d = a - b
    n = len(d)
    obs = float(d.mean())
    idx = rng.integers(0, n, size=(B, n))
    samples = d[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    p = 2.0 * min(float((samples <= 0).mean()), float((samples >= 0).mean()))
    return {
        "mean_A": round(float(a.mean()), 5),
        "mean_B": round(float(b.mean()), 5),
        "obs_diff": round(obs, 5),
        "ci95": [round(float(lo), 5), round(float(hi), 5)],
        "p_two_sided": round(min(p, 1.0), 4),
        "significant_at_05": bool(lo > 0 or hi < 0),
    }


def main():
    ap = argparse.ArgumentParser(description="ScoutBench Task B significance bootstrap")
    ap.add_argument("--gallery", default=DEF_GALLERY)
    ap.add_argument("--pairs", default=DEF_PAIRS)
    ap.add_argument("--join", default=DEF_JOIN)
    ap.add_argument("--checkpoint", default=DEF_CKPT)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/benchmark/scoutbench_significance.json")
    args = ap.parse_args()

    gallery = pd.read_parquet(args.gallery)
    queries, subpos = load_queries(gallery, pd.read_parquet(args.join), pd.read_parquet(args.pairs))
    qlist = list(queries.items())
    print(f"n_queries = {len(qlist)}, bootstrap B = {args.bootstrap}\n")

    pq = {}
    for name, spec in METHODS.items():
        emb = card_embeddings(spec, gallery, args.checkpoint)
        pq[name] = per_query_metrics(emb, qlist, subpos)
        print(f"  computed per-query metrics: {name}")

    rng = np.random.default_rng(args.seed)
    results = {"n_queries": len(qlist), "bootstrap": args.bootstrap, "comparisons": []}
    print(f"\n{'comparison':28s}{'metric':10s}{'A':>9}{'B':>9}{'diff':>9}{'95% CI':>20}{'p':>8}  sig")
    for A, B, metric, scope in COMPARISONS:
        r = boot_diff(pq[A][scope][metric], pq[B][scope][metric], args.bootstrap, rng)
        r.update({"A": A, "B": B, "metric": metric, "scope": scope})
        results["comparisons"].append(r)
        ci = f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]"
        flag = "YES" if r["significant_at_05"] else "no"
        print(f"{A+' vs '+B:28s}{scope+' '+metric:10s}{r['mean_A']:>9.4f}{r['mean_B']:>9.4f}"
              f"{r['obs_diff']:>+9.4f}{ci:>20}{r['p_two_sided']:>8.3f}  {flag}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
