#!/usr/bin/env python3
"""ScoutBench Task B -- CORRECTED statistics (block bootstrap + Holm).

Review round 1 (adversarial) found the per-query bootstrap in scoutbench_significance.py
badly understates variance: each replacement pair makes BOTH players queries, and the
query->relevant graph collapses into ~10-22 connected components (≈ sub-positions), so
the 1363 queries are NOT independent. Resampling queries inflates significance
(headline p 0.003 -> ~0.022). This module:

  - assigns each query to a CLUSTER = connected component of the pair graph (union-find
    over replacement pairs in transfermarkt-id space), mapped to gallery indices;
  - block-bootstraps by resampling CLUSTERS with replacement (the correct unit);
  - also reports a cluster-robust paired t on per-cluster mean differences (defensible
    with only ~10-22 clusters, where percentile CIs are shaky);
  - applies Holm-Bonferroni across the core comparison family;
  - floors bootstrap p at (r+1)/(B+1) (never reports p=0.0);
  - reports BOTH all-candidates (primary) and same-position (stratified) scopes.

Usage:
    .venv/bin/python3 -m football_embed.evaluation.scoutbench_blockboot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from football_embed.evaluation.scoutbench import (
    DEF_CKPT, DEF_GALLERY, DEF_JOIN, DEF_PAIRS,
    card_embeddings, load_queries, _qm,
)

METHODS = {
    "raw_card": "raw_card",
    "fbref": "file:data/processed/benchmark/repr_fbref.parquet",
    "pca": "file:data/processed/benchmark/repr_pca.parquet",
    "nmf": "file:data/processed/benchmark/repr_nmf.parquet",   # NMF over card features
    "player_vectors": "file:data/processed/benchmark/repr_player_vectors.parquet",  # FAITHFUL Player-Vectors (Decroos&Davis heatmap-NMF, external published baseline)
    "card_vaep": "file:data/processed/benchmark/repr_card_vaep.parquet",
    "text_tfidf": "file:data/processed/benchmark/repr_text.parquet",
    "v11": "v11",
    "random": "random",
}

# core comparison family for Holm correction (A, B, metric, scope)
CORE = [
    ("fbref", "v11", "map", "sp"),       # robust headline candidate
    ("raw_card", "v11", "map", "sp"),    # original headline
    ("raw_card", "v11", "map", "all"),   # all-candidates sibling
    ("fbref", "v11", "map", "all"),
    ("raw_card", "nmf", "map", "sp"),    # vs card-feature NMF
    ("raw_card", "player_vectors", "map", "sp"),   # vs FAITHFUL Player-Vectors (published learned baseline)
    ("raw_card", "player_vectors", "map", "all"),
    ("raw_card", "random", "map", "sp"),
    ("raw_card", "random", "map", "all"),
]


def _tm2gi(gallery, join):
    name2gi = {n: i for i, n in enumerate(gallery["player_name"].values)}
    tm2gi = {}
    for _, r in join.iterrows():
        gi = name2gi.get(r["player_name"])
        if gi is not None:
            tm2gi.setdefault(int(r["tm_player_id"]), gi)
    return tm2gi


def _components(pairs, tm2gi):
    """Union-find connected components over replacement pairs (tm-id space)."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for x, y in zip(pairs["player_x_tmid"], pairs["player_y_tmid"]):
        gx, gy = tm2gi.get(int(x)), tm2gi.get(int(y))
        if gx is not None and gy is not None:
            union(int(x), int(y))
    return {tm: find(tm) for tm in parent}


def per_query(emb, qlist, subpos):
    sims = emb.astype(np.float64) @ emb.astype(np.float64).T
    sp, al = [], []
    for qi, rel in qlist:
        m = (subpos == subpos[qi]); m[qi] = False
        sp.append(_qm(sims[qi], qi, rel, m)["map"])
        al.append(_qm(sims[qi], qi, rel, None)["map"])
    return {"sp": np.asarray(sp), "all": np.asarray(al)}


def block_boot(a, b, cl, B, rng):
    """Resample clusters with replacement; mean(a-b) over drawn clusters' members."""
    d = a - b
    uc = np.unique(cl)
    members = {c: np.where(cl == c)[0] for c in uc}
    obs = float(d.mean())
    samples = np.empty(B)
    for i in range(B):
        drawn = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([members[c] for c in drawn])
        samples[i] = d[idx].mean()
    lo, hi = np.percentile(samples, [2.5, 97.5])
    r = int(min((samples <= 0).sum(), (samples >= 0).sum()))
    p = 2.0 * (r + 1) / (B + 1)          # floored, never 0
    # cluster-robust paired t on per-cluster mean diffs (few-cluster safe)
    cmeans = np.array([d[members[c]].mean() for c in uc])
    t_p = float(sps.ttest_1samp(cmeans, 0.0).pvalue) if len(cmeans) > 1 else float("nan")
    return {"mean_A": round(float(a.mean()), 4), "mean_B": round(float(b.mean()), 4),
            "diff": round(obs, 5), "ci95_block": [round(float(lo), 5), round(float(hi), 5)],
            "p_block": round(min(p, 1.0), 4), "p_cluster_t": round(t_p, 4),
            "n_clusters": int(len(uc)), "n_queries": int(len(d))}


def holm(results, key="p_block"):
    items = sorted(results, key=lambda r: r[key])
    m = len(items)
    for i, r in enumerate(items):
        thr = 0.05 / (m - i)
        r["holm_thresh"] = round(thr, 5)
        r["holm_sig"] = bool(r[key] <= thr)
    # Holm stops at first failure
    failed = False
    for r in items:
        if failed:
            r["holm_sig"] = False
        elif not r["holm_sig"]:
            failed = True
    return items


def main():
    ap = argparse.ArgumentParser(description="ScoutBench Task B block-bootstrap stats")
    ap.add_argument("--gallery", default=DEF_GALLERY); ap.add_argument("--pairs", default=DEF_PAIRS)
    ap.add_argument("--join", default=DEF_JOIN); ap.add_argument("--checkpoint", default=DEF_CKPT)
    ap.add_argument("--bootstrap", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/benchmark/scoutbench_blockboot.json")
    args = ap.parse_args()

    gallery = pd.read_parquet(args.gallery); join = pd.read_parquet(args.join); pairs = pd.read_parquet(args.pairs)
    queries, subpos = load_queries(gallery, join, pairs)
    qlist = list(queries.items())
    tm2gi = _tm2gi(gallery, join)
    comp = _components(pairs, tm2gi)
    gi2comp = {}
    for tm, root in comp.items():
        gi = tm2gi.get(tm)
        if gi is not None:
            gi2comp[gi] = root
    # cluster id per query (fallback: own index if unmapped)
    cl = np.array([gi2comp.get(qi, -qi - 1) for qi, _ in qlist])
    uc = np.unique(cl)
    print(f"n_queries={len(qlist)}  n_clusters(connected-components)={len(uc)}  "
          f"sizes(top)={sorted([int((cl==c).sum()) for c in uc], reverse=True)[:12]}")

    pq = {nm: per_query(card_embeddings(spec, gallery, args.checkpoint), qlist, subpos)
          for nm, spec in METHODS.items()}
    for nm in METHODS:
        print(f"  {nm:12s} SP-MAP={pq[nm]['sp'].mean():.4f}  ALL-MAP={pq[nm]['all'].mean():.4f}")

    rng = np.random.default_rng(args.seed)
    core = []
    for A, B, metric, scope in CORE:
        r = block_boot(pq[A][scope], pq[B][scope], cl, args.bootstrap, rng)
        r.update({"A": A, "B": B, "metric": metric, "scope": scope})
        core.append(r)
    core = holm(core, key="p_block")

    print(f"\n{'comparison':22s}{'scope':6s}{'diff':>9}{'p_block':>9}{'p_clstrT':>9}{'holm✓':>7}")
    for r in sorted(core, key=lambda r: r["p_block"]):
        print(f"{r['A']+' vs '+r['B']:22s}{r['scope']:6s}{r['diff']:>+9.4f}{r['p_block']:>9.4f}"
              f"{r['p_cluster_t']:>9.4f}{('YES' if r['holm_sig'] else 'no'):>7}")

    out = {"n_queries": len(qlist), "n_clusters": int(len(uc)), "bootstrap": args.bootstrap,
           "method_means": {nm: {"sp_map": round(float(pq[nm]["sp"].mean()), 4),
                                 "all_map": round(float(pq[nm]["all"].mean()), 4)} for nm in METHODS},
           "core_comparisons_holm": core}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
