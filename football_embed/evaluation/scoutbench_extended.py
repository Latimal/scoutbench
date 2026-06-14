#!/usr/bin/env python3
"""ScoutBench Task B -- extended rigor for the paper (all bootstrap, no new data).

KT1 established raw-stat kNN significantly beats the learned v11 model and that the
task carries real signal. This adds the analyses a reviewer will demand before
believing the negative result:

  1. Method ranking with 95% bootstrap CIs on same-position MAP.
  2. Full pairwise significance vs v11 (does EVERY learned rep lose?) and vs random
     (does EVERY raw rep beat noise?), paired bootstrap, per-comparison p.
  3. Per-sub-position breakdown of raw_card vs v11 -- does the model ever win in a
     role, or is the loss uniform? (heterogeneity check)
  4. Market-value-tier robustness -- raw_card vs v11 within low/mid/high value
     terciles, to rule out the negative result being a value/popularity artifact.

Paired bootstrap over the 1363 query players (B=10000). Offline, CPU.

Usage:
    .venv/bin/python3 -m football_embed.evaluation.scoutbench_extended
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
    "pca": "file:data/processed/benchmark/repr_pca.parquet",
    "fbref": "file:data/processed/benchmark/repr_fbref.parquet",
    "card_vaep": "file:data/processed/benchmark/repr_card_vaep.parquet",
    "text_tfidf": "file:data/processed/benchmark/repr_text.parquet",
    "v11": "v11",
    "random": "random",
}


def per_query(emb, qlist, subpos):
    """Per-query same-position MAP + the query gallery index (for grouping)."""
    sims = emb.astype(np.float64) @ emb.astype(np.float64).T
    sp_map, qis = [], []
    for qi, rel in qlist:
        m = (subpos == subpos[qi]); m[qi] = False
        sp_map.append(_qm(sims[qi], qi, rel, m)["map"]); qis.append(qi)
    return np.asarray(sp_map), np.asarray(qis)


def mean_ci(a, B, rng):
    idx = rng.integers(0, len(a), size=(B, len(a)))
    s = a[idx].mean(axis=1)
    return round(float(a.mean()), 4), [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def paired(a, b, B, rng):
    d = a - b
    idx = rng.integers(0, len(d), size=(B, len(d)))
    s = d[idx].mean(axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    p = 2.0 * min(float((s <= 0).mean()), float((s >= 0).mean()))
    return {"diff": round(float(d.mean()), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p": round(min(p, 1.0), 4), "n": int(len(d)), "sig": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser(description="ScoutBench Task B extended analysis")
    ap.add_argument("--gallery", default=DEF_GALLERY); ap.add_argument("--pairs", default=DEF_PAIRS)
    ap.add_argument("--join", default=DEF_JOIN); ap.add_argument("--checkpoint", default=DEF_CKPT)
    ap.add_argument("--bootstrap", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/benchmark/scoutbench_extended.json")
    args = ap.parse_args()

    gallery = pd.read_parquet(args.gallery)
    join = pd.read_parquet(args.join)
    queries, subpos = load_queries(gallery, join, pd.read_parquet(args.pairs))
    qlist = list(queries.items())
    n = len(qlist)
    print(f"n_queries={n}, B={args.bootstrap}")

    # market value per gallery index (for tier robustness)
    n2val = dict(zip(join["player_name"], join["tm_market_value_in_eur"]))
    val_g = np.array([n2val.get(nm, np.nan) for nm in gallery["player_name"].values], dtype=float)

    pq, qref = {}, None
    for name, spec in METHODS.items():
        m, qis = per_query(card_embeddings(spec, gallery, args.checkpoint), qlist, subpos)
        pq[name] = m; qref = qis
        print(f"  per-query SP-MAP computed: {name} (mean={m.mean():.4f})")
    q_sub = subpos[qref]
    q_val = val_g[qref]

    rng = np.random.default_rng(args.seed)
    out = {"n_queries": n, "bootstrap": args.bootstrap}

    # 1. ranking with CIs
    print("\n# 1. Method ranking (same-position MAP, 95% CI)")
    ranking = []
    for nm in sorted(METHODS, key=lambda k: -pq[k].mean()):
        mean, ci = mean_ci(pq[nm], args.bootstrap, np.random.default_rng(args.seed))
        ranking.append({"method": nm, "sp_map": mean, "ci95": ci})
        print(f"  {nm:12s} {mean:.4f}  CI[{ci[0]:.4f},{ci[1]:.4f}]")
    out["ranking"] = ranking

    # 2. pairwise vs v11 and vs random
    print("\n# 2. Every method vs v11 (learned model) and vs random")
    vs_v11, vs_rand = [], []
    for nm in METHODS:
        if nm != "v11":
            r = paired(pq[nm], pq["v11"], args.bootstrap, rng); r["method"] = nm; vs_v11.append(r)
        if nm != "random":
            r2 = paired(pq[nm], pq["random"], args.bootstrap, rng); r2["method"] = nm; vs_rand.append(r2)
    print("  vs v11:    " + ", ".join(f"{r['method']}{'+' if r['diff']>0 else ''}{r['diff']:.4f}(p={r['p']:.3f}{'*' if r['sig'] else ''})" for r in vs_v11))
    print("  vs random: " + ", ".join(f"{r['method']}{'+' if r['diff']>0 else ''}{r['diff']:.4f}(p={r['p']:.3f}{'*' if r['sig'] else ''})" for r in vs_rand))
    out["vs_v11"] = vs_v11; out["vs_random"] = vs_rand

    # 3. per-sub-position: raw_card vs v11
    print("\n# 3. raw_card vs v11 by sub-position (does the model ever win a role?)")
    per_pos = []
    for pos in pd.Series(q_sub).value_counts().index:
        mask = q_sub == pos
        if mask.sum() < 20:
            continue
        r = paired(pq["raw_card"][mask], pq["v11"][mask], args.bootstrap, rng)
        r["sub_position"] = pos; per_pos.append(r)
        print(f"  {pos:20s} n={r['n']:4d}  raw-v11={r['diff']:+.4f} p={r['p']:.3f} {'sig' if r['sig'] else ''}")
    out["per_sub_position_raw_vs_v11"] = per_pos

    # 4. market-value-tier robustness: raw_card vs v11
    print("\n# 4. raw_card vs v11 by market-value tercile (rules out value/popularity artifact)")
    fin = np.isfinite(q_val)
    tiers = []
    if fin.sum() > 60:
        qt = np.nanpercentile(q_val[fin], [33.3, 66.6])
        labels = ["low", "mid", "high"]
        bucket = np.where(q_val <= qt[0], 0, np.where(q_val <= qt[1], 1, 2))
        for ti, lab in enumerate(labels):
            mask = fin & (bucket == ti)
            if mask.sum() < 20:
                continue
            r = paired(pq["raw_card"][mask], pq["v11"][mask], args.bootstrap, rng)
            r["tier"] = lab; r["value_range_eur"] = [None if ti == 0 else float(qt[ti-1]),
                                                      None if ti == 2 else float(qt[ti])]
            tiers.append(r)
            print(f"  {lab:5s} n={r['n']:4d}  raw-v11={r['diff']:+.4f} p={r['p']:.3f} {'sig' if r['sig'] else ''}")
    out["value_tier_raw_vs_v11"] = tiers

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
