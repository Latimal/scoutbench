#!/usr/bin/env python3
"""ScoutBench label validity -- do realized replacements carry SIMILARITY signal?

The strongest reviewer attack: a replacement transfer is driven by budget / availability
/ position need, not stylistic similarity, so the silver label may measure transfer-market
mechanics rather than similarity. Direct test: is the realized replacement Y MORE similar
(raw-card cosine) to the lost player X than a RANDOM same-sub-position player is?

For each same-position replacement pair (X,Y) we compute the PERCENTILE of cosine(X,Y)
among {cosine(X,Z): Z same sub-position}. Pure position+noise => mean percentile ~50%.
Genuine similarity signal => mean percentile >> 50%. We report the mean percentile, the
fraction above median, and replacement vs random-same-position mean cosine, with a
sub-position-clustered bootstrap CI.

Offline, CPU.

Usage:
    .venv/bin/python3 -m football_embed.evaluation.scoutbench_label_validity
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from football_embed.evaluation.scoutbench import (
    DEF_GALLERY, DEF_JOIN, DEF_PAIRS, _card_matrix, load_queries,
)
from football_embed.evaluation.scoutbench_blockboot import _tm2gi


def main():
    ap = argparse.ArgumentParser(description="ScoutBench silver-label validity")
    ap.add_argument("--gallery", default=DEF_GALLERY); ap.add_argument("--pairs", default=DEF_PAIRS)
    ap.add_argument("--join", default=DEF_JOIN); ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default="data/processed/benchmark/scoutbench_label_validity.json")
    args = ap.parse_args()

    gallery = pd.read_parquet(args.gallery); join = pd.read_parquet(args.join); pairs = pd.read_parquet(args.pairs)
    _, subpos = load_queries(gallery, join, pairs)
    tm2gi = _tm2gi(gallery, join)

    cards = np.clip(_card_matrix(gallery), -50.0, 50.0).astype(np.float64)  # _card_matrix already sanitizes; float64 for stable norm
    n = np.linalg.norm(cards, axis=1, keepdims=True); n[n < 1e-9] = 1.0      # floor tiny norms (near-zero rows) to avoid inf division
    raw = np.nan_to_num(cards / n, nan=0.0, posinf=0.0, neginf=0.0)

    # same-position candidate index lists
    pos_idx = defaultdict(list)
    for i, p in enumerate(subpos):
        if p is not None and p != "?":
            pos_idx[p].append(i)
    pos_idx = {p: np.array(v) for p, v in pos_idx.items()}

    pct, abovemed, repl_cos, rand_cos, clusters, seen = [], [], [], [], [], set()
    rng = np.random.default_rng(0)
    for x, y in zip(pairs["player_x_tmid"], pairs["player_y_tmid"]):
        gx, gy = tm2gi.get(int(x)), tm2gi.get(int(y))
        if gx is None or gy is None or gx == gy:
            continue
        sp = subpos[gx]
        if sp is None or sp == "?" or subpos[gy] != sp:
            continue
        cand = pos_idx[sp]; cand = cand[cand != gx]
        if len(cand) < 5:
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):  # raw verified finite/bounded; silence spurious BLAS matmul warning
            sims = raw[gx] @ raw[cand].T
        cyx = float(raw[gx] @ raw[gy])
        p = float((sims < cyx).mean())  # percentile of the replacement's similarity
        pct.append(p); abovemed.append(1.0 if p > 0.5 else 0.0)
        repl_cos.append(cyx); rand_cos.append(float(sims.mean()))
        clusters.append(sp)
        seen.add((gx, gy))

    pct = np.array(pct); clusters = np.array(clusters, dtype=object)

    def clustered_ci(a, cl, B):
        uc = np.unique(cl); members = {c: np.where(cl == c)[0] for c in uc}
        s = np.empty(B)
        for i in range(B):
            drawn = rng.choice(uc, size=len(uc), replace=True)
            idx = np.concatenate([members[c] for c in drawn])
            s[i] = a[idx].mean()
        return round(float(a.mean()), 4), [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]

    mean_pct, ci_pct = clustered_ci(pct, clusters, args.bootstrap)
    out = {
        "n_pairs_scored": int(len(pct)), "n_subpos_clusters": int(len(np.unique(clusters))),
        "mean_similarity_percentile": mean_pct, "ci95_clustered": ci_pct,
        "frac_above_median": round(float(np.mean(abovemed)), 4),
        "mean_cos_replacement": round(float(np.mean(repl_cos)), 4),
        "mean_cos_random_same_pos": round(float(np.mean(rand_cos)), 4),
        "interpretation": ("VALID: realized replacements are markedly more similar than random same-position "
                           "players (label carries similarity signal beyond position)" if mean_pct > 0.60
                           else "WEAK-but-present similarity signal above the 0.50 null" if mean_pct > 0.53
                           else "INVALID: label ~ position + noise (no similarity signal beyond position)"),
    }
    print(f"n_pairs={out['n_pairs_scored']}  clusters={out['n_subpos_clusters']}")
    print(f"mean similarity-percentile of realized replacement = {mean_pct:.3f}  CI{ci_pct}  (null=0.50)")
    print(f"frac above median = {out['frac_above_median']:.3f}")
    print(f"mean cos: replacement={out['mean_cos_replacement']:.4f}  random-same-pos={out['mean_cos_random_same_pos']:.4f}")
    print(f"=> {out['interpretation']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
