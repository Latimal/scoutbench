#!/usr/bin/env python3
"""ScoutBench Task B generalization: does the result replicate across player pools?

Reviewer gate: the headline ("no learned representation beats a raw-stat kNN; all beat
random") is shown on one gallery. This re-runs Task B INDEPENDENTLY on two disjoint
player pools split by the league the player's card was built from:
  A = top-5 men's club leagues (La Liga, Ligue 1, Serie A, Premier League, Bundesliga)
  B = international tournaments + women's + ISL/others (everything else)
For each pool: subset the gallery, restrict replacement pairs to within-pool, rank within
the pool, block-bootstrap raw_card vs {v11, faithful Player-Vectors, random}. If the
pattern holds in both genuinely different pools, the conclusion is not a pool artifact.

Offline, CPU.

Usage:
    .venv/bin/python3 -m football_embed.evaluation.scoutbench_generalization
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from football_embed.evaluation.scoutbench import (
    DEF_CKPT, DEF_GALLERY, DEF_JOIN, DEF_PAIRS, card_embeddings, _qm,
)
from football_embed.evaluation.scoutbench_blockboot import _tm2gi, block_boot

TOP5 = {"La Liga", "Ligue 1", "Serie A", "Premier League", "Bundesliga"}
METHODS = {
    "raw_card": "raw_card",
    "v11": "v11",
    "player_vectors": "file:data/processed/benchmark/repr_player_vectors.parquet",
    "random": "random",
}


def _components_local(pairs_local):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs_local:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {n: find(n) for n in parent}


def eval_pool(full_emb, sub_idx, subpos_full, queries_local, clusters, B, rng):
    """Per-query SP/ALL MAP for each method on the pool subset, then raw vs others."""
    sub_idx = np.asarray(sub_idx)
    subpos = subpos_full[sub_idx]
    qlist = list(queries_local.items())
    pq = {}
    for m, full in full_emb.items():
        emb = full[sub_idx].astype(np.float64)
        sims = emb @ emb.T
        sp, al = [], []
        for qi, rel in qlist:
            mask = (subpos == subpos[qi]); mask[qi] = False
            sp.append(_qm(sims[qi], qi, rel, mask)["map"])
            al.append(_qm(sims[qi], qi, rel, None)["map"])
        pq[m] = {"sp": np.asarray(sp), "all": np.asarray(al)}
    cl = np.array([clusters.get(qi, -qi - 1) for qi, _ in qlist])
    out = {"n_queries": len(qlist), "n_clusters": int(len(np.unique(cl))),
           "method_sp_map": {m: round(float(pq[m]["sp"].mean()), 4) for m in METHODS},
           "method_all_map": {m: round(float(pq[m]["all"].mean()), 4) for m in METHODS},
           "comparisons": {}}
    for opp in ["v11", "player_vectors", "random"]:
        for scope in ["sp", "all"]:
            r = block_boot(pq["raw_card"][scope], pq[opp][scope], cl, B, rng)
            out["comparisons"][f"raw_vs_{opp}_{scope}"] = {"diff": r["diff"], "p_block": r["p_block"],
                                                           "p_cluster_t": r["p_cluster_t"]}
    return out


def build_pool(group_gi, gallery, join, pairs):
    """queries (local idx) + clusters for a pool = set of gallery indices."""
    tm2gi = _tm2gi(gallery, join)
    gi_set = set(group_gi)
    local = {gi: i for i, gi in enumerate(sorted(gi_set))}
    queries, pairs_local = defaultdict(set), []
    for x, y in zip(pairs["player_x_tmid"], pairs["player_y_tmid"]):
        gx, gy = tm2gi.get(int(x)), tm2gi.get(int(y))
        if gx in gi_set and gy in gi_set and gx != gy:
            lx, ly = local[gx], local[gy]
            queries[lx].add(ly); queries[ly].add(lx); pairs_local.append((lx, ly))
    comp = _components_local(pairs_local)
    return sorted(gi_set), queries, comp


def main():
    ap = argparse.ArgumentParser(description="ScoutBench Task B cross-pool generalization")
    ap.add_argument("--gallery", default=DEF_GALLERY); ap.add_argument("--pairs", default=DEF_PAIRS)
    ap.add_argument("--join", default=DEF_JOIN); ap.add_argument("--checkpoint", default=DEF_CKPT)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default="data/processed/benchmark/scoutbench_generalization.json")
    args = ap.parse_args()

    gallery = pd.read_parquet(args.gallery); join = pd.read_parquet(args.join); pairs = pd.read_parquet(args.pairs)
    league = gallery["league"].values
    groupA = [i for i in range(len(gallery)) if league[i] in TOP5]
    groupB = [i for i in range(len(gallery)) if league[i] not in TOP5]
    print(f"Pool A (top-5 men's leagues): {len(groupA)} players | Pool B (tournaments/women's/other): {len(groupB)}")

    # full embeddings once
    full_emb = {m: card_embeddings(spec, gallery, args.checkpoint) for m, spec in METHODS.items()}
    # subpos over full gallery (from join)
    name2gi = {n: i for i, n in enumerate(gallery["player_name"].values)}
    sub = np.array(["?"] * len(gallery), dtype=object)
    for _, r in join.iterrows():
        gi = name2gi.get(r["player_name"])
        if gi is not None:
            sub[gi] = r["tm_sub_position"]

    rng = np.random.default_rng(0)
    results = {}
    for nm, grp in [("A_top5_mens_leagues", groupA), ("B_tournaments_womens_other", groupB)]:
        sub_idx, queries_local, clusters = build_pool(grp, gallery, join, pairs)
        res = eval_pool(full_emb, sub_idx, sub, queries_local, clusters, args.bootstrap, rng)
        results[nm] = res
        print(f"\n=== Pool {nm}: n_queries={res['n_queries']} clusters={res['n_clusters']} ===")
        print("  SP-MAP:", res["method_sp_map"])
        for k, v in res["comparisons"].items():
            if k.endswith("_sp"):
                print(f"    {k:28s} diff={v['diff']:+.4f} block_p={v['p_block']:.3f} clusterT={v['p_cluster_t']:.3f}")

    # replication verdict
    def holds(res):
        c = res["comparisons"]
        learned_not_better = c["raw_vs_v11_sp"]["diff"] >= -0.001 and c["raw_vs_player_vectors_sp"]["diff"] >= -0.001
        beats_random = c["raw_vs_random_all"]["diff"] > 0 and c["raw_vs_random_all"]["p_block"] < 0.05
        return bool(learned_not_better and beats_random)
    results["replication"] = {k: ("PATTERN HOLDS" if holds(results[k]) else "differs")
                              for k in ["A_top5_mens_leagues", "B_tournaments_womens_other"]}
    print(f"\nREPLICATION: {results['replication']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
