#!/usr/bin/env python3
"""ScoutBench Task B -- the decisive non-tautological test (review round 1, B1).

The v11 vs raw_card comparison is near-tautological: v11 shares raw_card's 88-d input
and was trained for TEXT alignment, not replacement retrieval, so it can only lose
information. The fair question a reviewer demands: can a model trained DIRECTLY for the
replacement objective beat a raw-cosine kNN on the SAME features, evaluated on
PLAYER-DISJOINT held-out players?

This trains a small projection (88->h->d, L2) with symmetric InfoNCE on replacement
pairs where BOTH players are in the train split, then evaluates on queries that are
TEST players (their pairs never seen in training), comparing to raw_card cosine on the
exact same held-out queries. Block bootstrap (connected-component clusters) + the
conservative cluster-t for significance.

Outcomes:
  - metric_model <= raw_card on held-out players  => CLEAN negative result: learning the
    metric on these features does NOT help (not just "a lossy off-objective projection
    loses"). This is the defensible, non-tautological version of the claim.
  - metric_model >  raw_card                       => the negative result was an artifact
    of v11 being off-objective; the paper pivots. Report honestly.

Offline, MPS/CPU.

Usage:
    .venv/bin/python3 -m football_embed.evaluation.scoutbench_metric_baseline --steps 3000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from football_embed.evaluation.scoutbench import (
    DEF_GALLERY, DEF_JOIN, DEF_PAIRS, _card_matrix, _l2, load_queries, _qm,
)
from football_embed.evaluation.scoutbench_blockboot import _tm2gi, _components, block_boot


class Metric(torch.nn.Module):
    def __init__(s, din, h=256, d=128, p=0.1):
        super().__init__()
        s.net = torch.nn.Sequential(
            torch.nn.Linear(din, h), torch.nn.GELU(), torch.nn.Dropout(p),
            torch.nn.Linear(h, d))

    def forward(s, x):
        z = s.net(x)
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def per_query_masked(sims, qlist, subpos, keep_q):
    sp, al, cl_ok = [], [], []
    for k, (qi, rel) in enumerate(qlist):
        if not keep_q[k]:
            continue
        m = (subpos == subpos[qi]); m[qi] = False
        sp.append(_qm(sims[qi], qi, rel, m)["map"])
        al.append(_qm(sims[qi], qi, rel, None)["map"])
        cl_ok.append(k)
    return np.asarray(sp), np.asarray(al), np.asarray(cl_ok)


def main():
    ap = argparse.ArgumentParser(description="ScoutBench Task B trained-metric baseline")
    ap.add_argument("--gallery", default=DEF_GALLERY); ap.add_argument("--pairs", default=DEF_PAIRS)
    ap.add_argument("--join", default=DEF_JOIN)
    ap.add_argument("--h", type=int, default=256); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.30); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--equiv-margin", type=float, default=0.005, help="TOST equivalence margin on MAP (predeclared)")
    ap.add_argument("--out", default="data/processed/benchmark/scoutbench_metric_baseline.json")
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    gallery = pd.read_parquet(args.gallery); join = pd.read_parquet(args.join); pairs = pd.read_parquet(args.pairs)
    queries, subpos = load_queries(gallery, join, pairs)
    qlist = list(queries.items())
    cards = _card_matrix(gallery)
    N = len(gallery)

    # player-disjoint split over gallery indices
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    test_set = set(perm[: int(N * args.test_frac)].tolist())
    is_test = np.array([i in test_set for i in range(N)])

    # training pairs: both players in TRAIN (player-disjoint from test queries)
    tm2gi = _tm2gi(gallery, join)
    train_edges = []
    for x, y in zip(pairs["player_x_tmid"], pairs["player_y_tmid"]):
        gx, gy = tm2gi.get(int(x)), tm2gi.get(int(y))
        if gx is not None and gy is not None and gx != gy and (gx not in test_set) and (gy not in test_set):
            train_edges.append((gx, gy))
    train_edges = np.array(train_edges)
    print(f"device={dev} N={N} test_players={len(test_set)} train_edges={len(train_edges)}")
    # multi-positive structure: a player has MANY true replacements; encode all positive
    # (anchor, partner) pairs so in-batch false negatives can be masked (else the loss
    # treats a player's other true replacements as negatives -- contradictory labels).
    pos_codes = set()
    for a, b in train_edges:
        pos_codes.add(int(a) * N + int(b)); pos_codes.add(int(b) * N + int(a))
    pos_codes = np.array(sorted(pos_codes), dtype=np.int64)

    Xc = torch.tensor(cards, dtype=torch.float32, device=dev)
    model = Metric(cards.shape[1], args.h, args.d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    model.train()
    for step in range(args.steps):
        bi = rng.choice(len(train_edges), size=min(args.batch, len(train_edges)), replace=False)
        ax = train_edges[bi, 0].astype(np.int64); ay = train_edges[bi, 1].astype(np.int64)
        za = model(Xc[ax]); zy = model(Xc[ay])
        # mask in-batch false negatives (other true replacements of the same anchor)
        m1 = np.isin(ax[:, None] * N + ay[None, :], pos_codes); np.fill_diagonal(m1, False)
        m2 = np.isin(ay[:, None] * N + ax[None, :], pos_codes); np.fill_diagonal(m2, False)
        M1 = torch.tensor(m1, device=dev); M2 = torch.tensor(m2, device=dev)
        lab = torch.arange(len(bi), device=dev)
        l1 = (za @ zy.T / args.tau).masked_fill(M1, float("-inf"))
        l2 = (zy @ za.T / args.tau).masked_fill(M2, float("-inf"))
        loss = (F.cross_entropy(l1, lab) + F.cross_entropy(l2, lab)) / 2
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 500 == 0:
            print(f"  step {step} loss {loss.item():.3f}", flush=True)
    model.eval()

    with torch.no_grad():
        Z = model(Xc).cpu().numpy().astype(np.float64)
    sims_metric = Z @ Z.T
    raw = _l2(cards).astype(np.float64)
    sims_raw = raw @ raw.T

    # evaluate only TEST-player queries (their pairs were never trained on)
    keep_q = np.array([qi in test_set for qi, _ in qlist])
    sp_m, al_m, idx = per_query_masked(sims_metric, qlist, subpos, keep_q)
    sp_r, al_r, _ = per_query_masked(sims_raw, qlist, subpos, keep_q)

    # clusters for the kept test queries
    comp = _components(pairs, tm2gi)
    gi2comp = {tm2gi[tm]: root for tm, root in comp.items() if tm in tm2gi}
    cl = np.array([gi2comp.get(qlist[k][0], -qlist[k][0] - 1) for k in idx])

    rng2 = np.random.default_rng(0)
    comp_sp = block_boot(sp_m, sp_r, cl, args.bootstrap, rng2)   # metric - raw, same-position
    comp_all = block_boot(al_m, al_r, cl, args.bootstrap, rng2)  # metric - raw, all-candidates
    mgn = args.equiv_margin
    def tost(comp):
        lo, hi = comp["ci95_block"]
        comp["equiv_margin"] = mgn
        comp["equivalent_within_margin"] = bool(lo > -mgn and hi < mgn)
        comp["tost_verdict"] = (f"EQUIVALENT within +-{mgn} MAP" if (lo > -mgn and hi < mgn)
                                else f"INCONCLUSIVE: no improvement detected AND cannot establish equivalence (95% CI [{lo},{hi}] exceeds +-{mgn})")
        return comp
    comp_sp = tost(comp_sp); comp_all = tost(comp_all)

    winner_sp = "metric_model" if comp_sp["diff"] > 0 else "raw_card"
    verdict = ("METRIC BEATS RAW -> negative result was an off-objective artifact; PIVOT"
               if comp_sp["diff"] > 0 and comp_sp["p_block"] < 0.05
               else "RAW >= TRAINED METRIC on held-out players -> CLEAN non-tautological negative result"
               if comp_sp["diff"] <= 0
               else "STATISTICAL TIE -> learning the metric does not help; supports negative result")
    out = {
        "n_test_queries": int(len(sp_m)), "n_clusters": int(len(np.unique(cl))),
        "train_edges": int(len(train_edges)), "config": vars(args),
        "same_position": {"metric_model_map": round(float(sp_m.mean()), 4),
                          "raw_card_map": round(float(sp_r.mean()), 4), **comp_sp},
        "all_candidates": {"metric_model_map": round(float(al_m.mean()), 4),
                           "raw_card_map": round(float(al_r.mean()), 4), **comp_all},
        "verdict": verdict,
    }
    print("\n=== METRIC-LEARNING BASELINE (held-out players) ===")
    print(f"  same-pos:  metric={sp_m.mean():.4f}  raw={sp_r.mean():.4f}  diff={comp_sp['diff']:+.4f} "
          f"block_p={comp_sp['p_block']:.4f} clusterT_p={comp_sp['p_cluster_t']:.4f}")
    print(f"  all-cand:  metric={al_m.mean():.4f}  raw={al_r.mean():.4f}  diff={comp_all['diff']:+.4f} "
          f"block_p={comp_all['p_block']:.4f} clusterT_p={comp_all['p_cluster_t']:.4f}")
    print(f"  VERDICT: {verdict}")
    print(f"  TOST(margin +-{mgn}): same-pos -> {comp_sp['tost_verdict']}")
    print(f"  TOST(margin +-{mgn}): all-cand -> {comp_all['tost_verdict']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    seed_out = args.out.replace(".json", f"_seed{args.seed}.json")
    Path(seed_out).write_text(json.dumps(out, indent=2, default=str))
    # upsert into a combined all-seeds artifact so the paper's multi-seed claim is reproducible
    comb_path = args.out.replace(".json", "_allseeds.json")
    comb = json.loads(Path(comb_path).read_text()) if Path(comb_path).exists() else {"seeds": {}}
    comb["seeds"][str(args.seed)] = {
        "same_position": {k: out["same_position"][k] for k in ("metric_model_map", "raw_card_map", "diff", "p_block", "p_cluster_t", "tost_verdict")},
        "all_candidates": {k: out["all_candidates"][k] for k in ("metric_model_map", "raw_card_map", "diff", "p_block", "p_cluster_t", "tost_verdict")},
        "verdict": verdict}
    diffs = [v["same_position"]["diff"] for v in comb["seeds"].values()]
    comb["aggregate_same_position"] = {"n_seeds": len(diffs), "mean_diff": round(float(np.mean(diffs)), 5),
        "all_nonpositive_or_ns": bool(all(d <= 0.001 for d in diffs)),
        "n_equivalent": sum("EQUIVALENT" in v["same_position"]["tost_verdict"] for v in comb["seeds"].values())}
    Path(comb_path).write_text(json.dumps(comb, indent=2, default=str))
    print(f"  saved -> {seed_out}  +  {comb_path} (seeds: {sorted(comb['seeds'])})")


if __name__ == "__main__":
    main()
