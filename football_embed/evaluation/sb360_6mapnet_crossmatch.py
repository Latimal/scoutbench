#!/usr/bin/env python3
"""Does the match-context leakage (P2) also hit HEATMAP-based identity methods?

Our set-transformer (sb360_crossmatch_killtest.py) collapses from 0.878 within-tournament
Top-1 (random split-half) to 0.188 under a match-disjoint split. A reviewer asks whether
this is specific to our architecture or a property of the protocol. The closest published
identity neighbours -- 6MapNet (heatmap-CNN + triplet) and Player Vectors (NMF over
heatmaps) -- both build heatmap representations and split by sampling phases, NOT by match,
so they may suffer the same leak.

We cannot reproduce 6MapNet exactly (it needs continuous tracking; our 360 data is on-ball
freeze-frames), but we test its FAMILY: a multi-channel spatial heatmap (actor / teammate-
density / opponent-density), parameter-free, and its NMF reduction (Player-Vectors-style).
We run both through the same protocol (split-half = phase-random ~6MapNet's split; same-match
control; cross-match = leakage-free) on held-out players, with player-bootstrap CIs. If these
heatmap-family representations also collapse split-half -> cross-match, the leak is a property
of the within-tournament random-split protocol, not our model -- so published heatmap-identity
methods that split by phase are exposed to it.

Offline, CPU. Usage: .venv/bin/python3 -m football_embed.evaluation.sb360_6mapnet_crossmatch
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from football_embed.evaluation.sb360_crossmatch_killtest import (
    eval_splithalf, eval_crossmatch, eval_samematch,
)

DATA = "data/processed/benchmark/sb360_sets_matchkeyed.npz"
OUT = "data/processed/benchmark/sb360_6mapnet_crossmatch.json"
GX, GY = 12, 8  # pitch grid (matches the original 360 heatmap baseline)


def heatmap_raw(ev):
    """Multi-channel spatial map of an event set: actor / teammate-density / opponent-density.
    ev: (n, 23, 5) tokens [x, y, is_actor, is_teammate, is_opponent]. Returns concatenated,
    per-channel sum-normalized histogram (non-L2; raw for NMF input)."""
    chans = []
    # channel 0: actor location (token 0 of each event)
    actor = ev[:, 0, :2]
    masks = [actor, None, None]
    # channels 1,2: teammate / opponent point clouds across all tokens
    flat = ev.reshape(-1, 5)
    tm = flat[flat[:, 3] == 1.0][:, :2]
    op = flat[flat[:, 4] == 1.0][:, :2]
    for P in (actor, tm, op):
        if len(P) == 0:
            chans.append(np.zeros(GX * GY, dtype=np.float64)); continue
        xs = np.clip((P[:, 0] * GX).astype(int), 0, GX - 1)
        ys = np.clip((P[:, 1] * GY).astype(int), 0, GY - 1)
        h = np.bincount(ys * GX + xs, minlength=GX * GY).astype(np.float64)
        s = h.sum()
        chans.append(h / s if s > 0 else h)
    return np.concatenate(chans)


def _l2(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v


def main():
    ap = argparse.ArgumentParser(description="6MapNet/heatmap-family cross-match leakage test")
    ap.add_argument("--data", default=DATA); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nmf-k", type=int, default=24); ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    X, pid, tourn, match = z["X"], z["pid"].astype(str), z["tourn"].astype(int), z["match"].astype(int)
    players = sorted(set(pid))
    rng0 = np.random.default_rng(args.seed)
    test_players = set(np.array(players)[rng0.permutation(len(players))[: len(players) // 3]].tolist())
    test_mask = np.array([p in test_players for p in pid])
    train_mask = ~test_mask
    print(f"events={len(X)} test_players={len(test_players)}")

    # fit NMF (Player-Vectors style) on TRAIN-profile heatmaps
    from sklearn.decomposition import NMF
    tr = defaultdict(list)
    for i in np.where(train_mask)[0]:
        tr[(pid[i], int(tourn[i]))].append(i)
    H = np.array([heatmap_raw(X[np.array(ix)]) for ix in tr.values() if len(ix) >= 40])
    nmf = NMF(n_components=args.nmf_k, init="nndsvda", max_iter=400, random_state=0).fit(H)
    print(f"NMF fit on {len(H)} train profiles")

    reps = {
        "heatmap": lambda ev: _l2(heatmap_raw(ev)),                                   # param-free maps
        "nmf_playervectors": lambda ev: _l2(nmf.transform(heatmap_raw(ev)[None])[0]), # Player-Vectors-style
    }

    out = {"n_test_players": len(test_players), "grid": [GX, GY], "nmf_k": args.nmf_k, "representations": {}}
    print(f"\n{'rep':18s}{'split-half':>12}{'same-match':>12}{'cross-match':>13}{'delta':>8}  verdict")
    for name, emb in reps.items():
        sh = eval_splithalf(emb, X, pid, tourn, test_mask, np.random.default_rng(1))
        smc = eval_samematch(emb, X, pid, tourn, match, test_mask, np.random.default_rng(1))
        cm = eval_crossmatch(emb, X, pid, tourn, match, test_mask, np.random.default_rng(1))
        delta = round(sh["top1"] - cm["top1"], 3)
        # collapse = split-half and cross-match CIs are disjoint AND the drop is large
        collapse = sh["top1_ci95"][0] > cm["top1_ci95"][1] and (sh["top1"] - cm["top1"]) > 0.25
        out["representations"][name] = {"split_half": sh, "same_match_control": smc, "cross_match": cm,
                                        "delta_splithalf_minus_crossmatch": delta,
                                        "collapses": bool(collapse)}
        print(f"{name:18s}{sh['top1']:>12.3f}{smc['top1']:>12.3f}{cm['top1']:>13.3f}{delta:>8.3f}  "
              f"{'COLLAPSES (leakage)' if collapse else 'no collapse'}")
        print(f"{'':18s}  CIs: split-half {sh.get('top1_ci95')}  cross-match {cm.get('top1_ci95')}")

    out["verdict"] = ("STATIC heatmap representations (parameter-free heatmap + Player-Vectors-style NMF) "
                      "collapse under a match-disjoint split -> the leak is a property of the within-tournament "
                      "random-split protocol, not any single architecture. SCOPE: this tests the static heatmap "
                      "FAMILY; we do NOT claim 6MapNet's trained triplet-CNN collapses (untested -- needs tracking "
                      "data; a learned net could acquire match-invariant features a static histogram cannot)."
                      if all(r["collapses"] for r in out["representations"].values())
                      else "Mixed: see per-representation results.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nverdict: {out['verdict']}\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
