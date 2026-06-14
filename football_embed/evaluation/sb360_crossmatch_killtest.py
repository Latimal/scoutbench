#!/usr/bin/env python3
"""KT2: is the 0.882 within-tournament 360 identity real style, or match leakage?

The headline 360 result (set-transformer, within-tournament Top-1 0.882) splits a
player's events by RANDOM half, so events from the SAME match can land in both the
query and target halves -- the model may be recognizing "this specific game"
(same opponents, score state, pitch) rather than enduring individual style.

This trains ONE model (the small d96/L2/2k-step config that produced 0.882) and
evaluates it two ways, changing ONLY the eval split:
  - split-half : original protocol (random half/half within a profile)  -> reproduction check
  - cross-match: query and target drawn from DISJOINT matches             -> leakage-free KILL TEST
Same trained weights, same held-out players, same retrieval metric. If cross-match
Top-1 holds (>= ~0.5-0.6) the individual-style claim survives; if it collapses
toward the cross-tournament number (0.178), the 0.882 was largely match-context.

Train split is identical to train_sb360_identity.py (seed 7, 1/3 held-out players).
Runs offline on local MPS/CPU.

Usage:
    .venv/bin/python3 -m football_embed.evaluation.sb360_crossmatch_killtest \
        --d 96 --layers 2 --heads 4 --steps 2000
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MAXP = 23
DATA = "data/processed/benchmark/sb360_sets_matchkeyed.npz"
OUT = "data/processed/benchmark/sb360_crossmatch_killtest.json"


class SetID(torch.nn.Module):
    def __init__(s, d=96, o=128, heads=4, layers=2):
        super().__init__()
        s.embed = torch.nn.Linear(5, d)
        layer = torch.nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True, dropout=0.1, activation="gelu")
        s.enc = torch.nn.TransformerEncoder(layer, layers)
        s.head = torch.nn.Linear(d, o)

    def event_emb(s, x):
        mask = (x.abs().sum(-1) == 0); mask[:, 0] = False
        h = s.embed(x)
        h = s.enc(h, src_key_padding_mask=mask)
        h = h.masked_fill(mask.unsqueeze(-1), 0.0).sum(1) / (~mask).sum(1, keepdim=True).clamp_min(1)
        return s.head(h)


def _retr(A, B):
    pids = [p for p in (A.keys() & B.keys()) if np.all(np.isfinite(A[p])) and np.all(np.isfinite(B[p]))]
    if len(pids) < 5:
        return None
    Am = np.array([A[p] for p in pids]); Bm = np.array([B[p] for p in pids])
    S = Am @ Bm.T
    hit1 = np.empty(len(pids)); rr = np.empty(len(pids))
    for i in range(len(pids)):
        r = int(np.where(np.argsort(-S[i]) == i)[0][0]); hit1[i] = float(r < 1); rr[i] = 1.0 / (r + 1)
    return len(pids), int(hit1.sum()), float(rr.sum()), hit1, rr


def _agg(rows, boot=2000, seed=0):
    """Pooled (player-weighted) Top-1/MRR + player-level bootstrap CI + per-tournament
    breakdown + macro Top-1 (each tournament equal-weight, so no single pool dominates)."""
    rows = [r for r in rows if r]
    if not rows:
        return {"n": 0, "n_groups": 0, "top1": 0.0, "mrr": 0.0}
    w = sum(x[0] for x in rows)
    per_t = [round(x[1] / x[0], 3) for x in rows]          # per-tournament Top-1
    all_hit1 = np.concatenate([x[3] for x in rows])         # pooled per-player hit@1 (each player once)
    rng = np.random.default_rng(seed)
    bs = all_hit1[rng.integers(0, len(all_hit1), size=(boot, len(all_hit1)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": int(w), "n_groups": len(rows),
            "top1": round(sum(x[1] for x in rows) / w, 4), "mrr": round(sum(x[2] for x in rows) / w, 4),
            "top1_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "per_tournament_top1": per_t, "macro_top1": round(float(np.mean(per_t)), 4)}


def _drop_team(ev, drop):
    if not drop:
        return ev
    ev = ev.copy(); ev[ev[:, :, 3] == 1.0] = 0.0
    return ev


def eval_samematch(emb_of, X, pid, tourn, match, test_mask, rng, min_events=40, min_side=10, cap=200):
    """POSITIVE CONTROL: query/target are disjoint event-halves of the SAME match.

    If match-context is the leakage source, this should be HIGH (≈split-half), proving
    the cross-match collapse is driven by match recognition, not lost individual signal.
    """
    idx_by = defaultdict(lambda: defaultdict(list))
    for i in np.where(test_mask)[0]:
        idx_by[(pid[i], int(tourn[i]))][int(match[i])].append(i)
    by_t = defaultdict(dict)
    for (p, t), mdict in idx_by.items():
        # pick the player's highest-volume match with enough events to split
        best = max(mdict.values(), key=len)
        if len(best) < max(min_events, 2 * min_side):
            continue
        ix = np.array(best); pm = rng.permutation(len(ix)); h = len(ix) // 2
        by_t[t][p] = (emb_of(X[ix[pm[:h]][:cap]]), emb_of(X[ix[pm[h:]][:cap]]))
    return _agg([_retr({p: v[0] for p, v in d.items()}, {p: v[1] for p, v in d.items()}) for d in by_t.values()])


def eval_splithalf(emb_of, X, pid, tourn, test_mask, rng, min_events=40, cap=200):
    """Original protocol: random half/half within each (player, tournament)."""
    idx_by = defaultdict(list)
    for i in np.where(test_mask)[0]:
        idx_by[(pid[i], int(tourn[i]))].append(i)
    by_t = defaultdict(dict)
    for (p, t), ix in idx_by.items():
        ix = np.array(ix)
        if len(ix) < min_events:
            continue
        pm = rng.permutation(len(ix)); h = len(ix) // 2
        by_t[t][p] = (emb_of(X[ix[pm[:h]][:cap]]), emb_of(X[ix[pm[h:]][:cap]]))
    return _agg([_retr({p: v[0] for p, v in d.items()}, {p: v[1] for p, v in d.items()}) for d in by_t.values()])


def eval_crossmatch(emb_of, X, pid, tourn, match, test_mask, rng, min_events=40, min_side=10, cap=200, drop_teammate=False):
    """Leakage-free: query and target drawn from DISJOINT matches of the same profile.

    Matches of a profile are sorted and split alternately (even->query, odd->target)
    to balance volume; both sides must clear min_side events and the profile min_events.
    drop_teammate zeros teammate tokens (characterizes the surviving fingerprint).
    """
    idx_by = defaultdict(lambda: defaultdict(list))
    for i in np.where(test_mask)[0]:
        idx_by[(pid[i], int(tourn[i]))][int(match[i])].append(i)
    by_t = defaultdict(dict); n_skip_1match = 0; side_sizes = []
    for (p, t), mdict in idx_by.items():
        total = sum(len(v) for v in mdict.values())
        if total < min_events:
            continue
        if len(mdict) < 2:
            n_skip_1match += 1; continue
        ms = sorted(mdict)
        qa = [i for k, m in enumerate(ms) if k % 2 == 0 for i in mdict[m]]
        qb = [i for k, m in enumerate(ms) if k % 2 == 1 for i in mdict[m]]
        if len(qa) < min_side or len(qb) < min_side:
            continue
        side_sizes += [min(len(qa), cap), min(len(qb), cap)]
        a = _drop_team(X[np.array(qa)[:cap]], drop_teammate)
        b = _drop_team(X[np.array(qb)[:cap]], drop_teammate)
        by_t[t][p] = (emb_of(a), emb_of(b))
    res = _agg([_retr({p: v[0] for p, v in d.items()}, {p: v[1] for p, v in d.items()}) for d in by_t.values()])
    res["profiles_skipped_single_match"] = n_skip_1match
    res["mean_events_per_side"] = round(float(np.mean(side_sizes)), 1) if side_sizes else 0.0
    return res


def main():
    ap = argparse.ArgumentParser(description="KT2: cross-match vs split-half 360 identity")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--d", type=int, default=96); ap.add_argument("--o", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--events", type=int, default=64); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    try:
        torch.backends.mha.set_fastpath_enabled(False)  # MPS lacks the nested-tensor fastpath
    except Exception:
        pass
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"device={dev}")

    z = np.load(args.data, allow_pickle=True)
    X, pid, tourn, match = z["X"], z["pid"].astype(str), z["tourn"].astype(int), z["match"].astype(int)
    print(f"events={len(X)} players={len(set(pid))} profiles={len(set(zip(pid, tourn)))} matches={len(set(match))}")

    # held-out player split -- IDENTICAL to train_sb360_identity.py (seed 7, 1/3 test)
    players = sorted(set(pid))
    rng0 = np.random.default_rng(7)
    test_players = set(np.array(players)[rng0.permutation(len(players))[: len(players) // 3]].tolist())
    train_mask = np.array([p not in test_players for p in pid]); test_mask = ~train_mask
    print(f"train players={len(players) - len(test_players)} test players={len(test_players)}")

    # train profiles
    tr = defaultdict(list)
    for i in np.where(train_mask)[0]:
        tr[(pid[i], int(tourn[i]))].append(i)
    E = args.events
    keys = [k for k, ix in tr.items() if len(ix) >= 2 * E]
    kidx = {k: np.array(ix) for k, ix in tr.items()}
    print(f"train profiles (>= {2*E} events) = {len(keys)}; d={args.d} layers={args.layers} heads={args.heads} steps={args.steps}")

    model = SetID(args.d, args.o, args.heads, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    Xt = torch.tensor(X); rng = np.random.default_rng(0)
    model.train()
    for step in range(args.steps):
        bk = [keys[i] for i in rng.choice(len(keys), size=min(args.batch, len(keys)), replace=False)]
        v1, v2 = [], []
        for k in bk:
            ix = kidx[k]; pm = rng.permutation(len(ix))
            z1 = model.event_emb(Xt[ix[pm[:E]]].to(dev)).mean(0)
            z2 = model.event_emb(Xt[ix[pm[E:2 * E]]].to(dev)).mean(0)
            v1.append(z1 / z1.norm().clamp_min(1e-9)); v2.append(z2 / z2.norm().clamp_min(1e-9))
        Z1, Z2 = torch.stack(v1), torch.stack(v2)
        lab = torch.arange(len(bk), device=dev)
        loss = (F.cross_entropy(Z1 @ Z2.T / 0.07, lab) + F.cross_entropy(Z2 @ Z1.T / 0.07, lab)) / 2
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 500 == 0:
            print(f"  step {step} loss {loss.item():.3f}", flush=True)
    model.eval()

    @torch.no_grad()
    def emb(ev):
        z = model.event_emb(torch.tensor(ev).to(dev)).mean(0)
        return (z / z.norm().clamp_min(1e-9)).cpu().numpy()

    sh = eval_splithalf(emb, X, pid, tourn, test_mask, np.random.default_rng(1))
    cm = eval_crossmatch(emb, X, pid, tourn, match, test_mask, np.random.default_rng(1))
    smc = eval_samematch(emb, X, pid, tourn, match, test_mask, np.random.default_rng(1))
    cmn = eval_crossmatch(emb, X, pid, tourn, match, test_mask, np.random.default_rng(1), drop_teammate=True)

    verdict = ("SURVIVES (style, not leakage)" if cm["top1"] >= 0.50
               else "PARTIAL (leakage inflates split-half)" if cm["top1"] >= 0.30
               else "COLLAPSES (0.882 was largely match-context leakage)")
    # positive control: same-match HIGH while cross-match LOW => match-context IS the driver
    control = ("CONFIRMED: same-match high + cross-match low => leakage is match-context"
               if smc["top1"] >= 0.5 and cm["top1"] < 0.3 else "INCONCLUSIVE")
    out = {
        "config": {"d": args.d, "layers": args.layers, "heads": args.heads, "steps": args.steps,
                   "note": "one model; eval-split-only change; training unchanged (random-half positives)"},
        "split_half_within": sh,
        "cross_match_within": cm,
        "cross_match_no_teammate": cmn,
        "same_match_control": smc,
        "reference": {"orig_split_half_top1": 0.882, "cross_tournament_top1": 0.178, "random_top1": 0.007},
        "delta_top1_splithalf_minus_crossmatch": round(sh["top1"] - cm["top1"], 4),
        "verdict": verdict,
        "leakage_mechanism_control": control,
        "event_count_comparable": {"crossmatch_mean_per_side": cm.get("mean_events_per_side"),
                                   "note": "both scopes cap at --cap; compare to rule out data-volume artifact"},
    }
    print("\n=== KT2 RESULT (hardened) ===")
    print(f"  split-half  within Top-1 = {sh['top1']:.3f}  MRR={sh['mrr']:.3f}  (n_groups={sh['n_groups']}, reproduction check vs 0.882)")
    print(f"  cross-match within Top-1 = {cm['top1']:.3f}  MRR={cm['mrr']:.3f}  (KILL TEST; mean ev/side={cm.get('mean_events_per_side')})")
    print(f"  cross-match NO-TEAMMATE   = {cmn['top1']:.3f}  MRR={cmn['mrr']:.3f}  (surviving fingerprint vs teammate-config)")
    print(f"  same-match CONTROL Top-1  = {smc['top1']:.3f}  MRR={smc['mrr']:.3f}  (positive control: should be HIGH)")
    print(f"  delta (leakage estimate) = {out['delta_top1_splithalf_minus_crossmatch']:.3f}")
    print(f"  VERDICT: {verdict}  |  mechanism: {control}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
