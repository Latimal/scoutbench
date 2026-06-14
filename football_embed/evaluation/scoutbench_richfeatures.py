#!/usr/bin/env python3
"""ScoutBench Task B -- can RICHER INPUTS beat the raw-stat kNN?

The paper's negative result is "no learned representation beats a raw-stat kNN on
Task B". But every rep tested so far (v11, the trained metric, NMF, PCA,
Player-Vectors) is a projection of the SAME 88-d variance-pruned stat card, so it can
only RESHAPE the card's information, never ADD any. This script tests the one thing
that could break the ceiling: a representation built from inputs the card does NOT
contain.

Richer inputs available (all keyed to the 5668-player gallery by player_name):
  - event_emb : 256-d event-transformer embedding from event SEQUENCES (trained from
    scratch on SPADL). 100% gallery coverage. Linear R^2=0.14 predicting the card from
    it, similarity-structure Pearson r=0.01 vs card -> genuinely ORTHOGONAL.
  - vaep      : 3-d value-added-per-action (from repr_card_vaep f88/f89/f90). 100%.
  - sb360     : StatsBomb 360 spatial set-transformer fingerprint. ~923 gallery players
    (tournament subset); evaluated ON its covered subset with a FAIR raw_card-on-the-
    same-subset comparison (same query set AND same candidate pool).

PRIMARY test = fuse + direct cosine kNN (training-free, zero hyperparameters, zero seed
variance -- un-gameable). It answers exactly "do richer INPUTS add information?".
SECONDARY test = a metric MLP trained over the fused inputs on a player-disjoint split,
>=3 seeds (it conflates input-richness with metric-learning, and the ~22 pair-graph
components are ≈ sub-positions so component-disjoint training is degenerate -- reported
with that caveat).

All fused vectors are built block-wise: each block is L2-normalized, optionally
weighted, then concatenated and L2-normalized again, so no single block's raw scale
dominates the cosine. raw_card here is the EXACT _l2(_card_matrix) the harness uses.

Significance: block bootstrap over connected-component clusters + conservative
component-t, identical to scoutbench_blockboot.py (imported, not reimplemented).

Does NOT modify scoutbench.py / scoutbench_blockboot.py / scoutbench_metric_baseline.py /
sb360_crossmatch_killtest.py. Offline, MPS/CPU.

Usage:
    .venv/bin/python3 -m football_embed.evaluation.scoutbench_richfeatures --all
    .venv/bin/python3 -m football_embed.evaluation.scoutbench_richfeatures --build-360-only
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd

from football_embed.evaluation.scoutbench import (
    DEF_GALLERY, DEF_JOIN, DEF_PAIRS, _card_matrix, _l2, load_queries, _qm,
)
from football_embed.evaluation.scoutbench_blockboot import _tm2gi, _components, block_boot

EVENT_EMB = "data/processed/embeddings/player_event_embeddings.parquet"
CARD_VAEP = "data/processed/benchmark/repr_card_vaep.parquet"
SB360_NPZ = "data/processed/benchmark/sb360_sets_matchkeyed.npz"
OUT_DIR = Path("data/processed/benchmark")
REPR_FUSED = OUT_DIR / "repr_rich_fused.parquet"
REPR_360 = OUT_DIR / "repr_sb360_fingerprint.parquet"
OUT_JSON = OUT_DIR / "scoutbench_richfeatures.json"


# --------------------------------------------------------------------------- #
# Signal loaders -> per-gallery-row matrices (NaN-imputed, NOT yet normalized)
# --------------------------------------------------------------------------- #

def load_event_emb(gallery: pd.DataFrame) -> np.ndarray:
    ee = pd.read_parquet(EVENT_EMB)
    cols = [c for c in ee.columns if c.startswith("emb_")]
    ee = ee.drop_duplicates("player_name", keep="first")  # a few players appear twice
    m = gallery[["player_name"]].merge(ee[["player_name"] + cols], on="player_name", how="left")
    assert len(m) == len(gallery), "merge changed row count -- duplicate keys"
    X = m[cols].values.astype(np.float64)
    return np.nan_to_num(X, nan=float(np.nanmean(X)))


def load_vaep(gallery: pd.DataFrame) -> np.ndarray:
    cv = pd.read_parquet(CARD_VAEP).drop_duplicates("player_name", keep="first")
    cols = [c for c in ("f88", "f89", "f90") if c in cv.columns]
    m = gallery[["player_name"]].merge(cv[["player_name"] + cols], on="player_name", how="left")
    assert len(m) == len(gallery), "merge changed row count -- duplicate keys"
    X = m[cols].values.astype(np.float64)
    return np.nan_to_num(X, nan=0.0)


def _sb_id_to_name() -> dict[int, str]:
    """StatsBomb player_id -> gallery player_name, via the event-embedding parquet
    (which carries both player_id and player_name)."""
    ee = pd.read_parquet(EVENT_EMB)
    return dict(zip(ee["player_id"].astype(int), ee["player_name"]))


def build_360_fingerprint(gallery: pd.DataFrame, *, d=96, o=128, heads=4, layers=2,
                          events=64, steps=2000, batch=64, lr=2e-3, seed=0) -> pd.DataFrame:
    """Train the SetID set-transformer (same arch/config as the 0.882 identity run) on
    TRAIN players, then embed EVERY 360 player's full event set into one fingerprint.

    Player-disjoint train/embed split (seed 7, 1/3 held out) is irrelevant for the
    fingerprint USED in Task B -- there the fingerprint is just a feature, and Task B's
    own leakage control is the realized-transfer label, not the 360 model. We still train
    only on the train-player split so the encoder is not fit to test players' identities.
    """
    import torch
    import torch.nn.functional as F
    from football_embed.evaluation.sb360_crossmatch_killtest import SetID

    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)

    z = np.load(SB360_NPZ, allow_pickle=True)
    X = z["X"]; pid = z["pid"].astype(str); tourn = z["tourn"].astype(int)
    sbid = np.array([int(float(p)) for p in pid])

    players = sorted(set(sbid))
    rng0 = np.random.default_rng(7)
    test_players = set(np.array(players)[rng0.permutation(len(players))[: len(players) // 3]].tolist())

    tr = defaultdict(list)
    for i in range(len(sbid)):
        if sbid[i] not in test_players:
            tr[(int(sbid[i]), int(tourn[i]))].append(i)
    E = events
    keys = [k for k, ix in tr.items() if len(ix) >= 2 * E]
    kidx = {k: np.array(ix) for k, ix in tr.items()}
    print(f"  [360] device={dev} train_profiles(>={2*E}ev)={len(keys)} "
          f"steps={steps} d={d} layers={layers}")

    model = SetID(d, o, heads, layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xt = torch.tensor(X); rng = np.random.default_rng(0)
    model.train()
    for step in range(steps):
        bk = [keys[i] for i in rng.choice(len(keys), size=min(batch, len(keys)), replace=False)]
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
            print(f"    step {step} loss {loss.item():.3f}", flush=True)
    model.eval()

    # one fingerprint per SB player: mean of event embeddings over ALL their events (cap)
    by_player = defaultdict(list)
    for i in range(len(sbid)):
        by_player[int(sbid[i])].append(i)
    id2name = _sb_id_to_name()
    rows = []
    with torch.no_grad():
        for spid, idxs in by_player.items():
            nm = id2name.get(spid)
            if nm is None:
                continue
            idxs = np.array(idxs)
            if len(idxs) > 600:
                idxs = np.random.default_rng(spid).choice(idxs, 600, replace=False)
            emb = model.event_emb(Xt[idxs].to(dev)).mean(0)
            emb = (emb / emb.norm().clamp_min(1e-9)).cpu().numpy()
            rows.append((nm, emb))
    fp = pd.DataFrame({"player_name": [r[0] for r in rows]})
    M = np.stack([r[1] for r in rows])
    for j in range(M.shape[1]):
        fp[f"s{j}"] = M[:, j]
    fp = fp.drop_duplicates("player_name").reset_index(drop=True)
    print(f"  [360] fingerprint players: {len(fp)} (gallery-named)")
    return fp


# --------------------------------------------------------------------------- #
# Block-wise fusion: per-block L2 -> weight -> concat -> L2
# --------------------------------------------------------------------------- #

def fuse(blocks: list[tuple[np.ndarray, float]]) -> np.ndarray:
    parts = []
    for X, w in blocks:
        parts.append(_l2(X.astype(np.float64)) * w)
    return _l2(np.hstack(parts))


def write_repr(path: Path, gallery: pd.DataFrame, emb: np.ndarray, prefix: str):
    df = pd.DataFrame({"player_name": gallery["player_name"].values})
    for j in range(emb.shape[1]):
        df[f"{prefix}{j}"] = emb[:, j]
    df.to_parquet(path, index=False)
    print(f"  wrote {path} ({emb.shape})")


# --------------------------------------------------------------------------- #
# Task B per-query MAP on an arbitrary candidate mask (fair subset comparison)
# --------------------------------------------------------------------------- #

def per_query_map(emb, qlist, subpos, cand_mask=None):
    """Per-query SP/ALL MAP. cand_mask (bool over gallery) restricts the candidate pool
    (and thus which queries are scorable) -- used for the 360 subset, applied IDENTICALLY
    to every method so the comparison is fair."""
    sims = emb.astype(np.float64) @ emb.astype(np.float64).T
    sp, al, keep = [], [], []
    for k, (qi, rel) in enumerate(qlist):
        if cand_mask is not None:
            if not cand_mask[qi]:
                continue
            rel = {r for r in rel if cand_mask[r]}
            if not rel:
                continue
        # same-position mask
        m = (subpos == subpos[qi]); m[qi] = False
        if cand_mask is not None:
            m = m & cand_mask
        a = (None if cand_mask is None else cand_mask.copy())
        if a is not None:
            a[qi] = False
        sp.append(_qm(sims[qi], qi, rel, m)["map"])
        al.append(_qm(sims[qi], qi, rel, a)["map"])
        keep.append(k)
    return np.asarray(sp), np.asarray(al), np.asarray(keep)


def clusters_for(qlist, keep, gi2comp):
    return np.array([gi2comp.get(qlist[k][0], -qlist[k][0] - 1) for k in keep])


def compare(name, emb, raw_emb, qlist, subpos, gi2comp, cand_mask, B, rng):
    sp_m, al_m, keep = per_query_map(emb, qlist, subpos, cand_mask)
    sp_r, al_r, keep_r = per_query_map(raw_emb, qlist, subpos, cand_mask)
    assert np.array_equal(keep, keep_r), "query sets diverged -- unfair comparison"
    cl = clusters_for(qlist, keep, gi2comp)
    csp = block_boot(sp_m, sp_r, cl, B, rng)   # method - raw, same-position
    cal = block_boot(al_m, al_r, cl, B, rng)   # method - raw, all-candidates
    print(f"  {name:28s} n_q={len(keep):4d} clusters={len(np.unique(cl)):2d}  "
          f"SP: rich={sp_m.mean():.4f} raw={sp_r.mean():.4f} d={csp['diff']:+.4f} "
          f"p_blk={csp['p_block']:.3f} p_t={csp['p_cluster_t']:.3f}  |  "
          f"ALL: rich={al_m.mean():.4f} raw={al_r.mean():.4f} d={cal['diff']:+.4f} "
          f"p_blk={cal['p_block']:.3f} p_t={cal['p_cluster_t']:.3f}")
    return {"n_queries": int(len(keep)),
            "same_position": {"rich_map": round(float(sp_m.mean()), 4),
                              "raw_map": round(float(sp_r.mean()), 4), **csp},
            "all_candidates": {"rich_map": round(float(al_m.mean()), 4),
                               "raw_map": round(float(al_r.mean()), 4), **cal}}


# --------------------------------------------------------------------------- #
# Secondary: trained metric over fused inputs (player-disjoint, >=3 seeds)
# --------------------------------------------------------------------------- #

def train_metric_eval(fused_in, qlist, subpos, gi2comp, tm2gi, pairs, gallery,
                      seeds, steps, B):
    """Mirror scoutbench_metric_baseline.py exactly, but the metric's INPUT is the fused
    richer-input matrix instead of the 88-d card. Player-disjoint test queries; multi-
    positive masking. Reported across seeds (mean +- std), no best-seed selection."""
    import torch
    import torch.nn.functional as F

    class Metric(torch.nn.Module):
        def __init__(s, din, h=256, dout=128, p=0.1):
            super().__init__()
            s.net = torch.nn.Sequential(
                torch.nn.Linear(din, h), torch.nn.GELU(), torch.nn.Dropout(p),
                torch.nn.Linear(h, dout))

        def forward(s, x):
            z = s.net(x)
            return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    N = fused_in.shape[0]
    Xc = torch.tensor(fused_in, dtype=torch.float32, device=dev)
    raw_full = fused_in  # raw baseline on the SAME fused inputs (cosine), apples-to-apples

    sp_diffs, al_diffs, sp_rich, sp_raw, al_rich, al_raw = [], [], [], [], [], []
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)
        test_set = set(perm[: int(N * 0.30)].tolist())

        train_edges = []
        for x, y in zip(pairs["player_x_tmid"], pairs["player_y_tmid"]):
            gx, gy = tm2gi.get(int(x)), tm2gi.get(int(y))
            if gx is not None and gy is not None and gx != gy and gx not in test_set and gy not in test_set:
                train_edges.append((gx, gy))
        train_edges = np.array(train_edges)
        pos_codes = set()
        for a, b in train_edges:
            pos_codes.add(int(a) * N + int(b)); pos_codes.add(int(b) * N + int(a))
        pos_codes = np.array(sorted(pos_codes), dtype=np.int64)

        model = Metric(fused_in.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        model.train()
        for step in range(steps):
            bi = rng.choice(len(train_edges), size=min(256, len(train_edges)), replace=False)
            ax = train_edges[bi, 0].astype(np.int64); ay = train_edges[bi, 1].astype(np.int64)
            za = model(Xc[ax]); zy = model(Xc[ay])
            m1 = np.isin(ax[:, None] * N + ay[None, :], pos_codes); np.fill_diagonal(m1, False)
            m2 = np.isin(ay[:, None] * N + ax[None, :], pos_codes); np.fill_diagonal(m2, False)
            M1 = torch.tensor(m1, device=dev); M2 = torch.tensor(m2, device=dev)
            lab = torch.arange(len(bi), device=dev)
            l1 = (za @ zy.T / 0.1).masked_fill(M1, float("-inf"))
            l2 = (zy @ za.T / 0.1).masked_fill(M2, float("-inf"))
            loss = (F.cross_entropy(l1, lab) + F.cross_entropy(l2, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        model.eval()
        with torch.no_grad():
            Z = model(Xc).cpu().numpy().astype(np.float64)
        sims_m = Z @ Z.T
        raw = _l2(raw_full).astype(np.float64); sims_r = raw @ raw.T

        keep = np.array([qi in test_set for qi, _ in qlist])
        kidx = np.where(keep)[0]
        spm = np.array([_qm(sims_m[qlist[k][0]], qlist[k][0], qlist[k][1],
                            ((subpos == subpos[qlist[k][0]]) & (np.arange(N) != qlist[k][0])))["map"] for k in kidx])
        alm = np.array([_qm(sims_m[qlist[k][0]], qlist[k][0], qlist[k][1], None)["map"] for k in kidx])
        spr = np.array([_qm(sims_r[qlist[k][0]], qlist[k][0], qlist[k][1],
                            ((subpos == subpos[qlist[k][0]]) & (np.arange(N) != qlist[k][0])))["map"] for k in kidx])
        alr = np.array([_qm(sims_r[qlist[k][0]], qlist[k][0], qlist[k][1], None)["map"] for k in kidx])
        cl = clusters_for(qlist, kidx, gi2comp)
        cb_sp = block_boot(spm, spr, cl, B, np.random.default_rng(0))
        cb_al = block_boot(alm, alr, cl, B, np.random.default_rng(0))
        sp_diffs.append(cb_sp["diff"]); al_diffs.append(cb_al["diff"])
        sp_rich.append(spm.mean()); sp_raw.append(spr.mean())
        al_rich.append(alm.mean()); al_raw.append(alr.mean())
        per_seed.append({"seed": seed, "n_test_q": int(keep.sum()),
                         "sp": {"rich": round(float(spm.mean()), 4), "raw": round(float(spr.mean()), 4), **cb_sp},
                         "all": {"rich": round(float(alm.mean()), 4), "raw": round(float(alr.mean()), 4), **cb_al}})
        print(f"    [metric seed {seed}] SP d={cb_sp['diff']:+.4f} (p_blk={cb_sp['p_block']:.3f}) "
              f"ALL d={cb_al['diff']:+.4f} (p_blk={cb_al['p_block']:.3f})")

    return {"seeds": list(seeds), "steps": steps,
            "sp_diff_mean": round(float(np.mean(sp_diffs)), 4), "sp_diff_std": round(float(np.std(sp_diffs)), 4),
            "all_diff_mean": round(float(np.mean(al_diffs)), 4), "all_diff_std": round(float(np.std(al_diffs)), 4),
            "sp_rich_mean": round(float(np.mean(sp_rich)), 4), "sp_raw_mean": round(float(np.mean(sp_raw)), 4),
            "all_rich_mean": round(float(np.mean(al_rich)), 4), "all_raw_mean": round(float(np.mean(al_raw)), 4),
            "per_seed": per_seed,
            "caveat": "pair-graph components ~= sub-positions; player-disjoint split is "
                      "not component-disjoint, so a test query's component can include "
                      "trained players. Treat as secondary corroboration only."}


def main():
    ap = argparse.ArgumentParser(description="ScoutBench Task B: richer-input vs raw_card")
    ap.add_argument("--gallery", default=DEF_GALLERY)
    ap.add_argument("--pairs", default=DEF_PAIRS)
    ap.add_argument("--join", default=DEF_JOIN)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric-seeds", default="7,13,29")
    ap.add_argument("--metric-steps", type=int, default=3000)
    ap.add_argument("--sb360-steps", type=int, default=2000)
    ap.add_argument("--skip-metric", action="store_true")
    ap.add_argument("--skip-360", action="store_true")
    ap.add_argument("--build-360-only", action="store_true")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    gallery = pd.read_parquet(args.gallery)
    join = pd.read_parquet(args.join); pairs = pd.read_parquet(args.pairs)
    N = len(gallery)

    # 360 fingerprint (build & cache)
    if not args.skip_360:
        if REPR_360.exists() and not args.build_360_only:
            print(f"using cached 360 fingerprint {REPR_360}")
        else:
            fp = build_360_fingerprint(gallery, steps=args.sb360_steps)
            fp.to_parquet(REPR_360, index=False)
            print(f"  wrote {REPR_360}")
            if args.build_360_only:
                return

    # signal matrices
    card = _card_matrix(gallery)
    raw_card = _l2(card)
    event = load_event_emb(gallery)
    vaep = load_vaep(gallery)

    # full-coverage fused reps (PRIMARY: direct kNN)
    fused_ce = fuse([(card, 1.0), (event, 1.0)])                 # card + event
    fused_cev = fuse([(card, 1.0), (event, 1.0), (vaep, 0.5)])   # card + event + vaep
    fused_cv = fuse([(card, 1.0), (vaep, 0.5)])                  # card + vaep (isolate vaep)
    event_only = _l2(event)                                       # event alone
    write_repr(REPR_FUSED, gallery, fused_cev, "z")

    queries, subpos = load_queries(gallery, join, pairs)
    qlist = list(queries.items())
    tm2gi = _tm2gi(gallery, join)
    comp = _components(pairs, tm2gi)
    gi2comp = {tm2gi[tm]: root for tm, root in comp.items() if tm in tm2gi}

    rng = np.random.default_rng(args.seed)
    results = {"n_gallery": N, "bootstrap": args.bootstrap}

    print(f"\n=== PRIMARY: fuse + direct kNN, FULL gallery ({len(qlist)} queries) ===")
    print("    (raw baseline = _l2(card), the exact harness raw_card)")
    full = {}
    full["card+event"] = compare("card+event", fused_ce, raw_card, qlist, subpos, gi2comp, None, args.bootstrap, rng)
    full["card+event+vaep"] = compare("card+event+vaep", fused_cev, raw_card, qlist, subpos, gi2comp, None, args.bootstrap, rng)
    full["card+vaep"] = compare("card+vaep", fused_cv, raw_card, qlist, subpos, gi2comp, None, args.bootstrap, rng)
    full["event_only"] = compare("event_only", event_only, raw_card, qlist, subpos, gi2comp, None, args.bootstrap, rng)
    results["primary_direct_knn_full"] = full

    # 360 subset: fair same-subset, same-candidate-pool comparison
    if not args.skip_360 and REPR_360.exists():
        fp = pd.read_parquet(REPR_360)
        scols = [c for c in fp.columns if c != "player_name"]
        m = gallery[["player_name"]].merge(fp, on="player_name", how="left")
        has360 = m[scols[0]].notna().values
        S360 = np.nan_to_num(m[scols].values.astype(np.float64), nan=0.0)
        cand_mask = has360.copy()
        n_cov = int(cand_mask.sum())
        print(f"\n=== 360 SUBSET: candidate pool restricted to {n_cov} 360-covered players ===")
        print("    (raw_card evaluated on the IDENTICAL query set + candidate pool)")
        sub = {}
        # 360 fingerprint alone
        sub["sb360_only"] = compare("sb360_only", _l2(S360), raw_card, qlist, subpos, gi2comp, cand_mask, args.bootstrap, rng)
        # card + 360
        fused_c360 = fuse([(card, 1.0), (S360, 1.0)])
        sub["card+sb360"] = compare("card+sb360", fused_c360, raw_card, qlist, subpos, gi2comp, cand_mask, args.bootstrap, rng)
        # card + event + 360 (everything)
        fused_all = fuse([(card, 1.0), (event, 1.0), (S360, 1.0), (vaep, 0.5)])
        sub["card+event+sb360+vaep"] = compare("card+event+sb360+vaep", fused_all, raw_card, qlist, subpos, gi2comp, cand_mask, args.bootstrap, rng)
        # and card+event (no 360) ON the same subset -- isolates 360's marginal contribution
        sub["card+event(subset)"] = compare("card+event(subset)", fused_ce, raw_card, qlist, subpos, gi2comp, cand_mask, args.bootstrap, rng)
        results["sub360_direct_knn"] = {"n_covered": n_cov, **sub}

    # SECONDARY: trained metric over fused inputs
    if not args.skip_metric:
        seeds = [int(s) for s in args.metric_seeds.split(",")]
        print(f"\n=== SECONDARY: metric MLP trained on fused inputs (card+event+vaep), seeds={seeds} ===")
        results["secondary_trained_metric"] = train_metric_eval(
            fused_cev, qlist, subpos, gi2comp, tm2gi, pairs, gallery, seeds, args.metric_steps, args.bootstrap)

    # ----- verdict -----
    def beats(c):  # significant + positive on SP (primary scope) by block p and component-t
        sp = c["same_position"]
        return sp["diff"] > 0 and sp["p_block"] < 0.05 and sp["p_cluster_t"] < 0.05

    winners = [k for k, c in full.items() if beats(c)]
    if "sub360_direct_knn" in results:
        winners += [f"360:{k}" for k, c in results["sub360_direct_knn"].items()
                    if k != "n_covered" and beats(c)]
    if winners:
        verdict = ("RICHER INPUTS BEAT RAW (robust): " + ", ".join(winners) +
                   " -> the negative result has a counterexample; paper pivots to "
                   "'leakage-free protocol + first method that earns its keep'")
    else:
        best = max(full.items(), key=lambda kv: kv[1]["same_position"]["diff"])
        verdict = ("HONEST NEGATIVE: no richer-input representation beats raw_card "
                   "robustly (block-p<.05 AND component-t<.05) on SP-MAP. Best richer rep = "
                   f"{best[0]} (SP diff {best[1]['same_position']['diff']:+.4f}, "
                   f"p_block={best[1]['same_position']['p_block']}, "
                   f"p_t={best[1]['same_position']['p_cluster_t']}). The card already "
                   "captures the linearly-available similarity; orthogonal richer inputs "
                   "(event sequences, VAEP, 360 spatial) do NOT add usable signal on this "
                   "noisy realized-transfer label. Strengthens the paper.")
    results["verdict"] = verdict
    print(f"\n=== VERDICT ===\n{verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
