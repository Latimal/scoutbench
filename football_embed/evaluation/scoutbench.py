#!/usr/bin/env python3
"""ScoutBench -- the externally-validated football player-retrieval benchmark.

Unlike the legacy text-retrieval bench (whose zone/archetype relevance is derived
from the model's OWN K-means/rule labels -> circular, saturates at nDCG~0.99),
ScoutBench grades against ground truth that does NOT come from the model under
test. It is the field's first external referee for player similarity / NL search.

TASK B -- realized replacement / similar-player retrieval.
  Ground truth = realized replacement transfers (CC0 transfermarkt): a club lost
  player X at sub-position P and signed Y at P in the same/next window -> (X,Y) is
  a silver similarity-positive pair. For each query player, rank all others by
  embedding cosine and measure whether the realized replacement ranks highly.
  Metrics: Hit@{1,5,10}, Recall@{5,10}, MRR, MAP -- over ALL candidates and
  SAME-sub-position candidates (the hard within-role test).

TASK A -- natural-language attribute retrieval (v11 NL model only).
  NL queries graded against EXTERNAL transfermarkt attributes (height / foot /
  age) plus position -- never the model's labels. Reports position precision@10,
  attribute precision@10, and LIFT over the position-only base rate. lift<1 means
  the model ignores the descriptive adjective.

Key finding (2026-05): no learned representation (incl. v11) beats a raw-stat
kNN on Task B, and v11 ignores attributes on Task A (lift 0.27x). The dense
embedding is not the lever; ship within-position raw-stat kNN + filters instead.

Usage:
    .venv/bin/python3 -m football_embed.evaluation.scoutbench --task both
    .venv/bin/python3 -m football_embed.evaluation.scoutbench --task b \
        --methods v11,raw_card,random,file:data/processed/benchmark/repr_pca.parquet
"""

import argparse
import json
import os
from pathlib import Path

# v11 text branch loads ModernBERT base from HF cache; force offline so the
# hub-update check does not hang under a no-network sandbox (weights must be
# pre-downloaded once with network).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd

DEF_GALLERY = "data/processed/text/player_card_gallery.parquet"
DEF_PAIRS = "data/processed/benchmark/replacement_pairs.parquet"
DEF_JOIN = "data/processed/benchmark/transfermarkt_join.parquet"
DEF_CKPT = "checkpoints/text_branch/v11/best"
DEF_PLAYERS_CSV = "data/raw/transfermarkt/players.csv"


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _card_matrix(g: pd.DataFrame) -> np.ndarray:
    cols = sorted([c for c in g.columns if c.startswith("card_")], key=lambda c: int(c.split("_")[1]))
    X = g[cols].values.astype(np.float32)
    # sanitize: a handful of cards carry nan/inf or runaway values (data-construction
    # artifacts) that overflow float32 matmuls. Clip to a generous z-score range (legit
    # values are within ~±15) and zero out non-finite -- aggregate metrics unchanged.
    return np.nan_to_num(np.clip(X, -50.0, 50.0), nan=0.0, posinf=0.0, neginf=0.0)


def card_embeddings(method: str, gallery: pd.DataFrame, checkpoint: str) -> np.ndarray:
    """Gallery embeddings for a retrieval method. method: v11|raw_card|random|file:<path>."""
    cards = _card_matrix(gallery)
    if method == "raw_card":
        return _l2(cards)
    if method == "random":
        return _l2(np.random.default_rng(0).standard_normal((len(gallery), 64)).astype(np.float32))
    if method == "v11":
        import torch
        from football_embed.model.text_branch import PlayerCardProjector
        proj = PlayerCardProjector.load(checkpoint, device="cpu"); proj.eval()
        with torch.no_grad():
            return _l2(proj(torch.tensor(cards)).cpu().numpy())
    if method.startswith("file:"):
        r = pd.read_parquet(method[5:])
        m = gallery[["player_name"]].merge(r, on="player_name", how="left")
        X = m[[c for c in r.columns if c != "player_name"]].values.astype(np.float32)
        return _l2(np.nan_to_num(X, nan=float(np.nanmean(X))))
    raise ValueError(f"unknown method {method}")


# --------------------------------------------------------------------------- #
# Task B
# --------------------------------------------------------------------------- #

def load_queries(gallery, join, pairs):
    name2gi = {n: i for i, n in enumerate(gallery["player_name"].values)}
    tm2gi, sub = {}, [None] * len(gallery)
    for _, r in join.iterrows():
        gi = name2gi.get(r["player_name"])
        if gi is None:
            continue
        tm2gi.setdefault(int(r["tm_player_id"]), gi)
        sub[gi] = r["tm_sub_position"]
    subpos = np.array([s if s is not None else "?" for s in sub], dtype=object)
    queries = {}
    for x, y in zip(pairs["player_x_tmid"], pairs["player_y_tmid"]):
        gx, gy = tm2gi.get(int(x)), tm2gi.get(int(y))
        if gx is not None and gy is not None and gx != gy:
            queries.setdefault(gx, set()).add(gy)
            queries.setdefault(gy, set()).add(gx)
    return queries, subpos


def _qm(sims_row, qi, rel, mask):
    s = sims_row.copy(); s[qi] = -np.inf
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    order = np.argsort(-s)
    rk = np.empty(len(order), dtype=np.int64); rk[order] = np.arange(len(order))
    rr = np.sort([rk[r] for r in rel]); first = rr[0]
    ap = float(np.mean(np.arange(1, len(rr) + 1) / (rr + 1)))
    return {"hit@1": float(first < 1), "hit@5": float(first < 5), "hit@10": float(first < 10),
            "recall@10": float(np.mean(rr < 10)), "mrr": float(1 / (first + 1)), "map": ap}


def eval_task_b(emb, queries, subpos):
    sims = emb @ emb.T
    A, P = [], []
    for qi, rel in queries.items():
        A.append(_qm(sims[qi], qi, rel, None))
        m = (subpos == subpos[qi]); m[qi] = False
        P.append(_qm(sims[qi], qi, rel, m))
    mean = lambda rows: {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
    return {"n_queries": len(queries), "all_candidates": mean(A), "same_position": mean(P)}


# --------------------------------------------------------------------------- #
# Task A
# --------------------------------------------------------------------------- #

_WING = {"Left Winger", "Right Winger"}
_FB = {"Left-Back", "Right-Back"}
# (text, position predicate, attribute predicate) over per-player arrays S/H/F/Ag
NL_ATTRIBUTE_QUERIES = [
    ("a tall centre back dominant in the air", lambda S, H, F, Ag, i: S[i] == "Centre-Back", lambda S, H, F, Ag, i: H[i] >= 188),
    ("a left footed centre back", lambda S, H, F, Ag, i: S[i] == "Centre-Back", lambda S, H, F, Ag, i: F[i] == "left"),
    ("a young attacking midfielder", lambda S, H, F, Ag, i: S[i] == "Attacking Midfield", lambda S, H, F, Ag, i: Ag[i] <= 23),
    ("a tall target man centre forward", lambda S, H, F, Ag, i: S[i] == "Centre-Forward", lambda S, H, F, Ag, i: H[i] >= 188),
    ("a left footed full back", lambda S, H, F, Ag, i: S[i] in _FB, lambda S, H, F, Ag, i: F[i] == "left"),
    ("a young winger", lambda S, H, F, Ag, i: S[i] in _WING, lambda S, H, F, Ag, i: Ag[i] <= 22),
    ("a left footed winger", lambda S, H, F, Ag, i: S[i] in _WING, lambda S, H, F, Ag, i: F[i] == "left"),
    ("a tall commanding goalkeeper", lambda S, H, F, Ag, i: S[i] == "Goalkeeper", lambda S, H, F, Ag, i: H[i] >= 190),
    ("a short creative central playmaker", lambda S, H, F, Ag, i: S[i] in ("Attacking Midfield", "Central Midfield"), lambda S, H, F, Ag, i: H[i] <= 173),
    ("an experienced veteran goalkeeper", lambda S, H, F, Ag, i: S[i] == "Goalkeeper", lambda S, H, F, Ag, i: Ag[i] >= 32),
]


def load_external_attrs(gallery, join, players_csv):
    from datetime import datetime
    tm = pd.read_csv(players_csv, low_memory=False)
    attr = {int(r.player_id): (r.height_in_cm, r.foot, r.date_of_birth, r.sub_position) for _, r in tm.iterrows()}
    n2t = dict(zip(join.player_name, join.tm_player_id))
    N = len(gallery)
    H = np.full(N, np.nan); F = np.array([None] * N, dtype=object); Ag = np.full(N, np.nan); S = np.array([None] * N, dtype=object)
    for i, nm in enumerate(gallery.player_name):
        tid = n2t.get(nm)
        if tid is None or int(tid) not in attr:
            continue
        h, f, dob, sp = attr[int(tid)]
        H[i] = h if pd.notna(h) else np.nan; F[i] = f; S[i] = sp
        if isinstance(dob, str) and len(dob) >= 10:
            try:
                Ag[i] = (datetime(2020, 7, 1) - datetime.strptime(dob[:10], "%Y-%m-%d")).days / 365.25
            except Exception:
                pass
    return S, H, F, Ag


def eval_task_a(checkpoint, gallery, S, H, F, Ag):
    import torch
    from football_embed.model.text_branch import TextBranch, PlayerCardProjector
    proj = PlayerCardProjector.load(checkpoint, device="cpu"); proj.eval()
    with torch.no_grad():
        G = _l2(proj(torch.tensor(_card_matrix(gallery))).cpu().numpy())
    model = TextBranch.load(checkpoint, device="cpu"); model.eval()
    qemb = model.encode_queries([q for q, _, _ in NL_ATTRIBUTE_QUERIES]).numpy()
    sims = qemb @ G.T
    N = len(gallery); rows = []; pos_all = []; attr_all = []; lift_all = []
    for qi, (q, posf, attrf) in enumerate(NL_ATTRIBUTE_QUERIES):
        top = np.argsort(-sims[qi])[:10]
        posok = [1 if (S[j] is not None and posf(S, H, F, Ag, j)) else 0 for j in top]
        attrok = [1 if (S[j] is not None and posf(S, H, F, Ag, j) and attrf(S, H, F, Ag, j)) else 0 for j in top]
        pos_idx = [i for i in range(N) if S[i] is not None and posf(S, H, F, Ag, i)]
        base = np.mean([1 if attrf(S, H, F, Ag, i) else 0 for i in pos_idx]) if pos_idx else 0.0
        p10, a10 = float(np.mean(posok)), float(np.mean(attrok))
        lift = (a10 / base) if base > 0 else float("nan")
        pos_all.append(p10); attr_all.append(a10); lift_all.append(lift)
        rows.append({"query": q, "posP@10": round(p10, 3), "attrP@10": round(a10, 3),
                     "base": round(float(base), 3), "lift": round(lift, 3) if base > 0 else None})
    return {"per_query": rows, "mean_posP@10": round(float(np.mean(pos_all)), 3),
            "mean_attrP@10": round(float(np.mean(attr_all)), 3),
            "mean_lift": round(float(np.nanmean(lift_all)), 3)}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="ScoutBench evaluator")
    ap.add_argument("--task", choices=["a", "b", "both"], default="both")
    ap.add_argument("--gallery", default=DEF_GALLERY)
    ap.add_argument("--pairs", default=DEF_PAIRS)
    ap.add_argument("--join", default=DEF_JOIN)
    ap.add_argument("--players-csv", default=DEF_PLAYERS_CSV)
    ap.add_argument("--checkpoint", default=DEF_CKPT)
    ap.add_argument("--methods", default="raw_card,v11,random,file:data/processed/benchmark/repr_pca.parquet,file:data/processed/benchmark/repr_fbref.parquet,file:data/processed/benchmark/repr_text.parquet,file:data/processed/benchmark/repr_card_vaep.parquet")
    ap.add_argument("--out", default="data/processed/benchmark/scoutbench_results.json")
    args = ap.parse_args()

    gallery = pd.read_parquet(args.gallery)
    results = {}

    if args.task in ("b", "both"):
        queries, subpos = load_queries(gallery, pd.read_parquet(args.join), pd.read_parquet(args.pairs))
        print(f"=== Task B: replacement retrieval ({len(queries)} query players) ===")
        print(f"{'method':42s}{'SP MAP':>8}{'SP Hit@10':>10}{'ALL Hit@10':>11}")
        tb = {}
        for m in [x.strip() for x in args.methods.split(",") if x.strip()]:
            try:
                r = eval_task_b(card_embeddings(m, gallery, args.checkpoint), queries, subpos)
            except Exception as e:
                print(f"{m:42s} ERROR {e}"); continue
            tb[m] = r
            star = "  <- baseline" if m == "raw_card" else ("  <- model" if m == "v11" else "")
            print(f"{m:42s}{r['same_position']['map']:>8.3f}{r['same_position']['hit@10']:>10.3f}{r['all_candidates']['hit@10']:>11.3f}{star}")
        results["task_b"] = tb

    if args.task in ("a", "both"):
        S, H, F, Ag = load_external_attrs(gallery, pd.read_parquet(args.join), args.players_csv)
        print(f"\n=== Task A: NL attribute retrieval (v11) ===")
        ta = eval_task_a(args.checkpoint, gallery, S, H, F, Ag)
        for r in ta["per_query"]:
            print(f"  {r['query'][:42]:42s} posP@10={r['posP@10']:.2f} attrP@10={r['attrP@10']:.2f} lift={r['lift']}")
        print(f"  MEAN posP@10={ta['mean_posP@10']} attrP@10={ta['mean_attrP@10']} lift={ta['mean_lift']} (lift<1 => ignores attribute)")
        results["task_a"] = ta

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
