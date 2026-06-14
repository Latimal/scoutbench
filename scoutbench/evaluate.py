#!/usr/bin/env python3
"""ScoutBench Task B -- submission-facing evaluator.

Scores a third-party embedding submission against the CC0 transfer-anchored labels.
Runs entirely from the released files (``release/``) -- no StatsBomb gallery, no
non-redistributable data. The candidate pool is the 1363 CC0 query players; scoring
uses the same ``_qm`` / ``eval_task_b`` metric code that produced the paper's numbers.

A submission is a parquet/CSV with one row per player and an embedding vector:
    player_id, e0, e1, ..., e{D-1}          (player_id == transfermarkt id; preferred)
  or
    player_name, e0, e1, ..., e{D-1}         (transfermarkt display name; fallback)
Any non-key column is treated as an embedding dimension. Vectors are L2-normalized;
cosine similarity is the retrieval metric. Players missing from the submission get
a zero vector (they never retrieve / get retrieved).

Usage:
    from scoutbench import evaluate
    metrics = evaluate("my_embeddings.parquet")
    print(metrics["all_candidates"]["map"])

    scoutbench-eval --embeddings my_embeddings.parquet --out metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE_DIR = Path(__file__).resolve().parent.parent / "release"
DEF_PLAYERS = str(RELEASE_DIR / "scoutbench_taskb_players.csv")
DEF_PAIRS = str(RELEASE_DIR / "scoutbench_taskb_pairs.csv")


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n


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


def _eval_task_b(emb, queries, subpos):
    sims = emb @ emb.T
    A, P = [], []
    for qi, rel in queries.items():
        A.append(_qm(sims[qi], qi, rel, None))
        m = (subpos == subpos[qi]); m[qi] = False
        P.append(_qm(sims[qi], qi, rel, m))
    mean = lambda rows: {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
    return {"n_queries": len(queries), "all_candidates": mean(A), "same_position": mean(P)}


def _load_release(players_csv: str, pairs_csv: str):
    """Build the query structure from the CC0 release files alone."""
    players = pd.read_csv(players_csv)
    pairs = pd.read_csv(pairs_csv)
    pool = players[players["in_gallery"].astype(str).str.lower() == "true"].reset_index(drop=True)
    pid2idx = {int(r["player_id"]): i for i, r in pool.iterrows()}
    subpos = np.array(pool["sub_position"].values, dtype=object)
    queries: dict[int, set[int]] = {}
    for _, r in pairs.iterrows():
        gx, gy = pid2idx.get(int(r["player_x_id"])), pid2idx.get(int(r["player_y_id"]))
        if gx is not None and gy is not None and gx != gy:
            queries.setdefault(gx, set()).add(gy)
            queries.setdefault(gy, set()).add(gx)
    return pool, pid2idx, subpos, queries


def _load_submission(path: str) -> pd.DataFrame:
    p = str(path)
    return pd.read_csv(p) if p.endswith(".csv") else pd.read_parquet(p)


def _submission_matrix(sub: pd.DataFrame, pool: pd.DataFrame, pid2idx: dict) -> np.ndarray:
    """Align a submission to pool order; return an (n_pool, D) L2-normalized matrix."""
    key_cols = [c for c in ("player_id", "player_name") if c in sub.columns]
    if not key_cols:
        raise ValueError("submission must have a 'player_id' or 'player_name' column")
    emb_cols = [c for c in sub.columns if c not in ("player_id", "player_name")]
    if not emb_cols:
        raise ValueError("submission has no embedding columns")
    D = len(emb_cols)

    idx2vec: dict[int, np.ndarray] = {}
    if "player_id" in sub.columns:
        for _, r in sub.iterrows():
            idx = pid2idx.get(int(r["player_id"]))
            if idx is not None:
                idx2vec[idx] = r[emb_cols].to_numpy(dtype=np.float32)
    if "player_name" in sub.columns:
        name2idx = {n: i for i, n in enumerate(pool["player_name"].values)}
        for _, r in sub.iterrows():
            idx = name2idx.get(r["player_name"])
            if idx is not None:
                idx2vec.setdefault(idx, r[emb_cols].to_numpy(dtype=np.float32))

    X = np.zeros((len(pool), D), dtype=np.float32)
    for idx, vec in idx2vec.items():
        X[idx] = vec
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"submission: {len(sub)} rows, dim={D}, matched {len(idx2vec)}/{len(pool)} pool players")
    return _l2(X)


def evaluate(embeddings: str,
             players: str = DEF_PLAYERS,
             pairs: str = DEF_PAIRS,
             out: str | None = None) -> dict:
    """Score a submission against ScoutBench Task B. Returns the metrics dict."""
    pool, pid2idx, subpos, queries = _load_release(players, pairs)
    emb = _submission_matrix(_load_submission(embeddings), pool, pid2idx)
    metrics = _eval_task_b(emb, queries, subpos)
    metrics["benchmark"] = "ScoutBench Task B (replacement retrieval, CC0 pool)"
    metrics["primary_metric"] = "all_candidates.map"
    metrics["pool_size"] = len(pool)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="ScoutBench Task B evaluator")
    ap.add_argument("--embeddings", required=True,
                    help="submission parquet/CSV: player_id|player_name + embedding dims")
    ap.add_argument("--players", default=DEF_PLAYERS,
                    help="CC0 player list (default: release/scoutbench_taskb_players.csv)")
    ap.add_argument("--pairs", default=DEF_PAIRS,
                    help="CC0 pair labels (default: release/scoutbench_taskb_pairs.csv)")
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args()

    metrics = evaluate(args.embeddings, args.players, args.pairs, args.out)
    print(json.dumps(metrics, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
