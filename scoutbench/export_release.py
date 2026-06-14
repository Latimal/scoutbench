#!/usr/bin/env python3
"""Export the frozen, REDISTRIBUTABLE ScoutBench Task B ground-truth release.

What this produces is derived ENTIRELY from CC0 transfermarkt data (player ids,
names, sub-positions) plus the *membership* of the gallery (which players are
scored). It contains NO StatsBomb-derived features, so it is redistributable
even though the underlying card features are research-only (see DATA.md).

Two artifacts, written to ``release/`` by default:

  scoutbench_taskb_pairs.csv      one row per directed replacement edge used as a
                                  silver similarity-positive:
                                    player_x_id, player_y_id, sub_position
                                  (transfermarkt ids; CC0). Both directions are
                                  emitted so (x,y) and (y,x) are explicit, matching
                                  the harness which makes both players queries.

  scoutbench_taskb_players.csv    one row per player that participates (is a query
                                  and/or a relevant target), with gallery membership:
                                    player_id, player_name, sub_position,
                                    is_query, in_gallery
                                  player_id == transfermarkt id; player_name is the
                                  transfermarkt display name (CC0). A submission keys
                                  its embeddings by player_id (preferred) or player_name.

Only players present in BOTH the replacement graph AND the ScoutBench gallery are
"live" (in_gallery=True, is_query=True) -- these are the 1363 queries scored by the
harness. Players in the graph but absent from the gallery are emitted with
in_gallery=False / is_query=False so the label graph is fully self-describing.

Usage:
    .venv/bin/python3 -m scoutbench.export_release
    .venv/bin/python3 -m scoutbench.export_release --out-dir release
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from football_embed.evaluation.scoutbench import (
    DEF_GALLERY,
    DEF_JOIN,
    DEF_PAIRS,
    load_queries,
)


def main():
    ap = argparse.ArgumentParser(description="Export frozen ScoutBench Task B release")
    ap.add_argument("--gallery", default=DEF_GALLERY)
    ap.add_argument("--pairs", default=DEF_PAIRS)
    ap.add_argument("--join", default=DEF_JOIN)
    ap.add_argument("--out-dir", default="release")
    args = ap.parse_args()

    gallery = pd.read_parquet(args.gallery)
    join = pd.read_parquet(args.join)
    pairs = pd.read_parquet(args.pairs)

    # gallery membership in transfermarkt-id space (same mapping the harness uses)
    name2gi = {n: i for i, n in enumerate(gallery["player_name"].values)}
    tm2gi, tm2sub, tm2name = {}, {}, {}
    for _, r in join.iterrows():
        gi = name2gi.get(r["player_name"])
        tid = int(r["tm_player_id"])
        tm2sub.setdefault(tid, r["tm_sub_position"])
        tm2name.setdefault(tid, r.get("tm_name", r["player_name"]))
        if gi is not None:
            tm2gi.setdefault(tid, gi)

    # which gallery players are actually scored as queries (matches the harness exactly)
    queries, _ = load_queries(gallery, join, pairs)
    live_gi = set(queries.keys())
    gi2tm = {gi: tm for tm, gi in tm2gi.items()}
    live_tm = {gi2tm[gi] for gi in live_gi if gi in gi2tm}

    # ---- pairs table (directed, deduped) ----
    edges = set()
    for x, y in zip(pairs["player_x_tmid"], pairs["player_y_tmid"]):
        x, y = int(x), int(y)
        if x == y:
            continue
        edges.add((x, y))
        edges.add((y, x))
    pair_rows = [
        {"player_x_id": x, "player_y_id": y,
         "sub_position": tm2sub.get(x) or tm2sub.get(y) or "?"}
        for x, y in sorted(edges)
    ]
    pairs_df = pd.DataFrame(pair_rows)

    # ---- players table ----
    all_tm = set(pairs["player_x_tmid"].astype(int)) | set(pairs["player_y_tmid"].astype(int))
    player_rows = []
    for tid in sorted(all_tm):
        in_gallery = tid in tm2gi
        player_rows.append({
            "player_id": tid,
            "player_name": tm2name.get(tid),
            "sub_position": tm2sub.get(tid, "?"),
            "is_query": tid in live_tm,
            "in_gallery": in_gallery,
        })
    players_df = pd.DataFrame(player_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "scoutbench_taskb_pairs.csv"
    players_path = out_dir / "scoutbench_taskb_players.csv"
    pairs_df.to_csv(pairs_path, index=False)
    players_df.to_csv(players_path, index=False)

    # also a parquet mirror of pairs for fast loading
    pairs_df.to_parquet(out_dir / "scoutbench_taskb_pairs.parquet", index=False)

    manifest = {
        "name": "ScoutBench Task B ground-truth release",
        "task": "external transfer-anchored player-similarity retrieval",
        "license": "CC0 (derived solely from transfermarkt CC0 fields: id, name, sub-position)",
        "provenance": "see DATA.md -- labels are realized like-for-like replacement transfers",
        "n_directed_pairs": int(len(pairs_df)),
        "n_players_in_graph": int(len(players_df)),
        "n_query_players_scored": int(players_df["is_query"].sum()),
        "n_in_gallery": int(players_df["in_gallery"].sum()),
        "note": ("Players with in_gallery=False are part of the label graph but absent "
                 "from the StatsBomb-derived gallery, so they are NOT scored; the harness "
                 "scores the is_query=True players (gallery members with >=1 relevant)."),
        "files": {
            "pairs_csv": pairs_path.name,
            "pairs_parquet": "scoutbench_taskb_pairs.parquet",
            "players_csv": players_path.name,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"pairs        -> {pairs_path}  ({len(pairs_df)} directed rows)")
    print(f"players      -> {players_path}  ({len(players_df)} players, "
          f"{int(players_df['is_query'].sum())} scored queries, "
          f"{int(players_df['in_gallery'].sum())} in gallery)")
    print(f"manifest     -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
