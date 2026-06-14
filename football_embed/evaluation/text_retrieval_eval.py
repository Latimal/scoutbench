#!/usr/bin/env python3
"""Evaluate text-card cross-modal alignment quality.

Loads a trained TextBranch checkpoint and a player-card gallery, projects cards
via PlayerCardProjector, then measures how well text descriptions retrieve the
correct player from the player-card gallery. Also runs natural language search
demos, zone-match evaluation, and optionally generates a UMAP visualization.

Evaluations:
1. Cross-modal retrieval accuracy: Recall@1, Recall@5, Recall@10
2. Alignment score: cosine sim distribution for paired text-card embeddings
3. Natural language search: top-5 players for hardcoded scouting queries
4. Zone match: whether NL queries retrieve players from expected positional zones
5. UMAP visualization: 2D scatter of text + card embeddings colored by position

Usage:
    .venv/bin/python3 -m football_embed.evaluation.text_retrieval_eval \
        --text-model-path checkpoints/text_branch/ \
        --gallery-path data/processed/embeddings/player_card_gallery.parquet \
        --val-path data/processed/text/text_card_pairs_val.parquet \
        --player-lookup data/processed/players_lookup.parquet \
        --output-dir evaluation/text_retrieval/ \
        --skip-umap

    .venv/bin/python3 -m football_embed.evaluation.text_retrieval_eval \
        --text-model-path checkpoints/text_branch/ \
        --gallery-path data/processed/embeddings/player_card_gallery.parquet \
        --val-path data/processed/text/text_card_pairs_val.parquet \
        --player-lookup data/processed/players_lookup.parquet \
        --output-dir evaluation/text_retrieval/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from football_embed.model.text_branch import TextBranch, PlayerCardProjector


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "fast winger with many dribbles and crosses",
    "defensive midfielder who controls possession and reads the game",
    "tall centre-back dominant in aerial duels",
    "creative playmaker with excellent passing vision",
    "prolific striker who scores goals consistently",
    "versatile full-back who attacks and defends",
    "shot-stopping goalkeeper with good distribution",
    "energetic box-to-box midfielder who covers ground",
]

ZONE_QUERIES = [
    ("fast winger with many dribbles and crosses", {"Winger", "Full-back"}),
    ("defensive midfielder who controls possession", {"Defensive midfielder", "Central midfielder"}),
    ("tall centre-back dominant in aerial duels", {"Centre-back"}),
    ("creative playmaker with excellent passing", {"Attacking midfielder", "Central midfielder"}),
    ("prolific striker who scores goals", {"Attacking midfielder"}),
    ("shot-stopping goalkeeper", {"Goalkeeper"}),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_card_gallery(
    gallery_path: Path,
    projector: PlayerCardProjector,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Load player card gallery, project to 256-dim via card projector.

    Returns:
        projected_embs: (N, 256) float32, L2-normalized projected embeddings.
        player_names: (N,) player name array.
    """
    df = pd.read_parquet(gallery_path)
    card_cols = [c for c in df.columns if c.startswith("card_")]
    cards = torch.tensor(df[card_cols].values, dtype=torch.float32).to(device)

    projector.eval()
    with torch.no_grad():
        projected = projector(cards).cpu().numpy()

    player_names = df["player_name"].values
    return projected, player_names


def _load_val_pairs(
    path: Path,
    projector: PlayerCardProjector,
    device: str = "cpu",
) -> tuple[list[str], list[str], np.ndarray]:
    """Load validation text-card pairs, project cards to 256-dim.

    Returns:
        texts: list of text descriptions.
        names: list of player names.
        projected_cards: (N, 256) float32 projected card embeddings.
    """
    df = pd.read_parquet(path)
    texts = df["text_content"].tolist()
    names = df["player_name"].tolist()

    card_cols = [c for c in df.columns if c.startswith("card_")]
    cards = torch.tensor(df[card_cols].values, dtype=torch.float32).to(device)

    projector.eval()
    with torch.no_grad():
        projected = projector(cards).cpu().numpy()

    return texts, names, projected


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a and rows of b.

    Both a and b are assumed L2-normalized, so cosine sim = dot product.

    Args:
        a: (M, D) query embeddings.
        b: (N, D) gallery embeddings.

    Returns:
        (M, N) similarity matrix.
    """
    return a @ b.T


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def eval_cross_modal_retrieval(
    text_embs: np.ndarray,
    val_names: list[str],
    gallery_embs: np.ndarray,
    gallery_names: np.ndarray,
) -> dict:
    """Cross-modal retrieval: text query -> event embedding gallery.

    For each val text embedding, rank all gallery players by cosine similarity
    and check if the correct player appears in top-K.

    Returns dict with recall@1, recall@5, recall@10 and total query count.
    """
    sim_matrix = _cosine_sim_matrix(text_embs, gallery_embs)  # (n_val, n_gallery)
    ranked_indices = np.argsort(-sim_matrix, axis=1)  # descending

    gallery_names_arr = np.asarray(gallery_names)
    hits = {1: 0, 5: 0, 10: 0}
    n_queries = len(val_names)

    if n_queries == 0:
        return {"n_queries": 0, "n_gallery": len(gallery_names),
                "recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0}

    for i, gt_name in enumerate(val_names):
        ranked_names = gallery_names_arr[ranked_indices[i]]
        for k in hits:
            if gt_name in ranked_names[:k]:
                hits[k] += 1

    results = {
        "n_queries": n_queries,
        "n_gallery": len(gallery_names),
        "recall_at_1": hits[1] / n_queries,
        "recall_at_5": hits[5] / n_queries,
        "recall_at_10": hits[10] / n_queries,
    }
    return results


def eval_alignment_score(
    text_embs: np.ndarray,
    event_embs: np.ndarray,
) -> dict:
    """Pairwise cosine similarity between matched text and event embeddings.

    text_embs[i] and event_embs[i] correspond to the same player.
    """
    # Row-wise dot product (both L2-normalized)
    sims = np.sum(text_embs * event_embs, axis=1)

    results = {
        "n_pairs": len(sims),
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "std": float(np.std(sims)),
        "pct_above_0.5": float(np.mean(sims > 0.5)),
    }
    return results


def eval_nl_search(
    model: TextBranch,
    gallery_embs: np.ndarray,
    gallery_names: np.ndarray,
    queries: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Natural language search demo: encode queries and retrieve top-K players."""
    query_embs = model.encode(queries).numpy()  # (n_queries, 256)
    sim_matrix = _cosine_sim_matrix(query_embs, gallery_embs)
    ranked_indices = np.argsort(-sim_matrix, axis=1)

    results = []
    for i, query in enumerate(queries):
        top_indices = ranked_indices[i, :top_k]
        top_names = gallery_names[top_indices].tolist()
        top_sims = sim_matrix[i, top_indices].tolist()
        results.append({
            "query": query,
            "results": [
                {"player": name, "similarity": round(sim, 4)}
                for name, sim in zip(top_names, top_sims)
            ],
        })
    return results


def eval_zone_match(
    model: TextBranch,
    gallery_embs: np.ndarray,
    gallery_names: np.ndarray,
    gallery_path: Path,
    queries_with_zones: list[tuple[str, set[str]]],
    top_k: int = 10,
) -> dict:
    """Evaluate whether NL queries retrieve players from the expected zone.

    Args:
        queries_with_zones: list of (query_text, set_of_expected_zones).
    """
    # Load zone labels from gallery
    df = pd.read_parquet(gallery_path)
    name_to_zone = dict(zip(df["player_name"], df["zone"]))

    query_texts = [q for q, _ in queries_with_zones]
    query_embs = model.encode(query_texts).numpy()
    sim_matrix = _cosine_sim_matrix(query_embs, gallery_embs)
    ranked_indices = np.argsort(-sim_matrix, axis=1)

    results = []
    total_match = 0
    for i, (query, expected_zones) in enumerate(queries_with_zones):
        top_indices = ranked_indices[i, :top_k]
        top_names = gallery_names[top_indices]
        top_zones = [name_to_zone.get(n, "Unknown") for n in top_names]
        matches = sum(1 for z in top_zones if z in expected_zones)
        match_rate = matches / top_k
        total_match += match_rate
        results.append({
            "query": query,
            "expected_zones": sorted(expected_zones),
            "top_zones": top_zones,
            "zone_match_rate": round(match_rate, 4),
        })

    avg_match = total_match / len(queries_with_zones) if queries_with_zones else 0
    return {"avg_zone_match_rate": round(avg_match, 4), "details": results}


def make_umap_plot(
    text_embs: np.ndarray,
    card_embs: np.ndarray,
    val_names: list[str],
    player_lookup_path: Path,
    output_path: Path,
):
    """Generate UMAP 2D scatter of text + card embeddings colored by position.

    Each player gets two points (text and card) connected by a thin line.
    Points are colored by playing position from the lookup table.
    """
    try:
        import umap
    except ImportError:
        print("umap-learn not installed, skipping UMAP visualization.")
        print("Install with: pip install umap-learn")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load position data
    lookup_df = pd.read_parquet(player_lookup_path)
    name_to_position = dict(
        zip(lookup_df["player_name"], lookup_df["starting_position_name"])
    )

    positions = [name_to_position.get(n, "Unknown") for n in val_names]
    unique_positions = sorted(set(positions))
    pos_to_idx = {p: i for i, p in enumerate(unique_positions)}
    pos_indices = np.array([pos_to_idx[p] for p in positions])

    # Stack all embeddings: text first, then card
    n = len(val_names)
    all_embs = np.vstack([text_embs, card_embs])  # (2N, 256)

    print(f"Running UMAP on {all_embs.shape[0]} points...")
    t0 = time.time()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(all_embs)  # (2N, 2)
    print(f"UMAP completed in {time.time() - t0:.1f}s")

    text_coords = coords[:n]
    card_coords = coords[n:]

    cmap = plt.cm.get_cmap("tab20", len(unique_positions))

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw connecting lines (same player text <-> card)
    for i in range(n):
        ax.plot(
            [text_coords[i, 0], card_coords[i, 0]],
            [text_coords[i, 1], card_coords[i, 1]],
            color="gray", alpha=0.15, linewidth=0.5, zorder=1,
        )

    # Plot text embeddings (circles)
    for pidx, pname in enumerate(unique_positions):
        mask = pos_indices == pidx
        ax.scatter(
            text_coords[mask, 0], text_coords[mask, 1],
            c=[cmap(pidx)], label=f"{pname} (text)", marker="o",
            s=20, alpha=0.7, zorder=2,
        )

    # Plot card embeddings (triangles)
    for pidx, pname in enumerate(unique_positions):
        mask = pos_indices == pidx
        ax.scatter(
            card_coords[mask, 0], card_coords[mask, 1],
            c=[cmap(pidx)], label=f"{pname} (card)", marker="^",
            s=20, alpha=0.7, zorder=2,
        )

    ax.set_title("Text vs Card Embeddings (UMAP 2D)", fontsize=14)
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, ncol=1,
        markerscale=1.5,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate text-card cross-modal alignment quality",
    )
    parser.add_argument(
        "--text-model-path", type=str, required=True,
        help="Path to TextBranch checkpoint directory (config.json + trainable_weights.pt + card_projector.pt)",
    )
    parser.add_argument(
        "--gallery-path", type=str, required=True,
        help="Path to player_card_gallery.parquet (gallery of all player cards)",
    )
    parser.add_argument(
        "--val-path", type=str, required=True,
        help="Path to text_card_pairs_val.parquet (validation text-card pairs)",
    )
    parser.add_argument(
        "--player-lookup", type=str, default=None,
        help="Path to players_lookup.parquet (needed for UMAP position coloring)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write results.json and UMAP plot",
    )
    parser.add_argument(
        "--skip-umap", action="store_true",
        help="Skip UMAP visualization (faster, no umap-learn dependency)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for TextBranch inference (default: cpu)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for text encoding (default: 32)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load TextBranch + PlayerCardProjector
    # ------------------------------------------------------------------
    print(f"Loading TextBranch from {args.text_model_path}...")
    model = TextBranch.load(args.text_model_path, device=args.device)
    model.eval()
    print(f"  Trainable params: {model.get_trainable_params():,}")

    print(f"Loading PlayerCardProjector from {args.text_model_path}...")
    projector = PlayerCardProjector.load(args.text_model_path, device=args.device)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading player card gallery...")
    gallery_embs, gallery_names = _load_card_gallery(
        Path(args.gallery_path), projector, device=args.device,
    )
    print(f"  Gallery: {gallery_embs.shape[0]} players, {gallery_embs.shape[1]}-dim")

    print("Loading validation pairs...")
    val_texts, val_names, val_card_embs = _load_val_pairs(
        Path(args.val_path), projector, device=args.device,
    )
    print(f"  Val set: {len(val_texts)} text-card pairs")

    # ------------------------------------------------------------------
    # Encode val texts
    # ------------------------------------------------------------------
    print("Encoding validation texts...")
    t0 = time.time()
    val_text_embs = model.encode(val_texts, batch_size=args.batch_size).numpy()
    encode_time = time.time() - t0
    print(f"  Encoded {len(val_texts)} texts in {encode_time:.1f}s")

    # ------------------------------------------------------------------
    # 1. Cross-modal retrieval
    # ------------------------------------------------------------------
    print("\n--- Cross-modal Retrieval ---")
    retrieval = eval_cross_modal_retrieval(
        val_text_embs, val_names, gallery_embs, gallery_names,
    )
    print(f"  Gallery size: {retrieval['n_gallery']}")
    print(f"  Queries:      {retrieval['n_queries']}")
    print(f"  Recall@1:     {retrieval['recall_at_1']:.4f}")
    print(f"  Recall@5:     {retrieval['recall_at_5']:.4f}")
    print(f"  Recall@10:    {retrieval['recall_at_10']:.4f}")

    # ------------------------------------------------------------------
    # 2. Alignment score
    # ------------------------------------------------------------------
    print("\n--- Alignment Score (paired text <-> card) ---")
    alignment = eval_alignment_score(val_text_embs, val_card_embs)
    print(f"  Pairs:          {alignment['n_pairs']}")
    print(f"  Mean cosine:    {alignment['mean']:.4f}")
    print(f"  Median cosine:  {alignment['median']:.4f}")
    print(f"  Min cosine:     {alignment['min']:.4f}")
    print(f"  Max cosine:     {alignment['max']:.4f}")
    print(f"  Std:            {alignment['std']:.4f}")
    print(f"  % above 0.5:    {alignment['pct_above_0.5']:.2%}")
    target_met = alignment["mean"] > 0.5
    print(f"  Target (mean > 0.5): {'PASS' if target_met else 'FAIL'}")

    # ------------------------------------------------------------------
    # 3. Natural language search demo
    # ------------------------------------------------------------------
    print("\n--- Natural Language Search Demo ---")
    nl_results = eval_nl_search(
        model, gallery_embs, gallery_names, DEMO_QUERIES, top_k=5,
    )
    for entry in nl_results:
        print(f"\n  Query: \"{entry['query']}\"")
        for rank, r in enumerate(entry["results"], 1):
            print(f"    {rank}. {r['player']} (sim={r['similarity']:.4f})")

    # ------------------------------------------------------------------
    # 4. Zone match evaluation
    # ------------------------------------------------------------------
    print("\n--- Zone Match Evaluation ---")
    zone_results = eval_zone_match(
        model, gallery_embs, gallery_names,
        Path(args.gallery_path), ZONE_QUERIES, top_k=10,
    )
    print(f"  Avg zone match rate: {zone_results['avg_zone_match_rate']:.2%}")
    for detail in zone_results["details"]:
        print(f"  Query: \"{detail['query']}\"")
        print(f"    Expected: {detail['expected_zones']}, Got: {detail['top_zones']}")
        print(f"    Match rate: {detail['zone_match_rate']:.2%}")

    # ------------------------------------------------------------------
    # 5. UMAP visualization
    # ------------------------------------------------------------------
    umap_generated = False
    if not args.skip_umap:
        if args.player_lookup is None:
            print("\n--player-lookup not provided, skipping UMAP.")
        else:
            print("\n--- UMAP Visualization ---")
            umap_path = output_dir / "umap_text_card.png"
            make_umap_plot(
                val_text_embs,
                val_card_embs,
                val_names,
                Path(args.player_lookup),
                umap_path,
            )
            umap_generated = True

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "cross_modal_retrieval": retrieval,
        "alignment_score": alignment,
        "nl_search_demo": nl_results,
        "zone_match": zone_results,
        "meta": {
            "text_model_path": args.text_model_path,
            "gallery_path": args.gallery_path,
            "val_path": args.val_path,
            "n_gallery": int(gallery_embs.shape[0]),
            "n_val": len(val_texts),
            "encode_time_s": round(encode_time, 2),
            "device": args.device,
            "umap_generated": umap_generated,
        },
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
