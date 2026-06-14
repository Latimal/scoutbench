#!/usr/bin/env python3
"""football-embed-Bench runner.

Runs the evaluation suite against a trained text-card model:
- role: Role classification accuracy (linear probe on projected card embeddings)
- temporal: Temporal consistency (same player across seasons)
- ndcg: Natural language retrieval nDCG@10
- match_outcome: Match outcome prediction from team-averaged cards

Usage:
    .venv/bin/python3 -m football_embed.evaluation.bench \
        --text-model-path checkpoints/text_branch/best/ \
        --gallery-path data/processed/text/player_card_gallery.parquet \
        --spadl-path data/processed/spadl_unified.parquet \
        --games-path data/processed/statsbomb/games.parquet \
        --norm-stats-path data/processed/text/card_normalization.json \
        --output-dir evaluation/bench_v1/

    # Run specific metrics:
    .venv/bin/python3 -m football_embed.evaluation.bench \
        --text-model-path checkpoints/text_branch/best/ \
        --gallery-path data/processed/text/player_card_gallery.parquet \
        --spadl-path data/processed/spadl_unified.parquet \
        --games-path data/processed/statsbomb/games.parquet \
        --norm-stats-path data/processed/text/card_normalization.json \
        --output-dir evaluation/bench_v1/ \
        --metrics role,ndcg
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from football_embed.evaluation.metrics import (
    archetype_classification_accuracy,
    match_outcome_prediction,
    nlq_ndcg_at_k,
    nlq_ndcg_graded_at_k,
    role_classification_accuracy,
    temporal_consistency,
    within_zone_retrieval_accuracy,
    zca_whiten_fit,
    zca_whiten_transform,
)
from football_embed.model.text_branch import PlayerCardProjector, TextBranch


# ---------------------------------------------------------------------------
# NL queries with expected zones (superset of text_retrieval_eval.py ZONE_QUERIES)
# ---------------------------------------------------------------------------

NL_QUERIES = [
    ("fast winger with many dribbles and crosses", {"Winger", "Full-back"}),
    ("defensive midfielder who controls possession", {"Defensive midfielder", "Central midfielder"}),
    ("tall centre-back dominant in aerial duels", {"Centre-back"}),
    ("creative playmaker with excellent passing", {"Attacking midfielder", "Central midfielder"}),
    ("prolific striker who scores goals", {"Attacking midfielder"}),
    ("shot-stopping goalkeeper", {"Goalkeeper"}),
    ("box-to-box midfielder with high work rate", {"Central midfielder", "Defensive midfielder"}),
    ("physical centre-forward who scores and wins aerial duels", {"Attacking midfielder"}),
    ("modern full-back who defends and supports attacks", {"Full-back"}),
]

ARCHETYPE_QUERIES = [
    ("deep-lying regista who dictates play with precise passing", {"Defensive midfielder"}, {"Regista"}),
    ("aggressive stopper centre-back who wins tackles", {"Centre-back"}, {"Stopper CB"}),
    ("overlapping full-back who delivers crosses", {"Full-back"}, {"Overlapping FB", "Wing-back"}),
    ("inverted winger who cuts inside to shoot", {"Winger"}, {"Inside forward", "Inverted winger"}),
    ("classic number 10 playmaker who creates chances", {"Attacking midfielder"}, {"Classic 10"}),
    ("box-to-box midfielder who covers every blade of grass", {"Central midfielder", "Defensive midfielder"}, {"Box-to-box CM", "Box-to-box DM"}),
    ("goal-scoring winger who finishes like a striker", {"Winger"}, {"Goal-scoring winger", "Inside forward"}),
    ("sweeper-keeper comfortable with the ball at his feet", {"Goalkeeper"}, {"Sweeper-keeper", "Ball-playing GK"}),
]

# 50 queries with 3-level graded relevance: (text, relevant_zones, relevant_archetypes)
NL_QUERIES_V2 = [
    # --- Original 9 queries (zone-only, empty archetype set for graded scoring) ---
    ("fast winger with many dribbles and crosses", {"Winger", "Full-back"}, set()),
    ("defensive midfielder who controls possession", {"Defensive midfielder", "Central midfielder"}, set()),
    ("tall centre-back dominant in aerial duels", {"Centre-back"}, set()),
    ("creative playmaker with excellent passing", {"Attacking midfielder", "Central midfielder"}, set()),
    ("prolific striker who scores goals", {"Attacking midfielder"}, set()),
    ("shot-stopping goalkeeper", {"Goalkeeper"}, set()),
    ("box-to-box midfielder with high work rate", {"Central midfielder", "Defensive midfielder"}, set()),
    ("physical centre-forward who scores and wins aerial duels", {"Attacking midfielder"}, set()),
    ("modern full-back who defends and supports attacks", {"Full-back"}, set()),

    # --- 26 archetype-specific queries ---
    ("classic number 10 who creates chances and plays through balls", {"Attacking midfielder"}, {"Classic 10"}),
    ("false nine who drops deep and links midfield play", {"Attacking midfielder"}, {"False 9"}),
    ("second striker who makes runs behind the defence", {"Attacking midfielder"}, {"Shadow striker"}),
    ("wide playmaker who drifts inside to create from the half-space", {"Attacking midfielder"}, {"Wide playmaker"}),
    ("all-action box-to-box midfielder who covers every blade of grass", {"Central midfielder"}, {"Box-to-box CM"}),
    ("deep-lying playmaker who dictates tempo from central midfield", {"Central midfielder"}, {"Deep playmaker"}),
    ("energetic mezzala who drives forward from midfield", {"Central midfielder"}, {"Mezzala"}),
    ("disciplined defensive midfielder in central midfield", {"Central midfielder"}, {"Defensive CM"}),
    ("ball-playing centre-back comfortable on the ball", {"Centre-back"}, {"Ball-playing CB"}),
    ("commanding centre-back dominant in the air", {"Centre-back"}, {"Commanding CB"}),
    ("aggressive stopper who wins tackles and duels", {"Centre-back"}, {"Stopper CB"}),
    ("wide centre-back who covers the flanks", {"Centre-back"}, {"Wide CB"}),
    ("anchor defensive midfielder who shields the back line", {"Defensive midfielder"}, {"Anchor DM"}),
    ("deep-lying regista who sprays long passes", {"Defensive midfielder"}, {"Regista"}),
    ("box-to-box defensive midfielder who tackles and drives forward", {"Defensive midfielder"}, {"Box-to-box DM"}),
    ("defensive full-back who prioritises stopping crosses", {"Full-back"}, {"Defensive FB"}),
    ("inverted full-back who tucks into midfield", {"Full-back"}, {"Inverted FB"}),
    ("overlapping full-back who delivers crosses from the byline", {"Full-back"}, {"Overlapping FB"}),
    ("wing-back who bombs forward and tracks back", {"Full-back"}, {"Wing-back"}),
    ("ball-playing goalkeeper comfortable with distribution", {"Goalkeeper"}, {"Ball-playing GK"}),
    ("shot-stopping goalkeeper with quick reflexes", {"Goalkeeper"}, {"Shot-stopper"}),
    ("sweeper-keeper who comes off the line aggressively", {"Goalkeeper"}, {"Sweeper-keeper"}),
    ("traditional winger who hugs the touchline and crosses", {"Winger"}, {"Traditional winger"}),
    ("inside forward who cuts in from the wing to shoot", {"Winger"}, {"Inside forward"}),
    ("inverted winger who plays on the opposite flank", {"Winger"}, {"Inverted winger"}),
    ("goal-scoring winger who finishes like a striker", {"Winger"}, {"Goal-scoring winger"}),

    # --- 15 cross-cutting style queries ---
    ("press-resistant midfielder who retains possession under pressure", {"Central midfielder", "Defensive midfielder"}, {"Deep playmaker", "Regista", "Mezzala"}),
    ("progressive ball carrier who drives past opponents", {"Central midfielder", "Winger", "Full-back"}, {"Mezzala", "Box-to-box CM", "Inside forward", "Wing-back"}),
    ("set-piece specialist who delivers dangerous free kicks and corners", {"Central midfielder", "Attacking midfielder", "Winger"}, {"Classic 10", "Wide playmaker", "Deep playmaker"}),
    ("aggressive presser who wins the ball high up the pitch", {"Attacking midfielder", "Central midfielder", "Winger"}, {"Shadow striker", "Box-to-box CM", "Mezzala", "Inside forward"}),
    ("long-range passer who switches play with diagonal balls", {"Centre-back", "Defensive midfielder", "Central midfielder"}, {"Ball-playing CB", "Regista", "Deep playmaker"}),
    ("penalty box goal scorer with clinical finishing", {"Attacking midfielder", "Winger"}, {"False 9", "Shadow striker", "Inside forward", "Goal-scoring winger"}),
    ("dribbler who takes on defenders in one-on-one situations", {"Winger", "Attacking midfielder"}, {"Traditional winger", "Inside forward", "Inverted winger", "Classic 10"}),
    ("aerial threat who wins headers from crosses and set pieces", {"Centre-back", "Attacking midfielder"}, {"Commanding CB", "Stopper CB", "Shadow striker", "False 9"}),
    ("defensive midfielder who reads the game and intercepts passes", {"Defensive midfielder", "Central midfielder"}, {"Anchor DM", "Defensive CM"}),
    ("creative wide player who provides assists from the flank", {"Winger", "Full-back"}, {"Traditional winger", "Goal-scoring winger", "Overlapping FB", "Wing-back"}),
    ("versatile defender comfortable at centre-back or full-back", {"Centre-back", "Full-back"}, {"Wide CB", "Defensive FB", "Ball-playing CB"}),
    ("goalkeeper who starts attacks with accurate distribution", {"Goalkeeper"}, {"Ball-playing GK", "Sweeper-keeper"}),
    ("midfield destroyer who breaks up opposition attacks", {"Defensive midfielder", "Central midfielder"}, {"Anchor DM", "Defensive CM", "Box-to-box DM"}),
    ("counter-attacking winger with pace and direct running", {"Winger", "Attacking midfielder"}, {"Traditional winger", "Inside forward", "Goal-scoring winger", "Shadow striker"}),
    ("full-back who provides width and overlapping runs", {"Full-back"}, {"Overlapping FB", "Wing-back", "Defensive FB"}),
]

ALL_METRICS = {"role", "temporal", "ndcg", "ndcg_v2", "match_outcome", "within_zone", "archetype"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_gallery(
    gallery_path: Path,
    projector: PlayerCardProjector,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Load gallery, project cards via projector.

    Returns:
        projected_embs: (N, 256) L2-normalized projected embeddings.
        zones: (N,) zone label strings.
        player_names: (N,) player name strings.
        archetype_names: (N,) archetype name strings, or None.
        raw_cards: (N, D) raw L2-normed card vectors (pre-projection).
    """
    df = pd.read_parquet(gallery_path)
    card_cols = [c for c in df.columns if c.startswith("card_")]
    raw_cards_np = df[card_cols].values.astype(np.float32)
    cards = torch.tensor(raw_cards_np, dtype=torch.float32).to(device)

    projector.eval()
    with torch.no_grad():
        projected = projector(cards).cpu().numpy()

    zones = df["zone"].values
    names = df["player_name"].values
    arch_names = df["archetype_name"].values if "archetype_name" in df.columns else None
    return projected, zones, names, arch_names, raw_cards_np


def _print_table(title: str, rows: list[tuple[str, str]]):
    """Print a two-column summary table."""
    key_width = max(len(r[0]) for r in rows)
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for key, val in rows:
        print(f"  {key:<{key_width}}  {val}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run football-embed-Bench evaluation suite.",
    )
    parser.add_argument(
        "--text-model-path", type=str, required=True,
        help="Path to TextBranch checkpoint dir (config.json + trainable_weights.pt + card_projector.pt)",
    )
    parser.add_argument(
        "--gallery-path", type=str, required=True,
        help="Path to player_card_gallery.parquet",
    )
    parser.add_argument(
        "--spadl-path", type=str, default="data/processed/spadl_unified.parquet",
        help="Path to spadl_unified.parquet (default: data/processed/spadl_unified.parquet)",
    )
    parser.add_argument(
        "--games-path", type=str, default="data/processed/statsbomb/games.parquet",
        help="Path to games.parquet (default: data/processed/statsbomb/games.parquet)",
    )
    parser.add_argument(
        "--norm-stats-path", type=str, default="data/processed/text/card_normalization.json",
        help="Path to card_normalization.json (default: data/processed/text/card_normalization.json)",
    )
    parser.add_argument(
        "--season-cards-path", type=str, default=None,
        help="Optional path to pre-computed season_cards.parquet (from "
             "build_player_cards.py --season-first). When provided, the "
             "temporal metric reads these directly instead of recomputing "
             "per-season cards at eval time — fixes the v11 train/test "
             "mismatch that kept temporal stuck at ~0.71.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write bench_results.json",
    )
    parser.add_argument(
        "--metrics", type=str, default="all",
        help="Comma-separated metrics to run: role,temporal,ndcg,match_outcome (default: all)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for TextBranch inference (default: cpu)",
    )
    parser.add_argument(
        "--whiten", action="store_true", default=False,
        help="Apply ZCA whitening to gallery embeddings (post-hoc isotropization)",
    )
    parser.add_argument(
        "--benchmark-version", type=str, choices=["v1", "v2"], default="v2",
        help="Benchmark version: v1 = original 9 queries, v2 = v1 + 50 graded queries (default: v2)",
    )
    parser.add_argument(
        "--no-query-prefix", action="store_true", default=False,
        help="Disable 'search_query: ' prefix on NL queries (use when prefix collapses embeddings)",
    )
    parser.add_argument(
        "--enable-rerank", action="store_true", default=False,
        help="Enable hybrid reranking (stat-based skill scoring on top of dense retrieval)",
    )
    parser.add_argument(
        "--rerank-alpha", type=float, default=0.7,
        help="Blend weight: alpha * dense + (1-alpha) * skill (default: 0.7)",
    )
    parser.add_argument(
        "--rerank-top-k", type=int, default=50,
        help="Candidates to consider for reranking (default: 50)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse requested metrics
    if args.metrics == "all":
        requested = set(ALL_METRICS)
        if args.benchmark_version == "v1":
            requested.discard("ndcg_v2")
    else:
        requested = {m.strip() for m in args.metrics.split(",")}
        unknown = requested - ALL_METRICS
        if unknown:
            parser.error(f"Unknown metrics: {unknown}. Choose from: {sorted(ALL_METRICS)}")
        # When v1, don't auto-include ndcg_v2 even if explicitly requested is fine
        # but if user explicitly requested ndcg_v2 with v1, allow it

    # Load feature names from normalization stats (tracks actual post-filter features)
    with open(args.norm_stats_path) as _f:
        _norm = json.load(_f)
    feature_names = _norm.get("feature_names", [])
    if not feature_names:
        # Fallback: import static list (pre-variance-filter)
        from football_embed.training.build_player_cards import FEATURE_NAMES
        feature_names = FEATURE_NAMES

    print(f"football-embed-Bench")
    print(f"  Metrics: {sorted(requested)}")
    print(f"  Features: {len(feature_names)}-dim cards")
    print(f"  Output:  {output_dir}")

    # ------------------------------------------------------------------
    # Load model + projector (needed for role and ndcg)
    # ------------------------------------------------------------------
    need_model = bool(requested & {"role", "ndcg", "ndcg_v2", "within_zone", "archetype"})

    model = None
    projector = None
    gallery_embs = None
    gallery_zones = None
    gallery_names = None
    gallery_archetypes = None
    gallery_raw_cards = None

    if need_model:
        print(f"\nLoading TextBranch from {args.text_model_path}...")
        model = TextBranch.load(args.text_model_path, device=args.device)
        model.eval()
        print(f"  Trainable params: {model.get_trainable_params():,}")

        print(f"Loading PlayerCardProjector from {args.text_model_path}...")
        projector = PlayerCardProjector.load(args.text_model_path, device=args.device)

        print(f"Loading gallery from {args.gallery_path}...")
        gallery_embs, gallery_zones, gallery_names, gallery_archetypes, gallery_raw_cards = _load_gallery(
            Path(args.gallery_path), projector, device=args.device,
        )
        print(f"  Gallery: {gallery_embs.shape[0]} players, {gallery_embs.shape[1]}-dim")

        whiten_W = None
        whiten_mean = None
        if args.whiten:
            print("  Applying ZCA whitening to gallery embeddings...")
            gallery_embs, whiten_W, whiten_mean = zca_whiten_fit(gallery_embs)
            print("  Whitening applied (transform saved for query-side).")

    # Query encoding setup (shared by ndcg and ndcg_v2)
    use_prefix = getattr(args, "no_query_prefix", False) is False
    enc_fn = model.encode_queries if (need_model and use_prefix) else (model.encode if need_model else None)

    # ------------------------------------------------------------------
    # Run metrics
    # ------------------------------------------------------------------
    results: dict = {}
    summary_rows: list[tuple[str, str]] = []
    t_total = time.time()

    # --- role ---
    if "role" in requested:
        print("\n--- Role Classification ---")
        t0 = time.time()
        role_result = role_classification_accuracy(gallery_embs, gallery_zones)
        dt = time.time() - t0

        results["role_classification"] = role_result
        summary_rows.append(("Role accuracy", f"{role_result['accuracy']:.4f}"))
        summary_rows.append(("Role macro F1", f"{role_result['macro_f1']:.4f}"))

        print(f"  Accuracy:  {role_result['accuracy']:.4f}")
        print(f"  Macro F1:  {role_result['macro_f1']:.4f}")
        print(f"  Classes:   {role_result['n_classes']}")
        print(f"  Train/Test: {role_result['n_train']} / {role_result['n_test']}")
        print(f"  Per-class accuracy:")
        for cls, acc in sorted(role_result["per_class"].items()):
            print(f"    {cls:25s} {acc:.4f}")
        print(f"  Time: {dt:.1f}s")

    # --- temporal ---
    if "temporal" in requested:
        print("\n--- Temporal Consistency ---")
        t0 = time.time()
        temp_result = temporal_consistency(
            args.spadl_path,
            args.norm_stats_path,
            feature_names,
            season_cards_path=args.season_cards_path,
        )
        dt = time.time() - t0

        results["temporal_consistency"] = temp_result
        summary_rows.append(("Temporal mean cosine", f"{temp_result['mean_cosine']:.4f}"))
        summary_rows.append(("Temporal pct > 0.8", f"{temp_result['pct_above_0.8']:.2%}"))

        print(f"  Mean cosine:  {temp_result['mean_cosine']:.4f}")
        print(f"  Median cosine: {temp_result['median_cosine']:.4f}")
        print(f"  % above 0.8:  {temp_result['pct_above_0.8']:.2%}")
        print(f"  Players:      {temp_result['n_players']}")
        print(f"  Pairs:        {temp_result['n_pairs']}")
        target_met = temp_result["mean_cosine"] > 0.8
        print(f"  Target (mean > 0.8): {'PASS' if target_met else 'FAIL'}")
        print(f"  Time: {dt:.1f}s")

    # --- ndcg ---
    if "ndcg" in requested:
        print("\n--- NL Retrieval nDCG@10 ---")
        t0 = time.time()

        query_texts = [q for q, _ in NL_QUERIES]
        query_relevant_zones = [z for _, z in NL_QUERIES]

        print(f"  Encoding {len(query_texts)} queries (prefix={use_prefix})...")
        query_embs = enc_fn(query_texts).numpy()
        if whiten_W is not None:
            query_embs = zca_whiten_transform(query_embs, whiten_W, whiten_mean)

        ndcg_result = nlq_ndcg_at_k(
            query_embs, gallery_embs, gallery_zones, query_relevant_zones, k=10,
        )
        dt = time.time() - t0

        results["nlq_ndcg"] = ndcg_result
        results["nlq_ndcg"]["queries"] = [
            {"text": q, "relevant_zones": sorted(z), "ndcg": pq["ndcg"]}
            for (q, z), pq in zip(NL_QUERIES, ndcg_result["per_query"])
        ]
        summary_rows.append(("nDCG@10 (mean)", f"{ndcg_result['mean_ndcg']:.4f}"))

        print(f"  Mean nDCG@10: {ndcg_result['mean_ndcg']:.4f}")
        for entry in results["nlq_ndcg"]["queries"]:
            print(f"    \"{entry['text'][:50]}...\"  nDCG={entry['ndcg']:.4f}")
        print(f"  Time: {dt:.1f}s")

    # --- ndcg_v2 (graded) ---
    if "ndcg_v2" in requested and gallery_archetypes is None:
        print("\n--- NL Retrieval nDCG@10 (v2) ---")
        print("  SKIPPED: gallery has no archetype_name column")
    elif "ndcg_v2" in requested:
        print("\n--- NL Retrieval nDCG@10 (v2 graded, 50 queries) ---")
        t0 = time.time()

        v2_texts = [q for q, _, _ in NL_QUERIES_V2]
        v2_zones = [z for _, z, _ in NL_QUERIES_V2]
        v2_archs = [a for _, _, a in NL_QUERIES_V2]

        print(f"  Encoding {len(v2_texts)} queries (prefix={use_prefix})...")
        v2_embs = enc_fn(v2_texts).numpy()
        if whiten_W is not None:
            v2_embs = zca_whiten_transform(v2_embs, whiten_W, whiten_mean)

        v2_result = nlq_ndcg_graded_at_k(
            v2_embs, gallery_embs, gallery_zones, gallery_archetypes,
            v2_zones, v2_archs, k=10,
        )
        dt = time.time() - t0

        results["nlq_ndcg_v2"] = v2_result
        results["nlq_ndcg_v2"]["queries"] = [
            {"text": q, "relevant_zones": sorted(z), "relevant_archetypes": sorted(a), "ndcg": pq["ndcg"]}
            for (q, z, a), pq in zip(NL_QUERIES_V2, v2_result["per_query"])
        ]
        summary_rows.append(("nDCG@10 v2 graded (mean)", f"{v2_result['mean_ndcg']:.4f}"))

        print(f"  Mean nDCG@10 (graded): {v2_result['mean_ndcg']:.4f}")
        # Print per-category stats
        orig_9 = [pq["ndcg"] for pq in v2_result["per_query"][:9]]
        arch_26 = [pq["ndcg"] for pq in v2_result["per_query"][9:35]]
        cross_15 = [pq["ndcg"] for pq in v2_result["per_query"][35:]]
        print(f"    Original 9 queries:    mean={np.mean(orig_9):.4f}")
        print(f"    Archetype 26 queries:  mean={np.mean(arch_26):.4f}")
        print(f"    Cross-cutting 15:      mean={np.mean(cross_15):.4f}")
        # Print worst 5
        sorted_q = sorted(
            zip(NL_QUERIES_V2, v2_result["per_query"]),
            key=lambda x: x[1]["ndcg"],
        )
        print(f"  Worst 5:")
        for (q, z, a), pq in sorted_q[:5]:
            print(f"    nDCG={pq['ndcg']:.4f}  \"{q[:60]}\"")
        print(f"  Time: {dt:.1f}s")

    # --- ndcg_v2 + rerank ---
    if "ndcg_v2" in requested and args.enable_rerank and gallery_raw_cards is not None and "nlq_ndcg_v2" in results:
        from football_embed.evaluation.rerank import rerank_candidates
        print(f"\n--- NL Retrieval nDCG@10 (v2 graded + RERANK alpha={args.rerank_alpha}) ---")
        t0 = time.time()

        # Dense scores: query_embs @ gallery_embs.T
        dense_scores = v2_embs @ gallery_embs.T  # (Q, N)

        reranked_idx, _ = rerank_candidates(
            dense_scores, gallery_raw_cards, feature_names,
            v2_texts, alpha=args.rerank_alpha, top_k=args.rerank_top_k, rerank_k=10,
        )

        # Compute graded nDCG on reranked results
        discounts = np.log2(np.arange(2, 12))  # k=10
        rerank_per_query = []
        for qi in range(len(v2_texts)):
            rel_zones = v2_zones[qi]
            rel_archs = v2_archs[qi]
            top_idx = reranked_idx[qi]

            rels = np.zeros(10, dtype=np.float64)
            for j, idx in enumerate(top_idx):
                z = gallery_zones[idx]
                a = str(gallery_archetypes[idx]) if gallery_archetypes is not None else ""
                if z in rel_zones:
                    if rel_archs and a in rel_archs:
                        rels[j] = 3.0
                    else:
                        rels[j] = 1.0

            dcg = float(np.sum(rels / discounts))

            # IDCG (same as dense, precompute from gallery)
            if rel_archs:
                n3 = int(np.sum([
                    gallery_zones[i] in rel_zones and str(gallery_archetypes[i]) in rel_archs
                    for i in range(len(gallery_zones))
                ]))
            else:
                n3 = 0
            n_zone = int(np.sum([gallery_zones[i] in rel_zones for i in range(len(gallery_zones))]))
            n1 = n_zone - n3
            ideal = np.zeros(10, dtype=np.float64)
            filled = 0
            for val, cnt in [(3.0, n3), (1.0, n1)]:
                take = min(cnt, 10 - filled)
                if take > 0:
                    ideal[filled:filled + take] = val
                    filled += take
            idcg = float(np.sum(ideal / discounts))
            ndcg_val = dcg / idcg if idcg > 0 else 0.0
            rerank_per_query.append({"ndcg": ndcg_val})

        mean_rerank = np.mean([pq["ndcg"] for pq in rerank_per_query])
        dt = time.time() - t0

        results["nlq_ndcg_v2_rerank"] = {
            "mean_ndcg": float(mean_rerank),
            "alpha": args.rerank_alpha,
            "per_query": [
                {"text": q, "ndcg": pq["ndcg"]}
                for q, pq in zip(v2_texts, rerank_per_query)
            ],
        }
        summary_rows.append(("nDCG@10 v2 RERANK (mean)", f"{mean_rerank:.4f}"))

        print(f"  Mean nDCG@10 (reranked): {mean_rerank:.4f}")
        r_orig_9 = [pq["ndcg"] for pq in rerank_per_query[:9]]
        r_arch_26 = [pq["ndcg"] for pq in rerank_per_query[9:35]]
        r_cross_15 = [pq["ndcg"] for pq in rerank_per_query[35:]]
        print(f"    Original 9 queries:    mean={np.mean(r_orig_9):.4f}")
        print(f"    Archetype 26 queries:  mean={np.mean(r_arch_26):.4f}")
        print(f"    Cross-cutting 15:      mean={np.mean(r_cross_15):.4f}")
        # Show biggest improvements
        if "nlq_ndcg_v2" in results:
            dense_pq = results["nlq_ndcg_v2"]["queries"]
            deltas = [(v2_texts[i], rerank_per_query[i]["ndcg"] - dense_pq[i]["ndcg"])
                      for i in range(len(v2_texts))]
            deltas.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top 5 rerank improvements:")
            for q, d in deltas[:5]:
                print(f"    +{d:.4f}  \"{q[:60]}\"")
        print(f"  Time: {dt:.1f}s")

    # --- within_zone ---
    if "within_zone" in requested:
        print("\n--- Within-Zone Retrieval ---")
        t0 = time.time()
        wz_result = within_zone_retrieval_accuracy(gallery_embs, gallery_zones, gallery_names, k=5)
        dt = time.time() - t0

        results["within_zone"] = wz_result
        summary_rows.append(("Within-zone match@5", f"{wz_result['mean_zone_match_rate']:.4f}"))

        print(f"  Mean zone match@5: {wz_result['mean_zone_match_rate']:.4f}")
        print(f"  Per-zone:")
        for zone, zdata in sorted(wz_result["per_zone"].items()):
            print(f"    {zone:25s} {zdata['mean_zone_match']:.4f} (n={zdata['n_players']})")
        print(f"  Time: {dt:.1f}s")

    # --- archetype ---
    if "archetype" in requested and gallery_archetypes is not None:
        print("\n--- Archetype Classification ---")
        t0 = time.time()
        arch_result = archetype_classification_accuracy(
            gallery_embs, gallery_archetypes, gallery_zones,
        )
        dt = time.time() - t0

        results["archetype_classification"] = arch_result
        summary_rows.append(("Archetype accuracy", f"{arch_result['accuracy']:.4f}"))
        summary_rows.append(("Archetype macro F1", f"{arch_result['macro_f1']:.4f}"))

        print(f"  Accuracy:  {arch_result['accuracy']:.4f}")
        print(f"  Macro F1:  {arch_result['macro_f1']:.4f}")
        print(f"  Classes:   {arch_result['n_classes']}")
        print(f"  Train/Test: {arch_result['n_train']} / {arch_result['n_test']}")
        if arch_result.get("per_class"):
            print(f"  Per-class accuracy:")
            for cls, acc in sorted(arch_result["per_class"].items()):
                print(f"    {cls:30s} {acc:.4f}")
        print(f"  Time: {dt:.1f}s")
    elif "archetype" in requested:
        print("\n--- Archetype Classification ---")
        print("  SKIPPED: gallery has no archetype_name column")

    # --- match_outcome ---
    if "match_outcome" in requested:
        print("\n--- Match Outcome Prediction ---")
        t0 = time.time()
        match_result = match_outcome_prediction(
            args.spadl_path, args.games_path, feature_names, args.norm_stats_path,
        )
        dt = time.time() - t0

        results["match_outcome"] = match_result
        summary_rows.append(("Match outcome accuracy", f"{match_result['accuracy']:.4f}"))
        summary_rows.append(("Match outcome baseline", f"{match_result['baseline_accuracy']:.4f}"))

        print(f"  Accuracy:  {match_result['accuracy']:.4f}")
        print(f"  Macro F1:  {match_result['macro_f1']:.4f}")
        print(f"  Baseline:  {match_result['baseline_accuracy']:.4f} (most common class)")
        print(f"  Matches:   {match_result['n_matches']}")
        print(f"  Train/Test: {match_result['n_train']} / {match_result['n_test']}")
        print(f"  Classes:   {match_result['class_distribution']}")
        above = match_result["accuracy"] > match_result["baseline_accuracy"]
        print(f"  Beats baseline: {'YES' if above else 'NO'}")
        print(f"  Time: {dt:.1f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - t_total
    summary_rows.append(("Total time", f"{total_time:.1f}s"))

    _print_table("football-embed-Bench Summary", summary_rows)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results["meta"] = {
        "text_model_path": args.text_model_path,
        "gallery_path": args.gallery_path,
        "spadl_path": args.spadl_path,
        "games_path": args.games_path,
        "norm_stats_path": args.norm_stats_path,
        "metrics_run": sorted(requested),
        "feature_names": feature_names,
        "total_time_s": round(total_time, 2),
        "device": args.device,
    }

    results_path = output_dir / "bench_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
