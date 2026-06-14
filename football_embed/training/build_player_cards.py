#!/usr/bin/env python3
"""Build player-card vectors (~114-dim statistical profiles) for contrastive text-branch training.

Computes ~114 per-player features from SPADL action data (99 static + ~15 dynamic
bigram features), applies a variance filter to prune zero/near-zero variance
features, z-score normalizes (train-split stats only), L2-normalizes, and pairs
with text descriptions. Outputs train/val pair parquets, a full player gallery,
and normalization stats.

Identity anchor features (height_cm, age_normalized, preferred_foot_encoded,
position_detail_encoded, matches_played_log) are joined from the players_lookup
when available. Final card_dim is auto-detected after variance filtering.

Usage:
    # Default paths:
    .venv/bin/python3 -m football_embed.training.build_player_cards

    # Custom paths:
    .venv/bin/python3 -m football_embed.training.build_player_cards \
        --spadl-path data/processed/spadl_unified.parquet \
        --lookup-path data/processed/players_lookup.parquet \
        --text-path data/processed/text/all_text_data_v2.parquet \
        --output-dir data/processed/text/ \
        --min-matches 3 --val-fraction 0.1 --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from football_embed.data.generate_stat_descriptions import (
    classify_zone,
    compute_player_stats,
)

# StatsBomb / Wyscout competition ID -> human-readable league name.
COMPETITION_NAMES = {
    2: "Premier League",
    7: "Ligue 1",
    9: "Bundesliga",
    11: "La Liga",
    12: "Serie A",
    16: "Champions League",
    35: "UEFA Europa League",
    37: "FA WSL",
    43: "FIFA World Cup",
    44: "MLS",
    49: "NWSL",
    53: "UEFA Women's Euro",
    55: "UEFA Euro",
    72: "Women's World Cup",
    81: "Liga Profesional",
    87: "Copa del Rey",
    116: "North American League",
    223: "Copa America",
    1238: "Indian Super League",
    1267: "African Cup of Nations",
    1470: "FIFA U20 World Cup",
}

# All candidate features (99 static SPADL-derived + ~15 dynamic bigram features).
# After variance filtering, the actual card_dim may be smaller.
FEATURE_NAMES = [
    # --- Original 30 ---
    "passes_per_match",
    "tackles_per_match",
    "shots_per_match",
    "goals_per_match",
    "crosses_per_match",
    "interceptions_per_match",
    "clearances_per_match",
    "fouls_per_match",
    "keeper_actions_per_match",
    "take_ons_per_match",
    "pass_completion",
    "tackle_success_rate",
    "cross_accuracy",
    "foot_pct",
    "forward_third_pct",
    "defensive_third_pct",
    "avg_start_x",
    "avg_start_y",
    "defensive_actions_per_match",
    "actions_per_match",
    "final_third_passes_per_match",
    "progressive_passes_per_match",
    "passes_into_box_per_match",
    "carries_into_final_third_per_match",
    "avg_pass_length",
    "final_third_pass_completion",
    "xt_gain_per_match",
    "xt_gain_per_action",
    "xt_pass_gain_per_match",
    "shots_created_per_match",
    # --- Spatial spread (4) ---
    "std_start_x",
    "std_start_y",
    "avg_end_x",
    "avg_end_y",
    # --- Pass direction (5) ---
    "forward_pass_pct",
    "backward_pass_pct",
    "lateral_pass_pct",
    "long_pass_pct",
    "short_pass_pct",
    # --- Carry features (3) ---
    "avg_carry_distance",
    "carries_into_box_per_match",
    "progressive_carry_pct",
    # --- Zonal occupation (4) ---
    "half_space_left_pct",
    "half_space_right_pct",
    "central_zone_pct",
    "penalty_box_actions_pct",
    # --- Action quality (4) ---
    "turnover_rate",
    "aerial_pct",
    "shot_conversion_rate",
    "dribble_success_rate",
    # --- Set piece (3) ---
    "corner_delivery_per_match",
    "freekick_delivery_per_match",
    "throw_in_per_match",
    # --- Temporal (3, conditional on data columns) ---
    "actions_per_minute",
    "late_game_actions_pct",
    "first_half_pct",
    # --- xT extensions (3) ---
    "xt_carry_gain_per_match",
    "xt_loss_per_match",
    "xt_gain_std",
    # --- GK-specific (3) ---
    "keeper_save_per_match",
    "keeper_claim_per_match",
    "goalkick_per_match",
    # --- Per-match rates from new totals ---
    "bad_touch_per_match",
    # --- Spatial Heatmap PCA (15 dims) ---
    "spatial_pca_0",
    "spatial_pca_1",
    "spatial_pca_2",
    "spatial_pca_3",
    "spatial_pca_4",
    "spatial_pca_5",
    "spatial_pca_6",
    "spatial_pca_7",
    "spatial_pca_8",
    "spatial_pca_9",
    "spatial_pca_10",
    "spatial_pca_11",
    "spatial_pca_12",
    "spatial_pca_13",
    "spatial_pca_14",
    # --- Action Bigram Frequencies (15 dims, names populated at runtime) ---
    # These are dynamically named bigram_X_Y; we list common ones here.
    # The actual bigram names depend on data. We handle them dynamically below.
    # --- Action Entropy (1 dim) ---
    "action_entropy",
    # --- Pass Network Centrality (4 dims) ---
    "pass_in_degree",
    "pass_out_degree",
    "pass_betweenness",
    "pass_clustering_coeff",
    # --- Pressing Intensity (2 dims) ---
    "pressing_actions_per_match",
    "counterpressing_rate",
    # --- VAEP Approximation (3 dims) ---
    "vaep_offensive_per_match",
    "vaep_defensive_per_match",
    "vaep_total_per_action",
    # --- Distributional Stats (8 dims: std + iqr for passes, shots, tackles, goals) ---
    "passes_per_match_std",
    "passes_per_match_iqr",
    "shots_per_match_std",
    "shots_per_match_iqr",
    "tackles_per_match_std",
    "tackles_per_match_iqr",
    "goals_per_match_std",
    "goals_per_match_iqr",
    # --- Convex Hull Area (1 dim) ---
    "action_convex_hull_area",
    # --- Pre-Assist Rate (1 dim) ---
    "pre_assist_per_match",
    # --- Identity anchors (joined from lookup, optional) ---
    "matches_played_log",
]

# Per-feature NaN fill values. Rates/ratios get median (computed from data),
# everything else gets a fixed constant.
_FIXED_NAN_FILLS = {
    # Original 30
    "passes_per_match": 0,
    "tackles_per_match": 0,
    "shots_per_match": 0,
    "goals_per_match": 0,
    "crosses_per_match": 0,
    "interceptions_per_match": 0,
    "clearances_per_match": 0,
    "fouls_per_match": 0,
    "keeper_actions_per_match": 0,
    "take_ons_per_match": 0,
    "foot_pct": 0.95,
    "forward_third_pct": 0,
    "defensive_third_pct": 0,
    "avg_start_x": 52.9,
    "avg_start_y": 34.4,
    "defensive_actions_per_match": 0,
    "actions_per_match": 0,
    "final_third_passes_per_match": 0,
    "progressive_passes_per_match": 0,
    "passes_into_box_per_match": 0,
    "carries_into_final_third_per_match": 0,
    "avg_pass_length": 15.0,
    "xt_gain_per_match": 0,
    "xt_gain_per_action": 0,
    "xt_pass_gain_per_match": 0,
    "shots_created_per_match": 0,
    # Spatial spread
    "std_start_x": 10.0,
    "std_start_y": 10.0,
    "avg_end_x": 52.9,
    "avg_end_y": 34.4,
    # Pass direction (pcts default to even split)
    "forward_pass_pct": 0.33,
    "backward_pass_pct": 0.33,
    "lateral_pass_pct": 0.34,
    "long_pass_pct": 0.2,
    "short_pass_pct": 0.3,
    # Carry features
    "avg_carry_distance": 5.0,
    "carries_into_box_per_match": 0,
    "progressive_carry_pct": 0.1,
    # Zonal occupation
    "half_space_left_pct": 0.25,
    "half_space_right_pct": 0.25,
    "central_zone_pct": 0.5,
    "penalty_box_actions_pct": 0,
    # Action quality
    "turnover_rate": 0.1,
    "aerial_pct": 0.05,
    # Set piece
    "corner_delivery_per_match": 0,
    "freekick_delivery_per_match": 0,
    "throw_in_per_match": 0,
    # Temporal
    "actions_per_minute": 0.3,
    "late_game_actions_pct": 0.2,
    "first_half_pct": 0.5,
    # xT extensions
    "xt_carry_gain_per_match": 0,
    "xt_loss_per_match": 0,
    "xt_gain_std": 0,
    # GK-specific
    "keeper_save_per_match": 0,
    "keeper_claim_per_match": 0,
    "goalkick_per_match": 0,
    # New per-match
    "bad_touch_per_match": 0,
    # Spatial Heatmap PCA
    "spatial_pca_0": 0,
    "spatial_pca_1": 0,
    "spatial_pca_2": 0,
    "spatial_pca_3": 0,
    "spatial_pca_4": 0,
    "spatial_pca_5": 0,
    "spatial_pca_6": 0,
    "spatial_pca_7": 0,
    "spatial_pca_8": 0,
    "spatial_pca_9": 0,
    "spatial_pca_10": 0,
    "spatial_pca_11": 0,
    "spatial_pca_12": 0,
    "spatial_pca_13": 0,
    "spatial_pca_14": 0,
    # Action Entropy
    "action_entropy": 1.5,
    # Pass Network Centrality
    "pass_in_degree": 0,
    "pass_out_degree": 0,
    "pass_betweenness": 0,
    "pass_clustering_coeff": 0,
    # Pressing Intensity
    "pressing_actions_per_match": 0,
    "counterpressing_rate": 0,
    # VAEP Approximation
    "vaep_offensive_per_match": 0,
    "vaep_defensive_per_match": 0,
    "vaep_total_per_action": 0,
    # Distributional Stats
    "passes_per_match_std": 0,
    "passes_per_match_iqr": 0,
    "shots_per_match_std": 0,
    "shots_per_match_iqr": 0,
    "tackles_per_match_std": 0,
    "tackles_per_match_iqr": 0,
    "goals_per_match_std": 0,
    "goals_per_match_iqr": 0,
    # Convex Hull Area
    "action_convex_hull_area": 0,
    # Pre-Assist Rate
    "pre_assist_per_match": 0,
    # Identity anchors
    "matches_played_log": 1.0,
}
_MEDIAN_FILL_FEATURES = {
    "pass_completion", "tackle_success_rate", "cross_accuracy",
    "final_third_pass_completion", "shot_conversion_rate", "dribble_success_rate",
}

# Minimum variance threshold for feature inclusion (raw scale).
# Features with train-split std below this are pruned.  0.02 removes spatial
# PCA 5-14, low-signal bigrams, and duplicate VAEP/xT per-action features.
_MIN_VARIANCE_THRESHOLD = 0.02


def _fill_nans(cards: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Fill NaN values per the feature spec."""
    for feat in feature_names:
        if feat not in cards.columns:
            # Feature missing entirely (conditional feature not computed)
            cards[feat] = _FIXED_NAN_FILLS.get(feat, 0)
            continue
        if feat in _MEDIAN_FILL_FEATURES:
            med = cards[feat].median()
            cards[feat] = cards[feat].fillna(med)
        else:
            cards[feat] = cards[feat].fillna(_FIXED_NAN_FILLS.get(feat, 0))
    return cards


def _zscore_normalize(
    cards: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Z-score normalize, skipping features with zero std."""
    safe_std = std.copy()
    safe_std[safe_std == 0] = 1.0  # avoid division by zero
    return (cards - mean) / safe_std


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize each row vector. Zero vectors are left as-is."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _cosine_diagnostic(cards: np.ndarray, n_sample: int = 500, seed: int = 42) -> float:
    """Compute mean pairwise cosine similarity on a random sample."""
    rng = np.random.RandomState(seed)
    n = min(n_sample, len(cards))
    idx = rng.choice(len(cards), size=n, replace=False)
    sample = cards[idx]

    # Normalize (should already be L2-normed, but be safe)
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = sample / norms

    # Pairwise cosine = dot product of L2-normed vectors
    sim_matrix = normed @ normed.T
    # Exclude diagonal (self-similarity = 1.0)
    n_pairs = n * (n - 1)
    total = sim_matrix.sum() - np.trace(sim_matrix)
    return float(total / n_pairs)


def _compute_per_season_stats(
    df: pd.DataFrame,
    min_matches: int,
    min_season_matches: int = 20,
) -> pd.DataFrame:
    """Compute `compute_player_stats` per-season and concatenate.

    Seasons with fewer than ``min_season_matches`` matches are skipped
    (too noisy to learn from). Returns a DataFrame with one row per
    (player_id, season_id) pair, plus all stat columns.
    """
    seasons = sorted(df["season_id"].unique())
    print(f"  Found {len(seasons)} raw seasons")

    all_season_stats = []
    for sid in seasons:
        season_df = df[df["season_id"] == sid]
        n_matches = season_df["game_id"].nunique()
        if n_matches < min_season_matches:
            continue
        n_actions = len(season_df)
        n_players = season_df["player_id"].nunique()
        print(
            f"  Season {sid}: {n_actions:,} actions, {n_players:,} players, "
            f"{n_matches:,} matches"
        )
        s_stats = compute_player_stats(season_df)
        s_stats = s_stats[s_stats["matches_played"] >= min_matches].copy()
        s_stats["zone"] = s_stats.apply(classify_zone, axis=1)
        s_stats.loc[s_stats["zone"] == "Striker", "zone"] = "Attacking midfielder"
        s_stats["season_id"] = sid
        s_stats["player_id"] = s_stats.index
        all_season_stats.append(s_stats.reset_index(drop=True))

    if not all_season_stats:
        raise ValueError(
            f"No seasons with >= {min_season_matches} matches found"
        )
    per_season = pd.concat(all_season_stats, ignore_index=True)
    print(
        f"  Total (player, season) rows: {len(per_season):,} from "
        f"{len(all_season_stats)} seasons"
    )
    return per_season


def _apply_eb_shrinkage(
    per_season: pd.DataFrame,
    feature_names: list[str],
    eb_k: float = 5.0,
    train_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of per-season features toward GLOBAL zone mean.

    For each feature, compute the **cross-season** zone mean (ignoring season_id)
    and blend each player's value toward it:

        shrunk = (n * x + k * zone_mean_global) / (n + k)

    where ``n`` is matches_played and ``k`` is the prior strength. Low-sample
    player-seasons are pulled toward the global zone mean (regularization),
    while high-sample seasons pass through nearly unchanged.

    **Critical: zone means are GLOBAL (cross-season), not per-season.**
    Per-season means inject season-specific bias that breaks temporal cosine
    (same player in different seasons gets pulled toward different zone means).
    Global means preserve identity: same player in different seasons gets
    pulled toward the same target, so the gap between their cards shrinks
    uniformly. This is the whole point of shrinkage for temporal stability.

    If ``train_mask`` is provided, zone means are computed from train rows only
    (proper train/val hygiene). Otherwise the full dataset is used.

    Identity features (``matches_played_log``) are NOT shrunk.
    """
    skip = {"matches_played_log"}
    src = (
        per_season.loc[train_mask] if train_mask is not None else per_season
    )
    for feat in feature_names:
        if feat in skip or feat not in per_season.columns:
            continue
        # Global (cross-season) zone mean from the train split.
        zone_mean_map = src.groupby("zone")[feat].mean()
        zone_mean = per_season["zone"].map(zone_mean_map)
        n = per_season["matches_played"]
        x = per_season[feat]
        per_season[feat] = (n * x + eb_k * zone_mean) / (n + eb_k)
    return per_season


def _build_cross_season_direct_pairs(
    per_season: pd.DataFrame,
    text_df: pd.DataFrame,
    card_cols: list[str],
    train_players: set[str],
    max_pairs: int = 10_000,
    texts_per_player: int = 2,
    seed: int = 42,
) -> pd.DataFrame | None:
    """Direct cross-season positive pairs: (text, season_S_card) rows
    emitted for each season a player appears in.

    For each train player with >= 2 seasons, pick ``texts_per_player`` random
    text variants and emit one pair row per (text, season). The multi-positive
    masking in InfoNCE pulls these together during training. Cap total pairs
    at ``max_pairs``. Replaces the EMA smoothing approach from v10/v11.
    """
    rng = np.random.RandomState(seed)
    player_groups = per_season.groupby("player_name")

    # Only players with >= 2 seasons generate cross-season signal
    multi_season_players = [
        p for p, g in player_groups if len(g) >= 2 and p in train_players
    ]
    if not multi_season_players:
        return None
    print(
        f"  [cross-season] {len(multi_season_players):,} train players "
        f"have 2+ seasons"
    )

    texts_by_player = text_df.groupby("player_name")["text_content"].apply(
        list
    ).to_dict()

    rows = []
    for pname in multi_season_players:
        if pname not in texts_by_player:
            continue
        texts = texts_by_player[pname]
        n_pick = min(texts_per_player, len(texts))
        picked_idx = rng.choice(len(texts), size=n_pick, replace=False)
        picked_texts = [texts[i] for i in picked_idx]
        group = player_groups.get_group(pname)
        for text in picked_texts:
            for _, card_row in group.iterrows():
                row = {
                    "player_name": pname,
                    "text_content": text,
                    "zone": card_row["zone"],
                    "season_id": int(card_row["season_id"]),
                    **{c: card_row[c] for c in card_cols},
                }
                rows.append(row)
        if len(rows) >= max_pairs * 2:  # oversample buffer
            break

    if not rows:
        return None
    result = pd.DataFrame(rows)
    if len(result) > max_pairs:
        result = result.sample(n=max_pairs, random_state=seed).reset_index(
            drop=True
        )
    print(f"  [cross-season] {len(result):,} direct cross-season pairs")
    return result


def build_cross_season_pairs(
    spadl_path: str,
    text_path: str,
    feature_names: list[str],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    ema_alpha: float = 0.7,
    max_fraction: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame | None:
    """Build cross-season training pairs: text from season S, card from season S+1.

    For players appearing in 2+ consecutive seasons, generate pairs where the
    text description comes from one season and the card vector from the next.
    This teaches temporal robustness.

    Season smoothing: card_ema[t] = alpha * card[t] + (1 - alpha) * card_ema[t-1]

    Args:
        spadl_path: Path to spadl_unified.parquet.
        text_path: Path to text data parquet with text_content, player_name columns.
        feature_names: Feature column names matching norm stats.
        norm_mean: Z-score mean array.
        norm_std: Z-score std array.
        ema_alpha: Weight for current season (default 0.7).
        max_fraction: Cap cross-season pairs at this fraction of existing training data.
        seed: Random seed.

    Returns:
        DataFrame with columns: player_name, text_content, zone, card_0..card_N,
        or None if no cross-season data found.
    """
    from football_embed.data.generate_stat_descriptions import (
        classify_zone,
        compute_player_stats,
    )

    df = pd.read_parquet(spadl_path)
    if "season_id" not in df.columns:
        print("  [cross-season] No season_id column, skipping")
        return None

    seasons = sorted(df["season_id"].unique())
    if len(seasons) < 2:
        print(f"  [cross-season] Only {len(seasons)} season(s), need 2+, skipping")
        return None

    print(f"  [cross-season] Found {len(seasons)} seasons: {seasons}")

    safe_std = norm_std.copy()
    safe_std[safe_std == 0] = 1.0

    # Compute per-season player cards
    season_cards = {}  # season -> {player_id: raw_card_array}
    season_zones = {}  # season -> {player_id: zone_string}
    for sid in seasons:
        season_df = df[df["season_id"] == sid]
        stats = compute_player_stats(season_df)
        stats = stats[stats["matches_played"] >= 3]
        stats["zone"] = stats.apply(classify_zone, axis=1)
        stats.loc[stats["zone"] == "Striker", "zone"] = "Attacking midfielder"

        for feat in feature_names:
            if feat not in stats.columns:
                stats[feat] = 0

        raw = stats[feature_names].fillna(0).values.astype(np.float64)
        season_cards[sid] = dict(zip(stats.index, raw))
        season_zones[sid] = dict(zip(stats.index, stats["zone"]))

    # Apply EMA smoothing across consecutive seasons
    ema_cards = {}  # season -> {player_id: smoothed_card}
    for i, sid in enumerate(seasons):
        ema_cards[sid] = {}
        for pid, card in season_cards[sid].items():
            if i == 0 or pid not in ema_cards[seasons[i - 1]]:
                ema_cards[sid][pid] = card
            else:
                prev = ema_cards[seasons[i - 1]][pid]
                ema_cards[sid][pid] = ema_alpha * card + (1 - ema_alpha) * prev

    # Load text data and build name->player_id map
    text_df = pd.read_parquet(text_path)
    # We need player_name -> text_content mapping
    # Group by player_name, sample one text per player
    rng = np.random.RandomState(seed)

    # Build cross-season pairs
    pairs = []
    card_dim = len(feature_names)
    for i in range(len(seasons) - 1):
        s_text = seasons[i]
        s_card = seasons[i + 1]
        common_pids = set(season_cards[s_text].keys()) & set(ema_cards[s_card].keys())

        for pid in common_pids:
            # Z-score normalize the EMA-smoothed card
            card_raw = ema_cards[s_card][pid]
            card_normed = (card_raw - norm_mean) / safe_std
            # L2-normalize
            norm_val = np.linalg.norm(card_normed)
            if norm_val > 0:
                card_normed = card_normed / norm_val

            zone = season_zones.get(s_card, {}).get(pid, season_zones.get(s_text, {}).get(pid, "Unknown"))

            pairs.append({
                "player_id": pid,
                "zone": zone,
                **{f"card_{j}": card_normed[j] for j in range(card_dim)},
            })

    if not pairs:
        print("  [cross-season] No common players across seasons")
        return None

    pairs_df = pd.DataFrame(pairs)
    print(f"  [cross-season] {len(pairs_df)} cross-season card entries from {pairs_df['player_id'].nunique()} players")

    # Join with text data by player_name (we need to map player_id back to name)
    # Load lookup for name mapping
    lookup_path = Path(spadl_path).parent / "players_lookup.parquet"
    if not lookup_path.exists():
        print("  [cross-season] No players_lookup.parquet, cannot join names")
        return None

    lookup = pd.read_parquet(lookup_path)
    pid_to_name = dict(zip(lookup["player_id"], lookup["player_name"]))
    pairs_df["player_name"] = pairs_df["player_id"].map(pid_to_name)
    pairs_df = pairs_df.dropna(subset=["player_name"])

    # Join with text: pick a random text for each player
    player_texts = (
        text_df.groupby("player_name")["text_content"]
        .apply(list)
        .to_dict()
    )

    rows = []
    card_cols = [f"card_{j}" for j in range(card_dim)]
    for _, row in pairs_df.iterrows():
        pname = row["player_name"]
        if pname not in player_texts:
            continue
        texts = player_texts[pname]
        text = texts[rng.randint(len(texts))]
        rows.append({
            "player_name": pname,
            "text_content": text,
            "zone": row["zone"],
            **{c: row[c] for c in card_cols},
        })

    if not rows:
        print("  [cross-season] No text matches for cross-season players")
        return None

    result = pd.DataFrame(rows)
    print(f"  [cross-season] {len(result)} cross-season text-card pairs")
    return result


def build_player_cards_season_first(
    spadl_path: str,
    lookup_path: str,
    text_path: str,
    output_dir: str,
    min_matches: int = 3,
    min_season_matches: int = 20,
    val_fraction: float = 0.1,
    seed: int = 42,
    eb_k: float = 5.0,
    texts_per_player_cross_season: int = 2,
    max_cross_season_pairs: int = 10_000,
    variance_threshold: float = 0.02,
) -> None:
    """Season-first card rebuild for v12a.

    Key differences from ``build_player_cards``:

    1. Computes stats per (player_id, season_id) instead of career-aggregated.
    2. Applies empirical-Bayes shrinkage toward (season, zone) mean to
       regularize low-sample player-seasons.
    3. Every training pair is a (text, season_card) row — no career cards.
    4. Career gallery is derived as the mean of a player's season cards
       (re-L2-normalized), so bench retrieval remains 1 vector per player.
    5. Cross-season positive pairs emitted explicitly (no EMA smoothing).
    6. Saves ``season_cards.parquet`` for the fixed ``temporal_consistency``
       metric to read directly (closes the train/eval mismatch).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load SPADL ---
    print(f"Loading SPADL data from {spadl_path}...")
    df = pd.read_parquet(spadl_path)
    print(
        f"  {len(df):,} actions, {df['player_id'].nunique():,} players, "
        f"{df['game_id'].nunique():,} matches"
    )
    if "season_id" not in df.columns:
        raise ValueError("season-first mode requires season_id in SPADL data")

    # --- 2. Per-season stats ---
    print("Computing per-season stats (expensive, one pass per season)...")
    per_season = _compute_per_season_stats(
        df, min_matches=min_matches, min_season_matches=min_season_matches
    )

    # --- 3. Zone distribution diagnostic ---
    zone_counts = per_season["zone"].value_counts()
    print("  Per-season zone distribution:")
    for zone, count in zone_counts.items():
        print(f"    {zone}: {count:,}")

    # --- 4. matches_played_log identity anchor ---
    per_season["matches_played_log"] = np.log1p(per_season["matches_played"])

    # --- 5. Determine available features ---
    available_features = []
    for feat in FEATURE_NAMES:
        if feat in per_season.columns:
            available_features.append(feat)
        elif feat in _FIXED_NAN_FILLS or feat in _MEDIAN_FILL_FEATURES:
            available_features.append(feat)

    bigram_cols = sorted(
        [c for c in per_season.columns if c.startswith("bigram_")]
    )
    for col in bigram_cols:
        if col not in available_features:
            available_features.append(col)
            if col not in _FIXED_NAN_FILLS:
                _FIXED_NAN_FILLS[col] = 0

    n_bigram = len(bigram_cols)
    print(
        f"  Candidate features: {len(available_features)} "
        f"(+ {n_bigram} dynamic bigrams)"
    )

    # --- 6. Fill NaNs ---
    per_season = _fill_nans(per_season, available_features)

    # --- 7. Load lookup for player names ---
    print(f"Loading player lookup from {lookup_path}...")
    lookup = pd.read_parquet(lookup_path)
    lookup_dedup = lookup.drop_duplicates(subset="player_id").set_index(
        "player_id"
    )
    name_map = lookup_dedup["player_name"]
    per_season["player_name"] = per_season["player_id"].map(name_map)
    before = len(per_season)
    per_season = per_season[per_season["player_name"].notna()].reset_index(
        drop=True
    )
    print(f"  Dropped {before - len(per_season)} rows without names")

    # --- 8. Player-level train/val split (same split across all seasons) ---
    rng = np.random.RandomState(seed)
    unique_players = np.array(sorted(per_season["player_name"].unique()))
    rng.shuffle(unique_players)
    n_val = max(1, int(len(unique_players) * val_fraction))
    val_players = set(unique_players[:n_val])
    train_players = set(unique_players[n_val:])
    train_mask = per_season["player_name"].isin(train_players).values
    print(
        f"  Train: {train_mask.sum():,} rows, "
        f"Val: {(~train_mask).sum():,} rows"
    )

    # --- 9. Empirical-Bayes shrinkage (global zone means, train-split) ---
    print(
        f"Applying EB shrinkage toward global zone mean (k={eb_k}, "
        f"train-only means)..."
    )
    per_season = _apply_eb_shrinkage(
        per_season, available_features, eb_k=eb_k, train_mask=train_mask
    )

    # --- 10. Z-score norm (train split only) ---
    print("Computing z-score stats from train split...")
    train_features = per_season.loc[train_mask, available_features].values.astype(
        np.float64
    )
    feat_mean = train_features.mean(axis=0)
    feat_std = train_features.std(axis=0, ddof=0)

    # --- 11. Variance filter ---
    vt = variance_threshold
    keep_mask = feat_std > vt
    n_pruned = (~keep_mask).sum()
    if n_pruned > 0:
        pruned_names = [
            available_features[i]
            for i in range(len(available_features))
            if not keep_mask[i]
        ]
        print(
            f"  Variance filter (threshold={vt}): pruning {n_pruned} features: {pruned_names}"
        )
        available_features = [
            available_features[i]
            for i in range(len(available_features))
            if keep_mask[i]
        ]
        feat_mean = feat_mean[keep_mask]
        feat_std = feat_std[keep_mask]
    final_card_dim = len(available_features)
    print(f"  Final card_dim: {final_card_dim}")

    # --- 12. Normalize + L2 ---
    all_features_arr = per_season[available_features].values.astype(np.float64)
    all_normed = _zscore_normalize(all_features_arr, feat_mean, feat_std)
    all_normed = _l2_normalize(all_normed)

    card_cols = [f"card_{i}" for i in range(final_card_dim)]
    for i, col in enumerate(card_cols):
        per_season[col] = all_normed[:, i]

    # --- 13. Save season_cards.parquet (for temporal metric) ---
    season_cards_out = output_dir / "season_cards.parquet"
    season_cards_df = per_season[
        [
            "player_id",
            "player_name",
            "season_id",
            "zone",
            "matches_played",
        ]
        + card_cols
    ].copy()
    season_cards_df.to_parquet(season_cards_out, index=False)
    print(
        f"Saved season cards -> {season_cards_out} "
        f"({len(season_cards_df):,} rows)"
    )

    # --- 14. Career gallery: mean of season cards per player, re-L2 ---
    agg_spec = {col: (col, "mean") for col in card_cols}
    agg_spec["total_matches"] = ("matches_played", "sum")
    career = per_season.groupby(
        ["player_id", "player_name"], as_index=False
    ).agg(**agg_spec)
    career_arr = career[card_cols].values
    career_arr = _l2_normalize(career_arr)
    for i, col in enumerate(card_cols):
        career[col] = career_arr[:, i]

    # Majority-vote zone across seasons (career-level zone label)
    zone_mode = (
        per_season.groupby("player_name")["zone"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "Unknown")
        .rename("zone")
    )
    career = career.merge(zone_mode, on="player_name", how="left")

    # Metadata from lookup
    career["team_name"] = career["player_id"].map(
        lookup_dedup["team_name"]
        if "team_name" in lookup_dedup.columns
        else pd.Series(dtype=str)
    )
    career["position"] = career["player_id"].map(
        lookup_dedup["starting_position_name"]
        if "starting_position_name" in lookup_dedup.columns
        else pd.Series(dtype=str)
    )
    # Primary league
    player_comp = (
        df.groupby(["player_id", "competition_id"])
        .size()
        .reset_index(name="n_actions")
    )
    primary_comp = (
        player_comp.sort_values("n_actions", ascending=False)
        .drop_duplicates(subset="player_id")
        .set_index("player_id")["competition_id"]
    )
    career["league"] = career["player_id"].map(primary_comp).map(
        COMPETITION_NAMES
    )

    gallery_path = output_dir / "player_card_gallery.parquet"
    career.to_parquet(gallery_path, index=False)
    print(
        f"Saved career gallery -> {gallery_path} ({len(career):,} players)"
    )

    # --- 15. Normalization stats ---
    norm_path = output_dir / "card_normalization.json"
    with open(norm_path, "w") as f:
        json.dump(
            {
                "mean": feat_mean.tolist(),
                "std": feat_std.tolist(),
                "feature_names": available_features,
                "card_dim": final_card_dim,
                "eb_k": eb_k,
                "season_first": True,
            },
            f,
            indent=2,
        )
    print(f"Saved normalization stats -> {norm_path}")

    # --- 16. Build text<>season_card training pairs (vectorized) ---
    print(f"\nLoading text data from {text_path}...")
    text_df = pd.read_parquet(text_path)
    print(
        f"  {len(text_df):,} rows, {text_df['player_name'].nunique():,} players"
    )

    # Drop text rows whose player has no season cards.
    text_df = text_df[
        text_df["player_name"].isin(per_season["player_name"])
    ].copy()
    text_df = text_df[
        text_df["text_content"].notna()
        & (text_df["text_content"].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    # Pre-compute a flat (player_name, season_id, weight, *cards, zone) index
    # then group into per-player numpy arrays — O(N_seasons), not O(N_texts).
    pn_arr = per_season["player_name"].values
    sid_arr = per_season["season_id"].values.astype(np.int64)
    mp_arr = per_season["matches_played"].values.astype(np.float64)
    zone_arr = per_season["zone"].values
    card_arr = per_season[card_cols].values.astype(np.float32)

    # Build per-player contiguous slices (sort by player_name first).
    order = np.argsort(pn_arr, kind="stable")
    pn_sorted = pn_arr[order]
    sid_sorted = sid_arr[order]
    mp_sorted = mp_arr[order]
    zone_sorted = zone_arr[order]
    card_sorted = card_arr[order]
    unique_pn, start_idx, counts = np.unique(
        pn_sorted, return_index=True, return_counts=True
    )
    pn_to_slice = {
        pn: (int(start_idx[i]), int(counts[i]))
        for i, pn in enumerate(unique_pn)
    }

    rng2 = np.random.RandomState(seed + 1)
    n_texts = len(text_df)
    sampled_sid = np.empty(n_texts, dtype=np.int64)
    sampled_zone = np.empty(n_texts, dtype=object)
    sampled_cards = np.empty((n_texts, len(card_cols)), dtype=np.float32)

    text_pnames = text_df["player_name"].values
    for i in range(n_texts):
        start, cnt = pn_to_slice[text_pnames[i]]
        w = mp_sorted[start : start + cnt]
        total = w.sum()
        if total <= 0:
            pick = 0
        else:
            pick = rng2.choice(cnt, p=(w / total))
        row = start + pick
        sampled_sid[i] = sid_sorted[row]
        sampled_zone[i] = zone_sorted[row]
        sampled_cards[i] = card_sorted[row]

    merged = pd.DataFrame(
        {
            "player_name": text_df["player_name"].values,
            "text_content": text_df["text_content"].values,
            "zone": sampled_zone,
            "season_id": sampled_sid,
        }
    )
    for j, col in enumerate(card_cols):
        merged[col] = sampled_cards[:, j]

    print(
        f"  After text join: {len(merged):,} pairs, "
        f"{merged['player_name'].nunique():,} players"
    )

    # --- 17. Train/val split by player ---
    is_val = merged["player_name"].isin(val_players)
    train_df = merged[~is_val].reset_index(drop=True)
    val_df = merged[is_val].reset_index(drop=True)

    # --- 18. Merge archetypes (reuse existing archetypes.parquet) ---
    archetype_path = Path("data/processed/text/archetypes.parquet")
    if archetype_path.exists():
        print(f"Merging archetypes from {archetype_path}...")
        arch_df = pd.read_parquet(archetype_path)[
            ["player_name", "archetype_id"]
        ]
        train_df = train_df.merge(arch_df, on="player_name", how="left")
        train_df["archetype_id"] = train_df["archetype_id"].fillna(-1).astype(
            int
        )
        val_df = val_df.merge(arch_df, on="player_name", how="left")
        val_df["archetype_id"] = val_df["archetype_id"].fillna(-1).astype(int)
        career = career.merge(arch_df, on="player_name", how="left")
        career["archetype_id"] = career["archetype_id"].fillna(-1).astype(int)
        career = career.drop_duplicates(
            subset="player_name", keep="first"
        ).reset_index(drop=True)
        career.to_parquet(gallery_path, index=False)

    train_out = output_dir / "text_card_pairs_train.parquet"
    val_out = output_dir / "text_card_pairs_val.parquet"
    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)
    print(f"Saved train pairs: {len(train_df):,} rows -> {train_out}")
    print(f"Saved val pairs:   {len(val_df):,} rows -> {val_out}")

    # --- 19. Cross-season direct positive pairs ---
    cross_season_df = _build_cross_season_direct_pairs(
        per_season=per_season,
        text_df=text_df,
        card_cols=card_cols,
        train_players=train_players,
        max_pairs=max_cross_season_pairs,
        texts_per_player=texts_per_player_cross_season,
        seed=seed + 2,
    )
    if cross_season_df is not None and len(cross_season_df) > 0:
        if archetype_path.exists():
            arch_df2 = pd.read_parquet(archetype_path)[
                ["player_name", "archetype_id"]
            ]
            cross_season_df = cross_season_df.merge(
                arch_df2, on="player_name", how="left"
            )
            cross_season_df["archetype_id"] = (
                cross_season_df["archetype_id"].fillna(-1).astype(int)
            )
        train_df = pd.concat([train_df, cross_season_df], ignore_index=True)
        train_df.to_parquet(train_out, index=False)
        print(f"  Training after cross-season merge: {len(train_df):,} rows")

    # --- 20. Diagnostics ---
    print("\n--- Cosine diagnostics ---")
    mean_cos_season = _cosine_diagnostic(all_normed, n_sample=500, seed=seed)
    print(f"  Season cards mean pairwise cosine: {mean_cos_season:.4f}")
    career_arr_np = career[card_cols].values
    mean_cos_career = _cosine_diagnostic(
        career_arr_np, n_sample=min(500, len(career_arr_np)), seed=seed
    )
    print(f"  Career gallery mean pairwise cosine: {mean_cos_career:.4f}")

    # Offline temporal consistency sanity check (from just-built cards)
    print("\n--- Offline temporal consistency (from new season cards) ---")
    temporal_cosines = []
    ps_sorted = per_season.sort_values(["player_id", "season_id"])
    for pid, g in ps_sorted.groupby("player_id"):
        if len(g) < 2:
            continue
        vecs = g[card_cols].values.astype(np.float64)
        for i in range(len(vecs) - 1):
            v1 = vecs[i]
            v2 = vecs[i + 1]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                continue
            temporal_cosines.append(float(np.dot(v1 / n1, v2 / n2)))
    if temporal_cosines:
        arr = np.array(temporal_cosines)
        print(
            f"  Temporal mean cosine: {arr.mean():.4f} | "
            f"median: {np.median(arr):.4f} | "
            f"%>0.8: {(arr > 0.8).mean() * 100:.1f}% | "
            f"n_pairs: {len(arr):,}"
        )
    print("\nDone.")


def build_player_cards(
    spadl_path: str,
    lookup_path: str,
    text_path: str,
    output_dir: str,
    min_matches: int = 3,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load SPADL and compute stats ---
    print(f"Loading SPADL data from {spadl_path}...")
    df = pd.read_parquet(spadl_path)
    print(f"  {len(df):,} actions, {df['player_id'].nunique():,} players, "
          f"{df['game_id'].nunique():,} matches")

    print("Computing per-player stats...")
    stats = compute_player_stats(df)
    print(f"  {len(stats):,} players computed")

    # --- 2. Filter by min matches ---
    before = len(stats)
    stats = stats[stats["matches_played"] >= min_matches].copy()
    print(f"  Filtered to {len(stats):,} players (>= {min_matches} matches, "
          f"dropped {before - len(stats):,})")

    # --- 3. Classify zones ---
    print("Classifying playing zones...")
    stats["zone"] = stats.apply(classify_zone, axis=1)
    # Merge Striker back into Attacking midfielder (too few players to learn)
    n_striker = (stats["zone"] == "Striker").sum()
    if n_striker > 0:
        stats.loc[stats["zone"] == "Striker", "zone"] = "Attacking midfielder"
        print(f"  Merged {n_striker} Strikers into Attacking midfielder")
    zone_counts = stats["zone"].value_counts()
    for zone, count in zone_counts.items():
        print(f"  {zone}: {count:,}")

    # --- 4. Add identity anchor: matches_played_log ---
    stats["matches_played_log"] = np.log1p(stats["matches_played"])

    # --- 5. Load player lookup and join identity anchors ---
    print(f"Loading player lookup from {lookup_path}...")
    lookup = pd.read_parquet(lookup_path)

    # Deduplicate: one row per player_id
    lookup_dedup = lookup.drop_duplicates(subset="player_id").set_index("player_id")
    name_map = lookup_dedup["player_name"] if "player_name" in lookup_dedup.columns else pd.Series(dtype=str)

    # --- 6. Determine available features ---
    # Start with FEATURE_NAMES, keep only those present in stats or fillable.
    # Also discover dynamically-named bigram features from the stats columns.
    available_features = []
    for feat in FEATURE_NAMES:
        if feat in stats.columns:
            available_features.append(feat)
        elif feat in _FIXED_NAN_FILLS or feat in _MEDIAN_FILL_FEATURES:
            available_features.append(feat)  # will be filled

    # Discover bigram_* columns (dynamically named based on data)
    bigram_cols = sorted([c for c in stats.columns if c.startswith("bigram_")])
    for col in bigram_cols:
        if col not in available_features:
            available_features.append(col)
            if col not in _FIXED_NAN_FILLS:
                _FIXED_NAN_FILLS[col] = 0  # bigram frequencies default to 0

    n_bigram = len(bigram_cols)
    print(f"  Available features: {len(available_features)} / {len(FEATURE_NAMES)} candidates "
          f"(+ {n_bigram} dynamic bigram features)")

    # --- 7. Extract features and fill NaNs ---
    print(f"Extracting {len(available_features)}-dim feature vectors...")
    cards = stats[
        [f for f in available_features if f in stats.columns]
    ].copy()
    cards = _fill_nans(cards, available_features)

    nan_count = cards[available_features].isna().sum().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values remain after filling")
    else:
        print(f"  No NaN values remaining")

    cards["player_name"] = cards.index.map(name_map)
    cards["zone"] = stats["zone"]

    named = cards["player_name"].notna().sum()
    print(f"  {named:,} / {len(cards):,} players have names")

    # Drop players without names (can't join with text)
    cards = cards[cards["player_name"].notna()].copy()
    print(f"  {len(cards):,} players after dropping unnamed")

    # --- 8. Player-level train/val split ---
    rng = np.random.RandomState(seed)
    unique_players = np.array(cards["player_name"].unique())
    rng.shuffle(unique_players)

    n_val = max(1, int(len(unique_players) * val_fraction))
    val_players = set(unique_players[:n_val])
    train_players = set(unique_players[n_val:])

    train_mask = cards["player_name"].isin(train_players)
    val_mask = cards["player_name"].isin(val_players)

    print(f"  Train: {train_mask.sum():,} players, Val: {val_mask.sum():,} players")

    # --- 9. Z-score normalization from TRAIN split only ---
    print("Computing z-score normalization stats from train split...")
    train_features = cards.loc[train_mask, available_features].values.astype(np.float64)
    feat_mean = train_features.mean(axis=0)
    feat_std = train_features.std(axis=0, ddof=0)

    # --- 10. Variance filter: prune near-zero-variance features ---
    keep_mask = feat_std > _MIN_VARIANCE_THRESHOLD
    n_pruned = (~keep_mask).sum()
    if n_pruned > 0:
        pruned_names = [available_features[i] for i in range(len(available_features)) if not keep_mask[i]]
        print(f"  Variance filter: pruning {n_pruned} zero-variance features: {pruned_names}")
        available_features = [available_features[i] for i in range(len(available_features)) if keep_mask[i]]
        feat_mean = feat_mean[keep_mask]
        feat_std = feat_std[keep_mask]
    else:
        print(f"  Variance filter: all {len(available_features)} features passed")

    final_card_dim = len(available_features)
    print(f"  Final card_dim: {final_card_dim}")

    # Show per-feature stats
    for i, name in enumerate(available_features):
        print(f"  {name:40s}  mean={feat_mean[i]:8.4f}  std={feat_std[i]:8.4f}")

    all_features = cards[available_features].values.astype(np.float64)
    all_normed = _zscore_normalize(all_features, feat_mean, feat_std)

    # --- 11. L2-normalize per player ---
    all_normed = _l2_normalize(all_normed)

    # Write normalized values back into the DataFrame
    card_cols = [f"card_{i}" for i in range(final_card_dim)]
    for i, col in enumerate(card_cols):
        cards[col] = all_normed[:, i]

    # --- 14. Build the full gallery (all players, not just those with text) ---
    gallery = cards[["player_name"] + card_cols + ["zone"]].copy()

    # Enrich with team_name and position from lookup
    gallery["team_name"] = gallery.index.map(
        lookup_dedup["team_name"] if "team_name" in lookup_dedup.columns else pd.Series(dtype=str)
    )
    gallery["position"] = gallery.index.map(
        lookup_dedup["starting_position_name"] if "starting_position_name" in lookup_dedup.columns else pd.Series(dtype=str)
    )

    # Derive primary league: competition where each player has the most actions
    player_comp = (
        df.groupby(["player_id", "competition_id"])
        .size()
        .reset_index(name="n_actions")
    )
    primary_comp = (
        player_comp.sort_values("n_actions", ascending=False)
        .drop_duplicates(subset="player_id")
        .set_index("player_id")["competition_id"]
    )
    gallery["league"] = gallery.index.map(primary_comp).map(COMPETITION_NAMES)

    gallery = gallery.reset_index(drop=True)
    gallery_path = output_dir / "player_card_gallery.parquet"
    gallery.to_parquet(gallery_path, index=False)

    meta_fill = gallery[["team_name", "position", "league"]].notna().sum()
    print(f"\nSaved gallery: {len(gallery):,} players -> {gallery_path}")
    print(f"  Metadata coverage: team={meta_fill['team_name']:,}, "
          f"position={meta_fill['position']:,}, league={meta_fill['league']:,}")

    # --- 13. Save normalization stats ---
    norm_path = output_dir / "card_normalization.json"
    norm_data = {
        "mean": feat_mean.tolist(),
        "std": feat_std.tolist(),
        "feature_names": available_features,
        "card_dim": final_card_dim,
    }
    with open(norm_path, "w") as f:
        json.dump(norm_data, f, indent=2)
    print(f"Saved normalization stats -> {norm_path} (card_dim={final_card_dim})")

    # --- 16. Join with text data ---
    print(f"\nLoading text data from {text_path}...")
    text_df = pd.read_parquet(text_path)
    print(f"  {len(text_df):,} rows, {text_df['player_name'].nunique():,} unique players")

    # Inner join: only players that have both card and text
    cards_for_join = cards[["player_name", "zone"] + card_cols].reset_index(drop=True)
    merged = text_df.merge(cards_for_join, on="player_name", how="inner")

    # Drop empty text
    merged = merged[merged["text_content"].notna() & (merged["text_content"].str.strip() != "")]
    merged = merged.reset_index(drop=True)

    print(f"  After join: {len(merged):,} rows, {merged['player_name'].nunique():,} players")

    if len(merged) == 0:
        print("No valid text-card pairs found. Exiting.")
        return

    # Keep only needed columns (include zone for zone-aware training)
    keep_cols = ["player_name", "text_content", "zone"] + card_cols
    merged = merged[keep_cols]

    # Split into train/val by player name
    is_val = merged["player_name"].isin(val_players)
    train_df = merged[~is_val].reset_index(drop=True)
    val_df = merged[is_val].reset_index(drop=True)

    # --- Merge archetype_id if archetype file provided ---
    archetype_path = output_dir / "archetypes.parquet"
    if archetype_path.exists():
        print(f"\nMerging archetype labels from {archetype_path}...")
        arch_df = pd.read_parquet(archetype_path)[["player_name", "archetype_id", "archetype_name"]]
        train_df = train_df.merge(arch_df, on="player_name", how="left")
        train_df["archetype_id"] = train_df["archetype_id"].fillna(-1).astype(int)
        val_df = val_df.merge(arch_df, on="player_name", how="left")
        val_df["archetype_id"] = val_df["archetype_id"].fillna(-1).astype(int)
        # Also add to gallery
        gallery = gallery.merge(arch_df, on="player_name", how="left")
        gallery["archetype_id"] = gallery["archetype_id"].fillna(-1).astype(int)
        gallery = gallery.drop_duplicates(subset="player_name", keep="first").reset_index(drop=True)
        gallery.to_parquet(gallery_path, index=False)
        n_with = (train_df["archetype_id"] >= 0).sum()
        print(f"  Train: {n_with}/{len(train_df)} rows with archetype_id")
        print(f"  Gallery archetypes: {(gallery['archetype_id'] >= 0).sum()}/{len(gallery)}")
        # Drop archetype_name from train/val (not needed for training)
        train_df = train_df.drop(columns=["archetype_name"], errors="ignore")
        val_df = val_df.drop(columns=["archetype_name"], errors="ignore")
    else:
        print(f"\n  No archetypes.parquet found, skipping archetype_id merge")

    train_out = output_dir / "text_card_pairs_train.parquet"
    val_out = output_dir / "text_card_pairs_val.parquet"
    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)

    print(f"\nSaved train pairs: {len(train_df):,} rows, "
          f"{train_df['player_name'].nunique():,} players -> {train_out}")
    print(f"Saved val pairs:   {len(val_df):,} rows, "
          f"{val_df['player_name'].nunique():,} players -> {val_out}")

    # --- Cross-season pairs (Phase III) ---
    cross_season_df = build_cross_season_pairs(
        spadl_path=spadl_path,
        text_path=text_path,
        feature_names=available_features,
        norm_mean=feat_mean,
        norm_std=feat_std,
    )
    if cross_season_df is not None:
        # Cap at 30% of existing training data
        max_cross = int(len(train_df) * 0.30)
        if len(cross_season_df) > max_cross:
            cross_season_df = cross_season_df.sample(n=max_cross, random_state=seed)
            print(f"  Capped cross-season pairs to {max_cross}")

        # Only keep cross-season pairs for train-split players
        cross_season_df = cross_season_df[cross_season_df["player_name"].isin(train_players)]

        # Merge archetype_id into cross-season pairs if available
        if archetype_path.exists() and "archetype_id" not in cross_season_df.columns:
            arch_df2 = pd.read_parquet(archetype_path)[["player_name", "archetype_id"]]
            cross_season_df = cross_season_df.merge(arch_df2, on="player_name", how="left")
            cross_season_df["archetype_id"] = cross_season_df["archetype_id"].fillna(-1).astype(int)

        if len(cross_season_df) > 0:
            train_df = pd.concat([train_df, cross_season_df], ignore_index=True)
            print(f"  Training data after cross-season: {len(train_df)} rows")
            # Re-save
            train_df.to_parquet(train_out, index=False)

    # --- 17. Cosine similarity diagnostic ---
    print("\n--- Cosine similarity diagnostic ---")
    mean_cos = _cosine_diagnostic(all_normed, n_sample=500, seed=seed)
    print(f"Mean pairwise cosine similarity (500-player sample): {mean_cos:.4f}")
    if mean_cos > 0.90:
        print("  WARNING: Cosine similarity is very high (> 0.90). "
              "Cards may lack discriminative power.")
    elif mean_cos > 0.70:
        print("  Moderate similarity. Cards have some spread but could be more diverse.")
    else:
        print("  Good spread. Cards are well-differentiated across players.")

    # Distribution breakdown
    sample_n = min(500, len(all_normed))
    rng2 = np.random.RandomState(seed)
    sample_idx = rng2.choice(len(all_normed), size=sample_n, replace=False)
    sample_vecs = all_normed[sample_idx]
    sim_matrix = sample_vecs @ sample_vecs.T
    # Extract upper triangle (no diagonal)
    triu_idx = np.triu_indices(sample_n, k=1)
    pairwise = sim_matrix[triu_idx]
    print(f"  Pairwise cosine distribution (n={len(pairwise):,} pairs):")
    print(f"    min={pairwise.min():.4f}  p25={np.percentile(pairwise, 25):.4f}  "
          f"median={np.median(pairwise):.4f}  p75={np.percentile(pairwise, 75):.4f}  "
          f"max={pairwise.max():.4f}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description=f"Build {len(FEATURE_NAMES)}-dim player-card vectors paired with text for contrastive training."
    )
    parser.add_argument(
        "--spadl-path",
        type=str,
        default="data/processed/spadl_unified.parquet",
        help="Path to unified SPADL parquet (default: data/processed/spadl_unified.parquet)",
    )
    parser.add_argument(
        "--lookup-path",
        type=str,
        default="data/processed/players_lookup.parquet",
        help="Path to players_lookup parquet (default: data/processed/players_lookup.parquet)",
    )
    parser.add_argument(
        "--text-path",
        type=str,
        default="data/processed/text/all_text_data_v2.parquet",
        help="Path to text data parquet (default: data/processed/text/all_text_data_v2.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/text/",
        help="Output directory (default: data/processed/text/)",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=3,
        help="Minimum matches to include a player (default: 3)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of players for validation split (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "--cross-season", action="store_true", default=True,
        help="Generate cross-season training pairs (default: True)",
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=0.7,
        help="EMA weight for current season in cross-season smoothing (default: 0.7)",
    )
    parser.add_argument(
        "--season-first", action="store_true",
        help="v12a: per-(player, season) cards with EB shrinkage, direct cross-season pairs",
    )
    parser.add_argument(
        "--min-season-matches", type=int, default=20,
        help="Skip seasons with fewer matches than this (season-first only, default: 20)",
    )
    parser.add_argument(
        "--eb-k", type=float, default=5.0,
        help="Empirical-Bayes prior strength for shrinkage (season-first only, default: 5.0)",
    )
    parser.add_argument(
        "--variance-threshold", type=float, default=0.02,
        help="Minimum std for feature inclusion (default: 0.02; lower keeps more features)",
    )
    parser.add_argument(
        "--max-cross-season-pairs", type=int, default=10_000,
        help="Cap on direct cross-season positive pairs (season-first only, default: 10000)",
    )
    parser.add_argument(
        "--texts-per-player-cross-season", type=int, default=2,
        help="Text variants per player in cross-season pairs (season-first only, default: 2)",
    )
    args = parser.parse_args()

    if args.season_first:
        build_player_cards_season_first(
            spadl_path=args.spadl_path,
            lookup_path=args.lookup_path,
            text_path=args.text_path,
            output_dir=args.output_dir,
            min_matches=args.min_matches,
            min_season_matches=args.min_season_matches,
            val_fraction=args.val_fraction,
            seed=args.seed,
            eb_k=args.eb_k,
            texts_per_player_cross_season=args.texts_per_player_cross_season,
            max_cross_season_pairs=args.max_cross_season_pairs,
            variance_threshold=args.variance_threshold,
        )
    else:
        build_player_cards(
            spadl_path=args.spadl_path,
            lookup_path=args.lookup_path,
            text_path=args.text_path,
            output_dir=args.output_dir,
            min_matches=args.min_matches,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
