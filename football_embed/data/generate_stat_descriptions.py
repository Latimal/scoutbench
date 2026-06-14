#!/usr/bin/env python3
"""Generate natural language stat descriptions per player from SPADL action data.

Reads the unified SPADL parquet (6.8M actions, 9024 players, 3464 matches),
computes per-player aggregate statistics, classifies playing zones, and
produces up to 11 text description variants per player in different styles:
    - stat_scouting: scouting report style (with pass direction, set pieces, dribble success)
    - stat_comparative: compares against population percentiles (with consistency stats)
    - stat_profile: concise profile summary
    - stat_strengths: strength-focused narrative (with convex hull, action entropy)
    - stat_statistical: raw per-match stat line
    - stat_nl_role: natural language role description (bridges NL queries)
    - stat_pressing_profile: defensive work rate, pressing, counterpressing, entropy
    - stat_creative_profile: pass network centrality, progressive carries, xT, pre-assists
    - stat_archetype: archetype-led description with differentiating stats (optional)
    - stat_comparative_archetype: within-cluster percentile comparison (optional)
    - stat_wikipedia_hybrid: stat+wiki combined description (optional, needs wiki data)

The archetype variants require an archetype_df to be passed to generate_all_descriptions().
The wikipedia hybrid variant requires wiki_lookup to be passed.

These descriptions feed into the text-event contrastive alignment pipeline.
Players must include no player names in the text (model should learn from
stats alone, not identity leakage), except for the wikipedia hybrid which
inherently references the player context.

Usage:
    # Default paths:
    .venv/bin/python3 -m football_embed.data.generate_stat_descriptions

    # Custom paths:
    .venv/bin/python3 -m football_embed.data.generate_stat_descriptions \
        --spadl-path data/processed/spadl_unified.parquet \
        --lookup-path data/processed/players_lookup.parquet \
        --output-path data/processed/text/stat_descriptions.parquet \
        --min-matches 3

    # Dry run (compute stats, print samples, don't write):
    .venv/bin/python3 -m football_embed.data.generate_stat_descriptions --dry-run
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import entropy as _scipy_entropy

from football_embed.data.xt_values import ExpectedThreat


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PITCH_X_MAX = 105.8
_PITCH_Y_MAX = 68.7
_FORWARD_THIRD_X = 70.0
_DEFENSIVE_THIRD_X = 35.0
_CENTRAL_Y_LOW = 25.0
_CENTRAL_Y_HIGH = 45.0
_BOX_X = 89.3
_BOX_Y_LOW = 14.2
_BOX_Y_HIGH = 54.5

# Action type groupings
_SHOT_TYPES = {"shot", "shot_freekick", "shot_penalty"}
_CROSS_TYPES = {"cross", "corner_crossed", "freekick_crossed"}
_KEEPER_TYPES = {"keeper_claim", "keeper_punch", "keeper_save"}
_DEFENSIVE_TYPES = {"tackle", "interception", "clearance"}

# Semantic role preambles: inject the exact NL tokens users will search for
_ZONE_PREAMBLE = {
    "Goalkeeper": "A shot-stopper and last line of defense.",
    "Centre-back": "A central defender who organizes the backline and wins aerial duels.",
    "Full-back": "A wide defender who contributes to both defense and attack down the flanks.",
    "Defensive midfielder": "A deep-lying midfielder who shields the defense and controls tempo.",
    "Central midfielder": "A box-to-box midfielder who links defense and attack with energy and passing.",
    "Attacking midfielder": "A creative playmaker who operates between the lines, orchestrating attacks with vision and flair.",
    "Winger": "A wide attacker who stretches play with pace, dribbling, and crossing.",
    "Striker": "A goal-scoring forward who leads the line and finishes chances.",
    "Defender": "A central defender.",
    "Midfielder": "A midfielder.",
    "Forward": "An attacking forward.",
}

# Zone-specific strength pools: only strengths relevant to the zone are candidates
_ZONE_STRENGTH_POOL = {
    "Goalkeeper": {"passing accuracy", "pass volume", "ball progression", "clearances",
                   "pressing intensity"},
    "Centre-back": {"tackling", "interceptions", "clearances", "defensive awareness",
                    "passing accuracy", "pass volume", "ball progression",
                    "pressing intensity", "aerial presence"},
    "Full-back": {"tackling", "interceptions", "crossing", "ball progression",
                  "forward presence", "take-on ability", "defensive awareness",
                  "progressive passing", "pressing intensity", "progressive carrying"},
    "Defensive midfielder": {"tackling", "interceptions", "passing accuracy", "pass volume",
                             "defensive awareness", "progressive passing", "ball progression",
                             "pressing intensity", "network hub"},
    "Central midfielder": {"passing accuracy", "pass volume", "tackling", "interceptions",
                           "progressive passing", "ball progression", "shooting volume",
                           "chance creation", "passing threat", "threat generation",
                           "pressing intensity", "network hub", "progressive carrying"},
    "Attacking midfielder": {"goal scoring", "shooting volume", "take-on ability",
                             "forward presence", "final-third passing", "passes into box",
                             "chance creation", "passing threat", "threat generation",
                             "progressive passing", "crossing", "progressive carrying",
                             "pre-assists"},
    "Winger": {"take-on ability", "crossing", "goal scoring", "shooting volume",
               "forward presence", "ball progression", "chance creation",
               "threat generation", "passes into box", "progressive carrying"},
    "Striker": {"goal scoring", "shooting volume", "take-on ability", "forward presence",
                "chance creation", "threat generation", "aerial presence"},
}

# Zone classification thresholds
_ZONE_RULES = [
    # (label, x_min, x_max, y_central, requires_keeper)
    # Order matters: first match wins
    ("Goalkeeper", None, 15, True, True),
    ("Centre-back", None, 40, True, False),
    ("Full-back", None, 50, False, False),
    ("Defensive midfielder", 35, 55, True, False),
    ("Central midfielder", 40, 65, True, False),
    ("Attacking midfielder", 55, None, True, False),
    ("Winger", 45, None, False, False),
    ("Striker", 60, None, True, False),
]


# Spatial heatmap grid resolution
_HEATMAP_COLS = 12
_HEATMAP_ROWS = 8
_HEATMAP_BINS = _HEATMAP_COLS * _HEATMAP_ROWS  # 96
_HEATMAP_PCA_DIMS = 15

# Top action bigrams (determined empirically from SPADL data -- the most common
# consecutive action type pairs across all players). We define the canonical
# set here; any bigrams not observed for a player get 0 frequency.
_TOP_BIGRAMS: list[tuple[str, str]] | None = None  # populated at runtime


def _compute_spatial_heatmap_pca(
    df: pd.DataFrame, n_components: int = _HEATMAP_PCA_DIMS
) -> pd.DataFrame:
    """Compute spatial heatmap PCA features per player.

    Discretizes the pitch into a 12x8 grid, counts action frequencies per bin,
    normalizes per player (so bins sum to 1), then applies PCA to reduce to
    n_components dimensions.

    Returns DataFrame indexed by player_id with columns spatial_pca_0..N.
    """
    from sklearn.decomposition import PCA

    # Discretize start_x, start_y into grid bins
    col_idx = (df["start_x"] / _PITCH_X_MAX * _HEATMAP_COLS).astype(int).clip(0, _HEATMAP_COLS - 1)
    row_idx = (df["start_y"] / _PITCH_Y_MAX * _HEATMAP_ROWS).astype(int).clip(0, _HEATMAP_ROWS - 1)
    bin_idx = row_idx * _HEATMAP_COLS + col_idx

    player_ids = df["player_id"].values
    unique_players = np.unique(player_ids)
    pid_to_idx = {pid: i for i, pid in enumerate(unique_players)}

    # Build the heatmap matrix (n_players x 96)
    heatmap = np.zeros((len(unique_players), _HEATMAP_BINS), dtype=np.float64)
    for pid, b in zip(player_ids, bin_idx.values):
        heatmap[pid_to_idx[pid], b] += 1

    # Normalize rows to sum to 1 (frequency distribution)
    row_sums = heatmap.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    heatmap /= row_sums

    # PCA
    n_comp = min(n_components, _HEATMAP_BINS, len(unique_players))
    pca = PCA(n_components=n_comp, random_state=42)
    pca_result = pca.fit_transform(heatmap)

    explained = pca.explained_variance_ratio_.sum()
    print(f"  Spatial heatmap PCA: {n_comp} components explain {explained:.1%} variance")

    cols = {f"spatial_pca_{i}": pca_result[:, i] for i in range(n_comp)}
    result = pd.DataFrame(cols, index=unique_players)
    result.index.name = "player_id"
    return result


def _compute_action_bigrams(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Compute action type bigram frequency features per player.

    For each player, counts consecutive (within same game) action type pairs,
    selects the top_n most common bigrams across the dataset, and normalizes
    per player so frequencies sum to 1.

    Returns DataFrame indexed by player_id with columns bigram_X_Y.
    """
    global _TOP_BIGRAMS

    # Sort by game_id and action sequence
    sorted_df = df.sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)
    types = sorted_df["type_name"].values
    players = sorted_df["player_id"].values
    games = sorted_df["game_id"].values

    # Count all bigrams per player
    from collections import Counter
    player_bigrams: dict[int, Counter] = {}
    global_bigrams: Counter = Counter()

    for i in range(len(sorted_df) - 1):
        if games[i] != games[i + 1]:
            continue
        if players[i] != players[i + 1]:
            continue
        bigram = (types[i], types[i + 1])
        pid = players[i]
        if pid not in player_bigrams:
            player_bigrams[pid] = Counter()
        player_bigrams[pid][bigram] += 1
        global_bigrams[bigram] += 1

    # Select top_n most common bigrams globally
    top = [bg for bg, _ in global_bigrams.most_common(top_n)]
    _TOP_BIGRAMS = top
    top_set = set(top)

    print(f"  Action bigrams: top {len(top)} bigrams selected")

    # Build feature matrix
    unique_players = sorted(player_bigrams.keys())
    col_names = [f"bigram_{a}_{b}" for a, b in top]
    data = np.zeros((len(unique_players), len(top)), dtype=np.float64)
    pid_to_idx = {pid: i for i, pid in enumerate(unique_players)}

    for pid, bg_counts in player_bigrams.items():
        total = sum(bg_counts.values())
        if total == 0:
            continue
        for j, bg in enumerate(top):
            data[pid_to_idx[pid], j] = bg_counts.get(bg, 0) / total

    result = pd.DataFrame(data, index=unique_players, columns=col_names)
    result.index.name = "player_id"
    return result


def _compute_action_entropy(df: pd.DataFrame) -> pd.Series:
    """Compute Shannon entropy of each player's action type distribution.

    Higher entropy = more unpredictable action selection.
    Returns Series indexed by player_id.
    """
    # Count action types per player
    counts = df.groupby(["player_id", "type_name"]).size().unstack(fill_value=0)
    # Normalize to probabilities
    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums, axis=0)
    # Compute entropy per row
    ent = probs.apply(lambda row: _scipy_entropy(row[row > 0]), axis=1)
    ent.name = "action_entropy"
    return ent


def _compute_pass_network_centrality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pass network centrality features per player.

    For each match, builds a directed pass network (player -> player for
    successful passes on the same team), computes degree and clustering
    centrality metrics, then averages across matches.

    Returns DataFrame indexed by player_id with columns:
        pass_in_degree, pass_out_degree, pass_betweenness, pass_clustering_coeff
    """
    import networkx as nx

    is_pass = df["type_name"] == "pass"
    is_success = df["result_name"] == "success"
    pass_df = df[is_pass & is_success].copy()

    # We need to link passer to receiver. SPADL doesn't have receiver_id,
    # but within the same game and team, the next action's player is a proxy
    # for the receiver when the pass is successful.
    sorted_all = df.sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)
    # For each successful pass, the receiver is the player who performs the next
    # action on the same team in the same game.
    pass_indices = sorted_all.index[
        (sorted_all["type_name"] == "pass") & (sorted_all["result_name"] == "success")
    ].values

    passer_ids = sorted_all.loc[pass_indices, "player_id"].values
    game_ids = sorted_all.loc[pass_indices, "game_id"].values
    team_ids = sorted_all.loc[pass_indices, "team_id"].values

    # Next action lookup
    next_idx = pass_indices + 1
    valid = next_idx < len(sorted_all)
    next_pid = np.full(len(pass_indices), -1, dtype=np.int64)
    next_gid = np.full(len(pass_indices), -1, dtype=np.int64)
    next_tid = np.full(len(pass_indices), -1, dtype=np.int64)
    next_pid[valid] = sorted_all.loc[next_idx[valid], "player_id"].values
    next_gid[valid] = sorted_all.loc[next_idx[valid], "game_id"].values
    next_tid[valid] = sorted_all.loc[next_idx[valid], "team_id"].values

    # Only keep passes where next action is same game and team (receiver proxy)
    keeper = valid & (game_ids == next_gid) & (team_ids == next_tid) & (passer_ids != next_pid)
    edges = pd.DataFrame({
        "game_id": game_ids[keeper],
        "passer": passer_ids[keeper],
        "receiver": next_pid[keeper],
    })

    # Per-match network centrality
    all_in_degree = []
    all_out_degree = []
    all_betweenness = []
    all_clustering = []

    for gid, gdf in edges.groupby("game_id"):
        G = nx.DiGraph()
        for _, row in gdf.iterrows():
            if G.has_edge(row["passer"], row["receiver"]):
                G[row["passer"]][row["receiver"]]["weight"] += 1
            else:
                G.add_edge(row["passer"], row["receiver"], weight=1)

        if len(G) < 2:
            continue

        n = len(G)
        # Normalize in/out degree by max possible
        for node in G.nodes():
            in_d = G.in_degree(node, weight="weight")
            out_d = G.out_degree(node, weight="weight")
            total_w = sum(d.get("weight", 1) for _, _, d in G.edges(data=True))
            all_in_degree.append({"player_id": node, "game_id": gid, "val": in_d / max(total_w, 1)})
            all_out_degree.append({"player_id": node, "game_id": gid, "val": out_d / max(total_w, 1)})

        # Betweenness (on unweighted for speed)
        try:
            between = nx.betweenness_centrality(G)
            for node, val in between.items():
                all_betweenness.append({"player_id": node, "game_id": gid, "val": val})
        except Exception:
            pass

        # Clustering coefficient (on undirected version)
        try:
            G_undir = G.to_undirected()
            clust = nx.clustering(G_undir)
            for node, val in clust.items():
                all_clustering.append({"player_id": node, "game_id": gid, "val": val})
        except Exception:
            pass

    def _mean_per_player(records):
        if not records:
            return pd.Series(dtype=np.float64)
        rdf = pd.DataFrame(records)
        return rdf.groupby("player_id")["val"].mean()

    result = pd.DataFrame({
        "pass_in_degree": _mean_per_player(all_in_degree),
        "pass_out_degree": _mean_per_player(all_out_degree),
        "pass_betweenness": _mean_per_player(all_betweenness),
        "pass_clustering_coeff": _mean_per_player(all_clustering),
    })
    result.index.name = "player_id"
    return result


def _compute_pressing_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pressing intensity features per player.

    pressing_actions_per_match: defensive actions (tackle, interception,
        clearance, foul) within 5 seconds of the team losing possession.
    counterpressing_rate: pressing_actions / total defensive actions after
        turnovers.

    Returns DataFrame indexed by player_id.
    """
    sorted_df = df.sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)
    types = sorted_df["type_name"].values
    results = sorted_df["result_name"].values
    times = sorted_df["time_seconds"].values
    teams = sorted_df["team_id"].values
    players = sorted_df["player_id"].values
    games = sorted_df["game_id"].values

    defensive_types = {"tackle", "interception", "clearance", "foul"}
    turnover_window = 5.0  # seconds

    press_count = {}   # player_id -> count of pressing actions
    def_after_turnover = {}  # player_id -> count of all defensive after turnover

    n = len(sorted_df)
    for i in range(1, n):
        if games[i] != games[i - 1]:
            continue
        if types[i] not in defensive_types:
            continue

        # Check if there was a turnover recently: opponent had possession before
        # A turnover = previous action by opponent team (different team_id)
        # and we're doing a defensive action within 5s
        team_now = teams[i]
        pid = players[i]

        # Look backward for the most recent action by the other team
        for j in range(i - 1, max(i - 20, -1), -1):
            if games[j] != games[i]:
                break
            if teams[j] != team_now:
                # Found opponent action: check time window
                dt = times[i] - times[j]
                if 0 < dt <= turnover_window:
                    press_count[pid] = press_count.get(pid, 0) + 1
                    def_after_turnover[pid] = def_after_turnover.get(pid, 0) + 1
                elif dt > 0:
                    # Defensive action after turnover but outside window
                    def_after_turnover[pid] = def_after_turnover.get(pid, 0) + 1
                break

    matches_played = df.groupby("player_id")["game_id"].nunique()
    press_s = pd.Series(press_count, dtype=np.float64)
    def_s = pd.Series(def_after_turnover, dtype=np.float64)

    result = pd.DataFrame(index=matches_played.index)
    result["pressing_actions_per_match"] = press_s / matches_played.replace(0, np.nan)
    result["counterpressing_rate"] = press_s / def_s.replace(0, np.nan)
    result = result.fillna(0)
    result.index.name = "player_id"
    return result


def _compute_vaep_approx(df: pd.DataFrame, xt_ratings: np.ndarray) -> pd.DataFrame:
    """Compute VAEP-approximation features per player.

    Uses xT gain + action success as a VAEP proxy:
    - Offensive value: positive xT from successful passes/dribbles/crosses/shots
    - Defensive value: xT recovered from tackles/interceptions/clearances
    - Total value per action: mean of all signed xT deltas

    Returns DataFrame indexed by player_id.
    """
    df_v = df.copy()
    df_v["xt_delta"] = xt_ratings

    mp = df_v.groupby("player_id")["game_id"].nunique().replace(0, np.nan)

    # Offensive VAEP proxy: positive xT from attacking actions
    attack_types = {"pass", "dribble", "cross", "shot", "shot_freekick",
                    "shot_penalty", "corner_crossed", "freekick_crossed", "take_on"}
    is_attack = df_v["type_name"].isin(attack_types)
    is_success = df_v["result_name"] == "success"
    off_mask = is_attack & is_success & (df_v["xt_delta"] > 0)
    vaep_off = df_v[off_mask].groupby("player_id")["xt_delta"].sum()

    # Defensive VAEP proxy: xT of the ball position when winning it back
    # For tackles/interceptions, the xT value of the recovered position
    def_types = {"tackle", "interception", "clearance"}
    is_def = df_v["type_name"].isin(def_types) & is_success
    # Defensive value = xT at the start position (they prevented threat)
    from football_embed.data.xt_values import ExpectedThreat, _cell_indexes
    # Reuse the xT grid that was already fitted
    # We approximate defensive value as xT(start) for successful defensive actions
    def_df = df_v[is_def].copy()
    if len(def_df) > 0 and hasattr(_compute_vaep_approx, "_xt_grid"):
        xt_grid = _compute_vaep_approx._xt_grid
        l, w = xt_grid.shape[1], xt_grid.shape[0]
        xi, yj = _cell_indexes(def_df["start_x"], def_df["start_y"], l, w)
        row = w - 1 - yj
        def_df["def_value"] = xt_grid[row.values, xi.values]
        vaep_def = def_df.groupby("player_id")["def_value"].sum()
    else:
        vaep_def = pd.Series(dtype=np.float64)

    # Total per action
    valid_xt = df_v[df_v["xt_delta"].notna()]
    vaep_total = valid_xt.groupby("player_id")["xt_delta"].mean()

    result = pd.DataFrame(index=mp.index)
    result["vaep_offensive_per_match"] = vaep_off / mp
    result["vaep_defensive_per_match"] = vaep_def / mp
    result["vaep_total_per_action"] = vaep_total
    result = result.fillna(0)
    result.index.name = "player_id"
    return result


def _compute_distributional_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distributional (consistency) features per player.

    For key per-match stats, computes std and IQR across matches to capture
    how consistent vs volatile a player is.

    Returns DataFrame indexed by player_id.
    """
    # Per-match action counts -- use pre-computed boolean columns to avoid
    # referencing the outer df inside lambdas (which breaks after groupby).
    df_d = df[["player_id", "game_id", "type_name", "result_name"]].copy()
    df_d["is_pass"] = df_d["type_name"] == "pass"
    df_d["is_shot"] = df_d["type_name"].isin(_SHOT_TYPES)
    df_d["is_tackle"] = df_d["type_name"] == "tackle"
    df_d["is_goal"] = df_d["is_shot"] & (df_d["result_name"] == "success")
    match_stats = df_d.groupby(["player_id", "game_id"]).agg(
        passes=("is_pass", "sum"),
        shots=("is_shot", "sum"),
        tackles=("is_tackle", "sum"),
        goals=("is_goal", "sum"),
    ).reset_index()

    # Also get per-match xT gain -- this requires xt_delta column which we
    # don't have here. We'll compute std/iqr for available per-match counts.
    result_parts = []
    for col in ["passes", "shots", "tackles", "goals"]:
        grouped = match_stats.groupby("player_id")[col]
        std_vals = grouped.std().fillna(0)
        std_vals.name = f"{col}_per_match_std"

        q75 = grouped.quantile(0.75)
        q25 = grouped.quantile(0.25)
        iqr_vals = (q75 - q25).fillna(0)
        iqr_vals.name = f"{col}_per_match_iqr"

        result_parts.extend([std_vals, iqr_vals])

    result = pd.concat(result_parts, axis=1)
    result.index.name = "player_id"
    return result


def _compute_convex_hull_area(df: pd.DataFrame) -> pd.Series:
    """Compute convex hull area of each player's action locations.

    Normalized by pitch area (105 x 68 = 7140 sq m).
    Players with fewer than 3 unique action locations get 0.

    Returns Series indexed by player_id.
    """
    pitch_area = _PITCH_X_MAX * _PITCH_Y_MAX

    hull_areas = {}
    for pid, gdf in df.groupby("player_id"):
        points = gdf[["start_x", "start_y"]].drop_duplicates().values
        if len(points) < 3:
            hull_areas[pid] = 0.0
            continue
        try:
            hull = ConvexHull(points)
            hull_areas[pid] = hull.volume / pitch_area  # 2D: volume = area
        except Exception:
            hull_areas[pid] = 0.0

    result = pd.Series(hull_areas, dtype=np.float64)
    result.index.name = "player_id"
    result.name = "action_convex_hull_area"
    return result


def _compute_pre_assists(df: pd.DataFrame) -> pd.Series:
    """Compute pre-assists per player.

    A pre-assist is a successful pass that directly precedes an assist
    (a successful pass/cross immediately followed by a goal).
    Pass A -> Pass B (assist) -> Shot (goal).

    Returns Series indexed by player_id with total pre-assist counts.
    """
    pre_assist_count = pd.Series(0, index=df["player_id"].unique(), dtype=int)

    for game_id, game_df in df.groupby("game_id"):
        game_df = game_df.reset_index(drop=True)
        n = len(game_df)
        types = game_df["type_name"].values
        results = game_df["result_name"].values
        teams = game_df["team_id"].values
        players = game_df["player_id"].values

        pass_types = {"pass", "cross", "corner_crossed", "freekick_crossed"}
        shot_types = {"shot", "shot_freekick", "shot_penalty"}

        for i in range(n - 2):
            # Action i = pre-assist pass (successful)
            if types[i] not in pass_types or results[i] != "success":
                continue
            # Action i+1 = assist pass (successful, same team)
            if types[i + 1] not in pass_types or results[i + 1] != "success":
                continue
            if teams[i] != teams[i + 1]:
                continue
            # Action i+2 = goal (shot + success, same team)
            if types[i + 2] not in shot_types or results[i + 2] != "success":
                continue
            if teams[i] != teams[i + 2]:
                continue
            pre_assist_count[players[i]] += 1

    return pre_assist_count


# ---------------------------------------------------------------------------
# 1. Compute per-player stats
# ---------------------------------------------------------------------------

def compute_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SPADL actions into per-player stat profiles.

    Args:
        df: SPADL DataFrame with standard columns (type_name, result_name,
            bodypart_name, start_x, start_y, game_id, player_id, etc.)

    Returns:
        DataFrame indexed by player_id with all computed stats.
    """
    # Pre-compute boolean masks for efficiency
    is_pass = df["type_name"] == "pass"
    is_tackle = df["type_name"] == "tackle"
    is_dribble = df["type_name"] == "dribble"
    is_take_on = df["type_name"] == "take_on"
    is_foul = df["type_name"] == "foul"
    is_interception = df["type_name"] == "interception"
    is_clearance = df["type_name"] == "clearance"
    is_shot = df["type_name"].isin(_SHOT_TYPES)
    is_cross = df["type_name"].isin(_CROSS_TYPES)
    is_keeper = df["type_name"].isin(_KEEPER_TYPES)
    is_success = df["result_name"] == "success"
    is_foot = df["bodypart_name"] == "foot"
    is_head = df["bodypart_name"] == "head"
    is_forward_third = df["start_x"] > _FORWARD_THIRD_X
    is_defensive_third = df["start_x"] < _DEFENSIVE_THIRD_X
    is_bad_touch = df["type_name"] == "bad_touch"
    is_corner = df["type_name"] == "corner_crossed"
    is_fk_cross = df["type_name"] == "freekick_crossed"
    is_fk_short = df["type_name"] == "freekick_short"
    is_throw_in = df["type_name"] == "throw_in"
    is_keeper_save = df["type_name"] == "keeper_save"
    is_keeper_claim = df["type_name"] == "keeper_claim"
    is_goalkick = df["type_name"] == "goalkick"

    g = df.groupby("player_id")

    stats = pd.DataFrame({
        "matches_played": df.groupby("player_id")["game_id"].nunique(),
        "total_actions": g.size(),

        # Passing
        "total_passes": df[is_pass].groupby("player_id").size(),
        "successful_passes": df[is_pass & is_success].groupby("player_id").size(),

        # Tackling
        "total_tackles": df[is_tackle].groupby("player_id").size(),
        "successful_tackles": df[is_tackle & is_success].groupby("player_id").size(),

        # Shots and goals
        "total_shots": df[is_shot].groupby("player_id").size(),
        "total_goals": df[is_shot & is_success].groupby("player_id").size(),

        # Ball carries (SPADL "dribble" = successful ball movement, always success)
        "total_carries": df[is_dribble].groupby("player_id").size(),

        # Take-ons (SPADL records only successful take-ons; failed = other event)
        "total_take_ons": df[is_take_on].groupby("player_id").size(),

        # Crosses
        "total_crosses": df[is_cross].groupby("player_id").size(),
        "successful_crosses": df[is_cross & is_success].groupby("player_id").size(),

        # Other
        "total_interceptions": df[is_interception].groupby("player_id").size(),
        "total_clearances": df[is_clearance].groupby("player_id").size(),
        "total_fouls": df[is_foul].groupby("player_id").size(),
        "total_keeper_actions": df[is_keeper].groupby("player_id").size(),

        # Body part usage
        "foot_actions": df[is_foot].groupby("player_id").size(),
        "head_actions": df[is_head].groupby("player_id").size(),

        # Zone activity
        "forward_third_actions": df[is_forward_third].groupby("player_id").size(),
        "defensive_third_actions": df[is_defensive_third].groupby("player_id").size(),

        # Average position
        "avg_start_x": g["start_x"].mean(),
        "avg_start_y": g["start_y"].mean(),

        # Spatial spread
        "std_start_x": g["start_x"].std(),
        "std_start_y": g["start_y"].std(),
        "avg_end_x": g["end_x"].mean(),
        "avg_end_y": g["end_y"].mean(),

        # Bad touches / set pieces / GK-specific
        "total_bad_touches": df[is_bad_touch].groupby("player_id").size(),
        "total_corners": df[is_corner].groupby("player_id").size(),
        "total_fk_deliveries": df[is_fk_cross | is_fk_short].groupby("player_id").size(),
        "total_throw_ins": df[is_throw_in].groupby("player_id").size(),
        "total_keeper_saves": df[is_keeper_save].groupby("player_id").size(),
        "total_keeper_claims": df[is_keeper_claim].groupby("player_id").size(),
        "total_goalkicks": df[is_goalkick].groupby("player_id").size(),
    })

    # Zone-conditional stats (added for 26-dim card)
    is_final_third_pass = is_pass & (df["start_x"] > _FORWARD_THIRD_X)
    is_progressive_pass = is_pass & is_success & (
        (df["end_x"] - df["start_x"]) > 10
    )
    is_pass_into_box = is_pass & is_success & (
        (df["end_x"] > _BOX_X)
        & (df["end_y"] > _BOX_Y_LOW)
        & (df["end_y"] < _BOX_Y_HIGH)
    )
    is_carry_into_ft = is_dribble & (
        (df["start_x"] <= _FORWARD_THIRD_X) & (df["end_x"] > _FORWARD_THIRD_X)
    )

    stats["total_final_third_passes"] = df[is_final_third_pass].groupby("player_id").size()
    stats["successful_final_third_passes"] = df[is_final_third_pass & is_success].groupby("player_id").size()
    stats["total_progressive_passes"] = df[is_progressive_pass].groupby("player_id").size()
    stats["total_passes_into_box"] = df[is_pass_into_box].groupby("player_id").size()
    stats["total_carries_into_ft"] = df[is_carry_into_ft].groupby("player_id").size()

    # Average pass length (Euclidean distance of all passes)
    pass_df = df[is_pass].copy()
    pass_df["pass_length"] = np.sqrt(
        (pass_df["end_x"] - pass_df["start_x"]) ** 2
        + (pass_df["end_y"] - pass_df["start_y"]) ** 2
    )
    stats["avg_pass_length"] = pass_df.groupby("player_id")["pass_length"].mean()

    # Pass direction features (computed from pass_df which already has pass_length)
    pass_dx = pass_df["end_x"] - pass_df["start_x"]
    is_forward_pass = pass_dx > 5
    is_backward_pass = pass_dx < -5
    is_lateral_pass = pass_dx.abs() <= 5
    is_long_pass = pass_df["pass_length"] > 25
    is_short_pass = pass_df["pass_length"] < 10

    total_passes_per_player = pass_df.groupby("player_id").size()
    stats["forward_pass_pct"] = (
        pass_df[is_forward_pass].groupby("player_id").size()
        / total_passes_per_player.replace(0, np.nan)
    )
    stats["backward_pass_pct"] = (
        pass_df[is_backward_pass].groupby("player_id").size()
        / total_passes_per_player.replace(0, np.nan)
    )
    stats["lateral_pass_pct"] = (
        pass_df[is_lateral_pass].groupby("player_id").size()
        / total_passes_per_player.replace(0, np.nan)
    )
    stats["long_pass_pct"] = (
        pass_df[is_long_pass].groupby("player_id").size()
        / total_passes_per_player.replace(0, np.nan)
    )
    stats["short_pass_pct"] = (
        pass_df[is_short_pass].groupby("player_id").size()
        / total_passes_per_player.replace(0, np.nan)
    )

    # Carry (dribble) features
    dribble_df = df[is_dribble].copy()
    dribble_df["carry_distance"] = np.sqrt(
        (dribble_df["end_x"] - dribble_df["start_x"]) ** 2
        + (dribble_df["end_y"] - dribble_df["start_y"]) ** 2
    )
    stats["avg_carry_distance"] = dribble_df.groupby("player_id")["carry_distance"].mean()

    is_carry_into_box = (
        (dribble_df["end_x"] > _BOX_X)
        & (dribble_df["end_y"] > _BOX_Y_LOW)
        & (dribble_df["end_y"] < _BOX_Y_HIGH)
    )
    stats["total_carries_into_box"] = dribble_df[is_carry_into_box].groupby("player_id").size()

    # Progressive carries (gain > 10m in x)
    is_progressive_carry = (dribble_df["end_x"] - dribble_df["start_x"]) > 10
    total_dribbles_per_player = dribble_df.groupby("player_id").size()
    stats["progressive_carry_pct"] = (
        dribble_df[is_progressive_carry].groupby("player_id").size()
        / total_dribbles_per_player.replace(0, np.nan)
    )

    # Dribble success proxy: take_ons / (take_ons + fouls) per player
    # (SPADL only records successful take-ons; fouls approximate failures)
    # Computed later after fillna from totals.

    # Zonal occupation
    is_half_space_left = df["start_y"] < _CENTRAL_Y_LOW
    is_half_space_right = df["start_y"] > _CENTRAL_Y_HIGH
    is_central_zone = (df["start_y"] >= _CENTRAL_Y_LOW) & (df["start_y"] <= _CENTRAL_Y_HIGH)
    is_penalty_box = (
        (df["start_x"] > _BOX_X)
        & (df["start_y"] > _BOX_Y_LOW)
        & (df["start_y"] < _BOX_Y_HIGH)
    )

    total_actions_per_player = g.size()
    stats["half_space_left_pct"] = (
        df[is_half_space_left].groupby("player_id").size()
        / total_actions_per_player.replace(0, np.nan)
    )
    stats["half_space_right_pct"] = (
        df[is_half_space_right].groupby("player_id").size()
        / total_actions_per_player.replace(0, np.nan)
    )
    stats["central_zone_pct"] = (
        df[is_central_zone].groupby("player_id").size()
        / total_actions_per_player.replace(0, np.nan)
    )
    stats["penalty_box_actions_pct"] = (
        df[is_penalty_box].groupby("player_id").size()
        / total_actions_per_player.replace(0, np.nan)
    )

    # Temporal features
    has_time_seconds = "time_seconds" in df.columns
    has_period_id = "period_id" in df.columns

    if has_time_seconds:
        is_late_game = df["time_seconds"] > 75 * 60
        stats["late_game_actions_pct"] = (
            df[is_late_game].groupby("player_id").size()
            / total_actions_per_player.replace(0, np.nan)
        )

    if has_period_id:
        is_first_half = df["period_id"] == 1
        stats["first_half_pct"] = (
            df[is_first_half].groupby("player_id").size()
            / total_actions_per_player.replace(0, np.nan)
        )

    stats = stats.fillna(0)

    # Per-match rates
    mp = stats["matches_played"].replace(0, np.nan)
    stats["actions_per_match"] = stats["total_actions"] / mp
    stats["passes_per_match"] = stats["total_passes"] / mp
    stats["tackles_per_match"] = stats["total_tackles"] / mp
    stats["shots_per_match"] = stats["total_shots"] / mp
    stats["goals_per_match"] = stats["total_goals"] / mp
    stats["crosses_per_match"] = stats["total_crosses"] / mp
    stats["interceptions_per_match"] = stats["total_interceptions"] / mp
    stats["clearances_per_match"] = stats["total_clearances"] / mp
    stats["fouls_per_match"] = stats["total_fouls"] / mp
    stats["keeper_actions_per_match"] = stats["total_keeper_actions"] / mp
    stats["carries_per_match"] = stats["total_carries"] / mp
    stats["take_ons_per_match"] = stats["total_take_ons"] / mp
    stats["final_third_passes_per_match"] = stats["total_final_third_passes"] / mp
    stats["progressive_passes_per_match"] = stats["total_progressive_passes"] / mp
    stats["passes_into_box_per_match"] = stats["total_passes_into_box"] / mp
    stats["carries_into_final_third_per_match"] = stats["total_carries_into_ft"] / mp

    # New per-match rates
    stats["bad_touch_per_match"] = stats["total_bad_touches"] / mp
    stats["carries_into_box_per_match"] = stats["total_carries_into_box"] / mp
    stats["corner_delivery_per_match"] = stats["total_corners"] / mp
    stats["freekick_delivery_per_match"] = stats["total_fk_deliveries"] / mp
    stats["throw_in_per_match"] = stats["total_throw_ins"] / mp
    stats["keeper_save_per_match"] = stats["total_keeper_saves"] / mp
    stats["keeper_claim_per_match"] = stats["total_keeper_claims"] / mp
    stats["goalkick_per_match"] = stats["total_goalkicks"] / mp
    stats["actions_per_minute"] = stats["total_actions"] / (
        stats["matches_played"].replace(0, np.nan) * 90
    )

    # Rates
    stats["pass_completion"] = (
        stats["successful_passes"] / stats["total_passes"].replace(0, np.nan)
    )
    stats["tackle_success_rate"] = (
        stats["successful_tackles"] / stats["total_tackles"].replace(0, np.nan)
    )
    # NOTE: SPADL records only successful carries and take-ons, so success
    # rates are always 100% and not meaningful. We track volume only.
    stats["cross_accuracy"] = (
        stats["successful_crosses"] / stats["total_crosses"].replace(0, np.nan)
    )
    stats["final_third_pass_completion"] = (
        stats["successful_final_third_passes"]
        / stats["total_final_third_passes"].replace(0, np.nan)
    )

    # Action quality rates
    stats["turnover_rate"] = (
        (stats["total_fouls"] + stats["total_bad_touches"]
         + stats["total_passes"] - stats["successful_passes"])
        / stats["total_actions"].replace(0, np.nan)
    )
    stats["aerial_pct"] = stats["head_actions"] / stats["total_actions"].replace(0, np.nan)
    stats["shot_conversion_rate"] = (
        stats["total_goals"] / stats["total_shots"].replace(0, np.nan)
    )
    # Dribble success proxy: take_ons / (take_ons + fouls)
    stats["dribble_success_rate"] = (
        stats["total_take_ons"]
        / (stats["total_take_ons"] + stats["total_fouls"]).replace(0, np.nan)
    )

    # Foot preference
    total_bp = (stats["foot_actions"] + stats["head_actions"]).replace(0, np.nan)
    stats["foot_pct"] = stats["foot_actions"] / total_bp
    stats["head_pct"] = stats["head_actions"] / total_bp

    # Zone activity rates
    ta = stats["total_actions"].replace(0, np.nan)
    stats["forward_third_pct"] = stats["forward_third_actions"] / ta
    stats["defensive_third_pct"] = stats["defensive_third_actions"] / ta

    # Defensive actions per match (combined)
    stats["defensive_actions_per_match"] = (
        stats["tackles_per_match"] + stats["interceptions_per_match"] + stats["clearances_per_match"]
    )

    # --- xT (Expected Threat) features ---
    print("  Fitting xT model (16x12 grid)...")
    xt_model = ExpectedThreat()
    xt_model.fit(df)
    xt_ratings = xt_model.rate(df)
    df_xt = df.copy()
    df_xt["xt_delta"] = xt_ratings

    # xT gain per match: sum of positive xT deltas
    positive_xt = df_xt[df_xt["xt_delta"] > 0]
    xt_gain_total = positive_xt.groupby("player_id")["xt_delta"].sum()
    stats["xt_gain_per_match"] = xt_gain_total / mp
    stats["xt_gain_per_match"] = stats["xt_gain_per_match"].fillna(0)

    # xT gain per action: mean xT delta per successful move
    xt_valid = df_xt[df_xt["xt_delta"].notna()]
    stats["xt_gain_per_action"] = xt_valid.groupby("player_id")["xt_delta"].mean()
    stats["xt_gain_per_action"] = stats["xt_gain_per_action"].fillna(0)

    # xT pass gain per match: xT from passes only
    is_pass_xt = df_xt["type_name"] == "pass"
    pass_xt = df_xt[is_pass_xt & (df_xt["xt_delta"] > 0)]
    xt_pass_total = pass_xt.groupby("player_id")["xt_delta"].sum()
    stats["xt_pass_gain_per_match"] = xt_pass_total / mp
    stats["xt_pass_gain_per_match"] = stats["xt_pass_gain_per_match"].fillna(0)

    # xT carry (dribble) gain per match
    is_dribble_xt = df_xt["type_name"] == "dribble"
    dribble_xt = df_xt[is_dribble_xt & (df_xt["xt_delta"] > 0)]
    xt_carry_total = dribble_xt.groupby("player_id")["xt_delta"].sum()
    stats["xt_carry_gain_per_match"] = xt_carry_total / mp
    stats["xt_carry_gain_per_match"] = stats["xt_carry_gain_per_match"].fillna(0)

    # xT loss per match: sum of negative xT deltas
    negative_xt = df_xt[df_xt["xt_delta"] < 0]
    xt_loss_total = negative_xt.groupby("player_id")["xt_delta"].sum()
    stats["xt_loss_per_match"] = xt_loss_total.abs() / mp
    stats["xt_loss_per_match"] = stats["xt_loss_per_match"].fillna(0)

    # xT gain standard deviation (of positive deltas)
    stats["xt_gain_std"] = positive_xt.groupby("player_id")["xt_delta"].std()
    stats["xt_gain_std"] = stats["xt_gain_std"].fillna(0)

    # --- Shot-Creating Actions ---
    # An SCA is a successful pass/cross/dribble followed by a shot from the
    # same team within 3 actions.
    print("  Computing shot-creating actions...")
    sca_counts = _compute_shot_creating_actions(df)
    stats["shots_created_per_match"] = sca_counts / mp
    stats["shots_created_per_match"] = stats["shots_created_per_match"].fillna(0)

    # ===================================================================
    # NEW FEATURE GROUPS (Strategy 1: expand to ~120-150 dims)
    # ===================================================================

    # --- 1. Spatial Heatmap PCA (15 dims) ---
    print("  Computing spatial heatmap PCA...")
    heatmap_pca = _compute_spatial_heatmap_pca(df, n_components=_HEATMAP_PCA_DIMS)

    # --- 2. Action Bigram Frequencies (15 dims) ---
    print("  Computing action bigram frequencies...")
    bigrams = _compute_action_bigrams(df, top_n=15)

    # --- 3. Action Entropy (1 dim) ---
    print("  Computing action entropy...")
    action_entropy = _compute_action_entropy(df)

    # --- 4. Pass Network Centrality (4 dims) ---
    print("  Computing pass network centrality...")
    centrality = _compute_pass_network_centrality(df)

    # --- 5. Pressing Intensity (2 dims) ---
    print("  Computing pressing intensity...")
    pressing = _compute_pressing_intensity(df)

    # --- 6. VAEP Approximation (3 dims) ---
    print("  Computing VAEP approximation...")
    _compute_vaep_approx._xt_grid = xt_model.xT
    vaep = _compute_vaep_approx(df, xt_ratings)

    # --- 7. Distributional Stats (8 dims) ---
    print("  Computing distributional stats (consistency)...")
    distrib = _compute_distributional_stats(df)

    # --- 8. Convex Hull Area (1 dim) ---
    print("  Computing convex hull area...")
    hull_area = _compute_convex_hull_area(df)

    # --- 9. Pre-Assist Rate (1 dim) ---
    print("  Computing pre-assists...")
    pre_assists = _compute_pre_assists(df)
    pre_assist_rate = (pre_assists / mp).fillna(0)
    pre_assist_rate.name = "pre_assist_per_match"

    # Join all new features at once via pd.concat (avoids DataFrame fragmentation)
    new_parts = [
        heatmap_pca,
        bigrams,
        action_entropy.to_frame(),
        centrality,
        pressing,
        vaep,
        distrib,
        hull_area.to_frame(),
        pre_assist_rate.to_frame(),
    ]
    new_features = pd.concat(new_parts, axis=1)
    stats = pd.concat([stats, new_features.reindex(stats.index)], axis=1)

    # Fill any remaining NaN from new features
    stats = stats.fillna(0)

    return stats


def _compute_shot_creating_actions(df: pd.DataFrame) -> pd.Series:
    """Count shot-creating actions per player.

    An SCA is a successful pass, cross, or dribble that is followed by a
    shot (by any teammate on the same team) within the next 3 actions in
    the same game.

    Returns a Series indexed by player_id with SCA counts.
    """
    sca_types = {"pass", "cross", "dribble", "corner_crossed", "freekick_crossed"}
    shot_types = {"shot", "shot_freekick", "shot_penalty"}

    sca_count = pd.Series(0, index=df["player_id"].unique(), dtype=int)

    for game_id, game_df in df.groupby("game_id"):
        game_df = game_df.reset_index(drop=True)
        n = len(game_df)
        types = game_df["type_name"].values
        results = game_df["result_name"].values
        teams = game_df["team_id"].values
        players = game_df["player_id"].values

        for i in range(n):
            if types[i] not in sca_types:
                continue
            if results[i] != "success":
                continue

            team = teams[i]
            # Look ahead up to 3 actions for a shot by same team
            for j in range(i + 1, min(i + 4, n)):
                if teams[j] != team:
                    break  # possession lost
                if types[j] in shot_types:
                    sca_count[players[i]] += 1
                    break

    return sca_count


# ---------------------------------------------------------------------------
# 2. Zone classification
# ---------------------------------------------------------------------------

def classify_zone(row: pd.Series) -> str:
    """Infer a playing zone label from average position and action profile.

    Args:
        row: A Series from the player stats DataFrame.

    Returns:
        Human-readable zone string (e.g. "Central midfielder", "Winger").
    """
    avg_x = row["avg_start_x"]
    avg_y = row["avg_start_y"]
    has_keeper = row["total_keeper_actions"] > 0

    is_central = _CENTRAL_Y_LOW <= avg_y <= _CENTRAL_Y_HIGH

    # Goalkeeper: low avg_x and has keeper actions
    if avg_x < 15 and has_keeper:
        return "Goalkeeper"

    # Centre-back: low avg_x, central
    if avg_x < 40 and is_central:
        return "Centre-back"

    # Full-back: low avg_x, wide
    if avg_x < 50 and not is_central:
        return "Full-back"

    # Defensive midfielder
    if 35 <= avg_x < 55 and is_central:
        return "Defensive midfielder"

    # Central midfielder
    if 40 <= avg_x < 65 and is_central:
        return "Central midfielder"

    # Striker: very advanced position, central, high shot rate (penalty-box player)
    if avg_x >= 68 and is_central:
        shots = row.get("shots_per_match", 0)
        if shots >= 2.0:
            return "Striker"

    # Attacking midfielder: high avg_x, central (but not as advanced/shot-heavy as striker)
    if avg_x >= 55 and is_central:
        return "Attacking midfielder"

    # Winger: high avg_x, wide
    if avg_x >= 45 and not is_central:
        return "Winger"

    # Fallback: use avg_x to guess
    if avg_x < 30:
        return "Defender"
    if avg_x < 55:
        return "Midfielder"
    return "Forward"


# ---------------------------------------------------------------------------
# 3. Percentile computation
# ---------------------------------------------------------------------------

def compute_percentiles(stats: pd.DataFrame) -> pd.DataFrame:
    """Compute population percentile ranks for key stats.

    Args:
        stats: Player stats DataFrame.

    Returns:
        DataFrame with the same index, columns suffixed with '_pctile' (0-100).
    """
    rate_cols = [
        # Original 28 rate columns
        "actions_per_match", "passes_per_match", "pass_completion",
        "tackles_per_match", "tackle_success_rate", "shots_per_match",
        "goals_per_match", "take_ons_per_match", "carries_per_match",
        "crosses_per_match", "cross_accuracy", "interceptions_per_match",
        "clearances_per_match", "fouls_per_match", "keeper_actions_per_match",
        "defensive_actions_per_match", "forward_third_pct", "defensive_third_pct",
        "final_third_passes_per_match", "progressive_passes_per_match",
        "passes_into_box_per_match", "carries_into_final_third_per_match",
        "avg_pass_length", "final_third_pass_completion",
        "xt_gain_per_match", "xt_gain_per_action",
        "xt_pass_gain_per_match", "shots_created_per_match",
        # Spatial spread
        "std_start_x", "std_start_y", "avg_end_x", "avg_end_y",
        # Pass direction
        "forward_pass_pct", "backward_pass_pct", "lateral_pass_pct",
        "long_pass_pct", "short_pass_pct",
        # Carry features
        "avg_carry_distance", "carries_into_box_per_match",
        "dribble_success_rate", "progressive_carry_pct",
        # Zonal occupation
        "half_space_left_pct", "half_space_right_pct",
        "central_zone_pct", "penalty_box_actions_pct",
        # Action quality
        "bad_touch_per_match", "turnover_rate", "aerial_pct",
        "shot_conversion_rate",
        # Set pieces
        "corner_delivery_per_match", "freekick_delivery_per_match",
        "throw_in_per_match",
        # Temporal
        "late_game_actions_pct", "actions_per_minute", "first_half_pct",
        # xT extensions
        "xt_carry_gain_per_match", "xt_loss_per_match", "xt_gain_std",
        # GK-specific
        "keeper_save_per_match", "keeper_claim_per_match",
        "goalkick_per_match",
        # New: entropy, centrality, pressing, VAEP, distributional, hull, pre-assist
        "action_entropy",
        "pass_in_degree", "pass_out_degree", "pass_betweenness",
        "pass_clustering_coeff",
        "pressing_actions_per_match", "counterpressing_rate",
        "vaep_offensive_per_match", "vaep_defensive_per_match",
        "vaep_total_per_action",
        "passes_per_match_std", "passes_per_match_iqr",
        "shots_per_match_std", "shots_per_match_iqr",
        "tackles_per_match_std", "tackles_per_match_iqr",
        "goals_per_match_std", "goals_per_match_iqr",
        "action_convex_hull_area",
        "pre_assist_per_match",
    ]

    pctile = pd.DataFrame(index=stats.index)
    for col in rate_cols:
        if col in stats.columns:
            pctile[col + "_pctile"] = stats[col].rank(pct=True) * 100

    return pctile


# ---------------------------------------------------------------------------
# 4. Text generation helpers
# ---------------------------------------------------------------------------

def _fmt(val: float, decimals: int = 1) -> str:
    """Format a float to a string with given decimal places."""
    if pd.isna(val):
        return "0"
    return f"{val:.{decimals}f}"


def _pct(val: float) -> str:
    """Format a 0-1 ratio as a percentage string like '82%'."""
    if pd.isna(val):
        return "N/A"
    return f"{val * 100:.0f}%"


def _pctile_label(pctile: float) -> str:
    """Convert a 0-100 percentile to a human-readable label."""
    if pd.isna(pctile):
        return "average"
    if pctile >= 90:
        return "elite (top 10%)"
    if pctile >= 75:
        return "above-average (top quartile)"
    if pctile >= 50:
        return "average"
    if pctile >= 25:
        return "below-average"
    return "low (bottom quartile)"


def _pctile_short(pctile: float) -> str:
    """Short comparison label for the comparative variant."""
    if pd.isna(pctile):
        return "average"
    if pctile >= 90:
        return "in the top 10%"
    if pctile >= 75:
        return "in the top quartile"
    if pctile >= 50:
        return "above the median"
    if pctile >= 25:
        return "below the median"
    return "in the bottom quartile"


def _foot_label(foot_pct: float) -> str:
    """Describe foot preference from the foot % value."""
    if pd.isna(foot_pct) or foot_pct < 0.5:
        return "no clear foot preference"
    if foot_pct >= 0.95:
        return "almost exclusively foot-based"
    if foot_pct >= 0.85:
        return "strongly foot-dominant"
    return "primarily foot-based"


def _zone_third_label(fwd_pct: float, def_pct: float) -> str:
    """Describe which third of the pitch the player favours."""
    if pd.isna(fwd_pct) or pd.isna(def_pct):
        return "across the pitch"
    if fwd_pct > 0.35:
        return "primarily in the attacking third"
    if def_pct > 0.45:
        return "primarily in the defensive third"
    if fwd_pct > 0.2 and def_pct > 0.2:
        return "across both halves of the pitch"
    return "mainly in the middle third"


# ---------------------------------------------------------------------------
# 5. The five text variants
# ---------------------------------------------------------------------------

def generate_scouting(row: pd.Series) -> str:
    """Variant 1: Scouting report style."""
    zone = row["zone"]
    parts = []

    # Lead with archetype (if present) then semantic role preamble
    archetype_name = row.get("archetype_name")
    if archetype_name:
        parts.append(f"A {archetype_name}.")
    preamble = _ZONE_PREAMBLE.get(zone, zone + ".")
    parts.append(preamble)

    # Foot preference
    foot_pct = row.get("foot_pct", 0)
    if not pd.isna(foot_pct) and foot_pct >= 0.5:
        foot_str = f"Predominantly foot-based ({_pct(foot_pct)})"
    else:
        foot_str = ""

    pass_line = f"Averages {_fmt(row['passes_per_match'])} passes per match with {_pct(row['pass_completion'])} completion."
    if foot_str:
        pass_line = foot_str + ". " + pass_line
    parts.append(pass_line)

    # Defensive line
    tackles = row["tackles_per_match"]
    interceptions = row["interceptions_per_match"]
    if tackles > 0.5 or interceptions > 0.5:
        parts.append(f"Makes {_fmt(tackles)} tackles and {_fmt(interceptions)} "
                     f"interceptions per match.")

    # Attacking line
    shots = row["shots_per_match"]
    goals = row["goals_per_match"]
    if shots > 0.3:
        parts.append(f"Averages {_fmt(shots)} shots per match, scoring "
                     f"{_fmt(goals)} goals per match.")

    # Take-ons (1v1 dribbling)
    take_ons = row["take_ons_per_match"]
    if take_ons > 1.0:
        parts.append(f"Attempts {_fmt(take_ons)} take-ons per match.")

    # Crossing
    crosses = row["crosses_per_match"]
    if crosses > 0.3:
        parts.append(f"Contributes {_fmt(crosses)} crosses per match "
                     f"({_pct(row['cross_accuracy'])} accuracy).")

    # Ball progression (carries)
    carries = row["carries_per_match"]
    if carries > 30:
        parts.append(f"High ball-carrying volume ({_fmt(carries)} carries/match).")

    # Progressive passing
    ft_passes = row.get("final_third_passes_per_match", 0)
    prog_passes = row.get("progressive_passes_per_match", 0)
    if not pd.isna(ft_passes) and ft_passes > 3:
        parts.append(f"Plays {_fmt(ft_passes)} final-third passes per match.")
    if not pd.isna(prog_passes) and prog_passes > 2:
        parts.append(f"Completes {_fmt(prog_passes)} progressive passes per match.")

    # xT (threat generation)
    xt_gain = row.get("xt_gain_per_match", 0)
    xt_pass = row.get("xt_pass_gain_per_match", 0)
    if not pd.isna(xt_gain) and xt_gain > 0.5:
        parts.append(f"Generates {_fmt(xt_gain, 2)} expected threat per match.")
    if not pd.isna(xt_pass) and xt_pass > 0.3:
        parts.append(f"Creates {_fmt(xt_pass, 2)} xT through passing alone.")

    # Shot-creating actions
    sca = row.get("shots_created_per_match", 0)
    if not pd.isna(sca) and sca > 0.5:
        parts.append(f"Creates {_fmt(sca)} shots per match for teammates.")

    # Pass direction tendencies
    fwd_pass = row.get("forward_pass_pct", 0) or 0
    bwd_pass = row.get("backward_pass_pct", 0) or 0
    long_pass = row.get("long_pass_pct", 0) or 0
    if not pd.isna(fwd_pass) and fwd_pass > 0.45:
        parts.append(f"Direct in possession, playing {_pct(fwd_pass)} of passes forward.")
    elif not pd.isna(bwd_pass) and bwd_pass > 0.25:
        parts.append(f"Conservative distributor with {_pct(bwd_pass)} of passes going backwards.")
    if not pd.isna(long_pass) and long_pass > 0.15:
        parts.append(f"Comfortable with long-range passing ({_pct(long_pass)} long balls).")

    # Dribble success
    dribble_sr = row.get("dribble_success_rate", 0) or 0
    if not pd.isna(dribble_sr) and dribble_sr > 0.5 and row.get("take_ons_per_match", 0) > 0.5:
        parts.append(f"Effective dribbler with {_pct(dribble_sr)} dribble success rate.")

    # Set piece involvement
    corners = row.get("corner_delivery_per_match", 0) or 0
    fks = row.get("freekick_delivery_per_match", 0) or 0
    if not pd.isna(corners) and corners > 0.5:
        parts.append(f"Regular corner taker ({_fmt(corners)} deliveries/match).")
    if not pd.isna(fks) and fks > 0.3:
        parts.append(f"Takes free kicks ({_fmt(fks)} deliveries/match).")

    # Zone
    third = _zone_third_label(row["forward_third_pct"], row["defensive_third_pct"])
    parts.append(f"Active {third}.")

    return " ".join(parts)


def generate_comparative(row: pd.Series, pctiles: pd.Series) -> str:
    """Variant 2: Comparative style (vs population percentiles)."""
    zone = row["zone"]
    parts = []

    # Lead with archetype (if present) then semantic preamble
    archetype_name = row.get("archetype_name")
    if archetype_name:
        parts.append(f"A {archetype_name}.")
    preamble = _ZONE_PREAMBLE.get(zone, zone + ".")
    parts.append(preamble)

    # Overall profile
    pass_pctile = pctiles.get("pass_completion_pctile", 50)
    pass_label = _pctile_short(pass_pctile)
    parts.append(f"Pass completion rate ({_pct(row['pass_completion'])}) sits {pass_label}.")

    # Offensive comparison
    shots_pctile = pctiles.get("shots_per_match_pctile", 50)
    goals_pctile = pctiles.get("goals_per_match_pctile", 50)
    if shots_pctile >= 60 or goals_pctile >= 60:
        parts.append(f"Goal threat is {_pctile_short(goals_pctile)} "
                     f"with {_fmt(row['shots_per_match'])} shots and "
                     f"{_fmt(row['goals_per_match'])} goals per match.")

    # Defensive comparison
    def_pctile = pctiles.get("defensive_actions_per_match_pctile", 50)
    if def_pctile >= 50:
        parts.append(f"Defensive contribution is {_pctile_short(def_pctile)}, "
                     f"averaging {_fmt(row['defensive_actions_per_match'])} "
                     f"defensive actions per match.")
    else:
        parts.append(f"Defensive output is {_pctile_short(def_pctile)}, "
                     f"with {_fmt(row['defensive_actions_per_match'])} "
                     f"defensive actions per match.")

    # Volume comparison
    actions_pctile = pctiles.get("actions_per_match_pctile", 50)
    if actions_pctile >= 75:
        parts.append(f"High-volume player ({_fmt(row['actions_per_match'])} "
                     f"actions/match, {_pctile_short(actions_pctile)}).")
    elif actions_pctile <= 25:
        parts.append(f"Low-volume player ({_fmt(row['actions_per_match'])} "
                     f"actions/match).")

    # Take-ons (1v1 dribbling, volume only since SPADL only records successes)
    take_on_pctile = pctiles.get("take_ons_per_match_pctile", 50)
    if row["total_take_ons"] >= 10 and not pd.isna(take_on_pctile):
        parts.append(f"Take-on frequency ({_fmt(row['take_ons_per_match'])}/match) "
                     f"is {_pctile_short(take_on_pctile)}.")

    # xT and chance creation
    xt_pctile = pctiles.get("xt_gain_per_match_pctile", 50)
    sca_pctile = pctiles.get("shots_created_per_match_pctile", 50)
    if not pd.isna(xt_pctile) and xt_pctile >= 60:
        parts.append(f"Threat generation ({_fmt(row.get('xt_gain_per_match', 0), 2)} xT/match) "
                     f"is {_pctile_short(xt_pctile)}.")
    if not pd.isna(sca_pctile) and sca_pctile >= 60:
        parts.append(f"Shot creation ({_fmt(row.get('shots_created_per_match', 0))}/match) "
                     f"is {_pctile_short(sca_pctile)}.")

    # Consistency (distributional stats)
    pass_std_pctile = pctiles.get("passes_per_match_std_pctile", 50)
    goals_iqr_pctile = pctiles.get("goals_per_match_iqr_pctile", 50)
    if not pd.isna(pass_std_pctile) and pass_std_pctile <= 25:
        parts.append("Highly consistent passer with low match-to-match variation.")
    elif not pd.isna(pass_std_pctile) and pass_std_pctile >= 75:
        parts.append("Inconsistent passer with high match-to-match variability.")
    if not pd.isna(goals_iqr_pctile) and goals_iqr_pctile >= 75 and row.get("goals_per_match", 0) > 0.1:
        parts.append("Streaky goalscorer with volatile output across matches.")

    return " ".join(parts)


def generate_profile(row: pd.Series) -> str:
    """Variant 3: Concise profile summary."""
    zone = row["zone"]

    # Build a compact summary
    tags = []

    # Pass volume
    ppm = row["passes_per_match"]
    if ppm > 40:
        tags.append(f"high pass volume ({_fmt(ppm)}/match)")
    elif ppm > 25:
        tags.append(f"moderate pass volume ({_fmt(ppm)}/match)")
    else:
        tags.append(f"low pass volume ({_fmt(ppm)}/match)")

    # Defensive
    dpm = row["defensive_actions_per_match"]
    if dpm > 4:
        tags.append(f"strong defensive contribution ({_fmt(dpm)} actions/match)")
    elif dpm > 2:
        tags.append(f"solid defensive contribution ({_fmt(dpm)} actions/match)")

    # Attacking
    gpm = row["goals_per_match"]
    spm = row["shots_per_match"]
    if gpm > 0.3:
        tags.append(f"prolific scorer ({_fmt(gpm)} goals/match)")
    elif spm > 1.0:
        tags.append(f"regular shot-taker ({_fmt(spm)} shots/match)")

    # Take-ons
    topm = row["take_ons_per_match"]
    if topm > 2.0:
        tags.append(f"frequent dribbler ({_fmt(topm)} take-ons/match)")

    # Creativity (xT + SCA)
    xt_gain = row.get("xt_gain_per_match", 0)
    sca = row.get("shots_created_per_match", 0)
    if not pd.isna(xt_gain) and xt_gain > 1.0:
        tags.append(f"high threat generation ({_fmt(xt_gain, 2)} xT/match)")
    if not pd.isna(sca) and sca > 1.0:
        tags.append(f"creative chance maker ({_fmt(sca)} shots created/match)")

    # Foot
    foot_pct = row.get("foot_pct", 0)
    if not pd.isna(foot_pct) and foot_pct >= 0.85:
        tags.append(f"primarily foot-based ({_pct(foot_pct)})")

    # Zone
    third = _zone_third_label(row["forward_third_pct"], row["defensive_third_pct"])
    tags.append(f"operates {third}")

    archetype_name = row.get("archetype_name")
    archetype_prefix = f"A {archetype_name}. " if archetype_name else ""
    preamble = _ZONE_PREAMBLE.get(zone, zone + ".")
    summary = archetype_prefix + preamble + " " + ", ".join(tags).capitalize() + "."

    # Add matches played context
    mp = int(row["matches_played"])
    summary += f" Based on {mp} matches."

    return summary


def generate_strengths(row: pd.Series, pctiles: pd.Series) -> str:
    """Variant 4: Strength-focused narrative."""
    zone = row["zone"]

    # Identify top 3 strengths by percentile
    strength_map = {
        "passing accuracy": ("pass_completion_pctile", row["pass_completion"], True),
        "pass volume": ("passes_per_match_pctile", row["passes_per_match"], False),
        "tackling": ("tackles_per_match_pctile", row["tackles_per_match"], False),
        "interceptions": ("interceptions_per_match_pctile", row["interceptions_per_match"], False),
        "clearances": ("clearances_per_match_pctile", row["clearances_per_match"], False),
        "goal scoring": ("goals_per_match_pctile", row["goals_per_match"], False),
        "shooting volume": ("shots_per_match_pctile", row["shots_per_match"], False),
        "crossing": ("crosses_per_match_pctile", row["crosses_per_match"], False),
        "defensive awareness": ("defensive_actions_per_match_pctile", row["defensive_actions_per_match"], False),
        "take-on ability": ("take_ons_per_match_pctile", row["take_ons_per_match"], False),
        "ball progression": ("carries_per_match_pctile", row["carries_per_match"], False),
        "forward presence": ("forward_third_pct_pctile", row["forward_third_pct"], True),
        "final-third passing": ("final_third_passes_per_match_pctile", row.get("final_third_passes_per_match", 0), False),
        "progressive passing": ("progressive_passes_per_match_pctile", row.get("progressive_passes_per_match", 0), False),
        "passes into box": ("passes_into_box_per_match_pctile", row.get("passes_into_box_per_match", 0), False),
        "chance creation": ("shots_created_per_match_pctile", row.get("shots_created_per_match", 0), False),
        "passing threat": ("xt_pass_gain_per_match_pctile", row.get("xt_pass_gain_per_match", 0), False),
        "threat generation": ("xt_gain_per_match_pctile", row.get("xt_gain_per_match", 0), False),
        "pressing intensity": ("pressing_actions_per_match_pctile", row.get("pressing_actions_per_match", 0), False),
        "aerial presence": ("aerial_pct_pctile", row.get("aerial_pct", 0), True),
        "progressive carrying": ("progressive_carry_pct_pctile", row.get("progressive_carry_pct", 0), True),
        "network hub": ("pass_out_degree_pctile", row.get("pass_out_degree", 0), False),
        "pre-assists": ("pre_assist_per_match_pctile", row.get("pre_assist_per_match", 0), False),
    }

    # Filter to zone-relevant strengths only
    allowed = _ZONE_STRENGTH_POOL.get(zone, set(strength_map.keys()))

    scored = []
    for name, (pctile_col, val, is_rate) in strength_map.items():
        if name not in allowed:
            continue
        p = pctiles.get(pctile_col, 0)
        if pd.isna(p):
            p = 0
        scored.append((p, name, val, is_rate))

    scored.sort(reverse=True)
    top = scored[:3]

    # Lead with archetype (if present) then semantic preamble
    archetype_name = row.get("archetype_name")
    archetype_prefix = f"A {archetype_name}. " if archetype_name else ""
    preamble = _ZONE_PREAMBLE.get(zone, zone + ".")

    strength_strs = []
    for _, name, val, is_rate in top:
        if is_rate:
            strength_strs.append(f"{name} ({_pct(val)})")
        else:
            strength_strs.append(f"{name} ({_fmt(val)}/match)")

    strength_line = archetype_prefix + preamble + " Key strengths: " + ", ".join(strength_strs) + "."

    # Convex hull: roaming vs positionally fixed
    hull_area = row.get("action_convex_hull_area", 0) or 0
    if not pd.isna(hull_area) and hull_area > 0:
        hull_pctile = pctiles.get("action_convex_hull_area_pctile", 50)
        if not pd.isna(hull_pctile) and hull_pctile >= 80:
            strength_line += " A roaming presence who covers a large area of the pitch."
        elif not pd.isna(hull_pctile) and hull_pctile <= 20:
            strength_line += " Positionally disciplined, operating within a tight area."

    # Action entropy: versatile vs specialist
    entropy_val = row.get("action_entropy", 0) or 0
    if not pd.isna(entropy_val) and entropy_val > 0:
        ent_pctile = pctiles.get("action_entropy_pctile", 50)
        if not pd.isna(ent_pctile) and ent_pctile >= 80:
            strength_line += " Tactically versatile, performing a wide range of actions."
        elif not pd.isna(ent_pctile) and ent_pctile <= 20:
            strength_line += " A specialist who focuses on a narrow set of key actions."

    # Add pitch coverage
    third = _zone_third_label(row["forward_third_pct"], row["defensive_third_pct"])
    strength_line += f" Contributes {third} with an average position in the {zone.lower()} zone."

    # Matches
    mp = int(row["matches_played"])
    strength_line += f" Sample: {mp} matches, {int(row['total_actions'])} total actions."

    return strength_line


def generate_statistical(row: pd.Series) -> str:
    """Variant 5: Pure statistical line."""
    zone = row["zone"]

    # Lead with archetype (if present) then zone preamble
    archetype_name = row.get("archetype_name")
    archetype_prefix = f"A {archetype_name}. " if archetype_name else ""
    preamble = _ZONE_PREAMBLE.get(zone, zone + ".")
    lines = [archetype_prefix + preamble, "Per-match averages:"]

    # Core stats -- only include non-trivial values (skip 0.0 noise)
    stats_parts = []
    ppm = row['passes_per_match']
    if ppm > 0.5:
        stats_parts.append(f"{_fmt(ppm)} passes ({_pct(row['pass_completion'])} accurate)")
    if row['tackles_per_match'] > 0.05:
        stats_parts.append(f"{_fmt(row['tackles_per_match'])} tackles ({_pct(row['tackle_success_rate'])} success)")
    if row['interceptions_per_match'] > 0.05:
        stats_parts.append(f"{_fmt(row['interceptions_per_match'])} interceptions")
    if row['shots_per_match'] > 0.05:
        stats_parts.append(f"{_fmt(row['shots_per_match'])} shots")
    if row['goals_per_match'] > 0.01:
        stats_parts.append(f"{_fmt(row['goals_per_match'])} goals")
    if row['crosses_per_match'] > 0.1:
        stats_parts.append(f"{_fmt(row['crosses_per_match'])} crosses ({_pct(row['cross_accuracy'])} accurate)")
    if row['clearances_per_match'] > 0.1:
        stats_parts.append(f"{_fmt(row['clearances_per_match'])} clearances")
    if row['take_ons_per_match'] > 0.1:
        stats_parts.append(f"{_fmt(row['take_ons_per_match'])} take-ons")
    if row['carries_per_match'] > 1.0:
        stats_parts.append(f"{_fmt(row['carries_per_match'])} ball carries")
    if row['fouls_per_match'] > 0.1:
        stats_parts.append(f"{_fmt(row['fouls_per_match'])} fouls committed")
    ft_passes = row.get('final_third_passes_per_match', 0)
    if not pd.isna(ft_passes) and ft_passes > 0.5:
        stats_parts.append(f"{_fmt(ft_passes)} final-third passes")
    prog_passes = row.get('progressive_passes_per_match', 0)
    if not pd.isna(prog_passes) and prog_passes > 0.3:
        stats_parts.append(f"{_fmt(prog_passes)} progressive passes")
    box_passes = row.get('passes_into_box_per_match', 0)
    if not pd.isna(box_passes) and box_passes > 0.1:
        stats_parts.append(f"{_fmt(box_passes)} passes into box")
    stats_parts.append(f"avg pass length {_fmt(row.get('avg_pass_length', 15), 1)}m")
    xt_gain = row.get('xt_gain_per_match', 0)
    if not pd.isna(xt_gain) and xt_gain > 0.01:
        stats_parts.append(f"{_fmt(xt_gain, 2)} xT generated")
    xt_pass = row.get('xt_pass_gain_per_match', 0)
    if not pd.isna(xt_pass) and xt_pass > 0.01:
        stats_parts.append(f"{_fmt(xt_pass, 2)} xT from passes")
    sca = row.get('shots_created_per_match', 0)
    if not pd.isna(sca) and sca > 0.05:
        stats_parts.append(f"{_fmt(sca)} shots created")

    lines.append(", ".join(stats_parts) + ".")

    # Pressing and defensive work
    pressing = row.get("pressing_actions_per_match", 0) or 0
    counterpress = row.get("counterpressing_rate", 0) or 0
    pressing_parts = []
    if not pd.isna(pressing) and pressing > 0.1:
        pressing_parts.append(f"{_fmt(pressing)} pressing actions/match")
    if not pd.isna(counterpress) and counterpress > 0.05:
        pressing_parts.append(f"{_pct(counterpress)} counterpress rate")
    if pressing_parts:
        lines.append("Pressing: " + ", ".join(pressing_parts) + ".")

    # Set pieces
    sp_parts = []
    corners = row.get("corner_delivery_per_match", 0) or 0
    fks = row.get("freekick_delivery_per_match", 0) or 0
    throws = row.get("throw_in_per_match", 0) or 0
    if not pd.isna(corners) and corners > 0.1:
        sp_parts.append(f"{_fmt(corners)} corners/match")
    if not pd.isna(fks) and fks > 0.1:
        sp_parts.append(f"{_fmt(fks)} free kicks/match")
    if not pd.isna(throws) and throws > 0.3:
        sp_parts.append(f"{_fmt(throws)} throw-ins/match")
    if sp_parts:
        lines.append("Set pieces: " + ", ".join(sp_parts) + ".")

    # Foot
    foot_pct = row.get("foot_pct", 0)
    foot_str = _pct(foot_pct) if not pd.isna(foot_pct) else "N/A"
    lines.append(f"Foot usage: {foot_str}.")

    # Positional
    lines.append(f"Avg position: ({_fmt(row['avg_start_x'], 0)}m, {_fmt(row['avg_start_y'], 0)}m). "
                 f"Forward third: {_pct(row['forward_third_pct'])}. "
                 f"Defensive third: {_pct(row['defensive_third_pct'])}.")

    # Keeper
    kpm = row["keeper_actions_per_match"]
    if kpm > 0.1:
        ks = row.get("keeper_save_per_match", 0) or 0
        kc = row.get("keeper_claim_per_match", 0) or 0
        gk = row.get("goalkick_per_match", 0) or 0
        gk_parts = [f"{_fmt(kpm)} keeper actions/match"]
        if not pd.isna(ks) and ks > 0.05:
            gk_parts.append(f"{_fmt(ks)} saves")
        if not pd.isna(kc) and kc > 0.05:
            gk_parts.append(f"{_fmt(kc)} claims")
        if not pd.isna(gk) and gk > 0.05:
            gk_parts.append(f"{_fmt(gk)} goal kicks")
        lines.append("GK: " + ", ".join(gk_parts) + ".")

    lines.append(f"Sample: {int(row['matches_played'])} matches.")

    return " ".join(lines)


def generate_nl_role(row: pd.Series) -> str:
    """Variant 6: Natural language role description.

    Uses the semantic phrases users would actually search for, bridging
    the gap between stat-template training data and NL queries at inference.
    """
    zone = row["zone"]
    archetype_name = row.get("archetype_name")
    _arch_prefix = f"A {archetype_name}. " if archetype_name else ""
    gpm = row["goals_per_match"]
    spm = row["shots_per_match"]
    topm = row["take_ons_per_match"]
    cpm = row["crosses_per_match"]
    dpm = row["defensive_actions_per_match"]
    ppm = row["passes_per_match"]
    kpm = row["keeper_actions_per_match"]
    xt_gain = row.get("xt_gain_per_match", 0) or 0
    sca = row.get("shots_created_per_match", 0) or 0
    prog = row.get("progressive_passes_per_match", 0) or 0
    ft_passes = row.get("final_third_passes_per_match", 0) or 0

    if zone == "Goalkeeper":
        parts = ["A shot-stopping goalkeeper"]
        if ppm > 20:
            parts.append("comfortable on the ball with good distribution")
        if kpm > 2:
            parts.append("commanding in the box")
        result = parts[0] + (", " + " and ".join(parts[1:]) if len(parts) > 1 else "") + "."
        return _arch_prefix + result

    if zone == "Centre-back":
        parts = ["A tall centre-back, dominant in aerial duels"]
        if dpm > 5:
            parts.append("strong in the tackle")
        if ppm > 30:
            parts.append("capable of playing out from the back")
        if prog > 2:
            parts.append("progressive in possession")
        extras = parts[1:]
        if extras:
            result = parts[0] + ", " + ", ".join(extras) + "."
        else:
            result = parts[0] + "."
        return _arch_prefix + result

    if zone == "Full-back":
        if cpm > 1.5:
            base = "An overlapping full-back who delivers crosses from wide areas"
        elif dpm > 4:
            base = "A defensively solid full-back who prioritizes stopping attacks"
        else:
            base = "A versatile full-back who attacks and defends"
        extras = []
        if topm > 1.5:
            extras.append("willing to take on opponents")
        if sca > 1:
            extras.append("creates chances from wide positions")
        if extras:
            result = base + ", " + " and ".join(extras) + "."
        else:
            result = base + "."
        return _arch_prefix + result

    if zone == "Defensive midfielder":
        base = "A deep-lying midfielder who controls the tempo"
        extras = []
        if dpm > 5:
            extras.append("breaks up opposition play with tackles and interceptions")
        if ppm > 35:
            extras.append("dictates possession with high pass volume")
        if prog > 3:
            extras.append("progresses play with forward passing")
        if extras:
            result = base + ". " + extras[0].capitalize() + ("." if len(extras) == 1 else " and " + extras[1] + ".")
        else:
            result = base + "."
        return _arch_prefix + result

    if zone == "Central midfielder":
        base = "An energetic box-to-box midfielder with high work rate"
        extras = []
        if dpm > 3 and spm > 0.5:
            extras.append("covers ground in both directions")
        elif dpm > 3:
            extras.append("energetic in defensive transitions")
        elif spm > 1:
            extras.append("gets forward to support attacks")
        if sca > 1:
            extras.append("creates shooting opportunities for teammates")
        if extras:
            result = base + " who " + " and ".join(extras) + "."
        else:
            result = base + "."
        return _arch_prefix + result

    if zone == "Attacking midfielder":
        base = "A creative playmaker with excellent passing vision"
        extras = []
        if gpm > 0.2:
            extras.append("chips in with goals regularly")
        if ft_passes > 5:
            extras.append("threads passes into dangerous areas")
        if sca > 2:
            extras.append("creates numerous chances for teammates")
        if extras:
            result = base + " who " + " and ".join(extras[:2]) + "."
        else:
            result = base + "."
        return _arch_prefix + result

    if zone == "Winger":
        base = "A fast winger"
        extras = []
        if topm > 2:
            extras.append("takes on defenders with pace and skill")
        if cpm > 2:
            extras.append("delivers dangerous crosses")
        elif gpm > 0.2:
            extras.append("cuts inside to shoot")
        if sca > 1.5:
            extras.append("creates chances from wide areas")
        if extras:
            result = base + " who " + " and ".join(extras[:2]) + "."
        else:
            result = base + "."
        return _arch_prefix + result

    if zone in ("Striker", "Forward"):
        base = "A prolific striker"
        extras = []
        if gpm > 0.3:
            extras.append("scores goals consistently")
        elif spm > 2:
            extras.append("takes plenty of shots")
        if topm > 1.5:
            extras.append("can beat defenders on the dribble")
        if extras:
            result = base + " who " + " and ".join(extras) + "."
        else:
            result = base + "."
        return _arch_prefix + result

    return _arch_prefix + _ZONE_PREAMBLE.get(zone, zone + ".")


# ---------------------------------------------------------------------------
# 5b. Archetype text variants (require archetype labels)
# ---------------------------------------------------------------------------

# Mapping from archetype name to the stat columns that most differentiate it
# from other archetypes in the same zone. Each value is a list of
# (stat_col, display_label, is_rate) tuples.
_ARCHETYPE_DIFFERENTIATORS = {
    # Goalkeeper archetypes
    "Shot-stopper": [
        ("keeper_save_per_match", "saves/match", False),
        ("keeper_claim_per_match", "claims/match", False),
        ("keeper_actions_per_match", "keeper actions/match", False),
    ],
    "Sweeper-keeper": [
        ("avg_start_x", "avg starting position", False),
        ("passes_per_match", "passes/match", False),
        ("pass_completion", "pass completion", True),
    ],
    "Ball-playing GK": [
        ("pass_completion", "pass completion", True),
        ("progressive_passes_per_match", "progressive passes/match", False),
        ("avg_pass_length", "avg pass length", False),
    ],
    # Centre-back archetypes
    "Stopper CB": [
        ("tackles_per_match", "tackles/match", False),
        ("interceptions_per_match", "interceptions/match", False),
        ("defensive_actions_per_match", "defensive actions/match", False),
    ],
    "Ball-playing CB": [
        ("pass_completion", "pass completion", True),
        ("progressive_passes_per_match", "progressive passes/match", False),
        ("avg_pass_length", "avg pass length", False),
    ],
    "Wide CB": [
        ("std_start_y", "lateral spread", False),
        ("half_space_left_pct", "left half-space activity", True),
        ("half_space_right_pct", "right half-space activity", True),
        ("crosses_per_match", "crosses/match", False),
    ],
    "Commanding CB": [
        ("clearances_per_match", "clearances/match", False),
        ("aerial_pct", "aerial action rate", True),
        ("defensive_actions_per_match", "defensive actions/match", False),
    ],
    # Full-back archetypes
    "Overlapping FB": [
        ("crosses_per_match", "crosses/match", False),
        ("forward_third_pct", "forward-third activity", True),
        ("xt_gain_per_match", "xT generated/match", False),
    ],
    "Inverted FB": [
        ("central_zone_pct", "central zone activity", True),
        ("passes_per_match", "passes/match", False),
        ("progressive_passes_per_match", "progressive passes/match", False),
    ],
    "Defensive FB": [
        ("defensive_actions_per_match", "defensive actions/match", False),
        ("tackles_per_match", "tackles/match", False),
    ],
    "Wing-back": [
        ("avg_start_x", "avg starting position", False),
        ("shots_per_match", "shots/match", False),
        ("xt_gain_per_match", "xT generated/match", False),
    ],
    # Defensive midfielder archetypes
    "Anchor DM": [
        ("defensive_actions_per_match", "defensive actions/match", False),
        ("tackles_per_match", "tackles/match", False),
        ("interceptions_per_match", "interceptions/match", False),
    ],
    "Regista": [
        ("pass_completion", "pass completion", True),
        ("progressive_passes_per_match", "progressive passes/match", False),
        ("xt_pass_gain_per_match", "xT from passes/match", False),
    ],
    "Box-to-box DM": [
        ("actions_per_match", "actions/match", False),
        ("forward_third_pct", "forward-third activity", True),
        ("defensive_actions_per_match", "defensive actions/match", False),
    ],
    # Central midfielder archetypes
    "Box-to-box CM": [
        ("actions_per_match", "actions/match", False),
        ("tackles_per_match", "tackles/match", False),
        ("forward_third_pct", "forward-third activity", True),
    ],
    "Deep playmaker": [
        ("progressive_passes_per_match", "progressive passes/match", False),
        ("pass_completion", "pass completion", True),
        ("xt_pass_gain_per_match", "xT from passes/match", False),
    ],
    "Mezzala": [
        ("forward_third_pct", "forward-third activity", True),
        ("goals_per_match", "goals/match", False),
        ("xt_gain_per_match", "xT generated/match", False),
    ],
    "Defensive CM": [
        ("defensive_actions_per_match", "defensive actions/match", False),
        ("tackles_per_match", "tackles/match", False),
        ("interceptions_per_match", "interceptions/match", False),
    ],
    # Attacking midfielder archetypes
    "Classic 10": [
        ("shots_created_per_match", "shots created/match", False),
        ("final_third_passes_per_match", "final-third passes/match", False),
        ("passes_into_box_per_match", "passes into box/match", False),
    ],
    "Shadow striker": [
        ("goals_per_match", "goals/match", False),
        ("shots_per_match", "shots/match", False),
        ("penalty_box_actions_pct", "penalty box activity", True),
    ],
    "Wide playmaker": [
        ("crosses_per_match", "crosses/match", False),
        ("shots_created_per_match", "shots created/match", False),
        ("half_space_left_pct", "left half-space activity", True),
        ("half_space_right_pct", "right half-space activity", True),
    ],
    "False 9": [
        ("avg_start_x", "avg starting position", False),
        ("take_ons_per_match", "take-ons/match", False),
        ("progressive_passes_per_match", "progressive passes/match", False),
    ],
    # Winger archetypes
    "Traditional winger": [
        ("crosses_per_match", "crosses/match", False),
        ("forward_third_pct", "forward-third activity", True),
    ],
    "Inside forward": [
        ("goals_per_match", "goals/match", False),
        ("shots_per_match", "shots/match", False),
        ("central_zone_pct", "central zone activity", True),
    ],
    "Inverted winger": [
        ("central_zone_pct", "central zone activity", True),
        ("pass_completion", "pass completion", True),
        ("progressive_passes_per_match", "progressive passes/match", False),
    ],
    "Goal-scoring winger": [
        ("goals_per_match", "goals/match", False),
        ("shot_conversion_rate", "shot conversion rate", True),
        ("shots_per_match", "shots/match", False),
    ],
}


def generate_archetype_description(row: pd.Series, pctiles: pd.Series) -> str:
    """Variant 7: Archetype-led description with differentiating stats.

    Opens with the archetype name prominently and lists the 3-4 stats that
    most differentiate this archetype from others in the same zone. Uses
    within-zone percentile context when available.

    Falls back to generate_profile() if no archetype_name in row.

    Args:
        row: Player stats row (must include 'zone', may include 'archetype_name').
        pctiles: Population percentiles for the player.

    Returns:
        Natural language description led by the archetype name.
    """
    archetype_name = row.get("archetype_name")
    if not archetype_name or (isinstance(archetype_name, float) and pd.isna(archetype_name)):
        return generate_profile(row)

    zone = row.get("zone", "Unknown")
    parts = [f"A {archetype_name} playing as a {zone}."]

    # Look up differentiating stats for this archetype
    diff_stats = _ARCHETYPE_DIFFERENTIATORS.get(archetype_name, [])

    stat_lines = []
    for stat_col, stat_label, is_rate in diff_stats:
        val = row.get(stat_col, 0) or 0
        if pd.isna(val):
            continue

        # Format the value
        if is_rate:
            val_str = _pct(val)
        else:
            val_str = _fmt(val)

        # Check percentile for within-zone context
        pctile_col = f"{stat_col}_pctile"
        pctile_val = pctiles.get(pctile_col, None)
        if pctile_val is not None and not pd.isna(pctile_val):
            pctile_ctx = _pctile_short(pctile_val)
            stat_lines.append(f"{stat_label.capitalize()} ({val_str}) is {pctile_ctx}.")
        else:
            stat_lines.append(f"{stat_label.capitalize()} at {val_str}.")

    if stat_lines:
        parts.append("Distinguished by " + stat_lines[0][0].lower() + stat_lines[0][1:])
        for line in stat_lines[1:4]:
            parts.append(line)

    # Add overall volume and pitch zone context
    ppm = row.get("passes_per_match", 0) or 0
    if ppm > 5:
        third = _zone_third_label(
            row.get("forward_third_pct", 0) or 0,
            row.get("defensive_third_pct", 0) or 0,
        )
        parts.append(f"Controls play with {_fmt(ppm)} passes per match, primarily {third}.")

    # xT contribution
    xt_gain = row.get("xt_gain_per_match", 0) or 0
    if xt_gain > 0.1:
        xt_source = "passing" if (row.get("xt_pass_gain_per_match", 0) or 0) > xt_gain * 0.6 else "all actions"
        parts.append(f"Generates {_fmt(xt_gain, 2)} xT per match through {xt_source}.")

    return " ".join(parts)


# Alias for the requested API name
generate_archetype = generate_archetype_description


def generate_comparative_archetype(row: pd.Series, pctiles: pd.Series) -> str:
    """Variant 8: Within-cluster percentile comparison.

    Opens with archetype context and shows how key stats rank among the
    population, framed as within-archetype ranking. Uses _pctile_short
    for readable percentile labels.

    Falls back to generate_comparative() if no archetype_name in row.

    Args:
        row: Player stats row (must include 'zone', may include 'archetype_name').
        pctiles: Population percentiles for the player.

    Returns:
        Natural language comparison framed around the archetype identity.
    """
    archetype_name = row.get("archetype_name")
    if not archetype_name or (isinstance(archetype_name, float) and pd.isna(archetype_name)):
        return generate_comparative(row, pctiles)

    parts = [f"Compared to other {archetype_name} players:"]

    # Look up the differentiating stats for this archetype
    diff_stats = _ARCHETYPE_DIFFERENTIATORS.get(archetype_name, [])

    comparisons = []
    for stat_col, stat_label, is_rate in diff_stats:
        val = row.get(stat_col, 0) or 0
        if pd.isna(val):
            continue

        pctile_col = f"{stat_col}_pctile"
        pctile_val = pctiles.get(pctile_col, None)
        if pctile_val is None or pd.isna(pctile_val):
            continue

        if is_rate:
            val_str = _pct(val)
        else:
            val_str = _fmt(val)

        rank_label = _pctile_short(pctile_val)
        comparisons.append(f"{stat_label.capitalize()} ({val_str}) is {rank_label}.")

    # Show up to 4 comparisons from the archetype differentiators
    for comp in comparisons[:4]:
        parts.append(comp)

    # Add broader stats for context (overall threat and defensive contribution)
    xt_pctile = pctiles.get("xt_gain_per_match_pctile", 50)
    xt_val = row.get("xt_gain_per_match", 0) or 0
    if not pd.isna(xt_pctile):
        parts.append(f"Overall threat generation ({_fmt(xt_val, 2)} xT/match) is "
                     f"{_pctile_short(xt_pctile)}.")

    dpm_pctile = pctiles.get("defensive_actions_per_match_pctile", 50)
    dpm_val = row.get("defensive_actions_per_match", 0) or 0
    if not pd.isna(dpm_pctile):
        parts.append(f"Defensive contribution ({_fmt(dpm_val)} actions/match) is "
                     f"{_pctile_short(dpm_pctile)}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# 5c. New text variants (pressing, creative, wikipedia hybrid)
# ---------------------------------------------------------------------------

def generate_pressing_profile(row: pd.Series, pctiles: pd.Series) -> str:
    """Variant 9: Pressing and defensive work-rate profile.

    Focuses on pressing intensity, counterpressing, turnover rate,
    and action entropy to paint a picture of the player's defensive
    identity and tactical versatility.

    Args:
        row: Player stats row.
        pctiles: Population percentiles for the player.

    Returns:
        Natural language pressing/work-rate profile.
    """
    zone = row["zone"]
    archetype_name = row.get("archetype_name")
    _arch_prefix = f"A {archetype_name}. " if archetype_name else ""
    preamble = _ZONE_PREAMBLE.get(zone, zone + ".")

    parts = [_arch_prefix + preamble]

    # Pressing actions
    pressing = row.get("pressing_actions_per_match", 0) or 0
    pressing_pctile = pctiles.get("pressing_actions_per_match_pctile", 50)
    if not pd.isna(pressing) and pressing > 0.1:
        label = _pctile_label(pressing_pctile) if not pd.isna(pressing_pctile) else "moderate"
        parts.append(f"Pressing intensity is {label} with {_fmt(pressing)} "
                     f"pressing actions per match.")
    else:
        parts.append("Limited pressing activity in the data.")

    # Counterpressing rate
    counterpress = row.get("counterpressing_rate", 0) or 0
    if not pd.isna(counterpress) and counterpress > 0.05:
        cp_pctile = pctiles.get("counterpressing_rate_pctile", 50)
        if not pd.isna(cp_pctile) and cp_pctile >= 75:
            parts.append(f"Aggressive in transition, counterpressing at "
                         f"{_pct(counterpress)} of opportunities.")
        elif not pd.isna(cp_pctile) and cp_pctile <= 25:
            parts.append(f"Selective in counterpressing ({_pct(counterpress)} rate), "
                         f"preferring to drop into shape.")
        else:
            parts.append(f"Counterpresses at a {_pct(counterpress)} rate after turnovers.")

    # Turnover rate
    turnover = row.get("turnover_rate", 0) or 0
    if not pd.isna(turnover) and turnover > 0:
        to_pctile = pctiles.get("turnover_rate_pctile", 50)
        if not pd.isna(to_pctile) and to_pctile >= 75:
            parts.append(f"Prone to losing possession (turnover rate {_pct(turnover)}).")
        elif not pd.isna(to_pctile) and to_pctile <= 25:
            parts.append(f"Reliable in possession with a low turnover rate ({_pct(turnover)}).")

    # Defensive actions context
    dpm = row.get("defensive_actions_per_match", 0) or 0
    tackles = row.get("tackles_per_match", 0) or 0
    interceptions = row.get("interceptions_per_match", 0) or 0
    if dpm > 3:
        parts.append(f"Contributes {_fmt(dpm)} defensive actions per match "
                     f"({_fmt(tackles)} tackles, {_fmt(interceptions)} interceptions).")

    # Action entropy (tactical identity)
    entropy_val = row.get("action_entropy", 0) or 0
    ent_pctile = pctiles.get("action_entropy_pctile", 50)
    if not pd.isna(entropy_val) and entropy_val > 0:
        if not pd.isna(ent_pctile) and ent_pctile >= 75:
            parts.append("Performs a diverse range of actions, suggesting tactical flexibility.")
        elif not pd.isna(ent_pctile) and ent_pctile <= 25:
            parts.append("Focused on a specific set of duties, a role specialist.")

    # VAEP defensive value
    vaep_def = row.get("vaep_defensive_per_match", 0) or 0
    if not pd.isna(vaep_def) and vaep_def > 0.05:
        vd_pctile = pctiles.get("vaep_defensive_per_match_pctile", 50)
        if not pd.isna(vd_pctile) and vd_pctile >= 70:
            parts.append(f"High defensive value ({_fmt(vaep_def, 2)} VAEP/match), "
                         f"recovering the ball in dangerous areas.")

    return " ".join(parts)


def generate_creative_profile(row: pd.Series, pctiles: pd.Series) -> str:
    """Variant 10: Creative and progressive profile.

    Focuses on pass network centrality, progressive carries, carries
    into box, pre-assists, and xT extensions to capture the player's
    creative and ball-progression identity.

    Args:
        row: Player stats row.
        pctiles: Population percentiles for the player.

    Returns:
        Natural language creative/progressive profile.
    """
    zone = row["zone"]
    archetype_name = row.get("archetype_name")
    _arch_prefix = f"A {archetype_name}. " if archetype_name else ""
    preamble = _ZONE_PREAMBLE.get(zone, zone + ".")

    parts = [_arch_prefix + preamble]

    # Pass network centrality
    out_degree = row.get("pass_out_degree", 0) or 0
    in_degree = row.get("pass_in_degree", 0) or 0
    betweenness = row.get("pass_betweenness", 0) or 0
    clustering = row.get("pass_clustering_coeff", 0) or 0

    out_pctile = pctiles.get("pass_out_degree_pctile", 50)
    between_pctile = pctiles.get("pass_betweenness_pctile", 50)

    if not pd.isna(out_pctile) and out_pctile >= 75:
        parts.append(f"A hub in the passing network, distributing the ball widely "
                     f"(out-degree {_pctile_short(out_pctile)}).")
    elif not pd.isna(in_pctile := pctiles.get("pass_in_degree_pctile", 50)) and in_pctile >= 75:
        parts.append("Frequently sought out by teammates as a passing target.")

    if not pd.isna(between_pctile) and between_pctile >= 70:
        parts.append("Acts as a key connector between different areas of the team's build-up.")

    if not pd.isna(clustering) and clustering > 0:
        clust_pctile = pctiles.get("pass_clustering_coeff_pctile", 50)
        if not pd.isna(clust_pctile) and clust_pctile >= 75:
            parts.append("Engages in tight passing combinations with nearby teammates.")
        elif not pd.isna(clust_pctile) and clust_pctile <= 25:
            parts.append("Prefers switching play rather than short passing triangles.")

    # Progressive carries and carries into box
    prog_carry = row.get("progressive_carry_pct", 0) or 0
    box_carry = row.get("carries_into_box_per_match", 0) or 0
    carry_dist = row.get("avg_carry_distance", 0) or 0

    if not pd.isna(prog_carry) and prog_carry > 0.15:
        parts.append(f"Progressive ball carrier, driving forward on {_pct(prog_carry)} of carries.")
    if not pd.isna(box_carry) and box_carry > 0.3:
        parts.append(f"Dangerous with the ball at feet, carrying into the box "
                     f"{_fmt(box_carry)} times per match.")
    elif not pd.isna(carry_dist) and carry_dist > 8:
        parts.append(f"Covers {_fmt(carry_dist, 1)}m on average per carry.")

    # Pre-assists
    pre_assist = row.get("pre_assist_per_match", 0) or 0
    if not pd.isna(pre_assist) and pre_assist > 0.1:
        pa_pctile = pctiles.get("pre_assist_per_match_pctile", 50)
        if not pd.isna(pa_pctile) and pa_pctile >= 70:
            parts.append(f"Initiates attacking sequences with {_fmt(pre_assist)} "
                         f"pre-assists per match.")

    # xT extensions
    xt_carry = row.get("xt_carry_gain_per_match", 0) or 0
    xt_loss = row.get("xt_loss_per_match", 0) or 0
    xt_std = row.get("xt_gain_std", 0) or 0
    xt_total = row.get("xt_gain_per_match", 0) or 0

    if not pd.isna(xt_carry) and xt_carry > 0.1:
        parts.append(f"Creates {_fmt(xt_carry, 2)} xT per match from carries alone.")

    if not pd.isna(xt_total) and xt_total > 0.3:
        net_xt = xt_total - xt_loss if not pd.isna(xt_loss) else xt_total
        if net_xt > 0.2:
            parts.append(f"Strong net threat contributor (generates {_fmt(xt_total, 2)} xT, "
                         f"loses {_fmt(xt_loss, 2)}).")

    # xT volatility
    if not pd.isna(xt_std) and xt_std > 0:
        xt_std_pctile = pctiles.get("xt_gain_std_pctile", 50)
        if not pd.isna(xt_std_pctile) and xt_std_pctile >= 75:
            parts.append("Creates threat in bursts rather than steady accumulation.")

    # VAEP offensive
    vaep_off = row.get("vaep_offensive_per_match", 0) or 0
    if not pd.isna(vaep_off) and vaep_off > 0.1:
        vo_pctile = pctiles.get("vaep_offensive_per_match_pctile", 50)
        if not pd.isna(vo_pctile) and vo_pctile >= 70:
            parts.append(f"Offensive value ({_fmt(vaep_off, 2)} VAEP/match) "
                         f"ranks {_pctile_short(vo_pctile)}.")

    return " ".join(parts)


def generate_wikipedia_hybrid(
    row: pd.Series,
    pctiles: pd.Series,
    wiki_lookup: dict[str, str] | None = None,
) -> str | None:
    """Variant 11: Wikipedia + stat hybrid description.

    For players with a Wikipedia entry, blends wiki text context with
    key stat highlights. For players without wiki text, returns None
    so the caller can skip this variant.

    Args:
        row: Player stats row (must include 'player_name' or 'zone').
        pctiles: Population percentiles for the player.
        wiki_lookup: dict mapping player_name -> wiki text content.

    Returns:
        Hybrid description string, or None if no wiki text available.
    """
    if wiki_lookup is None:
        return None

    player_name = row.get("player_name")
    if not player_name or pd.isna(player_name):
        return None

    wiki_text = wiki_lookup.get(player_name)
    if not wiki_text or pd.isna(wiki_text):
        return None

    zone = row["zone"]
    preamble = _ZONE_PREAMBLE.get(zone, zone + ".")

    # Truncate wiki text to first 2 sentences for brevity
    sentences = wiki_text.replace("\n", " ").split(". ")
    wiki_snippet = ". ".join(sentences[:2]).strip()
    if wiki_snippet and not wiki_snippet.endswith("."):
        wiki_snippet += "."
    # Cap at 300 chars
    if len(wiki_snippet) > 300:
        wiki_snippet = wiki_snippet[:297] + "..."

    parts = [preamble, wiki_snippet]

    # Add stat context that complements the wiki bio
    ppm = row.get("passes_per_match", 0) or 0
    gpm = row.get("goals_per_match", 0) or 0
    dpm = row.get("defensive_actions_per_match", 0) or 0
    xt_gain = row.get("xt_gain_per_match", 0) or 0
    sca = row.get("shots_created_per_match", 0) or 0

    stat_highlights = []
    if gpm > 0.2:
        stat_highlights.append(f"averages {_fmt(gpm)} goals per match")
    if not pd.isna(xt_gain) and xt_gain > 0.5:
        stat_highlights.append(f"generates {_fmt(xt_gain, 2)} xT per match")
    if dpm > 4:
        stat_highlights.append(f"contributes {_fmt(dpm)} defensive actions per match")
    if not pd.isna(sca) and sca > 1.0:
        stat_highlights.append(f"creates {_fmt(sca)} shots per match for teammates")
    if ppm > 40:
        stat_highlights.append(f"averages {_fmt(ppm)} passes per match")

    if stat_highlights:
        parts.append("Statistically, " + ", ".join(stat_highlights[:3]) + ".")

    # Progressive profile addition
    prog = row.get("progressive_passes_per_match", 0) or 0
    prog_carry = row.get("progressive_carry_pct", 0) or 0
    if not pd.isna(prog) and prog > 3:
        parts.append(f"Progresses the ball with {_fmt(prog)} progressive passes per match.")
    if not pd.isna(prog_carry) and prog_carry > 0.2:
        parts.append(f"Drives forward on {_pct(prog_carry)} of ball carries.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def generate_all_descriptions(
    stats: pd.DataFrame,
    pctiles: pd.DataFrame,
    lookup: pd.DataFrame | None = None,
    archetype_df: pd.DataFrame | None = None,
    wiki_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Generate text variants for every player in the stats DataFrame.

    Produces 8 base variants per player (6 original + pressing + creative).
    When archetype_df is provided, injects archetype labels and adds 2 more
    archetype-specific variants (10 total with archetypes, 8 without).
    When wiki_lookup is provided, adds a wikipedia hybrid variant for players
    with wiki text.

    Args:
        stats: Per-player stats (output of compute_player_stats).
        pctiles: Percentile ranks (output of compute_percentiles).
        lookup: Optional players_lookup DataFrame to join player_name.
        archetype_df: Optional DataFrame with columns (player_name, archetype_name, zone).
            When provided, archetype labels are injected into text variants.
            Players not in archetype_df still get the base variants without
            archetype labels.
        wiki_lookup: Optional dict mapping player_name -> wiki text content.
            When provided, adds stat_wikipedia_hybrid variant for matched players.

    Returns:
        DataFrame with columns: player_id, text_type, text_content, player_name.
    """
    # Build archetype lookup: player_id -> archetype_name
    archetype_map: dict = {}
    if archetype_df is not None:
        # Join archetype_df to stats via player_name through lookup
        if lookup is not None and "player_name" in lookup.columns:
            # Map player_name -> player_id from lookup
            name_to_id = (
                lookup[["player_id", "player_name"]]
                .drop_duplicates(subset="player_name")
                .set_index("player_name")["player_id"]
            )
            for _, arow in archetype_df.iterrows():
                pname = arow.get("player_name")
                aname = arow.get("archetype_name")
                if pname and aname and pname in name_to_id.index:
                    pid = name_to_id[pname]
                    archetype_map[pid] = aname
        else:
            # If archetype_df has player_id directly
            if "player_id" in archetype_df.columns:
                for _, arow in archetype_df.iterrows():
                    pid = arow.get("player_id")
                    aname = arow.get("archetype_name")
                    if pid is not None and aname:
                        archetype_map[pid] = aname

    # Build player_id -> player_name map for wiki hybrid
    pid_to_name: dict = {}
    if lookup is not None and "player_name" in lookup.columns:
        name_map_df = (
            lookup[["player_id", "player_name"]]
            .drop_duplicates(subset="player_id")
        )
        pid_to_name = dict(zip(name_map_df["player_id"], name_map_df["player_name"]))

    rows = []

    for player_id, row in stats.iterrows():
        p_row = pctiles.loc[player_id] if player_id in pctiles.index else pd.Series(dtype=float)

        # Inject archetype_name into the row if available
        aname = archetype_map.get(player_id)
        if aname:
            row = row.copy()
            row["archetype_name"] = aname

        texts = {
            "stat_scouting": generate_scouting(row),
            "stat_comparative": generate_comparative(row, p_row),
            "stat_profile": generate_profile(row),
            "stat_strengths": generate_strengths(row, p_row),
            "stat_statistical": generate_statistical(row),
            "stat_nl_role": generate_nl_role(row),
            "stat_pressing_profile": generate_pressing_profile(row, p_row),
            "stat_creative_profile": generate_creative_profile(row, p_row),
        }

        # Add archetype-specific variants when archetype is available
        if aname:
            texts["stat_archetype"] = generate_archetype_description(row, p_row)
            texts["stat_comparative_archetype"] = generate_comparative_archetype(
                row, p_row,
            )

        # Add wikipedia hybrid variant if wiki text available for this player
        if wiki_lookup is not None:
            pname = pid_to_name.get(player_id)
            if pname:
                wiki_row = row.copy() if aname else row
                if not aname:
                    wiki_row = row.copy()
                wiki_row["player_name"] = pname
                hybrid = generate_wikipedia_hybrid(wiki_row, p_row, wiki_lookup)
                if hybrid is not None:
                    texts["stat_wikipedia_hybrid"] = hybrid

        for text_type, text_content in texts.items():
            rows.append({
                "player_id": player_id,
                "text_type": text_type,
                "text_content": text_content,
            })

    result = pd.DataFrame(rows)

    # Join player names if lookup provided
    if lookup is not None and "player_name" in lookup.columns:
        name_map = (
            lookup[["player_id", "player_name"]]
            .drop_duplicates(subset="player_id")
            .set_index("player_id")["player_name"]
        )
        result["player_name"] = result["player_id"].map(name_map)
    else:
        result["player_name"] = None

    # Reorder columns
    result = result[["player_id", "player_name", "text_type", "text_content"]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate natural language stat descriptions from SPADL action data"
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
        help="Path to players_lookup parquet for name mapping (default: data/processed/players_lookup.parquet)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/text/stat_descriptions.parquet",
        help="Output parquet path (default: data/processed/text/stat_descriptions.parquet)",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=3,
        help="Minimum matches to include a player (default: 3)",
    )
    parser.add_argument(
        "--archetype-path",
        type=str,
        default=None,
        help="Path to archetype parquet (output of archetypes.py) for archetype-aware variants",
    )
    parser.add_argument(
        "--wiki-path",
        type=str,
        default="data/processed/text/wikipedia_intros.parquet",
        help="Path to Wikipedia texts parquet (default: data/processed/text/wikipedia_intros.parquet)",
    )
    parser.add_argument(
        "--all-text-output",
        type=str,
        default="data/processed/text/all_text_data_v3.parquet",
        help="Path to save consolidated all_text_data output (default: data/processed/text/all_text_data_v3.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats and print samples without writing output",
    )
    args = parser.parse_args()

    spadl_path = Path(args.spadl_path)
    lookup_path = Path(args.lookup_path)
    output_path = Path(args.output_path)

    # Load SPADL
    print(f"Loading SPADL data from {spadl_path}...")
    df = pd.read_parquet(spadl_path)
    print(f"  {len(df):,} actions, {df['player_id'].nunique():,} players, "
          f"{df['game_id'].nunique():,} matches")

    # Compute per-player stats
    print("Computing per-player stats...")
    stats = compute_player_stats(df)
    print(f"  {len(stats):,} players computed")

    # Filter by min matches
    before = len(stats)
    stats = stats[stats["matches_played"] >= args.min_matches].copy()
    print(f"  Filtered to {len(stats):,} players (>= {args.min_matches} matches, "
          f"dropped {before - len(stats):,})")

    # Classify zones
    print("Classifying playing zones...")
    stats["zone"] = stats.apply(classify_zone, axis=1)
    zone_counts = stats["zone"].value_counts()
    for zone, count in zone_counts.items():
        print(f"  {zone}: {count:,}")

    # Compute percentiles
    print("Computing population percentiles...")
    pctiles = compute_percentiles(stats)

    # Load lookup
    lookup = None
    if lookup_path.exists():
        print(f"Loading player lookup from {lookup_path}...")
        lookup = pd.read_parquet(lookup_path)
        print(f"  {len(lookup):,} rows, {lookup['player_id'].nunique():,} unique players")
    else:
        print(f"  [warn] Lookup file not found: {lookup_path}")

    # Load archetype data if provided
    archetype_df = None
    if args.archetype_path:
        archetype_path = Path(args.archetype_path)
        if archetype_path.exists():
            print(f"Loading archetype data from {archetype_path}...")
            archetype_df = pd.read_parquet(archetype_path)
            print(f"  {len(archetype_df):,} rows, "
                  f"{archetype_df['archetype_name'].nunique():,} archetypes")
        else:
            print(f"  [warn] Archetype file not found: {archetype_path}")

    # Load Wikipedia texts for hybrid descriptions
    wiki_lookup: dict[str, str] | None = None
    wiki_path = Path(args.wiki_path)
    wiki_df = None
    if wiki_path.exists():
        print(f"Loading Wikipedia texts from {wiki_path}...")
        wiki_df = pd.read_parquet(wiki_path)
        wiki_lookup = dict(zip(wiki_df["player_name"], wiki_df["wiki_text"]))
        print(f"  {len(wiki_lookup):,} players with Wikipedia text")
    else:
        print(f"  [warn] Wikipedia texts not found: {wiki_path}")

    # Generate descriptions
    base_count = 8  # 6 original + pressing + creative
    variant_label = f"{base_count + 2} variants" if archetype_df is not None else f"{base_count} variants"
    if wiki_lookup:
        variant_label += " + wikipedia hybrid"
    print(f"Generating text descriptions ({variant_label} per player)...")
    result = generate_all_descriptions(
        stats, pctiles, lookup,
        archetype_df=archetype_df,
        wiki_lookup=wiki_lookup,
    )
    print(f"  {len(result):,} total descriptions for {result['player_id'].nunique():,} players")

    # Summary by text_type
    for text_type, group in result.groupby("text_type"):
        print(f"  {text_type}: {len(group):,} texts")

    # Print samples
    print("\n--- Sample descriptions (first player with name) ---")
    sample_player = result[result["player_name"].notna()].iloc[0]["player_id"] if result["player_name"].notna().any() else result.iloc[0]["player_id"]
    sample = result[result["player_id"] == sample_player]
    sample_name = sample.iloc[0]["player_name"] if pd.notna(sample.iloc[0]["player_name"]) else f"player_id={sample_player}"
    print(f"Player: {sample_name}")
    for _, row in sample.iterrows():
        print(f"\n  [{row['text_type']}]")
        print(f"  {row['text_content']}")

    if args.dry_run:
        print("\n[dry-run] Skipping file write.")
        return

    # Write stat descriptions output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\nSaved {len(result):,} descriptions to {output_path}")

    # Quick quality stats
    avg_len = result["text_content"].str.len().mean()
    min_len = result["text_content"].str.len().min()
    max_len = result["text_content"].str.len().max()
    print(f"Text length: avg={avg_len:.0f}, min={min_len}, max={max_len} chars")

    named = result["player_name"].notna().sum()
    unnamed = result["player_name"].isna().sum()
    print(f"Named: {named:,}, unnamed: {unnamed:,}")

    # Save consolidated all_text_data_v3 (stat descriptions + wikipedia intros)
    all_text_output = Path(args.all_text_output)
    all_text_parts = [result[["player_name", "text_type", "text_content"]]]
    if wiki_df is not None:
        # Add raw wikipedia texts (not the hybrid, which is already in result)
        wiki_rows = wiki_df[["player_name", "wiki_text"]].rename(columns={"wiki_text": "text_content"})
        wiki_rows["text_type"] = "wikipedia_raw"
        all_text_parts.append(wiki_rows[["player_name", "text_type", "text_content"]])
    all_text = pd.concat(all_text_parts, ignore_index=True)
    all_text.to_parquet(all_text_output, index=False)
    print(f"\nSaved consolidated {len(all_text):,} texts to {all_text_output}")
    for tt, grp in all_text.groupby("text_type"):
        print(f"  {tt}: {len(grp):,}")


if __name__ == "__main__":
    main()
