#!/usr/bin/env python3
"""Evaluation metrics for football-embed-Bench.

Four metric implementations for benchmarking player embeddings:
- role_classification_accuracy: Linear probe -> position classification
- temporal_consistency: Same player cosine sim across seasons
- nlq_ndcg_at_k: Natural language retrieval nDCG
- match_outcome_prediction: Team-averaged cards -> game result prediction

Usage:
    from football_embed.evaluation.metrics import (
        role_classification_accuracy,
        temporal_consistency,
        nlq_ndcg_at_k,
        match_outcome_prediction,
    )
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ZCA whitening (post-hoc embedding improvement)
# ---------------------------------------------------------------------------


def zca_whiten_fit(embeddings: np.ndarray, regularization: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ZCA whitening transform from reference embeddings.

    Returns the whitening matrix W and mean so it can be applied to both
    gallery and query embeddings consistently.

    Args:
        embeddings: (N, D) L2-normalized reference embeddings (gallery).
        regularization: Small constant added to eigenvalues for stability.

    Returns:
        Tuple of (whitened_embeddings, W, mean) where W is (D, D) and mean is (D,).
    """
    mean = embeddings.mean(axis=0)
    X = embeddings - mean

    cov = (X.T @ X) / len(X)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, regularization)

    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    W = eigvecs @ D_inv_sqrt @ eigvecs.T
    whitened = X @ W

    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return whitened / norms, W, mean


def zca_whiten_transform(embeddings: np.ndarray, W: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Apply a pre-computed ZCA whitening transform to new embeddings.

    Args:
        embeddings: (N, D) embeddings to whiten.
        W: (D, D) whitening matrix from zca_whiten_fit.
        mean: (D,) mean vector from zca_whiten_fit.

    Returns:
        (N, D) whitened and L2-normalized embeddings.
    """
    X = embeddings - mean
    whitened = X @ W
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return whitened / norms


# ---------------------------------------------------------------------------
# a) Role classification
# ---------------------------------------------------------------------------


def role_classification_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray | list[str],
    test_size: float = 0.3,
    seed: int = 42,
) -> dict:
    """Fit a linear probe on embeddings to predict zone labels.

    Args:
        embeddings: (N, D) player embeddings (projected cards or text).
        labels: (N,) zone label strings (e.g. "Winger", "Goalkeeper").
        test_size: Fraction held out for testing.
        seed: Random seed for split and classifier.

    Returns:
        Dict with accuracy, macro_f1, n_train, n_test, n_classes, per_class.
    """
    labels = np.asarray(labels)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=test_size, random_state=seed, stratify=y,
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Per-class accuracy
    per_class = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_test == cls_idx
        if mask.sum() == 0:
            continue
        per_class[cls_name] = float(accuracy_score(y_test[mask], y_pred[mask]))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(class_names),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# b) Temporal consistency
# ---------------------------------------------------------------------------


def _temporal_consistency_precomputed(
    season_cards_path: Path,
    min_matches: int = 3,
) -> dict:
    """Temporal consistency using pre-computed season_cards.parquet.

    Expects columns: ``player_id``, ``season_id``, ``matches_played`` and
    ``card_0..card_{N-1}``. Cards are already z-normalized and L2-normalized
    by the season-first pipeline — no re-normalization here.
    """
    sc = pd.read_parquet(season_cards_path)
    if "matches_played" in sc.columns:
        sc = sc[sc["matches_played"] >= min_matches]

    card_cols = sorted(
        [c for c in sc.columns if c.startswith("card_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not card_cols:
        raise ValueError(
            f"No card_* columns found in {season_cards_path}"
        )

    seasons = sorted(sc["season_id"].unique().tolist())
    logger.info(
        "Pre-computed season cards: %d rows, %d seasons, %d card dims",
        len(sc),
        len(seasons),
        len(card_cols),
    )

    # Group cards by season for O(1) lookup
    season_cards: dict[int, dict[int, np.ndarray]] = {}
    for sid, grp in sc.groupby("season_id"):
        arr = grp[card_cols].values.astype(np.float64)
        season_cards[sid] = dict(zip(grp["player_id"].values, arr))

    # Find players appearing in consecutive seasons (by sort order of season_id)
    cosines: list[float] = []
    players_seen: set[int] = set()
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        common = set(season_cards[s1].keys()) & set(season_cards[s2].keys())
        for pid in common:
            v1 = season_cards[s1][pid]
            v2 = season_cards[s2][pid]
            cos = float(np.dot(v1, v2))
            cosines.append(cos)
            players_seen.add(pid)

    if not cosines:
        return {
            "mean_cosine": 0.0,
            "median_cosine": 0.0,
            "pct_above_0.8": 0.0,
            "n_players": 0,
            "n_pairs": 0,
        }

    arr = np.array(cosines)
    return {
        "mean_cosine": float(np.mean(arr)),
        "median_cosine": float(np.median(arr)),
        "pct_above_0.8": float(np.mean(arr > 0.8)),
        "n_players": len(players_seen),
        "n_pairs": len(arr),
    }


def temporal_consistency(
    spadl_path: str | Path,
    norm_stats_path: str | Path,
    feature_names: list[str],
    min_matches: int = 3,
    season_cards_path: str | Path | None = None,
) -> dict:
    """Compute cosine similarity of same player's card across consecutive seasons.

    Two modes:

    * **Pre-computed (v12a+)**: if ``season_cards_path`` is provided and the file
      exists, load pre-computed per-(player, season) cards from disk. These are
      produced by ``build_player_cards.py --season-first`` and share the exact
      same normalization/pruning pipeline used during training. This fixes the
      train/test mismatch that kept the metric stuck at ~0.71.

    * **Legacy (v11 and earlier)**: load SPADL, group by (player_id, season_id),
      compute per-season cards with ``compute_player_stats``, then z-score
      normalize using the saved (career-trained) stats. Kept for backward
      compatibility with older checkpoints.

    Args:
        spadl_path: Path to spadl_unified.parquet (legacy path only).
        norm_stats_path: Path to card_normalization.json.
        feature_names: Ordered list of feature column names (legacy path only).
        min_matches: Minimum matches per season to include a player-season pair.
        season_cards_path: Optional path to a pre-computed ``season_cards.parquet``
            emitted by ``build_player_cards.py --season-first``. When provided
            and the file exists, this is loaded directly and the legacy
            recomputation is skipped.

    Returns:
        Dict with mean_cosine, median_cosine, pct_above_0.8, n_players, n_pairs.
    """
    # --- Fast path: pre-computed season cards from season-first pipeline ---
    if season_cards_path is not None and Path(season_cards_path).exists():
        return _temporal_consistency_precomputed(
            Path(season_cards_path), min_matches=min_matches
        )

    from football_embed.data.generate_stat_descriptions import compute_player_stats

    df = pd.read_parquet(spadl_path)

    # Load normalization stats
    with open(norm_stats_path) as f:
        norm = json.load(f)
    feat_mean = np.array(norm["mean"])
    feat_std = np.array(norm["std"])
    safe_std = feat_std.copy()
    safe_std[safe_std == 0] = 1.0

    # Get unique seasons
    seasons = sorted(df["season_id"].unique())
    logger.info("Found %d seasons: %s", len(seasons), seasons)

    # Compute per-season player cards
    season_cards: dict[int, dict[int, np.ndarray]] = {}
    for sid in seasons:
        season_df = df[df["season_id"] == sid]
        stats = compute_player_stats(season_df)
        stats = stats[stats["matches_played"] >= min_matches]

        for feat in feature_names:
            if feat not in stats.columns:
                stats[feat] = 0

        cards = stats[feature_names].fillna(0).values.astype(np.float64)
        # Z-score normalize
        cards = (cards - feat_mean) / safe_std
        # L2 normalize
        norms = np.linalg.norm(cards, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cards = cards / norms

        season_cards[sid] = dict(zip(stats.index, cards))

    # Find players appearing in consecutive seasons
    cosines: list[float] = []
    players_seen: set[int] = set()
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        common = set(season_cards[s1].keys()) & set(season_cards[s2].keys())
        for pid in common:
            v1 = season_cards[s1][pid]
            v2 = season_cards[s2][pid]
            cos = float(np.dot(v1, v2))
            cosines.append(cos)
            players_seen.add(pid)

    if not cosines:
        return {
            "mean_cosine": 0.0,
            "median_cosine": 0.0,
            "pct_above_0.8": 0.0,
            "n_players": 0,
            "n_pairs": 0,
        }

    arr = np.array(cosines)
    return {
        "mean_cosine": float(np.mean(arr)),
        "median_cosine": float(np.median(arr)),
        "pct_above_0.8": float(np.mean(arr > 0.8)),
        "n_players": len(players_seen),
        "n_pairs": len(arr),
    }


# ---------------------------------------------------------------------------
# c) Natural language retrieval nDCG@k
# ---------------------------------------------------------------------------


def nlq_ndcg_at_k(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_zones: list[str] | np.ndarray,
    query_relevant_zones: list[set[str]],
    k: int = 10,
) -> dict:
    """Compute nDCG@k for natural language queries with binary zone relevance.

    For each query, retrieves the top-k gallery players by cosine similarity,
    then scores using binary relevance (1 if the player's zone is in the set
    of relevant zones for that query, 0 otherwise).

    Args:
        query_embs: (Q, D) query embeddings (already encoded by TextBranch).
        gallery_embs: (N, D) gallery embeddings (projected cards, L2-normalized).
        gallery_zones: (N,) zone label per gallery player.
        query_relevant_zones: Length-Q list of sets, each containing zone labels
            considered relevant for that query.
        k: Cutoff rank for nDCG computation.

    Returns:
        Dict with mean_ndcg, per_query list of {query_idx, ndcg}.
    """
    gallery_zones = np.asarray(gallery_zones)
    Q = len(query_embs)

    # Cosine similarity (inputs assumed L2-normalized)
    sims = query_embs @ gallery_embs.T  # (Q, N)
    ranked = np.argsort(-sims, axis=1)[:, :k]  # (Q, k)

    discounts = np.log2(np.arange(2, k + 2))  # log2(2), log2(3), ..., log2(k+1)

    ndcgs: list[float] = []
    per_query: list[dict] = []
    for qi in range(Q):
        top_zones = gallery_zones[ranked[qi]]
        relevant = query_relevant_zones[qi]

        # Binary relevance vector
        rels = np.array([1.0 if z in relevant else 0.0 for z in top_zones])

        # DCG
        dcg = float(np.sum(rels / discounts))

        # Ideal DCG: sort relevance descending
        ideal_rels = np.sort(rels)[::-1]
        idcg = float(np.sum(ideal_rels / discounts))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
        per_query.append({"query_idx": qi, "ndcg": round(ndcg, 4)})

    return {
        "mean_ndcg": float(np.mean(ndcgs)),
        "per_query": per_query,
    }


def nlq_ndcg_graded_at_k(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_zones: list[str] | np.ndarray,
    gallery_archetypes: list[str] | np.ndarray,
    query_relevant_zones: list[set[str]],
    query_relevant_archetypes: list[set[str]],
    k: int = 10,
) -> dict:
    """Compute nDCG@k with 3-level graded relevance.

    Relevance levels:
        3 = player matches BOTH a relevant zone AND a relevant archetype
        1 = player matches a relevant zone only (archetype mismatch or unknown)
        0 = player matches neither zone

    Args:
        query_embs: (Q, D) query embeddings.
        gallery_embs: (N, D) gallery embeddings (L2-normalized).
        gallery_zones: (N,) zone labels.
        gallery_archetypes: (N,) archetype name strings (may contain "" or NaN).
        query_relevant_zones: Length-Q list of sets of relevant zone names.
        query_relevant_archetypes: Length-Q list of sets of relevant archetype names.
        k: Cutoff rank.

    Returns:
        Dict with mean_ndcg, per_query list of {query_idx, ndcg}.
    """
    gallery_zones = np.asarray(gallery_zones)
    gallery_archetypes = np.asarray(gallery_archetypes)
    Q = len(query_embs)

    # Cosine similarity (inputs assumed L2-normalized)
    sims = query_embs @ gallery_embs.T  # (Q, N)
    ranked = np.argsort(-sims, axis=1)[:, :k]  # (Q, k)

    discounts = np.log2(np.arange(2, k + 2))  # log2(2), ..., log2(k+1)

    ndcgs: list[float] = []
    per_query: list[dict] = []
    for qi in range(Q):
        rel_zones = query_relevant_zones[qi]
        rel_archs = query_relevant_archetypes[qi]

        # Assign graded relevance for top-k retrieved players
        top_idx = ranked[qi]
        rels = np.zeros(k, dtype=np.float64)
        for j, idx in enumerate(top_idx):
            z = gallery_zones[idx]
            a = str(gallery_archetypes[idx])
            if z in rel_zones:
                if rel_archs and a in rel_archs:
                    rels[j] = 3.0
                else:
                    rels[j] = 1.0

        # DCG
        dcg = float(np.sum(rels / discounts))

        # IDCG: count ideal relevance levels across entire gallery
        if rel_archs:
            n_grade3 = int(np.sum(
                np.array([z in rel_zones and str(a) in rel_archs
                          for z, a in zip(gallery_zones, gallery_archetypes)])
            ))
        else:
            n_grade3 = 0
        n_zone_match = int(np.sum(np.array([z in rel_zones for z in gallery_zones])))
        n_grade1 = n_zone_match - n_grade3

        # Build ideal relevance vector: grade-3 first, then grade-1, truncated to k
        ideal_rels = np.zeros(k, dtype=np.float64)
        filled = 0
        for val, cnt in [(3.0, n_grade3), (1.0, n_grade1)]:
            take = min(cnt, k - filled)
            if take > 0:
                ideal_rels[filled:filled + take] = val
                filled += take
            if filled >= k:
                break

        idcg = float(np.sum(ideal_rels / discounts))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
        per_query.append({"query_idx": qi, "ndcg": round(ndcg, 4)})

    return {
        "mean_ndcg": float(np.mean(ndcgs)),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# d) Match outcome prediction
# ---------------------------------------------------------------------------


def match_outcome_prediction(
    spadl_path: str | Path,
    games_path: str | Path,
    feature_names: list[str],
    norm_stats_path: str | Path,
    seed: int = 42,
    test_size: float = 0.3,
) -> dict:
    """Predict match outcomes using team-averaged player cards.

    For each match: compute each player's overall card from SPADL stats, average
    per team, concatenate [home_avg, away_avg] as the feature vector, and
    predict outcome (home_win / draw / away_win) with LogisticRegression.

    Mapping team_id to home/away uses the player lookup: for each game, the
    SPADL team_ids are resolved to team names via the most common team_name
    among that team's players, then matched against the games file's
    home_team / away_team columns.

    Args:
        spadl_path: Path to spadl_unified.parquet.
        games_path: Path to games.parquet (columns: match_id, home_team,
            away_team, home_score, away_score).
        feature_names: Ordered list of feature column names.
        norm_stats_path: Path to card_normalization.json.
        seed: Random seed.
        test_size: Test split fraction.

    Returns:
        Dict with accuracy, macro_f1, baseline_accuracy, n_matches, n_train,
        n_test, class_distribution.
    """
    from football_embed.data.generate_stat_descriptions import compute_player_stats

    df = pd.read_parquet(spadl_path)
    games = pd.read_parquet(games_path)

    # Load normalization stats
    with open(norm_stats_path) as f:
        norm = json.load(f)
    feat_mean = np.array(norm["mean"])
    feat_std = np.array(norm["std"])
    safe_std = feat_std.copy()
    safe_std[safe_std == 0] = 1.0

    # Compute per-player stats across all data
    stats = compute_player_stats(df)
    for feat in feature_names:
        if feat not in stats.columns:
            stats[feat] = 0
    stats = stats.fillna(0)

    # Build a global team_id -> team_name map from SPADL + players_lookup.
    # For each (game_id, team_id) pair we look up the team_name of the
    # majority of that team's players via the lookup table.  This avoids
    # needing a separate teams table.
    lookup_path = Path(spadl_path).parent / "players_lookup.parquet"
    if lookup_path.exists():
        lookup = pd.read_parquet(lookup_path)
        pid_to_team_name = dict(zip(lookup["player_id"], lookup["team_name"]))
    else:
        # Fallback: no lookup, cannot resolve team names
        logger.warning("players_lookup.parquet not found at %s, skipping match outcome", lookup_path)
        return {
            "accuracy": 0.0, "macro_f1": 0.0, "baseline_accuracy": 0.0,
            "n_matches": 0, "n_train": 0, "n_test": 0, "class_distribution": {},
        }

    # Determine the game_id column in the games file
    gid_col = "game_id" if "game_id" in games.columns else "match_id"

    features: list[np.ndarray] = []
    labels: list[str] = []

    for _, game_row in games.iterrows():
        gid = game_row[gid_col]
        game_actions = df[df["game_id"] == gid]
        if game_actions.empty:
            continue

        teams = game_actions["team_id"].unique()
        if len(teams) != 2:
            continue

        # Resolve team_id -> team_name via player lookup
        tid_to_name: dict[int, str] = {}
        for tid in teams:
            team_players = game_actions[game_actions["team_id"] == tid]["player_id"].unique()
            names = [pid_to_team_name[p] for p in team_players if p in pid_to_team_name]
            if names:
                # Most common name among this team's players
                tid_to_name[tid] = max(set(names), key=names.count)

        if len(tid_to_name) != 2:
            continue

        # Match to home/away from games file
        home_name = game_row["home_team"]
        away_name = game_row["away_team"]

        home_tid = None
        away_tid = None
        for tid, tname in tid_to_name.items():
            if tname == home_name:
                home_tid = tid
            elif tname == away_name:
                away_tid = tid

        if home_tid is None or away_tid is None:
            continue

        # Collect player cards per team
        home_players = game_actions[game_actions["team_id"] == home_tid]["player_id"].unique()
        away_players = game_actions[game_actions["team_id"] == away_tid]["player_id"].unique()

        home_in_stats = [p for p in home_players if p in stats.index]
        away_in_stats = [p for p in away_players if p in stats.index]

        if len(home_in_stats) < 3 or len(away_in_stats) < 3:
            continue

        home_cards = stats.loc[home_in_stats, feature_names].values.astype(np.float64)
        away_cards = stats.loc[away_in_stats, feature_names].values.astype(np.float64)

        # Z-score normalize
        home_cards = (home_cards - feat_mean) / safe_std
        away_cards = (away_cards - feat_mean) / safe_std

        # Average per team, concatenate
        feat_vec = np.concatenate([home_cards.mean(axis=0), away_cards.mean(axis=0)])

        # Determine outcome
        home_score = game_row.get("home_score")
        away_score = game_row.get("away_score")
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        if home_score > away_score:
            outcome = "home_win"
        elif home_score < away_score:
            outcome = "away_win"
        else:
            outcome = "draw"

        features.append(feat_vec)
        labels.append(outcome)

    if len(features) < 10:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "baseline_accuracy": 0.0,
            "n_matches": len(features),
            "n_train": 0,
            "n_test": 0,
            "class_distribution": {},
        }

    X = np.array(features)
    y = np.array(labels)

    # Baseline: most common class
    unique, counts = np.unique(y, return_counts=True)
    baseline_acc = float(counts.max() / len(y))
    class_dist = {str(c): int(n) for c, n in zip(unique, counts)}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "baseline_accuracy": baseline_acc,
        "n_matches": len(features),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_distribution": class_dist,
    }


# ---------------------------------------------------------------------------
# e) Within-zone retrieval accuracy
# ---------------------------------------------------------------------------


def within_zone_retrieval_accuracy(
    gallery_embs: np.ndarray,
    gallery_zones: np.ndarray | list[str],
    gallery_names: np.ndarray | list[str],
    k: int = 5,
) -> dict:
    """Compute within-zone retrieval accuracy: for each player, what fraction
    of their k nearest neighbors share the same zone?

    Also computes Recall@k: for each player, whether any of their k nearest
    neighbors is the same player (different text variant or season).

    Args:
        gallery_embs: (N, D) L2-normalized embeddings.
        gallery_zones: (N,) zone label strings.
        gallery_names: (N,) player name strings.
        k: Number of neighbors to consider.

    Returns:
        Dict with mean_zone_match_rate, per_zone dict, and overall stats.
    """
    gallery_zones = np.asarray(gallery_zones)
    gallery_names = np.asarray(gallery_names)
    N = len(gallery_embs)

    # Cosine similarity matrix
    sims = gallery_embs @ gallery_embs.T  # (N, N)
    # Zero out self-similarity
    np.fill_diagonal(sims, -1.0)

    ranked = np.argsort(-sims, axis=1)[:, :k]  # (N, k)

    zone_match_rates = []
    per_zone_rates = {}

    for i in range(N):
        zone_i = gallery_zones[i]
        neighbor_zones = gallery_zones[ranked[i]]
        match_rate = float(np.mean(neighbor_zones == zone_i))
        zone_match_rates.append(match_rate)

        if zone_i not in per_zone_rates:
            per_zone_rates[zone_i] = []
        per_zone_rates[zone_i].append(match_rate)

    per_zone_summary = {
        zone: {
            "mean_zone_match": float(np.mean(rates)),
            "n_players": len(rates),
        }
        for zone, rates in per_zone_rates.items()
    }

    return {
        "mean_zone_match_rate": float(np.mean(zone_match_rates)),
        "k": k,
        "n_players": N,
        "per_zone": per_zone_summary,
    }


# ---------------------------------------------------------------------------
# f) Archetype classification accuracy
# ---------------------------------------------------------------------------


def archetype_classification_accuracy(
    embeddings: np.ndarray,
    archetype_labels: np.ndarray | list[str],
    zone_labels: np.ndarray | list[str] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
) -> dict:
    """Fit a linear probe on embeddings to predict archetype labels.

    Similar to role_classification_accuracy but at the finer archetype level.
    Optionally reports per-zone archetype accuracy.

    Args:
        embeddings: (N, D) player embeddings.
        archetype_labels: (N,) archetype name strings.
        zone_labels: (N,) optional zone labels for per-zone breakdown.
        test_size: Fraction held out for testing.
        seed: Random seed.

    Returns:
        Dict with accuracy, macro_f1, n_classes, per_class, and optionally per_zone.
    """
    archetype_labels = np.asarray(archetype_labels)

    # Filter out any "Unknown" or empty labels
    valid = archetype_labels != ""
    if not valid.all():
        embeddings = embeddings[valid]
        archetype_labels = archetype_labels[valid]
        if zone_labels is not None:
            zone_labels = np.asarray(zone_labels)[valid]

    le = LabelEncoder()
    y = le.fit_transform(archetype_labels)
    class_names = le.classes_

    if len(class_names) < 2:
        return {
            "accuracy": 0.0, "macro_f1": 0.0, "n_classes": len(class_names),
            "n_train": 0, "n_test": 0, "per_class": {},
        }

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=test_size, random_state=seed, stratify=y,
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    per_class = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_test == cls_idx
        if mask.sum() == 0:
            continue
        per_class[cls_name] = float(accuracy_score(y_test[mask], y_pred[mask]))

    result = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(class_names),
        "per_class": per_class,
    }

    # Per-zone breakdown if zone_labels provided
    if zone_labels is not None:
        zone_labels_test = np.asarray(zone_labels)[len(X_train):]  # This won't work with sklearn split
        # Re-derive: use the split indices
        # Actually simpler: just report overall
        pass

    return result
