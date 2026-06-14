#!/usr/bin/env python3
"""Hybrid reranker: combines dense retrieval scores with stat-based skill scores.

Pure inference module. No training required. Uses keyword extraction from NL queries
to identify skill intents, then scores raw card features against those intents.

Usage:
    from football_embed.evaluation.rerank import rerank_candidates
    new_indices = rerank_candidates(query_texts, top_k_indices, top_k_scores,
                                     raw_cards, feature_names, alpha=0.7)
"""

from __future__ import annotations

import re

import numpy as np

# Maps skill intent keywords to card feature names (indices into feature_names).
# Features should have POSITIVE correlation with the intent (higher = better).
INTENT_FEATURES: dict[str, list[str]] = {
    "aerial": [
        "clearances_per_match",
    ],
    "set_piece": [
        "crosses_per_match", "cross_accuracy", "corner_delivery_per_match",
        "freekick_delivery_per_match",
    ],
    "dribbler": [
        "take_ons_per_match", "dribble_success_rate",
        "carries_into_final_third_per_match", "carries_into_box_per_match",
    ],
    "box_to_box": [
        "actions_per_match", "defensive_actions_per_match",
        "forward_third_pct", "action_convex_hull_area",
    ],
    "long_range_passer": [
        "progressive_passes_per_match", "avg_pass_length",
        "final_third_passes_per_match",
    ],
    "penalty_box_scorer": [
        "goals_per_match", "shots_per_match", "shot_conversion_rate",
        "penalty_box_actions_pct",
    ],
    "presser": [
        "pressing_actions_per_match", "counterpressing_rate",
        "tackles_per_match", "interceptions_per_match",
    ],
    "creator": [
        "shots_created_per_match", "passes_into_box_per_match",
        "xt_gain_per_match", "final_third_passes_per_match",
    ],
    "deep_distributor": [
        "passes_per_match", "pass_completion", "avg_pass_length",
        "progressive_passes_per_match",
    ],
    "sweeper_keeper": [
        "keeper_actions_per_match", "keeper_claim_per_match",
        "goalkick_per_match",
    ],
}

# Keyword patterns that trigger each intent
INTENT_PATTERNS: dict[str, list[str]] = {
    "aerial": ["aerial", "header", "heads the ball", "dominant in the air", "wins aerial"],
    "set_piece": ["set.piece", "free.?kick", "corner", "dead.?ball", "delivery"],
    "dribbler": ["dribbl", "take.?on", "1.on.1", "one.on.one", "beat.?man", "carries"],
    "box_to_box": ["box.to.box", "b2b", "covers ground", "end.to.end", "both ends"],
    "long_range_passer": ["long.range pass", "long pass", "switching play", "diagonal"],
    "penalty_box_scorer": ["penalty.box", "goal.?scor", "poacher", "tap.?in", "clinical finish"],
    "presser": ["press", "counter.?press", "gegenpres", "high.?press", "intensity"],
    "creator": ["creat", "chance.?creat", "assist", "key pass", "through ball", "final ball"],
    "deep_distributor": ["distribut", "deep.?lying", "dictate", "metronome", "pass.?master"],
    "sweeper_keeper": ["sweeper.?keeper", "rush.?out", "command.?area"],
}


def parse_intents(query: str) -> list[str]:
    """Extract skill intents from a natural language query via regex matching."""
    q = query.lower()
    matched = []
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                matched.append(intent)
                break
    return matched


def _build_feature_index(feature_names: list[str]) -> dict[str, int]:
    """Map feature name to column index."""
    return {name: i for i, name in enumerate(feature_names)}


def score_by_intents(
    raw_cards: np.ndarray,
    feature_names: list[str],
    intents: list[str],
) -> np.ndarray:
    """Score each player by how well their raw stats match the given intents.

    For each intent, z-normalizes relevant features across the gallery, then
    averages the z-scores. Final score = mean across all matched intents.

    Args:
        raw_cards: (N, D) raw (un-normalized) card features for gallery.
        feature_names: list of D feature names matching raw_cards columns.
        intents: list of intent keys (e.g., ["aerial", "box_to_box"]).

    Returns:
        (N,) scores in [0, 1] range (min-max scaled).
    """
    if not intents:
        return np.zeros(raw_cards.shape[0])

    feat_idx = _build_feature_index(feature_names)
    intent_scores = []

    for intent in intents:
        feat_names = INTENT_FEATURES.get(intent, [])
        valid_cols = [feat_idx[f] for f in feat_names if f in feat_idx]
        if not valid_cols:
            continue
        # Z-normalize each feature, then average
        cols = raw_cards[:, valid_cols]
        mu = cols.mean(axis=0, keepdims=True)
        std = cols.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        z = (cols - mu) / std
        intent_scores.append(z.mean(axis=1))

    if not intent_scores:
        return np.zeros(raw_cards.shape[0])

    combined = np.mean(intent_scores, axis=0)
    # Min-max scale to [0, 1]
    lo, hi = combined.min(), combined.max()
    if hi - lo < 1e-8:
        return np.full(combined.shape, 0.5)
    return (combined - lo) / (hi - lo)


def rerank_candidates(
    dense_scores: np.ndarray,
    raw_cards: np.ndarray,
    feature_names: list[str],
    query_texts: list[str],
    alpha: float = 0.7,
    top_k: int = 50,
    rerank_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Rerank dense retrieval results using stat-based skill scoring.

    For each query:
    1. Take top_k candidates from dense retrieval
    2. Score them by intent-matched stat features
    3. Blend: final = alpha * dense_norm + (1-alpha) * skill_score
    4. Return top rerank_k

    Args:
        dense_scores: (Q, N) cosine similarities from dense retrieval.
        raw_cards: (N, D) raw card features for the full gallery.
        feature_names: list of D feature names.
        query_texts: list of Q query strings.
        alpha: blend weight for dense vs skill score.
        top_k: how many candidates to consider for reranking.
        rerank_k: how many to return.

    Returns:
        reranked_indices: (Q, rerank_k) gallery indices after reranking.
        reranked_scores: (Q, rerank_k) blended scores.
    """
    Q, N = dense_scores.shape
    all_indices = np.zeros((Q, rerank_k), dtype=np.int64)
    all_scores = np.zeros((Q, rerank_k), dtype=np.float64)

    for qi in range(Q):
        intents = parse_intents(query_texts[qi])

        # Get top_k from dense
        top_idx = np.argsort(dense_scores[qi])[::-1][:top_k]
        d_scores = dense_scores[qi, top_idx]

        if not intents:
            # No rerank needed, just return dense top-k
            final_idx = top_idx[:rerank_k]
            all_indices[qi] = final_idx
            all_scores[qi] = dense_scores[qi, final_idx]
            continue

        # Score the top_k candidates by intents
        skill_scores = score_by_intents(
            raw_cards[top_idx], feature_names, intents,
        )

        # Normalize dense scores to [0, 1] within this candidate set
        d_lo, d_hi = d_scores.min(), d_scores.max()
        if d_hi - d_lo < 1e-8:
            d_norm = np.full_like(d_scores, 0.5)
        else:
            d_norm = (d_scores - d_lo) / (d_hi - d_lo)

        blended = alpha * d_norm + (1 - alpha) * skill_scores
        rerank_order = np.argsort(blended)[::-1][:rerank_k]

        all_indices[qi] = top_idx[rerank_order]
        all_scores[qi] = blended[rerank_order]

    return all_indices, all_scores
