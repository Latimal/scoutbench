#!/usr/bin/env bash
# =============================================================================
# reproduce.sh -- regenerate every number in paper/PAPER.md
#                 from the pre-built inputs.
#
#   "When the Benchmark Grades Itself: Two Information Leaks in Football
#    Player-Representation Evaluation"
#
# Run from the repo root:   bash reproduce.sh
# Env:                      .venv/bin/python3 (CPython 3.13.x; see requirements.lock)
# Install:                  pip install -e ".[benchmark]"      # core Task B (light, torch-free)
#                           pip install -e ".[benchmark,v11]"  # + torch for the v11/360 runs
#
# Inputs are the pre-built, gitignored artifacts under data/ (rebuild instructions
# in DATA.md). This script does NOT rebuild inputs; it regenerates the RESULTS.
#
# Outputs land in data/processed/benchmark/*.json. Per-seed runs write distinct
# paths (no overwrite). Each stanza below maps:  PAPER number  ->  script  ->  JSON.
# =============================================================================
set -euo pipefail

PY="${PY:-.venv/bin/python3}"
BENCH="data/processed/benchmark"
mkdir -p "$BENCH"

echo "== ScoutBench / 360 reproduction =="
echo "interpreter: $($PY --version)"
echo

# -----------------------------------------------------------------------------
# [1] Per-query bootstrap (the WRONG test the paper critiques) + the CORRECT
#     block-bootstrap with cluster-t + Holm.
#
# PAPER -> Sec. 3 "Statistics done right"; Abstract "naive p ~10x too small".
# PAPER -> Sec. 3 Results table (SP-MAP / ALL-MAP per method):
#            raw_card 0.0498/0.0194, fbref 0.0514/0.0082, pca 0.0501/0.0195,
#            text_tfidf 0.0469/0.0165, player_vectors 0.0464/0.0172,
#            nmf 0.0450/0.0135, v11 0.0447/0.0162, card_vaep 0.0440/0.0088,
#            random 0.0369/0.0030.
# PAPER -> Sec. 3 "raw_card vs random ... block p=0.0002, cluster-t p=0.0001";
#            "raw_card vs v11 ... p~0.02 ... cluster-t p=0.36 (a tie)";
#            "raw - Player-Vectors: SP +0.0034 block p=0.15, component-t p=0.90".
# -----------------------------------------------------------------------------
echo "[1/8] significance (per-query bootstrap -- the critiqued test)"
$PY -m football_embed.evaluation.scoutbench_significance \
    --out "$BENCH/scoutbench_significance.json"
#   -> data/processed/benchmark/scoutbench_significance.json

echo "[2/8] blockboot (CORRECT: component block-bootstrap + cluster-t + Holm)"
$PY -m football_embed.evaluation.scoutbench_blockboot \
    --bootstrap 10000 --seed 0 \
    --out "$BENCH/scoutbench_blockboot.json"
#   -> data/processed/benchmark/scoutbench_blockboot.json
#      (n_clusters=22, the resampling unit; method_means feed the Sec. 3 table)
#   PAPER 3 (sensitivity): minimum detectable effect of the cluster-t -- the null is
#   well-powered on all-candidates (MDE~0.007 < spread) and near-floor within-role.
$PY -m football_embed.evaluation.scoutbench_power \
    --seed 0 \
    --out "$BENCH/scoutbench_power.json"
#   -> data/processed/benchmark/scoutbench_power.json

# -----------------------------------------------------------------------------
# [3] Extended rigor: per-method CI ranking, every-method-vs-v11 / vs-random,
#     per-sub-position and market-value-tier heterogeneity.
#
# PAPER -> Sec. 3 "no learned representation beats raw" robustness; value-tier
#          check rules out a popularity artifact.
# -----------------------------------------------------------------------------
echo "[3/8] extended (ranking CIs, pairwise vs v11/random, per-position, value tiers)"
$PY -m football_embed.evaluation.scoutbench_extended \
    --bootstrap 10000 --seed 0 \
    --out "$BENCH/scoutbench_extended.json"
#   -> data/processed/benchmark/scoutbench_extended.json

# -----------------------------------------------------------------------------
# [4] Purpose-trained, player-disjoint metric-learner (the decisive,
#     non-tautological test) -- multi-positive InfoNCE + TOST, across 3 seeds.
#     Each seed writes a DISTINCT path (no overwrite).
#
# PAPER -> Sec. 3 "Purpose-trained ... metric learning does not significantly
#          beat raw ... 3 seeds: same-pos dMAP {+0.0008, +0.0006, -0.0027}, all
#          p>0.37; all-cand {+0.0016, +0.0024, +0.0009} p 0.19-0.68; TOST(+-0.005)
#          equivalent in 1/3 seeds, inconclusive in 2/3."
#   seeds 7, 11, 23 -> ...metric_baseline_seed{7,11,23}.json
#   + combined      -> ...metric_baseline_allseeds.json (aggregate across seeds)
# (Needs torch; auto-uses MPS on Apple Silicon, else CPU. ~minutes/seed.)
# -----------------------------------------------------------------------------
echo "[4/8] metric-learner baseline x3 seeds (multi-positive InfoNCE + TOST)"
for SEED in 7 11 23; do
  echo "      seed=$SEED"
  $PY -m football_embed.evaluation.scoutbench_metric_baseline \
      --steps 3000 --seed "$SEED" \
      --out "$BENCH/scoutbench_metric_baseline.json"
done
#   -> ...metric_baseline_seed{7,11,23}.json + ...metric_baseline_allseeds.json
#      (the script auto-appends _seed{N} and upserts the combined all-seeds file)

# -----------------------------------------------------------------------------
# [5] Cross-pool generalization (does the verdict replicate on disjoint pools?).
#
# PAPER -> Sec. 3 "Generalization across player pools": Pool A top-5 men's
#          (raw 0.0639 vs v11 0.0637 ns, vs PV 0.0598 ns; raw-random +0.0182
#          block p<0.001, cluster-t p=0.003); Pool B tournaments/women's/other
#          (raw 0.0841 vs v11 0.0648, vs PV 0.0677; raw-random +0.0222 block p=0.007).
# -----------------------------------------------------------------------------
echo "[5/8] generalization (top-5 men's vs tournaments/women's/other pools)"
$PY -m football_embed.evaluation.scoutbench_generalization \
    --bootstrap 10000 \
    --out "$BENCH/scoutbench_generalization.json"
#   -> data/processed/benchmark/scoutbench_generalization.json

# -----------------------------------------------------------------------------
# [6] Silver-label validity (do realized replacements carry similarity signal?).
#
# PAPER -> Sec. 7 "realized replacements sit at mean similarity-percentile 0.549
#          (CI [0.521, 0.585], excludes 0.50); replacement cos 0.449 vs
#          random-same-position 0.404."
# -----------------------------------------------------------------------------
echo "[6/8] label validity (replacement-similarity percentile vs the 0.50 null)"
$PY -m football_embed.evaluation.scoutbench_label_validity \
    --bootstrap 10000 \
    --out "$BENCH/scoutbench_label_validity.json"
#   -> data/processed/benchmark/scoutbench_label_validity.json

# -----------------------------------------------------------------------------
# [7] 360 identity cross-match kill test (Pitfall 2).
#
# PAPER -> Sec. 4 table: split-half 0.878/0.924, same-match control 0.889/0.932,
#          cross-match 0.188/0.326, cross-match no-teammate 0.137/0.274;
#          "collapse split-half 0.878 -> cross-match 0.188 (delta 0.69)."
# (Needs torch + the offline-rebuilt sb360_sets_matchkeyed.npz; ~minutes.)
# -----------------------------------------------------------------------------
echo "[7/8] 360 cross-match kill test (split-half vs cross-match vs controls)"
$PY -m football_embed.evaluation.sb360_crossmatch_killtest \
    --d 96 --layers 2 --heads 4 --steps 2000 \
    --out "$BENCH/sb360_crossmatch_killtest.json"
#   -> data/processed/benchmark/sb360_crossmatch_killtest.json
#   PAPER 4 (static-heatmap generalization): the same collapse for STATIC heatmap reps
#   (param-free heatmap + Player-Vectors-style NMF), training-free. Bounded claim: does
#   NOT cover 6MapNet's trained triplet-CNN (untested; needs tracking data).
$PY -m football_embed.evaluation.sb360_6mapnet_crossmatch \
    --out "$BENCH/sb360_6mapnet_crossmatch.json"
#   -> data/processed/benchmark/sb360_6mapnet_crossmatch.json

# -----------------------------------------------------------------------------
# [8] Export the frozen, redistributable Task B label release (CC0-derived).
#     Not a paper number; ships the public benchmark artifact + leaderboard inputs.
# -----------------------------------------------------------------------------
echo "[8/8] export redistributable Task B release -> release/"
$PY -m scoutbench.export_release --out-dir release
#   -> release/scoutbench_taskb_pairs.csv|.parquet, scoutbench_taskb_players.csv, manifest.json

echo
echo "== done. paper numbers regenerated under $BENCH/, release under release/ =="
