# Changelog

All notable changes to ScoutBench are documented here (format: [Keep a Changelog](https://keepachangelog.com)).
The benchmark **labels are versioned** so leaderboard results stay comparable across time.

## [0.1.0] — 2026-06

### Added
- **ScoutBench Task B** — an external, transfer-anchored player-similarity benchmark.
  Frozen label release **v1**: 11,736 directed silver-positive pairs over 1,363 query
  players, derived solely from CC0 transfermarkt fields (`release/`, CC0 1.0).
- `scoutbench` evaluator (`scoutbench-eval`) for scoring third-party embedding submissions.
- Reproduction pipeline (`reproduce.sh`) and the statistics harness: block bootstrap over
  position components, cluster-t, Holm correction, TOST equivalence, and a minimum-detectable-effect
  (sensitivity) analysis.
- **360 cross-match identity protocol (P2)** plus the static-heatmap-family generalization check.
- Accompanying paper, *When the Benchmark Grades Itself* (`paper/`).
