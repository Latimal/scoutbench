# ScoutBench Task B -- leaderboard & submission format

ScoutBench is an **external referee** for football player-similarity representations.
Its purpose is to expose label leakage (P1): methods graded on their own clusters
score nDCG ~0.99, but on this transfer-anchored ground truth **no learned
representation we have tested beats a raw-stat kNN on the primary all-candidates
metric** -- a null this benchmark is *powered* to overturn. Submit a representation
and find out where it lands.

## How to submit

1. Produce a submission file per [`SCHEMA.md`](SCHEMA.md) (`player_id` + embedding dims).
2. Score it (runs entirely from the CC0 release -- no StatsBomb data needed):
   ```bash
   scoutbench-eval --embeddings <your_file>.parquet --out metrics.json
   ```
3. Report the resulting `metrics.json` with:
   - **method name** and a one-line description of the representation,
   - **provenance** of the input features (which data, which events, what training),
   - **dimensionality** `D`,
   - whether the method ever saw any ScoutBench label (it must not -- that is the leak),
   - pool coverage (`matched N/1363` printed by the evaluator).

## Primary metric

`all_candidates.map` (mean average precision ranking each of the 1363 query players
against all other players in the CC0 pool). Also report the stratified
`same_position.map` (within-sub-position, the hard test -- note: near-floor and
underpowered, so significance claims should be confined to all-candidates).

## Public leaderboard (CC0 pool, 1363 players)

Sorted by ALL-MAP (the primary metric). Higher is better. `raw_card` is the baseline
to beat; the finding is that nothing learned does.

| method | SP-MAP | ALL-MAP | notes |
|---|---|---|---|
| **raw_card cosine** | **0.1041** | **0.0553** | **baseline (raw-stat kNN)** |
| pca(card) | 0.1040 | 0.0553 | linear, raw |
| Player-Vectors (Decroos&Davis NMF) | 0.0984 | 0.0539 | published learned, ties raw |
| v11 (contrastive embed) | 0.0930 | 0.0492 | learned |
| TF-IDF text | 0.0952 | 0.0459 | text baseline |
| NMF (over card features) | 0.0958 | 0.0436 | learned |
| card+VAEP | 0.0983 | 0.0268 | raw+value |
| fbref percentile | 0.1014 | 0.0222 | raw stat percentiles |
| random | 0.0883 | 0.0113 | floor |

### Significance (block bootstrap over 22 components)

| comparison | scope | diff | p_block | p_cluster-t |
|---|---|---|---|---|
| raw_card vs random | all | +0.0440 | 0.0002 | 0.0000 |
| raw_card vs random | sp | +0.0157 | 0.0002 | 0.0277 |
| raw_card vs v11 | all | +0.0061 | 0.0002 | 0.2436 |
| raw_card vs v11 | sp | +0.0110 | 0.0002 | 0.0877 |
| raw_card vs Player-Vectors | all | +0.0014 | 0.4610 | 0.1768 |
| raw_card vs Player-Vectors | sp | +0.0056 | 0.0690 | 0.0432 |

Raw beats random on both metrics under both tests -- the benchmark carries real signal.
No learned representation beats raw on ALL-MAP (the primary, well-powered metric).

### Relationship to the paper's numbers

The paper reports results on the full 5668-player StatsBomb-derived gallery (a
research-only, non-redistributable configuration). The public leaderboard above uses
the **1363-player CC0 pool** (the query players in `scoutbench_taskb_players.csv`) so
that anyone can score a submission from the released files alone with zero
non-redistributable data. The finding -- no learned representation beats a raw-stat
kNN on all-candidates -- **replicates** on the CC0 pool. Absolute numbers are higher
in the smaller pool (MAP ~0.05 vs ~0.02) because the candidate set is more homogeneous.
To reproduce the paper's full-gallery configuration, rebuild the gallery per `DATA.md`.

## Label validity

Absolute numbers are low by design: realized replacements are a *weak* similarity
label (validity check: mean similarity-percentile 0.549, CI [0.521, 0.585], excludes
the 0.50 null). The contribution is the **ordering and the external protocol**, not
deployable accuracy. A submission that meaningfully beats `raw_card` on
`all_candidates.map` under the block bootstrap would be the first learned
representation to do so.

## Honesty rules

- A method must **never** have been trained or tuned on the released labels.
- Report the random floor alongside your number.
- For a significance claim, use the block bootstrap + cluster-t in
  `football_embed/evaluation/scoutbench_blockboot.py` (resamples the 22 components,
  not the 1363 dependent queries -- per-query bootstrap inflates significance ~10x).
- Report the test's **minimum detectable effect** (`football_embed/evaluation/scoutbench_power.py`).
  A null is only informative if the benchmark could have detected a real advantage of
  the relevant size: on `all_candidates.map` (primary) the cluster-t MDE is ~0.007 MAP
  (below the method spread -- the null is well-powered); on same-position it is ~0.017
  MAP (above the spread -- near-floor, so draw no within-role conclusions).
