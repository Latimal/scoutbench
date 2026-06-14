# ScoutBench

**An external, honest benchmark for football (soccer) player-similarity representations.**

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data: CC0](https://img.shields.io/badge/Labels-CC0%201.0-lightgrey.svg)](release/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/paper.pdf)

> **No learned representation we have tested beats a raw-statistics nearest-neighbour
> on this benchmark's primary all-candidates metric** — not a purpose-trained metric
> learner, not genuinely richer inputs, and not a faithful reproduction of the published
> *Player Vectors* method — a null the benchmark is *powered* to overturn. The same models
> score nDCG@10 ≈ 0.99 when graded on their own clusters. **Can yours beat raw?**

ScoutBench grades player-similarity representations against a signal the model never
sees: **realized like-for-like replacement transfers**. When a club loses player *X*
at sub-position *P* and signs *Y* at *P*, the pair *(X, Y)* is a silver
similarity-positive. It exists because the field usually validates similarity against
a model's *own* clusters/archetypes — which lets a model look near-perfect while
learning nothing externally valid.

This repo accompanies the paper **“When the Benchmark Grades Itself: Two Information
Leaks in Football Player-Representation Evaluation”** ([`paper/paper.pdf`](paper/paper.pdf)).

---

## The two findings

- **Label leakage (P1).** On ScoutBench's external transfer-anchored labels, every
  method beats random, but **no learned representation beats a raw-stat kNN on
  all-candidates retrieval (the primary metric)** — a null the benchmark is *powered*
  to overturn (minimum detectable effect ≈ 0.007 MAP, below the method spread). Holds
  for an off-objective contrastive embedding, a faithful Player Vectors reproduction,
  a purpose-trained metric learner, and orthogonal richer inputs (event-transformer,
  VAEP, 360 spatial). Within-role (same-position) retrieval is near-floor and
  underpowered, so we draw conclusions only on all-candidates.
- **Feature leakage (P2).** A set-transformer identifies players from anonymized 360
  freeze-frames at 0.878 Top-1 under a random within-tournament split, but **collapses
  to 0.188 under a match-disjoint split** (disjoint CIs); a same-match control scores
  0.889. The collapse reproduces for static heatmap representations, so it is a
  property of the random-split *protocol*, not one architecture. Fix: split by match
  (or season) and report a same-match positive control.

## Leaderboard & submission

`raw_card` (a raw-statistics kNN) is the baseline to beat. The reference results and
submission rules are in [`release/LEADERBOARD.md`](release/LEADERBOARD.md); the
embedding submission format is in [`release/SCHEMA.md`](release/SCHEMA.md). The headline
metric is `all_candidates.map` over the 1363-player CC0 pool.

**Scoring runs entirely from the CC0 release files — no StatsBomb data needed:**

```python
from scoutbench import evaluate
metrics = evaluate("my_embeddings.parquet", out="metrics.json")
print(metrics["all_candidates"]["map"])
```

or from the command line:

```bash
scoutbench-eval --embeddings my_embeddings.parquet --out metrics.json
```

A submission is one row per player — `player_id` (transfermarkt id) **or**
`player_name`, plus your embedding dimensions (any column names work; `e0..e{D-1}` is
just the convention). Vectors are L2-normalized; cosine is the similarity. The player
ids to cover are the `is_query=True` rows of
[`release/scoutbench_taskb_players.csv`](release/scoutbench_taskb_players.csv) (n=1363).
See [`release/SCHEMA.md`](release/SCHEMA.md) for details.

## Install

```bash
pip install -e ".[benchmark]"   # light, torch-free: scores submissions + runs all statistics
pip install -e ".[v11]"         # adds torch + sentence-transformers, only to reproduce the v11 learned baseline
```

## Data & license

- **Labels** ([`release/`](release/): `scoutbench_taskb_pairs.*`, `scoutbench_taskb_players.csv`):
  **CC0 1.0** — derived solely from CC0 transfermarkt fields (id, name, sub-position).
  Free to redistribute; cite via the paper / Zenodo DOI.
- **Scoring** runs from the release alone (1363-player CC0 pool). To reproduce the
  paper's full 5668-player research configuration, rebuild the StatsBomb-derived gallery
  with [`DATA.md`](DATA.md) (research / non-commercial, not shipped here).
- **Code**: MIT ([`LICENSE`](LICENSE)).

## Reproduce the paper

```bash
./reproduce.sh        # 8 ordered steps; regenerates every number in the paper
```
Requires the rebuilt gallery/labels per [`DATA.md`](DATA.md). Pinned dependencies are in
[`requirements.lock`](requirements.lock).

## Honesty rules (leaderboard)

- A method must **never** have been trained or tuned on the released labels.
- Report the random floor alongside your number.
- For significance, use the block bootstrap + cluster-t in
  `football_embed/evaluation/scoutbench_blockboot.py` (resamples the 22 position
  components, not the 1363 dependent queries — a per-query bootstrap inflates
  significance ~10×), and report the test's minimum detectable effect
  (`scoutbench_power.py`).

## Repository layout

```
scoutbench/                  # the submission-facing evaluator (pip: scoutbench-eval)
football_embed/evaluation/   # the research harness: ScoutBench + 360 protocols + statistics
football_embed/{data,model,training}/   # the card/feature builders that produce the baselines
release/                     # CC0 Task B labels + SCHEMA + LEADERBOARD + manifest
paper/                       # the paper (source + PDF)
tests/                       # evaluator + metric tests
reproduce.sh                 # end-to-end reproduction
DATA.md                      # how to rebuild the (non-redistributable) gallery
```

`football_embed/` is the original multimodal-embedding research project ScoutBench grew
out of; ScoutBench uses its `evaluation/` harness (scoring + statistics) and the
card/feature builders that generate the baselines — it is not a general-purpose dependency.

## Citation

```bibtex
@misc{patni2026scoutbench,
  title  = {When the Benchmark Grades Itself: Two Information Leaks in Football Player-Representation Evaluation},
  author = {Patni, Aditya},
  year   = {2026},
  note   = {ScoutBench: an external transfer-anchored player-similarity benchmark}
}
```
