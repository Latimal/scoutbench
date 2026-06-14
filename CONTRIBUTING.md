# Contributing to ScoutBench

The most valuable contribution is a **leaderboard submission** — a player
representation scored honestly against the external transfer-anchored labels.

## Submit to the leaderboard

1. Produce a submission file per [`release/SCHEMA.md`](release/SCHEMA.md)
   (`player_id` or `player_name` + your embedding dimensions).
2. Score it (you need the rebuilt gallery — see [`DATA.md`](DATA.md)):
   ```bash
   scoutbench-eval --embeddings my_embeddings.parquet --out metrics.json
   ```
3. Open a pull request adding a row to [`release/LEADERBOARD.md`](release/LEADERBOARD.md)
   with your `all_candidates.map`, a one-line method description, the input
   provenance, dimensionality `D`, gallery coverage, and a significance result
   (block bootstrap + cluster-t; see the honesty rules in `LEADERBOARD.md`).

### Honesty rules (non-negotiable)

- A method must **never** have been trained or tuned on the released labels.
- Report the random floor alongside your number.
- Report the test's minimum detectable effect (`scoutbench_power.py`) for any
  significance claim — a null is only informative if the benchmark could have
  detected a real advantage.

## Issues & code

Bug reports, reproduction problems, and protocol questions are welcome via
GitHub issues. For code changes, keep the style consistent (`ruff` config is in
`pyproject.toml`) and add a test under `tests/` where it makes sense.
