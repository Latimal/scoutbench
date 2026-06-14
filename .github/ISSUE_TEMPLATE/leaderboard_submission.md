---
name: Leaderboard submission
about: Submit a representation to the ScoutBench Task B leaderboard
title: "[Submission] <method name>"
labels: leaderboard
---

<!-- See release/SCHEMA.md for the submission format and release/LEADERBOARD.md for the rules. -->

**Method name:**
**One-line description:**
**Input provenance** (which data, which events, what training):
**Dimensionality D:**
**Gallery coverage** (`matched N/…` from the evaluator):

### Results (from `scoutbench-eval`)
- `all_candidates.map` (primary):
- `same_position.map` (stratified):
- random floor (report it):

### Significance (required for a "beats raw" claim)
- block bootstrap + cluster-t result vs `raw_card`:
- minimum detectable effect (`scoutbench_power.py`):

### Honesty checklist
- [ ] This method was **never** trained or tuned on the released ScoutBench labels.
- [ ] I report the random floor alongside my number.
- [ ] I used the block bootstrap over the 22 position components (not a per-query bootstrap).
