<!-- For leaderboard submissions, fill the honesty checklist below. For code, describe the change and add a test where it makes sense. -->

## What this changes

## If this is a leaderboard submission
- [ ] The method was **never** trained or tuned on the released ScoutBench labels.
- [ ] I report `all_candidates.map`, the random floor, and a significance result (block bootstrap + cluster-t vs `raw_card`).
- [ ] I report the test's minimum detectable effect (`scoutbench_power.py`).

## If this is code
- [ ] `ruff check scoutbench tests` passes.
- [ ] `pytest tests` passes.
