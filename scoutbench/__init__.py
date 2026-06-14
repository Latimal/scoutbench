"""ScoutBench -- externally-validated football player-retrieval benchmark.

This package is the *redistributable* front door to ScoutBench Task B. It ships:
  - a frozen ground-truth release (derived from transfermarkt CC0 -> redistributable),
  - a self-contained evaluator that scores submissions against the CC0 labels using
    only the released files (no StatsBomb gallery, no non-redistributable data),
  - input-schema and leaderboard-format docs.

Third parties submit their OWN embeddings (any provenance) keyed by the released
player ids and are scored against the released labels on the 1363-player CC0 pool.
"""

from .evaluate import evaluate  # noqa: F401

__all__ = ["evaluate"]
