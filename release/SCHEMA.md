# ScoutBench Task B -- input & submission schema

ScoutBench Task B scores a **player embedding** on *external transfer-anchored*
similarity retrieval: for each query player, rank all others by cosine similarity
and measure whether the player's realized like-for-like replacement(s) rank highly.

## Released ground-truth files (this directory, CC0)

### `scoutbench_taskb_pairs.csv` / `.parquet`
One row per **directed** silver-positive edge. Both (x,y) and (y,x) are present.

| column | type | meaning |
|---|---|---|
| `player_x_id` | int | transfermarkt player id of the lost player (query) |
| `player_y_id` | int | transfermarkt player id of the realized replacement (relevant) |
| `sub_position` | str | transfermarkt sub-position shared by the pair (e.g. `Centre-Back`) |

A query player's relevant set = every `player_y_id` paired with it. The graph is
many-to-many (a club may replace one player with several / a player anchors several
pairs); it collapses into 22 connected components (~sub-positions) used by the
block-bootstrap statistics.

### `scoutbench_taskb_players.csv`
One row per player in the label graph.

| column | type | meaning |
|---|---|---|
| `player_id` | int | transfermarkt player id (join key) |
| `player_name` | str | transfermarkt display name (fallback join key) |
| `sub_position` | str | transfermarkt sub-position |
| `is_query` | bool | scored as a query by the harness (gallery member with >=1 relevant) |
| `in_gallery` | bool | present in the ScoutBench gallery |

## Submission schema (what YOU provide)

A parquet or CSV, one row per player, with a key column and embedding dimensions:

```
player_id, e0, e1, ..., e{D-1}        # preferred: transfermarkt id
# or
player_name, e0, e1, ..., e{D-1}      # fallback: transfermarkt display name
```

- Any column other than `player_id` / `player_name` is treated as an embedding dim.
- `D` is arbitrary (your embedding dimensionality).
- Vectors are **L2-normalized** by the evaluator; cosine is the similarity.
- Players you omit get a zero vector (they never retrieve / get retrieved).
- Cover all `is_query=True` players for a comparable score.

## Scoring

```bash
.venv/bin/python3 -m scoutbench.evaluate --embeddings my_embeddings.parquet --out metrics.json
# or
python -c "from scoutbench import evaluate; print(evaluate('my_embeddings.parquet'))"
```

Metrics are reported over **all candidates** (primary) and **same-sub-position**
candidates (stratified, the hard within-role test): `hit@{1,5,10}`, `recall@10`,
`mrr`, `map`. The headline number is `all_candidates.map`.

## Important: the gallery is research-only

Scoring requires the ScoutBench gallery (`player_card_gallery.parquet`) for player
ordering + names. Those features are **StatsBomb-Open-derived = research/non-commercial**
and are NOT in this release (see `../DATA.md`). This release ships the **labels**
(CC0). To score a submission you also need the gallery + transfermarkt join, which a
researcher rebuilds per `../DATA.md`. The labels here are the redistributable,
citable artifact; the gallery is the non-redistributable input.
