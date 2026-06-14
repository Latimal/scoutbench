# DATA.md -- ScoutBench provenance, licensing, and how to rebuild every input

This documents where every input to the ScoutBench / 360 reproduction comes from,
what may be redistributed, and how a third party obtains or rebuilds each file.
The results scripts (`reproduce.sh`) consume these inputs; they
do **not** rebuild them.

## TL;DR licensing

| source | license | commercial use | in this repo? |
|---|---|---|---|
| **StatsBomb Open Data** | Research/educational, **non-commercial**, attribution | **NO** -- research only | derived features only (gitignored) |
| **Wyscout public dataset** | **CC BY 4.0** | yes, with attribution | derived features only (gitignored) |
| **transfermarkt-datasets (Kaggle)** | **CC0** (public domain) | yes | labels redistributable -> `release/` |
| Wikipedia/Wikidata bios | CC-BY-SA (attribution + share-alike) | with conditions | not used by ScoutBench Task B |

**Bottom line:** the ScoutBench player *gallery* (88-dim cards, the `repr_*`
features, the 360 freeze-frames) is **StatsBomb-Open-derived and therefore
research/non-commercial** -- it CANNOT be sold and is gitignored. The Task B
**labels** (replacement pairs, sub-positions, player ids/names) come solely from
**CC0 transfermarkt**, so they ARE redistributable and are exported to `release/`
(`scoutbench/export_release.py`). The StatsBomb Open Data licence (research /
non-commercial, attribution) is published at https://github.com/statsbomb/open-data.

## Inputs the result scripts read

All paths are relative to the repo root.

### Redistributable (CC0-derived) -- shipped in `release/`

#### `data/processed/benchmark/replacement_pairs.parquet`
- **What:** silver similarity-positive pairs. Columns: `player_x_tmid`,
  `player_y_tmid`, `sub_position`. 5,868 pairs. Both players are transfermarkt ids.
- **Provenance:** mined from **CC0 transfermarkt-datasets** (`transfers.csv` +
  `players.csv`): a club loses player X at sub-position P and signs player Y at the
  same sub-position P in the same/next transfer window -> (X, Y) is a like-for-like
  replacement positive. Restricted to pairs where BOTH players also exist in the
  gallery (so every pair is scorable).
- **Redistributable:** YES (CC0). Exported (deduped, directed, gallery-keyed) to
  `release/scoutbench_taskb_pairs.csv` + `.parquet`.
- **Builder status:** there is **no dedicated builder script committed** for the raw
  mining step; the file was constructed by joining `data/raw/transfermarkt/transfers.csv`
  and `players.csv` (the Kaggle CC0 dump) on the same-window same-sub-position rule,
  then filtering to gallery players. To rebuild: download the Kaggle CC0 dataset
  (below), reproduce the join (lost-player at P -> signing at P within the window),
  and intersect with the gallery names via `transfermarkt_join.parquet`.

#### `data/processed/benchmark/transfermarkt_join.parquet`
- **What:** maps gallery `player_name` <-> transfermarkt id + external attributes.
  Columns: `player_name`, `tm_player_id`, `tm_name`, `method`, `tm_sub_position`,
  `tm_date_of_birth`, `tm_market_value_in_eur`, `tm_foot`, `tm_country_of_citizenship`.
  3,464 rows.
- **Provenance:** fuzzy/exact name match between gallery players and **CC0
  transfermarkt** `players.csv` (`method` records exact vs fuzzy match). Attributes
  copied from transfermarkt CC0 fields.
- **Redistributable:** the transfermarkt fields are CC0; the *mapping to gallery
  names* is derived but the names themselves are CC0 transfermarkt display names.
- **Builder status:** **no dedicated builder script committed.** Rebuild by
  name-matching the gallery (`player_name`) against transfermarkt `players.csv`
  (exact, then normalized/fuzzy on `name`), carrying `player_id`, `sub_position`,
  `date_of_birth`, `market_value_in_eur`, `foot`, `country_of_citizenship`.

### Research-only (StatsBomb/Wyscout-derived) -- gitignored, NOT redistributable

#### `data/processed/text/player_card_gallery.parquet`
- **What:** the player gallery. 5,668 players x (7 meta cols + 88 `card_*` features).
  Meta: `player_name`, `zone`, `team_name`, `position`, `league`, `archetype_id`,
  `archetype_name`. The `card_*` columns are the z-normalized, variance-pruned 88-dim
  SPADL stat card.
- **Provenance:** **StatsBomb Open Data + Wyscout public**, converted to SPADL with
  `socceraction`, aggregated to 114 features, variance-pruned to 88, z-/L2-normalized.
- **Builder (committed):** `football_embed/training/build_player_cards.py` (consumes
  the SPADL feature tables from `football_embed/data/generate_stat_descriptions.py`).
  Treated read-only here (see the Pipeline section of the README).
- **Redistributable:** NO (StatsBomb Open = non-commercial; the derived features
  inherit the restriction). Researchers rebuild from the open data themselves.

#### `data/processed/benchmark/repr_pca.parquet`, `repr_fbref.parquet`, `repr_text.parquet`, `repr_nmf.parquet`
- **What:** alternative gallery representations used as Task B baselines. Each is
  `player_name` + feature columns; the harness L2-normalizes them.
  - `repr_pca` -- PCA of the 88-dim card matrix (32-d). (linear, raw)
  - `repr_fbref` -- per-position percentile-ranked stat features ("fbref-style").
  - `repr_text` -- TF-IDF over the player text descriptions.
  - `repr_nmf` -- NMF over the **card features** (a learned baseline, distinct from
    the faithful Player-Vectors below).
- **Provenance:** all derived from the StatsBomb-derived gallery / text -> research-only.
- **Builder status:** **no dedicated builder scripts committed** for these four. They
  were produced ad hoc from the gallery (`sklearn.decomposition.PCA` /
  `sklearn.decomposition.NMF` over `card_*`; `TfidfVectorizer` over the text;
  per-position percentile ranking for fbref). To rebuild: load
  `player_card_gallery.parquet`, fit the stated transform on `card_*` (or the text),
  write `player_name` + components. (Honest gap: these are not one-command
  reproducible; the transforms are simple and stated, but the exact ad-hoc scripts
  were not retained.)

#### `data/processed/benchmark/repr_player_vectors.parquet`
- **What:** faithful **Player-Vectors** (Decroos & Davis, ECML-PKDD 2019) baseline.
  `player_name` + `pv0..pv17` (18-dim). The one external *published* learned baseline.
- **Provenance:** built from on-ball SPADL events (50x50 action-location heatmaps per
  type -> per-type NMF), NOT card features. StatsBomb-derived -> research-only.
- **Builder (committed):** `football_embed/data/player_vectors_nmf.py`
  (`--spadl data/processed/spadl_unified.parquet --lookup data/processed/players_lookup.parquet`).
  Rebuild: `.venv/bin/python3 -m football_embed.data.player_vectors_nmf`.

#### `data/processed/benchmark/repr_card_vaep.parquet`
- **What:** 88 card dims + 3 scaled per-game/per-100 VAEP dims (91-d). `player_name` + `f0..f90`.
- **Provenance:** card features + VAEP computed from SPADL -> research-only.
- **Builder (committed):** `build_repr_card_vaep.py` (repo root). Needs
  `data/processed/benchmark/player_vaep.parquet` (built by
  `football_embed/data/compute_vaep.py`) and `data/processed/players_lookup.parquet`.
  Rebuild: `.venv/bin/python3 build_repr_card_vaep.py`.

#### `checkpoints/text_branch/v11/best/`
- **What:** the v11 contrastive text-branch + card projector (the `v11` method, and
  Task A). Trained on StatsBomb/Wyscout-derived cards + synthetic text.
- **Provenance/redistributable:** weights are gitignored. The base text model
  (`nomic-ai/modernbert-embed-base`, Apache-2.0) must be in the HF cache; the scripts
  set `HF_HUB_OFFLINE=1`, so download it once with network before an offline run.
- **Note:** the core Task B finding does NOT require v11 -- run `pip install -e ".[benchmark]"`
  and skip the `v11`/Task A rows.

#### `data/processed/benchmark/sb360_sets_matchkeyed.npz`
- **What:** anonymized StatsBomb 360 freeze-frames for the identity kill test.
  Arrays: `X` (events x 23 player point-tokens x 5 feats), `pid`, `tourn`, `match`.
  528,059 events, WC2022 + Euro2024 + Euro2020, 166 matches.
- **Provenance:** **StatsBomb Open Data 360** freeze-frames -> research-only.
- **Builder (committed):** `football_embed/data/sb360_extract_matchkeyed.py` (re-keys
  the original `sb360_sets.npz` by match; the original is downloaded by
  `football_embed/data/download_statsbomb360.py`).
- **Redistributable:** NO (StatsBomb Open non-commercial).

## How to obtain the raw sources

- **transfermarkt-datasets (CC0):** Kaggle `davidcariboo/player-scores` (a.k.a.
  transfermarkt-datasets). Download `players.csv` + `transfers.csv` to
  `data/raw/transfermarkt/`. CC0 -- no restriction.
- **StatsBomb Open Data (non-commercial):** `github.com/statsbomb/open-data` or
  `statsbombpy`. Free for research with attribution; **not for commercial use.**
- **Wyscout public dataset (CC BY 4.0):** Pappalardo et al., Figshare
  ("A public data set of spatio-temporal match events in soccer competitions").
  Attribution required.

## Reproducibility honesty

- **Fully one-command reproducible from committed builders:** `repr_player_vectors`,
  `repr_card_vaep`, `sb360_sets_matchkeyed`, and the gallery (via
  `build_player_cards.py` + the StatsBomb/Wyscout open data).
- **No committed builder (constructed ad hoc, transforms documented above):**
  `replacement_pairs`, `transfermarkt_join`, `repr_pca`, `repr_fbref`, `repr_text`,
  `repr_nmf`. These are simple, stated transforms over CC0 transfermarkt (labels) or
  the gallery (representations); the exact construction scripts were not retained.
- **Non-public by license:** the raw StatsBomb Open Data is freely downloadable but
  **non-commercial**; the derived gallery/360 features inherit that and are therefore
  gitignored and not shipped. Only the **CC0-derived labels** (`release/`) are
  redistributable. This is the load-bearing constraint on the benchmark's
  distribution.
