#!/usr/bin/env python3
"""Build card_vaep retrieval baseline: 88 stat-card dims + 3 scaled-VAEP dims (91-d)."""
import numpy as np
import pandas as pd

OUTPUT = "data/processed/benchmark/repr_card_vaep.parquet"

gal = pd.read_parquet("data/processed/text/player_card_gallery.parquet")
vaep = pd.read_parquet("data/processed/benchmark/player_vaep.parquet")
look = pd.read_parquet("data/processed/players_lookup.parquet")

card_cols = [f"card_{i}" for i in range(88)]
cards = gal[card_cols].to_numpy(dtype=np.float64)  # gallery order, 88-d

# Map vaep player_id -> player_name via lookup (task-specified path)
id2name = look.dropna(subset=["player_name"]).drop_duplicates("player_id").set_index("player_id")["player_name"]
vaep = vaep.copy()
vaep["mapped_name"] = vaep["player_id"].map(id2name)
vaep = vaep.dropna(subset=["mapped_name"])

# Aggregate per name from count-based totals (handles 3 name collisions across player_ids).
agg = vaep.groupby("mapped_name").agg(
    off_vaep=("off_vaep", "sum"),
    def_vaep=("def_vaep", "sum"),
    n_games=("n_games", "sum"),
    n_actions=("n_actions", "sum"),
)
agg["off_vaep_per_game"] = agg["off_vaep"] / agg["n_games"]
agg["def_vaep_per_game"] = agg["def_vaep"] / agg["n_games"]
agg["vaep_per100"] = (agg["off_vaep"] + agg["def_vaep"]) / agg["n_actions"] * 100.0

vaep_feat_cols = ["off_vaep_per_game", "def_vaep_per_game", "vaep_per100"]

# Align to gallery order; missing players -> NaN
v = agg.reindex(gal["player_name"].to_numpy())[vaep_feat_cols].to_numpy(dtype=np.float64)
matched_mask = ~np.isnan(v).any(axis=1)
coverage = int(matched_mask.sum())

# Z-score across players using only matched rows; fill missing with 0 after z-scoring.
mu = np.nanmean(v, axis=0)
sd = np.nanstd(v, axis=0)
sd = np.where(sd == 0, 1.0, sd)
vz = (v - mu) / sd
vz = np.where(np.isnan(vz), 0.0, vz)

# Scale VAEP block so its energy is comparable to the 88 card dims.
scale = np.sqrt(88.0 / 3.0)
vz_scaled = vz * scale

repr_mat = np.concatenate([cards, vz_scaled], axis=1).astype(np.float32)  # 91-d

out = pd.DataFrame({"player_name": gal["player_name"].to_numpy()})
for i in range(repr_mat.shape[1]):
    out[f"f{i}"] = repr_mat[:, i]
out.to_parquet(OUTPUT, index=False)

# Load back and verify
back = pd.read_parquet(OUTPUT)
n_feat = sum(c.startswith("f") for c in back.columns)
print(f"saved -> {OUTPUT}")
print(f"shape: {back.shape}  (rows={back.shape[0]}, dim={n_feat})")
print(f"5668 rows: {back.shape[0] == 5668}")
print(f"VAEP join coverage: {coverage}/5668 = {coverage/5668*100:.2f}%")
print(f"VAEP scale factor sqrt(88/3) = {scale:.4f}")
print(f"any NaN in output: {bool(back.isna().any().any())}")
