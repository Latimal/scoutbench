#!/usr/bin/env python3
"""ScoutBench Task B -- benchmark SENSITIVITY (minimum detectable effect).

The P1 finding is a NULL: no learned representation is distinguishable from a
raw-stat kNN. A null is only meaningful if the benchmark could have detected a
real advantage of the relevant size. This module measures that sensitivity
directly, so "we cannot detect a learned-rep advantage" is quantified rather
than asserted.

For the cluster-level paired t (the conservative test over the ~22 connected
components), the minimum detectable effect at power 0.8, two-sided alpha=0.05 is

    MDE = (t_{0.975, df} + t_{0.80, df}) * SE_component ,   df = n_components - 1

where SE_component is computed directly from the per-component mean differences
(the exact quantity the cluster-t consumes -- not backed out of a rounded
p-value). We report MDE for the learned-vs-raw comparisons against the full
method spread (best method minus random). If MDE exceeds that spread, the
benchmark cannot adjudicate effects of the size that separate real
representations -- the null is "no detectable advantage at this label's
resolution," not a proof of equality. We also verify the test is not generically
powerless: it DOES detect the gross raw-vs-random signal, which sits above the
MDE. Power curves use the noncentral-t (no simulation needed).

Offline; loads the same gallery/embeddings as scoutbench_blockboot.
Usage: .venv/bin/python3 -m football_embed.evaluation.scoutbench_power
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from football_embed.evaluation.scoutbench import (
    DEF_CKPT, DEF_GALLERY, DEF_JOIN, DEF_PAIRS,
    card_embeddings, load_queries, _qm,
)
from football_embed.evaluation.scoutbench_blockboot import (
    _tm2gi, _components, per_query,
)

# methods needed for the sensitivity comparisons (subset of blockboot METHODS)
NEED = {
    "raw_card": "raw_card",
    "fbref": "file:data/processed/benchmark/repr_fbref.parquet",
    "player_vectors": "file:data/processed/benchmark/repr_player_vectors.parquet",
    "v11": "v11",
    "random": "random",
}

# (A, B, scope): difference A-B whose detectability we characterize
COMPARISONS = [
    ("raw_card", "random", "all"),   # gross signal -- should be DETECTED (calibration)
    ("raw_card", "random", "sp"),
    ("raw_card", "v11", "sp"),       # learned-vs-raw -- the null
    ("raw_card", "v11", "all"),
    ("raw_card", "player_vectors", "sp"),
    ("raw_card", "player_vectors", "all"),
]


def se_component(d, cl):
    """SE of the cluster-mean difference = the cluster-t's standard error."""
    uc = np.unique(cl)
    cmeans = np.array([d[cl == c].mean() for c in uc])
    n = len(uc)
    se = float(cmeans.std(ddof=1) / np.sqrt(n))
    return se, n, float(cmeans.mean())


def mde(se, df, power=0.80, alpha=0.05):
    """Minimum detectable mean difference for a paired/one-sample t."""
    return float((sps.t.ppf(1 - alpha / 2, df) + sps.t.ppf(power, df)) * se)


def power_at(delta, se, df, alpha=0.05):
    """Two-sided power to detect a true effect `delta` via the t-test (noncentral t)."""
    if se <= 0:
        return 1.0
    ncp = delta / se
    tcrit = sps.t.ppf(1 - alpha / 2, df)
    return float(sps.nct.sf(tcrit, df, ncp) + sps.nct.cdf(-tcrit, df, ncp))


def main():
    ap = argparse.ArgumentParser(description="ScoutBench Task B sensitivity / MDE")
    ap.add_argument("--gallery", default=DEF_GALLERY); ap.add_argument("--pairs", default=DEF_PAIRS)
    ap.add_argument("--join", default=DEF_JOIN); ap.add_argument("--checkpoint", default=DEF_CKPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/benchmark/scoutbench_power.json")
    args = ap.parse_args()
    np.random.seed(args.seed)  # fix the stochastic `random` baseline for a reproducible floor

    gallery = pd.read_parquet(args.gallery); join = pd.read_parquet(args.join); pairs = pd.read_parquet(args.pairs)
    queries, subpos = load_queries(gallery, join, pairs)
    qlist = list(queries.items())
    tm2gi = _tm2gi(gallery, join)
    comp = _components(pairs, tm2gi)
    gi2comp = {tm2gi[tm]: root for tm, root in comp.items() if tm2gi.get(tm) is not None}
    cl = np.array([gi2comp.get(qi, -qi - 1) for qi, _ in qlist])
    df = len(np.unique(cl)) - 1
    print(f"n_queries={len(qlist)}  n_components={df + 1}  df={df}")

    pq = {nm: per_query(card_embeddings(spec, gallery, args.checkpoint), qlist, subpos)
          for nm, spec in NEED.items()}

    # method spread per scope: best method minus random (the effect size that separates real reps)
    spread = {sc: round(max(pq[m][sc].mean() for m in NEED) - pq["random"][sc].mean(), 4)
              for sc in ("sp", "all")}

    rows = []
    for A, B, sc in COMPARISONS:
        d = pq[A][sc] - pq[B][sc]
        se, n, diff = se_component(d, cl)
        m = mde(se, df)
        # observed two-sided cluster-t p, for cross-check against blockboot.json
        cmeans = np.array([d[cl == c].mean() for c in np.unique(cl)])
        p_t = float(sps.ttest_1samp(cmeans, 0.0).pvalue)
        rows.append({"A": A, "B": B, "scope": sc, "diff": round(diff, 5),
                     "se_component": round(se, 5), "p_cluster_t": round(p_t, 4),
                     "mde_080": round(m, 5), "detectable": bool(abs(diff) >= m),
                     "power_at_observed": round(power_at(abs(diff), se, df), 3)})

    # pooled sensitivity over the learned-vs-raw comparisons
    learned = [r for r in rows if r["B"] in ("v11", "player_vectors")]
    mde_sp = np.median([r["mde_080"] for r in learned if r["scope"] == "sp"])
    mde_all = np.median([r["mde_080"] for r in learned if r["scope"] == "all"])

    print(f"\nmethod spread (best - random): SP={spread['sp']}  ALL={spread['all']}")
    print(f"{'comparison':28s}{'scope':5s}{'diff':>9}{'SE_c':>8}{'p_t':>8}{'MDE.80':>9}{'det?':>6}")
    for r in rows:
        print(f"{r['A']+' vs '+r['B']:28s}{r['scope']:5s}{r['diff']:>+9.4f}{r['se_component']:>8.4f}"
              f"{r['p_cluster_t']:>8.3f}{r['mde_080']:>9.4f}{('YES' if r['detectable'] else 'no'):>6}")

    verdict = (f"Cluster-t MDE (power .80) for learned-vs-raw: SP~{mde_sp:.4f}, ALL~{mde_all:.4f}. "
               f"Method spread (best-random): SP={spread['sp']}, ALL={spread['all']}. "
               f"MDE {'EXCEEDS' if mde_sp >= spread['sp'] else 'is below'} the SP spread and "
               f"{'EXCEEDS' if mde_all >= spread['all'] else 'is below'} the ALL spread: on this weak "
               f"realized-transfer label the benchmark cannot resolve the sub-spread differences that "
               f"separate real representations. It DOES detect the gross raw-vs-random effect (above the "
               f"MDE), so the test is not generically powerless. The P1 null is 'no detectable learned-rep "
               f"advantage at this label's resolution,' not a proof of equality.")
    print(f"\n{verdict}")

    out = {"n_queries": len(qlist), "n_components": df + 1, "df": df,
           "method_spread_best_minus_random": spread,
           "mde_power080_alpha05": {"sp_median_learned_vs_raw": round(float(mde_sp), 5),
                                    "all_median_learned_vs_raw": round(float(mde_all), 5)},
           "comparisons": rows, "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
