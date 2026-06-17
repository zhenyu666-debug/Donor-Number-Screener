"""Step 16 (v3): Feature stability analysis.

Runs the full v2 descriptor + train + RF-importance pipeline N times
with different random seeds, and reports how stable the top-K feature
set is across runs (Jaccard index).

A Jaccard of 1.0 = identical top-K every run; 0.0 = disjoint.
We expect > 0.7 for the top-100 features if the descriptor set
is robust.

Outputs (under results/):
  feature_stability.json - per-K Jaccard, consensus top-K, runtimes
  feature_stability.csv  - per-run top-K

Usage:
  python src/16_feat_stability.py [--n-runs 5] [--k-list 50,100,200]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESULTS_DIR, get_logger  # noqa: E402

warnings.filterwarnings("ignore")
log = get_logger("feat_stability")

NON_FEATURE_COLS = {
    "mol_id", "smiles", "smiles_x", "smiles_y",
    "dn_rf", "dn_empirical", "dn_final", "confidence", "is_anchor",
}


def load_v2() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    desc = pd.read_csv(DATA_DIR / "descriptors_v2.csv")
    labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
    df = desc.merge(labels[["mol_id", "dn_final"]], on="mol_id", how="left")
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and df[c].dtype != "O"]
    X = df[feat_cols].values.astype(np.float64)
    y = df["dn_final"].values.astype(np.float64)
    return df, X, y, feat_cols


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def run_one(seed: int, X: np.ndarray, y: np.ndarray) -> tuple[list[str], float]:
    """One stability run: train an RF, return the ranked feature list."""
    t0 = time.perf_counter()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                              random_state=seed)
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=14, min_samples_split=4,
        min_samples_leaf=3, max_features=0.3, bootstrap=False,
        n_jobs=-1, random_state=seed,
    ).fit(X_tr, y_tr)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    ranked = [int(i) for i in order]
    return ranked, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--k-list", default="50,100,200,500",
                    help="comma-separated list of K values")
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "feature_stability.json"))
    ap.add_argument("--out-csv",  default=str(RESULTS_DIR / "feature_stability.csv"))
    args = ap.parse_args()

    k_list = [int(k) for k in args.k_list.split(",") if k.strip()]
    _df, X, y, feat_cols = load_v2()
    log.info("Loaded v2 X=%s  feats=%d  n_runs=%d  k_list=%s",
             X.shape, len(feat_cols), args.n_runs, k_list)

    # All runs
    rankings: list[list[int]] = []      # full rank-permutation, length = n_feats
    runtimes: list[float] = []
    seeds = [42 + 17 * i for i in range(args.n_runs)]
    for i, s in enumerate(seeds, start=1):
        ranking, dt = run_one(s, X, y)
        rankings.append(ranking)
        runtimes.append(dt)
        log.info("Run %d / %d  (seed=%d)  %.1f s", i, args.n_runs, s, dt)

    # Per-K Jaccard
    jaccard_by_k: dict[str, float] = {}
    for k in k_list:
        sets = [set(r[:k]) for r in rankings]
        # Average pairwise Jaccard
        vals: list[float] = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                vals.append(jaccard(sets[i], sets[j]))
        jaccard_by_k[f"top_{k}"] = float(np.mean(vals)) if vals else 1.0

    # Consensus top-K (intersection of all runs) + union
    consensus: dict[str, dict] = {}
    if rankings:
        for k in k_list:
            top_sets = [set(r[:k]) for r in rankings]
            inter = set.intersection(*top_sets) if top_sets else set()
            union = set.union(*top_sets)  if top_sets else set()
            consensus[f"top_{k}"] = {
                "intersection_size": len(inter),
                "union_size": len(union),
                "intersection_ratio": len(inter) / max(1, k),
                "union_ratio": len(union) / max(1, k),
                # Encode the consensus features (top intersection) by their column name
                "example_features": [feat_cols[i] for i in list(inter)[:10]],
            }

    # Save per-run top-100 CSV
    rows = []
    top_n = max(k_list)
    for i, r in enumerate(rankings):
        for j in range(min(top_n, len(r))):
            rows.append({
                "run": i,
                "rank": j + 1,
                "feature": feat_cols[r[j]],
            })
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    log.info("Wrote %s  (rows=%d)", args.out_csv, len(rows))

    out = {
        "n_runs":           args.n_runs,
        "seeds":            seeds,
        "k_list":           k_list,
        "n_features":       len(feat_cols),
        "jaccard_by_k":     jaccard_by_k,
        "consensus":        consensus,
        "runtime_per_run_s": runtimes,
        "total_runtime_s":  sum(runtimes),
        "interpretation": (
            "Jaccard > 0.7 = stable; > 0.85 = highly stable; "
            "< 0.5 = unstable (consider more features or re-run cleaning)."
        ),
    }
    Path(args.out_json).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote %s", args.out_json)

    print("\n===== Feature stability summary =====")
    for kk, jj in jaccard_by_k.items():
        verdict = "stable" if jj > 0.7 else ("marginal" if jj > 0.5 else "unstable")
        print(f"  {kk:>10s}  Jaccard = {jj:.4f}  ({verdict})")
    print(f"  consensus top-100  intersection ratio = "
          f"{consensus['top_100']['intersection_ratio']:.4f}")


if __name__ == "__main__":
    main()
