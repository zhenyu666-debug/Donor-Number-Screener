"""Step 4a: Data quality + feature selection.

Operates on the v2 descriptor set (`data/descriptors_v2.csv`) and the
existing `data/dn_labels.csv`.

Steps
-----
1. Drop descriptor columns with > 50 % NaN values
2. Drop duplicate SMILES rows (keep first)
3. Drop DN-label outliers beyond ±3σ
4. Drop rows with non-finite DN labels
5. Run RF feature-importance on the cleaned matrix and emit a
   `top_k_features.json` for downstream training, where k in
   {40, 80, 120, 200, 400, 996}.

Outputs (under data/ unless noted):
  descriptors_v2_clean.csv         — cleaned descriptor matrix
  dn_labels_clean.csv              — cleaned labels
  feature_importance_v2.csv        — full ranked importance
  top_k_features.json              — k -> list of feature names
  results/clean_meta.json          — row/col counts + thresholds
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils import DATA_DIR, RESULTS_DIR, get_logger, set_global_seed

warnings.filterwarnings("ignore")
set_global_seed(42)
log = get_logger("clean_feats")


NON_FEATURE_COLS = {
    "mol_id", "smiles", "smiles_x", "smiles_y",
    "dn_rf", "dn_empirical", "dn_final", "confidence", "is_anchor",
}


def main():
    t0 = time.time()
    desc = pd.read_csv(DATA_DIR / "descriptors_v2.csv")
    labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
    log.info("Raw: desc=%d rows x %d cols, labels=%d rows",
             len(desc), desc.shape[1], len(labels))

    # ---- 1. merge on mol_id ---- #
    df = desc.merge(labels[["mol_id", "dn_final"]], on="mol_id", how="inner")
    log.info("After merge on mol_id: %d rows", len(df))

    # ---- 2. drop NaN-heavy columns ---- #
    n_nan = df.isna().sum()
    cols_drop_nan = n_nan[n_nan > 0.5 * len(df)].index.tolist()
    log.info("Dropping %d columns with >50%% NaN: %s",
             len(cols_drop_nan), cols_drop_nan[:5])
    df = df.drop(columns=cols_drop_nan)

    # ---- 3. drop duplicate SMILES (keep first) ---- #
    n_dup = df.duplicated(subset=["smiles"]).sum()
    if n_dup:
        log.info("Dropping %d duplicate SMILES rows", n_dup)
        df = df.drop_duplicates(subset=["smiles"], keep="first")

    # ---- 4. drop non-finite DN ---- #
    n_nan_y = (~np.isfinite(df["dn_final"])).sum()
    if n_nan_y:
        log.info("Dropping %d non-finite DN rows", n_nan_y)
        df = df[np.isfinite(df["dn_final"])]

    # ---- 5. outlier removal: |DN - median| > 3 sigma ---- #
    y = df["dn_final"].values
    med = float(np.median(y))
    sigma = float(np.std(y))
    lo, hi = med - 3 * sigma, med + 3 * sigma
    mask = (y >= lo) & (y <= hi)
    n_out = int((~mask).sum())
    log.info("Outlier threshold [%.2f, %.2f]  dropping %d rows",
             lo, hi, n_out)
    df = df[mask].reset_index(drop=True)

    # ---- 6. identify feature columns ---- #
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and df[c].dtype != "O"]

    # Fill any remaining NaN with 0 (Mordred-like columns that
    # sometimes return NaN for malformed substructures).
    n_nan_left = df[feat_cols].isna().any(axis=1).sum()
    if n_nan_left:
        log.info("Filling %d remaining NaN rows with 0", n_nan_left)
        df[feat_cols] = df[feat_cols].fillna(0.0)

    log.info("Clean: %d rows x %d feature columns", len(df), len(feat_cols))

    # ---- 7. feature importance via RF ---- #
    X = df[feat_cols].values.astype(np.float32)
    y = df["dn_final"].values.astype(np.float64)
    log.info("Training RF for feature importance...")
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=None,
        n_jobs=-1, random_state=42
    )
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=feat_cols)
    imp = imp.sort_values(ascending=False)
    imp.to_csv(DATA_DIR / "feature_importance_v2.csv", header=["importance"])

    # ---- 8. emit top-k lists ---- #
    K_VALUES = (40, 80, 120, 200, 400, len(feat_cols))
    top_k = {str(k): imp.head(k).index.tolist() for k in K_VALUES}
    (DATA_DIR / "top_k_features.json").write_text(
        json.dumps(top_k, indent=2)
    )

    # ---- 9. write cleaned data ---- #
    front = ["mol_id", "smiles", "dn_final"]
    out = df[front + feat_cols]
    out.to_csv(DATA_DIR / "descriptors_v2_clean.csv", index=False)
    out[["mol_id", "smiles", "dn_final"]].to_csv(
        DATA_DIR / "dn_labels_clean.csv", index=False
    )

    # ---- 10. save metadata ---- #
    meta = {
        "raw_n_rows": int(len(desc)),
        "raw_n_cols": int(desc.shape[1]),
        "clean_n_rows": int(len(df)),
        "clean_n_cols": int(len(feat_cols)),
        "dropped_columns_nan50pct": cols_drop_nan,
        "dropped_dup_smiles": int(n_dup),
        "dropped_nonfinite_dn": int(n_nan_y),
        "dropped_outliers_3sigma": int(n_out),
        "dn_y_median": med,
        "dn_y_std": sigma,
        "wall_time_s": round(time.time() - t0, 2),
        "top10_features": imp.head(10).index.tolist(),
    }
    (RESULTS_DIR / "clean_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    log.info("Wrote clean_meta.json  top10: %s", meta["top10_features"])
    print("\n--- Cleaning summary ---")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
