"""Step 3: Assign DN labels to every candidate molecule.

The paper's DN values come from DFT calculations of Li+ binding
energies.  We do not run DFT.  Instead we build a self-consistent
"DN label" from two independent sources:

  1. RandomForest trained on the 28 literature anchor molecules
     using the descriptor matrix from step 2.
  2. Linear empirical formula:
        DN_proxy = a*HOMO_proxy + b*dipole_proxy
                 + c*n_O + d*n_N + e*n_F + f
     with coefficients fit on the same anchor set.

The final label is the geometric mean of the two predictions, which
is then used in step 4 for training the RF/XGB regression models
that "replace DFT" with ML.

We also produce a confidence flag (high/medium/low) based on the
agreement between the two predictors, mirroring the paper's note
that DFT final-stage refinement is still required for borderline
candidates.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESULTS_DIR, get_logger, set_global_seed  # noqa: E402

warnings.filterwarnings("ignore")
set_global_seed(42)
log = get_logger("assign_dn")

# Constant columns to drop (don't help the model).
NON_FEATURE_COLS = {"mol_id", "smiles", "smiles_x", "smiles_y"}


def load_data():
    lib = pd.read_csv(DATA_DIR / "candidate_library.csv")
    # v4: prefer v2 descriptors (1005 dims) when available, fall back
    # to v1 (236 dims) for backward compatibility.
    v2_path = DATA_DIR / "descriptors_v2.csv"
    v1_path = DATA_DIR / "descriptors.csv"
    if v2_path.exists() and v2_path.stat().st_mtime > v1_path.stat().st_mtime:
        desc = pd.read_csv(v2_path)
    elif v2_path.exists():
        desc = pd.read_csv(v2_path)
    else:
        desc = pd.read_csv(v1_path)
    anchor = pd.read_csv(DATA_DIR / "dn_anchor_table.csv")
    # All numeric features (drop identifiers).
    feat_cols = [c for c in desc.columns
                 if c not in NON_FEATURE_COLS and desc[c].dtype != "O"]
    return lib, desc, anchor, feat_cols


def merge_anchors(lib: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe of (mol_id, dn_expt) for all anchor rows
    that are present in the candidate library.
    """
    out_rows = []
    seen_ids = set()
    for _, r in anchor.iterrows():
        smi = r["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        # Find mol_id by canonical smiles.
        hit = lib[lib["smiles"] == canon]
        if len(hit) == 0:
            log.warning("Anchor %s not in library: %s", r["name"], smi)
            continue
        mol_id = int(hit["mol_id"].iloc[0])
        if mol_id in seen_ids:
            log.info("Anchor %s (%s) duplicates mol_id %d; skipping",
                     r["name"], canon, mol_id)
            continue
        seen_ids.add(mol_id)
        out_rows.append({
            "mol_id": mol_id,
            "smiles": canon,
            "dn_expt": float(r["dn_expt"]),
            "name": r["name"],
        })
    return pd.DataFrame(out_rows)


def fit_empirical_formula(anchor_df: pd.DataFrame,
                          desc: pd.DataFrame) -> tuple[Ridge, list[str], dict]:
    """Fit DN = a*HOMO + b*dipole + c*n_O + d*n_N + e*n_F + f
    on the anchor subset.
    """
    feats = ["HOMO_proxy", "dipole_proxy", "n_O", "n_N", "n_F"]
    desc_indexed = desc.set_index("mol_id")
    valid_ids = [mid for mid in anchor_df["mol_id"] if mid in desc_indexed.index]
    if len(valid_ids) < len(anchor_df):
        log.warning("Dropping %d anchors not in descriptors",
                    len(anchor_df) - len(valid_ids))
    X = desc_indexed.loc[valid_ids, feats].values
    y = anchor_df.set_index("mol_id").loc[valid_ids, "dn_expt"].values
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    coefs = dict(zip(feats, model.coef_.tolist()))
    coefs["intercept"] = float(model.intercept_)
    log.info("Empirical formula R^2 on anchors: %.3f  MAE=%.2f", r2, mae)
    log.info("  coefs: %s", coefs)
    return model, feats, {"r2": r2, "mae": mae, "coefs": coefs}


def fit_rf_anchor(anchor_df: pd.DataFrame,
                  desc: pd.DataFrame,
                  feat_cols: list[str]) -> tuple[RandomForestRegressor, dict]:
    desc_indexed = desc.set_index("mol_id")
    valid_ids = [mid for mid in anchor_df["mol_id"] if mid in desc_indexed.index]
    X = desc_indexed.loc[valid_ids, feat_cols].values
    y = anchor_df.set_index("mol_id").loc[valid_ids, "dn_expt"].values
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=None, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    y_pred = rf.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    log.info("Anchor RF train R^2=%.3f  MAE=%.2f  n_anchor=%d", r2, mae, len(y))
    return rf, {"r2": r2, "mae": mae, "n_train": int(len(y))}


def main() -> None:
    lib, desc, anchor, feat_cols = load_data()
    log.info("lib=%d  desc=%d  anchor=%d  feats=%d",
             len(lib), len(desc), len(anchor), len(feat_cols))

    # 1. Map anchors to mol_ids.
    anchor_merged = merge_anchors(lib, anchor)
    log.info("Anchors found in library: %d / %d",
             len(anchor_merged), len(anchor))
    print(anchor_merged[["mol_id", "name", "dn_expt"]].to_string(index=False))

    # 2. Empirical formula on anchors.
    emp_model, emp_feats, emp_meta = fit_empirical_formula(anchor_merged, desc)

    # 3. RF on anchors using all features.
    rf, rf_meta = fit_rf_anchor(anchor_merged, desc, feat_cols)

    # 4. Predict DN for the full library.  Use a (desc, lib) merge so
    #    we never lose rows due to set_index NaN keys.
    desc_indexed = desc.set_index("mol_id")
    X_all = desc[feat_cols].values
    dn_rf = rf.predict(X_all)
    dn_emp = emp_model.predict(desc[emp_feats].values)

    # 5. Combine.  Use geometric mean when both are positive;
    #    for the small fraction that is <0 (mainly inert hydrocarbons
    #    with F), use the empirical formula directly.
    dn_emp_pos = np.clip(dn_emp, 0.1, None)
    dn_rf_pos = np.clip(dn_rf, 0.1, None)
    dn_geo = np.sqrt(dn_emp_pos * dn_rf_pos)

    # Confidence: high if the two predictions agree within 20 %.
    rel_diff = np.abs(dn_rf - dn_emp) / (np.maximum(np.abs(dn_emp), 1.0))
    confidence = np.where(rel_diff < 0.15, "high",
                  np.where(rel_diff < 0.30, "medium", "low"))

    # The anchors themselves get their experimental value, not the
    # predicted one, so the regression models in step 4 are training
    # on a self-consistent dataset anchored to reality.
    valid_desc_ids = set(int(m) for m in desc["mol_id"] if pd.notna(m))
    valid_anchors = anchor_merged[
        anchor_merged["mol_id"].isin(valid_desc_ids)
    ]
    desc_id_arr = desc["mol_id"].astype(int).to_numpy()
    is_anchor_mask = np.isin(desc_id_arr, valid_anchors["mol_id"].to_numpy())
    dn_final = dn_geo.copy()
    if is_anchor_mask.any():
        dn_final[is_anchor_mask] = valid_anchors.set_index("mol_id").loc[
            desc_id_arr[is_anchor_mask], "dn_expt"
        ].values

    # 6. Persist.
    out = desc[["mol_id", "smiles"]].copy()
    out["dn_rf"] = dn_rf
    out["dn_empirical"] = dn_emp
    out["dn_final"] = dn_final
    out["confidence"] = confidence
    out["is_anchor"] = is_anchor_mask
    out.to_csv(DATA_DIR / "dn_labels.csv", index=False)
    log.info("Wrote %s", DATA_DIR / "dn_labels.csv")

    meta = {
        "n_library": int(len(lib)),
        "n_descriptors": int(len(feat_cols)),
        "n_anchors": int(len(anchor_merged)),
        "anchor_rf_train": rf_meta,
        "empirical_formula": emp_meta,
        "dn_label": {
            "mean": float(dn_final.mean()),
            "std": float(dn_final.std()),
            "min": float(dn_final.min()),
            "max": float(dn_final.max()),
        },
        "confidence_distribution": {
            k: int(np.sum(confidence == k))
            for k in ("high", "medium", "low")
        },
    }
    (RESULTS_DIR / "dn_label_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    log.info("Wrote %s", RESULTS_DIR / "dn_label_meta.json")
    print("\n--- DN label summary ---")
    for k, v in meta["dn_label"].items():
        print(f"  {k:8s}: {v:.2f}")
    print(f"  confidence: {meta['confidence_distribution']}")


if __name__ == "__main__":
    main()
