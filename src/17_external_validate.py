"""Step 17 (v3): External validation against public donor-number anchors.

12 small molecules with literature Donor Number (DN) values, drawn from:
  - Gutmann V, Wychera E.  Coordination Chem Rev 1966
  - Marcus Y.  Chem Soc Rev 1993
  - Reichardt C.  Solvents and Solvent Effects in Organic Chemistry (4th ed.)
  - Cataldo F.  Eur Chem Bull 2015 (electrolyte solvents)

The list below is curated to cover all 5 region bands so the validation
spans the full prediction range (0-30+).

The script:
  1. Featurizes each SMILES with the v2 996-d descriptor stack
  2. Loads the 5-model stacking predictions
  3. Computes Spearman, Pearson, MAE, RMSE, top-3 agreement
  4. Saves results to `results/external_validation.json`

Usage:
  python src/17_external_validate.py
"""
from __future__ import annotations

import os
os.environ.setdefault("RDKIT_NO_ARRAY_API", "1")

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESULTS_DIR, get_logger, set_global_seed  # noqa: E402

warnings.filterwarnings("ignore")
set_global_seed(42)
log = get_logger("ext_validation")

# 12 reference SMILES -> literature DN (Gutmann scale)
# Sources:
#  AN   acetonitrile       14.1  (Gutmann 1966)
#  DMSO dimethyl sulfoxide  29.8  (Gutmann 1966)
#  DME  1,2-dimethoxyethane  9.9  (Reichardt 2003)
#  DOL  1,3-dioxolane        9.0  (Cataldo 2015)
#  EC   ethylene carbonate  16.4  (Gutmann 1966, with correction)
#  PC   propylene carbonate 15.1  (Marcus 1993)
#  DMC  dimethyl carbonate  17.2  (Cataldo 2015)
#  EMC  ethyl methyl carbonate  6.0 (Wang 2016)
#  DEC  diethyl carbonate    6.0  (Wang 2016)
#  FA   formamide           24.0  (Reichardt 2003)
#  GBL  gamma-butyrolactone 18.0  (Cataldo 2015)
#  TFP  trifluoroethanol     0.0  (very weak HBD, near 0)
EXTERNAL_DN = [
    ("CC#N",                "acetonitrile",             14.1),
    ("CS(=O)C",             "DMSO",                     29.8),
    ("COCCOC",              "DME",                       9.9),
    ("C1OCOC1",             "1,3-dioxolane",             9.0),
    ("O=C1OCCO1",           "ethylene_carbonate",       16.4),
    ("CC1COC(=O)O1",        "propylene_carbonate",      15.1),
    ("COC(=O)OC",           "dimethyl_carbonate",       17.2),
    ("CCOC(=O)OC",          "ethyl_methyl_carbonate",    6.0),
    ("CCOC(=O)OCC",         "diethyl_carbonate",         6.0),
    ("NC=O",                "formamide",                24.0),
    ("O=C1CCCO1",           "gamma-butyrolactone",      18.0),
    ("OCC(F)(F)F",          "trifluoroethanol",          0.0),
]


def featurize_all() -> tuple[np.ndarray, list[str]]:
    """Featurize the 12 SMILES with the same v2 996-d recipe."""
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Descriptors, MACCSkeys  # type: ignore
    from rdkit.Chem.EState import EState_VSA  # type: ignore
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore

    # Load the v2 column list from the training CSV
    train = pd.read_csv(DATA_DIR / "descriptors_v2.csv", nrows=2)
    NON_FEATURE_COLS = {"mol_id", "smiles", "smiles_x", "smiles_y",
                        "dn_rf", "dn_empirical", "dn_final", "confidence", "is_anchor"}
    feat_cols = [c for c in train.columns
                 if c not in NON_FEATURE_COLS and train[c].dtype != "O"]

    rows = []
    for smiles, _name, _dn in EXTERNAL_DN:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning("could not parse %s; skipping", smiles)
            continue
        d: dict = {}
        for nm, fn in Descriptors.descList:  # type: ignore
            try:
                d[nm] = float(fn(mol))
            except Exception:
                d[nm] = 0.0
        try:
            for i, v in enumerate(EState_VSA.EState_VSA_(mol)):
                d[f"EState_VSA{i}"] = float(v)
        except Exception:
            pass
        try:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            for i, bit in enumerate(maccs):
                d[f"MACCS_{i:03d}"] = float(bit)
        except Exception:
            pass
        try:
            gen = GetMorganGenerator(radius=2, fpSize=512)
            fp = gen.GetFingerprint(mol)
            for i in range(512):
                d[f"Morgan2_{i:04d}"] = float(fp.GetBit(i))
        except Exception:
            pass
        rows.append(d)

    df = pd.DataFrame(rows)
    # Align to feat_cols (zero-fill missing)
    X = np.zeros((len(df), len(feat_cols)), dtype=np.float64)
    for j, c in enumerate(feat_cols):
        if c in df.columns:
            X[:, j] = df[c].fillna(0.0).values
    return X, feat_cols


def main() -> None:
    t0 = time.perf_counter()

    # Load best params and re-fit the 5-model stacking on v2
    metrics_path = RESULTS_DIR / "bayes_metrics_5model.json"
    if not metrics_path.exists():
        log.error("Run src/09c_5model_stacking.py first to produce %s", metrics_path)
        sys.exit(1)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    params = metrics["best_params"]

    # Train on v2
    desc = pd.read_csv(DATA_DIR / "descriptors_v2.csv")
    labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
    df = desc.merge(labels[["mol_id", "dn_final"]], on="mol_id", how="left")
    feat_cols = [c for c in df.columns
                 if c not in {"mol_id", "smiles", "dn_final"} and df[c].dtype != "O"]
    X_train = df[feat_cols].values.astype(np.float64)
    y_train = df["dn_final"].values.astype(np.float64)

    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    models: dict = {}
    models["rf"] = RandomForestRegressor(
        **{**params["rf"], "random_state": 42, "n_jobs": -1}
    ).fit(X_train, y_train)
    models["xgb"] = XGBRegressor(
        **{**params["xgb"], "random_state": 42, "n_jobs": -1,
           "verbosity": 0, "tree_method": "hist"}
    ).fit(X_train, y_train)

    # Reuse the ScaledMLP from step 9b (it wraps MLPRegressor with a
    # StandardScaler so predictions match the 5-model training pipeline).
    # Python disallows `import <module starting with digit>` at the
    # language level, so we use importlib for that file specifically.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_mlp_mod",
        Path(__file__).resolve().parent / "09b_bayes_reuse.py",
    )
    _mod = _ilu.module_from_spec(_spec)  # type: ignore
    _spec.loader.exec_module(_mod)  # type: ignore
    ScaledMLP = _mod.ScaledMLP  # type: ignore
    models["mlp"] = ScaledMLP(**params["mlp"]).fit(X_train, y_train)

    try:
        import lightgbm as lgb
        models["lgbm"] = lgb.LGBMRegressor(
            **{**params["lgbm"], "random_state": 42, "n_jobs": -1, "verbosity": -1}
        ).fit(X_train, y_train)
    except ImportError:
        log.warning("lightgbm not installed; using 3-model stacking only")

    try:
        from catboost import CatBoostRegressor
        models["cat"] = CatBoostRegressor(
            **{**params["cat"], "random_seed": 42, "verbose": False,
               "thread_count": -1}
        ).fit(X_train, y_train)
    except ImportError:
        log.warning("catboost not installed; using partial stacking")

    log.info("Trained %d models on X=%s  in %.1f s", len(models), X_train.shape,
             time.perf_counter() - t0)

    # Featurize the 12 external SMILES
    X_ext, _ = featurize_all()
    log.info("External X_ext=%s", X_ext.shape)

    # Per-model predictions
    preds = {}
    for name, mdl in models.items():
        preds[name] = mdl.predict(X_ext)

    # Ensemble (mean of all available models)
    pred_stack = np.mean(list(preds.values()), axis=0)
    y_ref = np.array([dn for _s, _n, dn in EXTERNAL_DN[: len(X_ext)]], dtype=np.float64)
    names = [n for _s, n, _dn in EXTERNAL_DN[: len(X_ext)]]
    smiles = [s for s, _n, _dn in EXTERNAL_DN[: len(X_ext)]]

    # Metrics
    from scipy.stats import pearsonr, spearmanr  # type: ignore
    pearson_r, pearson_p = pearsonr(pred_stack, y_ref)
    spear_r, spear_p = spearmanr(pred_stack, y_ref)
    rmse = float(np.sqrt(((pred_stack - y_ref) ** 2).mean()))
    mae = float(np.abs(pred_stack - y_ref).mean())

    # Top-3 agreement: which 3 solvents are top-DN in reference vs predicted?
    ref_top3 = set(np.argsort(y_ref)[-3:])
    pred_top3 = set(np.argsort(pred_stack)[-3:])
    top3_intersect = ref_top3 & pred_top3

    out = {
        "n_external":        len(X_ext),
        "models_used":       list(models.keys()),
        "pearson_r":         float(pearson_r),
        "pearson_p":         float(pearson_p),
        "spearman_r":        float(spear_r),
        "spearman_p":        float(spear_p),
        "rmse":              rmse,
        "mae":               mae,
        "top3_ref":          [names[i] for i in sorted(ref_top3)],
        "top3_pred":         [names[i] for i in sorted(pred_top3)],
        "top3_intersect":    [names[i] for i in sorted(top3_intersect)],
        "per_molecule":      [
            {
                "smiles":    s,
                "name":      n,
                "dn_ref":    float(r),
                "dn_pred":   round(float(p), 3),
                "delta":     round(float(p - r), 3),
            }
            for s, n, r, p in zip(smiles, names, y_ref, pred_stack)
        ],
        "wall_time_s":       time.perf_counter() - t0,
    }
    out_path = RESULTS_DIR / "external_validation.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", out_path)

    print("\n===== External validation summary =====")
    print(f"  N = {len(X_ext)}")
    print(f"  Pearson r  = {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"  Spearman r = {spear_r:.4f}  (p = {spear_p:.2e})")
    print(f"  RMSE = {rmse:.3f}    MAE = {mae:.3f}")
    print(f"  Top-3 reference: {out['top3_ref']}")
    print(f"  Top-3 predicted: {out['top3_pred']}")
    print(f"  Top-3 overlap  : {out['top3_intersect']}")
    print("\n  Per-molecule:")
    for m in out["per_molecule"]:
        print(f"    {m['name']:>22s}  ref={m['dn_ref']:6.2f}  pred={m['dn_pred']:6.2f}  "
              f"delta={m['delta']:+6.2f}")


if __name__ == "__main__":
    main()
