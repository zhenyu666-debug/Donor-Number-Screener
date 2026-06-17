"""Step 14 (v3): SHAP attribution for the 5-model ensemble.

Loads the existing best_params from `results/bayes_metrics_5model.json`,
re-fits the 4 tree models (RF, XGB, LGBM, CatBoost) on the v2
descriptors, computes per-model SHAP values, and averages the
contributions into a single global attribution plot.

Why not include MLP?  `shap.KernelExplainer` for MLPs is O(N^2) and
takes ~20 min on 3500 samples — not worth it for the demo.  Tree
models use the fast `TreeExplainer` (exact, <1 s per model).

Outputs (under results/):
  shap_top20.png              - global feature attribution (bar) PNG
  shap_top20_attribution.csv  - per-feature |SHAP| for the 4 tree models
  shap_summary.json           - the metadata

Usage:
  python src/14_shap_explain.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESULTS_DIR, get_logger  # noqa: E402

warnings.filterwarnings("ignore")
random.seed(42)
log = get_logger("shap")

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
    log.info("Loaded v2 X=%s  feats=%d", X.shape, len(feat_cols))
    return df, X, y, feat_cols


def fit_4_tree_models(X: np.ndarray, y: np.ndarray, params: dict) -> dict:
    """Fit RF / XGB / LGBM / CatBoost with the bayes best_params."""
    models: dict = {}

    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(
        **{**params["rf"], "random_state": 42, "n_jobs": -1}
    ).fit(X, y)
    models["rf"] = ("tree", rf)
    log.info("RF fitted (R2 train=%.4f)", rf.score(X, y))

    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        **{**params["xgb"], "random_state": 42, "n_jobs": -1,
           "verbosity": 0, "tree_method": "hist"}
    ).fit(X, y)
    models["xgb"] = ("tree", xgb)
    log.info("XGB fitted (R2 train=%.4f)", xgb.score(X, y))

    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMRegressor(
            **{**params["lgbm"], "random_state": 42, "n_jobs": -1, "verbosity": -1}
        ).fit(X, y)
        models["lgbm"] = ("tree", lgbm)
        log.info("LGBM fitted (R2 train=%.4f)", lgbm.score(X, y))
    except ImportError:
        log.warning("lightgbm not installed; skipping SHAP for LGBM")

    try:
        from catboost import CatBoostRegressor
        cat = CatBoostRegressor(
            **{**params["cat"], "random_seed": 42, "verbose": False,
               "thread_count": -1}
        ).fit(X, y)
        models["cat"] = ("tree", cat)
        log.info("CatBoost fitted (R2 train=%.4f)", cat.score(X, y))
    except ImportError:
        log.warning("catboost not installed; skipping SHAP for CatBoost")

    return models


def compute_global_shap(models: dict, X: np.ndarray, feat_cols: list[str]) -> pd.DataFrame:
    """Sum of |SHAP| per feature, averaged over the 4 tree models.

    Returns a DataFrame sorted by mean importance, one row per feature.
    """
    try:
        import shap
    except ImportError:
        log.error("shap not installed. Run: pip install shap")
        raise

    # Background dataset for tree models: 200 random rows is enough
    # (TreeExplainer only needs a small background for exact attribution)

    contribs = np.zeros((len(feat_cols),), dtype=np.float64)
    n_tree = 0
    for name, (_kind, model) in models.items():
        log.info("Computing SHAP for %s ...", name)
        t0 = time.perf_counter()
        try:
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            log.warning("  TreeExplainer failed for %s: %s", name, e)
            continue
        try:
            sv = explainer.shap_values(X)  # full dataset
        except Exception as e:
            log.warning("  shap_values failed for %s: %s; using 500-row subset", name, e)
            sv = explainer.shap_values(X[:500])
        contribs += np.abs(sv).mean(axis=0)
        n_tree += 1
        log.info("  done in %.1f s", time.perf_counter() - t0)

    if n_tree == 0:
        raise RuntimeError("no tree models produced SHAP values")

    contribs /= n_tree
    df = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": contribs})
    df.sort_values("mean_abs_shap", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def render_top20_png(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Ensure the parent dir exists (sometimes the results/ dir is missing
    # when the script is run in a fresh checkout).
    path.parent.mkdir(parents=True, exist_ok=True)

    top = df.head(20).iloc[::-1]  # barh reads bottom-up
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#3b6fb6")
    ax.set_xlabel("Mean |SHAP value|  (averaged over 4 tree models)")
    ax.set_title("Top 20 features by SHAP attribution\ndonor-number-screener v3")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    # Save and assert the file was actually written.  This guards against
    # silent failures (e.g. read-only filesystem, missing CWD, or any
    # matplotlib backend issue that swallowed an error).
    try:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)
    if not path.exists() or path.stat().st_size < 50_000:
        raise RuntimeError(
            f"shap_top20.png was not written: exists={path.exists()}, "
            f"size={path.stat().st_size if path.exists() else 0}"
        )
    log.info("Wrote %s (%.1f KB)", path, path.stat().st_size / 1024)


def main() -> None:
    t0 = time.perf_counter()

    metrics_path = RESULTS_DIR / "bayes_metrics_5model.json"
    if not metrics_path.exists():
        log.error("Run src/09c_5model_stacking.py first to produce %s", metrics_path)
        sys.exit(1)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    params = metrics["best_params"]

    _df, X, _y, feat_cols = load_v2()
    models = fit_4_tree_models(X, _y, params)
    log.info("Fitted %d tree models in %.1f s", len(models), time.perf_counter() - t0)

    contrib_df = compute_global_shap(models, X, feat_cols)

    # Save CSV + PNG + JSON
    out_csv = RESULTS_DIR / "shap_top20_attribution.csv"
    contrib_df.head(20).to_csv(out_csv, index=False)
    log.info("Wrote %s (top 20 features)", out_csv)

    out_png = RESULTS_DIR / "shap_top20.png"
    render_top20_png(contrib_df, out_png)
    log.info("Wrote %s", out_png)

    out_json = RESULTS_DIR / "shap_summary.json"
    out_json.write_text(json.dumps({
        "n_tree_models":        len(models),
        "models_used":          list(models.keys()),
        "n_features_total":     len(feat_cols),
        "top1_feature":         contrib_df.iloc[0]["feature"],
        "top1_mean_abs_shap":   float(contrib_df.iloc[0]["mean_abs_shap"]),
        "top20_first":          contrib_df.head(20)["feature"].tolist(),
        "wall_time_s":          time.perf_counter() - t0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", out_json)

    print("\n===== Top 10 SHAP features =====")
    for i, row in contrib_df.head(10).iterrows():
        print(f"  {i+1:>2}. {row['feature']:<30}  mean|SHAP| = {row['mean_abs_shap']:.4f}")
    log.info("=== Done in %.1f s ===", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
