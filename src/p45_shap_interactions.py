"""p45_shap_interactions.py - v5: SHAP interaction values for paired-feature chemistry.

Pairs the top-20 features from the global SHAP analysis and computes
SHAP interaction values using the 4 tree models (RF, XGB, LGBM, CatBoost).
Interaction values reveal paired-feature chemistry: e.g. (n_N, dipole) synergy
means N atoms boost the effect of high dipole moment on DN.

Outputs (under results/):
  shap_interactions_top20.png    - heatmap of top-20 feature pairs
  shap_interactions_summary.csv  - ranked list of strongest interactions
  shap_interactions_pairs.json   - structured interaction data

Usage:
  python src/p45_shap_interactions.py
  # Generates all outputs in results/

Dependencies: shap, matplotlib, pandas, numpy
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
log = get_logger("shap_interact")

NON_FEATURE_COLS = {
    "mol_id", "smiles", "smiles_x", "smiles_y",
    "dn_rf", "dn_empirical", "dn_final", "confidence", "is_anchor",
}

# ---------------------------------------------------------------------------
# Human-readable labels for chemical feature names
# ---------------------------------------------------------------------------
FEAT_LABELS: dict[str, str] = {
    "MaxEStateIndex": "Max E-state",
    "MinEStateIndex": "Min E-state",
    "MaxAbsEStateIndex": "Max |E-state|",
    "MinAbsEStateIndex": "Min |E-state|",
    "qed": "QED drug-likeness",
    "MolWt": "Molecular weight",
    "MolLogP": "LogP",
    "NumHDonors": "H-bond donors",
    "NumHAcceptors": "H-bond acceptors",
    "NumRotatableBonds": "Rotatable bonds",
    "NumHeteroatoms": "Heteroatoms",
    "NumAromaticHeterocycles": "Aromatic heterocycles",
    "NumAromaticCarbocycles": "Aromatic carbocycles",
    "NumSaturatedHeterocycles": "Saturated heterocycles",
    "NumSaturatedCarbocycles": "Saturated carbocycles",
    "NumAliphaticHeterocycles": "Aliphatic heterocycles",
    "NumAliphaticCarbocycles": "Aliphatic carbocycles",
    "RingCount": "Ring count",
    "HeavyAtomCount": "Heavy atoms",
    "NumValenceElectrons": "Valence electrons",
    "NumRadicalElectrons": "Radical electrons",
    "FractionCSP3": "Fraction CSP3",
    "NOCount": "N+O count",
    "NHOHCount": "N-H+O-H count",
    "NumC": "Carbon count",
    "NumN": "Nitrogen count",
    "NumO": "Oxygen count",
    "NumF": "Fluorine count",
    "NumP": "Phosphorus count",
    "NumS": "Sulfur count",
    "NumCl": "Chlorine count",
    "NumBr": "Bromine count",
    "NumI": "Iodine count",
    "MaxPartialCharge": "Max partial charge",
    "MinPartialCharge": "Min partial charge",
    "MaxAbsPartialCharge": "Max |partial charge|",
    "MinAbsPartialCharge": "Min |partial charge|",
    "TPSA": "Topological PSA",
    "LabuteASA": "Labute ASA",
    "Chi0": "Chi0",
    "Chi1": "Chi1",
    "Chi0n": "Chi0n",
    "Chi1n": "Chi1n",
    "Chi0v": "Chi0v",
    "Chi1v": "Chi1v",
    "Chi2v": "Chi2v",
    "Chi3v": "Chi3v",
    "Chi4v": "Chi4v",
    "HallKierAlpha": "Hall-Kier alpha",
    "Kappa1": "Kappa1",
    "Kappa2": "Kappa2",
    "Kappa3": "Kappa3",
    "NumBridgeheadAtoms": "Bridgehead atoms",
    "NumAmideBonds": "Amide bonds",
}


def _label(name: str) -> str:
    return FEAT_LABELS.get(name, name)


def load_top_features(n: int = 20) -> list[str]:
    """Load top-n features from existing SHAP attribution output."""
    path = RESULTS_DIR / "shap_top20_attribution.csv"
    if not path.exists():
        log.warning(
            "shap_top20_attribution.csv not found. "
            "Run `python src/14_shap_explain.py` first to generate it."
        )
        return []
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        log.warning("shap_top20_attribution.csv missing 'feature' column")
        return []
    return df["feature"].head(n).tolist()


def load_full_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load full library for training models."""
    desc = pd.read_csv(DATA_DIR / "descriptors_v2.csv")
    labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
    df = desc.merge(labels[["mol_id", "dn_final"]], on="mol_id", how="inner")
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and df[c].dtype != "O"]
    X = df[feat_cols].values.astype(np.float64)
    y = df["dn_final"].values.astype(np.float64)
    log.info("Loaded full data: X=%s", X.shape)
    return X, y, feat_cols


def load_best_params() -> dict:
    """Load best params from the 5-model results."""
    path = RESULTS_DIR / "bayes_metrics_5model.json"
    if not path.exists():
        log.warning("bayes_metrics_5model.json not found; using defaults")
        return {}
    with open(path) as f:
        return json.load(f)


def fit_tree_models(X: np.ndarray, y: np.ndarray, params: dict) -> dict:
    """Fit RF / XGB / LGBM / CatBoost on the full data."""
    models: dict = {}

    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(
        **{
            **params.get("rf", {"n_estimators": 200, "max_depth": 15}),
            "random_state": 42, "n_jobs": -1
        }
    ).fit(X, y)
    models["rf"] = rf
    log.info("RF fitted (R2=%.4f)", rf.score(X, y))

    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        **{
            **params.get("xgb", {"n_estimators": 200, "max_depth": 7}),
            "random_state": 42, "n_jobs": -1,
            "verbosity": 0, "tree_method": "hist"
        }
    ).fit(X, y)
    models["xgb"] = xgb
    log.info("XGB fitted (R2=%.4f)", xgb.score(X, y))

    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMRegressor(
            **{
                **params.get("lgbm", {"n_estimators": 200, "max_depth": 7}),
                "random_state": 42, "n_jobs": -1, "verbosity": -1
            }
        ).fit(X, y)
        models["lgbm"] = lgbm
        log.info("LGBM fitted (R2=%.4f)", lgbm.score(X, y))
    except ImportError:
        log.warning("lightgbm not available; skipping")

    try:
        from catboost import CatBoostRegressor
        cat = CatBoostRegressor(
            **{
                **params.get("cat", {"iterations": 200, "depth": 7}),
                "random_seed": 42, "verbose": False, "thread_count": -1
            }
        ).fit(X, y)
        models["cat"] = cat
        log.info("CatBoost fitted (R2=%.4f)", cat.score(X, y))
    except ImportError:
        log.warning("catboost not available; skipping")

    return models


def compute_interactions_shap(
    models: dict,
    X: np.ndarray,
    feat_cols: list[str],
    top_feats: list[str],
) -> np.ndarray:
    """Compute SHAP interaction values for top_feats x top_feats.

    Uses TreeExplainer.shap_interaction_values() on each model and averages.
    Returns a (n_top, n_top) matrix of mean |interaction| values.
    """
    try:
        import shap
    except ImportError:
        log.error("shap not installed. Run: pip install shap")
        raise

    feat_idx = {f: i for i, f in enumerate(feat_cols)}
    top_idx = [feat_idx[f] for f in top_feats]

    n = len(top_feats)
    interaction_matrix = np.zeros((n, n), dtype=np.float64)
    n_models = 0

    # Sample background for efficiency
    bg_size = min(100, X.shape[0])
    bg_idx = random.sample(range(X.shape[0]), bg_size)
    X_bg = X[bg_idx]

    for name, model in models.items():
        log.info("Computing interaction SHAP for %s ...", name)
        t0 = time.perf_counter()
        try:
            explainer = shap.TreeExplainer(model)
            interact = explainer.shap_interaction_values(X_bg)
            log.info(
                "  %s interaction matrix: %s (%.1fs)",
                name, interact.shape, time.perf_counter() - t0
            )
            interact_top = interact[:, top_idx][:, :, top_idx]
            interaction_matrix += np.abs(interact_top).mean(axis=0)
            n_models += 1
        except Exception as exc:
            log.warning("  %s failed: %s", name, exc)

    if n_models == 0:
        log.error("All models failed to produce interaction values")
        raise RuntimeError("SHAP interaction computation failed for all models")

    interaction_matrix /= n_models
    log.info("Averaged interaction matrix over %d models", n_models)
    return interaction_matrix


def plot_interaction_heatmap(
    matrix: np.ndarray,
    top_feats: list[str],
    out_path: Path,
) -> None:
    """Render the interaction matrix as a heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping heatmap")
        return

    n = len(top_feats)
    labels = [_label(f) for f in top_feats]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), max(8, n * 0.5)))

    vmax = float(np.percentile(matrix.max(axis=1), 95)) if n > 1 else 1.0
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_title(
        "SHAP Interaction Values (|mean|)\nTop-20 Feature Pairs — 4 Tree Models",
        fontsize=10, pad=10
    )
    ax.set_xlabel("Feature", fontsize=8)
    ax.set_ylabel("Feature", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Mean SHAP interaction|", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved interaction heatmap: %s", out_path)


def save_summary(
    matrix: np.ndarray,
    top_feats: list[str],
    out_csv: Path,
    out_json: Path,
) -> None:
    """Rank and save all feature pairs by interaction strength."""
    pairs = []
    n = len(top_feats)
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append({
                    "feature_a": top_feats[i],
                    "label_a": _label(top_feats[i]),
                    "feature_b": top_feats[j],
                    "label_b": _label(top_feats[j]),
                    "mean_abs_interaction": float(matrix[i, j]),
                })
    pairs.sort(key=lambda x: x["mean_abs_interaction"], reverse=True)

    df = pd.DataFrame(pairs)
    df.to_csv(out_csv, index=False)
    log.info("Saved interaction summary: %s (%d pairs)", out_csv, len(pairs))

    with open(out_json, "w") as f:
        json.dump({
            "n_features": n,
            "features": top_feats,
            "top_pairs": pairs[:50],
        }, f, indent=2)
    log.info("Saved interaction JSON: %s", out_json)


def chemical_insights(matrix: np.ndarray, top_feats: list[str]) -> list[str]:
    """Extract interpretable chemistry insights from the interaction matrix."""
    insights = []
    n = len(top_feats)
    feat_labels = [_label(f) for f in top_feats]

    pair_vals = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pair_vals.append((float(matrix[i, j]), feat_labels[i], feat_labels[j]))
    pair_vals.sort(key=lambda x: x[0], reverse=True)

    insights.append("Top-10 pairwise feature interactions (|mean SHAP interaction|):")
    for val, fa, fb in pair_vals[:10]:
        insights.append(f"  {fa} <-> {fb}: {val:.4f}")

    if n > 1:
        diag = [float(matrix[i, i]) for i in range(n)]
        off_diag = [float(matrix[i, j]) for i in range(n) for j in range(n) if i != j]
        avg_diag = np.mean(diag)
        avg_off = np.mean(off_diag)
        insights.append("")
        insights.append(f"Self-interaction (main effect) avg: {avg_diag:.4f}")
        insights.append(f"Cross-interaction (pairwise) avg: {avg_off:.4f}")
        if avg_diag > avg_off * 2:
            insights.append("Interpretation: features act largely independently")
        elif avg_off > avg_diag:
            insights.append("Interpretation: strong pairwise synergies detected")

    return insights


def main() -> None:
    t_start = time.perf_counter()

    log.info("=" * 60)
    log.info("p45 SHAP Interaction Analysis (v5)")
    log.info("=" * 60)

    top_feats = load_top_features(n=20)
    X_full, y_full, feat_cols = None, None, None
    if not top_feats:
        X_full, y_full, feat_cols = load_full_data()
        if len(feat_cols) < 20:
            log.error("Not enough features to compute interactions")
            return
        top_feats = feat_cols[:20]
        log.info("Using first 20 features as fallback set")

    params = load_best_params()
    if X_full is None:
        X_full, y_full, feat_cols = load_full_data()
    models = fit_tree_models(X_full, y_full, params)

    matrix = compute_interactions_shap(models, X_full, feat_cols, top_feats)

    out_png = RESULTS_DIR / "shap_interactions_top20.png"
    out_csv = RESULTS_DIR / "shap_interactions_summary.csv"
    out_json = RESULTS_DIR / "shap_interactions_pairs.json"

    plot_interaction_heatmap(matrix, top_feats, out_png)
    save_summary(matrix, top_feats, out_csv, out_json)

    for line in chemical_insights(matrix, top_feats):
        log.info(line)

    log.info("Total time: %.1fs", time.perf_counter() - t_start)
    log.info("Done. Outputs: %s/*.png|*.csv|*.json", RESULTS_DIR)


if __name__ == "__main__":
    main()
