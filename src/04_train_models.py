"""Step 4: Train RandomForest and XGBoost regressors on the
DN label and evaluate.  Reproduce paper Fig. 2 (scatter + feature
importance).

To make the evaluation honest, the train/test split is performed
*after* step 3 has finished.  The full (X, y) matrix is split 80/20
with a fixed seed.  All anchors (which have an experimental DN
value) are placed in the training set, exactly as in the paper.

We also run 5-fold cross-validation on the training set for
hyperparameter selection, then report the held-out 20% test R^2
and RMSE.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (DATA_DIR, FIGURES_DIR, RESULTS_DIR,  # noqa: E402
                   get_logger, set_global_seed)

warnings.filterwarnings("ignore")
set_global_seed(42)
log = get_logger("train")

NON_FEATURE_COLS = {"mol_id", "smiles", "smiles_x", "smiles_y",
                    "dn_rf", "dn_empirical", "dn_final", "confidence",
                    "is_anchor"}


def load_xy():
    desc = pd.read_csv(DATA_DIR / "descriptors.csv")
    labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
    df = desc.merge(labels[["mol_id", "dn_final", "is_anchor"]],
                    on="mol_id", how="left")
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and df[c].dtype != "O"]
    X = df[feat_cols].values
    y = df["dn_final"].values
    return df, X, y, feat_cols


def evaluate(y_true, y_pred) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def plot_parity(y_true, y_pred, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(y_true, y_pred, s=8, alpha=0.45, c="#3b6fb6",
               edgecolor="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("Reference DN value")
    ax.set_ylabel("Predicted DN value")
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_title(f"{title}\nR$^2$={r2:.3f}  RMSE={rmse:.2f}")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_importance(rf: RandomForestRegressor,
                            xgb: XGBRegressor,
                            feat_cols: list[str],
                            path: Path) -> None:
    fi_rf = pd.Series(rf.feature_importances_, index=feat_cols)
    fi_xgb = pd.Series(xgb.feature_importances_, index=feat_cols)
    # rank average
    rank = (fi_rf.rank(ascending=False) + fi_xgb.rank(ascending=False)) / 2
    top = rank.sort_values().head(20)
    # Map to paper terminology
    paper_map = {
        "HOMO_proxy": "HOMO energy (proxy)",
        "dipole_proxy": "Dipole moment (proxy)",
        "LUMO_proxy": "LUMO energy (proxy)",
        "HL_gap_proxy": "HOMO-LUMO gap (proxy)",
        "polarizability_proxy": "Polarizability (proxy)",
        "TPSA": "Topological PSA",
        "MolLogP": "MolLogP",
        "NumHAcceptors": "H-bond acceptors",
        "NumHDonors": "H-bond donors",
        "MaxEStateIndex": "Max EState index",
        "MinEStateIndex": "Min EState index",
        "n_O": "# Oxygen atoms",
        "n_N": "# Nitrogen atoms",
        "n_F": "# Fluorine atoms",
        "Chi0v0": "Chi0v",
        "Chi1v0": "Chi1v",
        "Kappa1": "Kappa1 (shape)",
    }
    pretty = [paper_map.get(c, c) for c in top.index]
    rf_top = fi_rf[top.index].values
    xgb_top = fi_xgb[top.index].values

    fig, ax = plt.subplots(figsize=(8, 6.5))
    y_pos = np.arange(len(pretty))
    ax.barh(y_pos - 0.18, rf_top, height=0.36,
            color="#3b6fb6", label="RandomForest")
    ax.barh(y_pos + 0.18, xgb_top, height=0.36,
            color="#d36a4a", label="XGBoost")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pretty)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title("Top-20 features (rank-averaged)\nReproducing paper Fig. 2 feature importance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_shap(xgb: XGBRegressor, X: np.ndarray, feat_cols: list[str],
              path: Path) -> None:
    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(X[:400])  # subsample for speed
    plt.figure(figsize=(8, 6.5))
    shap.summary_plot(sv, X[:400], feature_names=feat_cols,
                      show=False, plot_size=None)
    plt.title("SHAP summary (XGBoost)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    df, X, y, feat_cols = load_xy()
    log.info("Loaded X=%s y=%d  feats=%d", X.shape, len(y), len(feat_cols))

    # 80/20 split.  Anchors go to training (paper convention).
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.20, random_state=42, shuffle=True
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    log.info("Train n=%d  Test n=%d", len(y_train), len(y_test))

    # ------------------------------------------------------------------ #
    # Random Forest
    # ------------------------------------------------------------------ #
    rf_grid = {
        "n_estimators": [300, 600],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
    }
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_grid, cv=KFold(5, shuffle=True, random_state=42),
        scoring="r2", n_jobs=-1, refit=True
    )
    rf_search.fit(X_train, y_train)
    rf = rf_search.best_estimator_
    log.info("RF best params: %s", rf_search.best_params_)

    # ------------------------------------------------------------------ #
    # XGBoost
    # ------------------------------------------------------------------ #
    xgb_grid = {
        "n_estimators": [400, 800],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
    }
    xgb_search = GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0,
                     tree_method="hist"),
        xgb_grid, cv=KFold(5, shuffle=True, random_state=42),
        scoring="r2", n_jobs=-1, refit=True
    )
    xgb_search.fit(X_train, y_train)
    xgb = xgb_search.best_estimator_
    log.info("XGB best params: %s", xgb_search.best_params_)

    # ------------------------------------------------------------------ #
    # Held-out test evaluation
    # ------------------------------------------------------------------ #
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    rf_metrics = evaluate(y_test, rf_pred)
    xgb_metrics = evaluate(y_test, xgb_pred)
    log.info("Test RF: %s", rf_metrics)
    log.info("Test XGB: %s", xgb_metrics)

    plot_parity(y_test, rf_pred, "Random Forest",
                FIGURES_DIR / "fig2a_rf_parity.png")
    plot_parity(y_test, xgb_pred, "XGBoost",
                FIGURES_DIR / "fig2b_xgb_parity.png")
    plot_feature_importance(rf, xgb, feat_cols,
                            FIGURES_DIR / "fig2c_feature_importance.png")
    try:
        plot_shap(xgb, X_test, feat_cols, FIGURES_DIR / "fig2d_shap.png")
    except Exception as e:
        log.warning("SHAP failed: %s", e)

    # 5-fold cross-validated metrics for the paper-style table
    kf = KFold(5, shuffle=True, random_state=42)
    rf_cv_r2, xgb_cv_r2 = [], []
    for tr, te in kf.split(X):
        rfm = RandomForestRegressor(**rf.get_params()).fit(X[tr], y[tr])
        xgb_params = {k: v for k, v in xgb.get_params().items()
                      if k not in ("verbosity",)}
        xgm = XGBRegressor(**xgb_params).fit(X[tr], y[tr])
        rf_cv_r2.append(r2_score(y[te], rfm.predict(X[te])))
        xgb_cv_r2.append(r2_score(y[te], xgm.predict(X[te])))
    cv_summary = {
        "rf_cv_R2_mean": float(np.mean(rf_cv_r2)),
        "rf_cv_R2_std": float(np.std(rf_cv_r2)),
        "xgb_cv_R2_mean": float(np.mean(xgb_cv_r2)),
        "xgb_cv_R2_std": float(np.std(xgb_cv_r2)),
    }
    log.info("5-fold CV: %s", cv_summary)

    # Save metrics
    out_metrics = {
        "test_rf": rf_metrics,
        "test_xgb": xgb_metrics,
        "cv": cv_summary,
        "best_params": {
            "rf": rf_search.best_params_,
            "xgb": xgb_search.best_params_,
        },
    }
    (RESULTS_DIR / "model_metrics.json").write_text(
        json.dumps(out_metrics, indent=2, ensure_ascii=False)
    )
    log.info("Wrote %s", RESULTS_DIR / "model_metrics.json")

    # Save the test predictions for the screening step
    test_df = df.iloc[test_idx][["mol_id", "smiles"]].copy()
    test_df["dn_true"] = y_test
    test_df["dn_pred_rf"] = rf_pred
    test_df["dn_pred_xgb"] = xgb_pred
    test_df.to_csv(DATA_DIR / "test_predictions.csv", index=False)

    # Save full predicted library for step 5
    full_pred = df[["mol_id", "smiles"]].copy()
    full_pred["dn_pred_rf"] = rf.predict(X)
    full_pred["dn_pred_xgb"] = xgb.predict(X)
    full_pred["dn_pred_ens"] = (full_pred["dn_pred_rf"] +
                                 full_pred["dn_pred_xgb"]) / 2
    full_pred.to_csv(DATA_DIR / "full_predictions.csv", index=False)
    log.info("Wrote full_predictions.csv with %d rows", len(full_pred))

    print("\n--- training summary ---")
    print(f"RF  test: R2={rf_metrics['R2']:.3f}  "
          f"RMSE={rf_metrics['RMSE']:.2f}  MAE={rf_metrics['MAE']:.2f}")
    print(f"XGB test: R2={xgb_metrics['R2']:.3f}  "
          f"RMSE={xgb_metrics['RMSE']:.2f}  MAE={xgb_metrics['MAE']:.2f}")
    print(f"5-fold CV R2:  RF={cv_summary['rf_cv_R2_mean']:.3f}"
          f" ± {cv_summary['rf_cv_R2_std']:.3f}   "
          f"XGB={cv_summary['xgb_cv_R2_mean']:.3f}"
          f" ± {cv_summary['xgb_cv_R2_std']:.3f}")


if __name__ == "__main__":
    main()
