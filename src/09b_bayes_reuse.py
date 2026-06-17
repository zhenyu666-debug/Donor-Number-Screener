"""Step 9b: Reuse hardcoded best params from the previous Optuna run
and skip the search.  Used to regenerate plots and the metrics
JSON after the long search has been run once.

Best params (from logs):
  RF:  n_estimators=500  max_depth=14  min_samples_split=4
       min_samples_leaf=3  max_features=0.3  bootstrap=False
  XGB: n_estimators=1100  max_depth=4  learning_rate=0.04819
       subsample=0.8787  colsample_bytree=0.4945
       reg_alpha=1.981  reg_lambda=0.6693  min_child_weight=9
  MLP: hidden_layer_sizes=(192, 112)  activation=tanh
       alpha=0.6354  learning_rate_init=0.00870  max_iter=700
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from utils import (DATA_DIR, FIGURES_DIR, RESULTS_DIR,
                   get_logger, set_global_seed)

warnings.filterwarnings("ignore")
set_global_seed(42)
log = get_logger("bayes_reuse")

N_CV_FOLDS = 5
CV = KFold(N_CV_FOLDS, shuffle=True, random_state=42)


class ScaledMLP(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(128, 64, 32),
                 activation="relu", alpha=0.001,
                 learning_rate_init=0.001, max_iter=500,
                 early_stopping=True, validation_fraction=0.1,
                 n_iter_no_change=20, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def _make_mlp(self):
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        self.mlp_ = self._make_mlp()
        self.mlp_.fit(Xs, y)
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(X)
        return self.mlp_.predict(Xs)


def evaluate(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def plot_parity_matrix(y_true, preds_dict, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    axes = axes.flat
    colors = ["#3b6fb6", "#d36a4a", "#4a9c6d", "#7b5ea7"]
    titles = ["RandomForest (Bayesian)", "XGBoost (Bayesian)",
              "MLP Neural Network", "3-Model Stacking Ensemble"]
    keys = ["rf", "xgb", "mlp", "ensemble"]

    for ax, title, key, color in zip(axes, titles, keys, colors):
        yp = preds_dict[key]
        ax.scatter(y_true, yp, s=8, alpha=0.4, c=color, edgecolor="none")
        lo = min(y_true.min(), yp.min())
        hi = max(y_true.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        r2 = r2_score(y_true, yp)
        rmse = np.sqrt(mean_squared_error(y_true, yp))
        mae = mean_absolute_error(y_true, yp)
        ax.set_xlabel("Reference DN value")
        ax.set_ylabel("Predicted DN value")
        ax.set_title(f"{title}\nR$^2$={r2:.4f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(lo - 1, hi + 1)
        ax.set_ylim(lo - 1, hi + 1)

    fig.suptitle("Bayesian-optimized models vs reference DN\n"
                 "Step 9 — donor-number-screener pipeline",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cv_comparison(baseline_metrics, bayes_metrics, path):
    """baseline_metrics / bayes_metrics: {"rf": float, "xgb": float}."""
    models = ["RF", "XGB"]
    base = [baseline_metrics["rf"], baseline_metrics["xgb"]]
    bayes = [bayes_metrics["rf"], bayes_metrics["xgb"]]
    delta = [b - a for a, b in zip(base, bayes)]
    x = np.arange(len(models))
    width = 0.3
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, base, width, label="GridSearchCV (step 4)",
           color="#3b6fb6", alpha=0.85)
    ax.bar(x + width/2, bayes, width, label="Optuna Bayesian (step 9)",
           color="#4a9c6d", alpha=0.85)
    for xi, d in zip(x, delta):
        ax.annotate(f"+{d:.4f}" if d > 0 else f"{d:.4f}",
                    xy=(xi + width/2, bayes[xi] + 0.0005),
                    ha="center", va="bottom", fontsize=10,
                    color="#4a9c6d" if d >= 0 else "#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("5-fold CV R$^2$")
    ax.set_title("GridSearchCV vs Bayesian Optimization\n"
                 "60 Optuna trials per model")
    ax.set_ylim(min(min(base), min(bayes)) - 0.005,
                max(max(base), max(bayes)) + 0.005)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# Hardcoded best params from the run
BEST_PARAMS = {
    "rf": {
        "n_estimators": 500, "max_depth": 14,
        "min_samples_split": 4, "min_samples_leaf": 3,
        "max_features": 0.3, "bootstrap": False,
    },
    "xgb": {
        "n_estimators": 1100, "max_depth": 4,
        "learning_rate": 0.048189213238580546,
        "subsample": 0.8787285418507224,
        "colsample_bytree": 0.4945457183693173,
        "reg_alpha": 1.9809310941934146,
        "reg_lambda": 0.6692840166879787,
        "min_child_weight": 9,
    },
    "mlp": {
        "hidden_layer_sizes": (192, 112),
        "activation": "tanh",
        "alpha": 0.635374964687094,
        "learning_rate_init": 0.00869832693793297,
        "max_iter": 700,
    },
}


def main():
    t0 = time.time()

    baseline_path = RESULTS_DIR / "model_metrics.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    log.info("Baseline (GridSearchCV): RF R2=%.5f  XGB R2=%.5f",
             baseline.get("test_rf", {}).get("R2", float("nan")),
             baseline.get("test_xgb", {}).get("R2", float("nan")))

    desc = pd.read_csv(DATA_DIR / "descriptors.csv")
    labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
    df = desc.merge(labels[["mol_id", "dn_final"]], on="mol_id", how="left")
    NON_FEATURE_COLS = {
        "mol_id", "smiles", "smiles_x", "smiles_y",
        "dn_rf", "dn_empirical", "dn_final", "confidence", "is_anchor",
    }
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and df[c].dtype != "O"]
    X = df[feat_cols].values.astype(np.float64)
    y = df["dn_final"].values.astype(np.float64)
    log.info("Loaded X=%s  y=%d", X.shape, len(y))

    # Train all three
    log.info("Training RF with best params...")
    best_rf = RandomForestRegressor(
        **{**BEST_PARAMS["rf"], "random_state": 42, "n_jobs": -1}
    ).fit(X, y)

    log.info("Training XGB with best params...")
    best_xgb = XGBRegressor(
        **{**BEST_PARAMS["xgb"], "random_state": 42,
           "n_jobs": -1, "verbosity": 0, "tree_method": "hist"}
    ).fit(X, y)

    log.info("Training MLP with best params...")
    best_mlp = ScaledMLP(**BEST_PARAMS["mlp"]).fit(X, y)

    # OOF for stacking
    log.info("Computing OOF predictions for stacking...")
    oof_rf  = cross_val_predict(best_rf,  X, y, cv=CV, n_jobs=1)
    oof_xgb = cross_val_predict(best_xgb, X, y, cv=CV, n_jobs=1)
    oof_mlp = cross_val_predict(best_mlp, X, y, cv=CV, n_jobs=1)

    X_meta = np.column_stack([oof_rf, oof_xgb, oof_mlp])
    meta_learner = Ridge(alpha=1.0).fit(X_meta, y)
    oof_stack = meta_learner.predict(X_meta)
    log.info("Stacking meta-learner CV R2 = %.5f",
             r2_score(y, oof_stack))

    cv_preds = {
        "rf": oof_rf, "xgb": oof_xgb, "mlp": oof_mlp,
        "stack": oof_stack,
        "blend": (oof_rf + oof_xgb + oof_mlp) / 3,
    }
    cv_metrics = {k: evaluate(y, cv_preds[k]) for k in cv_preds}

    # Held-out test
    idx = np.arange(len(y))
    _, test_idx = train_test_split(idx, test_size=0.20, random_state=42)
    X_t, y_t = X[test_idx], y[test_idx]
    test_preds = {
        "rf":  best_rf.predict(X_t),
        "xgb": best_xgb.predict(X_t),
        "mlp": best_mlp.predict(X_t),
        "stack": meta_learner.predict(np.column_stack([
            best_rf.predict(X_t), best_xgb.predict(X_t), best_mlp.predict(X_t)
        ])),
    }
    test_metrics = {k: evaluate(y_t, test_preds[k]) for k in test_preds}

    # Plots
    parity_preds = {
        "rf": test_preds["rf"], "xgb": test_preds["xgb"],
        "mlp": test_preds["mlp"], "ensemble": test_preds["stack"],
    }
    plot_parity_matrix(y_t, parity_preds,
                       FIGURES_DIR / "fig_bayes_comparison.png")
    if baseline:
        # baseline: float per model, bayes: nested dict
        plot_cv_comparison(
            {"rf":  baseline["cv"]["rf_cv_R2_mean"],
             "xgb": baseline["cv"]["xgb_cv_R2_mean"]},
            {"rf":  cv_metrics["rf"]["R2"],
             "xgb": cv_metrics["xgb"]["R2"]},
            FIGURES_DIR / "fig_bayes_improvement.png",
        )

    # Save metrics JSON (always)
    out = {
        "n_trials_per_model": 60,
        "cv_folds": N_CV_FOLDS,
        "best_params": BEST_PARAMS,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "baseline_gridsearch": baseline,
    }
    (RESULTS_DIR / "bayes_metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )

    # Update full predictions
    full_pred = df[["mol_id", "smiles"]].copy()
    full_pred["dn_pred_rf_bayes"] = best_rf.predict(X)
    full_pred["dn_pred_xgb_bayes"] = best_xgb.predict(X)
    full_pred["dn_pred_mlp"] = best_mlp.predict(X)
    full_pred["dn_pred_stack"] = meta_learner.predict(
        np.column_stack([best_rf.predict(X), best_xgb.predict(X), best_mlp.predict(X)])
    )
    full_pred["dn_pred_3way_blend"] = (
        full_pred["dn_pred_rf_bayes"]
        + full_pred["dn_pred_xgb_bayes"]
        + full_pred["dn_pred_mlp"]
    ) / 3
    full_pred.to_csv(DATA_DIR / "full_predictions_bayes.csv", index=False)

    # Update top-20
    top20_bayes = full_pred.sort_values("dn_pred_stack", ascending=False).head(20)
    top20_bayes.to_csv(RESULTS_DIR / "top20_candidates_bayes.csv", index=False)

    elapsed = time.time() - t0
    log.info("=== Done in %.1f s ===", elapsed)
    print("\n===== Bayesian Optimization Summary =====")
    print("Trials per model: 60 (cached, reused)")
    print(f"CV folds:          {N_CV_FOLDS}")
    print(f"Total wall time:  {elapsed:.0f}s")
    print(f"RF   CV R2: {cv_metrics['rf']['R2']:.5f}")
    print(f"XGB  CV R2: {cv_metrics['xgb']['R2']:.5f}")
    print(f"MLP  CV R2: {cv_metrics['mlp']['R2']:.5f}")
    print(f"Stack CV R2: {cv_metrics['stack']['R2']:.5f}")
    print(f"Blend CV R2: {cv_metrics['blend']['R2']:.5f}")
    print(f"\nBaseline RF CV R2: {baseline.get('cv',{}).get('rf_cv_R2_mean','?'):.5f}")
    print(f"Baseline XGB CV R2: {baseline.get('cv',{}).get('xgb_cv_R2_mean','?'):.5f}")


if __name__ == "__main__":
    main()
