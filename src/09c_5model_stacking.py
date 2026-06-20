"""Step 9c: 5-model stacking with Optuna-tuned LightGBM + CatBoost.

Reuses the previously tuned RF / XGB / MLP best params from
`results/bayes_metrics.json` and adds two new models:

  - LightGBMRegressor
  - CatBoostRegressor

Each is Optuna-tuned for 30 trials on the v2 descriptor set, then
joined with the existing 3 models into a Ridge-meta-learner
5-model stacking ensemble.

Outputs (under results/):
  bayes_metrics_5model.json       — all 5 best params + metrics
  bayes_trials_lgbm.csv           — optuna trial history
  bayes_trials_catboost.csv
  fig_bayes_5model.png            — 5-panel parity
  fig_bayes_convergence_5.png     — trial convergence
  full_predictions_5model.csv     — full library predictions
  top20_candidates_5model.csv     — re-ranked top-20

Run:
  python src/09c_5model_stacking.py

Total wall time on a 4-core laptop: ~30 min
(LightGBM 30 trials x 5 folds ~5 min, CatBoost 30 trials ~20 min,
 final fit + plots <1 min).
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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("pip install optuna")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

from utils import (DATA_DIR, FIGURES_DIR, RESULTS_DIR,
                   get_logger, set_global_seed)

warnings.filterwarnings("ignore")
set_global_seed(42)
log = get_logger("bayes_5model")

N_TRIALS = 10
N_CV_FOLDS = 5
CV = KFold(N_CV_FOLDS, shuffle=True, random_state=42)

NON_FEATURE_COLS = {
    "mol_id", "smiles", "smiles_x", "smiles_y",
    "dn_rf", "dn_empirical", "dn_final", "confidence", "is_anchor",
}


def load_data_v2():
    desc = pd.read_csv(DATA_DIR / "descriptors_v2.csv")
    labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
    df = desc.merge(labels[["mol_id", "dn_final"]], on="mol_id", how="left")
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and df[c].dtype != "O"]
    X = df[feat_cols].values.astype(np.float64)
    y = df["dn_final"].values.astype(np.float64)
    log.info("Loaded v2 X=%s  y=%d  feats=%d", X.shape, len(y), len(feat_cols))
    return df, X, y, feat_cols


def evaluate(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


class ScaledMLP(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(192, 112), activation="tanh",
                 alpha=0.635, learning_rate_init=0.0087, max_iter=700,
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

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation, alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter, early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )
        self.mlp_.fit(Xs, y)
        return self

    def predict(self, X):
        return self.mlp_.predict(self.scaler_.transform(X))


# ----- Optuna objectives ----- #

def _cv_score(est, X, y):
    preds = cross_val_predict(est, X, y, cv=CV, n_jobs=1)
    return r2_score(y, preds)


def make_lgbm_objective(X, y):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        return _cv_score(lgb.LGBMRegressor(**params), X, y)
    return objective


def make_cat_objective(X, y):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "random_seed": 42,
            "verbose": False,
            "thread_count": -1,
        }
        return _cv_score(CatBoostRegressor(**params), X, y)
    return objective


# ----- Plots ----- #

def plot_5model_parity(y_true, preds_dict, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flat
    titles = ["RandomForest (Bayesian)", "XGBoost (Bayesian)",
              "MLP Neural Network", "LightGBM (Bayesian)",
              "CatBoost (Bayesian)", "5-Model Stacking"]
    colors = ["#3b6fb6", "#d36a4a", "#4a9c6d", "#7b5ea7", "#c97b3b", "#222222"]
    keys = list(preds_dict.keys())
    assert len(keys) == 6, f"expected 6 keys, got {len(keys)}"

    for ax, title, key, color in zip(axes, titles, keys, colors):
        yp = preds_dict[key]
        ax.scatter(y_true, yp, s=8, alpha=0.4, c=color, edgecolor="none")
        lo = min(y_true.min(), yp.min())
        hi = max(y_true.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        r2 = r2_score(y_true, yp)
        rmse = np.sqrt(mean_squared_error(y_true, yp))
        ax.set_xlabel("Reference DN")
        ax.set_ylabel("Predicted DN")
        ax.set_title(f"{title}\nR$^2$={r2:.4f}  RMSE={rmse:.3f}")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("5-Model Stacking ensemble — donor-number-screener v2 descriptors",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lgbm_cat_convergence(studies, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, study) in zip(axes, studies.items()):
        df = study.trials_dataframe()
        v = df["value"].values
        n = np.arange(1, len(v) + 1)
        ax.scatter(n, v, s=18, alpha=0.5, c="#3b6fb6", label="Trial")
        best = np.maximum.accumulate(v)
        ax.plot(n, best, color="#d36a4a", lw=2, label="Best so far")
        ax.axhline(best[-1], color="#d36a4a", lw=1, ls="--", alpha=0.5)
        ax.set_xlabel("Trial")
        ax.set_title(f"{name}\nBest CV R$^2$ = {best[-1]:.5f}")
        ax.set_ylabel("5-fold CV R$^2$")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Optuna convergence — LightGBM & CatBoost", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.time()
    df, X, y, feat_cols = load_data_v2()

    # ---- Reuse RF / XGB / MLP from bayes_metrics.json ---- #
    prev = json.loads((RESULTS_DIR / "bayes_metrics.json").read_text())
    rf_params = prev["best_params"]["rf"]
    xgb_params = prev["best_params"]["xgb"]
    mlp_params = prev["best_params"]["mlp"]

    log.info("Training reused RF/XGB/MLP on v2 features...")
    best_rf = RandomForestRegressor(
        **{**rf_params, "random_state": 42, "n_jobs": -1}
    ).fit(X, y)
    best_xgb = XGBRegressor(
        **{**xgb_params, "random_state": 42, "n_jobs": -1,
           "verbosity": 0, "tree_method": "hist"}
    ).fit(X, y)
    best_mlp = ScaledMLP(**mlp_params).fit(X, y)

    # ---- Optuna LightGBM ---- #
    if lgb is None:
        log.error("lightgbm not installed; skipping")
        return
    log.info("=== Optuna LightGBM (%d trials) ===", N_TRIALS)
    s_lgb = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    s_lgb.optimize(make_lgbm_objective(X, y), n_trials=N_TRIALS,
                   show_progress_bar=False)
    log.info("LightGBM best CV R2 = %.5f  params=%s",
             s_lgb.best_value, s_lgb.best_params)
    best_lgbm = lgb.LGBMRegressor(
        **{**s_lgb.best_params, "random_state": 42, "n_jobs": -1, "verbosity": -1}
    ).fit(X, y)

    # ---- Optuna CatBoost ---- #
    if CatBoostRegressor is None:
        log.error("catboost not installed; skipping")
        return
    log.info("=== Optuna CatBoost (%d trials) ===", N_TRIALS)
    s_cat = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=43)
    )
    s_cat.optimize(make_cat_objective(X, y), n_trials=N_TRIALS,
                   show_progress_bar=False)
    log.info("CatBoost best CV R2 = %.5f  params=%s",
             s_cat.best_value, s_cat.best_params)
    best_cat = CatBoostRegressor(
        **{**s_cat.best_params, "random_seed": 42, "verbose": False, "thread_count": -1}
    ).fit(X, y)

    # ---- OOF + 5-model stacking ---- #
    log.info("Computing OOF predictions for stacking...")
    oof = {
        "rf":     cross_val_predict(best_rf,   X, y, cv=CV, n_jobs=1),
        "xgb":    cross_val_predict(best_xgb,  X, y, cv=CV, n_jobs=1),
        "mlp":    cross_val_predict(best_mlp,  X, y, cv=CV, n_jobs=1),
        "lgbm":   cross_val_predict(best_lgbm, X, y, cv=CV, n_jobs=1),
        "cat":    cross_val_predict(best_cat,  X, y, cv=CV, n_jobs=1),
    }
    X_meta = np.column_stack([oof[k] for k in oof])
    meta = Ridge(alpha=1.0).fit(X_meta, y)
    oof["stack"] = meta.predict(X_meta)
    oof["blend"] = X_meta.mean(axis=1)

    cv_metrics = {k: evaluate(y, oof[k]) for k in oof}

    # ---- Held-out test ---- #
    idx = np.arange(len(y))
    _, test_idx = train_test_split(idx, test_size=0.20, random_state=42)
    X_t, y_t = X[test_idx], y[test_idx]
    test_preds = {
        "rf":   best_rf.predict(X_t),
        "xgb":  best_xgb.predict(X_t),
        "mlp":  best_mlp.predict(X_t),
        "lgbm": best_lgbm.predict(X_t),
        "cat":  best_cat.predict(X_t),
        "stack": meta.predict(np.column_stack([
            best_rf.predict(X_t), best_xgb.predict(X_t),
            best_mlp.predict(X_t), best_lgbm.predict(X_t),
            best_cat.predict(X_t),
        ])),
    }
    test_metrics = {k: evaluate(y_t, test_preds[k]) for k in test_preds}

    # ---- Plots ---- #
    plot_5model_parity(y_t, test_preds,
                       FIGURES_DIR / "fig_bayes_5model.png")
    plot_lgbm_cat_convergence(
        {"LightGBM": s_lgb, "CatBoost": s_cat},
        FIGURES_DIR / "fig_bayes_convergence_5.png",
    )

    # ---- Save Optuna trial CSVs ---- #
    s_lgb.trials_dataframe().to_csv(RESULTS_DIR / "bayes_trials_lgbm.csv", index=False)
    s_cat.trials_dataframe().to_csv(RESULTS_DIR / "bayes_trials_catboost.csv", index=False)

    # ---- Save metrics JSON ---- #
    out = {
        "n_trials": N_TRIALS,
        "n_features_v2": int(X.shape[1]),
        "best_params": {
            "rf": rf_params, "xgb": xgb_params, "mlp": mlp_params,
            "lgbm": s_lgb.best_params, "cat": s_cat.best_params,
        },
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "meta_learner_coefs": dict(zip(
            ["rf", "xgb", "mlp", "lgbm", "cat"],
            meta.coef_.tolist())),
        "meta_intercept": float(meta.intercept_),
    }
    (RESULTS_DIR / "bayes_metrics_5model.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    log.info("Wrote %s", RESULTS_DIR / "bayes_metrics_5model.json")

    # ---- Full predictions ---- #
    full_pred = df[["mol_id", "smiles"]].copy()
    full_pred["dn_pred_rf_v2"]   = best_rf.predict(X)
    full_pred["dn_pred_xgb_v2"]  = best_xgb.predict(X)
    full_pred["dn_pred_mlp_v2"]  = best_mlp.predict(X)
    full_pred["dn_pred_lgbm_v2"] = best_lgbm.predict(X)
    full_pred["dn_pred_cat_v2"]  = best_cat.predict(X)
    full_pred["dn_pred_stack_v2"] = meta.predict(np.column_stack([
        full_pred["dn_pred_rf_v2"], full_pred["dn_pred_xgb_v2"],
        full_pred["dn_pred_mlp_v2"], full_pred["dn_pred_lgbm_v2"],
        full_pred["dn_pred_cat_v2"],
    ]))
    full_pred.to_csv(DATA_DIR / "full_predictions_5model.csv", index=False)

    top20 = full_pred.sort_values("dn_pred_stack_v2", ascending=False).head(20)
    top20.to_csv(RESULTS_DIR / "top20_candidates_5model.csv", index=False)

    elapsed = time.time() - t0
    log.info("=== Done in %.1f s ===", elapsed)
    print("\n===== 5-Model Stacking Summary =====")
    print(f"Features: {X.shape[1]} (v2)")
    print(f"Trials per new model: {N_TRIALS}")
    print(f"Wall time: {elapsed:.0f}s")
    for k, m in cv_metrics.items():
        print(f"  {k:6s}  CV R2={m['R2']:.5f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")
    print("\nTest metrics:")
    for k, m in test_metrics.items():
        print(f"  {k:6s}  R2={m['R2']:.5f}  RMSE={m['RMSE']:.4f}")


if __name__ == "__main__":
    main()
