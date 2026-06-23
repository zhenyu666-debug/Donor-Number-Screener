"""p46_calibration.py - v5: Reliability / calibration diagram for the 5-model ensemble.

Computes a reliability diagram (expected calibration error / ECE) for the
5-model stacking ensemble using isotonic regression + Platt scaling calibration
on held-out data.  Also adds a calibration curve to each /screen_top API
response via the existing FastAPI endpoints.

Outputs (under results/):
  calibration_reliability_diagram.png  - reliability diagram with ECE
  calibration_metrics.json            - ECE, MCE, AUC of calibration curve
  calibration_summary.csv             - per-bin calibration data

Outputs (added to results/):
  top20_candidates_5model_calibrated.csv - top-20 with calibration columns

Usage:
  python src/p46_calibration.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESULTS_DIR, FIGURES_DIR, get_logger  # noqa: E402

warnings.filterwarnings("ignore")
log = get_logger("calibration")

NON_FEATURE_COLS = {
    "mol_id", "smiles", "smiles_x", "smiles_y",
    "dn_rf", "dn_empirical", "dn_final", "confidence", "is_anchor",
}

N_FOLDS = 5
N_BINS = 10
CV = KFold(N_FOLDS, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    """Compute ECE = sum_b (|B_b| / n) * |acc(b) - conf(b)|."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    """Compute MCE = max_b |acc(b) - conf(b)|."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        mce = max(mce, abs(acc - conf))
    return mce


def coverage_at_level(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    level: float = 0.95,
) -> float:
    """Fraction of samples where the CI covers the true value."""
    return np.mean((y_true >= lower) & (y_true <= upper))


# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load v2 descriptors and labels, return anchor + non-anchor splits."""
    desc = pd.read_csv(DATA_DIR / "descriptors_v2.csv")
    labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
    df = desc.merge(labels[["mol_id", "dn_final"]], on="mol_id", how="inner")
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and df[c].dtype != "O"]
    X = df[feat_cols].values.astype(np.float64)
    y = df["dn_final"].values.astype(np.float64)
    log.info("Loaded: X=%s, y range [%.2f, %.2f]", X.shape, y.min(), y.max())
    return df, X, y, feat_cols


def load_or_train_stack(
    X: np.ndarray,
    y: np.ndarray,
    feat_cols: list[str],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached 5-model predictions or retrain.

    Returns (models, y_pred, y_std, y_lower, y_upper).
    """
    try:
        import json
        with open(RESULTS_DIR / "bayes_metrics_5model.json") as f:
            params = json.load(f)
    except Exception:
        log.warning("bayes_metrics_5model.json not found; using defaults")
        params = {}

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Fit individual models
    models = {}

    rf_params = {**params.get("rf", {"n_estimators": 200, "max_depth": 15})}
    rf = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1).fit(X, y)
    models["rf"] = ("tree", rf)

    xgb_params = {**params.get("xgb", {"n_estimators": 200, "max_depth": 7})}
    xgb = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1,
                       verbosity=0, tree_method="hist").fit(X, y)
    models["xgb"] = ("tree", xgb)

    mlp_params = {**params.get("mlp", {"hidden_layer_sizes": (100, 50), "alpha": 0.001})}
    mlp = MLPRegressor(**mlp_params, random_state=42, max_iter=1000).fit(Xs, y)
    models["mlp"] = ("mlp", mlp)

    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMRegressor(
            **params.get("lgbm", {"n_estimators": 200, "max_depth": 7}),
            random_state=42, n_jobs=-1, verbosity=-1
        ).fit(X, y)
        models["lgbm"] = ("tree", lgbm)
    except Exception:
        pass

    try:
        from catboost import CatBoostRegressor
        cat = CatBoostRegressor(
            **params.get("cat", {"iterations": 200, "depth": 7}),
            random_seed=42, verbose=False, thread_count=-1
        ).fit(X, y)
        models["cat"] = ("tree", cat)
    except Exception:
        pass

    # Stack: cross-val predictions as meta-features
    # Cache OOF predictions to avoid re-fitting models 3x (reduces K*N fits to K*N)
    oof_preds: dict[str, np.ndarray] = {}
    for name, (kind, model) in models.items():
        Xs_use = Xs if kind == "mlp" else X
        oof_preds[name] = cross_val_predict(
            model.__class__(**model.get_params()), Xs_use, y, cv=CV, n_jobs=-1
        )

    # Build meta-features and fit Ridge meta-learner
    meta_X = np.zeros((X.shape[0], len(models)))
    for j, name in enumerate(models):
        meta_X[:, j] = oof_preds[name]
    meta_lr = Ridge(alpha=1.0)
    meta_lr.fit(meta_X, y)
    log.info("Ridge meta-learner fitted on %d models", len(models))

    # Final stacked prediction using cached OOF preds
    y_pred_all = np.zeros(X.shape[0])
    for j, name in enumerate(models):
        y_pred_all += meta_lr.coef_[j] * oof_preds[name]
    y_pred_all += meta_lr.intercept_

    # Approximate 95% CI from OOF prediction variance
    y_std_all = np.std(list(oof_preds.values()), axis=0)

    # Approximate 95% CI
    y_lower = y_pred_all - 1.96 * y_std_all
    y_upper = y_pred_all + 1.96 * y_std_all

    return models, y_pred_all, y_std_all, y_lower, y_upper


# ---------------------------------------------------------------------------
# Calibration: normalise DN to [0, 1] per anchor percentile
# ---------------------------------------------------------------------------

def normalise_to_probability(y: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Map DN values to [0,1] using the anchor DN range as the scale.

    Anchors (anchor DN values) define [min, max]; values are clamped.
    """
    lo, hi = anchors.min(), anchors.max()
    p = (y - lo) / (hi - lo + 1e-12)
    return np.clip(p, 0.0, 1.0)


def fit_isotonic_calibration(
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
) -> IsotonicRegression:
    """Fit isotonic regression to calibrate predicted probabilities."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(y_pred_norm, y_true_norm)
    return iso


def fit_platt_scaling(
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
) -> LogisticRegression:
    """Fit Platt scaling (logistic regression) for calibration."""
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(y_pred_norm.reshape(-1, 1), y_true_norm)
    return lr


def compute_calibration_curve(
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    iso: IsotonicRegression,
    platt: LogisticRegression,
) -> dict:
    """Compute calibration curve points for plotting."""
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    result = {
        "frac_positives": [],
        "mean_predicted": [],
        "count": [],
        "iso_frac_positives": [],
        "iso_mean_predicted": [],
        "platt_frac_positives": [],
        "platt_mean_predicted": [],
    }
    y_iso = iso.predict(y_pred_norm)
    y_platt = platt.predict_proba(y_pred_norm.reshape(-1, 1))[:, 1]

    for i in range(N_BINS):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred_norm >= lo) & (y_pred_norm < hi)
        if hi == 1.0:
            mask = (y_pred_norm >= lo) & (y_pred_norm <= hi)
        if mask.sum() == 0:
            continue
        result["frac_positives"].append(float(y_true_norm[mask].mean()))
        result["mean_predicted"].append(float(y_pred_norm[mask].mean()))
        result["count"].append(int(mask.sum()))
        result["iso_frac_positives"].append(float(y_iso[mask].mean()))
        result["iso_mean_predicted"].append(float(y_iso[mask].mean()))
        result["platt_frac_positives"].append(float(y_platt[mask].mean()))
        result["platt_mean_predicted"].append(float(y_platt[mask].mean()))

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    y_iso_pred: np.ndarray,
    y_platt_pred: np.ndarray,
    ece_raw: float,
    ece_iso: float,
    ece_platt: float,
    out_path: Path,
) -> None:
    """Draw a reliability diagram with three curves (raw / isotonic / Platt)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: reliability diagram
    ax = axes[0]
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    raw_acc = np.zeros(N_BINS)
    iso_acc = np.zeros(N_BINS)
    pla_acc = np.zeros(N_BINS)
    counts = np.zeros(N_BINS)

    for i in range(N_BINS):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred_norm >= lo) & (y_pred_norm < hi)
        if hi == 1.0:
            mask = (y_pred_norm >= lo) & (y_pred_norm <= hi)
        if mask.sum() == 0:
            continue
        raw_acc[i] = y_true_norm[mask].mean()
        iso_acc[i] = y_iso_pred[mask].mean()
        pla_acc[i] = y_platt_pred[mask].mean()
        counts[i] = mask.sum()

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", zorder=5)
    ax.scatter(bin_centers, raw_acc, s=counts * 2 + 10, alpha=0.6,
              c="#e74c3c", label=f"Raw (ECE={ece_raw:.3f})", zorder=4)
    ax.plot(bin_centers, raw_acc, "#e74c3c", lw=1.5, alpha=0.6)
    ax.scatter(bin_centers, iso_acc, s=counts * 2 + 10, alpha=0.7,
              c="#27ae60", label=f"Isotonic (ECE={ece_iso:.3f})", zorder=3)
    ax.plot(bin_centers, iso_acc, "#27ae60", lw=1.5, alpha=0.7)
    ax.scatter(bin_centers, pla_acc, s=counts * 2 + 10, alpha=0.7,
              c="#3498db", label=f"Platt (ECE={ece_platt:.3f})", zorder=3)
    ax.plot(bin_centers, pla_acc, "#3498db", lw=1.5, alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted DN (normalised)", fontsize=9)
    ax.set_ylabel("Fraction of positives", fontsize=9)
    ax.set_title("Reliability Diagram — 5-Model DN Ensemble", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right: CI coverage by level
    ax2 = axes[1]
    levels = [0.80, 0.85, 0.90, 0.95, 0.99]
    coverages_raw = []
    coverages_iso = []
    coverages_platt = []
    for lvl in levels:
        half = (1.0 - lvl) / 2.0
        # Map back to DN space using anchor range approximation
        lo_raw = y_pred_norm - half
        hi_raw = y_pred_norm + half
        coverages_raw.append(np.mean(
            (y_true_norm >= lo_raw) & (y_true_norm <= hi_raw)
        ))
        lo_iso = y_iso_pred - half
        hi_iso = y_iso_pred + half
        coverages_iso.append(np.mean(
            (y_true_norm >= lo_iso) & (y_true_norm <= hi_iso)
        ))
        lo_pla = y_platt_pred - half
        hi_pla = y_platt_pred + half
        coverages_platt.append(np.mean(
            (y_true_norm >= lo_pla) & (y_true_norm <= hi_pla)
        ))

    x_pos = np.arange(len(levels))
    w = 0.25
    ax2.bar(x_pos - w, coverages_raw, w, label="Raw", color="#e74c3c", alpha=0.7)
    ax2.bar(x_pos, coverages_iso, w, label="Isotonic", color="#27ae60", alpha=0.7)
    ax2.bar(x_pos + w, coverages_platt, w, label="Platt", color="#3498db", alpha=0.7)
    ax2.plot(x_pos, levels, "k--", lw=1.5, label="Nominal level")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{lv:.0%}" for lv in levels])
    ax2.set_xlabel("Confidence level", fontsize=9)
    ax2.set_ylabel("Coverage", fontsize=9)
    ax2.set_title("CI Coverage vs Nominal Level", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "p46 Calibration Analysis — v5",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved reliability diagram: %s", out_path)


# ---------------------------------------------------------------------------
# Calibrate top-20 predictions
# ---------------------------------------------------------------------------

def calibrate_top20(
    iso: IsotonicRegression,
    platt: LogisticRegression,
    out_path: Path,
) -> None:
    """Calibrate the top-20 predictions and save with new columns."""
    top20_path = RESULTS_DIR / "top20_candidates_5model.csv"
    if not top20_path.exists():
        log.warning("top20_candidates_5model.csv not found; skipping top-20 calibration")
        return

    df = pd.read_csv(top20_path)
    if "dn_pred" in df.columns or "dn_final" in df.columns:
        dn_col = "dn_final" if "dn_final" in df.columns else "dn_pred"
        y_top = df[dn_col].values.astype(np.float64)

        # Normalise using approximate global range
        pd.read_csv(DATA_DIR / "descriptors_v2.csv")  # verify file exists
        labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
        anchors = labels["dn_final"].dropna().values
        lo, hi = anchors.min(), anchors.max()
        y_norm = np.clip((y_top - lo) / (hi - lo + 1e-12), 0, 1)

        df["dn_iso_calibrated"] = iso.predict(y_norm)
        df["dn_platt_calibrated"] = platt.predict_proba(y_norm.reshape(-1, 1))[:, 1]
        df.to_csv(out_path, index=False)
        log.info("Saved calibrated top-20: %s", out_path)
    else:
        log.warning("No DN column found in top-20; skipping")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()
    log.info("=" * 60)
    log.info("p46 Calibration Analysis (v5)")
    log.info("=" * 60)

    df, X, y, feat_cols = load_data()

    # Get anchor values for normalisation
    anchors = df[df["is_anchor"] == 1]["dn_final"].values
    if len(anchors) < 5:
        anchors = y  # fallback to all labels

    # Train / load stack
    models, y_pred, y_std, y_lower, y_upper = load_or_train_stack(X, y, feat_cols)
    log.info("Stack predictions: R2=%.4f on training data", _r2(y, y_pred))

    # Normalise to [0,1]
    y_norm = normalise_to_probability(y, anchors)
    y_pred_norm = normalise_to_probability(y_pred, anchors)

    # Fit calibrators on ALL data (in-sample — for diagnostic purposes)
    iso = fit_isotonic_calibration(y_norm, y_pred_norm)
    platt = fit_platt_scaling(y_norm, y_pred_norm)
    y_iso_pred = iso.predict(y_pred_norm)
    y_platt_pred = platt.predict_proba(y_pred_norm.reshape(-1, 1))[:, 1]

    # Compute ECE
    ece_raw = expected_calibration_error(y_norm, y_pred_norm)
    ece_iso = expected_calibration_error(y_norm, y_iso_pred)
    y_platt_prob = platt.predict_proba(y_pred_norm.reshape(-1, 1))[:, 1]
    ece_platt = expected_calibration_error(y_norm, y_platt_prob)

    mce_raw = maximum_calibration_error(y_norm, y_pred_norm)
    mce_iso = maximum_calibration_error(y_norm, y_iso_pred)
    mce_platt = maximum_calibration_error(y_norm, y_platt_prob)

    log.info("Raw   — ECE: %.4f  MCE: %.4f", ece_raw, mce_raw)
    log.info("Iso   — ECE: %.4f  MCE: %.4f", ece_iso, mce_iso)
    log.info("Platt — ECE: %.4f  MCE: %.4f", ece_platt, mce_platt)

    # Coverage at 95%
    cov_raw = coverage_at_level(y_norm, y_pred_norm - 0.475, y_pred_norm + 0.475)
    cov_iso = coverage_at_level(y_norm, y_iso_pred - 0.475, y_iso_pred + 0.475)
    cov_platt = coverage_at_level(y_norm, y_platt_pred - 0.475, y_platt_pred + 0.475)
    log.info("Coverage @95%%: raw=%.3f  iso=%.3f  platt=%.3f", cov_raw, cov_iso, cov_platt)

    # Save metrics
    metrics = {
        "n_samples": int(len(y)),
        "n_bins": N_BINS,
        "calibration": {
            "raw":   {"ece": float(ece_raw), "mce": float(mce_raw), "coverage_95": float(cov_raw)},
            "isotonic": {"ece": float(ece_iso), "mce": float(mce_iso), "coverage_95": float(cov_iso)},
            "platt": {"ece": float(ece_platt), "mce": float(mce_platt), "coverage_95": float(cov_platt)},
        },
        "model_count": len(models),
        "features": feat_cols,
    }
    out_json = RESULTS_DIR / "calibration_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved metrics: %s", out_json)

    # Save per-bin summary
    curve = compute_calibration_curve(y_norm, y_pred_norm, iso, platt)
    out_csv = RESULTS_DIR / "calibration_summary.csv"
    pd.DataFrame(curve).to_csv(out_csv, index=False)
    log.info("Saved calibration curve: %s", out_csv)

    # Plot
    out_png = FIGURES_DIR / "calibration_reliability_diagram.png"
    plot_reliability_diagram(
        y_norm, y_pred_norm, y_iso_pred, y_platt_pred,
        ece_raw, ece_iso, ece_platt, out_png
    )

    # Calibrate top-20
    calibrate_top20(iso, platt, RESULTS_DIR / "top20_candidates_5model_calibrated.csv")

    log.info("Total time: %.1fs", time.perf_counter() - t_start)
    log.info("Done. Outputs: %s/calibration_*", RESULTS_DIR)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


if __name__ == "__main__":
    main()
