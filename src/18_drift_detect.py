"""Step 18 (v3): Population Stability Index (PSI) drift detector.

Computes per-feature PSI between a baseline distribution (the v2
training set) and a new batch.  PSI is the standard model-monitoring
metric:

    PSI_j = sum_i (p_i_new - p_i_base) * ln(p_i_new / p_i_base)

For each feature the new and baseline samples are bucketed into
10 quantile bins of the baseline.  A feature with PSI > 0.2 is
considered meaningfully drifted.

Outputs (under results/):
  drift_baseline.json   - the baseline per-feature bin edges (saved once)
  drift_report.json     - the per-feature + overall PSI for the new batch

Usage (one-time baseline):
  python src/18_drift_detect.py --mode baseline

Usage (compare a new batch):
  python src/18_drift_detect.py --mode batch --input batch.csv
  where batch.csv has the same 996 v2 descriptor columns.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESULTS_DIR, get_logger  # noqa: E402

warnings.filterwarnings("ignore")
log = get_logger("drift")

NON_FEATURE_COLS = {
    "mol_id", "smiles", "smiles_x", "smiles_y",
    "dn_rf", "dn_empirical", "dn_final", "confidence", "is_anchor",
}

N_BINS = 10
PSI_THRESHOLDS = {"stable": 0.10, "minor": 0.20}


def load_v2_Xy() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    desc = pd.read_csv(DATA_DIR / "descriptors_v2.csv")
    feat_cols = [c for c in desc.columns
                 if c not in NON_FEATURE_COLS and desc[c].dtype != "O"]
    X = desc[feat_cols].values.astype(np.float64)
    return desc, X, feat_cols


def compute_baseline(X: np.ndarray, feat_cols: list[str]) -> dict:
    """Compute per-feature quantile bin edges and the baseline histogram."""
    edges: dict = {}
    base_hist: dict = {}
    for j, c in enumerate(feat_cols):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if len(col) < N_BINS:
            edges[c] = []
            base_hist[c] = []
            continue
        quantiles = np.linspace(0, 1, N_BINS + 1)
        # Use unique values to avoid duplicate bin edges
        e = np.quantile(col, quantiles)
        # Ensure strictly increasing edges (np.quantile can return ties on
        # near-constant features; nudge by 1e-9)
        for k in range(1, len(e)):
            if e[k] <= e[k - 1]:
                e[k] = e[k - 1] + 1e-9
        edges[c] = e.tolist()
        hist, _ = np.histogram(col, bins=e)
        total = hist.sum()
        base_hist[c] = (hist / total).tolist() if total > 0 else [0.0] * N_BINS
    return {"edges": edges, "base_hist": base_hist,
            "n_features": len(feat_cols), "n_samples": int(X.shape[0])}


def psi_for_feature(baseline: np.ndarray, new: np.ndarray, edges: np.ndarray) -> float:
    """Standard PSI for a single feature."""
    if len(edges) < 2:
        return 0.0
    b, _ = np.histogram(baseline, bins=edges)
    n, _ = np.histogram(new,     bins=edges)
    bsum, nsum = b.sum(), n.sum()
    if bsum == 0 or nsum == 0:
        return 0.0
    p_b = b / bsum
    p_n = n / nsum
    # Replace 0s with a small number to avoid log(0)
    eps = 1e-6
    p_b = np.clip(p_b, eps, None)
    p_n = np.clip(p_n, eps, None)
    return float(((p_n - p_b) * np.log(p_n / p_b)).sum())


def evaluate_batch(X_new: np.ndarray, baseline: dict, feat_cols: list[str]) -> dict:
    per_feature: list[dict] = []
    X_base = pd.read_csv(DATA_DIR / "descriptors_v2.csv")[feat_cols].values.astype(np.float64)
    overall = 0.0
    n_drifted = 0
    for j, c in enumerate(feat_cols):
        edges = np.array(baseline["edges"].get(c, []))
        if len(edges) < 2:
            continue
        base_col = X_base[:, j]
        base_col = base_col[~np.isnan(base_col)]
        new_col = X_new[:, j]
        new_col = new_col[~np.isnan(new_col)]
        psi = psi_for_feature(base_col, new_col, edges)
        if psi > PSI_THRESHOLDS["minor"]:
            verdict = "drifted"
            n_drifted += 1
        elif psi > PSI_THRESHOLDS["stable"]:
            verdict = "minor"
        else:
            verdict = "stable"
        per_feature.append({
            "feature": c,
            "psi":     round(psi, 4),
            "verdict": verdict,
        })
        overall += psi
    per_feature.sort(key=lambda r: -r["psi"])
    return {
        "overall_psi":       round(overall, 4),
        "mean_psi":          round(overall / max(1, len(per_feature)), 4),
        "n_features":        len(per_feature),
        "n_drifted":         n_drifted,
        "thresholds":        PSI_THRESHOLDS,
        "top10_drifted":     per_feature[:10],
        "per_feature_first": per_feature[:25],
        "per_feature_last":  per_feature[-5:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "batch"], default="baseline")
    ap.add_argument("--input", help="path to a new-batch descriptors CSV")
    ap.add_argument("--out-baseline", default=str(RESULTS_DIR / "drift_baseline.json"))
    ap.add_argument("--out-report",   default=str(RESULTS_DIR / "drift_report.json"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    desc, X, feat_cols = load_v2_Xy()
    log.info("Loaded v2 X=%s  feats=%d", X.shape, len(feat_cols))

    if args.mode == "baseline":
        baseline = compute_baseline(X, feat_cols)
        Path(args.out_baseline).write_text(
            json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Wrote %s  (%.1f s)", args.out_baseline, time.perf_counter() - t0)
        print(f"\n  Baseline: {baseline['n_features']} features, "
              f"{baseline['n_samples']} samples, {N_BINS} quantile bins each")
        return

    # mode == batch
    if not args.input:
        log.error("--input required for batch mode")
        sys.exit(1)
    if not Path(args.out_baseline).exists():
        log.error("Run baseline first: python src/18_drift_detect.py --mode baseline")
        sys.exit(1)
    baseline = json.loads(Path(args.out_baseline).read_text(encoding="utf-8"))
    new = pd.read_csv(args.input)
    missing = [c for c in feat_cols if c not in new.columns]
    if missing:
        log.error("New batch missing %d features (e.g. %s); aborting",
                  len(missing), missing[:5])
        sys.exit(1)
    X_new = new[feat_cols].values.astype(np.float64)
    log.info("Loaded new batch X=%s", X_new.shape)

    report = evaluate_batch(X_new, baseline, feat_cols)
    report["input_path"]      = args.input
    report["n_new_samples"]   = int(X_new.shape[0])
    report["wall_time_s"]     = time.perf_counter() - t0
    Path(args.out_report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote %s", args.out_report)
    print(f"\n  Overall PSI = {report['overall_psi']:.4f}  "
          f"(mean {report['mean_psi']:.4f})")
    print(f"  {report['n_drifted']} features drifted (PSI > 0.20)")
    print("\n  Top 10 drifted features:")
    for r in report["top10_drifted"]:
        print(f"    {r['feature']:>32s}  PSI = {r['psi']:6.4f}  ({r['verdict']})")


if __name__ == "__main__":
    main()
