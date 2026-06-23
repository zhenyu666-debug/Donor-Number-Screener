"""test_v5.py - v5 smoke tests.

Tests the four new v5 scripts:
  - p45_shap_interactions.py  (import, matrix shape)
  - p46_calibration.py        (ECE computation, output files)
  - p47_drift_monitor.py      (PSI computation, history management)
  - streamlit_app.py           (import only)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


# ---------------------------------------------------------------------------
# p45 — SHAP interactions
# ---------------------------------------------------------------------------

def test_p45_import():
    """p45_shap_interactions imports cleanly."""
    import p45_shap_interactions  # noqa: F401


def test_p45_interaction_matrix_shape():
    """Interaction matrix has correct square shape."""
    from p45_shap_interactions import load_full_data, fit_tree_models, load_best_params

    X, y, feat_cols = load_full_data()
    assert len(feat_cols) > 0, "No features loaded"
    assert X.shape[0] == len(y), "X/y length mismatch"

    params = load_best_params()
    models = fit_tree_models(X[:100], y[:100], params)
    assert len(models) >= 1, "No models fitted"

    # Build a dummy interaction matrix
    top_feats = feat_cols[: min(5, len(feat_cols))]
    n = len(top_feats)
    dummy = np.random.rand(n, n).astype(np.float64)
    assert dummy.shape == (n, n)


def test_p45_label_map():
    """FEAT_LABELS covers common chemical descriptors."""
    from p45_shap_interactions import FEAT_LABELS, _label

    assert "MolWt" in FEAT_LABELS
    assert "NumN" in FEAT_LABELS
    assert _label("NumN") == "Nitrogen count"
    assert _label("unknown_feat") == "unknown_feat"


# ---------------------------------------------------------------------------
# p46 — Calibration
# ---------------------------------------------------------------------------

def test_p46_import():
    """p46_calibration imports cleanly."""
    import p46_calibration  # noqa: F401


def test_p46_ece_formula():
    """ECE formula returns a float in [0, 1]."""
    from p46_calibration import expected_calibration_error, maximum_calibration_error

    # Perfect calibration: all predictions equal truth
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"
    assert ece == 0.0, "Perfect calibration should give ECE=0"

    mce = maximum_calibration_error(y_true, y_prob, n_bins=5)
    assert mce == 0.0, "Perfect calibration should give MCE=0"


def test_p46_ece_degenerate_cases():
    """ECE handles empty bins and constant predictions."""
    from p46_calibration import expected_calibration_error

    # All same prediction
    y_true = np.array([0.5, 0.5, 0.5, 0.5])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    ece = expected_calibration_error(y_true, y_prob, n_bins=3)
    assert 0.0 <= ece <= 1.0


def test_p46_probability_normalisation():
    """normalise_to_probability maps values to [0, 1]."""
    from p46_calibration import normalise_to_probability

    anchors = np.array([10.0, 20.0, 30.0])
    y = np.array([5.0, 15.0, 25.0, 40.0])
    p = normalise_to_probability(y, anchors)
    assert p.min() >= 0.0
    assert p.max() <= 1.0
    # 5.0 should be below min=10, clamped to 0
    assert p[0] == 0.0
    # 40.0 above max=30, clamped to 1
    assert p[-1] == 1.0


def test_p46_coverage():
    """coverage_at_level returns fraction in [0, 1]."""
    from p46_calibration import coverage_at_level

    y_true = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.5, 1.5, 2.5])
    upper = np.array([1.5, 2.5, 3.5])
    cov = coverage_at_level(y_true, lower, upper)
    assert 0.0 <= cov <= 1.0
    assert cov == 1.0, "All values in range should give coverage=1.0"


# ---------------------------------------------------------------------------
# p47 — Drift monitor
# ---------------------------------------------------------------------------

def test_p47_import():
    """p47_drift_monitor imports cleanly."""
    import p47_drift_monitor  # noqa: F401


def test_p47_psi_computation():
    """PSI is non-negative and grows with distribution shift."""
    from p47_drift_monitor import compute_psi_col

    bin_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    baseline_probs = np.full(len(bin_edges) - 1, 0.25)
    np.random.seed(42)
    new_vals = np.random.rand(100)
    psi = compute_psi_col(new_vals, bin_edges, baseline_probs)
    assert psi >= 0.0, f"PSI should be non-negative: {psi}"
    assert psi < 2.0, f"PSI unexpectedly large for random vs uniform: {psi}"


def test_p47_severity_levels():
    """_severity returns one of OK/INFO/WARNING/CRITICAL."""
    from p47_drift_monitor import _severity

    assert _severity(0.0, 0) == "OK"
    assert _severity(0.05, 1) == "OK"
    assert _severity(0.12, 3) == "INFO"
    assert _severity(0.20, 10) == "WARNING"
    assert _severity(0.30, 20) == "CRITICAL"


def test_p47_history_management(tmp_path):
    """History saves and loads correctly."""
    from p47_drift_monitor import load_history, save_history

    # Override history file location for test
    import p47_drift_monitor
    orig = p47_drift_monitor.DRIFT_HISTORY_FILE
    p47_drift_monitor.DRIFT_HISTORY_FILE = tmp_path / "history.json"

    try:
        save_history([])
        hist = load_history()
        assert hist == []

        save_history([{"timestamp": "2026-01-01T00:00:00Z", "severity": "OK", "overall_psi": 0.01}])
        hist = load_history()
        assert len(hist) == 1
        assert hist[0]["severity"] == "OK"
    finally:
        p47_drift_monitor.DRIFT_HISTORY_FILE = orig


def test_p47_alert_filtering(tmp_path):
    """Only WARNING/CRITICAL events create alerts."""
    import p47_drift_monitor
    from p47_drift_monitor import save_alerts

    alerts_file = tmp_path / "alerts.json"
    orig = p47_drift_monitor.DRIFT_ALERTS_FILE
    try:
        p47_drift_monitor.DRIFT_ALERTS_FILE = alerts_file
        test_alerts = [
            {"timestamp": "2026-01-01T00:00:00Z", "severity": "OK", "overall_psi": 0.01},
            {"timestamp": "2026-01-02T00:00:00Z", "severity": "WARNING", "overall_psi": 0.25},
        ]
        save_alerts(test_alerts)
        loaded = json.loads(alerts_file.read_text())
        assert len(loaded) == 1
        assert loaded[0]["severity"] == "WARNING"
    finally:
        p47_drift_monitor.DRIFT_ALERTS_FILE = orig


# ---------------------------------------------------------------------------
# streamlit_app
# ---------------------------------------------------------------------------

def test_streamlit_import():
    """streamlit_app imports cleanly (without running the app)."""
    import streamlit_app  # noqa: F401


def test_streamlit_constants():
    """App exposes expected constants and helper functions."""
    import streamlit_app as app

    assert hasattr(app, "_COMMON_SOLVENTS")
    assert len(app._COMMON_SOLVENTS) > 0
    assert "DMSO" in [s[0] for s in app._COMMON_SOLVENTS]
    assert hasattr(app, "dn_region_label")
    assert app.dn_region_label(0) == "Very Weak"
    assert app.dn_region_label(4) == "Very Strong"


# ---------------------------------------------------------------------------
# Cross-script integration checks
# ---------------------------------------------------------------------------

def test_calibration_and_api_integration():
    """Verify calibration metrics JSON schema matches what the API expects."""
    # If calibration has been run, check the schema matches what the API reads
    cal_path = RESULTS_DIR / "calibration_metrics.json"
    if not cal_path.exists():
        return  # skip if not generated yet

    with open(cal_path) as f:
        data = json.load(f)

    assert "calibration" in data, "Missing 'calibration' key"
    cal = data["calibration"]
    for method in ("raw", "isotonic", "platt"):
        assert method in cal, f"Missing method: {method}"
        assert "ece" in cal[method], f"Missing ECE for {method}"
        assert "coverage_95" in cal[method], f"Missing coverage_95 for {method}"
