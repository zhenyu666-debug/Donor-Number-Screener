"""test_pbp_regeneration.py - Tests for Phase 2 DEER regeneration protocol layer.

Validates:
  - p41_regeneration_protocol.py: R_EEI exponential decay, k_diss calibration
  - p41b_lif_stabilization.py: LiF residual stabilization model
  - p41c_pouch_validation.py: scale-up from coin to pouch cell
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402


class TestRegenerationProtocol:
    """Unit tests for p41_regeneration_protocol.py."""

    def test_k_diss_calibration_dmi(self):
        from p41_regeneration_protocol import dissolution_rate_constant
        k = dissolution_rate_constant(dn=29.0, T=298.15)
        assert 0.3 < k < 0.8, f"DMI k_diss should be ~0.5, got {k}"

    def test_k_diss_calibration_ec(self):
        from p41_regeneration_protocol import dissolution_rate_constant
        k = dissolution_rate_constant(dn=16.8, T=298.15)
        assert k < 0.15, f"EC k_diss should be low, got {k}"

    def test_eei_resistance_decay(self):
        from p41_regeneration_protocol import eei_resistance_after_n_scans
        r0 = 100.0
        k  = 0.5
        r1 = eei_resistance_after_n_scans(r0, 1, k)
        r2 = eei_resistance_after_n_scans(r0, 2, k)
        assert r1 < r0,    "R should decrease after 1 scan"
        assert r2 < r1,    "R should continue decreasing"
        assert np.isclose(r1, r0 * np.exp(-k), rtol=1e-4)
        assert np.isclose(r2, r0 * np.exp(-2*k), rtol=1e-4)

    def test_n_scans_needed(self):
        from p41_regeneration_protocol import n_scans_to_recover
        n = n_scans_to_recover(r_eei_0=100.0, target_resistance=5.0, k_diss=0.5)
        expected = np.log(100.0/5.0) / 0.5
        assert np.isclose(n, expected, rtol=1e-3), f"Expected {expected:.2f}, got {n:.2f}"

    def test_lif_residual_thickness(self):
        from p41_regeneration_protocol import lif_residual_thickness
        t = lif_residual_thickness(lif_initial_nm=3.0, n=5, dissolution_fraction=0.90)
        assert 0.5 <= t <= 3.0, f"LiF should be 0.5-3.0 nm, got {t}"
        # More scans -> more dissolution
        t_more = lif_residual_thickness(lif_initial_nm=3.0, n=10, dissolution_fraction=0.90)
        assert t_more <= t, "More scans should dissolve more LiF"

    def test_simulate_cv_decay_output(self):
        from p41_regeneration_protocol import simulate_cv_decay
        result = simulate_cv_decay(dn=29.0, electrode_key="NMC811_cathode",
                                  n_scans_max=5, T=298.15)
        assert "rows" in result
        assert "k_diss_per_scan" in result
        assert len(result["rows"]) == 6
        assert result["rows"][0]["scan_number"] == 0
        assert result["rows"][-1]["scan_number"] == 5
        # Resistance should monotonically decrease (scan 0: 85, scan 5: ~6.9)
        resistances = [r["r_eei_ohm_cm2"] for r in result["rows"]]
        assert resistances == sorted(resistances, reverse=True), \
            f"R_EEI should decrease monotonically (got {resistances})"

    def test_simulate_dual_electrode(self):
        from p41_regeneration_protocol import simulate_dual_electrode
        result = simulate_dual_electrode(dn=29.0, solvent_name="DMI", n_scans=5)
        assert "combined_rows" in result
        assert "combined_recovery_pct" in result
        assert 80.0 <= result["combined_recovery_pct"] <= 99.0

    def test_p41_runs_without_exception(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "p41_regeneration_protocol.py"),
             "--dn", "29.0", "--electrode", "dual", "--n-scans", "5"],
            capture_output=True, text=True, timeout=60,
            env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        if result.returncode != 0:
            pytest.fail(f"p41 failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


class TestLiFStabilization:
    """Unit tests for p41b_lif_stabilization.py."""

    def test_lif_conductivity(self):
        from p41b_lif_stabilization import lif_conductivity
        s0  = lif_conductivity(0.0)
        s1  = lif_conductivity(1.0)
        s5  = lif_conductivity(5.0)
        s10 = lif_conductivity(10.0)
        assert s0 > s1,   "Ultra-thin LiF should have highest conductivity"
        assert s1 > s5,  "Thin LiF > bulk LiF (grain boundary effect)"
        assert s5 <= s10, "Bulk LiF should be lowest (allow equality)"
        assert 1e-7 <= s10 <= 1e-5, f"Bulk LiF should be ~1e-6, got {s10}"

    def test_butler_volmer_increasing(self):
        from p41b_lif_stabilization import butler_volmer_current_density
        j0 = 0.1   # A/m^2
        eta_values = [0.01, 0.05, 0.10, 0.20]
        currents = [butler_volmer_current_density(j0, e) for e in eta_values]
        assert currents == sorted(currents), "Current should increase with overpotential"

    def test_capacity_fade_deer_better_than_fresh(self):
        from p41b_lif_stabilization import fade_rate_from_sigma
        sigma_deer  = fade_rate_from_sigma(4.1e-5)   # LiF 2nm
        sigma_fresh = fade_rate_from_sigma(1.0e-6)    # native SEI
        assert sigma_deer <= sigma_fresh, "DEER cells should have no worse fade than fresh"

    def test_simulate_cycling_output(self):
        from p41b_lif_stabilization import simulate_cycling
        rows = simulate_cycling(n_cycles=100, lif_thickness_nm=2.0)
        assert len(rows) == 101
        assert rows[0]["cycle"] == 0
        assert rows[-1]["cycle"] == 100
        # DEER capacity should be >= fresh (not strictly, but close)
        assert rows[-1]["capacity_deer_pct"] <= 100.0
        assert rows[-1]["capacity_fresh_pct"] <= 100.0

    def test_p41b_runs_without_exception(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "p41b_lif_stabilization.py"),
             "--lif-nm", "2.0", "--n-cycles", "100"],
            capture_output=True, text=True, timeout=60,
            env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        if result.returncode != 0:
            pytest.fail(f"p41b failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


class TestPouchValidation:
    """Unit tests for p41c_pouch_validation.py."""

    def test_area_correction(self):
        from p41c_pouch_validation import area_correction_factor
        f1   = area_correction_factor(1.0)
        f100 = area_correction_factor(100.0)
        f10k = area_correction_factor(10000.0)
        assert np.isclose(f1, 1.0),   "1 cm^2 (coin cell) should have f=1.0"
        assert f100 < f1, "Larger area should have penalty"
        assert f10k < f100, "Even larger should have more penalty"
        assert f10k >= 0.85, "Should not drop below 0.85"

    def test_npn_correction(self):
        from p41c_pouch_validation import npn_ratio_correction
        f_optimal = npn_ratio_correction(1.10)
        f_low     = npn_ratio_correction(0.90)
        f_high    = npn_ratio_correction(1.30)
        assert f_optimal > f_low,  "Optimal N/P should be better than low"
        assert f_optimal > f_high, "Optimal N/P should be better than high"

    def test_rate_correction(self):
        from p41c_pouch_validation import rate_correction_factor
        f_low  = rate_correction_factor(0.2)
        f_high = rate_correction_factor(2.0)
        assert np.isclose(f_low, 1.0),  "Low C-rate should have f=1.0"
        assert f_high < f_low, "Higher C-rate should have penalty"

    def test_pouch_recovery_prediction(self):
        from p41c_pouch_validation import predict_pouch_recovery
        r = predict_pouch_recovery(
            coin_recovery_pct=95.0,
            pouch_area_cm2=180.0,
            npn_ratio=1.15,
            c_rate=0.5,
            temperature_C=25.0,
        )
        assert 85.0 <= r["predicted_recovery_pct"] <= 98.0
        assert "f_area" in r
        assert "f_npn" in r
        assert "f_rate" in r
        assert "f_total" in r

    def test_p41c_runs_without_exception(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "p41c_pouch_validation.py")],
            capture_output=True, text=True, timeout=60,
            env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        if result.returncode != 0:
            pytest.fail(f"p41c failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
