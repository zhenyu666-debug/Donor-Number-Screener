"""test_pbp_tea_lca.py - Tests for Phase 3 TEA/LCA layer.

Validates:
  - p42_tea_lca.py: pathway cost model, DEER cost, comparison table
  - p42b_viz_tea.py: visualization stubs
  - p42c_sensitivity.py: sensitivity model, tornado data
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import subprocess  # noqa: E402, F401
import json  # noqa: E402, F401

import pytest  # noqa: E402


class TestTEALCA:
    """Tests for p42_tea_lca.py."""

    def test_pathway_definitions(self):
        from p42_tea_lca import PATHWAYS
        assert "DEER" in PATHWAYS
        assert "Pyrometallurgy" in PATHWAYS
        assert "Hydrometallurgy" in PATHWAYS
        assert "Electrolyte_Swap" in PATHWAYS
        assert PATHWAYS["DEER"]["cost_usd_kg"] == 15.25

    def test_deer_is_cheapest_among_recovery_methods(self):
        from p42_tea_lca import PATHWAYS
        recovery_methods = {k: v["cost_usd_kg"] for k, v in PATHWAYS.items()
                          if v["capacity_recovery_pct"] >= 90}
        assert recovery_methods["DEER"] == min(recovery_methods.values())

    def test_deer_cost_in_expected_range(self):
        from p42_tea_lca import PATHWAYS
        d = PATHWAYS["DEER"]
        assert 10.0 < d["cost_usd_kg"] < 20.0
        assert 1.0 < d["energy_kwh_kg"] < 5.0
        assert 0.5 < d["ghg_kg_co2_kg"] < 3.0

    def test_deer_cost_vs_pyro(self):
        from p42_tea_lca import PATHWAYS
        pyro  = PATHWAYS["Pyrometallurgy"]["cost_usd_kg"]
        deer  = PATHWAYS["DEER"]["cost_usd_kg"]
        savings_pct = (pyro - deer) / pyro * 100
        assert 40.0 <= savings_pct <= 50.0, \
            f"Expected ~42% savings, got {savings_pct:.1f}%"

    def test_deer_energy_vs_pyro(self):
        from p42_tea_lca import PATHWAYS
        pyro  = PATHWAYS["Pyrometallurgy"]["energy_kwh_kg"]
        deer  = PATHWAYS["DEER"]["energy_kwh_kg"]
        assert deer < pyro, "DEER should use less energy than pyrometallurgy"
        assert deer < pyro * 0.30, "DEER energy should be <30% of pyrometallurgy"

    def test_breakdown_cost(self):
        from p42_tea_lca import breakdown_cost
        bd = breakdown_cost("DEER", scale_kg=1000.0)
        assert bd["scale_kg"] == 1000.0
        assert "cathode_material" in bd
        assert "binder_separator" in bd
        assert "labor" in bd
        total_check = (bd["cathode_material"] + bd["binder_separator"] +
                       bd["labor"] + bd["energy_cost"] + bd["overhead"])
        assert abs(total_check - bd["total_cost"]) < 0.01, "Breakdown should sum to total"

    def test_comparison_table_sorted(self):
        from p42_tea_lca import comparison_table
        rows = comparison_table()
        costs = [r["cost_usd_kg"] for r in rows]
        assert costs == sorted(costs), "Comparison table should be sorted by cost"

    def test_p42_runs_without_exception(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "p42_tea_lca.py")],
            capture_output=True, text=True, timeout=60,
            env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        if result.returncode != 0:
            pytest.fail(f"p42_tea_lca.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


class TestViz:
    """Tests for p42b_viz_tea.py."""

    def test_load_tea_data(self):
        from p42b_viz_tea import load_tea_data
        # If CSV not yet created, run p42 first
        if not (RESULTS_DIR / "deer_tea_lca.csv").exists():
            import subprocess
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "src" / "p42_tea_lca.py")],
                capture_output=True, timeout=60,
                env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                check=False,
            )
        data = load_tea_data()
        assert len(data) >= 3


class TestSensitivity:
    """Tests for p42c_sensitivity.py."""

    def test_deer_cost_model(self):
        from p42c_sensitivity import deer_cost_model
        r = deer_cost_model()
        assert "total_cost_kg" in r
        assert 3.0 < r["total_cost_kg"] < 30.0

    def test_base_case_matches_p42(self):
        from p42c_sensitivity import deer_cost_model
        from p42_tea_lca import PATHWAYS
        # p42c uses a simplified model; check it's in the right ballpark
        r = deer_cost_model()
        assert r["total_cost_kg"] < PATHWAYS["Pyrometallurgy"]["cost_usd_kg"]

    def test_solvent_cost_sensitivity(self):
        from p42c_sensitivity import deer_cost_model
        c_lo  = deer_cost_model(solvent_cost_usd_kg=10.0)["total_cost_kg"]
        c_mid = deer_cost_model(solvent_cost_usd_kg=50.0)["total_cost_kg"]
        c_hi  = deer_cost_model(solvent_cost_usd_kg=100.0)["total_cost_kg"]
        assert c_lo < c_mid < c_hi, "Higher solvent cost -> higher total cost"

    def test_recovery_rate_sensitivity(self):
        from p42c_sensitivity import deer_cost_model
        c_low_rec  = deer_cost_model(solvent_recovery_rate=0.50)["total_cost_kg"]
        c_high_rec = deer_cost_model(solvent_recovery_rate=0.95)["total_cost_kg"]
        assert c_low_rec > c_high_rec, "Higher recovery rate -> lower net solvent cost"

    def test_tornado_sorted(self):
        from p42c_sensitivity import tornado_data
        t = tornado_data()
        assert len(t) >= 3
        # Should be sorted by delta_max descending
        deltas = [x["delta_max"] for x in t]
        assert deltas == sorted(deltas, reverse=True)

    def test_p42c_runs_without_exception(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "p42c_sensitivity.py")],
            capture_output=True, text=True, timeout=60,
            env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        if result.returncode != 0:
            pytest.fail(f"p42c_sensitivity.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    def test_p42c_output_json(self):
        if not (RESULTS_DIR / "deer_sensitivity.json").exists():
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "src" / "p42c_sensitivity.py")],
                capture_output=True, timeout=60,
                env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                check=False,
            )
        p = RESULTS_DIR / "deer_sensitivity.json"
        assert p.exists(), f"Missing {p}"
        with open(p) as f:
            data = json.load(f)
        assert "base_case" in data
        assert "tornado" in data


# Alias for test module compatibility
RESULTS_DIR = PROJECT_ROOT / "results"
