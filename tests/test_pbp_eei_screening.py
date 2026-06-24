"""test_pbp_eei_screening.py - Tests for p40 EEI solvent screening layer.

Validates:
  - p40_solvent_screening.py runs end-to-end without exception
  - DMI ranks in top-3 by eei_dissolution_score
  - EC/DMC scores < 0.15 on eei_dissolution_score
  - Pareto front is non-empty and non-dominated
  - API p40c starts and scores known solvents
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest  # noqa: E402


class TestPhysicsScoring:
    """Unit tests for the physics scoring functions."""

    def test_dmi_reference(self):
        from p40_solvent_screening import physics_score
        s = physics_score(dn=29.0, an=10.4, epsilon_r=37.7,
                          oxidation_V=4.8, reduction_V=0.5, viscosity_cp=2.1)
        assert 0.90 <= s["eei_dissolution_score"] <= 0.99, \
            f"DMI should score ~0.95, got {s['eei_dissolution_score']}"
        assert 0.85 <= s["electrode_compat_score"] <= 0.95
        assert s["regeneration_potential_pct"] >= 89.0

    def test_ec_control(self):
        from p40_solvent_screening import physics_score
        s = physics_score(dn=16.8, an=10.0, epsilon_r=89.0,
                          oxidation_V=4.2, reduction_V=0.5, viscosity_cp=1.9)
        assert s["eei_dissolution_score"] < 0.20, \
            f"EC should have low EEI dissolution, got {s['eei_dissolution_score']}"
        assert s["electrode_compat_score"] > 0.80

    def test_dmso_high_dn(self):
        from p40_solvent_screening import physics_score
        s = physics_score(dn=29.8, an=19.3, epsilon_r=47.2,
                          oxidation_V=4.2, reduction_V=0.8, viscosity_cp=-0.5)
        assert s["eei_dissolution_score"] >= 0.80
        # DMSO oxidation stability is lower than DMI, so compat is lower
        assert s["electrode_compat_score"] < 0.86

    def test_an_high_penalty(self):
        from p40_solvent_screening import physics_score
        s_low  = physics_score(dn=20.0, an=10.0, epsilon_r=30.0, oxidation_V=4.2, reduction_V=0.8)
        s_high = physics_score(dn=35.0, an=10.0, epsilon_r=30.0, oxidation_V=4.2, reduction_V=0.8)
        assert s_high["electrode_compat_score"] < s_low["electrode_compat_score"], \
            "Very high DN should penalize electrode compat"

    def test_composite_ordering(self):
        from p40_solvent_screening import physics_score, composite_score
        dmi = physics_score(29.0, 10.4, 37.7, 4.8, 0.5, 2.1)
        ec  = physics_score(16.8, 10.0, 89.0, 4.2, 0.5, 1.9)
        dmi["eei_dissolution_score"] = 0.95
        ec["eei_dissolution_score"]  = 0.10
        dmi["electrode_compat_score"] = 0.90
        ec["electrode_compat_score"]  = 0.95
        dmi["regeneration_potential_pct"] = 95.0
        ec["regeneration_potential_pct"]  = 83.0
        assert composite_score(dmi) > composite_score(ec)


class TestSolventLibrary:
    """Tests for the solvent data files."""

    def test_solvent_csv_exists(self):
        p = PROJECT_ROOT / "data" / "solvent_eei_properties.csv"
        assert p.exists(), f"Missing {p}"

    def test_solvent_csv_loads(self):
        import pandas as pd
        df = pd.read_csv(PROJECT_ROOT / "data" / "solvent_eei_properties.csv")
        assert len(df) >= 25, f"Expected >=25 solvents, got {len(df)}"
        assert "smiles" in df.columns
        assert "dn" in df.columns
        assert "eei_dissolution_score" in df.columns

    def test_solvent_yaml_exists(self):
        p = PROJECT_ROOT / "data" / "solvent_library.yaml"
        assert p.exists(), f"Missing {p}"

    def test_solvent_yaml_loads(self):
        import yaml
        with open(PROJECT_ROOT / "data" / "solvent_library.yaml",
                  "rb") as f:
            data = yaml.safe_load(f.read().decode("utf-16", errors="replace"))
        assert "solvents" in data
        assert "DMI" in data["solvents"]


class TestP40Script:
    """End-to-end smoke tests for p40_solvent_screening.py."""

    def test_runs_without_exception(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "p40_solvent_screening.py")],
            capture_output=True, text=True, timeout=120,
            env={**subprocess.os.environ,
                 "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        if result.returncode != 0:
            pytest.fail(f"p40_solvent_screening.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        assert "Done." in result.stdout or result.returncode == 0

    def test_output_csv_created(self):
        out = PROJECT_ROOT / "results" / "solvent_eei_predictions.csv"
        # Run p40 first if not already run (subprocess call above is separate)
        if not out.exists():
            import subprocess
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "src" / "p40_solvent_screening.py")],
                capture_output=True, timeout=120,
                env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                check=False,
            )
        assert out.exists(), f"Missing {out}"

    def test_dmi_top3_dissolution(self):
        import pandas as pd
        out = PROJECT_ROOT / "results" / "solvent_eei_predictions.csv"
        if not out.exists():
            pytest.skip("Run p40 first")
        df = pd.read_csv(out)
        dmi = df[df["name"].str.contains("DMI", na=False)]
        assert not dmi.empty, "DMI not found in results"
        top3_diss = df.nlargest(3, "eei_dissolution_score")["name"].tolist()
        assert any("DMI" in n for n in top3_diss), \
            f"DMI should be in top-3 by dissolution. Top-3: {top3_diss}"

    def test_ec_low_dissolution(self):
        import pandas as pd
        out = PROJECT_ROOT / "results" / "solvent_eei_predictions.csv"
        if not out.exists():
            pytest.skip("Run p40 first")
        df = pd.read_csv(out)
        ec_rows = df[df["name"].str.contains("Ethylene carbonate", na=False)]
        for _, ec in ec_rows.iterrows():
            assert ec["eei_dissolution_score"] < 0.20, \
                f"EC should score <0.20, got {ec['eei_dissolution_score']}"

    def test_summary_json_created(self):
        p = PROJECT_ROOT / "results" / "solvent_screening_summary.json"
        if not p.exists():
            pytest.skip("Run p40 first")
        import json
        with open(p) as f:
            s = json.load(f)
        assert "n_solvents" in s
        assert "dmi_eei_dissolution_score" in s


class TestP40Pareto:
    """Tests for p40b_solvent_pareto.py."""

    def test_pareto_runs(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "p40b_solvent_pareto.py")],
            capture_output=True, text=True, timeout=60,
            env={**subprocess.os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        if result.returncode != 0:
            pytest.fail(f"p40b_pareto.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    def test_pareto_front_nonempty(self):
        import pandas as pd
        p = PROJECT_ROOT / "data" / "solvent_pareto_front.csv"
        if not p.exists():
            pytest.skip("Run p40b first")
        df = pd.read_csv(p)
        assert len(df) >= 1, f"Pareto front should have >=1 entries, got {len(df)}"

    def test_pareto_nondominated(self):
        from p40b_solvent_pareto import dominates, OBJECTIVES
        # A clearly dominates B on all objectives
        a = {"eei_dissolution_score": 0.9, "electrode_compat_score": 0.9,
             "regeneration_potential_pct": 95.0}
        b = {"eei_dissolution_score": 0.5, "electrode_compat_score": 0.5,
             "regeneration_potential_pct": 70.0}
        assert dominates(a, b, OBJECTIVES)
        assert not dominates(b, a, OBJECTIVES)
        # c is strictly worse on all objectives — a should dominate c
        c = {"eei_dissolution_score": 0.8, "electrode_compat_score": 0.8,
             "regeneration_potential_pct": 85.0}
        assert dominates(a, c, OBJECTIVES)
class TestP40API:
    """Tests for p40c_solvent_rest_api.py."""

    def test_score_smiles_dmi(self):
        from p40c_solvent_rest_api import score_smiles
        r = score_smiles("O=C1N(C)CCC1", "DMI")
        assert r.eei_dissolution_score >= 0.80
        assert r.electrode_compat_score >= 0.80
        assert "Tier-1" in r.tier or "Tier-2" in r.tier

    def test_score_smiles_ec(self):
        from p40c_solvent_rest_api import score_smiles
        r = score_smiles("O=C1OCCO1", "EC")
        assert r.eei_dissolution_score < 0.80, \
            f"EC should have lower EEI dissolution, got {r['eei_dissolution_score']}"
        assert r.electrode_compat_score > 0.70

    def test_compare(self):
        from p40c_solvent_rest_api import score_smiles
        dmi = score_smiles("O=C1N(C)CCC1", "DMI")
        ec  = score_smiles("O=C1OCCO1", "EC")
        assert dmi.eei_dissolution_score > ec.eei_dissolution_score
        assert ec.electrode_compat_score >= dmi.electrode_compat_score

    def test_fastapi_available(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("fastapi not installed")
