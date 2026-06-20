"""test_fetch_sse.py - tests for the SSE dataset fetcher."""
import math
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))
from p34_fetch_sse_datasets import (  # noqa: E402
    classify_family, formula_cost, fetch_paper_extra,
    merge, fill_missing_with_heuristic, compute_dn_pbp_v2,
)
from p35_pareto_best_sse import dominates, non_dominated_sort  # noqa: E402


def test_classify_family_known():
    assert classify_family("Li10GeP2S12 (LGPS)", "Li10GeP2S12") == "LGPS"
    assert classify_family("Li6PS5Cl (argyrodite)", "Li6PS5Cl") == "argyrodite"
    assert classify_family("LLZO (Ta-doped)", "Li6.4La3Zr1.4Ta0.6O12") == "garnet"
    assert classify_family("LATP", "Li1.3Al0.3Ti1.7(PO4)3") == "NASICON"
    assert classify_family("PEO+LiTFSI", "polymer") == "polymer"
    assert classify_family("LiBH4", "LiBH4") == "hydride"


def test_classify_family_sulfide_default():
    assert classify_family("Li3PS4", "Li3PS4") == "sulfide"
    assert classify_family("Li7P3S11", "Li7P3S11") == "sulfide"


def test_formula_cost_basic():
    c1 = formula_cost("Li3PS4")
    c2 = formula_cost("Li10GeP2S12")
    # adding Ge (5) should raise cost per atom
    assert c2 > c1


def test_formula_cost_known_values():
    # Pure Li -> 1.0
    assert abs(formula_cost("Li") - 1.0) < 1e-6
    # LGPS has Ge (cost 5.0) - total weighted ~1.18 with 25 atoms
    assert formula_cost("Li10GeP2S12") > 1.1
    # Ga (6.0) makes it more expensive
    assert formula_cost("LiGaO2") > formula_cost("LiAlO2")


def test_fetch_paper_extra_loads():
    rows = fetch_paper_extra()
    assert len(rows) >= 5
    for r in rows:
        assert "formula" in r and r["formula"]
        assert r.get("sigma_ion_S_cm") > 0


def test_merge_dedupes_by_formula():
    obelix = [{"id": "A", "formula": "Li3PS4", "name": "Li3PS4",
               "family": "sulfide", "source": "OBELiX",
               "sigma_ion_S_cm": 1.6e-3, "E_g_eV": float("nan"),
               "stability_window_V": float("nan"),
               "migration_barrier_eV": float("nan"),
               "cost_index": 1.0}]
    paper = [{"id": "B", "formula": "Li3PS4", "name": "Li3PS4 (paper)",
              "family": "sulfide", "source": "paper",
              "sigma_ion_S_cm": 1.6e-3, "E_g_eV": 2.5,
              "stability_window_V": 5.0,
              "migration_barrier_eV": 0.3,
              "cost_index": 1.0},
             {"id": "C", "formula": "LGSSSI", "name": "LGSSSI",
              "family": "halide", "source": "paper",
              "sigma_ion_S_cm": 3.2e-2, "E_g_eV": 2.6,
              "stability_window_V": 5.5,
              "migration_barrier_eV": 0.18,
              "cost_index": 2.5}]
    rows = merge([obelix, paper])
    assert len(rows) == 2  # one Li3PS4, one LGSSSI
    keys = {r["formula"] for r in rows}
    assert keys == {"Li3PS4", "LGSSSI"}


def test_fill_missing_with_heuristic():
    row = {"family": "sulfide",
           "E_g_eV": float("nan"),
           "stability_window_V": float("nan"),
           "migration_barrier_eV": float("nan")}
    r = fill_missing_with_heuristic(row)
    assert r["E_g_eV"] == 2.5
    assert r["stability_window_V"] == 5.0
    assert r["migration_barrier_eV"] == 0.30


def test_compute_dn_pbp_v2_finite():
    row = {"sigma_ion_S_cm": 1.6e-3, "E_g_eV": 2.5,
           "stability_window_V": 5.0, "migration_barrier_eV": 0.3,
           "cost_index": 1.0}
    d = compute_dn_pbp_v2(row)
    assert math.isfinite(d)
    assert 0.0 < d < 50.0


def test_dominates_basic():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, 2.0])
    assert dominates(a, b) is True
    assert dominates(b, a) is False


def test_non_dominated_sort_minimal():
    v = np.array([[1, 1], [0.5, 0.5], [2, 0.5], [0.5, 2]])
    nd = non_dominated_sort(v)
    # row 0 (1,1) dominates row 1 (0.5,0.5) -- only one
    assert not bool(nd[1])
    # rows 2 (2,0.5) and 3 (0.5,2) are non-dominated vs row 0 (1,1)
    # and vs each other (2,0.5 vs 0.5,2 -- neither dominates)
    assert bool(nd[0])
    assert bool(nd[2])
    assert bool(nd[3])


if __name__ == "__main__":
    test_classify_family_known()
    test_classify_family_sulfide_default()
    test_formula_cost_basic()
    test_formula_cost_known_values()
    test_fetch_paper_extra_loads()
    test_merge_dedupes_by_formula()
    test_fill_missing_with_heuristic()
    test_compute_dn_pbp_v2_finite()
    test_dominates_basic()
    test_non_dominated_sort_minimal()
    print("OK: all fetch_sse_datasets tests passed")
