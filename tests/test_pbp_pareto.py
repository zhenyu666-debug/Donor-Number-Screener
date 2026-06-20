"""test_pareto.py - tests for the Pareto front computation."""
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))
from p35_pareto_best_sse import (  # noqa: E402
    dominates, non_dominated_sort, build_objective_matrix, normalize,
    representative_by_distance, representative_per_family,
)


def test_dominates_strict():
    a = np.array([2.0, 2.0])
    b = np.array([1.0, 1.0])
    assert dominates(a, b)
    assert not dominates(b, a)
    # equal in one dim, strictly better in the other -> still dominates
    # (this is the standard Pareto definition)
    c = np.array([2.0, 1.0])
    assert dominates(a, c)
    # equal in all dims -> not a strict dominance
    d = np.array([2.0, 2.0])
    assert not dominates(a, d)
    assert not dominates(c, a)


def test_non_dominated_sort_simple():
    # 3 points: A dominates B, C is non-dominated with A
    v = np.array([
        [1.0, 1.0],   # A
        [0.5, 0.5],   # B
        [0.8, 1.1],   # C
    ])
    nd = non_dominated_sort(v)
    assert bool(nd[0])  # A
    assert not nd[1]    # B
    assert bool(nd[2])  # C


def test_non_dominated_sort_empty():
    v = np.zeros((0, 2))
    nd = non_dominated_sort(v)
    assert len(nd) == 0


def test_build_objective_matrix_signs():
    rows = [
        {"sigma_ion_S_cm": 1.0, "E_g_eV": 2.0,
         "stability_window_V": 4.0, "migration_barrier_eV": 0.3,
         "cost_index": 1.0},
        {"sigma_ion_S_cm": 1.0e-6, "E_g_eV": 6.0,
         "stability_window_V": 6.0, "migration_barrier_eV": 0.7,
         "cost_index": 5.0},
    ]
    objs = [("sigma_ion_S_cm", 1.0), ("E_g_eV", 1.0),
            ("stability_window_V", 1.0),
            ("migration_barrier_eV", -1.0), ("cost_index", -1.0)]
    M = build_objective_matrix(rows, objs)
    # migration and cost have negative signs -> first row should have higher
    # values in those columns
    assert M[0, 3] > M[1, 3]  # -0.3 > -0.7
    assert M[0, 4] > M[1, 4]  # -1.0 > -5.0


def test_normalize_zero_range():
    M = np.array([[1.0, 1.0], [1.0, 1.0]])
    out = normalize(M)
    assert np.allclose(out, 0.5)


def test_normalize_basic():
    M = np.array([[0.0, 0.0], [1.0, 1.0]])
    out = normalize(M)
    assert out[0, 0] == 0.0 and out[1, 0] == 1.0


def test_representative_by_distance():
    M = np.array([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]])
    mask = np.array([True, False, True])
    M_norm = normalize(M)
    idx = representative_by_distance(M_norm, mask)
    assert idx == 0  # row 0 is the ideal point


def test_representative_per_family():
    rows = [
        {"family": "sulfide", "name": "Li3PS4"},
        {"family": "oxide", "name": "LLZO"},
        {"family": "sulfide", "name": "LGPS"},
    ]
    M = np.array([[1.0, 1.0], [0.5, 0.5], [0.9, 0.9]])
    mask = np.array([True, True, True])
    M_norm = normalize(M)
    d = representative_per_family(rows, M_norm, mask)
    assert d["sulfide"] == 0
    assert d["oxide"] == 1


if __name__ == "__main__":
    test_dominates_strict()
    test_non_dominated_sort_simple()
    test_non_dominated_sort_empty()
    test_build_objective_matrix_signs()
    test_normalize_zero_range()
    test_normalize_basic()
    test_representative_by_distance()
    test_representative_per_family()
    print("OK: all pareto tests passed")
