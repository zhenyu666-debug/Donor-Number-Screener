"""test_ml_aimd.py - tests for the ML-AIMD module.

These tests do not require MACE / CHGNet. They cover:
  - LJ fallback calculator: forces/energy sanity
  - Per-SSE row construction
  - Library loading
  - Heuristic DN correction bounds
"""
import math
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))
from p30_ml_aimd import (  # noqa: E402
    LJCalculator, load_sse_library, run_one_sse, SSE_COMPOSITIONS,
)
from utils_pb import load_yaml  # noqa: E402


def test_lj_calculator_energy_finite():
    from ase import Atoms
    a = Atoms("Li3", positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0]], cell=[6, 6, 6], pbc=True)
    calc = LJCalculator(eps_eV=0.005, sigma_A=3.0)
    e = calc.get_potential_energy(a)
    assert math.isfinite(e)
    f = calc.get_forces(a)
    assert f.shape == (3, 3)
    assert np.isfinite(f).all()


def test_lj_calculator_pair_signs():
    from ase import Atoms
    a_close = Atoms("Li2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    a_far = Atoms("Li2", positions=[[0, 0, 0], [3.0, 0, 0]], cell=[5, 5, 5], pbc=True)
    calc = LJCalculator(eps_eV=0.005, sigma_A=3.0)
    e_close = calc.get_potential_energy(a_close)
    e_far = calc.get_potential_energy(a_far)
    # at r=1.5 (below LJ minimum ~3.16) energy is repulsive
    assert e_close > e_far


def test_sse_library_loadable():
    lib = load_sse_library()
    assert len(lib) == 14
    names = [x["name"] for x in lib]
    # Library uses "Name (alias)" format
    for n in ("Li3PS4 (beta)", "LGPS", "LLZO (Ta-doped)", "LATP"):
        # either exact match OR (alias) suffix
        assert any((n == nm) or (nm.endswith(f"({n})")) for nm in names), f"{n} not in {names}"


def test_sse_compositions_known():
    # All 14 SSEs must have a composition entry; allow alias or full name.
    # Library keys are like "Li10GeP2S12 (LGPS)" — match by substring (case-insensitive).
    keys = list(SSE_COMPOSITIONS.keys())
    def find(full: str) -> str | None:
        if full in keys:
            return full
        fl = full.lower()
        for k in keys:
            kl = k.lower()
            if fl in kl:
                return k
        return None
    for full in ("Li3PS4 (beta)", "LGPS", "LLZO (Ta-doped)", "PEO+LiTFSI (polymer)"):
        match = find(full)
        assert match is not None, f"missing {full} in {sorted(keys)[:5]}..."
        assert len(SSE_COMPOSITIONS[match]) > 0


def test_run_one_sse_lib_only_fallback():
    """When ASE is missing, run_one_sse should still return a row from the lib entry."""
    params = load_yaml("ml_aimd_params.yaml")
    lib = load_sse_library()
    li3ps4 = next(x for x in lib if x["name"] == "Li3PS4 (beta)")
    r = run_one_sse("Li3PS4 (beta)", li3ps4, params, calc=None)
    assert r["sse"] == "Li3PS4 (beta)"
    assert r["dn_aimd"] >= 0.0
    assert r["barrier_eV"] > 0


def test_dn_aimd_sign_log_sanity():
    """Higher sigma_ion -> higher DN. Compare Li3PS4 (1.6e-3) vs LiPON (2e-6)."""
    params = load_yaml("ml_aimd_params.yaml")
    lib = load_sse_library()
    by_name = {x["name"]: x for x in lib}
    r1 = run_one_sse("Li3PS4 (beta)", by_name["Li3PS4 (beta)"], params, calc=None)
    r2 = run_one_sse("LiPON", by_name["LiPON"], params, calc=None)
    assert r1["dn_aimd"] > r2["dn_aimd"]


if __name__ == "__main__":
    test_lj_calculator_energy_finite()
    test_lj_calculator_pair_signs()
    test_sse_library_loadable()
    test_sse_compositions_known()
    test_run_one_sse_lib_only_fallback()
    test_dn_aimd_sign_log_sanity()
    print("OK: all ML-AIMD tests passed")
