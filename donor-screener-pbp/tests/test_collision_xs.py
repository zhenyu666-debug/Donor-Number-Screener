"""test_collision_xs.py - sanity tests for the classical scattering module."""
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))
from p25_collision_xs import (  # noqa: E402
    lj_potential, deflection_chi, transport_cross_section, run,
)
from utils_pb import load_yaml  # noqa: E402


def test_lj_potential_minimum():
    r = np.linspace(0.9, 5.0, 200)
    e = lj_potential(r, eps=1.0, sigma=1.0)
    # minimum near r = 2**(1/6)
    imin = int(np.argmin(e))
    assert abs(r[imin] - 2.0 ** (1.0 / 6.0)) < 0.02
    assert e[imin] < -0.99  # well depth ~ -1


def test_deflection_chi_zero_far_b():
    b = np.linspace(2.5, 3.5, 10)  # far outside LJ range
    chi = deflection_chi(b, eps=1.0, sigma=1.0, mu=10.0, E=1.0)
    # For large b beyond r0, chi should be ~0
    assert (np.abs(chi) < 1e-6).all()


def test_deflection_chi_signs():
    # At very small b, chi -> pi (head-on backscatter)
    chi_small = deflection_chi(np.array([0.01]), eps=1.0, sigma=1.0, mu=10.0, E=0.5)
    assert chi_small[0] > 3.0  # close to pi


def test_transport_xs_positive_and_sane():
    r = transport_cross_section(eps=0.005, sigma=3.0, mu_amu=10.0, T=298.0)
    assert r["sigma_star_A2"] > 0
    assert r["omega_11"] > 0
    # Hard sphere pi sigma^2 = ~28 A^2; LJ transport xs typically O(sigma^2)
    assert r["sigma_star_A2"] < 1e3
    assert 1e-5 < r["mobility_cm2_V_s"] < 1.0
    # 1 M LiPF6 + LJ mobility -> kappa typically O(1e-2 .. 1e3) S/m depending on sigma*
    assert 1e-5 < r["ionic_conductivity_S_m"] < 1e5


def test_run_real_smiles():
    params = load_yaml("particle_params.yaml")
    atoms = params["atoms"]
    r = run("CCO", params, atoms, T=298.15)
    assert r["T_K"] == 298.15
    assert r["sigma_star_A2"] > 0


if __name__ == "__main__":
    test_lj_potential_minimum()
    test_deflection_chi_zero_far_b()
    test_deflection_chi_signs()
    test_transport_xs_positive_and_sane()
    test_run_real_smiles()
    print("OK: all collision_xs tests passed")
