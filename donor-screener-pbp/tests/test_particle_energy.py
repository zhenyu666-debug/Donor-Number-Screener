"""test_particle_energy.py - LJ + Coulomb sanity tests."""
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))
from utils_pb import lj_energy, coulomb_energy, pair_energy  # noqa: E402
from p24_particle_md import _ATOMIC_MASS  # noqa: E402


def test_lj_minimum():
    r = np.array([0.95, 1.0, 1.122, 1.5, 2.5])  # sigma=1.0, eps=1.0
    e = lj_energy(r, eps=1.0, sigma=1.0)
    # minimum near r = 2^(1/6) sigma -> e[2] is the well depth ~ -1
    assert e[2] < e[1]  # r=1.122 well below r=sigma
    assert e[2] < 0  # negative at the bottom
    assert e[0] > 0  # below 0.95*sigma -> strong repulsion
    # Monotonic decay after minimum
    assert e[4] > e[3] > e[2]


def test_coulomb_signs():
    r = np.array([1.0, 2.0, 4.0])  # A
    e_pos = coulomb_energy(r, q_i=1.0, q_j=1.0, eps_r=12.0)
    e_neg = coulomb_energy(r, q_i=1.0, q_j=-1.0, eps_r=12.0)
    assert (e_pos > 0).all()
    assert (e_neg < 0).all()
    # Magnitude drops with 1/r
    assert e_pos[0] > e_pos[1] > e_pos[2]


def test_pair_energy_symmetry():
    r = np.linspace(1.5, 6.0, 10)
    e_ij = pair_energy(r, q_i=0.3, q_j=-0.3, eps_lj=0.005, sig_lj=3.0, eps_r=12.0, cutoff=12.0)
    e_ji = pair_energy(r, q_i=-0.3, q_j=0.3, eps_lj=0.005, sig_lj=3.0, eps_r=12.0, cutoff=12.0)
    np.testing.assert_allclose(e_ij, e_ji, atol=1e-12)


def test_pair_energy_cutoff_zero():
    rc = 6.0
    r = np.array([rc, rc + 0.1, rc - 0.1])
    e = pair_energy(r, q_i=0.3, q_j=-0.3, eps_lj=0.005, sig_lj=3.0, eps_r=12.0, cutoff=rc)
    # Energy at cutoff should be 0 (cutoff-shifted)
    assert abs(e[0]) < 1e-6


def test_atomic_mass_complete():
    # Smoke: the dict covers what particle_md uses
    needed = {"H", "C", "N", "O", "F", "P", "S", "Cl", "B", "Li"}
    assert needed.issubset(_ATOMIC_MASS.keys())
    assert _ATOMIC_MASS["Li"] < 7.5  # ~6.94 amu


if __name__ == "__main__":
    test_lj_minimum()
    test_coulomb_signs()
    test_pair_energy_symmetry()
    test_pair_energy_cutoff_zero()
    test_atomic_mass_complete()
    print("OK: all particle_energy tests passed")
