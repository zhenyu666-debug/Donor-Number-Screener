"""test_sei.py - physics constraints + monotonicity tests for SEI/EDL."""
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))
from p27_sei_edl import (  # noqa: E402
    helmholtz_capacitance, debye_length, ionic_conductivity_bulk,
    sei_resistance, butler_volmer_j, dn_attenuation, run,
)
from utils_pb import load_yaml  # noqa: E402


def test_helmholtz_capacitance_units():
    c = helmholtz_capacitance(eps_r=10.0)
    # eps_r * eps_0 ~ 10 * 8.85e-12 ~ 8.85e-11 F/m^2
    assert 5e-11 < c < 1e-10


def test_debye_length_decreases_with_conc():
    l1 = debye_length(eps_r=25.0, c_bulk=100.0, T=298.0)
    l2 = debye_length(eps_r=25.0, c_bulk=1000.0, T=298.0)
    assert l2 < l1


def test_ionic_conductivity_positive():
    k = ionic_conductivity_bulk(c=1000.0, D_li=1e-9, T=298.0)
    assert k > 0
    # Typical 1 M LiPF6 in EC/DMC ~ 10 S/m
    assert 0.5 < k < 20.0


def test_sei_resistance_increases_with_thickness():
    r1 = sei_resistance(thickness_nm=5.0, sigma_ion_S_m=1e-6, area_cm2=1.0)
    r2 = sei_resistance(thickness_nm=50.0, sigma_ion_S_m=1e-6, area_cm2=1.0)
    assert r2 > r1
    # At typical params, 5 nm should be O(0.1 - 10) Ohm
    assert 0.01 < r1 < 100.0


def test_butler_volmer_zero_overpotential():
    j = butler_volmer_j(j0=0.05, eta=0.0, T=298.0)
    assert abs(j) < 1e-6


def test_butler_volmer_signs():
    j_pos = butler_volmer_j(j0=0.05, eta=0.1, T=298.0)  # anodic
    j_neg = butler_volmer_j(j0=0.05, eta=-0.1, T=298.0)  # cathodic
    assert j_pos > 0
    assert j_neg < 0


def test_dn_attenuation_floor():
    d0 = dn_attenuation(thickness_nm=0.0, dn_bulk=20.0, floor=0.5)
    d_inf = dn_attenuation(thickness_nm=1000.0, dn_bulk=20.0, floor=0.5)
    assert abs(d0 - 20.0) < 1e-9
    assert abs(d_inf - 10.0) < 1e-3  # floor * 20


def test_run_full_sweep():
    params = load_yaml("sei_params.yaml")
    res = run(params, dn_bulk=20.0, T=298.0)
    rows = res["rows"]
    assert len(rows) > 10
    # R_total must increase with thickness
    r_vals = [r["r_total_ohm"] for r in rows]
    assert all(b > a for a, b in zip(r_vals, r_vals[1:]))
    # DN attenuation monotonically decreases
    dn_vals = [r["dn_eff"] for r in rows]
    assert all(b < a for a, b in zip(dn_vals, dn_vals[1:]))


if __name__ == "__main__":
    test_helmholtz_capacitance_units()
    test_debye_length_decreases_with_conc()
    test_ionic_conductivity_positive()
    test_sei_resistance_increases_with_thickness()
    test_butler_volmer_zero_overpotential()
    test_butler_volmer_signs()
    test_dn_attenuation_floor()
    test_run_full_sweep()
    print("OK: all SEI/EDL tests passed")
