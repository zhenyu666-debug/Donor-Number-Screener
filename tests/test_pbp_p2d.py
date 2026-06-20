"""test_p2d.py - physics constraints for the P2D / 3D module."""
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))
from p31_p2d_3d_micro import (  # noqa: E402
    build_x_grid, build_r_grid, solve_radial_diffusion, butler_volmer_j, solve_heat, solve_stress,
    random_close_packing, micro_structure_currents, solve_p2d,
)
from utils_pb import load_yaml  # noqa: E402


def test_build_x_grid_regions():
    p = {"n_x": 30, "cathode_thickness_um": 80.0, "anode_thickness_um": 80.0,
         "separator_thickness_um": 25.0}
    x, region, L, Lc, La, Ls = build_x_grid(p)
    assert len(x) == 30
    assert len(region) == 30
    assert (region == 0).sum() > 0
    assert (region == 1).sum() > 0
    assert (region == 2).sum() > 0
    assert abs(L - (Lc + Ls + La)) < 1e-9


def test_solve_radial_diffusion_positive():
    p = load_yaml("p2d_3d_params.yaml")["p2d"]
    r = build_r_grid(p)
    c0 = np.full(len(r), 0.5 * p["c_s_max_cathode_mol_m3"])
    c1 = solve_radial_diffusion(c0, r, float(p["D_s_cathode_m2_s"]), 0.1, 0.0, p["c_s_max_cathode_mol_m3"])
    assert (c1 > 0).all()
    assert (c1 <= p["c_s_max_cathode_mol_m3"]).all()


def test_butler_volmer_signs():
    j_pos = butler_volmer_j(phi_s=4.2, phi_e=0.0, T=298.0, c_s_surf=24500,
                            c_e=1000, c_s_max=49000, j0=36.0, U=4.0, alpha=0.5)
    j_neg = butler_volmer_j(phi_s=3.8, phi_e=0.0, T=298.0, c_s_surf=24500,
                            c_e=1000, c_s_max=49000, j0=36.0, U=4.0, alpha=0.5)
    assert j_pos > 0
    assert j_neg < 0


def test_solve_heat_equilibrates_to_T_amb():
    p = load_yaml("p2d_3d_params.yaml")["thermal"]
    n = 20
    x = np.linspace(0.0, 1e-4, n)
    T = np.full(n, 350.0)  # hot initial
    j = np.zeros(n)
    sigma_h = np.zeros(n)
    T_final = solve_heat(T, x, j, sigma_h, p, dt=10.0)
    # After enough time, the centre should cool towards T_amb
    assert T_final[0] < 350.0


def test_solve_stress_finite():
    p = load_yaml("p2d_3d_params.yaml")["mechanical"]
    c_s = np.linspace(0.4, 0.6, 20) * 49000
    T = np.full(20, 298.0)
    sigma = solve_stress(c_s, 49000.0, T, p)
    assert np.isfinite(sigma).all()
    assert np.all(np.abs(sigma) < 1000.0)  # sane range, MPa


def test_random_close_packing_count():
    pos = random_close_packing(80, 1e-4, seed=0)
    assert pos.shape == (80, 3)
    assert (pos >= 0).all() and (pos <= 1e-4).all()


def test_micro_structure_currents_shape():
    pos = random_close_packing(20, 1e-4, seed=0)
    j_1d = np.linspace(0.0, 5.0, 30)
    x = np.linspace(0.0, 1e-4, 30)
    j = micro_structure_currents(pos, j_1d, x)
    assert j.shape == (20,)
    assert np.isfinite(j).all()


def test_solve_p2d_short_run():
    params = load_yaml("p2d_3d_params.yaml")
    res = solve_p2d(params, n_steps=20, dt=0.5)
    assert len(res["rows"]) == 20
    s = res["summary"]
    assert 0.0 < s["V_end_V"] < 5.0
    assert 290.0 < s["T_max_K"] < 500.0
    # micro was generated
    assert res["micro"]["n_particles"] == params["micro3d"]["n_particles"]


if __name__ == "__main__":
    test_build_x_grid_regions()
    test_solve_radial_diffusion_positive()
    test_butler_volmer_signs()
    test_solve_heat_equilibrates_to_T_amb()
    test_solve_stress_finite()
    test_random_close_packing_count()
    test_micro_structure_currents_shape()
    test_solve_p2d_short_run()
    print("OK: all P2D/3D tests passed")
