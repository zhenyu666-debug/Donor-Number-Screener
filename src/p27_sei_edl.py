"""27_sei_edl.py - Solid|solid|electrolyte SEI / EDL impedance + ionic conductivity.

Models the three-sandwich cell:
  NMC cathode  |  CEI  |  electrolyte  |  SEI  |  Li metal anode

We compute:
  - SEI / CEI ionic resistance (parallel + series combo) for a sweep of
    thicknesses.
  - Helmholtz double-layer capacitance at the Li | electrolyte interface.
  - Butler-Volmer exchange current + plating overpotential.
  - Nernst-Planck ionic conductivity of the bulk electrolyte.
  - An interface "DN attenuation factor" that multiplies the bulk DN to
    reflect the chemistry loss in the dense SEI/CEI layers.

This is intentionally a 0-D / 1-D analytical model; we are not solving
PDEs. See the report markdown for the equations and ranges.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import (  # noqa: E402
    RESULTS_DIR, EPS_0, E_CHARGE, K_B, N_A, load_yaml, write_csv, write_json,
)


# --------------------------------------------------------------------------- #
# Component computations
# --------------------------------------------------------------------------- #

def helmholtz_capacitance(eps_r: float) -> float:
    """C_H = eps_r eps_0 / lambda_D [F/m^2]."""
    return eps_r * EPS_0


def debye_length(eps_r: float, c_bulk: float, T: float) -> float:
    """Debye length [m] for a symmetric 1:1 electrolyte at concentration c [mol/m^3]."""
    F = E_CHARGE
    return math.sqrt(eps_r * EPS_0 * K_B * T / (2.0 * c_bulk * N_A * F ** 2))


def ionic_conductivity_bulk(c: float, D_li: float, z_li: int = 1, T: float = 298.15) -> float:
    """Nernst-Einstein conductivity [S/m] of a 1:1 electrolyte at conc c [mol/m^3]."""
    F = E_CHARGE
    return c * N_A * F ** 2 * D_li / (K_B * T)


def sei_resistance(thickness_nm: float, sigma_ion_S_m: float, area_cm2: float = 1.0) -> float:
    """R = L / (sigma A) in Ohm for an SEI of given thickness and area."""
    L_m = thickness_nm * 1e-9
    A_m2 = area_cm2 * 1e-4
    return L_m / (max(sigma_ion_S_m, 1e-30) * A_m2)


def butler_volmer_j(j0: float, eta: float, T: float, alpha_a: float = 0.5,
                    alpha_c: float = 0.5) -> float:
    """Net current density [A/m^2] for overpotential eta (V)."""
    F = E_CHARGE
    RT = K_B * T
    return j0 * (math.exp(alpha_a * F * eta / RT) - math.exp(-alpha_c * F * eta / RT))


def plating_overpotential(j: float, j0: float, T: float, alpha: float = 0.5) -> float:
    """Invert Butler-Volmer (high-field limit) for overpotential.
    eta = (RT / (alpha F)) ln(j / j0)
    """
    if j0 <= 0 or j <= 0:
        return 0.0
    F = E_CHARGE
    return (K_B * T) / (alpha * F) * math.log(j / j0)


def dn_attenuation(thickness_nm: float, dn_bulk: float,
                   saturation_nm: float = 30.0, floor: float = 0.5) -> float:
    """Multiplicative DN attenuation through a dense SEI layer.

    d_eff = d_bulk * (floor + (1 - floor) * exp(- thickness / saturation))
    """
    return dn_bulk * (floor + (1.0 - floor) * math.exp(-thickness_nm / saturation_nm))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def sweep_thickness(params: dict, dn_bulk: float = 20.0, T: float = 298.15) -> list:
    sei = params["sei"]
    edl = params["edl"]
    cath = params["cathode"]
    op = params["operating"]

    n_steps = int(sei["thickness_nm_steps"])
    th_min = float(sei["thickness_nm_min"])
    th_max = float(sei["thickness_nm_max"])
    thicknesses = np.linspace(th_min, th_max, n_steps)
    sigma_ion = float(sei["ionic_conductivity_S_m"])
    c_bulk = 1000.0  # mol/m^3 ~ 1 M
    D_li = 1e-9      # m^2/s
    j0 = float(sei["j0_A_m2"])
    area_cm2 = float(op["cell_area_cm2"])
    c_rate = float(op["c_rate"])
    # Reference current density for 1C of 1 mAh/cm^2 capacity
    j_ref = c_rate * 1.0 * area_cm2 * 3600.0 / 3600.0  # A/m^2 (1C of 1 mAh/cm2)

    rows = []
    for th in thicknesses:
        # SEI resistance (one layer, the CEI gets half the thickness)
        r_sei = sei_resistance(th, sigma_ion, area_cm2)
        r_cei = sei_resistance(float(cath["cei_thickness_nm"]), sigma_ion, area_cm2)
        r_total = r_sei + r_cei

        # Bulk electrolyte conductivity
        kappa = ionic_conductivity_bulk(c_bulk, D_li, T=T)
        # Bulk resistance across typical 25 um separator
        r_bulk = (25e-6) / (kappa * area_cm2 * 1e-4)

        # Capacitance
        c_h_sei = helmholtz_capacitance(float(sei["permittivity_relative"]))
        c_h_edl = helmholtz_capacitance(float(edl["helmholtz_eps_r"]))
        lam_d = debye_length(float(edl["electrolyte_eps_r"]), c_bulk, T)

        # Plating overpotential (Li)
        eta_pl = plating_overpotential(j_ref, j0, T)

        # Total R and tau
        r_total_with_bulk = r_total + r_bulk
        tau = (c_h_edl + c_h_sei) * r_total_with_bulk

        # DN attenuation
        d_eff = dn_attenuation(th, dn_bulk, saturation_nm=30.0, floor=0.5)

        rows.append({
            "thickness_nm": float(th),
            "r_sei_ohm": float(r_sei),
            "r_cei_ohm": float(r_cei),
            "r_bulk_ohm": float(r_bulk),
            "r_total_ohm": float(r_total_with_bulk),
            "c_h_sei_F_m2": float(c_h_sei),
            "c_h_edl_F_m2": float(c_h_edl),
            "debye_length_nm": float(lam_d * 1e9),
            "eta_plating_V": float(eta_pl),
            "tau_s": float(tau),
            "kappa_bulk_S_m": float(kappa),
            "dn_bulk": float(dn_bulk),
            "dn_eff": float(d_eff),
            "dn_attenuation": float(d_eff / dn_bulk),
        })
    return rows


def run(params: dict, dn_bulk: float = 20.0, T: float = 298.15) -> dict:
    rows = sweep_thickness(params, dn_bulk=dn_bulk, T=T)
    if not rows:
        return {"rows": [], "summary": {}}
    mid = rows[len(rows) // 2]
    return {
        "rows": rows,
        "summary": {
            "thickness_nm_mid": mid["thickness_nm"],
            "r_total_ohm_mid": mid["r_total_ohm"],
            "dn_eff_mid": mid["dn_eff"],
            "kappa_bulk_S_m": mid["kappa_bulk_S_m"],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dn_bulk", type=float, default=22.0)
    p.add_argument("--T", type=float, default=298.15)
    p.add_argument("--out", default=str(RESULTS_DIR / "sei_impedance.csv"))
    args = p.parse_args()
    params = load_yaml("sei_params.yaml")
    res = run(params, dn_bulk=args.dn_bulk, T=args.T)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, res["rows"])
    write_json(out.with_suffix(".json"), res["summary"])
    s = res["summary"]
    print(f"[sei] T={args.T}K, mid thickness {s['thickness_nm_mid']:.1f} nm: "
          f"R_total={s['r_total_ohm_mid']:.3e} Ohm, "
          f"kappa_bulk={s['kappa_bulk_S_m']:.3e} S/m, "
          f"DN_eff={s['dn_eff_mid']:.2f} (from {args.dn_bulk})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
