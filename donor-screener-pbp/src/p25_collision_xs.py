"""25_collision_xs.py - classical scattering cross-section from LJ potential.

Inputs:  SMILES, T (K), reduced mass, eps_lj, sig_lj
Method:  Integrate transport cross-section
             sigma* = 2 pi integral_0^inf (1 - cos chi(b)) b db
         where chi(b) is the classical deflection function
         from the LJ potential V(r) = 4 eps [ (sig/r)^12 - (sig/r)^6 ].
Outputs: sigma* (A^2), mobility mu (cm^2/V/s), ionic conductivity (S/cm)
         + collision integral Omega^(1,1) (dimensionless).
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
    K_B, K_B_eV, E_CHARGE, N_A, load_yaml, write_csv,
)


def lj_potential(r: np.ndarray, eps: float, sigma: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    sr = sigma / np.maximum(r, 1e-12)
    sr6 = sr ** 6
    return 4.0 * eps * (sr6 * sr6 - sr6)


def deflection_chi(b: np.ndarray, eps: float, sigma: float,
                   mu: float, E: float) -> np.ndarray:
    """Classical deflection angle for the LJ potential at impact parameter b,
    relative energy E, reduced mass mu (in eV * s^2 / A^2 if you use A/fs).
    Returns chi in radians.

    mu units: we treat the LJ energy eps in eV and length in Angstrom; the
    angular momentum L = b sqrt(2 mu E) where mu has units of amu * (eV/A^2)
    and E in eV.  We adopt the standard reduced form via the dimensionless
    parameter b* = b/sigma and energy E* = E/eps.
    """
    b = np.asarray(b, dtype=float)
    eps_eps = max(eps, 1e-12)
    E_star = max(E / eps_eps, 1e-12)

    def chi_one(b_val: float) -> float:
        if b_val <= 0:
            return math.pi
        # root of V(r) - E = 0  (turning point)
        # V(r) = 4 eps [ (s/r)^12 - (s/r)^6 ] = E   =>  (s/r)^6 = (1 + sqrt(1+E/E))/2
        x = (1.0 + math.sqrt(1.0 + E_star)) / 2.0  # = (sigma/r0)^6
        if x <= 0:
            return 0.0
        r0 = sigma / (x ** (1.0 / 6.0))
        if b_val >= r0:
            return 0.0
        # integral: chi = pi - 2 b integral_{r0}^infty dr / r^2 / sqrt(1 - V/E - (b/r)^2)
        # Use Simpson on a transformed grid.
        rs = np.linspace(r0, 20.0 * sigma, 4000)
        u = 1.0 - lj_potential(rs, eps_eps, sigma) / E - (b_val / rs) ** 2
        u = np.clip(u, 0.0, None)
        integrand = np.where(u > 0, 1.0 / (rs * rs * np.sqrt(u + 1e-30)), 0.0)
        integral = np.trapezoid(integrand, rs)
        return math.pi - 2.0 * b_val * integral

    return np.array([chi_one(bv) for bv in b])


def transport_cross_section(eps: float, sigma: float, mu_amu: float, T: float,
                            n_b: int = 120) -> dict:
    """Compute sigma*(T) and the dimensionless collision integral Omega^(1,1)."""
    mu_eV_A2 = mu_amu  # amu -> 1.6605e-27 kg, but in LJ units we just need amu
    # average relative kinetic energy <E_rel> = 3/2 kT  (in eV)
    kT_eV = K_B_eV * T
    E_avg = 1.5 * kT_eV
    # impact parameters: from 0 to ~3 sigma (LJ range)
    b_grid = np.linspace(0.0, 3.0 * sigma, n_b)
    chi = deflection_chi(b_grid, eps, sigma, mu_eV_A2, E_avg)
    one_minus_cos = 1.0 - np.cos(chi)
    integrand = one_minus_cos * b_grid
    # Trapezoid
    sigma_star = 2.0 * math.pi * float(np.trapezoid(integrand, b_grid))  # A^2

    # Hard-sphere reference pi sigma^2 to make Omega^(1,1) dimensionless
    sigma_hs = math.pi * sigma * sigma
    omega_11 = sigma_star / sigma_hs

    # Chapman-Enskog mobility:
    # mu = 3 e / (16 N sigma_star) * sqrt(2 pi / (mu_kg kT))  (SI)
    mu_kg = mu_amu * 1.66053906660e-27
    mu_mob = (3.0 * E_CHARGE / (16.0 * N_A * 1e6 * sigma_star * 1e-20)  # A^2 -> m^2
              * math.sqrt(2.0 * math.pi / (mu_kg * K_B * T)))
    mu_cm = mu_mob * 1e4  # m^2/V/s -> cm^2/V/s

    # Nernst-Einstein conductivity (1 M LiPF6 in EC/DMC ~ c = 1000 mol/m^3)
    c_molar = 1000.0
    kappa = c_molar * N_A * E_CHARGE ** 2 * mu_mob / (K_B * T)  # S/m
    return {
        "sigma_star_A2": float(sigma_star),
        "sigma_hs_A2": float(sigma_hs),
        "omega_11": float(omega_11),
        "mobility_cm2_V_s": float(mu_cm),
        "ionic_conductivity_S_m": float(kappa),
    }


def run(smiles: str, params: dict, atom_table: dict, T: float = 298.15) -> dict:
    """Compute xs for a Li+ vs solvent pair. Solvent mass is approximated as the
    average atomic mass in the SMILES; eps/sig from Lorentz-Berthelot mixing."""
    from utils_pb import parse_atoms
    atoms = parse_atoms(smiles)
    if not atoms:
        atoms = ["C"]
    avg_mass = float(np.mean([{"H": 1.0, "C": 12.0, "N": 14.0, "O": 16.0,
                               "F": 19.0, "P": 31.0, "S": 32.0, "Cl": 35.5,
                               "B": 10.8, "Li": 6.9}.get(a, 12.0) for a in atoms]))
    # We treat the collision pair as Li vs "average solvent atom"
    li = atom_table["Li"]
    # Most common heavy atom in solvent
    heavy = max(set(atoms) - {"H", "Li"}, key=atoms.count) if any(a not in ("H", "Li") for a in atoms) else "C"
    solv = atom_table.get(heavy, atom_table["C"])
    eps = math.sqrt(li["eps"] * solv["eps"])
    sig = 0.5 * (li["sigma"] + solv["sigma"])
    mu_amu = (6.941 * avg_mass) / (6.941 + avg_mass)  # reduced mass
    res = transport_cross_section(eps, sig, mu_amu, T)
    res.update({"smiles": smiles, "T_K": T, "mu_amu": mu_amu,
                "eps_eV": eps, "sigma_A": sig, "heavy_atom": heavy,
                "avg_solvent_mass_amu": avg_mass})
    return res


def sweep_temperature(smiles: str, params: dict, atom_table: dict,
                       T_values=(200, 250, 298, 350, 400)) -> list:
    return [run(smiles, params, atom_table, T=T) for T in T_values]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smiles", default="CCO")
    p.add_argument("--out", default=str(THIS_DIR.parent / "results" / "collision_xs.csv"))
    args = p.parse_args()

    params = load_yaml("particle_params.yaml")
    atoms = params["atoms"]
    rows = sweep_temperature(args.smiles, params, atoms)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows)
    r = rows[2]  # 298 K entry
    print(f"[xs] {args.smiles} T=298K: sigma* = {r['sigma_star_A2']:.2f} A^2, "
          f"omega = {r['omega_11']:.2f}, mu = {r['mobility_cm2_V_s']:.3e} cm^2/V/s, "
          f"kappa = {r['ionic_conductivity_S_m']:.3e} S/m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
