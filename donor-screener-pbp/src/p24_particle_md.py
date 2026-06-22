"""24_particle_md.py - Lennard-Jones + Coulomb MD, NVT Berendsen.

Inputs:  SMILES, T (K), rho (g/cm3), n_steps, density
Method:  64-particle cubic box, periodic BC, LJ(eps,sigma) + Coulomb(q_i q_j/4 pi eps r),
         Velocity-Verlet integration, Berendsen thermostat.
Outputs: g_Li-O(r) radial distribution + coordination number
         + DN correction (alpha * n_coord)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import (  # noqa: E402
    RESULTS_DIR, K_B_eV, EPS_0, E_CHARGE, PI,
    load_yaml, set_seed, write_csv, write_json, parse_atoms,
)


# --------------------------------------------------------------------------- #
# Parameter loading + atom building
# --------------------------------------------------------------------------- #

def atom_params_from_smiles(smiles: str, atom_table: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (masses, eps_lj_eV, sigma_A, q_e) for each particle we will simulate.

    Heuristic composition: 1 Li+ plus the atoms of the SMILES. Masses are atomic weights.
    """
    atoms = parse_atoms(smiles)
    atoms = ["Li"] + atoms
    masses, eps_lj, sigma, q = [], [], [], []
    for a in atoms:
        spec = atom_table.get(a, atom_table["C"])
        masses.append(_ATOMIC_MASS.get(a, 12.0))
        eps_lj.append(spec["eps"])
        sigma.append(spec["sigma"])
        q.append(spec["q"])
    # Force total charge to +1 (Li+ + neutral solvent)
    q = np.asarray(q, dtype=float)
    q[0] += 1.0  # Li+
    q_sum = q.sum()
    q -= q_sum / len(q)  # neutralize residual
    return (np.asarray(masses, dtype=float),
            np.asarray(eps_lj, dtype=float),
            np.asarray(sigma, dtype=float),
            q)


_ATOMIC_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
                "P": 30.974, "S": 32.06, "Cl": 35.45, "B": 10.81, "Li": 6.941}


# --------------------------------------------------------------------------- #
# Pair energy / forces
# --------------------------------------------------------------------------- #

def lj_pair(r2: np.ndarray, eps_ij: np.ndarray, sig_ij: np.ndarray) -> np.ndarray:
    """LJ energy per pair for r2 = r^2 with cutoff-shifted tail correction handled by caller."""
    r6 = (sig_ij * sig_ij / np.maximum(r2, 1e-12)) ** 3
    return 4.0 * eps_ij * (r6 * r6 - r6)


def lj_force_mag(r2: np.ndarray, eps_ij: np.ndarray, sig_ij: np.ndarray) -> np.ndarray:
    """Magnitude of the radial LJ force, F_r = -dU/dr (positive = repulsive)."""
    r6 = (sig_ij * sig_ij / np.maximum(r2, 1e-12)) ** 3
    return 24.0 * eps_ij * (2.0 * r6 * r6 - r6) / np.maximum(np.sqrt(r2), 1e-12)


def coulomb_pair(r2: np.ndarray, q_i: np.ndarray, q_j: np.ndarray, eps_r: float) -> np.ndarray:
    """Coulomb energy in eV from r^2 in A^2."""
    r_m = np.sqrt(np.maximum(r2, 1e-12)) * 1e-10
    k = 1.0 / (4.0 * PI * EPS_0 * eps_r)  # J*m
    j = k * q_i * q_j / r_m
    return j / E_CHARGE  # eV


def coulomb_force_mag(r2: np.ndarray, q_i: np.ndarray, q_j: np.ndarray, eps_r: float) -> np.ndarray:
    """Magnitude of the radial Coulomb force (positive = repulsive)."""
    r_m = np.sqrt(np.maximum(r2, 1e-12)) * 1e-10
    k = 1.0 / (4.0 * PI * EPS_0 * eps_r)
    f = k * q_i * q_j / (r_m * r_m)  # N
    f_per_a = f * 1e-10  # convert to eV/A
    return f_per_a / E_CHARGE  # eV/A


# --------------------------------------------------------------------------- #
# Box + neighbor setup
# --------------------------------------------------------------------------- #

def build_box(n: int, density_g_cm3: float, molar_mass: float, padding: float = 1.5) -> float:
    """Return the cubic box edge in Angstrom for n particles at given mass density."""
    mass_g = n * molar_mass / 6.02214076e23
    vol_cm3 = mass_g / density_g_cm3
    vol_a3 = vol_cm3 * 1e24
    edge = (vol_a3) ** (1.0 / 3.0) * padding
    return edge


def initial_positions(n: int, edge: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, edge, size=(n, 3))
    # crude min-distance: shake if too close (rare for n=64 in big box)
    for i in range(n):
        for _ in range(50):
            d = np.linalg.norm(pos - pos[i], axis=1)
            d[i] = np.inf
            j = int(np.argmin(d))
            if d[j] > 2.0:
                break
            pos[i] = rng.uniform(0.0, edge, size=3)
    return pos


def initial_velocities(n: int, mass: np.ndarray, T: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    sigma_v = np.sqrt(K_B_eV * T / mass)  # sqrt(eV / amu) ~ A/fs
    v = rng.normal(0.0, 1.0, size=(n, 3)) * sigma_v[:, None]
    v -= v.mean(axis=0)  # remove COM drift
    return v


# --------------------------------------------------------------------------- #
# Core MD loop
# --------------------------------------------------------------------------- #

def assign_forces(positions: np.ndarray, edge: float, eps_ij: np.ndarray, sig_ij: np.ndarray,
                  q: np.ndarray, eps_r: float, cutoff2: float) -> np.ndarray:
    n = positions.shape[0]
    forces = np.zeros_like(positions)
    for i in range(n - 1):
        d = positions[i + 1:] - positions[i]
        d -= edge * np.round(d / edge)  # minimum image
        r2 = (d * d).sum(axis=1)
        mask = r2 < cutoff2
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        r2m = r2[idx]
        f_mag = lj_force_mag(r2m, eps_ij[idx], sig_ij[idx]) + \
                coulomb_force_mag(r2m, q[i] * q[i + 1 + idx], np.ones_like(r2m), eps_r)
        # Force vector points from j to i: F_ij = f_mag * d_hat
        d_hat = d[idx] / np.sqrt(r2m)[:, None]
        f_vec = f_mag[:, None] * d_hat
        forces[i] += f_vec.sum(axis=0)
        forces[i + 1 + idx] -= f_vec
    return forces


def total_energy(positions: np.ndarray, edge: float, eps_ij: np.ndarray, sig_ij: np.ndarray,
                 q: np.ndarray, eps_r: float, cutoff2: float) -> float:
    n = positions.shape[0]
    e = 0.0
    for i in range(n - 1):
        d = positions[i + 1:] - positions[i]
        d -= edge * np.round(d / edge)
        r2 = (d * d).sum(axis=1)
        mask = r2 < cutoff2
        if not mask.any():
            continue
        r2m = r2[mask]
        e += lj_pair(r2m, eps_ij[mask], sig_ij[mask]).sum()
        e += coulomb_pair(r2m, q[i] * q[i + 1 + np.where(mask)[0]], np.ones_like(r2m), eps_r).sum()
    return float(e)


def run_md(smiles: str, params: dict, atom_table: dict,
           n_steps: int = None, T: float = None, device: str = "cpu") -> dict:
    """Run an NVT MD simulation and return g(r) summary + DN correction."""
    md = params["md"]
    set_seed(int(md.get("seed", 42)))
    n = int(md["n_particles"])
    T = T if T is not None else float(md["temperature_K"])
    dt = float(md["timestep_fs"])
    n_steps = n_steps or int(md["n_steps"])
    cutoff = float(md["cutoff_angstrom"])
    cutoff2 = cutoff * cutoff
    eps_r = float(md["epsilon_r_relative"])
    tau = float(md["thermostat_tau_fs"])
    eq = int(md["equilibration_steps"])
    density = float(md["density_g_cm3"])

    masses, eps_lj, sigma, q = atom_params_from_smiles(smiles, atom_table)
    n = max(n, len(masses))
    # Pad/trim to fixed N
    if len(masses) < n:
        pad = n - len(masses)
        masses = np.concatenate([masses, np.full(pad, 12.0)])
        eps_lj = np.concatenate([eps_lj, np.full(pad, eps_lj.mean())])
        sigma = np.concatenate([sigma, np.full(pad, sigma.mean())])
        q = np.concatenate([q, np.zeros(pad)])
    else:
        masses = masses[:n]
        eps_lj = eps_lj[:n]
        sigma = sigma[:n]
        q = q[:n]

    # Pair mixing (Lorentz-Berthelot)
    eps_ij = np.sqrt(np.outer(eps_lj, eps_lj))
    sig_ij = 0.5 * (sigma[:, None] + sigma[None, :])
    # Flatten upper triangle to a per-pair array
    iu = np.triu_indices(n, k=1)
    eps_pair = eps_ij[iu]
    sig_pair = sig_ij[iu]

    avg_mass = float(masses.mean())
    edge = build_box(n, density, avg_mass, padding=float(md["box_padding_factor"]))
    pos = initial_positions(n, edge, int(md["seed"]))
    vel = initial_velocities(n, masses, T, int(md["seed"]) + 1)
    forces = assign_forces(pos, edge, eps_pair, sig_pair, q, eps_r, cutoff2)

    # Velocity-Verlet with Berendsen
    e_history = []
    t_history = []
    sample_every = max(1, n_steps // 200)
    rdf_bins = np.linspace(0.0, min(cutoff, edge / 2.0), 60)
    rdf_hist = np.zeros(len(rdf_bins) - 1)
    rdf_samples = 0

    for step in range(n_steps):
        # half-kick
        vel += 0.5 * (forces / masses[:, None]) * dt
        pos += vel * dt
        # PBC wrap
        pos -= edge * np.floor(pos / edge)
        forces = assign_forces(pos, edge, eps_pair, sig_pair, q, eps_r, cutoff2)
        vel += 0.5 * (forces / masses[:, None]) * dt
        # Berendsen
        if step >= eq:
            ke_now = 0.5 * (masses[:, None] * vel * vel).sum() * E_CHARGE / 1.602176634e-19
            dof = 3.0 * n
            T_inst = 2.0 * ke_now / (dof * K_B_eV)
            lam = math.sqrt(1.0 + dt / tau * (T / max(T_inst, 1e-3) - 1.0))
            vel *= lam
        if step >= eq and step % sample_every == 0:
            e_pot = total_energy(pos, edge, eps_pair, sig_pair, q, eps_r, cutoff2)
            ke_ev = 0.5 * (masses[:, None] * vel * vel).sum() * E_CHARGE / 1.602176634e-19
            e_history.append(e_pot + ke_ev)
            t_history.append(2.0 * ke_ev / (dof * K_B_eV))
            # RDF
            d = pos[:, None, :] - pos[None, :, :]
            d -= edge * np.round(d / edge)
            r = np.sqrt((d * d).sum(axis=2))
            iu = np.triu_indices(n, k=1)
            rs = r[iu]
            h, _ = np.histogram(rs, bins=rdf_bins)
            rdf_hist += h
            rdf_samples += 1

    rdf = rdf_hist / max(rdf_samples, 1)
    shell = 0.5 * (rdf_bins[1:] + rdf_bins[:-1])
    shell_v = (4.0 / 3.0) * PI * (rdf_bins[1:] ** 3 - rdf_bins[:-1] ** 3)
    n_total_pairs = n * (n - 1) / 2.0
    density_n = n / (edge ** 3)
    g_r = rdf / max(n_total_pairs * density_n * shell_v, 1e-12)

    # Coordination number: integrate g(r) * 4 pi r^2 rho for O around Li (index 0)
    coord_radius_A = 3.5
    coord_mask = shell < coord_radius_A
    n_coord = float(np.sum(g_r[coord_mask] * shell[coord_mask] ** 2 * shell_v[coord_mask]) * density_n)

    dn_correction = 0.5 * n_coord  # heuristic scale, see report
    return {
        "smiles": smiles,
        "n_particles": n,
        "edge_angstrom": float(edge),
        "n_steps": n_steps,
        "T_K": T,
        "rho_g_cm3": density,
        "mean_T_K": float(np.mean(t_history)) if t_history else float("nan"),
        "std_T_K": float(np.std(t_history)) if t_history else float("nan"),
        "n_coord_li_O": n_coord,
        "dn_correction": dn_correction,
        "rdf_radius_A": shell.tolist(),
        "rdf_g": g_r.tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smiles", default="CCO")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", default=str(RESULTS_DIR / "particle_md_rdf.csv"))
    args = p.parse_args()

    params = load_yaml("particle_params.yaml")
    atoms = params["atoms"]
    res = run_md(args.smiles, params, atoms, n_steps=args.steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"r_A": r, "g_r": g} for r, g in zip(res.pop("rdf_radius_A"), res.pop("rdf_g"))]
    write_csv(out, rows)
    write_json(out.with_suffix(".json"), res)
    print(f"[md] {args.smiles} -> mean T = {res['mean_T_K']:.1f} K, "
          f"n_coord = {res['n_coord_li_O']:.2f}, dn_correction = {res['dn_correction']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
