"""utils_pb.py - shared helpers for the Particle-Bayes-Physics package.

This module is intentionally light: CSV I/O, simple LJ / Coulomb helpers
and a small featurizer. We avoid any hard import on the sibling
`donor-number-screener` package so this repo can be used standalone.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
PARAMS_DIR = DATA_DIR
EXTERNAL_DATA_DIR = DATA_DIR  # external mirrored datasets live alongside internal data

K_B = 1.380649e-23       # J/K
K_B_eV = 8.617333262e-5   # eV/K
N_A = 6.02214076e23       # 1/mol
E_CHARGE = 1.602176634e-19  # C
EPS_0 = 8.8541878128e-12  # F/m
PI = math.pi


def load_yaml(name: str) -> dict:
    p = DATA_DIR / name
    if not p.exists():
        return {}
    with p.open(encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fns = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fns})


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def parse_atoms(smiles: str) -> List[str]:
    """Return a coarse atom-type list from SMILES (uses RDKit if present)."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ["C"]
        return [a.GetSymbol() for a in mol.GetAtoms()]
    except Exception:
        return ["C"]


def lj_energy(r: np.ndarray, eps: float, sigma: float) -> np.ndarray:
    """Lennard-Jones pair energy for an array of distances r > 0."""
    r = np.asarray(r, dtype=float)
    sr = sigma / np.maximum(r, 1e-12)
    sr6 = sr ** 6
    sr12 = sr6 ** 2
    return 4.0 * eps * (sr12 - sr6)


def coulomb_energy(r: np.ndarray, q_i: float, q_j: float, eps_r: float = 12.0) -> np.ndarray:
    """Coulomb pair energy in eV for an array of distances r in Angstrom."""
    r = np.asarray(r, dtype=float) * 1e-10  # A -> m
    k = 1.0 / (4.0 * PI * EPS_0 * eps_r)  # J*m
    j_per_pair = k * q_i * q_j / np.maximum(r, 1e-12)
    return j_per_pair / E_CHARGE  # eV


def coulomb_force(r_angstrom: np.ndarray, q_i: float, q_j: float,
                  eps_r: float = 12.0) -> np.ndarray:
    """Coulomb pair force magnitude in eV/A for an array of distances r in A.
    F = -dU/dr where U = k q_i q_j / r.  Returns magnitude (positive when repulsive).
    """
    r = np.asarray(r_angstrom, dtype=float) * 1e-10  # A -> m
    k = 1.0 / (4.0 * PI * EPS_0 * eps_r)  # J*m
    # dU/dr = -k q_i q_j / r^2, force along r-hat = -dU/dr = +k q_i q_j / r^2
    j_per_pair_per_m = k * q_i * q_j / np.maximum(r * r, 1e-24)
    # J/m -> eV/A: 1 J = 1/eV_C eV; per m = per A * 1e-10; so J/m -> eV/A: /E_CHARGE * 1e-10
    return j_per_pair_per_m / E_CHARGE * 1e-10  # eV/A (sign-preserving)


def pair_energy(r_angstrom: np.ndarray, q_i: float, q_j: float,
                eps_lj: float, sig_lj: float, eps_r: float = 12.0,
                cutoff: float = 12.0) -> np.ndarray:
    """Combined LJ + Coulomb energy per pair, cutoff-shifted to 0 at `cutoff`."""
    r = np.asarray(r_angstrom, dtype=float)
    mask = r > 0
    e = np.zeros_like(r)
    e[mask] = lj_energy(r[mask], eps_lj, sig_lj) + coulomb_energy(r[mask], q_i, q_j, eps_r)
    if cutoff is not None:
        rc = np.asarray(cutoff, dtype=float)
        if rc > 0:
            e_rc = lj_energy(np.array([rc]), eps_lj, sig_lj)[0] + \
                   coulomb_energy(np.array([rc]), q_i, q_j, eps_r)[0]
            e[mask] -= e_rc
    e[~mask] = 0.0
    return e


def ring_buffer_max(x: np.ndarray, n: int) -> float:
    """Naive IACT-free max for chains: just return the last 1/(2n) range."""
    if len(x) < 2 * n:
        return float(np.std(x) + 1e-12)
    return float(np.std(x[-2 * n:]) + 1e-12)


def gelman_rubin(chains: np.ndarray) -> float:
    """Compute R-hat (potential scale reduction factor) for an array of
    independent chains (shape: [n_chains, n_samples])."""
    if chains.ndim != 2 or chains.shape[0] < 2:
        return float("nan")
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    var_hat = (1.0 - 1.0 / n) * W + B / n
    if W <= 0:
        # All chains identical -> perfect agreement.
        return 1.0
    return float(np.sqrt(var_hat / W))


def main() -> int:
    """Tiny smoke entrypoint: load params + parse atoms."""
    p = load_yaml("particle_params.yaml")
    print(f"params: {len(p)} keys")
    for s in ("CCO", "CCN", "C1COC(=O)O1"):
        print(f"{s} -> {parse_atoms(s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
