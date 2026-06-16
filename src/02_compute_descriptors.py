"""Step 2: Compute molecular descriptors with RDKit.

Output features (per molecule):
- 200+ RDKit 2D descriptors (Descriptors.descList)
- Custom quantum-chemistry proxy features explicitly mapped to the
  paper's mention of "HOMO, LUMO, dipole moment, polarizability, TPSA,
  HBD/HBA, molecular connectivity indices, Kier shape index".
- Atom counts (O, N, F, S, P, halogen).
- Gasteiger charge statistics (proxy for dipole moment & HOMO).

We deliberately avoid an external DFT call (no MOPAC / xTB / Psi4) and
approximate HOMO/LUMO/dipole with cheap, RDKit-native proxies so the
reproduction runs in seconds.  The proxy column names match the
paper's terminology for direct comparison.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.EState import EState

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, get_logger, set_global_seed  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

set_global_seed(42)
log = get_logger("descriptors")

# Disable slow EState_VSA computation that prints a lot of warnings.
try:
    from rdkit.Chem.EState import EState_VSA
except Exception:
    EState_VSA = None


# --------------------------------------------------------------------------- #
# All default RDKit 2D descriptors.
# --------------------------------------------------------------------------- #
DESC_LIST = [(name, func) for name, func in Descriptors.descList]


# --------------------------------------------------------------------------- #
# Custom proxy features.
# --------------------------------------------------------------------------- #
ELECTRONEGATIVITY = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44,
                     "F": 3.98, "P": 2.19, "S": 2.58, "Cl": 3.16,
                     "Br": 2.96, "I": 2.66, "B": 2.04, "Si": 1.90}


def count_atoms(mol: Chem.Mol, symbol: str) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == symbol)


def gasteiger_features(mol: Chem.Mol) -> dict:
    """Compute Gasteiger charges and summary statistics.

    Sum |q| approximates electron-density polarisation -> proxy for
    dipole moment.  Max/min q are proxies for the most/least
    nucleophilic site (HOMO/LUMO location).
    """
    try:
        mol2 = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(mol2, throwOnParamFailure=False)
        charges = np.array([
            float(a.GetProp("_GasteigerCharge"))
            if a.HasProp("_GasteigerCharge")
            else 0.0
            for a in mol2.GetAtoms()
        ])
        if not np.isfinite(charges).all():
            charges = np.nan_to_num(charges, nan=0.0)
    except Exception:
        return {f"gast_{k}": 0.0 for k in
                ("sum_abs", "mean", "max", "min", "range", "std")}
    return {
        "gast_sum_abs": float(np.sum(np.abs(charges))),
        "gast_mean": float(np.mean(charges)),
        "gast_max": float(np.max(charges)),
        "gast_min": float(np.min(charges)),
        "gast_range": float(np.ptp(charges)),
        "gast_std": float(np.std(charges)),
    }


def get_chi_indices(mol: Chem.Mol) -> dict:
    """Molecular connectivity (Chi) and shape (Kappa) indices."""
    out = {}
    try:
        for k in (0, 1):
            for v, fn in (("Chi0v", rdMolDescriptors.CalcChi0v),
                          ("Chi1v", rdMolDescriptors.CalcChi1v),
                          ("Chi2v", rdMolDescriptors.CalcChi2v),
                          ("Chi3v", rdMolDescriptors.CalcChi3v),
                          ("Chi4v", rdMolDescriptors.CalcChi4v)):
                try:
                    out[f"{v}{k}"] = float(fn(mol, k))
                except Exception:
                    out[f"{v}{k}"] = 0.0
        for k in (1, 2, 3):
            try:
                out[f"Kappa{k}"] = float(rdMolDescriptors.CalcKappa(mol, k))
            except Exception:
                out[f"Kappa{k}"] = 0.0
        for k in (1, 2, 3):
            try:
                out[f"Phi{k}"] = float(rdMolDescriptors.CalcPhi(mol, k))
            except Exception:
                out[f"Phi{k}"] = 0.0
    except Exception:
        pass
    return out


def proxy_homo_lumo(mol: Chem.Mol) -> dict:
    """Cheap HOMO/LUMO proxy from EState indices and electronegativity.

    The EState MaxEStateIndex correlates with electron-donating
    ability and is monotonic with HOMO energy; MinEStateIndex with
    electron-withdrawing ability and LUMO energy.  We further mix in
    a halogen/oxygen bonus because strong Lewis bases are typically
    electron-rich at the donor atom.
    """
    try:
        from rdkit.Chem.EState import EStateIndices
        es = EStateIndices(mol)
        max_e = float(np.max(es))
        min_e = float(np.min(es))
        mean_e = float(np.mean(es))
    except Exception:
        max_e = min_e = mean_e = 0.0

    # Average electronegativity of the heavy atoms (without H) -> LUMO proxy.
    en_vals = [ELECTRONEGATIVITY.get(a.GetSymbol(), 2.5) for a in mol.GetAtoms()
               if a.GetSymbol() != "H"]
    mean_en = float(np.mean(en_vals)) if en_vals else 0.0

    # Halogen bonus raises HOMO_a (electronegativity) and lowers LUMO_a
    n_F = count_atoms(mol, "F")
    n_O = count_atoms(mol, "O")
    n_N = count_atoms(mol, "N")
    n_S = count_atoms(mol, "S")
    n_X = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in {"F", "Cl", "Br", "I"})

    # Final proxies:  HOMO is high when nucleophilic atoms and donor
    # EState are present; LUMO is high when electronegative atoms are
    # present.  The signs/offsets don't matter, only the correlations
    # across molecules (the model picks the scale).
    HOMO_proxy = 0.5 * max_e + 0.5 * (n_O + 1.5 * n_N) - 0.2 * n_X
    LUMO_proxy = 0.5 * min_e + 0.3 * mean_en - 0.3 * (n_O + n_N)
    return {
        "HOMO_proxy": HOMO_proxy,
        "LUMO_proxy": LUMO_proxy,
        "HL_gap_proxy": HOMO_proxy - LUMO_proxy,
        "MaxEStateIndex": max_e,
        "MinEStateIndex": min_e,
        "MeanEStateIndex": mean_e,
        "mean_electronegativity": mean_en,
    }


def proxy_dipole(mol: Chem.Mol) -> dict:
    """Dipole moment proxy: |sum of charge * position| approximated by
    Gasteiger charge displacement."""
    f = gasteiger_features(mol)
    return {
        "dipole_proxy": f["gast_sum_abs"],  # higher = more polar
        "polarizability_proxy": f["gast_range"],
    }


def basic_counts(mol: Chem.Mol) -> dict:
    return {
        "n_O": count_atoms(mol, "O"),
        "n_N": count_atoms(mol, "N"),
        "n_F": count_atoms(mol, "F"),
        "n_S": count_atoms(mol, "S"),
        "n_P": count_atoms(mol, "P"),
        "n_X_halogen": sum(1 for a in mol.GetAtoms()
                           if a.GetSymbol() in {"F", "Cl", "Br", "I"}),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "n_heavy": mol.GetNumHeavyAtoms(),
    }


def safe_desc(mol: Chem.Mol, name: str, fn) -> float:
    try:
        v = fn(mol)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Main pipeline.
# --------------------------------------------------------------------------- #
def compute_all(mol: Chem.Mol) -> dict:
    row = basic_counts(mol)
    row.update(proxy_homo_lumo(mol))
    row.update(proxy_dipole(mol))
    row.update(get_chi_indices(mol))
    for name, fn in DESC_LIST:
        row[name] = safe_desc(mol, name, fn)
    return row


def main() -> None:
    lib = pd.read_csv(DATA_DIR / "candidate_library.csv")
    log.info("Loaded %d molecules", len(lib))

    out_rows = []
    failed = 0
    for i, smi in enumerate(lib["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            failed += 1
            continue
        row = compute_all(mol)
        row["mol_id"] = int(lib["mol_id"].iloc[i])
        row["smiles"] = smi
        out_rows.append(row)
        if (i + 1) % 500 == 0:
            log.info("  processed %d / %d", i + 1, len(lib))

    df = pd.DataFrame(out_rows)
    # Re-order: id and smiles first
    front = ["mol_id", "smiles"]
    df = df[front + [c for c in df.columns if c not in front]]
    out = DATA_DIR / "descriptors.csv"
    df.to_csv(out, index=False)
    log.info("Wrote %s: %d rows x %d columns (failed %d)",
             out, len(df), df.shape[1], failed)
    print("\n--- descriptors summary ---")
    print("rows:", len(df))
    print("cols:", df.shape[1])
    print("\nkey paper features (means):")
    for k in ("HOMO_proxy", "LUMO_proxy", "HL_gap_proxy",
              "dipole_proxy", "polarizability_proxy",
              "TPSA", "MolLogP", "MolWt",
              "NumHAcceptors", "NumHDonors",
              "Chi0v0", "Chi1v0", "Kappa1", "n_O", "n_N", "n_F"):
        if k in df.columns:
            print(f"  {k:24s}  mean={df[k].mean():.3f}  std={df[k].std():.3f}")


if __name__ == "__main__":
    main()
