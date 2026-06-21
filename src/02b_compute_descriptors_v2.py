"""Step 2b (v2 descriptors): 升级版描述符计算。

在 `02_compute_descriptors.py` 的 236 维基础上,追加:
  - Morgan 指纹 (radius=2, nBits=512)        -> 512 维
  - MACCS keys                                 -> 167 维
  - RDKit 额外 `fr_*` 计数                    -> 11 维
  - Estate fingerprint (rdkit.Chem.EState.Fingerprint)  -> 79 维
---------------------------------------------------------------
合计 v2 描述符约 1005 维(原始 236 + 新增 769)。

性能:用 joblib.Parallel 多核并行,目标 < 5 秒(v1 约 20 秒)。

输出:data/descriptors_v2.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, get_logger, set_global_seed  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
set_global_seed(42)
log = get_logger("descriptors_v2")

# Additional RDKit fragment counts (fr_* family).
EXTRA_FR_DESCS = [
    "fr_Al_COO", "fr_Al_OH", "fr_Ar_NH", "fr_Ar_OH",
    "fr_COO", "fr_C_O", "fr_NH0", "fr_NH1", "fr_NH2",
    "fr_N_O", "fr_amide",
]

DESC_LIST = [(name, func) for name, func in Descriptors.descList
             if name not in EXTRA_FR_DESCS]
DESC_LIST += [(name, getattr(Descriptors, name))
              for name in EXTRA_FR_DESCS
              if hasattr(Descriptors, name)]

# Module-level (picklable) name list for worker processes.
DESC_NAMES = [name for name, _ in DESC_LIST]


def count_atoms(mol, symbol):
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == symbol)


def gasteiger_features(mol):
    try:
        mol2 = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(mol2, throwOnParamFailure=False)
        ch = np.array([
            float(a.GetProp("_GasteigerCharge"))
            if a.HasProp("_GasteigerCharge") else 0.0
            for a in mol2.GetAtoms()
        ])
        if not np.isfinite(ch).all():
            ch = np.nan_to_num(ch, nan=0.0)
    except Exception:
        return {f"gast_{k}": 0.0 for k in
                ("sum_abs", "mean", "max", "min", "range", "std")}
    return {
        "gast_sum_abs": float(np.sum(np.abs(ch))),
        "gast_mean":    float(np.mean(ch)),
        "gast_max":     float(np.max(ch)),
        "gast_min":     float(np.min(ch)),
        "gast_range":   float(np.ptp(ch)),
        "gast_std":     float(np.std(ch)),
    }


def get_chi_indices(mol):
    from rdkit.Chem import rdMolDescriptors
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


def proxy_homo_lumo(mol):
    ELECTRONEGATIVITY = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44,
                         "F": 3.98, "P": 2.19, "S": 2.58, "Cl": 3.16}
    try:
        from rdkit.Chem.EState import EStateIndices
        es = EStateIndices(mol)
        max_e, min_e, mean_e = (float(np.max(es)),
                                float(np.min(es)),
                                float(np.mean(es)))
    except Exception:
        max_e = min_e = mean_e = 0.0
    en_vals = [ELECTRONEGATIVITY.get(a.GetSymbol(), 2.5) for a in mol.GetAtoms()
               if a.GetSymbol() != "H"]
    mean_en = float(np.mean(en_vals)) if en_vals else 0.0
    _n_F, n_O, n_N = (count_atoms(mol, "F"),
                     count_atoms(mol, "O"),
                     count_atoms(mol, "N"))
    n_X = sum(1 for a in mol.GetAtoms()
              if a.GetSymbol() in {"F", "Cl", "Br", "I"})
    HOMO = 0.5 * max_e + 0.5 * (n_O + 1.5 * n_N) - 0.2 * n_X
    LUMO = 0.5 * min_e + 0.3 * mean_en - 0.3 * (n_O + n_N)
    return {"HOMO_proxy": HOMO, "LUMO_proxy": LUMO,
            "HL_gap_proxy": HOMO - LUMO,
            "MaxEStateIndex": max_e, "MinEStateIndex": min_e,
            "MeanEStateIndex": mean_e,
            "mean_electronegativity": mean_en}


def proxy_dipole(mol):
    f = gasteiger_features(mol)
    return {"dipole_proxy": f["gast_sum_abs"],
            "polarizability_proxy": f["gast_range"]}


def basic_counts(mol):
    from rdkit.Chem import rdMolDescriptors
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


def safe_desc(mol, name, fn):
    try:
        v = fn(mol)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


# ----- New: Morgan, MACCS, Estate FP ----- #

def morgan_fp(mol, n_bits=512, radius=2):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def maccs_keys(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def estate_fp(mol, n_bits=79):
    """Estate fingerprint (EState_VSA-like index vector)."""
    try:
        from rdkit.Chem.EState import EStateIndices
        es = EStateIndices(mol)
        if len(es) != n_bits:
            return np.zeros(n_bits, dtype=np.float32)
        return es.astype(np.float32)
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


# ----- Single-molecule compute ----- #

def _desc_lookup():
    """Worker-side lookup: avoid pickling the bound function objects."""
    return {n: getattr(Descriptors, n) for n in DESC_NAMES
            if hasattr(Descriptors, n)}


def compute_v2(smiles: str) -> dict | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    row = basic_counts(mol)
    row.update(proxy_homo_lumo(mol))
    row.update(proxy_dipole(mol))
    row.update(get_chi_indices(mol))
    for name, fn in _desc_lookup().items():
        row[name] = safe_desc(mol, name, fn)

    # Fingerprints
    morgan = morgan_fp(mol)
    for i, v in enumerate(morgan):
        row[f"morgan_{i}"] = int(v)
    maccs = maccs_keys(mol)
    for i, v in enumerate(maccs):
        row[f"maccs_{i}"] = int(v)
    es = estate_fp(mol)
    for i, v in enumerate(es):
        row[f"estate_{i}"] = float(v)
    return row


def compute_v2_with_id(args):
    i, mol_id, smi = args
    r = compute_v2(smi)
    if r is None:
        return None
    r["mol_id"] = int(mol_id)
    r["smiles"] = smi
    return r


def main():
    t0 = time.perf_counter()
    lib = pd.read_csv(DATA_DIR / "candidate_library.csv")
    log.info("Loaded %d molecules", len(lib))

    # Build arg list and run in parallel.
    args_list = [(i, int(lib["mol_id"].iloc[i]), smi)
                 for i, smi in enumerate(lib["smiles"])]

    n_jobs = -1  # use all cores
    log.info("Computing v2 descriptors with n_jobs=%d ...", n_jobs)
    results = Parallel(n_jobs=n_jobs, verbose=5, batch_size=32)(
        delayed(compute_v2_with_id)(a) for a in args_list
    )

    # Drop failures.
    rows = [r for r in results if r is not None]
    log.info("Computed %d / %d molecules successfully",
             len(rows), len(args_list))

    df = pd.DataFrame(rows)
    front = ["mol_id", "smiles"]
    df = df[front + [c for c in df.columns if c not in front]]
    out = DATA_DIR / "descriptors_v2.csv"
    df.to_csv(out, index=False)
    elapsed = time.perf_counter() - t0

    log.info("Wrote %s: %d rows x %d columns in %.1fs",
             out, len(df), df.shape[1], elapsed)
    print("\n--- v2 descriptors summary ---")
    print(f"rows: {len(df)}")
    print(f"cols: {df.shape[1]}")
    print(f"v1 was 236 cols, v2 has {df.shape[1]} cols (delta +{df.shape[1] - 236})")
    print(f"wall time: {elapsed:.1f}s")
    print(f"per molecule: {elapsed*1000/len(df):.1f} ms")


if __name__ == "__main__":
    main()
