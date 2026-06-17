"""Step 1: Build the candidate library of O/N/F Lewis-basic molecules.

Strategy
--------
- Use a hand-curated set of core scaffolds (with O/N/F atoms) plus a
  pool of substituent SMILES.  Assemble by concatenation (single
  bond) and RDKit MolFromSmiles -> MolToSmiles to canonicalize.
- Apply RDKit sanitization to drop invalid molecules.
- Filter by molecular weight, atom counts, and absence of toxic/
  hypervalent patterns.  Aim for 2000+ unique molecules.
- Save the library to data/candidate_library.csv with columns
  (mol_id, smiles, source, is_anchor).
- Anchors from the public literature are merged in (so they get a
  real experimental DN value attached downstream).

Output: data/candidate_library.csv, data/dn_anchor_table.csv (copy)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, get_logger, set_global_seed  # noqa: E402

RDLogger.DisableLog("rdApp.*")

set_global_seed(42)
log = get_logger("build_library")


# --------------------------------------------------------------------------- #
# Building blocks.
# --------------------------------------------------------------------------- #
# Each tuple is a SMILES of a *valid* fragment containing O/N/F as the
# coordination site.  We attach one of the "tail" groups via a single
# bond to create new molecules.

CORE_FRAGMENTS = [
    # carbonyl/ester O
    "C=O", "O=C", "C(=O)O", "C(=O)N", "C(=O)C", "O=C(O)C", "O=C(N)C",
    "C(=O)OC", "O=C(OC)C", "O=C(NC)C", "O=C(C)C", "C(=O)NC", "C(=O)OCC",
    "C(=O)OCCC", "O=C1OCC1", "O=C1NC(=O)NC1", "O=C1CCCC1",
    # ether O
    "CO", "CCO", "COC", "CCOC", "OCC", "OCO", "COCO", "OCCO",
    "OCCCO", "OCCOCC", "C1CCOC1", "C1CCOCC1", "C1COCCO1", "C1OCCO1",
    "C1COCC1", "C1COC1", "C1CC(=O)OC1", "C1OCOCO1",
    # amine N
    "CN", "NC", "CCN", "NCC", "CN(C)C", "CNC", "NCCN", "NCCCCN",
    "C1CCNC1", "C1CCNCC1", "C1CNCCN1", "C1COCCN1", "C1NCC1",
    "C1=NC=CN=C1", "c1ccncc1", "c1cnccn1", "c1cncnc1",
    "c1ccsc1", "C1CCSC1", "c1cc[nH]c1", "c1ccc2[nH]ccc2c1",
    "C1=NC2=NC=NC=C2N1", "c1ccc2nccnc2c1",
    # amide N/O
    "NC=O", "N(C)C=O", "C(=O)NCC", "C(=O)NCCC", "C(=O)N(C)C",
    "C(=O)N1CCCC1", "C(=O)N1CC1", "C(=O)NCCO",
    # nitrile N
    "C#N", "CC#N", "CCC#N", "C#CC", "CC#CC", "N#CCCC", "C#CCC#N",
    # F-containing
    "CF", "C(F)(F)F", "C(F)C", "C(F)(F)C", "OC(F)(F)F", "CC(F)(F)F",
    "C(F)(F)OC", "C(F)(F)O", "C(F)Cl", "FCF", "FC(F)F", "FC(=O)",
    "FCC", "FCCl", "FCBr", "FC(c1ccccc1)", "C(F)(F)(F)c1ccccc1",
    # nitro / sulfonyl
    "C[N+](=O)[O-]", "C(S(=O)(=O)C)", "C(S(=O)(=O)F)",
    "OS(=O)(=O)C", "NS(=O)(=O)C",
    # phosphine/phosphate
    "P", "OP", "O=P", "COP(=O)OC", "P(=O)(OC)OC",
    "N(C)P(=O)(N(C)C)N(C)C",  # HMPA proxy
]

# Tail groups that are grafted to a free valence.
TAIL_GROUPS = [
    "", "C", "CC", "CCC", "CCCC", "C(C)C", "C(C)(C)C",
    "O", "OC", "OCC", "OCCC", "OC(C)C",
    "N", "NC", "NCC", "N(C)C", "N1CC1",
    "F", "Cl",
    "C=O", "C(=O)C", "C(=O)O", "C(=O)N", "C(=O)NC",
    "C#N", "S", "SC", "S(=O)C", "S(=O)(=O)C",
    "c1ccccc1", "c1ccncc1", "c1cnccn1",
    "OC1CC1", "C1CC1", "C1CCOC1", "C1CCNC1",
    "C(O)CC", "C(O)O", "C(O)C",
    "P(=O)(OC)OC",
]

# Fragments that look dangerous or hypervalent -> reject.
FORBIDDEN_SUBSTRINGS = [
    "[N+](=O)=O",  # extra valence
    "OO",         # peroxide
    "SS",         # disulfide
    "C=C=C",      # allene
    "C#C#C",      # cumulene
]


def is_valid(mol: Chem.Mol) -> bool:
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    smi = Chem.MolToSmiles(mol)
    if any(p in smi for p in FORBIDDEN_SUBSTRINGS):
        return False
    return True


def has_lewis_base(mol: Chem.Mol) -> bool:
    """At least one O, N, or F atom in the molecule."""
    atoms = {a.GetSymbol() for a in mol.GetAtoms()}
    return bool(atoms & {"O", "N", "F"})


def passes_filter(mol: Chem.Mol) -> bool:
    if not is_valid(mol):
        return False
    if not has_lewis_base(mol):
        return False
    mw = Descriptors.MolWt(mol)
    if not (40.0 <= mw <= 280.0):
        return False
    # Disallow extreme oxygen count
    n_O = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    n_N = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    n_F = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F")
    if n_O + n_N + n_F > 8:
        return False
    if n_O + n_N + n_F < 1:
        return False
    return True


def assemble_mol(core: str, tail: str) -> Chem.Mol | None:
    """Concatenate core+tail as a SMILES string and parse.

    We insert a single bond 'C' in between if both fragments end/start
    with heavy atoms.  Easiest: just literal string concatenation,
    which RDKit interprets as a default single bond.
    """
    raw = core + tail
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return None
    return mol


def main() -> None:
    seen: dict[str, str] = {}  # canonical_smiles -> source

    # 1.  Plain cores (tail = "")
    for core in CORE_FRAGMENTS:
        mol = assemble_mol(core, "")
        if mol is None:
            continue
        if not passes_filter(mol):
            continue
        smi = Chem.MolToSmiles(mol)
        seen.setdefault(smi, f"core:{core}")

    # 2.  Core + tail combinations.
    for core in CORE_FRAGMENTS:
        for tail in TAIL_GROUPS:
            if tail == "":
                continue
            mol = assemble_mol(core, tail)
            if mol is None:
                continue
            if not passes_filter(mol):
                continue
            smi = Chem.MolToSmiles(mol)
            seen.setdefault(smi, f"core+tail:{core}+{tail}")

    # 3.  core+tail+tail2 to add diversity
    EXTRA_TAILS = ["C", "CC", "O", "N", "F", "C=O", "C#N", "OC"]
    for core in CORE_FRAGMENTS[:30]:
        for t1 in TAIL_GROUPS[:6]:
            for t2 in EXTRA_TAILS:
                mol = assemble_mol(core, t1 + t2)
                if mol is None or not passes_filter(mol):
                    continue
                smi = Chem.MolToSmiles(mol)
                seen.setdefault(smi, "core+t1+t2")

    log.info("Generated %d unique candidate SMILES", len(seen))

    # 4.  Merge in the literature anchors (these are validated molecules
    # that have an experimental DN value).
    anchor_path = DATA_DIR / "dn_anchor_table.csv"
    anchor_df = pd.read_csv(anchor_path)
    for smi, name in zip(anchor_df["smiles"], anchor_df["name"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not passes_filter(mol):
            log.warning("Anchor %s failed filter (%s)", name, smi)
            continue
        canon = Chem.MolToSmiles(mol)
        seen.setdefault(canon, f"anchor:{name}")

    log.info("After adding anchors: %d unique SMILES", len(seen))

    # 5.  Build the dataframe.
    rows = []
    for i, (smi, src) in enumerate(seen.items()):
        is_anchor = src.startswith("anchor:")
        rows.append({"mol_id": i, "smiles": smi, "source": src, "is_anchor": is_anchor})
    df = pd.DataFrame(rows)
    out = DATA_DIR / "candidate_library.csv"
    df.to_csv(out, index=False)
    log.info("Wrote %s with %d rows", out, len(df))

    # Quick distribution check
    print("\n--- library summary ---")
    print("total molecules:", len(df))
    print("anchor molecules:", int(df["is_anchor"].sum()))
    print("\nsource distribution (top 10):")
    print(df["source"].value_counts().head(10))


if __name__ == "__main__":
    main()
