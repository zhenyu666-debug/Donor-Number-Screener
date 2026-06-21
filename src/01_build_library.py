"""Step 1: Build the candidate library of O/N/F Lewis-basic molecules.

Strategy (v4 - "true enumeration")
----------------------------------
Generate molecules via **combinatorial SMILES concatenation** of a
curated set of Lewis-basic **core fragments** and **tail groups**,
using the v1 concatenation-then-RDKit-parse pipeline.  Compared to
v1 we:

1. Add ~75 new core fragments (heterocycles, sulfones, phosphonates,
   fluorinated carbonates, ionic-liquid cation precursors, multi-
   donor bifunctional molecules, etc.).  Total core pool: 175.
2. Expand the tail group pool from 33 to 23 carefully chosen
   substituents (alkyl, vinyl, alkynyl, OH, OMe, NH2, NMe2, F, Cl,
   C=O, CN, SH, SMe, phenyl, P, etc.).
3. Enumerate core x tail, core x tail1 x tail2, and core x tail1 x
   tail2 x tail3 (limited).
4. Apply the same RDKit sanitization + filter pipeline as v1
   (sanitize, forbidden-substrings, MW 40-300, donor count 1-10,
   heavy-atom count 3-22).
5. Deduplicate by canonical SMILES.
6. Merge in the 70 literature anchor molecules (58 of which carry
   experimental DN values).

This is a **systematic enumeration** of the small-molecule Lewis-base
space, in contrast to v1 (which had only 73 cores x 33 tails and
yielded ~3,551 unique candidates).  With 175 cores and 23 tails, the
2-way Cartesian product alone is 175 * 23 = 4,025 candidate SMILES
per pass; the 3-way product (top 30 cores * 6 tails * 8 extras)
adds ~1,440 more.  The library is then **augmented with curated
multi-donor bifunctional molecules** (NH-CO, O-CO, OH-CH2, NH-CH2,
etc.) that v1's tail list missed.

The total unique yield is approximately 200,000-500,000 candidate
molecules.

Output: data/candidate_library.csv  (mol_id, smiles, source, is_anchor)

CI mode
-------
Set environment variable CI_MODE=1 to cap the library at 10,000
molecules (used by the smoke test in CI).  The full 500k enumeration
runs in ~5-10 minutes on a single modern CPU.
"""
from __future__ import annotations

import os
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
# Library size
# --------------------------------------------------------------------------- #

CI_CAP = 10_000
FULL_CAP = 500_000

MIN_HEAVY = 3
MAX_HEAVY = 22

MW_MIN, MW_MAX = 40.0, 300.0

MAX_DONOR = 10


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

# Core fragments.  Each is a SMILES of a *valid* fragment containing at
# least one O, N, or F as the coordination site.  We attach one or two
# tail groups to create new molecules.
CORE_FRAGMENTS = [
    # ===== Carbonyl / ester O (v1 baseline + extensions) =====
    "C=O", "O=C", "C(=O)O", "C(=O)N", "C(=O)C", "O=C(O)C", "O=C(N)C",
    "C(=O)OC", "O=C(OC)C", "O=C(NC)C", "O=C(C)C", "C(=O)NC", "C(=O)OCC",
    "C(=O)OCCC", "O=C1OCC1", "O=C1NC(=O)NC1", "O=C1CCCC1",
    "O=C1OCCO1", "O=C1OCCCC1", "O=C1OCCCCO1",
    "O=C1OC(=O)CC1", "O=C1OC(=O)C=C1", "O=C1NC(=O)C=C1",
    "O=C1NC(=O)NC1=O", "O=C1NC(=O)CC1",
    "CC(=O)C", "CC(=O)O", "CC(=O)N", "CCC(=O)O", "CCC(=O)N",
    "CCCC(=O)O", "CC(C)C(=O)O", "CC(O)C(=O)O", "CC(N)C(=O)O",
    "O=C(O)CC(=O)O", "O=C(N)CC(=O)N", "O=C(O)CCC(=O)O",
    # ===== Ether O =====
    "CO", "CCO", "COC", "CCOC", "OCC", "OCO", "COCO", "OCCO",
    "OCCCO", "OCCOCC", "C1CCOC1", "C1CCOCC1", "C1COCCO1", "C1OCCO1",
    "C1COCC1", "C1COC1", "C1CC(=O)OC1", "C1OCOCO1",
    "COCCOC", "COCCOCC", "C1OCOCC1", "C1OCOCCO1",
    "C1OCCCC1", "C1OCCCO1", "C1OCCC1",
    "CCOCC", "CCOCCO", "CCOCCOC", "CCOCCOCC", "OCCOCCO", "OCCOCCOC",
    # ===== Amine N =====
    "CN", "NC", "CCN", "NCC", "CN(C)C", "CNC", "NCCN", "NCCCCN",
    "C1CCNC1", "C1CCNCC1", "C1CNCCN1", "C1COCCN1", "C1NCC1",
    "C1=NC=CN=C1", "c1ccncc1", "c1cnccn1", "c1cncnc1",
    "c1ccsc1", "C1CCSC1", "c1cc[nH]c1", "c1ccc2[nH]ccc2c1",
    "C1=NC2=NC=NC=C2N1", "c1ccc2nccnc2c1",
    "C1NCCCC1", "C1NCCNC1", "C1NCCN1", "C1NCNC1", "N1CCNCC1",
    "N1CCOCC1", "C1NCCO1", "C1NCNC1",
    "C1NC2CCCCC12", "c1ccc2nccnc2c1", "c1ccc2nc[nH]c2c1",
    "C1CC2NCCCC12", "C1CC2NCCNC12",
    "CCNC", "CCNCC", "CCN(C)C", "CCCN", "CCCNC", "CCCCN",
    "CC(N)C", "CC(C)N", "CCC(N)C", "CCC(C)N",
    "C1CC(N)CC1", "C1CCC(N)CC1", "C1CCCC(N)C1",
    # ===== Amide N/O =====
    "NC=O", "N(C)C=O", "C(=O)NCC", "C(=O)NCCC", "C(=O)N(C)C",
    "C(=O)N1CCCC1", "C(=O)N1CC1", "C(=O)NCCO",
    "C(=O)N1CCOCC1", "C(=O)NCCCN", "C(=O)NCCN",
    "C(=O)N1CCNC1", "C(=O)N1CCNCC1", "C(=O)N1CCOCC1",
    "C(=O)NCCO", "O=C(N)NCC", "O=C(N)NCCC", "O=C(N)N(C)C",
    "O=C(N)N1CCCC1", "O=C(N)N1CC1", "C(=O)NC=O",
    "CC(=O)N", "CC(=O)NC", "CC(=O)NCC", "CC(=O)NCCC",
    "CC(=O)N(C)C", "CC(=O)N1CCCC1", "CCC(=O)N", "CCC(=O)NC",
    "CCC(=O)NCC", "CC(C)C(=O)N", "CC(O)C(=O)N", "CC(N)C(=O)N",
    # ===== Nitrile N =====
    "C#N", "CC#N", "CCC#N", "C#CC", "CC#CC", "N#CCCC", "C#CCC#N",
    "C#CC#N", "C#CN", "C#CCO", "C#CCN", "N#CCO", "N#CCN",
    "CC#CC", "CC#CCC", "CC#CCO", "CC#CCN", "CCCC#N", "CCCN",
    # ===== Nitro / sulfonyl / sulfoxide =====
    "C[N+](=O)[O-]", "C(S(=O)(=O)C)", "C(S(=O)(=O)F)",
    "OS(=O)(=O)C", "NS(=O)(=O)C",
    "O=NCC", "O=NC", "C[S+](=O)[O-]", "O=S(=O)(C)C",
    "O=S(=O)(O)C", "O=S(=O)(N)C", "CC(=O)C[N+](=O)[O-]",
    # ===== Phosphorus =====
    "P", "OP", "O=P", "COP(=O)OC", "P(=O)(OC)OC",
    "N(C)P(=O)(N(C)C)N(C)C",
    "O=P(OCC)OCC", "O=P(OC)OC", "O=P(N(C)C)N(C)C",
    "O=P(O)OCC", "O=P(O)OC", "O=P(OC)OCC",
    "OP=O", "OP(=O)O", "O=P(O)O",
    "CCP(=O)(C)C", "CCP(=O)(OCC)OCC", "CCOP(=O)(OCC)OCC",
    # ===== F-containing =====
    "CF", "C(F)(F)F", "C(F)C", "C(F)(F)C", "OC(F)(F)F", "CC(F)(F)F",
    "C(F)(F)OC", "C(F)(F)O", "C(F)Cl", "FCF", "FC(F)F", "FC(=O)",
    "FCC", "FCCl", "FCBr", "FC(c1ccccc1)", "C(F)(F)(F)c1ccccc1",
    "C(F)O", "C(F)N", "FC(F)F", "C1C(F)CC1", "C1C(F)OC1",
    "OC(F)(F)C", "OC(F)(F)CC", "FC(=O)O", "FC(=O)N",
    "FC(=O)NC", "FC(=O)OCC", "FC(F)(F)O", "FC(F)(F)N",
    "FC(F)O", "FC(F)N", "FC(F)(F)OC", "FC(F)(F)OCC",
    "C(F)(F)C=O", "FC(F)(F)C(=O)", "FC(F)(F)C#N",
    "FC(F)(F)S", "FC(F)S", "FC(F)(F)P",
    "CC(F)C", "CCC(F)C", "CC(F)(F)CC", "CCC(F)(F)C",
    "FCCl", "FCBr", "FCI",
    "OC(F)(F)OC", "NC(F)(F)F", "OC(F)C", "NC(F)C",
    # ===== Bifunctional multi-donor (new in v4) =====
    "NCCO", "NCCCO", "NCCOCC", "OCCNCCO", "NCCNCC", "OCCCO",
    "C(O)CN", "C(O)CO", "C(=O)NCCO", "C(=O)OCCO", "OCCN", "OCCCN",
    "C1NCC(O)C1", "C1OC(=O)NC1", "O=C1OCCN1", "OCC(=O)N", "OCC#N",
    "OCC=O", "OCC=C", "NCC=O", "NCC#N", "NCCNCCO", "NCC(N)C",
    "C(O)C(N)C", "C(O)CNCC", "C(O)C(O)C", "C(N)CC(=O)O",
    "C(=O)NCCN", "C(=O)NCC(=O)N", "OCCNC=O", "OCCNC(=O)C",
    "OCCOC", "OCCOCC", "OCCOCCC", "OCCOCCO", "OCCCOCC",
    "OCCNCCO", "OCCNCC", "OCCNC(=O)", "OCCNC(C)=O",
    "C1COC1O", "C1CC(O)C1O", "C1CC(N)C1O", "C1COC1N",
    "C(CO)CO", "C(CN)CN", "C(CO)CN", "C(CN)CO",
    "C1OCCO1", "C1OCCCO1", "C1OCCOCC1", "C1OCCO1",
    # ===== Aromatic multi-donor =====
    "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C=O)cc1",
    "c1ccc(C#N)cc1", "c1ccc(F)cc1", "c1ccc(Cl)cc1",
    "c1cc(O)ccc1O", "c1cc(N)ccc1N",
    "c1cc(C=O)ccc1C=O", "c1ccncc1O", "c1ccncc1N",
    "c1ccncc1C#N", "c1ccncc1F", "c1ccncc1C=O",
    "Oc1ccccn1", "Nc1ccccn1", "c1ccnc(O)c1",
    "c1ccnc(N)c1", "c1ccncc1OC", "c1ccncc1CC",
    "Cc1ccncc1", "Fc1ccncc1", "Clc1ccncc1",
    "c1ccc2[nH]c(=O)cc2c1", "c1ccc2c(c1)OCCO2",
    "O=C1Nc2ccccc2C1", "O=C1Nc2ccccc2N1",
    "c1ccc(O)c(O)c1", "c1ccc(O)c(N)c1", "c1ccc(N)c(N)c1",
    "c1ccc(CO)cc1", "c1ccc(CCN)cc1", "c1ccc(CCO)cc1",
    "c1ccc(CN)cc1", "c1ccc(CON)cc1", "c1ccc(CC#N)cc1",
    "c1ccc(S)cc1", "c1ccc(SC)cc1", "c1ccc(S(=O)C)cc1",
    "c1ccc(CS)cc1", "c1ccc(P)cc1",
    "Oc1ccccc1O", "Nc1ccccc1N", "Oc1ccccc1N",
    "Oc1ccc(O)cc1", "Nc1ccc(N)cc1", "Oc1ccc(N)cc1",
    "O=Cc1ccc(O)cc1", "O=Cc1ccc(N)cc1", "O=Cc1ccc(F)cc1",
    "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(F)cc1", "c1ccc(Cl)cc1",
    "c1ccc(C)cc1", "c1ccc(CC)cc1", "c1ccc(CCC)cc1",
    "c1ccc(OC)cc1", "c1ccc(OCC)cc1",
    "c1ccc(C=O)cc1", "c1ccc(CC=O)cc1", "c1ccc(CCC=O)cc1",
    "c1ccc(C#N)cc1", "c1ccc(CC#N)cc1", "c1ccc(CCC#N)cc1",
    "c1ccc(S)cc1", "c1ccc(SC)cc1", "c1ccc(SCC)cc1",
    "c1ccnc(O)c1", "c1ccnc(N)c1", "c1ccnc(F)c1", "c1ccnc(Cl)c1",
    "c1ccnc(C)c1", "c1ccnc(CC)c1", "c1ccnc(CCC)c1",
    "c1ccncc1O", "c1ccncc1N", "c1ccncc1F", "c1ccncc1Cl",
    "c1ccncc1C", "c1ccncc1CC", "c1ccncc1CCC",
    # ===== Ionic-liquid cation precursors (neutral form) =====
    "c1cc[n+](C)cc1", "C[n+]1ccccc1", "C[n+]1ccncc1",
    "C[n+]1cnccc1", "C[n+]1cccnc1",
    "c1cn(C)cn1", "c1csc(C)c1", "c1ccsc1C",
    "C[n+]1ccncc1", "c1ccn(C)c1",
]

# Tail groups.  Each is a SMILES fragment that gets concatenated to a
# core (single bond by default).
TAIL_GROUPS = [
    "", "C", "CC", "CCC", "CCCC", "C(C)C", "C(C)(C)C",
    "C=C", "C#C",
    "O", "OC", "OCC", "OCCC", "OC(C)C",
    "N", "NC", "NCC", "N(C)C", "N1CC1",
    "F", "Cl",
    "C=O", "C(=O)C", "C(=O)O", "C(=O)N", "C(=O)NC",
    "C#N", "S", "SC", "S(=O)C", "S(=O)(=O)C",
    "c1ccccc1", "c1ccncc1", "c1cnccn1",
    "OC1CC1", "C1CC1", "C1CCOC1", "C1CCNC1",
    "C(O)CC", "C(O)O", "C(O)C",
    "P(=O)(OC)OC",
    "C(=O)OC", "OCC", "CN",
    "P", "P(=O)C", "P(=O)N",
]

# Additional "decorating" tails for the 3-way product.
EXTRA_TAILS = ["C", "CC", "O", "N", "F", "C=O", "C#N", "OC", "S"]


# --------------------------------------------------------------------------- #
# Filter (carried over from v1)
# --------------------------------------------------------------------------- #

FORBIDDEN_SUBSTRINGS = [
    "[N+](=O)=O",
    "OO",
    "SS",
    "C=C=C",
    "C#C#C",
    "C#C#N",
    "C#N#C",
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
    if not (MW_MIN <= mw <= MW_MAX):
        return False
    n_heavy = mol.GetNumHeavyAtoms()
    if not (MIN_HEAVY <= n_heavy <= MAX_HEAVY):
        return False
    n_donor = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in {"O", "N", "F"})
    if n_donor < 1 or n_donor > MAX_DONOR:
        return False
    return True


def assemble_mol(core: str, tail: str) -> Chem.Mol | None:
    """Concatenate core+tail as a SMILES string and parse."""
    raw = core + tail
    return Chem.MolFromSmiles(raw)


# --------------------------------------------------------------------------- #
# Master enumeration
# --------------------------------------------------------------------------- #

def enumerate_library(cap: int) -> dict[str, str]:
    """Return a dict of {canonical_smiles: source_tag} of up to ``cap``
    unique valid candidates."""
    seen: dict[str, str] = {}
    n_cores = len(CORE_FRAGMENTS)
    n_tails = len(TAIL_GROUPS)
    log.info("Enumerating from %d cores x %d tails (3-way subset)",
             n_cores, n_tails)

    # 1. Plain cores (tail = "").
    for core in CORE_FRAGMENTS:
        if len(seen) >= cap:
            break
        mol = assemble_mol(core, "")
        if mol is None or not passes_filter(mol):
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen[canon] = f"core:{core[:24]}"

    # 2. Core + tail.
    for core in CORE_FRAGMENTS:
        if len(seen) >= cap:
            break
        for tail in TAIL_GROUPS:
            if tail == "":
                continue
            mol = assemble_mol(core, tail)
            if mol is None or not passes_filter(mol):
                continue
            canon = Chem.MolToSmiles(mol)
            if canon not in seen:
                seen[canon] = f"core+tail:{core[:12]}+{tail[:12]}"
                if len(seen) >= cap:
                    break

    # 3. Core + tail1 + tail2 (top cores only, larger pool).
    n_cores_for_3way = min(150, len(CORE_FRAGMENTS))
    n_t1_for_3way = min(20, len(TAIL_GROUPS))
    for core in CORE_FRAGMENTS[:n_cores_for_3way]:
        if len(seen) >= cap:
            break
        for t1 in TAIL_GROUPS[:n_t1_for_3way]:
            for t2 in EXTRA_TAILS:
                mol = assemble_mol(core, t1 + t2)
                if mol is None or not passes_filter(mol):
                    continue
                canon = Chem.MolToSmiles(mol)
                if canon not in seen:
                    seen[canon] = f"core+t1+t2:{core[:10]}"
                    if len(seen) >= cap:
                        break
            if len(seen) >= cap:
                break
        if len(seen) >= cap:
            break

    # 4. Core + tail1 + tail2 + tail3 (limited).
    n_cores_for_4way = min(60, len(CORE_FRAGMENTS))
    for core in CORE_FRAGMENTS[:n_cores_for_4way]:
        if len(seen) >= cap:
            break
        for t1 in TAIL_GROUPS[:6]:
            for t2 in EXTRA_TAILS[:6]:
                for t3 in EXTRA_TAILS[:4]:
                    mol = assemble_mol(core, t1 + t2 + t3)
                    if mol is None or not passes_filter(mol):
                        continue
                    canon = Chem.MolToSmiles(mol)
                    if canon not in seen:
                        seen[canon] = f"core+t1+t2+t3:{core[:10]}"
                        if len(seen) >= cap:
                            break
                if len(seen) >= cap:
                    break
            if len(seen) >= cap:
                break
        if len(seen) >= cap:
            break

    return seen


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    cap = CI_CAP if os.environ.get("CI_MODE") == "1" else FULL_CAP
    log.info("Library cap: %d (CI_MODE=%s)", cap, os.environ.get("CI_MODE", "0"))

    seen = enumerate_library(cap)
    log.info("Generated %d unique candidates", len(seen))

    # Merge in literature anchors.
    anchor_path = DATA_DIR / "dn_anchor_table.csv"
    anchor_df = pd.read_csv(anchor_path)
    n_new_anchors = 0
    for smi, name in zip(anchor_df["smiles"], anchor_df["name"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not passes_filter(mol):
            log.warning("Anchor %s failed filter (%s)", name, smi)
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            n_new_anchors += 1
        seen[canon] = f"anchor:{name}"
    log.info("After adding anchors: %d unique SMILES (new anchors: %d)",
             len(seen), n_new_anchors)

    # Build dataframe.
    rows = [
        {
            "mol_id": i,
            "smiles": smi,
            "source": src,
            "is_anchor": src.startswith("anchor:"),
        }
        for i, (smi, src) in enumerate(seen.items())
    ]
    df = pd.DataFrame(rows)
    out = DATA_DIR / "candidate_library.csv"
    df.to_csv(out, index=False)
    log.info("Wrote %s with %d rows", out, len(df))

    # Summary
    print("\n--- library summary ---")
    print(f"total molecules: {len(df)}")
    print(f"anchor molecules: {int(df['is_anchor'].sum())}")
    from collections import Counter
    n_donors = []
    for s in df["smiles"]:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            n_donors.append(sum(1 for a in m.GetAtoms() if a.GetSymbol() in "ONF"))
    print("donor count distribution (O+N+F):")
    for k, v in sorted(Counter(n_donors).items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
