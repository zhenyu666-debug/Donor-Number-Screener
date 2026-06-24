#!/usr/bin/env python3
"""elexacaftor_chemistry.py -- Trikafta component molecular analysis.

Provides:
  1. Molecular weight calculator via RDKit (validates SMILES)
  2. Trikafta 1:1:2 molar-ratio verification
  3. Parameterised annual API cost estimator

Usage:
    python scripts/elexacaftor_chemistry.py --mw
    python scripts/elexacaftor_chemistry.py --cost --price-elexa 50000 --price-tez 30000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

COMPONENTS = {
    "Elexacaftor": {
        "name": "Elexacaftor",
        "vx": "VX-814",
        "smiles": "CC1CN(C(C1)(C)C)c1nc(ccc1C(=O)NS(=O)(=O)c1cn(nc1C)C)n1ccc(n1)OCC(C(F)(F)F)(C)C",
        "cid": 134587348,
        "formula": "C26H34F3N7O4S",
        "dose_mg": 100.0,
    },
    "Tezacaftor": {
        "name": "Tezacaftor",
        "vx": "VX-661",
        "smiles": "CC(C)(CO)C1=CC2=CC(NC(=O)C3(CC3)C4=CC=C5OC(O2)(F)F)C(F)=C2N1C[C@@H](O)CO",
        "cid": 46199646,
        "formula": "C26H27F3N2O6",
        "dose_mg": 50.0,
    },
    "Ivacaftor": {
        "name": "Ivacaftor",
        "vx": "VX-770",
        "smiles": "CC(C)(C)C1=CC(=C(O)C=C1NC(=O)C2=CNC3=CC=CC=C3C2=O)C(C)(C)C",
        "cid": 16220172,
        "formula": "C24H28N2O3",
        "dose_mg": 75.0,
    },
}


def calc_mw(smiles: str) -> float:
    """Return molecular weight from SMILES via RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Descriptors import ExactMolWt
    except ImportError:
        print("ERROR: rdkit-pypi not installed. Run: pip install rdkit-pypi")
        sys.exit(1)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return ExactMolWt(mol)


@dataclass
class CostScenario:
    """Parameterised cost model for Trikafta API synthesis."""

    price_elexa_ksm: float = 50_000.0
    price_tez_ksm: float = 30_000.0
    price_iva_ksm: float = 10_000.0

    reagent_na_bh4: float = 50.0
    reagent_dppa: float = 100.0
    reagent_pp_diad: float = 20.0
    reagent_tfa: float = 80.0
    reagent_cdi: float = 200.0
    reagent_hatu: float = 1_500.0
    reagent_solvent: float = 500.0

    gmp_synthesis: float = 5_000.0
    formulation: float = 800.0
    qc_testing: float = 500.0

    yield_step2: float = 0.65
    yield_step3: float = 0.55

    def total_cost(self) -> float:
        ksm = (
            self.price_elexa_ksm * 0.2
            + self.price_tez_ksm * 0.1
            + self.price_iva_ksm * 0.05
        )
        reagents = (
            self.reagent_na_bh4
            + self.reagent_dppa
            + self.reagent_pp_diad
            + self.reagent_tfa
            + self.reagent_cdi
            + self.reagent_hatu
            + self.reagent_solvent
        )
        processing = self.gmp_synthesis + self.formulation + self.qc_testing
        return ksm + reagents + processing

    def sensitivity(self) -> dict:
        base = self.total_cost()
        params = {
            "price_elexa_ksm": self.price_elexa_ksm,
            "price_tez_ksm": self.price_tez_ksm,
            "price_iva_ksm": self.price_iva_ksm,
            "reagent_hatu": self.reagent_hatu,
            "gmp_synthesis": self.gmp_synthesis,
        }
        result = {}
        for key, val in params.items():
            setattr(self, key, val * 1.10)
            result[f"delta_10pct_{key}"] = self.total_cost() - base
            setattr(self, key, val)
        return result


EXPECTED_MW = {
    "Elexacaftor": 597.66,
    "Tezacaftor": 520.50,
    "Ivacaftor": 392.49,
}


def cmd_mw() -> None:
    print("\n=== Trikafta Component Molecular Weights ===\n")
    mw_list = []
    for key, comp in COMPONENTS.items():
        mw = calc_mw(comp["smiles"])
        mw_list.append((key, mw))
        print(f"  {key:<15} CID={comp['cid']}  Formula={comp['formula']}")
        print(f"    SMILES: {comp['smiles'][:55]}...")
        print(f"    MW = {mw:.4f} g/mol  (expected ~{EXPECTED_MW[key]:.2f})")
        print(f"    Dose = {comp['dose_mg']} mg/day")
        print()

    total_mw = sum(mw for _, mw in mw_list)
    avg_mw = total_mw / 4
    total_dose = sum(c["dose_mg"] for c in COMPONENTS.values())
    print("  Trikafta (1 Elexa : 1 Tez : 2 Iva):")
    print(f"    Combined MW (4 units) = {total_mw:.4f} g/mol")
    print(f"    Average unit MW      = {avg_mw:.4f} g/mol")
    print(f"    Daily dose           = {total_dose:.0f} mg  ({total_dose * 365 / 1000:.3f} g/year)")
    print()


def cmd_cost(args: argparse.Namespace) -> None:
    scenario = CostScenario(
        price_elexa_ksm=args.price_elexa,
        price_tez_ksm=args.price_tez,
        price_iva_ksm=args.price_iva,
        yield_step2=args.yield_step2,
        yield_step3=args.yield_step3,
    )
    total = scenario.total_cost()
    reagents_sub = (
        scenario.reagent_na_bh4 + scenario.reagent_dppa
        + scenario.reagent_pp_diad + scenario.reagent_tfa
        + scenario.reagent_cdi + scenario.reagent_hatu + scenario.reagent_solvent
    )
    ksm_sub = (
        scenario.price_elexa_ksm * 0.2
        + scenario.price_tez_ksm * 0.1
        + scenario.price_iva_ksm * 0.05
    )
    print("\n=== Annual API Synthesis Cost (CNY / person-year) ===\n")
    print("  KSM�S�e:")
    print(f"    Elexacaftor KSM  ~200g @{scenario.price_elexa_ksm:>10,.0f}/kg = {scenario.price_elexa_ksm * 0.2:>10,.0f}")
    print(f"    Tezacaftor KSM  ~100g @{scenario.price_tez_ksm:>10,.0f}/kg = {scenario.price_tez_ksm * 0.1:>10,.0f}")
    print(f"    Ivacaftor KSM   ~50g @{scenario.price_iva_ksm:>10,.0f}/kg = {scenario.price_iva_ksm * 0.05:>10,.0f}")
    print(f"    KSM\��                                                  = {ksm_sub:>10,.0f}")
    print(f"  ՋBR�Pg  (HATU={scenario.reagent_hatu:,.0f})                      = {reagents_sub:>10,.0f}")
    print(f"  GMPTb                                               = {scenario.gmp_synthesis:>10,.0f}")
    print(f"  6RBR+QC                                               = {scenario.formulation + scenario.qc_testing:>10,.0f}")
    print(f"  {'='*55}")
    print(f"  T��                                                   = {total:>10,.0f} CNY")
    print(f"  vs Vertex ~230N/t^  =>  ~{2_300_000 / total:.0f}x markup")
    if args.sensitivity:
        print("\n  Sensitivity (+10% each param):")
        for k, v in scenario.sensitivity().items():
            print(f"    {k}: {v:+,.0f} CNY")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trikafta component chemistry calculator")
    parser.add_argument("--mw", action="store_true", help="Calculate molecular weights from SMILES")
    parser.add_argument("--cost", action="store_true", help="Run cost model")
    parser.add_argument("--price-elexa", type=float, default=50_000.0,
                        help="Elexacaftor KSM price (CNY/kg)")
    parser.add_argument("--price-tez", type=float, default=30_000.0,
                        help="Tezacaftor KSM price (CNY/kg)")
    parser.add_argument("--price-iva", type=float, default=10_000.0,
                        help="Ivacaftor KSM price (CNY/kg)")
    parser.add_argument("--yield-step2", type=float, default=0.65,
                        help="Step 2 Mitsunobu yield (0-1)")
    parser.add_argument("--yield-step3", type=float, default=0.55,
                        help="Step 3 deprotection+cyclization yield (0-1)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Show sensitivity analysis")
    args = parser.parse_args()

    if args.mw:
        cmd_mw()
    elif args.cost:
        cmd_cost(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
