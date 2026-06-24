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
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

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
    print("  KSM:")
    print(f"    Elexacaftor KSM  ~200g @{scenario.price_elexa_ksm:>10,.0f}/kg = {scenario.price_elexa_ksm * 0.2:>10,.0f}")
    print(f"    Tezacaftor KSM  ~100g @{scenario.price_tez_ksm:>10,.0f}/kg = {scenario.price_tez_ksm * 0.1:>10,.0f}")
    print(f"    Ivacaftor KSM   ~50g @{scenario.price_iva_ksm:>10,.0f}/kg = {scenario.price_iva_ksm * 0.05:>10,.0f}")
    print(f"    KSM Total                                             = {ksm_sub:>10,.0f}")
    print(f"  Reagents (HATU={scenario.reagent_hatu:,.0f})                         = {reagents_sub:>10,.0f}")
    print(f"  GMP Synthesis                                       = {scenario.gmp_synthesis:>10,.0f}")
    print(f"  Formulation+QC                                     = {scenario.formulation + scenario.qc_testing:>10,.0f}")
    print(f"  {'='*55}")
    print(f"  Total                                               = {total:>10,.0f} CNY")
    print(f"  vs Vertex ~2,300,000 CNY/yr  =>  ~{2_300_000 / total:.0f}x markup")
    if args.sensitivity:
        print("\n  Sensitivity (+10% each param):")
        for k, v in scenario.sensitivity().items():
            print(f"    {k}: {v:+,.0f} CNY")
    print()


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run N rounds of Monte-Carlo optimisation (8 min each).

    Each round randomly varies:
      - KSM prices  (uniform within +/- 30% of base)
      - Reagent costs
      - GMP / formulation / QC processing costs
      - Yield rates for Step 2 and Step 3
      - Import tariff scenario (0 % / 5 % / 13 %)

    Reports per-round cost and aggregated statistics.
    """
    sys.stdout.reconfigure(line_buffering=True)
    base = CostScenario()
    n_rounds = args.rounds
    seconds_per_round = args.seconds_per_round

    print(
        f"\n{'='*65}\n"
        f"  Monte-Carlo Cost Optimisation  ({n_rounds} rounds x {seconds_per_round}s)\n"
        f"{'='*65}\n"
    )
    print(f"  Base cost (default params) : {base.total_cost():>12,.0f} CNY/yr\n")

    round_results: list[dict] = []
    start_total = time.monotonic()

    for i in range(1, n_rounds + 1):
        r_start = time.monotonic()

        # Randomise parameters within plausible ranges
        rng = random.Random(i)          # reproducible per-round seed
        price_elexa = base.price_elexa_ksm * rng.uniform(0.7, 1.30)
        price_tez   = base.price_tez_ksm   * rng.uniform(0.7, 1.30)
        price_iva   = base.price_iva_ksm   * rng.uniform(0.7, 1.30)

        reagent_hatu   = base.reagent_hatu   * rng.uniform(0.8, 1.20)
        reagent_cdi   = base.reagent_cdi   * rng.uniform(0.8, 1.20)
        reagent_solvent = base.reagent_solvent * rng.uniform(0.8, 1.20)

        gmp_synthesis = base.gmp_synthesis * rng.uniform(0.85, 1.15)
        formulation   = base.formulation   * rng.uniform(0.85, 1.15)
        qc_testing    = base.qc_testing    * rng.uniform(0.85, 1.15)

        yield_step2 = rng.uniform(0.50, 0.80)
        yield_step3 = rng.uniform(0.40, 0.70)

        # Tariff scenario
        tariff_options = [(0.0, "0%"),
                          (0.05, "5%"),
                          (0.13, "13%")]
        tariff_pct, tariff_label = rng.choices(
            tariff_options, weights=[5, 3, 1])[0]

        # Lecheng zero-tariff saves the 5 % scenario cost
        tariff_saving = price_elexa * 0.2 * tariff_pct  # Elexa KSM component

        scenario = CostScenario(
            price_elexa_ksm=price_elexa,
            price_tez_ksm=price_tez,
            price_iva_ksm=price_iva,
            reagent_na_bh4=base.reagent_na_bh4,
            reagent_dppa=base.reagent_dppa,
            reagent_pp_diad=base.reagent_pp_diad,
            reagent_tfa=base.reagent_tfa,
            reagent_cdi=reagent_cdi,
            reagent_hatu=reagent_hatu,
            reagent_solvent=reagent_solvent,
            gmp_synthesis=gmp_synthesis,
            formulation=formulation,
            qc_testing=qc_testing,
            yield_step2=yield_step2,
            yield_step3=yield_step3,
        )

        cost = scenario.total_cost()
        cost_with_tariff = cost + tariff_saving
        cost_saved = tariff_saving          # Lecheng saves tariff
        net_cost_lecheng = cost_with_tariff - cost_saved

        elapsed = time.monotonic() - r_start
        sleep_remaining = max(0, seconds_per_round - elapsed)
        if sleep_remaining > 0 and i < n_rounds:
            time.sleep(sleep_remaining)

        round_results.append({
            "round": i,
            "cost_base": round(cost, 2),
            "cost_with_tariff": round(cost_with_tariff, 2),
            "tariff_pct": tariff_pct,
            "tariff_saving": round(cost_saved, 2),
            "net_cost_lecheng": round(net_cost_lecheng, 2),
            "yield_step2": round(yield_step2, 4),
            "yield_step3": round(yield_step3, 4),
            "price_elexa_ksm": round(price_elexa, 2),
            "price_tez_ksm": round(price_tez, 2),
            "price_iva_ksm": round(price_iva, 2),
            "elapsed_s": round(elapsed, 3),
        })

        status = "OK" if elapsed >= seconds_per_round * 0.9 else "FAST"
        print(
            f"  Round {i:>2}/{n_rounds} | "
            f"cost={cost:>8,.0f} | "
            f"tariff={tariff_label:>3} saving={cost_saved:>6,.0f} | "
            f"lecheng_net={net_cost_lecheng:>8,.0f} | "
            f"y2={yield_step2:.2f} y3={yield_step3:.2f} | "
            f"{status}"
        )

    total_elapsed = time.monotonic() - start_total

    # --- Summary statistics ---
    costs_base = [r["cost_base"] for r in round_results]
    costs_lecheng = [r["net_cost_lecheng"] for r in round_results]
    costs_raw = [r["cost_with_tariff"] for r in round_results]

    def pct(arr: list[float], p: float) -> float:
        s = sorted(arr)
        return s[int(len(s) * p / 100)]

    print(f"\n{'='*65}")
    print(f"  Summary  (total wall time: {total_elapsed:.1f}s)\n")
    print(f"  {'Metric':<30} {'No-tariff':>12} {'With-tariff':>12} {'Lecheng-net':>12}")
    print(f"  {'-'*68}")

    for label, arr in [
        ("Mean",     costs_base),
        ("Median",   [sorted(costs_base)[len(costs_base)//2]]),
        ("P10",      [pct(costs_base, 10)]),
        ("P50",      [pct(costs_base, 50)]),
        ("P90",      [pct(costs_base, 90)]),
        ("Min",      [min(costs_base)]),
        ("Max",      [max(costs_base)]),
    ]:
        vals = arr if label != "Mean" else [sum(arr)/len(arr)]
        raw_vals = costs_raw[costs_base.index(vals[0])] if label != "Mean" else sum(costs_raw)/len(costs_raw)
        lecheng_vals = costs_lecheng[costs_base.index(vals[0])] if label != "Mean" else sum(costs_lecheng)/len(costs_lecheng)
        print(f"  {label:<30} {vals[0]:>12,.0f} {raw_vals:>12,.0f} {lecheng_vals:>12,.0f}")

    print(f"\n  vs Vertex ~2,300,000 CNY/yr:")
    mean_lecheng = sum(costs_lecheng) / len(costs_lecheng)
    print(f"    Mean  markup: {2_300_000 / (sum(costs_base)/len(costs_base)):.0f}x")
    print(f"    Lecheng-net mean: {2_300_000 / mean_lecheng:.0f}x")
    print(f"    Lecheng-net P10 (best): {2_300_000 / pct(costs_lecheng, 10):.0f}x")
    print(f"    Lecheng-net P90 (worst): {2_300_000 / pct(costs_lecheng, 90):.0f}x")

    # Save JSON
    out_path = args.json_out or "optimisation_results.json"
    summary = {
        "n_rounds": n_rounds,
        "seconds_per_round": seconds_per_round,
        "total_elapsed_s": round(total_elapsed, 2),
        "base_cost": base.total_cost(),
        "stats_no_tariff": {
            "mean": round(sum(costs_base) / len(costs_base), 2),
            "median": round(sorted(costs_base)[len(costs_base)//2], 2),
            "p10": round(pct(costs_base, 10), 2),
            "p50": round(pct(costs_base, 50), 2),
            "p90": round(pct(costs_base, 90), 2),
            "min": round(min(costs_base), 2),
            "max": round(max(costs_base), 2),
        },
        "stats_lecheng_net": {
            "mean": round(sum(costs_lecheng) / len(costs_lecheng), 2),
            "median": round(sorted(costs_lecheng)[len(costs_lecheng)//2], 2),
            "p10": round(pct(costs_lecheng, 10), 2),
            "p50": round(pct(costs_lecheng, 50), 2),
            "p90": round(pct(costs_lecheng, 90), 2),
            "min": round(min(costs_lecheng), 2),
            "max": round(max(costs_lecheng), 2),
        },
        "rounds": round_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}\n")


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
    parser.add_argument("--optimize", action="store_true",
                        help="Run Monte-Carlo cost optimisation (10 rounds x 8 min each)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of Monte-Carlo rounds (default: 10)")
    parser.add_argument("--seconds-per-round", type=int, default=480,
                        help="Seconds per round (default: 480 = 8 min)")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Output JSON file path (default: optimisation_results.json)")
    args = parser.parse_args()

    if args.mw:
        cmd_mw()
    elif args.cost:
        cmd_cost(args)
    elif args.optimize:
        cmd_optimize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
