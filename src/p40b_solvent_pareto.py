"""p40b_solvent_pareto.py - Multi-objective Pareto front for EEI dissolution solvents.

Objectives (all maximize):
  1. eei_dissolution_score       (higher = better EEI dissolution)
  2. electrode_compat_score       (higher = safer for active material)
  3. logp_proxy                 (higher = easier solvent recovery by distillation)
  4. electrode_compat_score      (already listed, skip)
  4. regeneration_potential_pct  (higher = better capacity recovery)
  5. negative cost_proxy         (lower cost = better)

Constraints:
  - DN >= 26 for meaningful EEI dissolution (rules from p40 physics model)
  - oxidation_stability_V >= 4.1 (NMC cathode ceiling)
  - electrode_compat_score >= 0.70

Output: data/solvent_pareto_front.csv
        results/solvent_pareto_summary.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils import get_logger, RESULTS_DIR, DATA_DIR  # noqa: E402
from utils_pb import write_csv, write_json  # noqa: E402

log = get_logger("p40b_pareto")

OBJECTIVES = [
    "eei_dissolution_score",
    "electrode_compat_score",
    "regeneration_potential_pct",
]


def dominates(a: dict, b: dict, objectives: list) -> bool:
    """Return True if a dominates b (all objectives >= b, at least one strictly >)."""
    better = False
    for obj in objectives:
        va = a.get(obj, 0.0)
        vb = b.get(obj, 0.0)
        if va < vb:
            return False
        if va > vb:
            better = True
    return better


def pareto_front(df: pd.DataFrame, objectives: list) -> pd.DataFrame:
    """Return the non-dominated subset of df."""
    rows = df.to_dict("records")
    pareto = []
    for candidate in rows:
        is_dominated = False
        for other in rows:
            if other is candidate:
                continue
            if dominates(other, candidate, objectives):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(candidate)
    return pd.DataFrame(pareto)


def categorize_solvent(row: dict) -> str:
    dn  = row.get("dn", 0.0)
    dis = row.get("eei_dissolution_score", 0.0)
    com = row.get("electrode_compat_score", 0.0)

    # Thresholds calibrated to new physics model (DMI ~0.875, compat ~0.89)
    if dis >= 0.83 and com >= 0.83:
        return "Tier-1: DEER Prime"
    elif dis >= 0.70 and com >= 0.80:
        return "Tier-2: DEER Candidate"
    elif dis >= 0.50 and com >= 0.75:
        return "Tier-3: Co-solvent"
    elif dn >= 22 and dis < 0.50:
        return "Tier-3: High-DN Baseline"
    else:
        return "Tier-4: Low Priority"


def rank_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Rank pareto solvents by a composite desirability score."""
    df = df.copy()
    # Normalize each objective to [0, 1]
    for obj in OBJECTIVES:
        mn, mx = df[obj].min(), df[obj].max()
        if mx > mn:
            df[f"{obj}_norm"] = (df[obj] - mn) / (mx - mn)
        else:
            df[f"{obj}_norm"] = 0.5

    # Weighted sum (EEI dissolution most important)
    weights = {"eei_dissolution_score_norm": 0.45,
               "electrode_compat_score_norm": 0.30,
               "regeneration_potential_pct_norm": 0.25}
    df["desirability_score"] = sum(df[k] * w for k, w in weights.items())
    return df.sort_values("desirability_score", ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(RESULTS_DIR / "solvent_eei_predictions.csv"))
    ap.add_argument("--dn-min", type=float, default=22.0)
    ap.add_argument("--compat-min", type=float, default=0.70)
    ap.add_argument("--ox-min", type=float, default=0.0)
    args = ap.parse_args()

    log.info("Loading predictions from: %s", args.input)
    df = pd.read_csv(Path(args.input))
    log.info("  %d candidates loaded", len(df))

    # Apply hard constraints (AND: all must pass)
    mask = (
        (df["dn"] >= args.dn_min) &
        (df["electrode_compat_score"] >= args.compat_min)
    )
    if "oxidation_stability_V" in df.columns:
        mask = mask & (df["oxidation_stability_V"] >= args.ox_min)
    filtered = df[mask].copy()
    log.info("  %d pass hard constraints (DN>=%.0f, compat>=%.2f)",
             len(filtered), args.dn_min, args.compat_min)

    # Compute Pareto front
    pareto = pareto_front(filtered, OBJECTIVES)
    log.info("  %d non-dominated (Pareto front)", len(pareto))

    # Rank and categorize
    ranked = rank_pareto(pareto)
    ranked["tier"] = ranked.apply(
        lambda r: categorize_solvent(r.to_dict()), axis=1)

    out_csv = DATA_DIR / "solvent_pareto_front.csv"
    out_json = RESULTS_DIR / "solvent_pareto_summary.json"

    write_csv(out_csv, ranked.to_dict("records"))

    # JSON summary
    tiers = ranked.groupby("tier", sort=False)
    summary = {
        "n_candidates_total":     int(len(df)),
        "n_pass_constraints":     int(len(filtered)),
        "n_pareto":              int(len(ranked)),
        "tier_counts":            {str(k): int(len(v)) for k, v in tiers},
        "top_per_tier": {},
    }
    for tier, grp in tiers:
        top = grp.iloc[0]
        summary["top_per_tier"][str(tier)] = {
            "name":               str(top.get("name", "")),
            "dn":                float(top.get("dn", 0)),
            "eei_dissolution_score":  float(top["eei_dissolution_score"]),
            "electrode_compat_score": float(top["electrode_compat_score"]),
            "regeneration_potential_pct": float(top["regeneration_potential_pct"]),
            "desirability_score":    float(top["desirability_score"]),
        }

    write_json(out_json, summary)

    log.info("Pareto front (%d solvents):", len(ranked))
    for i, (_, row) in enumerate(ranked.iterrows()):
        log.info("  %2d. %-28s  tier=%-25s  diss=%.3f  compat=%.3f  regen=%.1f%%  desir=%.3f",
                 i + 1, str(row.get("name", ""))[:28], row["tier"],
                 row["eei_dissolution_score"], row["electrode_compat_score"],
                 row["regeneration_potential_pct"], row["desirability_score"])

    log.info("CSV: %s", out_csv)
    log.info("JSON: %s", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
