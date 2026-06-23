"""p42_tea_lca.py - Techno-Economic Analysis (TEA) and LCA for DEER.

Implements a simplified EverBatt-style model comparing five battery recycling pathways:
  1. Pyrometallurgy    (fire assay, 26.31 $/kg)
  2. Hydrometallurgy   (acid Leach, 31.07 $/kg)
  3. Electrolyte swap  (electrolyte replacement only, 83% recovery)
  4. DEER              (direct electrode regeneration, 15.25 $/kg)
  5. Direct cathode    (cathode-to-cathode, intermediate cost)

Outputs:
  results/deer_tea_lca.csv
  results/deer_tea_lca_summary.json

References:
  - Kalra 2026 E&E: DEER 15.25 $/kg, 56% cost reduction vs pyrometallurgy
  - EverBatt (Argonne) 2019 model
  - Zhang 2021 Energy Storage Materials (cost breakdown)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils import get_logger, RESULTS_DIR  # noqa: E402
from utils_pb import write_csv, write_json  # noqa: E402

log = get_logger("p42_tea_lca")

# --------------------------------------------------------------------------- #
# Pathway definitions
# --------------------------------------------------------------------------- #

PATHWAYS = {
    "Pyrometallurgy": {
        "display_name": "火法冶金 (Pyrometallurgy)",
        "cost_usd_kg":     26.31,
        "energy_kwh_kg":   12.5,
        "ghg_kg_co2_kg":   8.2,
        "capacity_recovery_pct": 98.0,
        "process_steps": ["破碎/粉碎", "高温熔炼", "精炼", "金属还原", "电极重造"],
        "color": "#e74c3c",
        "labor_hours_kg":  0.15,
        "chemical_cost_kg": 1.5,
    },
    "Hydrometallurgy": {
        "display_name": "湿法冶金 (Hydrometallurgy)",
        "cost_usd_kg":     31.07,
        "energy_kwh_kg":   8.8,
        "ghg_kg_co2_kg":   5.5,
        "capacity_recovery_pct": 97.0,
        "process_steps": ["破碎/粉碎", "酸浸出", "溶剂萃取", "沉淀", "电极涂布"],
        "color": "#9b59b6",
        "labor_hours_kg":  0.25,
        "chemical_cost_kg": 4.2,
    },
    "Electrolyte_Swap": {
        "display_name": "换电解液 (Electrolyte Swap)",
        "cost_usd_kg":     5.20,
        "energy_kwh_kg":   0.8,
        "ghg_kg_co2_kg":   0.6,
        "capacity_recovery_pct": 83.0,
        "process_steps": ["放电", "拆解", "更换电解液", "注液封装"],
        "color": "#3498db",
        "labor_hours_kg":  0.08,
        "chemical_cost_kg": 3.5,
    },
    "DEER": {
        "display_name": "DEER 直接再生 (DEER)",
        "cost_usd_kg":     15.25,
        "energy_kwh_kg":   2.8,
        "ghg_kg_co2_kg":   1.8,
        "capacity_recovery_pct": 95.0,
        "process_steps": ["CV 扫描正极", "CV 扫描负极", "LiF 稳定化", "电解液补充", "封装"],
        "color": "#27ae60",
        "labor_hours_kg":  0.10,
        "chemical_cost_kg": 6.8,   # DMI solvent
    },
    "Direct_Cathode": {
        "display_name": "直接修复正极 (Direct Cathode)",
        "cost_usd_kg":     18.50,
        "energy_kwh_kg":   4.0,
        "ghg_kg_co2_kg":   2.5,
        "capacity_recovery_pct": 91.0,
        "process_steps": ["正极修复焙烧", "电解液补充", "界面优化", "封装"],
        "color": "#f39c12",
        "labor_hours_kg":  0.12,
        "chemical_cost_kg": 2.0,
    },
}


# --------------------------------------------------------------------------- #
# Cost breakdown model
# --------------------------------------------------------------------------- #

def breakdown_cost(pathway_key: str, scale_kg: float = 1000.0) -> dict:
    """Break down manufacturing cost into components for a pathway.

    Scale: kg of recovered battery material (not per kg of original battery).
    """
    p = PATHWAYS[pathway_key]
    total = p["cost_usd_kg"]

    # Component fractions (based on EverBatt model)
    cathode_mat_fraction = 0.31    # cathode active material (biggest cost)
    binder_sep_fraction  = 0.16    # binder + separator
    labor_fraction       = 0.10    # labor
    energy_fraction      = 0.08    # energy
    overhead_fraction    = 0.35    # facility, logistics, etc.

    # DEER has different breakdown: less material, more process
    if pathway_key == "DEER":
        cathode_mat_fraction = 0.50   # preserves cathode
        binder_sep_fraction = 0.16
        labor_fraction      = 0.12
        energy_fraction     = 0.06
        overhead_fraction   = 0.16

    elif pathway_key == "Electrolyte_Swap":
        cathode_mat_fraction = 0.00   # doesn't recover cathode
        binder_sep_fraction = 0.15
        labor_fraction      = 0.20
        energy_fraction     = 0.05
        overhead_fraction   = 0.60

    cathode_cost = total * cathode_mat_fraction
    binder_sep   = total * binder_sep_fraction
    labor       = total * labor_fraction
    energy      = total * energy_fraction
    overhead    = total * overhead_fraction

    # Scale to total
    total_cost = total * scale_kg
    return {
        "pathway":            pathway_key,
        "display_name":       p["display_name"],
        "scale_kg":           scale_kg,
        "cost_per_kg":        round(total, 4),
        "total_cost":         round(total_cost, 2),
        "cathode_material":   round(cathode_cost * scale_kg, 2),
        "binder_separator":   round(binder_sep   * scale_kg, 2),
        "labor":              round(labor       * scale_kg, 2),
        "energy_cost":        round(energy      * scale_kg, 2),
        "overhead":           round(overhead     * scale_kg, 2),
    }


# --------------------------------------------------------------------------- #
# LCA model
# --------------------------------------------------------------------------- #

def lca_metrics(pathway_key: str) -> dict:
    p = PATHWAYS[pathway_key]
    # Energy and GHG per kg of recovered material
    return {
        "pathway":            pathway_key,
        "energy_kwh_kg":      p["energy_kwh_kg"],
        "ghg_kg_co2_kg":     p["ghg_kg_co2_kg"],
        "capacity_recovery_pct": p["capacity_recovery_pct"],
        # Effective GHG per Ah recovered (accounting for capacity recovery)
        "effective_ghg": round(
            p["ghg_kg_co2_kg"] / (p["capacity_recovery_pct"] / 100.0), 3),
    }


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #

def comparison_table() -> list:
    rows = []
    deer = PATHWAYS["DEER"]

    for key, p in PATHWAYS.items():
        bd = breakdown_cost(key)
        _lca = lca_metrics(key)

        cost_vs_deer = round(p["cost_usd_kg"] / deer["cost_usd_kg"], 2)
        energy_vs_deer = round(p["energy_kwh_kg"] / deer["energy_kwh_kg"], 2)
        ghg_vs_deer = round(p["ghg_kg_co2_kg"] / deer["ghg_kg_co2_kg"], 2)

        rows.append({
            "pathway":                 key,
            "display_name":            p["display_name"],
            "cost_usd_kg":            p["cost_usd_kg"],
            "cost_vs_deer":           cost_vs_deer,
            "energy_kwh_kg":         p["energy_kwh_kg"],
            "energy_vs_deer":         energy_vs_deer,
            "ghg_kg_co2_kg":         p["ghg_kg_co2_kg"],
            "ghg_vs_deer":           ghg_vs_deer,
            "capacity_recovery_pct": p["capacity_recovery_pct"],
            "labor_h_kg":            p["labor_hours_kg"],
            "chemical_cost_kg":      p["chemical_cost_kg"],
            "color":                 p["color"],
            # Cost breakdown
            "cathode_material_cost":  round(bd["cathode_material"], 4),
            "binder_sep_cost":        round(bd["binder_separator"], 4),
            "labor_cost":             round(bd["labor"], 4),
            "energy_cost":            round(bd["energy_cost"], 4),
            "overhead_cost":          round(bd["overhead"], 4),
        })

    # Sort by cost
    rows.sort(key=lambda r: r["cost_usd_kg"])
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="TEA and LCA comparison for DEER")
    ap.add_argument("--scale-kg", type=float, default=1000.0,
                    help="Scale in kg of recovered material. Default: 1000")
    args = ap.parse_args()

    log.info("TEA/LCA comparison for %d kg scale:", int(args.scale_kg))

    rows = comparison_table()
    out_csv = RESULTS_DIR / "deer_tea_lca.csv"
    write_csv(out_csv, rows)

    # Summary JSON
    deer_row = next(r for r in rows if r["pathway"] == "DEER")
    summary = {
        "scale_kg":              args.scale_kg,
        "deer_reference_cost":  deer_row["cost_usd_kg"],
        "deer_reference_energy": deer_row["energy_kwh_kg"],
        "deer_reference_ghg":    deer_row["ghg_kg_co2_kg"],
        "pathway_count":         len(rows),
        "cheapest_pathway":      rows[0]["pathway"],
        "cheapest_cost":         rows[0]["cost_usd_kg"],
        "cheapest_vs_deer_pct": round(
            (rows[0]["cost_usd_kg"] - deer_row["cost_usd_kg"]) /
            deer_row["cost_usd_kg"] * 100, 2),
        "deer_cost_savings_vs_pyro_pct": round(
            (PATHWAYS["Pyrometallurgy"]["cost_usd_kg"] - deer_row["cost_usd_kg"]) /
            PATHWAYS["Pyrometallurgy"]["cost_usd_kg"] * 100, 2),
        "deer_energy_savings_vs_pyro_pct": round(
            (PATHWAYS["Pyrometallurgy"]["energy_kwh_kg"] - deer_row["energy_kwh_kg"]) /
            PATHWAYS["Pyrometallurgy"]["energy_kwh_kg"] * 100, 2),
        "deer_ghg_savings_vs_pyro_pct": round(
            (PATHWAYS["Pyrometallurgy"]["ghg_kg_co2_kg"] - deer_row["ghg_kg_co2_kg"]) /
            PATHWAYS["Pyrometallurgy"]["ghg_kg_co2_kg"] * 100, 2),
        "pathways": [
            {
                "pathway":    r["pathway"],
                "cost":       r["cost_usd_kg"],
                "cost_vs_deer": r["cost_vs_deer"],
                "energy_kwh_kg": r["energy_kwh_kg"],
                "ghg_kg_co2_kg": r["ghg_kg_co2_kg"],
                "capacity_pct":   r["capacity_recovery_pct"],
            }
            for r in rows
        ],
    }

    out_json = RESULTS_DIR / "deer_tea_lca_summary.json"
    write_json(out_json, summary)

    log.info("Pathway comparison (sorted by cost):")
    for r in rows:
        log.info("  %-22s  $%5.2f/kg  %.1fx vs DEER  energy=%.1f kWh/kg  ghg=%.1f kgCO2/kg  cap=%.0f%%",
                 r["pathway"], r["cost_usd_kg"], r["cost_vs_deer"],
                 r["energy_kwh_kg"], r["ghg_kg_co2_kg"], r["capacity_recovery_pct"])

    log.info("CSV: %s", out_csv)
    log.info("JSON: %s", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
