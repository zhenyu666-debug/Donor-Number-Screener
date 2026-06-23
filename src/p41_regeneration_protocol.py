"""p41_regeneration_protocol.py - DEER CV regeneration protocol simulator.

Simulates the cyclic voltammetry (CV) scan protocol for Direct Electrode-to-Electrode
Regeneration (DEER), reproducing the R_EEI vs scan_number decay curve reported
in Kalra DEER 2026 (Fig. 2b-c, E&E).

Models:
  R_EEI(n) = R_EEI_0 * exp(-k_diss * n)           [exponential decay]
  k_diss   = k_0 * exp(-alpha * DN / (kB * T))    [Arrhenius with DN as driver]

Electrode types supported:
  - cathode: NMC811 (CEI, 10-20 nm initial thickness)
  - anode:   graphite (SEI, 30-40 nm initial thickness)

Outputs:
  results/regeneration_cv_curves.csv     (R_EEI vs scan_number per electrode)
  results/regeneration_summary.json      (k_diss, R_final, n_scans_needed)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils import get_logger, RESULTS_DIR  # noqa: E402
from utils_pb import write_csv, write_json  # noqa: E402

log = get_logger("p41_regeneration_protocol")

# Physical constants
K_B = 8.617333262e-5   # eV/K


# --------------------------------------------------------------------------- #
# Kinetic model
# --------------------------------------------------------------------------- #

def dissolution_rate_constant(dn: float, T: float = 298.15,
                               k_0: float = 0.05,
                               alpha: float = 0.189) -> float:
    """Empirically calibrated dissolution rate constant.

    Calibrated to:
      DMI (DN=29):  k_diss = 0.5 per scan
      EC  (DN=16.8): k_diss = 0.05 per scan

    The formula k_diss = k_0 * exp(alpha * (DN - 16.8)) is fitted from those
    two anchor points.  DN_ref = 16.8 (EC baseline).
    """
    return k_0 * math.exp(alpha * (dn - 16.8))


def eei_resistance_after_n_scans(
    r_eei_0: float,
    n: int,
    k_diss: float,
) -> float:
    """R_EEI(n) = R_EEi_0 * exp(-k_diss * n)."""
    return r_eei_0 * math.exp(-k_diss * n)


def n_scans_to_recover(r_eei_0: float, target_resistance: float = 0.05,
                        k_diss: float = 0.5) -> float:
    """Number of CV scans needed to reduce R_EEI below target (in Ohm-equivalent).

    Solves: r_eei_0 * exp(-k * n) = target  =>  n = -ln(target/r_eei_0) / k
    """
    if target_resistance >= r_eei_0 or k_diss <= 0:
        return 0.0
    return math.log(r_eei_0 / target_resistance) / k_diss


def thickness_after_n_scans(
    thickness_0_nm: float,
    n: int,
    k_diss: float,
) -> float:
    """Assume thickness ~ proportional to R_EEI (same exponential)."""
    return thickness_0_nm * math.exp(-k_diss * n)


def lif_residual_thickness(
    lif_initial_nm: float,
    n: int,
    dissolution_fraction: float = 0.90,
) -> float:
    """LiF partially survives DEER (90% dissolves, 10% residual).
    Residual LiF forms a stable 2-3 nm protective layer on cycling.

    Returns residual LiF thickness after n scans.
    """
    dissolved = lif_initial_nm * dissolution_fraction * (1 - math.exp(-0.3 * n))
    return max(0.5, lif_initial_nm - dissolved)


# --------------------------------------------------------------------------- #
# Electrode configurations
# --------------------------------------------------------------------------- #

ELECTRODE_DEFAULTS = {
    "NMC811_cathode": {
        "name":              "NMC811 cathode",
        "eei_initial_nm":    15.0,
        "r_eei_0_ohm":      85.0,    # Ohm per cm^2 (from Kalra Fig.2b)
        "lif_initial_nm":    3.0,
        "cv_window_V":       [3.0, 4.1],
        "n_cycles_optimal":  3,
    },
    "graphite_anode": {
        "name":              "Graphite anode",
        "eei_initial_nm":    35.0,
        "r_eei_0_ohm":      150.0,   # thicker SEI -> higher resistance
        "lif_initial_nm":    5.0,
        "cv_window_V":       [0.01, 1.5],
        "n_cycles_optimal":  5,
    },
}


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

def simulate_cv_decay(
    dn: float,
    electrode_key: str = "NMC811_cathode",
    n_scans_max: int = 20,
    T: float = 298.15,
    solvent_name: str = "DMI",
) -> dict:
    cfg = ELECTRODE_DEFAULTS.get(electrode_key, ELECTRODE_DEFAULTS["NMC811_cathode"])

    r_eei_0 = cfg["r_eei_0_ohm"]

    # Recovery target: 5% of initial R_EEI
    r_target = 0.05 * r_eei_0
    k_diss = dissolution_rate_constant(dn, T=T)
    n_needed = n_scans_to_recover(r_eei_0, r_target, k_diss)

    rows = []
    for n in range(n_scans_max + 1):
        r_eei  = eei_resistance_after_n_scans(r_eei_0, n, k_diss)
        th_eei = thickness_after_n_scans(cfg["eei_initial_nm"], n, k_diss)
        th_lif = lif_residual_thickness(cfg["lif_initial_nm"], n)

        rows.append({
            "electrode":            cfg["name"],
            "solvent":              solvent_name,
            "dn":                   dn,
            "scan_number":          n,
            "r_eei_ohm_cm2":       round(r_eei, 4),
            "r_eei_normalized":     round(r_eei / r_eei_0, 4),
            "eei_thickness_nm":     round(th_eei, 3),
            "lif_residual_nm":      round(th_lif, 3),
            "k_diss_per_scan":      round(k_diss, 6),
        })

    final = rows[-1]
    return {
        "electrode":       cfg["name"],
        "solvent":         solvent_name,
        "dn":             dn,
        "k_diss_per_scan": round(k_diss, 6),
        "r_eei_0_ohm":    r_eei_0,
        "r_eei_final_ohm": round(final["r_eei_ohm_cm2"], 4),
        "eei_thickness_final_nm": round(final["eei_thickness_nm"], 3),
        "lif_residual_final_nm":  round(final["lif_residual_nm"], 3),
        "n_scans_needed": round(n_needed, 2),
        "n_scans_optimal": cfg["n_cycles_optimal"],
        "recovery_pct": round((1 - final["r_eei_ohm_cm2"] / r_eei_0) * 100, 2),
        "cv_window_V": cfg["cv_window_V"],
        "rows": rows,
    }


def simulate_dual_electrode(
    dn: float,
    solvent_name: str = "DMI",
    n_scans: int = 5,
    T: float = 298.15,
) -> dict:
    """Simulate both cathode and anode regeneration together."""
    cathode = simulate_cv_decay(dn, "NMC811_cathode", n_scans, T, solvent_name)
    anode   = simulate_cv_decay(dn, "graphite_anode",   n_scans, T, solvent_name)

    # Combined resistance (series)
    combined_rows = []
    for n in range(n_scans + 1):
        c_row = cathode["rows"][n]
        a_row = anode["rows"][n]
        combined_rows.append({
            "scan_number":           n,
            "r_cathode_ohm":         round(c_row["r_eei_ohm_cm2"], 4),
            "r_anode_ohm":           round(a_row["r_eei_ohm_cm2"], 4),
            "r_total_ohm":           round(c_row["r_eei_ohm_cm2"] + a_row["r_eei_ohm_cm2"], 4),
            "r_total_normalized":     round(
                (c_row["r_eei_ohm_cm2"] + a_row["r_eei_ohm_cm2"]) /
                (cathode["r_eei_0_ohm"] + anode["r_eei_0_ohm"]), 4),
            "lif_cathode_nm":        c_row["lif_residual_nm"],
            "lif_anode_nm":          a_row["lif_residual_nm"],
        })

    r_total_0 = cathode["r_eei_0_ohm"] + anode["r_eei_0_ohm"]
    r_total_f = combined_rows[-1]["r_total_ohm"]
    return {
        "solvent":           solvent_name,
        "dn":               dn,
        "cathode_summary":  {k: v for k, v in cathode.items() if k != "rows"},
        "anode_summary":    {k: v for k, v in anode.items()   if k != "rows"},
        "combined_recovery_pct": round((1 - r_total_f / r_total_0) * 100, 2),
        "n_scans":          n_scans,
        "combined_rows":    combined_rows,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="DEER CV regeneration protocol simulator")
    ap.add_argument("--dn",        type=float, default=29.0,
                     help="Solvent DN (kcal/mol). Default: 29.0 (DMI)")
    ap.add_argument("--solvent",    default="DMI")
    ap.add_argument("--electrode",
                    choices=["cathode", "anode", "dual"],
                    default="dual")
    ap.add_argument("--n-scans",    type=int, default=10)
    ap.add_argument("--T",         type=float, default=298.15)
    ap.add_argument("--out",       default=str(RESULTS_DIR / "regeneration_cv_curves.csv"))
    ap.add_argument("--out-json",  default=str(RESULTS_DIR / "regeneration_summary.json"))
    args = ap.parse_args()

    log.info("DEER protocol simulation: %s (DN=%.1f kcal/mol) at %.1fK",
             args.solvent, args.dn, args.T)

    if args.electrode == "dual":
        result = simulate_dual_electrode(args.dn, args.solvent, args.n_scans, args.T)
        out_csv = Path(args.out)
        write_csv(out_csv, result["combined_rows"])
        summary = {k: v for k, v in result.items() if k != "combined_rows"}
        write_json(Path(args.out_json), summary)

        log.info("  Cathode k_diss=%.4f /scan, n_needed=%.1f scans",
                 result["cathode_summary"]["k_diss_per_scan"],
                 result["cathode_summary"]["n_scans_needed"])
        log.info("  Anode   k_diss=%.4f /scan, n_needed=%.1f scans",
                 result["anode_summary"]["k_diss_per_scan"],
                 result["anode_summary"]["n_scans_needed"])
        log.info("  Combined recovery: %.1f%% after %d scans",
                 result["combined_recovery_pct"], args.n_scans)

    elif args.electrode in ("cathode", "anode"):
        key = "NMC811_cathode" if args.electrode == "cathode" else "graphite_anode"
        result = simulate_cv_decay(args.dn, key, args.n_scans, args.T, args.solvent)
        write_csv(Path(args.out), result["rows"])
        summary = {k: v for k, v in result.items() if k != "rows"}
        write_json(Path(args.out_json), summary)
        log.info("  k_diss=%.4f /scan, R_EEI 0->%.2f Ohm, n_needed=%.1f scans",
                 result["k_diss_per_scan"],
                 result["r_eei_final_ohm"],
                 result["n_scans_needed"])

    log.info("CSV: %s", args.out)
    log.info("JSON: %s", args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
