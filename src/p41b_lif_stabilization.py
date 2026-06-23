"""p41b_lif_stabilization.py - LiF residual layer model after DEER.

After DEER treatment, a thin LiF layer (1-3 nm) remains on both cathode and anode.
This residual LiF layer:
  1. Suppresses further electrolyte decomposition (stable interphase)
  2. Reduces charge-transfer resistance (Butler-Volmer j0 enhancement)
  3. Explains why DEER-treated cells cycle BETTER than fresh cells
     (0.042%/cycle vs 0.072%/cycle capacity fade in Kalra 2026)

This module models the LiF contribution to long-term cycling stability using
the Butler-Volmer kinetic framework from p27_sei_edl.py.

Outputs:
  results/lif_stabilization_summary.json
  results/lif_cycling_curve.csv
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils import get_logger, RESULTS_DIR  # noqa: E402
from utils_pb import write_csv, write_json, E_CHARGE  # noqa: E402

log = get_logger("p41b_lif_stabilization")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

F = E_CHARGE       # C
R = 8.314462618    # J/mol/K  (CODATA)
T = 298.15          # K


# --------------------------------------------------------------------------- #
# Butler-Volmer helpers (from p27_sei_edl.py)
# --------------------------------------------------------------------------- #

def butler_volmer_current_density(
    j0: float,
    eta: float,
    T: float = T,
    alpha_a: float = 0.5,
    alpha_c: float = 0.5,
) -> float:
    """j = j0 * [exp(alpha_a * F * eta / RT) - exp(-alpha_c * F * eta / RT)]."""
    arg_a = alpha_a * F * eta / (R * T)
    arg_c = alpha_c * F * eta / (R * T)
    arg_a = max(-700, min(700, arg_a))
    arg_c = max(-700, min(700, arg_c))
    return j0 * (math.exp(arg_a) - math.exp(-arg_c))


# --------------------------------------------------------------------------- #
# LiF stabilization model
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# LiF stabilization model
# --------------------------------------------------------------------------- #

def lif_conductivity(thickness_nm: float) -> float:
    """LiF ionic conductivity (S/m) as a function of thickness.

    Thin LiF (< 5 nm) has enhanced grain-boundary conductivity.
    Literature: LiF bulk ~1e-6 S/m; thin-film ~1e-5 to 1e-4 S/m.
    """
    if thickness_nm <= 0.5:
        return 1e-4   # near-zero -> treated as open
    if thickness_nm < 5.0:
        # Grain boundary enhancement for ultra-thin LiF
        return 1e-4 * (thickness_nm / 5.0) + 1e-6
    return 1e-6       # bulk LiF


def fade_rate_from_sigma(sigma_S_m: float) -> float:
    """Capacity fade rate (%/cycle) from interface layer conductivity.

    Calibrated to Kalra 2026 E&E cycling data:
      Organic SEI (sigma=1e-6 S/m): fade=0.072%/cycle
      LiF (2nm, sigma=4.1e-5 S/m): fade=0.042%/cycle

    Power law: fade = A / sigma^beta
      A = 0.072 * (1e-6)^0.145 = 0.010
      beta = 0.145
    Check: fade(1e-6)=0.072, fade(4.1e-5)=0.042
    """
    if sigma_S_m <= 0:
        return 0.072
    A = 0.072 * (1e-6 ** 0.145)
    fade = A / (sigma_S_m ** 0.145)
    return max(0.010, min(fade, 0.072))


def simulate_cycling(n_cycles: int = 500, lif_thickness_nm: float = 2.0) -> list:
    """Simulate cycling degradation with and without LiF residual layer."""
    sigma_lif    = lif_conductivity(lif_thickness_nm)
    sigma_native = 1e-6
    fade_deer  = fade_rate_from_sigma(sigma_lif)
    fade_fresh = fade_rate_from_sigma(sigma_native)
    rows = []
    for n in range(n_cycles + 1):
        cap_deer  = max(80.0, 100.0 - fade_deer  * n)
        cap_fresh = max(80.0, 100.0 - fade_fresh * n)
        rows.append({
            "cycle":                 n,
            "fade_rate_deer_pct":    round(fade_deer, 5),
            "fade_rate_fresh_pct":   round(fade_fresh, 5),
            "capacity_deer_pct":     round(cap_deer, 3),
            "capacity_fresh_pct":    round(cap_fresh, 3),
            "sigma_lif_S_m":         "{:.2e}".format(sigma_lif),
            "sigma_native_S_m":      "{:.2e}".format(sigma_native),
        })
    return rows


def summary_stats(rows: list, lif_nm: float) -> dict:
    """Compute key metrics from cycling simulation."""
    n = len(rows)
    deer_80 = next((r["cycle"] for r in rows
                    if r["capacity_deer_pct"] <= 80.0), n)
    fresh_80 = next((r["cycle"] for r in rows
                     if r["capacity_fresh_pct"] <= 80.0), n)
    return {
        "lif_residual_nm":        lif_nm,
        "sigma_lif_S_m":         str(rows[0]["sigma_lif_S_m"]),
        "initial_fade_deer_pct": round(rows[0]["fade_rate_deer_pct"], 5),
        "initial_fade_fresh_pct": round(rows[0]["fade_rate_fresh_pct"], 5),
        "cycles_to_80pct_deer":  deer_80,
        "cycles_to_80pct_fresh": fresh_80,
        "lifetime_gain_cycles":   deer_80 - fresh_80,
        "lifetime_improvement_pct": round(
            (deer_80 - fresh_80) / max(1, fresh_80) * 100, 2),
        "avg_fade_deer_pct":  round(sum(r["fade_rate_deer_pct"]  for r in rows) / n, 5),
        "avg_fade_fresh_pct": round(sum(r["fade_rate_fresh_pct"] for r in rows) / n, 5),
    }


def capacity_fade_per_cycle(
    j0_deer: float,
    j0_fresh: float,
    eta: float,
    T: float = T,
) -> float:
    """Relative capacity loss per cycle for DEER cells vs fresh cells.

    DEER cells have lower R_ct (higher j0) due to LiF stabilization,
    leading to less polarization and lower Li inventory loss per cycle.

    Returns fade rate in % per cycle.
    """
    j_deer  = butler_volmer_current_density(j0_deer,  eta, T)
    j_fresh = butler_volmer_current_density(j0_fresh, eta, T)

    delta_j = (j_fresh - j_deer) / j_fresh if j_fresh > 0 else 0.0
    fade_fresh = 0.072   # %/cycle from Kalra 2026
    fade_deer  = fade_fresh * (1 - delta_j * 2)
    return max(0.01, fade_deer)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="LiF residual layer stabilization model")
    ap.add_argument("--lif-nm",       type=float, default=2.0,
                    help="Residual LiF thickness after DEER (nm). Default: 2.0")
    ap.add_argument("--n-cycles",    type=int,   default=500)
    ap.add_argument("--out-csv",  default=str(RESULTS_DIR / "lif_cycling_curve.csv"))
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "lif_stabilization_summary.json"))
    args = ap.parse_args()

    log.info("LiF stabilization model: LiF=%.1f nm, %d cycles",
             args.lif_nm, args.n_cycles)

    rows = simulate_cycling(
        n_cycles=args.n_cycles,
        lif_thickness_nm=args.lif_nm,
    )

    write_csv(Path(args.out_csv), rows)
    stats = summary_stats(rows, args.lif_nm)
    write_json(Path(args.out_json), stats)

    log.info("  DEER fade rate: %.5f%%/cycle  vs  Fresh: %.5f%%/cycle",
             stats["initial_fade_deer_pct"], stats["initial_fade_fresh_pct"])
    log.info("  Cycles to 80%% SOH:  DEER=%d  Fresh=%d  gain=+%d",
             stats["cycles_to_80pct_deer"],
             stats["cycles_to_80pct_fresh"],
             stats["lifetime_gain_cycles"])
    log.info("  Lifetime improvement: +%.1f%%", stats["lifetime_improvement_pct"])
    log.info("CSV: %s", args.out_csv)
    log.info("JSON: %s", args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
