"""p41c_pouch_validation.py - Scale-up from coin cell to 3 Ah pouch cell.

Kalra 2026 reports 90.3% capacity recovery in a 3 Ah soft-pouch cell,
vs 95% in coin cells. This module extrapolates from coin-cell results
to pouch scale using electrode area scaling and N/P ratio corrections.

Model:
  Recovery_pouch ≈ Recovery_coin * f_area * f_npn * f_rate

  f_area:  area ratio (pouch_are / coin_area) → more edge effects at scale
  f_npn:   N/P ratio correction (graphite excess reduces Li inventory)
  f_rate:   C-rate correction (pouch tested at higher rate)

Outputs:
  results/pouch_scale_validation.json
  results/pouch_comparison.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils import get_logger, RESULTS_DIR  # noqa: E402
from utils_pb import write_csv, write_json  # noqa: E402

log = get_logger("p41c_pouch_validation")


# --------------------------------------------------------------------------- #
# Scale-up factors
# --------------------------------------------------------------------------- #

def area_correction_factor(area_cm2: float) -> float:
    """Larger electrodes have more edge/sealing losses.

    Empirically: f_area ≈ 1 - 0.02 * log10(area_cm2) for area > 1 cm^2.
    At 1 cm^2 (coin): f=1.0. At 100 cm^2: f≈0.96.
    """
    if area_cm2 <= 1.0:
        return 1.0
    return max(0.85, 1.0 - 0.02 * np.log10(area_cm2))


def npn_ratio_correction(npn_ratio: float, target_npn: float = 1.1) -> float:
    """Optimal N/P ratio for graphite is ~1.1.
    Excess graphite (high N/P) improves first-cycle efficiency but wastes capacity.
    Deficit (low N/P) causes Li plating -> lower recovery.

    Model: f_npn peaks at N/P=1.1, drops linearly away from it.
    """
    if npn_ratio <= 0:
        return 0.5
    # Parabolic penalty
    deviation = abs(npn_ratio - target_npn) / target_npn
    return max(0.80, 1.0 - deviation * 0.3)


def rate_correction_factor(c_rate: float, coin_c_rate: float = 0.5) -> float:
    """Higher C-rate → more polarization → lower accessible capacity.

    Approximate: capacity ~ 1 - 0.15 * log10(c_rate / coin_c_rate) for c_rate >= 0.5C.
    """
    if c_rate <= coin_c_rate:
        return 1.0
    penalty = 0.15 * np.log10(c_rate / coin_c_rate)
    return max(0.70, 1.0 - penalty)


def temperature_correction(t_C: float, t_ref_C: float = 25.0) -> float:
    """Arrhenius correction for low-temperature operation.
    Every 10°C below 25°C reduces kinetics by ~30%.
    """
    if t_C >= t_ref_C:
        return 1.0
    delta_t = t_ref_C - t_C
    return max(0.50, 1.0 - 0.03 * delta_t)


# --------------------------------------------------------------------------- #
# Pouch cell model
# --------------------------------------------------------------------------- #

def predict_pouch_recovery(
    coin_recovery_pct: float = 95.0,
    pouch_area_cm2:   float = 180.0,   # 3 Ah pouch: ~10 cm × 18 cm
    npn_ratio:        float = 1.15,
    c_rate:            float = 0.5,
    temperature_C:     float = 25.0,
    coin_area_cm2:    float = 1.0,
    coin_c_rate:      float = 0.5,
) -> dict:
    f_area = area_correction_factor(pouch_area_cm2)
    f_npn  = npn_ratio_correction(npn_ratio)
    f_rate = rate_correction_factor(c_rate, coin_c_rate)
    f_temp = temperature_correction(temperature_C)

    f_total = f_area * f_npn * f_rate * f_temp

    recovery = coin_recovery_pct * f_total

    return {
        "coin_recovery_pct":       coin_recovery_pct,
        "pouch_area_cm2":          pouch_area_cm2,
        "npn_ratio":               npn_ratio,
        "c_rate":                  c_rate,
        "temperature_C":            temperature_C,
        "f_area":                 round(f_area, 5),
        "f_npn":                  round(f_npn, 5),
        "f_rate":                 round(f_rate, 5),
        "f_temperature":          round(f_temp, 5),
        "f_total":               round(f_total, 5),
        "predicted_recovery_pct": round(recovery, 2),
        "reported_recovery_pct":  90.3,    # Kalra 2026 experimental
    }


def simulate_electrode_thickness(
    thickness_nm: float,
    area_cm2: float,
    density_g_cm3: float = 4.5,    # NMC density
    loading_mg_cm2: float = 20.0,  # areal loading
) -> dict:
    """Compute mass loading for pouch-scale electrode."""
    mass_per_area_g_cm2 = loading_mg_cm2 / 1000.0   # mg/cm2 → g/cm2
    active_mass_g = mass_per_area_g_cm2 * area_cm2
    volume_cm3    = active_mass_g / density_g_cm3
    thickness_cm  = volume_cm3 / area_cm2
    del thickness_cm  # needed for readability, not returned
    return {
        "active_mass_g":         round(active_mass_g, 4),
        "volume_cm3":            round(volume_cm3, 5),
        "loading_mg_cm2":        loading_mg_cm2,
        "density_g_cm3":         density_g_cm3,
    }


def comparison_table() -> list:
    """Compare coin cell vs pouch cell vs EV module."""
    configs = [
        {"label": "Coin cell (CR2032)", "area_cm2": 1.0,   "npn": 1.05, "c_rate": 0.5, "temp": 25.0},
        {"label": "Pouch 1 Ah",         "area_cm2": 50.0,  "npn": 1.10, "c_rate": 0.5, "temp": 25.0},
        {"label": "Pouch 3 Ah",         "area_cm2": 180.0, "npn": 1.15, "c_rate": 0.5, "temp": 25.0},
        {"label": "Pouch 3 Ah (cold)",  "area_cm2": 180.0, "npn": 1.15, "c_rate": 0.5, "temp": 10.0},
        {"label": "Pouch 3 Ah (high C)","area_cm2": 180.0, "npn": 1.15, "c_rate": 1.0, "temp": 25.0},
    ]
    rows = []
    for cfg in configs:
        r = predict_pouch_recovery(
            coin_recovery_pct=95.0,
            pouch_area_cm2=cfg["area_cm2"],
            npn_ratio=cfg["npn"],
            c_rate=cfg["c_rate"],
            temperature_C=cfg["temp"],
        )
        rows.append({
            "label":                   cfg["label"],
            "area_cm2":                cfg["area_cm2"],
            "npn_ratio":               cfg["npn"],
            "c_rate":                  cfg["c_rate"],
            "temperature_C":           cfg["temp"],
            "predicted_recovery_pct":   r["predicted_recovery_pct"],
            "f_total":                 r["f_total"],
            "f_area":                  r["f_area"],
            "f_npn":                   r["f_npn"],
            "f_rate":                  r["f_rate"],
            "f_temperature":           r["f_temperature"],
        })
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="3 Ah pouch cell scale-up validation")
    ap.add_argument("--coin-recovery",  type=float, default=95.0)
    ap.add_argument("--pouch-area",      type=float, default=180.0)
    ap.add_argument("--npn",            type=float, default=1.15)
    ap.add_argument("--c-rate",         type=float, default=0.5)
    ap.add_argument("--temp",           type=float, default=25.0)
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "pouch_scale_validation.json"))
    ap.add_argument("--out-csv",  default=str(RESULTS_DIR / "pouch_comparison.csv"))
    args = ap.parse_args()

    log.info("Pouch validation: area=%.0f cm^2, N/P=%.2f, C=%.1f, T=%d°C",
             args.pouch_area, args.npn, args.c_rate, int(args.temp))

    result = predict_pouch_recovery(
        coin_recovery_pct=args.coin_recovery,
        pouch_area_cm2=args.pouch_area,
        npn_ratio=args.npn,
        c_rate=args.c_rate,
        temperature_C=args.temp,
    )

    write_json(Path(args.out_json), result)

    comparison = comparison_table()
    write_csv(Path(args.out_csv), comparison)

    log.info("  Predicted recovery: %.1f%%  (vs Kalra reported: %.1f%%)",
             result["predicted_recovery_pct"], result["reported_recovery_pct"])
    log.info("  Scale factors: area=%.4f  npn=%.4f  rate=%.4f  temp=%.4f  total=%.4f",
             result["f_area"], result["f_npn"], result["f_rate"],
             result["f_temperature"], result["f_total"])
    log.info("JSON: %s", args.out_json)
    log.info("CSV:  %s", args.out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
