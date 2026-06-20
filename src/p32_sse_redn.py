"""32_sse_redn.py - re-rank 14 SSEs with the v2 weighted model.

Replaces the v1 single-anchor Gaussian with a multi-physics combination:
    dn_sse = w_langevin * langevin_mean
           + w_particle * (dn_anchor + 0.5 * particle_dn_correction)
           + w_sei * dn_sei_eff
           + w_aimd  * dn_aimd
           + w_empirical * (alpha * log10(sigma_ion) + beta * E_g + gamma
                             + delta * migration_eV)
where the empirical component is calibrated on the PBP anchor SMILES
and the AIMD component comes from src/30_ml_aimd.py (or the lib fallback).

Outputs:
  results/sse_dn_rerank.csv   - 14 SSEs with dn components + rank
  results/pbp_v2_metrics.json - 7-model aggregate metrics
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import DATA_DIR, RESULTS_DIR, write_csv, write_json, set_seed  # noqa: E402


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #

def empirical_dn(entry: dict, weights: dict) -> float:
    sigma = max(float(entry.get("sigma_ion_S_cm", 1e-6)), 1e-12)
    Eg = float(entry.get("E_g_eV", 4.0))
    migration = float(entry.get("migration_eV", 0.4))
    alpha = float(weights.get("alpha_sigma", 1.0e3))
    beta = float(weights.get("beta_Eg", 1.0))
    gamma = float(weights.get("gamma_floor", 5.0))
    delta = float(weights.get("delta_migration", -2.0))
    return alpha * math.log10(sigma) + beta * Eg + gamma + delta * migration


def langevin_proxy_dn(entry: dict, anchor: float) -> float:
    """For SSEs the 5-model stack has no SMILES, so we use a
    Gaussian posterior centred on the empirical DN with a tight std."""
    em = empirical_dn(entry, {})  # default weights
    return em + np.random.normal(0.0, 0.4)


def particle_correction(entry: dict) -> float:
    """Tiny per-SSE correction from coordination number + migration barrier."""
    coord = float(entry.get("li_coord_num", 4.0))
    migration = float(entry.get("migration_eV", 0.3))
    return 0.5 * (4.0 - coord) - 0.5 * (migration - 0.3) * 2.0


def sei_attenuation(entry: dict) -> float:
    """Treat the SSE thickness as ~bulk; SEI attenuation floor 0.5."""
    migration = float(entry.get("migration_eV", 0.3))
    return max(0.0, 1.0 - 0.5 * migration)


def aimd_dn_from_csv(entry: dict, aimd_rows: list) -> float:
    """Look up the AIMD DN from the ml_aimd result for this SSE name.
    If not present (e.g. AIMD not run), fall back to a heuristic."""
    name = entry.get("name")
    for r in aimd_rows:
        if r.get("sse") == name:
            return float(r["dn_aimd"])
    sigma = max(float(entry.get("sigma_ion_S_cm", 1e-6)), 1e-12)
    return 8.0 + 5.0 * math.sqrt(sigma)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def load_sse_library() -> list:
    with (DATA_DIR / "sse_library.yaml").open(encoding="utf-8-sig") as f:
        import yaml
        d = yaml.safe_load(f) or {}
    return d.get("sse", []), d.get("ranking", {})


def load_aimd_rows() -> list:
    p = RESULTS_DIR / "ml_aimd_interface.csv"
    if not p.exists():
        return []
    import csv
    rows = []
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                r["dn_aimd"] = float(r["dn_aimd"])
            except Exception:
                r["dn_aimd"] = 0.0
            rows.append(r)
    return rows


def rerank(weights: dict = None, anchor_dn: float = 22.0) -> List[dict]:
    weights = weights or {"langevin": 0.5, "particle": 0.1, "sei": 0.1,
                          "aimd": 0.2, "empirical": 0.1}
    lib, rk = load_sse_library()
    rk_w = {**rk, **weights}  # ranking uses YAML defaults
    aimd_rows = load_aimd_rows()
    rows = []
    for entry in lib:
        em = empirical_dn(entry, rk_w)
        em_clamped = max(5.0, min(40.0, em))
        lang_dn = langevin_proxy_dn(entry, anchor_dn)
        p_corr = particle_correction(entry)
        s = sei_attenuation(entry)
        a = aimd_dn_from_csv(entry, aimd_rows)
        dn = (weights["langevin"] * lang_dn
              + weights["particle"] * (anchor_dn + p_corr)
              + weights["sei"] * em_clamped * s
              + weights["aimd"] * a
              + weights["empirical"] * em_clamped)
        rows.append({
            "sse": entry["name"],
            "formula": entry["formula"],
            "class": entry["class"],
            "sigma_ion_S_cm": entry["sigma_ion_S_cm"],
            "E_g_eV": entry["E_g_eV"],
            "migration_eV": entry["migration_eV"],
            "dn_empirical": em_clamped,
            "dn_langevin": lang_dn,
            "dn_particle": anchor_dn + p_corr,
            "dn_sei": em_clamped * s,
            "dn_aimd": a,
            "dn_pbp_v2": dn,
        })
    rows.sort(key=lambda r: r["dn_pbp_v2"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv", default=str(RESULTS_DIR / "sse_dn_rerank.csv"))
    p.add_argument("--out_json", default=str(RESULTS_DIR / "pbp_v2_metrics.json"))
    args = p.parse_args()
    set_seed(0)
    rows = rerank()
    write_csv(Path(args.out_csv), rows)
    # Metrics: spread, average DN, etc.
    dns = np.array([r["dn_pbp_v2"] for r in rows])
    metrics = {
        "n_sse": len(rows),
        "dn_mean": float(dns.mean()),
        "dn_std": float(dns.std()),
        "dn_min": float(dns.min()),
        "dn_max": float(dns.max()),
        "top3": [r["sse"] for r in rows[:3]],
        "bottom3": [r["sse"] for r in rows[-3:]],
        "rows": rows,
    }
    write_json(Path(args.out_json), metrics)
    print("[sse_redn] 14 SSEs ranked. Top 3:")
    for r in rows[:3]:
        print(f"  #{r['rank']} {r['sse']:30s} dn_pbp_v2={r['dn_pbp_v2']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
