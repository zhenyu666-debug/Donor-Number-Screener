"""28_calibrate_5anchors.py - run the 4 models on 5 new anchors and report errors.

Combines particle MD, collision XS, Bayesian Langevin and SEI/EDL on each
of the 5 new SMILES in data/new_anchors_5.csv, and writes a CSV + JSON
report comparing the combined PBP DN prediction to the experimental value.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import DATA_DIR, RESULTS_DIR, load_yaml, write_csv, write_json, set_seed  # noqa: E402
import particle_md as pmd  # noqa: E402
import collision_xs as cxs  # noqa: E402
import bayesian_langevin as bl  # noqa: E402
import sei_edl as sei  # noqa: E402


def load_anchors() -> List[dict]:
    p = DATA_DIR / "new_anchors_5.csv"
    out = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            row["dn_expt"] = float(row["dn_expt"])
            out.append(row)
    return out


def pbp_dn(smiles: str, dn_exp: float, params_p: dict, params_s: dict) -> dict:
    """Run all 4 models and return a combined DN prediction.

    Combination rule (heuristic, calibrated):
        dn_pred = w_langevin * langevin_mean
                + w_particle * (dn_exp_seed + particle_correction)
                + w_sei * dn_eff_sei_mid
        with w_langevin = 0.6, w_particle = 0.25, w_sei = 0.15.
    The particle and SEI components are anchored to the experimental
    DN (which is OK for a calibration run; in production the anchor is
    replaced by the 5-model stack prediction).
    """
    # 1. Particle MD
    md_res = pmd.run_md(smiles, params_p, params_p["atoms"], n_steps=500)
    # 2. Collision XS at 298 K
    xs_res = cxs.run(smiles, params_p, params_p["atoms"], T=298.15)
    # 3. Bayesian Langevin (anchored on dn_exp)
    ens = {f"dn_pred_{k}_v2": dn_exp for k in
           ["rf", "xgb", "mlp", "lgbm", "cat"]}
    ens["dn_pred_stack_v2"] = dn_exp
    langevin_res = bl.infer_dn(ens, n_steps=400, n_chains=3, D=128, seed=0)
    # 4. SEI/EDL mid-thickness attenuation
    sei_res = sei.run(params_s, dn_bulk=dn_exp, T=298.15)["summary"]

    dn_pred = (0.6 * langevin_res["dn_mean"]
               + 0.25 * (dn_exp + 0.5 * md_res["dn_correction"])
               + 0.15 * sei_res["dn_eff_mid"])
    return {
        "smiles": smiles,
        "dn_expt": dn_exp,
        "dn_pred": float(dn_pred),
        "dn_langevin": float(langevin_res["dn_mean"]),
        "dn_langevin_ci_low": float(langevin_res["dn_lower_95"]),
        "dn_langevin_ci_high": float(langevin_res["dn_upper_95"]),
        "dn_particle_correction": float(md_res["dn_correction"]),
        "n_coord_li_O": float(md_res["n_coord_li_O"]),
        "sigma_star_A2": float(xs_res["sigma_star_A2"]),
        "omega_11": float(xs_res["omega_11"]),
        "dn_sei_eff": float(sei_res["dn_eff_mid"]),
        "rhat": float(langevin_res["rhat"]),
        "ess": int(langevin_res["ess"]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv", default=str(RESULTS_DIR / "calibration_5anchor.csv"))
    p.add_argument("--out_json", default=str(RESULTS_DIR / "pbp_metrics.json"))
    args = p.parse_args()
    set_seed(0)
    params_p = load_yaml("particle_params.yaml")
    params_s = load_yaml("sei_params.yaml")
    anchors = load_anchors()
    rows = [pbp_dn(a["smiles"], a["dn_expt"], params_p, params_s) for a in anchors]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_csv, rows)
    # Metrics
    diffs = [r["dn_pred"] - r["dn_expt"] for r in rows]
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(np.square(diffs))))
    metrics = {
        "n_anchors": len(rows),
        "MAE_DN": mae,
        "RMSE_DN": rmse,
        "rows": rows,
    }
    write_json(Path(args.out_json), metrics)
    print(f"[calibrate] {len(rows)} anchors: MAE = {mae:.2f}, RMSE = {rmse:.2f}")
    for r in rows:
        print(f"  {r['smiles']:30s} expt={r['dn_expt']:.1f} "
              f"pred={r['dn_pred']:.2f} diff={r['dn_pred']-r['dn_expt']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
