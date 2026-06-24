"""p40e_solvent_uncertainty.py - Bayesian/bootstrap uncertainty quantification for DEER solvent screening.

Adds uncertainty bands to the EEI dissolution scores from p40_solvent_screening.py
using two complementary approaches:
  1. Bootstrap Resampling  (N=100 noisy physics_score replicas)
  2. Ensemble RF Spread     (prediction variance from existing RF models, if available)

Outputs: results/solvent_uncertainty.json
         figures/deer_solvent_uncertainty.png
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils import get_logger, set_global_seed, RESULTS_DIR, FIGURES_DIR  # noqa: E402
from utils_pb import write_json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger("p40e_solvent_uncertainty")

# --------------------------------------------------------------------------- #
# Colour palette
# --------------------------------------------------------------------------- #
DEER_BLUE   = "#1A6EBF"
DEER_ORANGE = "#E8630A"
GRAY        = "#999999"

# --------------------------------------------------------------------------- #
# Reference values -- MUST match p40_solvent_screening.py
# --------------------------------------------------------------------------- #
DN_NORM_BOT = 15.0
DN_NORM_TOP = 29.0

# --------------------------------------------------------------------------- #
# Physics score (copied verbatim from p40_solvent_screening.py)
# --------------------------------------------------------------------------- #
def physics_score(dn: float, an: float, epsilon_r: float,
                  oxidation_V: float, reduction_V: float,
                  viscosity_cp: float = 2.0,
                  melting_point_c: float = -100.0) -> dict:
    dn_norm = max(0.0, min(1.0, (dn - DN_NORM_BOT) / (DN_NORM_TOP - DN_NORM_BOT)))

    _eps = max(0.0, epsilon_r - 35.0)
    eps_comp = max(0.0, 1.0 - 0.015 * _eps)
    if epsilon_r > 80.0:
        eps_comp = 0.0
    elif epsilon_r > 60.0:
        eps_comp = min(eps_comp, 0.2)

    ox_factor = min(1.0, max(0.0, (oxidation_V - 3.5) / 1.3))
    red_factor = min(1.0, max(0.0, (reduction_V - 0.1) / 0.9))
    visc_abs = max(abs(viscosity_cp), 0.1)
    visc_factor = min(1.2, 2.1 / visc_abs)

    raw = (
        dn_norm     * 0.43 +
        eps_comp    * 0.27 +
        ox_factor  * 0.20 +
        red_factor * 0.05 +
        visc_factor * 0.05
    )
    eps_mult = 0.5 if epsilon_r > 80.0 else 1.0
    raw = raw * eps_mult
    eei_diss = max(0.0, min(1.0, raw))

    dn_penalty = 0.0 if dn <= 32.0 else (dn - 32.0) * 0.04
    compat = max(0.0, min(1.0, 0.89 - dn_penalty - (1.0 - ox_factor) * 0.10))

    regen = max(0.0, min(1.0, (eei_diss ** 0.7) * (compat ** 0.3) * 0.98))
    regen_pct = round(regen * 100.0, 1)

    return {
        "eei_dissolution_score":       round(eei_diss, 4),
        "electrode_compat_score":      round(compat, 4),
        "regeneration_potential_pct":  regen_pct,
    }


def physics_score_noise(dn: float, an: float, epsilon_r: float,
                        oxidation_V: float, reduction_V: float,
                        viscosity_cp: float = 2.0,
                        melting_point_c: float = -100.0,
                        noise_std: float = 0.03) -> dict:
    result = physics_score(dn, an, epsilon_r, oxidation_V, reduction_V,
                          viscosity_cp, melting_point_c)
    rng = np.random.default_rng()
    result["eei_dissolution_score"] = float(np.clip(
        result["eei_dissolution_score"] + rng.normal(0.0, noise_std), 0.0, 1.0))
    return result


# --------------------------------------------------------------------------- #
# Bootstrap resampling
# --------------------------------------------------------------------------- #
N_BOOTSTRAP = 100
NOISE_STD   = 0.025


def bootstrap_scores(row: dict, n: int = N_BOOTSTRAP,
                     noise_std: float = NOISE_STD) -> dict:
    rng = np.random.default_rng()

    diss_scores   = np.empty(n)
    compat_scores = np.empty(n)
    regen_pcts    = np.empty(n)

    dn   = float(row.get("dn", 20.0))
    an_v = float(row.get("an", 10.0))
    eps  = float(row.get("epsilon_r", 30.0))
    ox   = float(row.get("oxidation_stability_V", 4.2))
    red  = float(row.get("reduction_stability_V", 0.8))
    visc = float(row.get("viscosity_cp", 2.0))
    mp   = float(row.get("melting_point_c", -100.0))

    for i in range(n):
        noise = rng.normal(0.0, noise_std, size=5)

        dn_n   = max(0.0, dn   + noise[0] * 2.5)
        eps_n  = max(1.0, eps  + noise[1] * 5.0)
        ox_n   = max(3.0, ox   + noise[2] * 0.2)
        red_n  = max(0.0, red  + noise[3] * 0.15)
        visc_n = max(0.1, visc + noise[4] * 0.2)

        r = physics_score(dn_n, an_v, eps_n, ox_n, red_n, visc_n, mp)
        diss_scores[i]    = r["eei_dissolution_score"]
        compat_scores[i] = r["electrode_compat_score"]
        regen_pcts[i]     = r["regeneration_potential_pct"]

    diss_arr   = np.array(diss_scores)
    compat_arr = np.array(compat_scores)
    regen_arr  = np.array(regen_pcts)

    w_d, w_c, w_r = 0.50, 0.30, 0.20
    comp_scores = w_d * diss_arr + w_c * compat_arr + w_r * (regen_arr / 100.0)

    return {
        "eei_dissolution_score_mean":   round(float(diss_arr.mean()), 6),
        "eei_dissolution_score_std":   round(float(diss_arr.std()),  6),
        "eei_dissolution_score_ci_lo": round(float(np.percentile(diss_arr,  2.5)), 6),
        "eei_dissolution_score_ci_hi": round(float(np.percentile(diss_arr, 97.5)), 6),
        "electrode_compat_score_mean":   round(float(compat_arr.mean()), 6),
        "electrode_compat_score_std":   round(float(compat_arr.std()),  6),
        "regeneration_potential_mean":   round(float(regen_arr.mean()),  6),
        "regeneration_potential_std":   round(float(regen_arr.std()),   6),
        "composite_score_mean":         round(float(comp_scores.mean()), 6),
        "composite_score_std":          round(float(comp_scores.std()),  6),
    }


# --------------------------------------------------------------------------- #
# Ensemble RF spread (optional -- requires sklearn)
# --------------------------------------------------------------------------- #
def ensemble_rf_spread(df: pd.DataFrame, row_dict: dict) -> dict | None:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return None

    TARGETS = ["eei_dissolution_score", "electrode_compat_score",
               "regeneration_potential_pct"]
    anchor = df.dropna(subset=["source_note"])
    if len(anchor) < 10:
        return None

    feat_cols = [c for c in [
        "dn", "an", "epsilon_r",
        "oxidation_stability_V", "reduction_stability_V", "viscosity_cp",
    ] if c in anchor.columns]

    X = anchor[feat_cols].fillna(0.0)
    preds: dict = {}

    for tgt in TARGETS:
        y = anchor[tgt]
        if y.nunique() < 3:
            continue
        trees = []
        for seed in range(10):
            rf = RandomForestRegressor(
                n_estimators=20, max_depth=6,
                random_state=seed, n_jobs=1,
            )
            rf.fit(X, y)
            trees.append(rf)
        x_new = np.array([[row_dict.get(c, 0.0) for c in feat_cols]])
        tree_preds = np.array([float(t.predict(x_new)[0]) for t in trees])
        preds[tgt] = tree_preds

    if not preds:
        return None

    return {
        tgt: {"mean": round(float(v.mean()), 6), "std": round(float(v.std()), 6)}
        for tgt, v in preds.items()
    }


# --------------------------------------------------------------------------- #
# Colour helper
# --------------------------------------------------------------------------- #
def solvent_color(name: str) -> str:
    n = str(name).lower()
    if "dmi" in n or "n,n-dimethyl-2-imidazolidinone" in n:
        return DEER_BLUE
    if "dmso" in n or "dimethyl sulfoxide" in n:
        return DEER_ORANGE
    return GRAY


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="DEER solvent uncertainty quantification",
    )
    ap.add_argument(
        "--input", default=str(RESULTS_DIR / "solvent_eei_predictions.csv"),
    )
    ap.add_argument(
        "--n-bootstrap", type=int, default=N_BOOTSTRAP,
        help="Number of bootstrap replicas (default 100)",
    )
    ap.add_argument(
        "--noise-std", type=float, default=NOISE_STD,
        help="Gaussian noise std per replica (default 0.025)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)
    np.random.seed(args.seed)

    log.info("Loading: %s", args.input)
    df = pd.read_csv(Path(args.input))
    log.info("  %d solvents loaded", len(df))

    results = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        name = str(row_dict.get("name", f"solvent_{idx}"))

        set_global_seed(args.seed + idx)
        bs_stats = bootstrap_scores(
            row_dict, n=args.n_bootstrap, noise_std=args.noise_std,
        )
        rf_stats = ensemble_rf_spread(df, row_dict)

        entry: dict = {
            "name":                              name,
            "dn":                                float(row_dict.get("dn", 0.0)),
            "eei_dissolution_score_mean":        bs_stats["eei_dissolution_score_mean"],
            "eei_dissolution_score_std":         bs_stats["eei_dissolution_score_std"],
            "eei_dissolution_score_ci_lo":       bs_stats["eei_dissolution_score_ci_lo"],
            "eei_dissolution_score_ci_hi":       bs_stats["eei_dissolution_score_ci_hi"],
            "electrode_compat_score_mean":        bs_stats["electrode_compat_score_mean"],
            "electrode_compat_score_std":         bs_stats["electrode_compat_score_std"],
            "regeneration_potential_mean":        bs_stats["regeneration_potential_mean"],
            "regeneration_potential_std":         bs_stats["regeneration_potential_std"],
            "composite_score_mean":              bs_stats["composite_score_mean"],
            "composite_score_std":               bs_stats["composite_score_std"],
        }
        if rf_stats:
            entry["rf_ensemble_spread"] = rf_stats

        results.append(entry)

        if idx < 5 or idx % 20 == 0:
            log.info(
                "  [%3d] %-30s  diss=%.4f +/- %.4f  CI=[%.4f, %.4f]",
                idx, name[:30],
                entry["eei_dissolution_score_mean"],
                entry["eei_dissolution_score_std"],
                entry["eei_dissolution_score_ci_lo"],
                entry["eei_dissolution_score_ci_hi"],
            )

    # Write JSON
    out_json = RESULTS_DIR / "solvent_uncertainty.json"
    write_json(out_json, {
        "solvents":    results,
        "n_solvents":  len(results),
        "n_bootstrap": args.n_bootstrap,
        "noise_std":   args.noise_std,
    })
    log.info("Results written: %s", out_json)

    # ------------------------------------------------------------------ #
    # Figure: DN vs dissolution score with 95% CI error bars
    # ------------------------------------------------------------------ #
    fig_path = FIGURES_DIR / "deer_solvent_uncertainty.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)

    xs, ys, yerr_lo, yerr_hi, cols = [], [], [], [], []
    for r in results:
        c = solvent_color(r["name"])
        xs.append(r["dn"])
        ys.append(r["eei_dissolution_score_mean"])
        half_ci = (
            r["eei_dissolution_score_ci_hi"]
            - r["eei_dissolution_score_ci_lo"]
        ) / 2.0
        yerr_lo.append(half_ci)
        yerr_hi.append(half_ci)
        cols.append(c)

    for i in range(len(xs)):
        ax.errorbar(
            xs[i], ys[i],
            yerr=[[yerr_lo[i]], [yerr_hi[i]]],
            fmt="o", capsize=3, capthick=0.8,
            elinewidth=0.8, markersize=4,
            color=cols[i], ecolor=cols[i], alpha=0.55,
        )

    for r in results:
        c = solvent_color(r["name"])
        if c != GRAY:
            ax.scatter([], [], marker="o", s=60, color=c, label=r["name"][:35])

    ax.set_xlabel("Donor Number (kcal mol\u207b\u00b9)", fontsize=12)
    ax.set_ylabel("EEI Dissolution Score", fontsize=12)
    ax.set_title(
        "DEER Solvent Screening \u2014 Bootstrap 95% CI Uncertainty\n"
        f"(N={args.n_bootstrap} bootstrap replicas, noise_std={args.noise_std})",
        fontsize=12,
    )
    ax.set_xlim(10, 42)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels, loc="lower right", fontsize=9,
            framealpha=0.8, title="Key solvents",
        )

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved: %s", fig_path)

    log.info("Done. %d solvents processed.", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
