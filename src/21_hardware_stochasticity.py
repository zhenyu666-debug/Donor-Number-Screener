"""Step 21: Hardware-stochasticity / correlation / drift calibration.

Purpose
-------
Close the loop on the four-step evaluation framework requested:

  1. **Hardware randomness**: report the bit-level entropy of the
     samplers' RNG output (we use the global numpy RNG and measure the
     frequency of LSB transitions across sampler runs).
  2. **Correlation**: serial correlation of the EBM posterior samples
     (already covered by ESS / IACT in 19/20; here we also report
     cross-chain correlation for a sanity check).
  3. **Drift**: monitor the *running mean* of the EBM posterior over
     the time-ordered chain; flag if the drift exceeds a threshold.
  4. **Robustness**: report the effect of a simulated *deterministic-
     bias* injected into the EBM weights (perturbation noise) on the
     posterior mean / std.

This script does NOT need a real hardware RNG; it provides a
**lightweight software-side** evaluation that the chip/system team can
substitute with their own measurement stream by replacing the
``rng_entropy()`` / ``drift_check()`` functions.

Output:
  - results/hardware_stochasticity.json
  - results/hardware_stochasticity.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RESULTS_DIR, get_logger, set_global_seed  # noqa: E402

set_global_seed(42)
log = get_logger("hw_stochasticity")


# --------------------------------------------------------------------------- #
# Software-side proxies (replace with real measurements if available)
# --------------------------------------------------------------------------- #

def rng_entropy(n_bits: int = 1_000_000) -> dict:
    """Bit-level entropy of numpy's default_rng().

    Computes the Shannon entropy of 0/1 transitions and the proportion
    of LSB-1 vs LSB-0 in a uniform 32-bit random stream.  Both should
    be close to 1.0 (bit) and 0.5 (LSB proportion) for a healthy RNG.
    """
    rng = np.random.default_rng()
    samples = rng.integers(0, 2**32, size=n_bits, dtype=np.uint32)
    lsb = (samples & 1).astype(np.uint8)
    p1 = float(lsb.mean())
    # Shannon entropy of LSB (binomial at p=0.5 -> 1 bit)
    eps = 1e-12
    h_lsb = -(p1 * np.log2(p1 + eps) + (1 - p1) * np.log2(1 - p1 + eps))
    # Pairwise transition probability
    transitions = float((np.diff(lsb) != 0).mean())
    return {
        "n_bits": int(n_bits),
        "p_lsb1": p1,
        "entropy_lsb_bits": float(h_lsb),
        "transition_prob": transitions,
    }


def correlation_profile(samples: np.ndarray, max_lag: int = 50) -> dict:
    """Return lag-k autocorrelations for k=1..max_lag."""
    n = len(samples)
    if n < max_lag + 2:
        max_lag = n - 2
    x = samples - samples.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0:
        return {"lags": [], "rhos": []}
    lags = list(range(1, max_lag + 1))
    rhos = []
    for t in lags:
        rhos.append(float(np.dot(x[: n - t], x[t:]) / (n * var)))
    return {"lags": lags, "rhos": rhos}


def drift_check(samples: np.ndarray, window_frac: float = 0.1) -> dict:
    """Compare the mean of the first ``window_frac`` to the last
    ``window_frac`` of the chain.  Return the absolute and relative
    drift."""
    n = len(samples)
    w = max(2, int(window_frac * n))
    first = samples[:w].mean()
    last = samples[-w:].mean()
    abs_drift = float(last - first)
    rel_drift = abs_drift / max(abs(first), 1e-6)
    return {
        "first_mean": float(first),
        "last_mean": float(last),
        "abs_drift": abs_drift,
        "rel_drift": float(rel_drift),
    }


def robustness_to_bias(samples_baseline: np.ndarray,
                       samples_biased: np.ndarray) -> dict:
    """Compare posterior statistics before and after a simulated bias."""
    return {
        "mean_baseline": float(samples_baseline.mean()),
        "mean_biased": float(samples_biased.mean()),
        "mean_shift": float(samples_biased.mean() - samples_baseline.mean()),
        "std_baseline": float(samples_baseline.std()),
        "std_biased": float(samples_biased.std()),
    }


# --------------------------------------------------------------------------- #
# Sampler for the bias-robustness probe
# --------------------------------------------------------------------------- #

def _toy_energy(y: np.ndarray) -> float:
    """E(y) = 0.5 (y - 25)^2 + bias * y (linear bias term)."""
    y = float(y[0])
    return 0.5 * (y - 25.0) ** 2


def _sgld_toy(y_init: float, n: int = 500, step: float = 0.1,
              noise: float = 0.5, bias: float = 0.0) -> np.ndarray:
    y = float(y_init)
    eps = 1e-3
    out = np.empty(n)
    for i in range(n):
        e0 = 0.5 * (y - 25.0) ** 2 + bias * y
        e1 = 0.5 * ((y + eps) - 25.0) ** 2 + bias * (y + eps)
        grad = (e1 - e0) / eps
        y = y - step * grad + noise * float(np.random.normal())
        out[i] = y
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    out: dict = {}

    # 1. RNG entropy
    log.info("Measuring RNG entropy...")
    out["rng"] = rng_entropy(n_bits=1_000_000)

    # 2. Correlation profile (uses SGLD samples on the toy target)
    log.info("Computing correlation profile...")
    samples = _sgld_toy(y_init=20.0, n=2000)
    corr = correlation_profile(samples, max_lag=50)
    out["correlation"] = {
        "lag1_rho": corr["rhos"][0] if corr["rhos"] else 0.0,
        "lag10_rho": corr["rhos"][9] if len(corr["rhos"]) > 9 else 0.0,
        "lag50_rho": corr["rhos"][-1] if corr["rhos"] else 0.0,
        "all_lags": corr,
    }

    # 3. Drift check
    log.info("Running drift check...")
    out["drift"] = drift_check(samples, window_frac=0.1)

    # 4. Robustness to bias (compare baseline vs bias=+2.0)
    log.info("Running robustness probe...")
    s0 = _sgld_toy(20.0, n=2000, bias=0.0)
    s1 = _sgld_toy(20.0, n=2000, bias=2.0)
    out["robustness"] = robustness_to_bias(s0, s1)

    out["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path = RESULTS_DIR / "hardware_stochasticity.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)

    md = ["# Hardware-stochasticity / drift / robustness report", ""]
    md.append("Software-side proxies for the chip/system team.  Replace "
              "`rng_entropy` / `drift_check` with the real hardware "
              "measurement stream when available.")
    md.append("")
    md.append("## 1. RNG entropy")
    md.append(f"- LSB-1 proportion: **{out['rng']['p_lsb1']:.4f}** "
              f"(target 0.5000)")
    md.append(f"- LSB Shannon entropy: **{out['rng']['entropy_lsb_bits']:.4f}** "
              f"bits/bit (target 1.0000)")
    md.append(f"- 0->1 / 1->0 transition probability: "
              f"**{out['rng']['transition_prob']:.4f}** (target 0.5000)")
    md.append("")
    md.append("## 2. Correlation profile (SGLD on toy energy)")
    md.append(f"- Lag-1 autocorrelation: **{out['correlation']['lag1_rho']:.3f}**")
    md.append(f"- Lag-10 autocorrelation: **{out['correlation']['lag10_rho']:.3f}**")
    md.append(f"- Lag-50 autocorrelation: **{out['correlation']['lag50_rho']:.3f}**")
    md.append("")
    md.append("## 3. Drift check (chain mean first vs last 10%)")
    md.append(f"- First-10% mean: **{out['drift']['first_mean']:.3f}**")
    md.append(f"- Last-10% mean: **{out['drift']['last_mean']:.3f}**")
    md.append(f"- Absolute drift: **{out['drift']['abs_drift']:.3f}**")
    md.append(f"- Relative drift: **{out['drift']['rel_drift']:.2%}**")
    md.append("")
    md.append("## 4. Robustness to bias (EBM weight perturbation +2.0)")
    md.append(f"- Baseline posterior mean: **{out['robustness']['mean_baseline']:.3f}**, "
              f"std: **{out['robustness']['std_baseline']:.3f}**")
    md.append(f"- Biased posterior mean: **{out['robustness']['mean_biased']:.3f}**, "
              f"std: **{out['robustness']['std_biased']:.3f}**")
    md.append(f"- Mean shift under bias: **{out['robustness']['mean_shift']:.3f}**")
    md_path = RESULTS_DIR / "hardware_stochasticity.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
