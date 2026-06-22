"""26_bayesian_langevin.py - 994-dim Bayesian Langevin diffusion sampler.

Method:
    Stochastic gradient Langevin dynamics on the log-posterior
        U(x) = EBM_energy(x) - log_prior(x)
    with update
        x_{k+1} = x_k - eps_k grad U(x_k) + sqrt(2 eps_k) xi_k
    Multiple independent chains; report R-hat, ESS, posterior mean, 95% CI.

Inputs:  feature vector x of length D (default 994 to match
         donor-number-screener's 5-model stack). The posterior is approximated
         as a Gaussian anchored on a 5-model stack prediction; we optionally
         load real ensemble predictions from a JSON file.

Outputs: chain samples, posterior mean, 95% CI, R-hat, ESS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import (  # noqa: E402
    RESULTS_DIR, write_json, gelman_rubin, set_seed,
)


# --------------------------------------------------------------------------- #
# Energy / posterior
# --------------------------------------------------------------------------- #

class GaussianPosterior:
    """Diagonal-Gaussian posterior proxy:

        U(x) = 0.5 * (x - mu)^T Sigma^{-1} (x - mu) + const

    mu is the 5-model stack prediction broadcasted to D, Sigma is diagonal
    (sigma^2) - same standard deviation in every direction.
    """

    def __init__(self, mu_d: np.ndarray, sigma: float = 1.5):
        self.mu = mu_d.astype(float)
        self.sigma = float(sigma)

    def energy(self, x: np.ndarray) -> np.ndarray:
        # x: (B, D), returns energy per row
        d = x - self.mu
        return 0.5 * np.sum((d / self.sigma) ** 2, axis=-1)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / (self.sigma ** 2)


class EBMEnergy:
    """Fallback toy EBM: a small MLP energy on a random projection of x.
    We deliberately keep it tiny so CPU eval is cheap.
    """

    def __init__(self, d_in: int, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.1, size=(d_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, 0.1, size=(hidden, 1))
        self.b2 = np.zeros(1)

    def energy(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out.squeeze(-1)

    def grad(self, x: np.ndarray) -> np.ndarray:
        # numerical gradient
        eps = 1e-3
        f0 = self.energy(x)
        grads = np.zeros_like(x)
        for i in range(x.shape[1]):
            x1 = x.copy()
            x1[:, i] += eps
            grads[:, i] = (self.energy(x1) - f0) / eps
        return grads


# --------------------------------------------------------------------------- #
# Langevin sampler
# --------------------------------------------------------------------------- #

def langevin_step(x: np.ndarray, energy_grad, eps: float, noise_scale: float) -> np.ndarray:
    """One SGLD step.  Returns the next state, same shape as x."""
    grad = energy_grad(x)
    noise = np.random.normal(0.0, 1.0, size=x.shape)
    return x - 0.5 * eps * grad + np.sqrt(eps) * noise_scale * noise


def run_chains(energy, x0: np.ndarray, n_steps: int, eps: float, n_chains: int = 4,
               burn_in: int = 200, seed: int = 0) -> dict:
    """Run multiple SGLD chains from x0 (replicated if needed) and return diagnostics."""
    rng = np.random.default_rng(seed)
    if x0.ndim == 1:
        x0 = np.tile(x0, (n_chains, 1))
    elif x0.shape[0] < n_chains:
        x0 = np.tile(x0, (n_chains // x0.shape[0] + 1, 1))[:n_chains]

    samples = np.zeros((n_chains, n_steps, x0.shape[1]))
    x = x0.copy()
    for k in range(n_steps):
        noise = rng.normal(0.0, 1.0, size=x.shape)
        grad = energy.grad(x)
        x = x - 0.5 * eps * grad + np.sqrt(eps) * noise
        samples[:, k] = x
    post = samples[:, burn_in:]
    flat = post.reshape(-1, post.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    # R-hat on the per-chain mean
    chain_means = post.mean(axis=1)
    rhat = gelman_rubin(chain_means)
    # ESS using lag-1 autocorrelation (rough)
    ess = max(1, int(flat.shape[0] / (1.0 + 2.0 * np.corrcoef(flat[:-1].mean(axis=1), flat[1:].mean(axis=1))[0, 1])))
    return {
        "mean": mean,
        "std": std,
        "rhat": rhat,
        "ess": ess,
        "samples": flat,
        "all_samples": samples,
    }


# --------------------------------------------------------------------------- #
# High-level DN inference
# --------------------------------------------------------------------------- #

def infer_dn(ensemble_pred: dict, D: int = 994, n_chains: int = 4, n_steps: int = 1500,
             eps: float = 5e-4, sigma_posterior: float = 1.5, seed: int = 0) -> dict:
    """Run a posterior over D-dim features and reduce to a 1-D DN posterior by
    averaging the first 5 dims (the 5-model stack) across all samples.

    ensemble_pred keys (any subset):
        dn_pred_rf_v2, dn_pred_xgb_v2, dn_pred_mlp_v2,
        dn_pred_lgbm_v2, dn_pred_cat_v2, dn_pred_stack_v2
    """
    set_seed(seed)
    keys = ["dn_pred_rf_v2", "dn_pred_xgb_v2", "dn_pred_mlp_v2",
            "dn_pred_lgbm_v2", "dn_pred_cat_v2"]
    available = [(k, float(ensemble_pred[k])) for k in keys
                if k in ensemble_pred and ensemble_pred[k] is not None]
    if not available:
        available = [("dn_pred_stack_v2", float(ensemble_pred.get("dn_pred_stack_v2", 20.0)))]
    avg = float(np.mean([v for _, v in available]))
    mu_d = np.full(D, fill_value=avg, dtype=float)
    mu_d[:5] = [v for _, v in available] + [avg] * (5 - len(available))

    posterior = GaussianPosterior(mu_d, sigma=sigma_posterior)
    ebm = EBMEnergy(D, hidden=32, seed=seed)
    combined = type("C", (), {})()  # simple namespace
    def grad(x):
        return posterior.grad(x) + 0.05 * ebm.grad(x)
    combined.grad = grad
    combined.energy = lambda x: posterior.energy(x) + 0.05 * ebm.energy(x)

    x0 = mu_d + np.random.normal(0.0, sigma_posterior, size=(n_chains, D))
    res = run_chains(combined, x0, n_steps=n_steps, eps=eps, n_chains=n_chains, burn_in=200)
    # DN estimate = mean of first 5 dims
    dn_samples = res["samples"][:, :5].mean(axis=1)
    dn_mean = float(dn_samples.mean())
    dn_std = float(dn_samples.std(ddof=1))
    lo, hi = float(np.quantile(dn_samples, 0.025)), float(np.quantile(dn_samples, 0.975))
    return {
        "dn_mean": dn_mean,
        "dn_std": dn_std,
        "dn_lower_95": lo,
        "dn_upper_95": hi,
        "rhat": float(res["rhat"]),
        "ess": int(res["ess"]),
        "n_chains": n_chains,
        "n_steps": n_steps,
        "burn_in": 200,
        "D": D,
        "ensemble_pred": ensemble_pred,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smiles", default="CCO")
    p.add_argument("--rf", type=float, default=20.0)
    p.add_argument("--xgb", type=float, default=21.0)
    p.add_argument("--mlp", type=float, default=20.5)
    p.add_argument("--lgbm", type=float, default=20.8)
    p.add_argument("--cat", type=float, default=20.3)
    p.add_argument("--stack", type=float, default=20.6)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--out", default=str(RESULTS_DIR / "langevin_samples.csv"))
    args = p.parse_args()
    ens = {"dn_pred_rf_v2": args.rf, "dn_pred_xgb_v2": args.xgb,
           "dn_pred_mlp_v2": args.mlp, "dn_pred_lgbm_v2": args.lgbm,
           "dn_pred_cat_v2": args.cat, "dn_pred_stack_v2": args.stack}
    res = infer_dn(ens, n_steps=args.steps, n_chains=args.chains)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save a sample-trace CSV (one column per chain)
    with out.open("w") as f:
        f.write("chain,step,dn_sample\n")
        for c in range(res["n_chains"]):
            trace = res["samples"][c::res["n_chains"]]  # rough split
            for i, v in enumerate(trace[:, :5].mean(axis=1)):
                f.write(f"{c},{i},{v:.4f}\n")
    write_json(out.with_suffix(".json"), {k: v for k, v in res.items() if k != "samples"})
    print(f"[langevin] {args.smiles} DN = {res['dn_mean']:.2f} +/- {res['dn_std']:.2f} "
          f"(95% CI [{res['dn_lower_95']:.2f}, {res['dn_upper_95']:.2f}], "
          f"R-hat={res['rhat']:.3f}, ESS={res['ess']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
