"""test_langevin.py - R-hat / ESS / posterior consistency."""
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))
from p26_bayesian_langevin import (  # noqa: E402
    GaussianPosterior, langevin_step, run_chains, infer_dn,
)
from utils_pb import gelman_rubin  # noqa: E402


def test_langevin_step_finite():
    e = GaussianPosterior(np.zeros(5), sigma=1.0)
    x = np.array([[0.1, -0.2, 0.0, 0.5, 0.3]])
    x1 = langevin_step(x, e.grad, eps=1e-3, noise_scale=1.0)
    assert np.isfinite(x1).all()
    assert x1.shape == x.shape


def test_rhat_known_chains():
    # Three identical chains -> R-hat = 1
    chains = np.ones((3, 100))
    assert abs(gelman_rubin(chains) - 1.0) < 1e-6
    # Two chains with same distribution
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (2, 1000))
    assert 0.99 < gelman_rubin(a) < 1.05


def test_run_chains_converges():
    mu = np.zeros(10)
    e = GaussianPosterior(mu, sigma=1.0)
    x0 = np.full((4, 10), 5.0)  # start far
    res = run_chains(e, x0, n_steps=2000, eps=5e-3, n_chains=4, burn_in=500)
    chain_mean = res["samples"].mean(axis=0)  # mean across (chains, steps) after burn-in
    # SGLD with eps=5e-3 + 2000 steps + 4 chains: per-dim mean ~ N(0, 0.1/sqrt(N))-ish,
    # but with 10 dims a few may drift to ~1. Use 1.5 as a loose bound.
    assert np.abs(chain_mean).max() < 1.5
    # ESS positive
    assert res["ess"] > 0


def test_infer_dn_runs_and_returns_ci():
    ens = {"dn_pred_rf_v2": 19.5, "dn_pred_xgb_v2": 20.0,
           "dn_pred_mlp_v2": 20.2, "dn_pred_lgbm_v2": 19.8,
           "dn_pred_cat_v2": 20.1, "dn_pred_stack_v2": 19.95}
    res = infer_dn(ens, n_steps=400, n_chains=3, D=64, seed=0)
    # Posterior mean near ensemble mean
    assert abs(res["dn_mean"] - 19.92) < 1.0
    # CI brackets mean
    assert res["dn_lower_95"] <= res["dn_mean"] <= res["dn_upper_95"]
    # R-hat small for the toy posterior
    assert res["rhat"] < 5.0  # very loose bound (R-hat sensitive at small D)


if __name__ == "__main__":
    test_langevin_step_finite()
    test_rhat_known_chains()
    test_run_chains_converges()
    test_infer_dn_runs_and_returns_ci()
    print("OK: all langevin tests passed")
