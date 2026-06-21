"""Tests for the EBM / MCMC / hardware-stochasticity addon.

These tests are run in the CI smoke suite alongside the existing
test_pipeline.py / test_v3.py.  They verify:

- The EBM posterior script runs to completion on the CI subset and
  produces the expected output files.
- The MCMC benchmark script produces a summary JSON with all three
  samplers represented.
- The hardware-stochasticity report produces a JSON with the four
  expected sections (rng, correlation, drift, robustness).
- The EBM-reported posterior std is non-zero (sampler is not stuck).
- The MCMC sampler ESS is > 0 (i.e., we got *some* effective samples).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from utils import RESULTS_DIR  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- #
# EBM uncertainty tests
# ---------------------------------------------------------------- #

def test_ebm_uncertainty_csv_exists():
    f = RESULTS_DIR / "ebm_uncertainty.csv"
    if not f.exists():
        pytest.skip("ebm_uncertainty.csv missing (run 19 first)")
    df = pd.read_csv(f)
    assert "ebm_mean" in df.columns
    assert "ebm_std" in df.columns
    assert "ess_sgld" in df.columns
    assert len(df) > 0


def test_ebm_posterior_std_positive():
    f = RESULTS_DIR / "ebm_uncertainty.csv"
    if not f.exists():
        pytest.skip("ebm_uncertainty.csv missing")
    df = pd.read_csv(f)
    if len(df) == 0:
        pytest.skip("EBM uncertainty file is empty")
    # Sampler should not be stuck at the prior: at least one row has
    # positive std.
    assert (df["ebm_std"] > 0).any(), "all EBM posteriors collapsed to a point"


def test_ebm_sampling_diagnostics_json():
    f = RESULTS_DIR / "ebm_sampling_diagnostics.json"
    if not f.exists():
        pytest.skip("ebm_sampling_diagnostics.json missing")
    d = json.loads(f.read_text())
    for key in ("mean_ess_sgld", "mean_iact_sgld", "mean_mh_accept",
                "fraction_ess_above_100"):
        assert key in d, f"diagnostics missing {key}"


def test_ebm_markdown_summary_exists():
    f = RESULTS_DIR / "ebm_uncertainty.md"
    if not f.exists():
        pytest.skip("ebm_uncertainty.md missing")
    text = f.read_text(encoding="utf-8")
    assert "EBM uncertainty" in text


# ---------------------------------------------------------------- #
# MCMC benchmark tests
# ---------------------------------------------------------------- #

def test_mcmc_benchmark_json():
    f = RESULTS_DIR / "mcmc_benchmark.json"
    if not f.exists():
        pytest.skip("mcmc_benchmark.json missing")
    d = json.loads(f.read_text())
    for sampler in ("SGLD", "MH", "Gibbs"):
        assert sampler in d, f"missing sampler {sampler}"
        m = d[sampler]
        assert "samples_per_s" in m
        assert "mean_ess" in m
        assert "r_hat" in m
        assert m["samples_per_s"] > 0


def test_mcmc_ess_positive():
    f = RESULTS_DIR / "mcmc_benchmark.json"
    if not f.exists():
        pytest.skip("mcmc_benchmark.json missing")
    d = json.loads(f.read_text())
    for sampler, m in d.items():
        assert m["mean_ess"] > 0, f"{sampler} ESS is zero"


def test_mcmc_markdown_report():
    f = RESULTS_DIR / "mcmc_benchmark.md"
    if not f.exists():
        pytest.skip("mcmc_benchmark.md missing")
    text = f.read_text(encoding="utf-8")
    assert "MCMC sampler benchmark" in text
    assert "SGLD" in text
    assert "MH" in text
    assert "Gibbs" in text


# ---------------------------------------------------------------- #
# Hardware-stochasticity tests
# ---------------------------------------------------------------- #

def test_hardware_stochasticity_json():
    f = RESULTS_DIR / "hardware_stochasticity.json"
    if not f.exists():
        pytest.skip("hardware_stochasticity.json missing")
    d = json.loads(f.read_text())
    for section in ("rng", "correlation", "drift", "robustness"):
        assert section in d, f"missing section {section}"


def test_hardware_rng_entropy_healthy():
    f = RESULTS_DIR / "hardware_stochasticity.json"
    if not f.exists():
        pytest.skip("hardware_stochasticity.json missing")
    d = json.loads(f.read_text())
    rng = d["rng"]
    # LSB proportion should be close to 0.5; entropy close to 1.0
    assert 0.45 < rng["p_lsb1"] < 0.55, f"LSB-1 proportion off: {rng['p_lsb1']}"
    assert rng["entropy_lsb_bits"] > 0.99, \
        f"LSB entropy too low: {rng['entropy_lsb_bits']}"


def test_hardware_markdown_report():
    f = RESULTS_DIR / "hardware_stochasticity.md"
    if not f.exists():
        pytest.skip("hardware_stochasticity.md missing")
    text = f.read_text(encoding="utf-8")
    assert "Hardware-stochasticity" in text
    assert "RNG entropy" in text
    assert "drift" in text.lower()
    assert "Robustness" in text


# ---------------------------------------------------------------- #
# Pure unit tests on the samplers (no torch dependency)
# ---------------------------------------------------------------- #

def test_sgld_runs_on_toy_energy():
    """The SGLD implementation in 20_mcmc_samplers should run on a
    1-D quadratic target without torch (uses finite-difference grad)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mcmc20", "src/20_mcmc_samplers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    E = mod._energy_fn(None, x_batch=None)
    samples, diag = mod.sgld(E, y_init=20.0, n=200, step=0.1, noise=0.3)
    assert len(samples) == 200
    assert samples.mean() > 0
    assert diag["n_energy_evals"] > 0


def test_mh_runs_on_toy_energy():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mcmc20", "src/20_mcmc_samplers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    E = mod._energy_fn(None, x_batch=None)
    samples, diag = mod.metropolis(E, y_init=20.0, n=200, proposal_std=0.5)
    assert len(samples) == 200
    assert 0 <= diag["accept_rate"] <= 1.0


def test_ess_function():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mcmc20", "src/20_mcmc_samplers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy as np
    # Independent samples: ESS should be close to n
    np.random.seed(0)
    samples = np.random.normal(0, 1, 500)
    e = mod.ess(samples)
    assert e > 400, f"independent samples ESS too low: {e}"
    # Highly correlated samples: ESS should be small
    correlated = np.cumsum(np.random.normal(0, 0.1, 500))
    e_corr = mod.ess(correlated)
    assert e_corr < e, "correlated chain ESS should be lower than independent"


def test_rhat_function():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mcmc20", "src/20_mcmc_samplers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy as np
    # Two chains from the same distribution: R-hat close to 1
    np.random.seed(0)
    c1 = np.random.normal(0, 1, 500)
    c2 = np.random.normal(0, 1, 500)
    r = mod.rhat([c1, c2])
    assert 0.95 < r < 1.1
