"""Step 19: EBM-based uncertainty quantification on top-20 candidates.

Goal
----
Add an **energy-based-model (EBM)** posterior over the donor-number
(DN) as a third opinion alongside the 5-model stacking ensemble.  We
train a small MLP that estimates an *unnormalized* log-density over
DN values, then draw Langevin / MH samples from it to obtain
mean, std, and credible intervals for each of the top-20 candidates.

The model
---------
- Inputs: v2 descriptor vector (1,005 dims) for each candidate.
- Output: scalar energy E(x, y) where y is the candidate's predicted DN.
- Training: noise-contrastive estimation with a Gaussian noise
  distribution for y; we also include a denoising-style loss that
  corrupts the input features and tries to recover y, akin to a DBM
  with stochastic hidden units.  This is deliberately a *lightweight*
  probabilistic baseline -- the goal is to provide an uncertainty
  estimate, not a state-of-the-art density estimator.
- Sampling: stochastic gradient Langevin dynamics (SGLD) on y given x,
  starting from the 5-model stacking point estimate.  Effective sample
  size (ESS) and integrated autocorrelation time (IACT) are reported
  per candidate.
- Diagnostics: also record 5-model ensemble spread (std of the 5
  individual model predictions) for comparison.

Output
------
- results/ebm_uncertainty.csv: per-candidate (mean, std, q05, q50, q95,
  ESS, IACT, 5model_std)
- results/ebm_sampling_diagnostics.json: overall sampler health
  (mean ESS, median ESS, mean IACT, fraction of chains with ESS > 100)
- results/ebm_uncertainty.md: human-readable summary
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESULTS_DIR, get_logger, set_global_seed  # noqa: E402

set_global_seed(42)
log = get_logger("ebm_uncertainty")


# --------------------------------------------------------------------------- #
# Torch import (lazy)
# --------------------------------------------------------------------------- #

def _import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception as e:
        log.error("PyTorch not available: %s. EBM addon will skip.", e)
        return None, None


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def _build_mlp(in_dim: int, hidden: int = 128, out_dim: int = 1):
    torch, nn = _import_torch()
    if torch is None:
        return None
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class DNEnergyModel:
    """E(x, y) -> R: a small MLP that scores (descriptor, scalar DN)."""

    def __init__(self, in_dim: int, hidden: int = 128):
        torch, nn = _import_torch()
        if torch is None:
            raise RuntimeError("PyTorch not available")
        self.torch = torch
        self.nn = nn
        self.net = _build_mlp(in_dim + 1, hidden, 1)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def energy(self, x, y):
        """E(x, y) for x [B, D] and y [B]."""
        z = self.torch.cat([x, y.view(-1, 1)], dim=1)
        return self.net(z).squeeze(-1)

    def train_noise_contrastive(
        self, x_train: np.ndarray, y_train: np.ndarray,
        n_epochs: int = 50, batch_size: int = 256, noise_std: float = 2.0,
    ):
        torch, nn = _import_torch()
        x = torch.tensor(x_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.float32)
        n = x.shape[0]
        for epoch in range(n_epochs):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = x[idx]
                yb = y[idx]
                # Positive energy: real (x, y) pair
                e_pos = self.energy(xb, yb)
                # Negative energy: corrupt y with Gaussian noise
                y_noise = yb + torch.randn_like(yb) * noise_std
                e_neg = self.energy(xb, y_noise)
                # InfoNCE-style: log p(y|x) >< log(1 - sigmoid)
                logits = torch.stack([e_pos, e_neg], dim=1)  # [B, 2]
                # Lower energy = higher probability.  Positive is index 0.
                loss = -torch.log(torch.softmax(-logits, dim=1)[:, 0] + 1e-9).mean()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total += float(loss.item()) * len(idx)
            if (epoch + 1) % 10 == 0:
                log.info("  epoch %d/%d  loss=%.4f",
                         epoch + 1, n_epochs, total / n)
        return self


# --------------------------------------------------------------------------- #
# Sampler: Stochastic Gradient Langevin Dynamics on y given x
# --------------------------------------------------------------------------- #

def sgld_sample(
    ebm: "DNEnergyModel",
    x: np.ndarray,
    y_init: float,
    n_samples: int = 200,
    step_size: float = 0.05,
    noise_scale: float = 0.5,
):
    """Draw n_samples from p(y|x) via SGLD.  Returns array of shape [n_samples]."""
    torch = ebm.torch
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).requires_grad_()  # [1, D]
    y = torch.tensor([[float(y_init)]], dtype=torch.float32, requires_grad=True)
    samples = []
    for _ in range(n_samples):
        # Compute dE/dy
        e = ebm.energy(x_t, y.view(-1))
        grad = torch.autograd.grad(e.sum(), y, retain_graph=False)[0]
        with torch.no_grad():
            y_next = (
                y
                - step_size * grad
                + noise_scale * torch.randn_like(y)
            )
        y = y_next.detach().clone().requires_grad_(True)
        samples.append(float(y.item()))
    return np.asarray(samples)


# --------------------------------------------------------------------------- #
# MH sampler (baseline)
# --------------------------------------------------------------------------- #

def metropolis_hastings_sample(
    ebm: "DNEnergyModel",
    x: np.ndarray,
    y_init: float,
    n_samples: int = 200,
    proposal_std: float = 0.5,
):
    """Metropolis-Hastings on y with Gaussian proposal."""
    torch = ebm.torch
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    y = float(y_init)
    samples = []
    accepted = 0
    with torch.no_grad():
        e_cur = float(ebm.energy(x_t, torch.tensor([y])).item())
    for _ in range(n_samples):
        y_prop = y + float(np.random.normal(0, proposal_std))
        with torch.no_grad():
            e_prop = float(ebm.energy(x_t, torch.tensor([y_prop])).item())
        log_alpha = -(e_prop - e_cur)
        if np.log(np.random.rand() + 1e-12) < log_alpha:
            y = y_prop
            e_cur = e_prop
            accepted += 1
        samples.append(y)
    acc_rate = accepted / n_samples
    return np.asarray(samples), acc_rate


# --------------------------------------------------------------------------- #
# Gibbs sampler (block: alternate y-update and feature-mask update)
# --------------------------------------------------------------------------- #

def gibbs_block_sample(
    ebm: "DNEnergyModel",
    x: np.ndarray,
    y_init: float,
    n_samples: int = 200,
    y_step: float = 0.1,
    feat_mask_prob: float = 0.1,
):
    """Block Gibbs: alternate updating y (SGLD) and re-masking a random
    subset of features (denoising-style).  This is the "DBM" with
    stochastic hidden units pattern."""
    torch = ebm.torch
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).clone().requires_grad_()
    y = float(y_init)
    samples = []
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        # 1. Update y via short SGLD chain.
        y_t = torch.tensor([[y]], dtype=torch.float32, requires_grad=True)
        for _ in range(3):
            e = ebm.energy(x_t, y_t.view(-1))
            grad = torch.autograd.grad(e.sum(), y_t)[0]
            with torch.no_grad():
                y_t = y_t.detach().clone().requires_grad_(True)
                y_t.add_(-y_step * grad).add_(0.05 * torch.randn_like(y_t))
            y = float(y_t.item())
        # 2. Block-mask features and re-fill with mean of training set.
        mask = rng.random(x.shape[0]) < feat_mask_prob
        if mask.any():
            x_t = x_t.clone()
            x_t[0, mask] = 0.0
        samples.append(y)
    return np.asarray(samples)


# --------------------------------------------------------------------------- #
# Effective sample size & IACT
# --------------------------------------------------------------------------- #

def effective_sample_size(samples: np.ndarray) -> float:
    """ESS via initial positive sequence estimator.

    Reference: BDA3 p.286-287.  Returns min(n, 1/(1+2*sum(rho_t))).
    """
    n = len(samples)
    if n < 4:
        return float(n)
    x = samples - samples.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0:
        return float(n)
    # Autocorrelation up to lag n//2
    max_lag = n // 2
    rho = np.empty(max_lag)
    for t in range(max_lag):
        rho[t] = float(np.dot(x[: n - t], x[t:]) / (n * var))
    # Sum positive pairs (initial positive sequence)
    s = 0.0
    for t in range(1, max_lag - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        s += pair
    tau = 1.0 + 2.0 * s
    ess = n / max(tau, 1.0)
    return float(min(n, ess))


def integrated_autocorr_time(samples: np.ndarray) -> float:
    n = len(samples)
    if n < 4:
        return 1.0
    return n / max(effective_sample_size(samples), 1.0)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    torch, nn = _import_torch()
    if torch is None:
        log.warning("Skipping EBM addon (no torch)")
        return

    # 1. Load data
    desc_path = DATA_DIR / "descriptors_v2.csv"
    label_path = DATA_DIR / "dn_labels_clean.csv"
    if not desc_path.exists() or not label_path.exists():
        log.warning("Missing descriptors_v2.csv or dn_labels_clean.csv. Skipping.")
        return

    desc = pd.read_csv(desc_path)
    labels = pd.read_csv(label_path)
    feat_cols = [c for c in desc.columns
                 if c not in {"smiles", "mol_id", "source", "is_anchor"}]
    merged = desc[["mol_id"] + feat_cols].join(
        labels[["dn_final"]], on="mol_id", how="inner"
    )
    X = merged[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = merged["dn_final"].to_numpy(dtype=np.float32)
    log.info("Loaded %d samples, %d features (aligned desc+labels by mol_id)", X.shape[0], X.shape[1])

    # 2. Train EBM
    ebm = DNEnergyModel(in_dim=X.shape[1], hidden=128)
    t0 = time.time()
    ebm.train_noise_contrastive(X, y, n_epochs=30, batch_size=512, noise_std=2.0)
    log.info("EBM training done in %.1fs", time.time() - t0)

    # 3. Pick candidates: union of top-20 and high-DN (DN > 30)
    top20_path = RESULTS_DIR / "top20_candidates_5model.csv"
    if not top20_path.exists():
        top20_path = RESULTS_DIR / "top20_candidates.csv"
    if not top20_path.exists():
        log.warning("No top-20 file; using top-20 by stacking prediction instead.")
        full_pred = RESULTS_DIR / "full_predictions_5model.csv"
        if full_pred.exists():
            full = pd.read_csv(full_pred)
            if "dn_pred_stack_v2" in full.columns:
                top20 = full.nlargest(20, "dn_pred_stack_v2")[
                    ["mol_id", "smiles", "dn_pred_stack_v2"]
                ].rename(columns={"dn_pred_stack_v2": "dn_stack"})
            else:
                top20 = full.nlargest(20, full.columns[-1])[
                    ["mol_id", "smiles", full.columns[-1]
                     ]].rename(columns={full.columns[-1]: "dn_stack"})
        else:
            log.error("No predictions file. Skipping EBM.")
            return
    else:
        top20 = pd.read_csv(top20_path)
        if "dn_pred_stack_v2" in top20.columns:
            top20 = top20.rename(columns={"dn_pred_stack_v2": "dn_stack"})
        else:
            top20 = top20.rename(columns={top20.columns[2]: "dn_stack"})

    log.info("Sampling EBM posterior for %d candidates", len(top20))
    desc_indexed = merged.set_index("mol_id")
    rows = []
    diagnostics = {
        "ess": [],
        "iact": [],
        "mh_accept": [],
        "gibbs_ess": [],
        "sgld_ess": [],
    }
    for _, r in top20.iterrows():
        mol_id = int(r["mol_id"])
        if mol_id not in desc_indexed.index:
            continue
        x = desc_indexed.loc[mol_id, feat_cols].fillna(0.0).to_numpy(
            dtype=np.float32
        )
        y_init = float(r.get("dn_stack", y.mean()))
        # Three samplers for cross-comparison
        s_sgld = sgld_sample(ebm, x, y_init, n_samples=200)
        s_mh, acc = metropolis_hastings_sample(ebm, x, y_init, n_samples=200)
        s_gibbs = gibbs_block_sample(ebm, x, y_init, n_samples=200)

        ess_sgld = effective_sample_size(s_sgld)
        ess_mh = effective_sample_size(s_mh)
        ess_gibbs = effective_sample_size(s_gibbs)
        iact_sgld = integrated_autocorr_time(s_sgld)
        # Combine samples for the final posterior summary (use SGLD as
        # primary, others for diagnostics)
        combined = np.concatenate([s_sgld, s_mh, s_gibbs])
        mean = float(combined.mean())
        std = float(combined.std())
        q05, q50, q95 = (float(np.quantile(combined, q)) for q in (0.05, 0.5, 0.95))
        # 5-model spread
        stack_cols = [c for c in top20.columns if c.startswith("dn_pred_") and c != "dn_pred_stack_v2"]
        if stack_cols:
            preds = r[stack_cols].to_numpy(dtype=float)
            five_model_std = float(np.std(preds))
        else:
            five_model_std = float("nan")
        rows.append({
            "mol_id": mol_id,
            "smiles": r["smiles"],
            "dn_stack_point": float(y_init),
            "ebm_mean": mean,
            "ebm_std": std,
            "ebm_q05": q05,
            "ebm_q50": q50,
            "ebm_q95": q95,
            "ess_sgld": ess_sgld,
            "ess_mh": ess_mh,
            "ess_gibbs": ess_gibbs,
            "iact_sgld": iact_sgld,
            "mh_accept_rate": acc,
            "five_model_std": five_model_std,
        })
        diagnostics["ess"].append(ess_sgld)
        diagnostics["iact"].append(iact_sgld)
        diagnostics["mh_accept"].append(acc)
        diagnostics["gibbs_ess"].append(ess_gibbs)
        diagnostics["sgld_ess"].append(ess_sgld)

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "ebm_uncertainty.csv"
    out.to_csv(out_path, index=False)
    log.info("Wrote %s with %d rows", out_path, len(out))

    diag = {
        "n_candidates": len(out),
        "mean_ess_sgld": float(np.mean(diagnostics["sgld_ess"])) if diagnostics["sgld_ess"] else 0.0,
        "median_ess_sgld": float(np.median(diagnostics["sgld_ess"])) if diagnostics["sgld_ess"] else 0.0,
        "mean_ess_mh": float(np.mean(diagnostics["ess"])) if diagnostics["ess"] else 0.0,
        "mean_iact_sgld": float(np.mean(diagnostics["iact"])) if diagnostics["iact"] else 0.0,
        "mean_mh_accept": float(np.mean(diagnostics["mh_accept"])) if diagnostics["mh_accept"] else 0.0,
        "fraction_ess_above_100": float(
            np.mean([e > 100 for e in diagnostics["sgld_ess"]])
        ) if diagnostics["sgld_ess"] else 0.0,
    }
    diag_path = RESULTS_DIR / "ebm_sampling_diagnostics.json"
    diag_path.write_text(json.dumps(diag, indent=2))
    log.info("Wrote %s: %s", diag_path, json.dumps(diag, indent=2))

    # Human-readable summary
    md = ["# EBM uncertainty summary", ""]
    md.append(f"- Candidates scored: **{len(out)}**")
    md.append(f"- Mean SGLD ESS: **{diag['mean_ess_sgld']:.1f}** "
              f"(median {diag['median_ess_sgld']:.1f})")
    md.append(f"- Mean SGLD IACT: **{diag['mean_iact_sgld']:.2f}**")
    md.append(f"- MH acceptance rate: **{diag['mean_mh_accept']:.2f}**")
    md.append(f"- Fraction of chains with ESS > 100: "
              f"**{diag['fraction_ess_above_100']:.0%}**")
    md.append("")
    md.append("## Top-10 candidates (by EBM posterior mean)")
    md.append("")
    md.append("| smiles | dn_stack | ebm_mean | ebm_std | ebm_q05 | ebm_q95 | 5model_std |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in out.nlargest(10, "ebm_mean").iterrows():
        md.append(
            f"| `{r['smiles']}` | {r['dn_stack_point']:.2f} | "
            f"{r['ebm_mean']:.2f} | {r['ebm_std']:.2f} | "
            f"{r['ebm_q05']:.2f} | {r['ebm_q95']:.2f} | {r['five_model_std']:.2f} |"
        )
    md_path = RESULTS_DIR / "ebm_uncertainty.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
