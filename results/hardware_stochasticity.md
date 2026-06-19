# Hardware-stochasticity / drift / robustness report

Software-side proxies for the chip/system team.  Replace `rng_entropy` / `drift_check` with the real hardware measurement stream when available.

## 1. RNG entropy
- LSB-1 proportion: **0.4999** (target 0.5000)
- LSB Shannon entropy: **1.0000** bits/bit (target 1.0000)
- 0->1 / 1->0 transition probability: **0.5006** (target 0.5000)

## 2. Correlation profile (SGLD on toy energy)
- Lag-1 autocorrelation: **0.897**
- Lag-10 autocorrelation: **0.374**
- Lag-50 autocorrelation: **0.029**

## 3. Drift check (chain mean first vs last 10%)
- First-10% mean: **24.610**
- Last-10% mean: **25.037**
- Absolute drift: **0.426**
- Relative drift: **1.73%**

## 4. Robustness to bias (EBM weight perturbation +2.0)
- Baseline posterior mean: **24.949**, std: **1.190**
- Biased posterior mean: **22.744**, std: **1.096**
- Mean shift under bias: **-2.204**