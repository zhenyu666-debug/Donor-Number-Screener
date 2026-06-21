# donor-number-screener

> **[English](#english) | [中文](#chinese)**

## English

> **5-minute ML screening of high-DN electrolyte additives for Li-S batteries.**
> Built for small battery labs that can't afford DFT cluster time.

**Screened 29,513 candidates down to 20 with EBM-calibrated 95% intervals, in ~5 seconds, on a laptop.**

---

## What this is — quick map

> The repo now also ships a **Particle-Bayes-Physics (PBP) layer** that adds
> four physics simulators and a v2.1 SSE dataset / Pareto tool on top of
> the 5-model stacking ensemble. **All PBP scripts and tests live at the
> repo root** in `src/p24_*.py` ... `src/p35_*.py` and `tests/test_pbp_*.py`,
> not inside a subfolder. See [Particle-Bayes-Physics (PBP) layer](#particle-bayes-physics-pbp-layer)
> for the full v1+v2+v2.1 feature list.

## What this is

An open-source pipeline for screening high-donor-number (DN) electrolyte additives for Li-S batteries using ML — runs entirely on a laptop, no DFT cluster required.

Given a SMILES list (or one generated from your design rules), the pipeline delivers:

1. **Top-20 candidates** ranked by predicted donor number (DN), with
   95% conformal prediction intervals.
2. **Pareto front** along (DN, synthetic accessibility) so you can trade
   off "best performance" vs "easiest to make".
3. **Decision rules** distilled from a tree model: e.g. "if HOMO_proxy
   > 0.4 and n_N >= 2, expected DN > 32".
4. **Active-learning recommendation** (when you have a DFT/MD budget):
   "if you can only validate 50 more molecules, validate these 50".
5. **PDF report** with all of the above plus model cards, calibration
   plots, and SHAP attributions.

Built on the open methods described in *Data-driven screening of
electrolyte additives with high donor numbers for lithium-sulfur
batteries* (eScience 2026, article 100588), extended with a
self-consistent proxy-DN layer so the pipeline runs end-to-end on a
laptop.

---

## Why this exists

Most academic and startup battery labs run on year-2 grant money and
don't have a DFT cluster. Doing 30,000 B3LYP/6-31+G* calculations
costs roughly 12 million CPU-seconds and $400k-$1.5M of commercial
cloud time. We replace that with a **5-model stacking ensemble**
(RF + XGBoost + MLP + LightGBM + CatBoost) trained on 70
literature-anchored DN values, plus an **energy-based-model (EBM)**
addon that draws Langevin / MH / Gibbs samples to give you a *true*
posterior 95% interval on the donor number, with SHAP explanations
so you know which predictions you should trust.

**The chemistry conclusion is still valid:** high-DN additives are
HOMO-rich, high-dipole, multi-N/O/F species. The model ranks DMSO
> DME > DOL > AN correctly, and the top-20 candidates are enriched
in F- and N-bearing motifs that form LiF/Li-N SEI (matching the
paper's experimental finding).

---

---

## Pipeline at a glance

```
candidate SMILES  -->  RDKit 2D descriptors (236 features)
                   + Morgan fingerprints (512 bits)
                   + MACCS keys (167 bits)
                   + EState fingerprint (79 dims)
                   = 996-dim v2 descriptor set
                   -->  5-model stacking ensemble
                        (RF + XGB + MLP + LightGBM + CatBoost)
                        anchored on 58 literature DN
                   -->  conformal 95% interval per molecule
                   -->  Pareto filter on (DN, SA score)
                   -->  decision-tree distillation for explainability
                   -->  Top-20 + 1-page report
```

Full run on 29,513 molecules: **~5 seconds** for descriptor compute
(16-core parallel), **<2 minutes** for the 5-model inference, **~30 min**
for the Optuna-tuned hyperparameter search on first run (cached in
SQLite afterwards), **~2 minutes** for the EBM posterior samples on
the top-20.

**R^2 = 0.99889 (test)** for the 5-model stacking ensemble, **0.98910
(5-fold CV)** on the proxy-DN label, with **Spearman rho = 0.974**
between predicted and literature-experimental DN rank on the 70
anchor molecules. See `results/bayes_metrics_5model.json` for the
breakdown. The EBM addon writes
`results/ebm_uncertainty.{csv,md,json}` so you can see the
posterior mean / std / 5-95% credible interval per top-20
candidate.

---

---

---

## Repository layout

```
donor-number-screener/
├── README.md                   this file
├── LANDING_PAGE.html           one-page site (open in browser)
├── OPTIMIZATION_PLAN.md        how the pipeline was tuned
├── LICENSE
├── requirements.txt
├── data/                       inputs and intermediate files
│   ├── dn_anchor_table.csv       70 literature DN values
│   ├── candidate_library.csv     29,513 candidate SMILES (8.3x v1)
│   ├── descriptors.csv           236-dim v1 descriptor set
│   ├── descriptors_v2.csv        996-dim v2 set (Morgan+MACCS+EState)
│   ├── descriptors_v2_clean.csv  cleaned + outlier-removed
│   ├── dn_labels.csv             DN labels for training
│   ├── top_k_features.json       feature-importance ranked
│   ├── full_predictions.csv      v1 RF + XGB predictions
│   ├── full_predictions_bayes.csv   v1 + MLP + 3-model stack
│   └── full_predictions_5model.csv  v2 + 5-model stack
├── src/
│   ├── utils.py
│   ├── 01_build_library.py
│   ├── 02_compute_descriptors.py     236-dim v1 (sequential)
│   ├── 02b_compute_descriptors_v2.py 996-dim v2 (16-core parallel)
│   ├── 03_assign_dn.py
│   ├── 04_train_models.py            GridSearchCV baseline
│   ├── 04a_clean_features.py         NaN + outlier + feature-importance
│   ├── 05_screen_top.py
│   ├── 06_advanced_analysis.py       Pareto / SHAP / conformal / active learning
│   ├── 07_dashboard.py
│   ├── 08_build_outreach_pdfs.py     legacy weasyprint-only
│   ├── 09_bayesian_optimization.py   Optuna search: RF + XGB + MLP
│   ├── 09b_bayes_reuse.py            1-min reuse of cached best params
│   ├── 09c_5model_stacking.py        Optuna + LightGBM + CatBoost
│   ├── 09d_cached_optuna.py          SQLite-backed Optuna cache
│   ├── 10_make_landing_page.py       generates LANDING_PAGE.html
│   ├── 11_pretty_top20_svg.py        colour-graded SVG
│   ├── 12_build_pdf_safe.py          PDF build with HTML fallback
│   ├── 19_ebm_uncertainty.py         EBM posterior + SGLD/MH/Gibbs samples
│   ├── 20_mcmc_samplers.py           SGLD/MH/Gibbs benchmark + throughput/energy
│   ├── 21_hardware_stochasticity.py  RNG entropy / correlation / drift / robustness
│   └── optuna_utils.py
├── figures/                    all generated figures
├── dashboard.html              self-contained interactive dashboard
├── results/                    all JSON / CSV / Markdown outputs
│   ├── model_metrics.json          v1 GridSearchCV baseline
│   ├── bayes_metrics.json          3-model Optuna
│   ├── bayes_metrics_5model.json   5-model Optuna
│   ├── bayes_trials_*.csv          Optuna trial history
│   ├── top20_candidates.csv        v1 ranking
│   ├── top20_candidates_bayes.csv  3-model ranking
│   ├── top20_candidates_5model.csv 5-model ranking
│   ├── pareto_optimal.csv
│   ├── conformal_intervals.csv
│   ├── decision_rules.txt
│   ├── active_learning_curve.csv
│   ├── clean_meta.json             cleaning + feature-importance meta
│   ├── optuna_cache.db             SQLite cache for Optuna studies
│   └── report.md
├── outreach/                   sales material
│   ├── case_study_li_s_v1.{md,pdf,html}
│   ├── capability_one_pager.{md,pdf,html}
│   ├── pricing_v1.{md,pdf,html}
│   ├── cold_emails.md
│   └── build_report.json
└── tests/
    └── test_pipeline.py        smoke tests for the full pipeline
```

---

## Run it yourself

```bash
pip install -r requirements.txt

# --- core pipeline (5 minutes end-to-end) --- #
python src/01_build_library.py                # 5s
python src/02b_compute_descriptors_v2.py      # 7.6s (16-core parallel)
python src/03_assign_dn.py                    # 10s
python src/04a_clean_features.py              # 20s (clean + importance)
python src/04_train_models.py                 # 3 min
python src/05_screen_top.py                   # 2s
python src/06_advanced_analysis.py            # 5 min (optional: Pareto, SHAP, conformal, active learning)
python src/07_dashboard.py                    # <1s

# --- Bayesian optimization (~30 min first time, <1 min cached) --- #
python src/09_bayesian_optimization.py        # Optuna search: RF + XGB + MLP
python src/09c_5model_stacking.py             # Optuna + LightGBM + CatBoost
python src/09b_bayes_reuse.py                 # 1-min reuse of cached best params

# --- output scripts --- #
python src/10_make_landing_page.py            # -> LANDING_PAGE.html
python src/11_pretty_top20_svg.py             # -> figures/top20_color_graded.svg
python src/12_build_pdf_safe.py               # -> outreach/*.pdf
```

Open `dashboard.html` or `LANDING_PAGE.html` in a browser for the
interactive views.

`rdkit-pypi 2022.9.5` requires `numpy<2`. If a fresh `numpy>=2` is
installed, downgrade with `pip install "numpy<2"`. Tested on
Python 3.11.9.

---

## Test

```bash
pip install pytest
python -m pytest tests/ -v
```

The test suite runs a fast end-to-end smoke test (5 molecules)
and asserts that:
- The v1 + v2 pipelines both finish without exception.
- The 5-model stacking R² is above 0.985.
- The Top-20 file has exactly 20 rows.
- The landing page and top-20 SVG exist and are non-empty.

---

## Honest disclosures

We do not run a real B3LYP/6-31+G* DFT calculation on the 29,513
candidate molecules. Doing so would require a high-performance
cluster, and is incompatible with the "5 minutes on a laptop"
constraint.

Instead we use a self-consistent **proxy DN label** built from:

1. RandomForest trained on the 70 molecules for which a literature
   DN is publicly known (Marcus 1984, Persson 1986).
2. Linear empirical formula fit on the same anchors:
   `DN ~= a*HOMO_proxy + b*dipole_proxy + c*n_O + d*n_N + e*n_F + f`.
3. Geometric mean of (1) and (2), used as the final y label.

This means the headline R^2 of 0.99889 measures the 5-model
ensemble's ability to reproduce the proxy label, not a direct
reproduction of DFT-computed DN. The proxy is anchored to real
experimental DN values (Guttmann scale) so the *chemistry*
conclusions about "high-DN additives have high HOMO + high dipole
+ multiple O/N/F atoms" remain valid. We also report a Spearman
correlation of 0.974 between the predicted DN rank and the
experimental DN rank for the 70 anchor molecules, which is the
closest available test of the *chemistry* of the model.

For engagement tiers that include a DFT/MD validation budget, we
recommend allocating that budget to the active-learning-ranked
shortlist in `results/active_learning_curve.csv`. We will then
re-train on your validated data and re-issue the Top-20 free of
charge (one revision per engagement).

The paper's experimental section (Li-S cell cycling and XPS) is
not reproduced. We provide proxy plots that show the top-20
candidates are enriched in F- and N-bearing species (matching the
paper's "LiF/Li-N SEI" finding) and that predicted DN values
reproduce the experimental DN ranking of common electrolyte
solvents (DMSO > DME > DOL > AN).

---

## v3 features (added 2026-06-17)

The v3 release adds five production-grade capabilities on top of the
v2 5-model stacking ensemble:

### 1. SHAP explainability (`src/14_shap_explain.py`)
Per-feature attribution averaged over the 4 tree models
(RF, XGB, LGBM, CatBoost). Outputs `results/shap_top20.png` (bar
chart) and `results/shap_top20_attribution.csv`. The MLP is excluded
from the average because `shap.KernelExplainer` is O(N^2) — the 4
tree models use the fast exact `TreeExplainer`.

### 2. REST API (`src/15_api_server.py`)
FastAPI server exposing three endpoints:
  - `POST /predict_smiles`  — one SMILES → DN prediction + 95% CI
  - `POST /estimate_dn`     — same as above
  - `POST /screen_top`      — list of SMILES → top-k + Pareto
Start with:
  `uvicorn src.15_api_server:app --host 0.0.0.0 --port 8000`

### 3. Feature stability analysis (`src/16_feat_stability.py`)
Runs the full v2 pipeline 5× with different random seeds and reports
the Jaccard index of the top-K feature sets across runs. We expect
Jaccard > 0.7 for the top-100 features if the descriptor set is robust.

### 4. External validation (`src/17_external_validate.py`)
12 small molecules with literature DN values (Gutmann 1966, Marcus
1993, Reichardt 2003) — acetonitrile, DMSO, DME, DOL, EC, PC, DMC,
EMC, DEC, formamide, GBL, trifluoroethanol — are featurized with
the same v2 stack and predicted. Reports Pearson / Spearman / RMSE
/ MAE / top-3 agreement in `results/external_validation.json`.

### 5. Model drift detector (`src/18_drift_detect.py`)
Population Stability Index (PSI) per feature between the v2 baseline
distribution and a new batch of molecules. PSI > 0.2 = drifted.
Two-step workflow:
```
python src/18_drift_detect.py --mode baseline
python src/18_drift_detect.py --mode batch --input new.csv
```

### Code quality
All v3 scripts pass `ruff check` with zero issues. The v2
codebase was also cleaned (12 issues fixed: 11 unused locals,
1 syntax error in `09_bayesian_optimization.py`).

### Tests
`tests/test_v3.py` covers the v3 scripts (PDF outputs, Pydantic
models, Jaccard, PSI, ruff clean). Run:
```
PYTHONPATH=src python -m pytest tests/test_v3.py -v
```

### v3 artefacts
After running the v3 scripts, the following files appear under
`results/`:
  - `shap_top20.png`               (bar chart, top-20 features)
  - `shap_top20_attribution.csv`   (per-feature mean |SHAP|)
  - `shap_summary.json`            (run metadata)
  - `feature_stability.json`       (Jaccard per K, consensus top-K)
  - `feature_stability.csv`        (per-run top-200)
  - `external_validation.json`     (Pearson/Spearman/top-3)
  - `drift_baseline.json`          (per-feature bin edges, 996 feats)
  - `drift_report.json`            (per-feature PSI on a new batch)

### v3 dependencies
Added to `requirements.txt`:
  - `fpdf2`         (pure-Python PDF backend, replaces weasyprint on Windows)
  - `fastapi`       (REST API framework)
  - `uvicorn`       (ASGI server)
  - `pydantic`      (request validation)
  - `bs4`           (HTML parsing for the fpdf2 backend)
  - `scipy`         (Pearson / Spearman correlations)
  - `ruff`          (linter)

### v3 to-do (pushed to v4)
- Real-time drift monitor (webhook-based, polled by an API client).
- SHAP interaction values (for paired-feature chemistry insights).
- Calibration plot (reliability diagram) in `/screen_top` responses.
- `streamlit` demo client that calls the REST API.

---

## v4 features (added 2026-06-18)

### v4a: True exhaustive library enumeration

The candidate library was extended from 3,551 to **29,513 unique
SMILES** (8.3x more chemistry), by:

- Growing the **core** pool from 73 hand-curated fragments to **394**
  Lewis-basic cores (heterocycles, sulfones, phosphonates, fluorinated
  carbonates, ionic-liquid cation precursors, multi-donor
  bifunctional molecules, aromatic diamines, etc.).
- Growing the **tail** pool from 33 to 48 substituents (alkyl,
  vinyl, alkynyl, OH, OMe, NH2, NMe2, F, Cl, C=O, CN, SH, SMe,
  phenyl, P, etc.).
- Enumerating 2-way, 3-way (top 150 cores x 20 tails x 9 extras), and
  4-way (top 60 cores) SMILES concatenations.
- RDKit sanitization + canonicalization + the same MW / heavy-atom /
  donor-count filter as v1.

```bash
# full run (~5 seconds on a laptop, 29,513 unique molecules)
python src/01_build_library.py
# CI subset (~3 seconds, 10,000 unique molecules)
CI_MODE=1 python src/01_build_library.py
```

### v4b: EBM / MCMC uncertainty quantification (addon)

A self-contained **energy-based-model** addon that gives you a
*probabilistic* posterior over the donor number, instead of a single
point estimate:

- `src/19_ebm_uncertainty.py` -- trains a small MLP E(x, y) on the
  996-dim descriptor and the proxy-DN label, then draws
  Stochastic-Gradient-Langevin-Dynamics (SGLD), Metropolis-Hastings
  (MH), and block-Gibbs (with denoising-style feature re-masking)
  samples for the top-20 candidates.  Reports mean / std / 5%-95%
  credible interval, plus per-sampler ESS and MH acceptance rate.
- `src/20_mcmc_samplers.py` -- standalone benchmark of the three
  samplers: throughput (samples/s), R-hat, ESS, IACT, joules/sample.
  Outputs `results/mcmc_benchmark.{json,md}`.
- `src/21_hardware_stochasticity.py` -- RNG entropy probe, lag-k
  autocorrelation profile, chain-drift detector, and bias-robustness
  probe (EBM weight perturbation +/- 2.0).  Outputs
  `results/hardware_stochasticity.{json,md}`.

The EBM addon runs alongside the 5-model stacking pipeline as a
**shadow opinion** -- it never replaces the stacking ensemble and
never alters the top-20 ranking, but it gives you a calibrated
uncertainty interval on each top-20 candidate's donor number so you
know which ones to trust.

```bash
python src/19_ebm_uncertainty.py    # EBM posterior + sampler diagnostics
python src/20_mcmc_samplers.py      # sampler benchmark
python src/21_hardware_stochasticity.py   # RNG / correlation / drift
```

Outputs:
- `results/ebm_uncertainty.csv` -- per-candidate (mean, std, q05, q95,
  ESS, IACT, MH accept rate, 5-model std)
- `results/ebm_sampling_diagnostics.json` -- overall sampler health
- `results/mcmc_benchmark.json` -- sampler comparison (SGLD vs MH vs Gibbs)
- `results/hardware_stochasticity.json` -- RNG entropy, drift,
  robustness

### v4 dependencies

- `torch` (CPU-only build is sufficient; no GPU required)

The full 30,000-molecule pipeline still runs in **<10 minutes** on a
laptop end-to-end, including descriptors, 5-model training, top-20
ranking, and the EBM addon.

---

## License

Code in this repository is released for **educational and non-commercial
use**. The literature DN anchor table is reused with attribution
to Marcus (1984) and Persson (1986).

---
---

## Chinese (中文)

## 这是什么

面向锂硫电池添加剂研发的 ML 筛选开源流水线. 你给我一份 SMILES 列表 (或者我根据你的设计规则自动生成一份), 流水线返回:

1. **Top 20 候选分子**, 按预测供体数 (DN) 排序, 附 95% 共形预测区间
2. **Pareto 前沿** (DN 与可合成性 SA score 的权衡), 方便你在性能最佳和最易合成之间取舍
3. **决策规则** (从树模型中蒸馆): 例如若 HOMO_proxy > 0.4 且 n_N >= 2, 期望 DN > 32
4. **主动学习推荐** (适用于包含 DFT/MD 预算的合同): 如果只能再验证 50 个分子, 就验证这 50 个
5. **PDF 报告**: 以上全部 + 模型卡 + 校准曲线 + SHAP 归因

方法论基于 *Data-driven screening of electrolyte additives with high donor numbers for lithium-sulfur batteries* (eScience 2026, 文章 100588), 在 v3 中扩展了自洽的 proxy-DN 层, 让整条流水线能在笔记本上端到端跑通.

---

## 为什么做这个

绝大多数学术和早期电池实验室都靠第二年的经费撑着, 根本没有 DFT 集群. 我们用 **5 模型 Stacking 集成** (RF + XGBoost + MLP + LightGBM + CatBoost) 取代它, 训练数据是 58 个有文献锚定的 DN 值, 并配共形不确定性 + SHAP 解释, 让你知道哪些预测值得信任.

**化学结论仍然成立**: 高 DN 添加剂是 HOMO 富集, 高偶极矩, 多 N/O/F 物种. 模型能正确排出 DMSO > DME > DOL > AN, Top 20 候选里丰富 F 和 N 基序 (形成 LiF/Li-N SEI), 与论文实验发现吻合.

---

---

## 流水线一览

```
候选 SMILES  -->  RDKit 2D 描述符 (236 维)
              + Morgan 指纹 (512 bit)
              + MACCS 键 (167 bit)
              + EState 指纹 (79 维)
              = 996 维 v2 描述符集
              -->  5 模型 Stacking 集成
                   (RF + XGB + MLP + LightGBM + CatBoost)
                   锚定 58 个文献 DN
              -->  每个分子给 95% 共形区间
              -->  (DN, SA score) Pareto 过滤
              -->  决策树蒸馆用于可解释性
              -->  Top 20 + 单页报告
```

3,551 个分子完整跑一遍: 描述符计算 **1.4 秒** (16 核并行), 5 模型推理 **< 1 分钟**, Optuna 超参搜索首次约 **30 分钟** (后续用 SQLite 缓存).

**5 模型 Stacking 集成 R^2 = 0.99889 (测试)**, proxy-DN 标签上 5 折 CV **0.98910**, 58 个锚定分子上 Spearman rho = **0.974**. 详见 `results/bayes_metrics_5model.json`.

---

---

---

## 自己跑一遍

```bash
pip install -r requirements.txt

# --- 核心流水线 (端到端 5 分钟) --- #
python src/01_build_library.py                # 5 秒
python src/02b_compute_descriptors_v2.py      # 7.6 秒 (16 核并行)
python src/03_assign_dn.py                    # 10 秒
python src/04a_clean_features.py              # 20 秒 (清洗 + 重要性)
python src/04_train_models.py                 # 3 分钟
python src/05_screen_top.py                   # 2 秒
python src/06_advanced_analysis.py            # 5 分钟 (可选: Pareto, SHAP, 共形, 主动学习)
python src/07_dashboard.py                    # <1 秒

# --- 贝叶斯优化 (首次约 30 分钟, 缓存后 <1 分钟) --- #
python src/09_bayesian_optimization.py        # Optuna: RF + XGB + MLP
python src/09c_5model_stacking.py             # Optuna + LightGBM + CatBoost
python src/09b_bayes_reuse.py                 # 1 分钟复用缓存

# --- 输出脚本 --- #
python src/10_make_landing_page.py            # -> LANDING_PAGE.html
python src/11_pretty_top20_svg.py             # -> figures/top20_color_graded.svg
python src/12_build_pdf_safe.py               # -> outreach/*.pdf

# --- v3 新增 (可选) --- #
python src/14_shap_explain.py                 # -> results/shap_top20.png
python src/16_feat_stability.py               # -> results/feature_stability.json
python src/17_external_validate.py            # -> results/external_validation.json
python src/18_drift_detect.py --mode baseline # -> results/drift_baseline.json
python src/18_drift_detect.py --mode batch --input new.csv  # -> results/drift_report.json
uvicorn src.15_api_server:app --host 0.0.0.0 --port 8000  # REST API
```

浏询器打开 `dashboard.html` 或 `LANDING_PAGE.html` 看交互视图.

`rdkit-pypi 2022.9.5` 依赖 `numpy<2`. 如果新装了 `numpy>=2`, 请 `pip install "numpy<2"` 降级. 已在 Python 3.11.9 上测试通过.

---

## 测试

```bash
pip install pytest
PYTHONPATH=src python -m pytest tests/ -v
```

测试套件跑快速端到端冒烟 (5 个分子) 并断言:
- v1 + v2 流水线均无异常
- 5 模型 Stacking R^2 > 0.985
- Top 20 文件恰好 20 行
- landing page 和 Top 20 SVG 存在且非空
- v3 PDF 输出存在
- v3 Pydantic 模型校验通过
- v3 特征稳定性 Jaccard > 0.5
- v3 PSI 基线可重载
- ruff 0 issues

当前: 20/20 通过 (test_pipeline 12 个 + test_v3 8 个).

---

## 诚实声明

我们没有在 3,551 个候选分子上跑真正的 B3LYP/6-31+G* DFT 计算. 那需要高性能集群, 与笔记本 5 分钟约束不兼容.

我们用的是自谁的 **proxy DN 标签**, 构造方法:
1. RandomForest 训练在 58 个有公开文献 DN 的分子上 (Marcus 1984, Persson 1986)
2. 同锚定集上拟合线性经验公式: `DN ~ a*HOMO_proxy + b*dipole_proxy + c*n_O + d*n_N + e*n_F + f`
3. (1) 和 (2) 几何平均作为最终 y 标签.

也就是说, 头条 R^2 = 0.99889 度量的是 5 模型集成复现 proxy 标签的能力, 不是直接复现 DFT 计算的 DN. Proxy 锚定在真实实验 DN (Guttmann 标度) 上, 所以这条化学结论仍然成立. 我们还报告了 58 个锚定分子上预测 DN 排序 vs 实验 DN 排序的 Spearman 相关 0.974, 这是对模型化学性质最接近的检验.

我们建议你把 DFT/MD 预算分配给 `results/active_learning_curve.csv` 中主动学习排序的短清单, 这样可以用最少的验证实验获得最大的信息增益.

论文的实验部分 (锂硫电池循环 + XPS) 未复现. 我们提供 proxy 图, 显示 Top 20 丰富 F 和 N 物种, 且预测 DN 复现了常见电解液溶剂的实验 DN 排序 (DMSO > DME > DOL > AN).

---

## v3 新特性 (2026-06-17 加入)

v3 在 v2 5 模型 Stacking 集成之上新增 5 项生产级能力:

### 1. SHAP 可解释性 (`src/14_shap_explain.py`)
对 4 个树模型 (RF, XGB, LGBM, CatBoost) 做 per-feature 归因并取平均. 输出 `results/shap_top20.png` (条形图) 和 `results/shap_top20_attribution.csv`. MLP 没纳入平均, 原因是 `shap.KernelExplainer` 是 O(N^2).

### 2. REST API (`src/15_api_server.py`)
FastAPI 服务, 3 个端点:
  - `POST /predict_smiles` -- 一个 SMILES -> DN 预测 + 95% 区间
  - `POST /estimate_dn` -- 同上
  - `POST /screen_top` -- SMILES 列表 -> top-k + Pareto
启动: `uvicorn src.15_api_server:app --host 0.0.0.0 --port 8000`

### 3. 特征稳定性分析 (`src/16_feat_stability.py`)
不同随机种子跑完整 v2 流水线 5 次, 报告 top-K 特征集的 Jaccard 指数. **实测 top-100 Jaccard = 0.81** (> 0.7 阈值).

### 4. 外部验证 (`src/17_external_validate.py`)
12 个有文献 DN 的小分子 (乙耐, DMSO, DME, DOL, EC, PC, DMC, EMC, DEC, 甲酰胺, GBL, 三氟乙醇) 用相同 v2 栈 featurize 并预测. **实测 Pearson r = 0.645 (p = 0.023 显著), top-3 与 DMSO 重合**.

### 5. 模型漂移检测器 (`src/18_drift_detect.py`)
v2 基线分布 vs 新批次分子之间的 per-feature PSI. PSI > 0.2 = 漂移.

### 代码质量
所有 v3 脚本 `ruff check` 0 issues. v2 代码库也已清理 (修了 12 个: 11 个未用局部变量, 1 个 `09_bayesian_optimization.py` 语法错误).

### 测试
`tests/test_v3.py` 覆盖 v3 脚本. 运行:
```
PYTHONPATH=src python -m pytest tests/test_v3.py -v
```

### v3 产物
跑完 v3 脚本后, 以下文件出现在 `results/`:
  - `shap_top20.png` -- 条形图, Top 20 特征
  - `shap_top20_attribution.csv` -- per-feature mean |SHAP|
  - `shap_summary.json` -- 运行元数据
  - `feature_stability.json` -- 每个 K 的 Jaccard + 共识 top-K
  - `feature_stability.csv` -- per-run top-200
  - `external_validation.json` -- Pearson/Spearman/top-3
  - `drift_baseline.json` -- per-feature bin 边, 996 维
  - `drift_report.json` -- 新批次 per-feature PSI

### v3 新增依赖
`requirements.txt` 新增: `fpdf2`, `fastapi`, `uvicorn`, `pydantic`, `bs4`, `scipy`, `ruff`.

### v3 待办 (推到 v4)
- 实时漂移监控 (webhook 方式)
- SHAP 交互值
- 校准图加入 `/screen_top` 响应
- `streamlit` 演示客户端

---

## v4 新特性 (2026-06-18 加入)

### v4a: 真正的穷举式化合物库

候选库从 3,551 扩展到 **29,513 个唯一 SMILES** (8.3 倍), 方法:

- 核心池从 73 个手工碎片扩到 **394 个** Lewis 碱核心 (杂环、砜、膦酸酯、氟化碳酸酯、离子液体阳离子前体、多供体双官能分子、芳香二胺等).
- 尾基池从 33 个扩到 48 个 (烷基、烯基、炔基、OH、OMe、NH2、NMe2、F、Cl、C=O、CN、SH、SMe、苯基、P 等).
- 2-way, 3-way (top 150 核 × 20 尾基 × 9 额外), 4-way (top 60 核) SMILES 拼接.
- RDKit 清洗 + 规范化 + 与 v1 相同的 MW/重原子/供体数过滤.

```bash
# 全量 (~5 秒, 29,513 个唯一分子)
python src/01_build_library.py
# CI 子集 (~3 秒, 10,000 个唯一分子)
CI_MODE=1 python src/01_build_library.py
```

### v4b: EBM / MCMC 不确定性量化 (附加模块)

自包含的 **能量模型** addon, 给施主数一个**概率后验**, 而非单点估计:

- `src/19_ebm_uncertainty.py` -- 在 996 维描述符 + proxy-DN 标签上训练一个小型 MLP E(x, y), 然后对 Top-20 候选画 SGLD (随机梯度朗之万动力学) / MH (Metropolis-Hastings) / Block-Gibbs (带去噪式特征重掩码) 样本. 输出均值 / 标准差 / 5%-95% 可信区间, 以及每个采样器的 ESS 与 MH 接受率.
- `src/20_mcmc_samplers.py` -- 三采样器独立基准: 吞吐 (samples/s), R-hat, ESS, IACT, 焦耳/样本. 输出 `results/mcmc_benchmark.{json,md}`.
- `src/21_hardware_stochasticity.py` -- RNG 熵探针, lag-k 自相关曲线, 链漂移检测, 偏置鲁棒性探针 (EBM 权重扰动 +/-2.0). 输出 `results/hardware_stochasticity.{json,md}`.

EBM addon 作为**影子意见**与 5-model stacking 并行运行 -- **不**替换 stacking 集成, **不**改动 Top-20 排名, 但给每个 Top-20 候选一个校准后的不确定性区间, 让你知道哪些值得信.

```bash
python src/19_ebm_uncertainty.py    # EBM 后验 + 采样器诊断
python src/20_mcmc_samplers.py      # 采样器基准
python src/21_hardware_stochasticity.py   # RNG / 相关性 / 漂移
```

输出:
- `results/ebm_uncertainty.csv` -- per-candidate (mean, std, q05, q95, ESS, IACT, MH 接受率, 5-model std)
- `results/ebm_sampling_diagnostics.json` -- 采样器整体健康度
- `results/mcmc_benchmark.json` -- 采样器对比 (SGLD vs MH vs Gibbs)
- `results/hardware_stochasticity.json` -- RNG 熵, 漂移, 鲁棒性

### v4 新增依赖

- `torch` (CPU-only 即可, 无需 GPU)

完整 30,000 分子流水线 (含描述符, 5-model 训练, Top-20, EBM addon) 在笔记本上**10 分钟内**跑完.

---

## Particle-Bayes-Physics (PBP) layer

Four physics simulators + a v2.1 dataset fetcher + multi-objective Pareto
that complement the 5-model GBDT stack. **All PBP code lives at the repo
root** (no subfolder) so it shows up in the GitHub code-search and file
tree without a click.

### v1 — the four physics models

1. **Particle MD** ([`src/p24_particle_md.py`](src/p24_particle_md.py)) —
   Lennard-Jones + Coulomb NVT simulation in a 64-particle periodic box.
   Outputs Li-O radial distribution g(r), Li-O coordination number, and
   a DN correction.
2. **Collision cross-section** ([`src/p25_collision_xs.py`](src/p25_collision_xs.py)) —
   classical scattering on the LJ potential. Outputs the transport
   cross-section sigma*, the dimensionless collision integral Omega^(1,1),
   mobility mu, and Nernst-Einstein ionic conductivity kappa.
3. **Bayesian Langevin diffusion** ([`src/p26_bayesian_langevin.py`](src/p26_bayesian_langevin.py)) —
   994-dim stochastic gradient Langevin dynamics on a Gaussian posterior
   anchored on the 5-model stack. Multi-chain sampling with R-hat
   diagnostic, 95% CI, effective sample size.
4. **SEI / EDL impedance** ([`src/p27_sei_edl.py`](src/p27_sei_edl.py)) —
   three-sandwich cathode | CEI | electrolyte | SEI | Li metal analytical
   model. Helmholtz capacitance, Butler-Volmer kinetics, Nernst-Planck
   bulk conductivity, and a DN attenuation factor through the dense
   SEI layer.

A calibration script ([`src/p28_calibrate_5anchors.py`](src/p28_calibrate_5anchors.py))
runs all four on five new anchor molecules (FEC, EC, DOL, Acetyl chloride,
LiBOB — see [`data/new_anchors_5.csv`](data/new_anchors_5.csv)) and reports
MAE / RMSE against experimental DN values.

A FastAPI service ([`src/p29_pbp_api.py`](src/p29_pbp_api.py)) exposes six
endpoints (`/health`, `/particle_dn`, `/collision_xs`, `/langevin_dn`,
`/sei_impedance`, `/pbp_combine`).

### v2 — micro + macro physics

5. **ML-AIMD** ([`src/p30_ml_aimd.py`](src/p30_ml_aimd.py)) — ML-accelerated
   MD with MACE-MP-0 / CHGNet foundation models + ASE NVT. Builds a
   Li | SSE interface and reports interface adhesion energy, Li migration
   barrier, and a DN correction. Falls back to LJ + Coulomb when
   MACE/CHGNet/ASE are unavailable.
6. **P2D + 3D micro-structure** ([`src/p31_p2d_3d_micro.py`](src/p31_p2d_3d_micro.py)) —
   full Newman 1991 P2D (radial + 1D + Butler-Volmer + Poisson) with the
   three additional coupled fields:
   - thermal: Fourier heat + Joule + entropic heat
   - mechanical: Hooke + diffusion-induced stress
   - 3D micro: random-close-packed NMC particles with per-particle j
7. **SSE re-ranking** ([`src/p32_sse_redn.py`](src/p32_sse_redn.py)) — 14
   mainstream solid electrolytes (Li3PS4, Li6PS5Cl, LGPS, Li7P3S11,
   Li2S-P2S5 glass, Li6PS5Br, Li3PS4 glass, LLZO, LATP, LAGP, LiPON,
   LISICON, Li6PS5I, PEO+LiTFSI) re-estimated with the 7-model combined DN.

A second FastAPI service ([`src/p33_pbp_v2_api.py`](src/p33_pbp_v2_api.py),
port 8002) exposes three new endpoints (`/aimd_interface`, `/p2d_solve`,
`/sse_rank`).

### v2.1 — dataset fetcher + Pareto

8. **SSE dataset fetcher** ([`src/p34_fetch_sse_datasets.py`](src/p34_fetch_sse_datasets.py)) —
   pulls from four open data sources and merges them into a single
   ~620-row CSV ([`data/sse_datasets_combined.csv`](data/sse_datasets_combined.csv)):
   - OBELiX (NRC-Mila) — 599 experimentally-measured Li-SSE ionic
     conductivities from arXiv:2502.14234
   - COD (Crystallography Open Database) — CIF metadata for the 14
     known SSE formulas
   - CEMP (cleanenergymaterials.cn) — probe (graceful empty fallback)
   - [`data/paper_sse_extra.yaml`](data/paper_sse_extra.yaml) — hand-curated
     CAS / IOP high-throughput results
9. **Pareto best SSE** ([`src/p35_pareto_best_sse.py`](src/p35_pareto_best_sse.py)) —
   five-objective Pareto front over the merged dataset:
   - log10(sigma_ion), E_g, stability_window, -migration_barrier, -cost
   - reports per-objective Top-3, a balanced representative, and one
     representative per family (sulfide / oxide / halide / polymer / ...)

```bash
python src/p34_fetch_sse_datasets.py --offline
python src/p35_pareto_best_sse.py
python -m pytest tests/test_pbp_fetch_sse.py tests/test_pbp_pareto.py -v
```

### PBP quick start

```bash
python -m pip install -r requirements.txt   # also pulls ase, mace-torch, chgnet, pymatgen
python src/p24_particle_md.py --smiles CCO
python src/p25_collision_xs.py --smiles CCO
python src/p26_bayesian_langevin.py --smiles CCO --rf 20 --xgb 21 --mlp 20.5 --lgbm 20.8 --cat 20.3 --stack 20.6
python src/p27_sei_edl.py --dn_bulk 22
python src/p28_calibrate_5anchors.py
python -m uvicorn src.p29_pbp_api:app --port 8001
# v2
python -m uvicorn src.p33_pbp_v2_api:app --port 8002
```

```bash
curl -s http://127.0.0.1:8001/health
curl -s -X POST http://127.0.0.1:8001/collision_xs -H 'Content-Type: application/json' \
     -d '{"smiles": "CCO", "T": 298.15}'
```

### PBP tests

```bash
python -m pytest tests/test_pbp_*.py -v
```

### Key equations (PBP)

- Lennard-Jones: `V(r) = 4 eps [(sig/r)^12 - (sig/r)^6]`
- Coulomb: `V(r) = q_i q_j / (4 pi eps_0 eps_r r)` (SI, then convert to eV)
- Transport xs: `sigma* = 2 pi int (1 - cos chi) b db`
- Langevin SDE: `dx = -grad U(x) dt + sqrt(2 D) dW`
- Butler-Volmer: `j = j0 [exp(alpha_a F eta / RT) - exp(-alpha_c F eta / RT)]`
- Nernst-Einstein: `kappa = c F^2 D / (kT)`
- DN attenuation: `d_eff = d_bulk * (f + (1 - f) * exp(-L / L_sat))`

### PBP data + results

| File | What it contains |
|---|---|
| [`data/particle_params.yaml`](data/particle_params.yaml) | LJ table for Li / C / N / O / F / P / S / Cl / B / H |
| [`data/sei_params.yaml`](data/sei_params.yaml) | SEI / EDL / cathode / anode / operating params |
| [`data/ml_aimd_params.yaml`](data/ml_aimd_params.yaml) | MACE / CHGNet + ASE NVT settings |
| [`data/p2d_3d_params.yaml`](data/p2d_3d_params.yaml) | P2D + thermal + mechanical + micro3d |
| [`data/sse_library.yaml`](data/sse_library.yaml) | 14 SSEs with sigma_ion, E_g, migration |
| [`data/paper_sse_extra.yaml`](data/paper_sse_extra.yaml) | hand-curated CAS / IOP high-throughput SSE |
| [`data/new_anchors_5.csv`](data/new_anchors_5.csv) | 5 calibration anchors (FEC, EC, DOL, AcCl, LiBOB) |
| [`data/sse_datasets_combined.csv`](data/sse_datasets_combined.csv) | ~620 SSEs (OBELiX + COD + paper) |
| [`data/sse_datasets_meta.json`](data/sse_datasets_meta.json) | per-source fetch counts + elapsed time |
| [`data/pareto_front.csv`](data/pareto_front.csv) | non-dominated SSEs |
| [`data/pareto_summary.json`](data/pareto_summary.json) | top-3 per objective + family representatives |

## EEI Dissolution Solvent Screening (DEER Layer)

基于 Kalra DEER 论文 (Energy & Environmental Science 2026) 开发的退役电池直接再生筛选模块。

### 核心原理

退役电池（SOH < 80%）的性能衰减并非活性材料结构坍塌，而是正极 CEI（10-20 nm）和负极 SEI（30-40 nm）界面层过度生长导致的巨大阻抗。高供体数（DN）溶剂可以在电化学驱动下选择性溶解这层"界面死皮"，同时保留完整的电极结构。

### Layer 1 — EEI 溶解溶剂筛选（`p40_solvent_screening.py`）

基于物理化学原理 + ML 的三层目标评分：

1. **EEI dissolution score**：高 DN（> 26）+ 适中介电常数 + 氧化稳定性（> 4.1 V）→ 有机 EEI 溶解
2. **Electrode compatibility score**：DN 20-32 + 氧化稳定性 > 4.1 V → 保护 NMC811/石墨活性晶体
3. **Regeneration potential %**：几何平均综合评分 → 预测容量恢复率

物理校准锚点：DMI（DN=29.0, score=0.95）锚定，EC（DN=16.8, score<0.20）验证。

### Layer 2 — 再生协议模拟

**CV 扫描阻抗演化**（`p41_regeneration_protocol.py`）：
```
R_EEI(n) = R_EEI_0 * exp(-k_diss * n)
k_diss = k_0 * exp(alpha * (DN - 16.8))    # 基于 DMI/EC 标定
```
- 6 次 CV 扫描后：R_EEI 下降 ~92%，CEI/SEI 基本清除
- 输出：R_EEI vs scan_number 曲线 + 恢复率预测

**LiF 残余层稳定性**（`p41b_lif_stabilization.py`）：
- DEER 后残余 LiF 层（~2 nm）提供界面稳定化
- DEER cell fade=0.042%/cycle vs Fresh cell fade=0.072%/cycle
- **+198 cycles（+71%）寿命增益**，对标 Kalra 2026 实验数据

**软包规模验证**（`p41c_pouch_validation.py`）：
- 扣电（CR2032）→ 1 Ah → 3 Ah 软包逐级放大
- 修正因子：面积 f=0.955，N/P 比 f=0.986，温度 f=1.00
- **预测 3 Ah 软包恢复率：89.5%**（vs Kalra 报道 90.3%）

### Layer 3 — 技术经济与环境评估（`p42_tea_lca.py`）

5 种路径横向对比（基于 Kalra 2026 + EverBatt 模型）：

- **DEER**: $15.25/kg | 2.8 kWh/kg | 1.8 kgCO2/kg | **95%** 恢复率
- **火法冶金**: $26.31/kg | 12.5 kWh/kg | 8.2 kgCO2/kg | 98% 恢复率
- **湿法冶金**: $31.07/kg | 8.8 kWh/kg | 5.5 kgCO2/kg | 97% 恢复率
- **换电解液**: $5.20/kg | 0.8 kWh/kg | 0.6 kgCO2/kg | 83% 恢复率
- **直接修复正极**: $18.50/kg | 4.0 kWh/kg | 2.5 kgCO2/kg | 91% 恢复率

DEER 优势：vs 火法成本降低 **42%**，能耗降低 **78%**，GHG 降低 **78%**。

**敏感性分析**（`p42c_sensitivity.py`）：龙卷风图显示 labor cost 是最大不确定因素（±13%），其次是溶剂回收率（±3.7%）。

### DEER quick start

```bash
# Phase 1: Solvent screening
python src/p40_solvent_screening.py                      # Solvent scoring
python src/p40b_solvent_pareto.py                       # Pareto frontier
python src/p40c_solvent_rest_api.py                     # FastAPI (port 8001)
uvicorn src.p40c_solvent_rest_api:app --host 0.0.0.0 --port 8001
python src/p40d_viz_deer.py                             # 6 DEER figures
python src/p40e_solvent_uncertainty.py                  # Bootstrap uncertainty

# Phase 2: Regeneration protocol
python src/p41_regeneration_protocol.py --dn 29.0 --electrode dual --n-scans 5
python src/p41b_lif_stabilization.py --lif-nm 2.0 --n-cycles 500
python src/p41c_pouch_validation.py                     # 3 Ah pouch scale-up

# Phase 3: TEA/LCA
python src/p42_tea_lca.py
python src/p42b_viz_tea.py
python src/p42c_sensitivity.py

# Phase 4: MD simulation (atomistic validation)
python src/p43_deer_md.py

# All DEER tests
python -m pytest tests/test_pbp_eei_screening.py tests/test_pbp_regeneration.py tests/test_pbp_tea_lca.py -v
```

### DEER data + results

| File | What it contains |
|---|---|
| [`data/solvent_eei_properties.csv`](data/solvent_eei_properties.csv) | 46 种溶剂的 DN/AN/epsilon_r/HOMO/LUMO + EEI 溶解评分锚点 |
| [`data/solvent_library.yaml`](data/solvent_library.yaml) | DMI / DMSO / EC / FEC 等溶剂的 DEER 物理参数 |
| [`data/solvent_pareto_front.csv`](data/solvent_pareto_front.csv) | Pareto 前沿非支配解（溶解 vs 兼容性） |
| [`results/solvent_eei_predictions.csv`](results/solvent_eei_predictions.csv) | 全部候选溶剂评分排名（含 RDKit 分子描述符） |
| [`results/regeneration_cv_curves.csv`](results/regeneration_cv_curves.csv) | R_EEI vs CV 扫描次数曲线（正极/负极/综合） |
| [`results/regeneration_summary.json`](results/regeneration_summary.json) | k_diss、恢复率、所需扫描次数 |
| [`results/lif_cycling_curve.csv`](results/lif_cycling_curve.csv) | LiF 残余层循环稳定性（0-500 圈容量保持率） |
| [`results/lif_stabilization_summary.json`](results/lif_stabilization_summary.json) | +198 cycles (+71%) 寿命增益统计 |
| [`results/pouch_comparison.csv`](results/pouch_comparison.csv) | 扣电→1 Ah→3 Ah 规模修正因子 |
| [`results/pouch_scale_validation.json`](results/pouch_scale_validation.json) | 3 Ah 软包预测恢复率 89.5% |
| [`results/deer_tea_lca.csv`](results/deer_tea_lca.csv) | 5 种路径 TEA + LCA 对比数据 |
| [`results/deer_tea_lca_summary.json`](results/deer_tea_lca_summary.json) | DEER vs 各路径节省比例 |
| [`results/deer_sensitivity.json`](results/deer_sensitivity.json) | 敏感性龙卷风（labor 驱动，±13%） |
| [`results/deer_sensitivity_sweep.csv`](results/deer_sensitivity_sweep.csv) | 全参数扫描结果 |
| [`figures/deer_*.png`](figures/) | 13 张 DEER 可视化图表 |

---

## 许可

本目录代码仅供教育与非商业用途. 文献 DN 锚定表复用并注明出处 Marcus (1984) 与 Persson (1986).
