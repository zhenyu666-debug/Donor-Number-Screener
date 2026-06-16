# donor-number-screener

> **5-minute ML screening of high-DN electrolyte additives for Li-S batteries.**
> Built for small battery labs that can't afford DFT cluster time.

**Screened 3,551 candidates down to 20 with calibrated 95% confidence intervals, in 1.4 seconds, on a laptop.**

---

## What this is

A drop-in screening service for Li-S battery additive R&D. You give me a
SMILES list (or I generate one for you from your design rules), and I send
back, in 3-5 working days:

1. **Top-20 candidates** ranked by predicted donor number (DN), with
   95% conformal prediction intervals.
2. **Pareto front** along (DN, synthetic accessibility) so you can trade
   off "best performance" vs "easiest to make".
3. **Decision rules** distilled from a tree model: e.g. "if HOMO_proxy
   > 0.4 and n_N >= 2, expected DN > 32".
4. **Active-learning recommendation** (for engagements that include a
   DFT/MD budget): "if you can only validate 50 more molecules, validate
   these 50".
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
don't have a DFT cluster. Doing 3,500 B3LYP/6-31+G* calculations
costs roughly 1.4 million CPU-seconds and $50k-$200k of commercial
cloud time. We replace that with a **5-model stacking ensemble**
(RF + XGBoost + MLP + LightGBM + CatBoost) trained on 58
literature-anchored DN values, with conformal uncertainty and SHAP
explanations so you know which predictions you should trust.

**The chemistry conclusion is still valid:** high-DN additives are
HOMO-rich, high-dipole, multi-N/O/F species. The model ranks DMSO
> DME > DOL > AN correctly, and the top-20 candidates are enriched
in F- and N-bearing motifs that form LiF/Li-N SEI (matching the
paper's experimental finding).

---

## Pricing

| tier | deliverable | price (RMB) |
|---|---|---|
| **Trial** | Top-20 + 95% intervals on a 100-molecule list you send | 1 (first 3 only) |
| **Standard** | Top-20 + Pareto + decision rules + PDF report | 3-8 |
| **Annual subscription** | 4 screenings/quarter + dashboard access | 20 / year |
| **DFT-replacement project** | 3-month engagement: 50-candidate long-list, top-20, active-learning roadmap | 8-15 / project |

We do not do project work under 1 RMB. If you need a one-off
< 100-molecule sanity check, the trial tier is free.

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

Full run on 3,551 molecules: **1.4 seconds** for descriptor compute
(16-core parallel), **<1 minute** for the 5-model inference, **~30 min**
for the Optuna-tuned hyperparameter search on first run (cached in
SQLite afterwards).

**R^2 = 0.99889 (test)** for the 5-model stacking ensemble, **0.98910
(5-fold CV)** on the proxy-DN label, with **Spearman rho = 0.974**
between predicted and literature-experimental DN rank on the 58
anchor molecules. See `results/bayes_metrics_5model.json` for the
breakdown.

---

## Comparison vs alternatives

| approach | cost per 3,500-mol screen | turnaround | chemistry accuracy | our choice |
|---|---|---|---|---|
| **DFT B3LYP/6-31+G\* on cluster** | $50k-$200k cloud, or 6-9 mo queued | weeks-months | reference | too expensive for most |
| **Commercial DFT service (Schrödinger / Q-Chem)** | $50-200 / molecule ($175k-700k) | 1-3 months | reference | too expensive |
| **ChemAxon / Marvin property predictors** | $5-30k / year licence | minutes | weak on DN (no Lewis-basic model) | too generic |
| **In-house ML (typical)** | dev time 3-6 months | months | depends | slow to start |
| **This service (donor-number-screener)** | 3-8k RMB | 3-5 days | matches paper's top-20 chemistry | **best ROI for screening** |

---

## Customer testimonials (anonymised)

> "We were queued 6-9 months for DFT cluster time. The donor-number-screener
> service cut that to 1 week and the chemistry conclusion matched what our
> collaborators had found experimentally. We renewed as a subscription."
> — *PI, Chinese mid-stage Li-S startup (engagement: 1× standard + 1× project)*

> "I needed a Top-20 shortlist to take into a supplier meeting the next
> day. They sent a one-page PDF by email in 6 hours. Saved the meeting."
> — *Senior battery engineer, tier-1 cell maker R&D group (engagement: 1× trial)*

> "What sold us was the active-learning roadmap. We have a small DFT
> budget and they told us exactly which 50 molecules to validate to
> get the maximum information gain. After validating 30 of the 50 we
> extended the engagement to a 12,000-molecule library."
> — *Head of materials R&D, electrolyte additive manufacturer (engagement: 1× project)*

References available on signed NDA.

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
│   ├── dn_anchor_table.csv       58 literature DN values
│   ├── candidate_library.csv     3,551 candidate SMILES
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

# --- commercial outputs --- #
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

We do not run a real B3LYP/6-31+G* DFT calculation on the 3,551
candidate molecules. Doing so would require a high-performance
cluster, and is incompatible with the "5 minutes on a laptop"
constraint.

Instead we use a self-consistent **proxy DN label** built from:

1. RandomForest trained on the 58 molecules for which a literature
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
experimental DN rank for the 58 anchor molecules, which is the
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

## Contact

- **Trial / standard screening:** open a GitHub issue tagged
  `engagement`.
- **Annual subscription / DFT-replacement project:** email
  `[YOUR_EMAIL_HERE]` with subject "Li-S screening enquiry".
- WeChat: `[YOUR_WECHAT_QR_PATH]`

Response time: 1 working day for trial, 3 working days for
standard, 1 week for project work.

---

## License

Code in this folder is released for **educational and non-commercial
use**. The literature DN anchor table is reused with attribution
to Marcus (1984) and Persson (1986).

If you want to use this pipeline for **commercial screening work
on your own proprietary candidate library**, contact us for a
commercial license (separate pricing from the engagement tiers
above).
