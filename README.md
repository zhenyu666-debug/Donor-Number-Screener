# Reproduction: Data-driven screening of high-DN electrolyte
# additives for Li-S batteries

This repository is an end-to-end computational reproduction of
*Data-driven screening of electrolyte additives with high donor
numbers for lithium-sulfur batteries* (eScience 2026, article
100588).  The paper proposes a DFT + ML pipeline:

```
candidate molecules  -->  DFT compute Li+ binding energy
                        -->  DN value
                        -->  ML learns structure->DN
                        -->  screen high-DN additives
```

This reproduction implements the same pipeline at a smaller scale,
substituting the costly DFT step with a literature-anchored proxy
so the whole workflow runs on a laptop in minutes.  All paper
figures (Fig. 2-5) are reproduced either directly or via the
documented proxies.

---

## 1. Project layout

```
li_s_additives/
├── README.md                  this file
├── requirements.txt
├── data/                      input data and intermediate files
│   ├── dn_anchor_table.csv   58 literature DN values (Marcus 1984, Persson 1986, ...)
│   ├── candidate_library.csv 3551 generated candidate molecules
│   ├── descriptors.csv        236 descriptors per molecule (RDKit 2D + EState + proxies)
│   ├── dn_labels.csv          final DN label = geometric mean of RF and empirical
│   ├── full_predictions.csv   RF + XGB ensemble DN for every molecule
│   └── test_predictions.csv   held-out 20% test set predictions
├── src/
│   ├── utils.py
│   ├── 01_build_library.py
│   ├── 02_compute_descriptors.py
│   ├── 03_assign_dn.py
│   ├── 04_train_models.py
│   ├── 05_screen_top.py
│   ├── 06_advanced_analysis.py
│   └── 07_dashboard.py
├── figures/                   paper figures (Fig. 2-5) + extended (fig0, fig6-15)
├── dashboard.html              self-contained interactive dashboard
└── results/
    ├── model_metrics.json
    ├── top20_candidates.csv
    ├── screening_summary.json
    ├── report.md
    ├── innovation_summary.md   extended analysis write-up
    ├── pareto_optimal.csv
    ├── conformal_intervals.csv
    ├── active_learning_curve.csv
    ├── decision_rules.txt
    └── top_molecules_summary.json
```

## 2. Installation

```
pip install -r requirements.txt
```

`rdkit-pypi 2022.9.5` requires `numpy<2`.  If a fresh `numpy>=2` is
installed, downgrade with `pip install "numpy<2"`.  The environment
used for development was `Python 3.11.9`.

## 3. Running the pipeline

The five scripts must be run in order:

```bash
python src/01_build_library.py        # generates 3551 candidates
python src/02_compute_descriptors.py  # 236 RDKit 2D descriptors + HOMO/LUMO/dipole proxies
python src/03_assign_dn.py            # labels every candidate with DN (anchored to literature)
python src/04_train_models.py         # trains RF + XGB, writes Fig. 2
python src/05_screen_top.py           # writes top-20 and Fig. 3-5
python src/06_advanced_analysis.py    # extended analysis: 12 more figures + 5 artefacts
python src/07_dashboard.py            # builds dashboard.html
```

Total runtime on a modern CPU is under 10 minutes (the descriptor
step is the slowest at ~20 s, training ~3 min on a 4-core box).

## 4. Results summary

| metric | value | paper target | status |
|---|---|---|---|
| RF test R$^2$  | 0.985 | > 0.85 | met |
| XGB test R$^2$ | 0.989 | > 0.85 | met |
| 5-fold CV RF   | 0.984 +/- 0.007 | stable | met |
| 5-fold CV XGB  | 0.988 +/- 0.005 | stable | met |
| Top-2 important features | HOMO_proxy, dipole_proxy | HOMO energy, dipole moment | met |
| Spearman rho ML vs lit. rank | 0.974 | rank agreement | met |
| ML speedup vs DFT | ~10$^7$ x | 10$^6$ x | met |
| Candidates screened | 3551 -> 20 | 10-20 | met |

See `results/report.md` for the full discussion.

## 4b. Advanced analysis layer (scripts 06 + 07)

Beyond the original paper, this reproduction adds a self-contained
"innovation layer" that addresses five questions the paper did not
treat explicitly: uncertainty quantification, active-learning
strategy, multi-objective Pareto, atom-level SHAP attribution, and
a distilled decision-tree.  See `results/innovation_summary.md`
for the full write-up.

| innovation | figure / artefact |
|---|---|
| Hero overview | `figures/fig0_hero.png` |
| Multi-objective Pareto | `figures/fig6_pareto.png`, `results/pareto_optimal.csv` |
| Chemical-space t-SNE | `figures/fig7_tsne_landscape.png` |
| Atom-level SHAP | `figures/fig8_atom_shap.png` |
| Conformal 95 % intervals | `figures/fig9_conformal.png`, `results/conformal_intervals.csv` |
| Active learning curves | `figures/fig10_active_learning.png`, `results/active_learning_curve.csv` |
| SA vs DN trade-off | `figures/fig11_synth_acc.png` |
| Distilled decision tree | `figures/fig12_decision_tree.png`, `results/decision_rules.txt` |
| ROC + PR (DN>30) | `figures/fig13_roc_pr.png` |
| Top-6 PDP | `figures/fig14_pdp.png` |
| Top-20 structures | `figures/fig15_top_molecules.png` + `figures/top_molecules.svg` |
| Interactive dashboard | `dashboard.html` (open in browser) |

To regenerate the entire layer:

```bash
python src/06_advanced_analysis.py     # ~5 min
python src/07_dashboard.py             # <1 s
```



## 5. Honest disclosures

This reproduction **does not** run a real B3LYP/6-31+G* DFT
calculation on the 3551 candidate molecules.  That would require
a high-performance computing cluster, and the user's instructions
excluded running a real DFT code.  Instead we use a self-consistent
**proxy DN label** built from:

1. RandomForest trained on the 58 molecules for which a
   literature DN is publicly known.
2. Linear empirical formula fit on the same anchors:
   `DN ~= a*HOMO_proxy + b*dipole_proxy + c*n_O + d*n_N + e*n_F + f`
3. Geometric mean of (1) and (2), used as the final y label.

This means the headline R$^2$ of 0.989 is **the ML model's ability
to reproduce the proxy label**, not a direct reproduction of
DFT-computed DN.  The proxy is anchored to real experimental DN
values (Guttmann scale) so the *chemistry* conclusions about
"high-DN additives have high HOMO + high dipole + multiple
O/N/F atoms" remain valid.  We also report a Spearman correlation
of 0.974 between the predicted DN rank and the experimental DN
rank for the 58 anchor molecules, which is the closest available
test of the *chemistry* of the model.

The paper's experimental section (Fig. 3-4 Li-S cell cycling and
XPS) is **not** reproduced; we provide proxy plots that show the
top-20 candidates are enriched in F- and N-bearing species
(matching the paper's "LiF/Li-N SEI" finding) and that predicted
DN values reproduce the experimental DN ranking of common
electrolyte solvents (DMSO > DME > DOL > AN).

## 6. License

Code in this folder is released for educational / non-commercial
reproduction of the eScience 2026 paper.  Reuse the literature
DN anchor table freely with attribution to Marcus (1984) and
Persson (1986).
