# donor-screener-pbp (Particle-Bayes-Physics)

A physics-layer extension on top of `donor-number-screener`. Adds four
analytical / Monte-Carlo models that complement the 5-model GBDT stack:

1. Particle MD (`src/24_particle_md.py`) - Lennard-Jones + Coulomb NVT
   simulation in a 64-particle periodic box. Outputs the Li-O radial
   distribution g(r), Li-O coordination number, and a DN correction.
2. Collision cross-section (`src/25_collision_xs.py`) - classical
   scattering on the LJ potential. Outputs the transport cross-section
   sigma*, the dimensionless collision integral Omega^(1,1), mobility
   mu, and Nernst-Einstein ionic conductivity kappa.
3. Bayesian Langevin diffusion (`src/26_bayesian_langevin.py`) -
   994-dim stochastic gradient Langevin dynamics on a Gaussian posterior
   anchored on the 5-model stack. Multi-chain sampling with R-hat
   diagnostic, 95% CI, effective sample size.
4. SEI / EDL impedance (`src/27_sei_edl.py`) - three-sandwich
   cathode | CEI | electrolyte | SEI | Li metal analytical model.
   Helmholtz capacitance, Butler-Volmer kinetics, Nernst-Planck bulk
   conductivity, and a DN attenuation factor through the dense SEI layer.

A calibration script (`src/28_calibrate_5anchors.py`) runs all four on
five new anchor molecules (FEC, EC, DOL, Acetyl chloride, LiBOB) and
reports MAE / RMSE against experimental DN values.

A FastAPI service (`src/29_pbp_api.py`) exposes six endpoints
(`/health`, `/particle_dn`, `/collision_xs`, `/langevin_dn`,
`/sei_impedance`, `/pbp_combine`).

## v2 additions

Three more physics layers (the most complex micro + macro layers used
in modern battery design):

5. **ML-AIMD** (`src/30_ml_aimd.py`) - Machine-Learning Accelerated MD
   with MACE-MP-0 / CHGNet foundation models + ASE NVT. We build a
   Li | SSE interface and report the interface adhesion energy, Li
   migration barrier, and a DN correction. Falls back to LJ + Coulomb
   when MACE/CHGNet/ASE are unavailable.
6. **P2D + 3D micro-structure** (`src/31_p2d_3d_micro.py`) - full
   Newman 1991 P2D (radial + 1D + Butler-Volmer + Poisson) with the
   three additional coupled fields:
   - thermal: Fourier heat + Joule + entropic heat
   - mechanical: Hooke + diffusion-induced stress
   - 3D micro: random-close-packed NMC particles with per-particle j
7. **SSE re-ranking** (`src/32_sse_redn.py`) - 14 mainstream solid
   electrolytes (Li3PS4, Li6PS5Cl, LGPS, Li7P3S11, Li2S-P2S5 glass,
   Li6PS5Br, Li3PS4 glass, LLZO, LATP, LAGP, LiPON, LISICON,
   Li6PS5I, PEO+LiTFSI) re-estimated with the 7-model combined DN.

A second FastAPI service (`src/33_pbp_v2_api.py`, port 8002) exposes
three new endpoints (`/aimd_interface`, `/p2d_solve`, `/sse_rank`).

## v2.1 additions

8. **SSE dataset fetcher** (`src/34_fetch_sse_datasets.py`) - pulls from
   four open data sources and merges them into a single ~620-row CSV:
   - OBELiX (NRC-Mila) - 599 experimentally-measured Li-SSE ionic
     conductivities from arXiv:2502.14234
   - COD (Crystallography Open Database) - CIF metadata for the 14
     known SSE formulas
   - CEMP (cleanenergymaterials.cn) - probe (graceful empty fallback)
   - `paper_sse_extra.yaml` - hand-curated CAS / IOP high-throughput
     results (LGSSSI, Li2SiO3, doped-Li3PS4, halides, ...)
9. **Pareto best SSE** (`src/35_pareto_best_sse.py`) - five-objective
   Pareto front over the merged dataset:
   - log10(sigma_ion), E_g, stability_window, -migration_barrier, -cost
   - reports per-objective Top-3, a balanced representative, and one
     representative per family (sulfide / oxide / halide / polymer / ...)

Run offline:
```bash
python src/34_fetch_sse_datasets.py --offline
python src/35_pareto_best_sse.py
python -m pytest tests/test_fetch_sse.py tests/test_pareto.py -v
```

## Layout

```
donor-screener-pbp/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- new_anchors_5.csv         5 new anchors with experimental DN
|   |-- particle_params.yaml      LJ + Coulomb + thermostat params
|   |-- sei_params.yaml           SEI / EDL / cathode / anode params
|   |-- sse_library.yaml          v2 14 SSEs with sigma_ion, E_g, migration
|   |-- ml_aimd_params.yaml       v2 MACE / CHGNet + ASE NVT settings
|   |-- p2d_3d_params.yaml        v2 P2D + thermal + mechanical + micro3d
|   |-- paper_sse_extra.yaml      v2.1 hand-curated CAS / IOP high-throughput SSE
|   |-- sse_datasets_combined.csv v2.1 merged ~620 SSEs (OBELiX + COD + paper)
|   |-- sse_datasets_meta.json    v2.1 per-source fetch counts
|   |-- pareto_front.csv          v2.1 non-dominated SSEs
|   `-- pareto_summary.json       v2.1 top-3 per objective + family reps
|-- src/
|   |-- 24_particle_md.py
|   |-- 25_collision_xs.py
|   |-- 26_bayesian_langevin.py
|   |-- 27_sei_edl.py
|   |-- 28_calibrate_5anchors.py
|   |-- 29_pbp_api.py
|   |-- 30_ml_aimd.py             v2 ML-AIMD (MACE / CHGNet + fallback)
|   |-- 31_p2d_3d_micro.py        v2 P2D + thermal + mechanical + 3D micro
|   |-- 32_sse_redn.py            v2 14 SSE re-ranking
|   |-- 33_pbp_v2_api.py          v2 FastAPI v2 (3 new endpoints)
|   |-- 34_fetch_sse_datasets.py  v2.1 fetch OBELiX + COD + CEMP + paper SSE
|   |-- 35_pareto_best_sse.py     v2.1 multi-objective Pareto front
|   `-- utils_pb.py
|-- tests/
|   |-- test_particle_energy.py
|   |-- test_collision_xs.py
|   |-- test_langevin.py
|   `-- test_sei.py
`-- results/                       (CSVs and JSONs written here at runtime)
```

## Quick start

```bash
python -m pip install -r requirements.txt
python src/24_particle_md.py --smiles CCO
python src/25_collision_xs.py --smiles CCO
python src/26_bayesian_langevin.py --smiles CCO --rf 20 --xgb 21 --mlp 20.5 --lgbm 20.8 --cat 20.3 --stack 20.6
python src/27_sei_edl.py --dn_bulk 22
python src/28_calibrate_5anchors.py
python -m uvicorn src.29_pbp_api:app --port 8001
# v2
python -m uvicorn src.33_pbp_v2_api:app --port 8002
```

Then probe the API:

```bash
curl -s http://127.0.0.1:8001/health
curl -s -X POST http://127.0.0.1:8001/collision_xs -H 'Content-Type: application/json' \
     -d '{"smiles": "CCO", "T": 298.15}'
```

## Run the tests

```bash
python -m pytest tests/
```

## Key equations

- Lennard-Jones: `V(r) = 4 eps [(sig/r)^12 - (sig/r)^6]`
- Coulomb: `V(r) = q_i q_j / (4 pi eps_0 eps_r r)` (SI, then convert to eV)
- Transport xs: `sigma* = 2 pi int (1 - cos chi) b db`
- Langevin SDE: `dx = -grad U(x) dt + sqrt(2 D) dW`
- Butler-Volmer: `j = j0 [exp(alpha_a F eta / RT) - exp(-alpha_c F eta / RT)]`
- Nernst-Einstein: `kappa = c F^2 D / (kT)`
- DN attenuation: `d_eff = d_bulk * (f + (1 - f) * exp(-L / L_sat))`

## Relationship to the existing screener

This repo does NOT modify or import from `donor-number-screener/`.
It can be used standalone. The Bayesian Langevin module is designed to
consume a 5-model stack prediction if one is available; otherwise it
falls back to a single-anchor Gaussian posterior. The particle MD module
reads `data/particle_params.yaml` (LJ table for Li / C / N / O / F / P /
S / Cl / B / H), and the SEI module reads `data/sei_params.yaml`.

## Output files

| File | What it contains |
|---|---|
| results/particle_md_rdf.csv   | r vs g(r) radial distribution |
| results/particle_md_rdf.json  | T, n_coord, dn_correction |
| results/collision_xs.csv      | sigma*, omega, mu, kappa at 5 temperatures |
| results/langevin_samples.csv  | 4 chains x N steps of DN posterior |
| results/langevin_samples.json | posterior mean, 95% CI, R-hat, ESS |
| results/sei_impedance.csv     | R_sei, R_cei, R_bulk, C_H, eta, tau, dn_eff at each thickness |
| results/sei_impedance.json    | summary at mid-thickness |
| results/calibration_5anchor.csv | per-anchor dn_pred vs dn_expt |
| results/pbp_metrics.json      | overall MAE / RMSE + per-row detail |

## License

Internal donor-screener project. Not for redistribution.
