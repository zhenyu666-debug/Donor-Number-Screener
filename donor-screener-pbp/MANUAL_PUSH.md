# Manual git + verification steps

If the shell tool is stuck, run these commands manually to push
`donor-screener-pbp` to GitHub.

## 1. Verify the code works (3-5 min)

```bash
cd c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp
python -m pip install -r requirements.txt

# Run the four model CLIs (each writes to results/)
python src/24_particle_md.py --smiles CCO
python src/25_collision_xs.py --smiles CCO
python src/26_bayesian_langevin.py --smiles CCO --rf 20 --xgb 21 --mlp 20.5 --lgbm 20.8 --cat 20.3 --stack 20.6
python src/27_sei_edl.py --dn_bulk 22

# Calibrate on 5 new anchors
python src/28_calibrate_5anchors.py

# Run unit tests
python -m pytest tests/ -v
```

Expected:
- 4 CSVs in `results/` with 20+ rows each
- `calibration_5anchor.csv` with 5 rows, MAE reported in stdout
- All 16 unit tests pass

## 2. Boot the FastAPI (1 min)

```bash
cd c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp
python -m uvicorn src.29_pbp_api:app --port 8001
```

Then in another shell:

```bash
curl -s http://127.0.0.1:8001/health
curl -s -X POST http://127.0.0.1:8001/collision_xs ^
     -H "Content-Type: application/json" ^
     -d "{\"smiles\": \"CCO\", \"T\": 298.15}"
```

## 3. git init + first commit (30 s)

```bash
cd c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp
git init
git config user.email "you@example.com"    # only if not set globally
git config user.name  "Your Name"          # only if not set globally
git add .
git commit -m "feat: 4 physics-layer models (particle MD, collision XS, Bayesian Langevin, SEI/EDL) with 5-anchor calibration"
```

## 4. Push to GitHub

Create the empty repo on GitHub first (UI), then:

```bash
git remote add origin https://github.com/<your-username>/donor-screener-pbp.git
git branch -M main
git push -u origin main
```

# 3. v2 verification

```bash
cd c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp
python -m pip install -r requirements.txt  # also pulls ase, mace-torch, chgnet, pymatgen

# Optional: only if you have a CUDA GPU
# python -c "from mace.calculators import mace_mp"  # smoke test

# v2 CLIs
python src/30_ml_aimd.py                     # all 14 SSEs, default backend
python src/30_ml_aimd.py --sse "Li10GeP2S12 (LGPS)"
python src/31_p2d_3d_micro.py --steps 100
python src/32_sse_redn.py

# Boot v2 API on port 8002
python -m uvicorn src.33_pbp_v2_api:app --port 8002
```

Probe the v2 API:

```bash
curl -s http://127.0.0.1:8002/health
curl -s -X POST http://127.0.0.1:8002/sse_rank ^
     -H "Content-Type: application/json" ^
     -d "{\"anchor_dn\": 22.0}"
curl -s -X POST http://127.0.0.1:8002/aimd_interface ^
     -H "Content-Type: application/json" ^
     -d "{\"sse\": \"Li10GeP2S12 (LGPS)\"}"
curl -s -X POST http://127.0.0.1:8002/p2d_solve ^
     -H "Content-Type: application/json" ^
     -d "{\"n_steps\": 50, \"dt\": 1.0}"
```

Run the v2 tests (no MACE/CHGNet required, they use the LJ fallback):

```bash
python -m pytest tests/test_ml_aimd.py tests/test_p2d.py tests/test_sse_redn.py -v
```

## 4. git commit + push v2 (incremental)

```bash
cd c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp

# stage v2 files
git add src/30_ml_aimd.py src/31_p2d_3d_micro.py src/32_sse_redn.py src/33_pbp_v2_api.py
git add data/sse_library.yaml data/ml_aimd_params.yaml data/p2d_3d_params.yaml
git add tests/test_ml_aimd.py tests/test_p2d.py tests/test_sse_redn.py
git add results/ml_aimd_interface.csv results/p2d_voltage_curve.csv \
        results/p2d_3d_micro.csv results/sse_dn_rerank.csv results/pbp_v2_metrics.json
git add requirements.txt README.md MANUAL_PUSH.md
git add start.bat

git commit -m "feat: v2 with ML-AIMD (MACE/CHGNet) + P2D/3D multi-field + 14 SSE re-rank"
git push
```

If v1 was never committed, do an initial commit first:

```bash
git add .
git commit -m "feat: PBP v1 (particle MD, collision XS, Bayesian Langevin, SEI/EDL) + v2 (ML-AIMD, P2D/3D, SSE)"
git push -u origin main
```

# 5. v2.1 verification

```bash
cd c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp

# v2.1 dataset fetcher
python src/34_fetch_sse_datasets.py            # online (OBELiX + COD + paper)
python src/34_fetch_sse_datasets.py --offline  # offline (paper only)

# v2.1 Pareto front
python src/35_pareto_best_sse.py

# v2.1 tests
python -m pytest tests/test_fetch_sse.py tests/test_pareto.py -v
```

Expected:

- `data/sse_datasets_combined.csv` >= 30 unique SSEs (offline)
  up to 620 (online w/ OBELiX)
- `data/pareto_front.csv` >= 5 non-dominated rows
- `data/pareto_summary.json` lists top-3 per objective and
  per-family representatives

## 6. git commit + push v2.1 (incremental)

```bash
cd c:\Users\Hasee\.qclaw\workspace\get_jobs\donor-screener-pbp

git add src/34_fetch_sse_datasets.py src/35_pareto_best_sse.py
git add data/paper_sse_extra.yaml
git add data/sse_datasets_combined.csv data/sse_datasets_meta.json
git add data/pareto_front.csv data/pareto_summary.json
git add tests/test_fetch_sse.py tests/test_pareto.py
git add README.md MANUAL_PUSH.md

git commit -m "feat: v2.1 SSE datasets fetch (OBELiX+COD+CEMP+paper) + Pareto front"
git push
```

If you have no remote, add one (create the empty repo on GitHub first):

```bash
git remote add origin https://github.com/<your-username>/donor-screener-pbp.git
git branch -M main
git push -u origin main
```
