"""29_pbp_api.py - FastAPI exposing the 4 PBP models.

Endpoints:
  GET  /health                   - liveness
  POST /particle_dn              - run MD on a SMILES, return RDF + DN correction
  POST /collision_xs             - run scattering for a SMILES at T
  POST /langevin_dn              - Bayesian Langevin posterior on an ensemble pred
  POST /sei_impedance            - SEI/EDL sweep at a DN bulk
  POST /pbp_combine              - 4-model weighted DN for a SMILES
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import load_yaml  # noqa: E402
import particle_md as pmd  # noqa: E402
import collision_xs as cxs  # noqa: E402
import bayesian_langevin as bl  # noqa: E402
import sei_edl as sei_mod  # noqa: E402


app = FastAPI(title="donor-screener-pbp", version="0.1.0")
_PARAMS_P = {}
_PARAMS_S = {}


@app.on_event("startup")
def _startup():
    global _PARAMS_P, _PARAMS_S
    _PARAMS_P = load_yaml("particle_params.yaml")
    _PARAMS_S = load_yaml("sei_params.yaml")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class ParticleReq(BaseModel):
    smiles: str
    n_steps: Optional[int] = None
    T: Optional[float] = None


class CollisionReq(BaseModel):
    smiles: str
    T: float = 298.15


class LangevinReq(BaseModel):
    smiles: Optional[str] = None
    rf: Optional[float] = None
    xgb: Optional[float] = None
    mlp: Optional[float] = None
    lgbm: Optional[float] = None
    cat: Optional[float] = None
    stack: Optional[float] = None
    n_steps: int = 1500
    chains: int = 4
    D: int = 994
    seed: int = 0


class SEIReq(BaseModel):
    dn_bulk: float = 22.0
    T: float = 298.15


class CombineReq(BaseModel):
    smiles: str
    rf: float
    xgb: float
    mlp: float
    lgbm: float
    cat: float
    stack: float
    dn_anchor: Optional[float] = None  # optional calibration anchor (e.g. experimental)
    weights: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    return {"status": "ok", "models": ["particle_md", "collision_xs", "bayesian_langevin", "sei_edl"]}


@app.post("/particle_dn")
def particle_dn(req: ParticleReq):
    if not req.smiles:
        raise HTTPException(400, "smiles required")
    res = pmd.run_md(req.smiles, _PARAMS_P, _PARAMS_P["atoms"],
                     n_steps=req.n_steps, T=req.T)
    return res


@app.post("/collision_xs")
def collision_xs(req: CollisionReq):
    if not req.smiles:
        raise HTTPException(400, "smiles required")
    return cxs.run(req.smiles, _PARAMS_P, _PARAMS_P["atoms"], T=req.T)


@app.post("/langevin_dn")
def langevin_dn(req: LangevinReq):
    ens = {
        "dn_pred_rf_v2": req.rf, "dn_pred_xgb_v2": req.xgb,
        "dn_pred_mlp_v2": req.mlp, "dn_pred_lgbm_v2": req.lgbm,
        "dn_pred_cat_v2": req.cat, "dn_pred_stack_v2": req.stack,
    }
    ens = {k: v for k, v in ens.items() if v is not None}
    if not ens:
        raise HTTPException(400, "at least one of rf/xgb/mlp/lgbm/cat/stack required")
    return bl.infer_dn(ens, n_steps=req.n_steps, n_chains=req.chains, D=req.D, seed=req.seed)


@app.post("/sei_impedance")
def sei_impedance(req: SEIReq):
    return sei_mod.run(_PARAMS_S, dn_bulk=req.dn_bulk, T=req.T)


@app.post("/pbp_combine")
def pbp_combine(req: CombineReq):
    """Combine the 4 models with optional user-supplied weights.

    Default weights: langevin=0.6, particle=0.25, sei=0.15.
    """
    weights = req.weights or {"langevin": 0.6, "particle": 0.25, "sei": 0.15}
    anchor = req.dn_anchor if req.dn_anchor is not None else req.stack
    md_res = pmd.run_md(req.smiles, _PARAMS_P, _PARAMS_P["atoms"], n_steps=300)
    sei_res = sei_mod.run(_PARAMS_S, dn_bulk=anchor, T=298.15)["summary"]
    ens = {
        "dn_pred_rf_v2": req.rf, "dn_pred_xgb_v2": req.xgb,
        "dn_pred_mlp_v2": req.mlp, "dn_pred_lgbm_v2": req.lgbm,
        "dn_pred_cat_v2": req.cat, "dn_pred_stack_v2": req.stack,
    }
    langevin_res = bl.infer_dn(ens, n_steps=400, n_chains=3, D=128, seed=0)
    dn = (weights.get("langevin", 0.6) * langevin_res["dn_mean"]
          + weights.get("particle", 0.25) * (anchor + 0.5 * md_res["dn_correction"])
          + weights.get("sei", 0.15) * sei_res["dn_eff_mid"])
    return {
        "smiles": req.smiles,
        "dn_pred": float(dn),
        "langevin": langevin_res,
        "particle": {"dn_correction": md_res["dn_correction"],
                     "n_coord_li_O": md_res["n_coord_li_O"]},
        "sei": sei_res,
        "weights": weights,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
