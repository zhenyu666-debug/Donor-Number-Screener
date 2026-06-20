"""33_pbp_v2_api.py - FastAPI exposing the v2 PBP models (ML-AIMD + P2D + SSE).

Endpoints (NEW in v2):
  POST /aimd_interface  - SSE name -> interface energy, barrier, dn_aimd
  POST /p2d_solve       - P2D params -> V(t) / T(t) / sigma(t)
  POST /sse_rank        - all 14 SSEs re-ranked

The v1 endpoints (/health, /particle_dn, /collision_xs, /langevin_dn,
/sei_impedance, /pbp_combine) live in src/29_pbp_api.py and remain
unchanged. This module is additive.
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
import ml_aimd as aimd  # noqa: E402
import p2d_3d_micro as p2d  # noqa: E402
import sse_redn as sse  # noqa: E402


app = FastAPI(title="donor-screener-pbp v2", version="0.2.0")
_PARAMS = {}
_CALC = None
_LIB = []
_RK = {}


@app.on_event("startup")
def _startup():
    global _PARAMS, _CALC, _LIB, _RK
    _PARAMS["aimd"] = load_yaml("ml_aimd_params.yaml")
    _PARAMS["p2d"] = load_yaml("p2d_3d_params.yaml")
    _CALC = aimd.load_calculator(_PARAMS["aimd"])
    _LIB, _RK = sse.load_sse_library()


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class AIMDReq(BaseModel):
    sse: str
    n_steps: Optional[int] = None


class P2DReq(BaseModel):
    n_steps: int = 200
    dt: float = 1.0
    c_rate: float = 1.0


class SSERankReq(BaseModel):
    anchor_dn: float = 22.0
    weights: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    return {
        "status": "ok",
        "v2_models": ["ml_aimd", "p2d_3d_micro", "sse_redn"],
        "aimd_backend": _CALC[0] if _CALC else "none",
        "n_sse_loaded": len(_LIB),
    }


@app.post("/aimd_interface")
def aimd_interface(req: AIMDReq):
    if not req.sse:
        raise HTTPException(400, "sse required")
    by_name = {x["name"]: x for x in _LIB}
    if req.sse not in by_name:
        raise HTTPException(404, f"unknown SSE: {req.sse}")
    params = _PARAMS["aimd"]
    if req.n_steps is not None:
        params = {**params, "md": {**params["md"], "n_steps": int(req.n_steps)}}
    r = aimd.run_one_sse(req.sse, by_name[req.sse], params, _CALC)
    return r


@app.post("/p2d_solve")
def p2d_solve(req: P2DReq):
    res = p2d.solve_p2d(_PARAMS["p2d"], n_steps=req.n_steps, dt=req.dt)
    return {
        "summary": res["summary"],
        "n_rows": len(res["rows"]),
        "first_5_rows": res["rows"][:5],
        "last_5_rows": res["rows"][-5:],
        "micro_n_particles": res["micro"]["n_particles"] if res["micro"] else 0,
    }


@app.post("/sse_rank")
def sse_rank(req: SSERankReq):
    rows = sse.rerank(weights=req.weights, anchor_dn=req.anchor_dn)
    return {
        "n_sse": len(rows),
        "rows": rows,
        "top3": [r["sse"] for r in rows[:3]],
        "bottom3": [r["sse"] for r in rows[-3:]],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    args = p.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
