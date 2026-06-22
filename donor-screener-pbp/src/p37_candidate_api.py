"""37_candidate_api.py - FastAPI exposing the candidate SSE scoring pipeline.

Endpoints:
  GET  /health              - liveness + library summary
  GET  /candidates          - all 14 candidates (current scores)
  POST /candidate_score     - {id: int} or {formula: str} -> single DN breakdown
  POST /candidate_enrich    - {id: int, force: bool=false} -> run p37 enrich on one id
  GET  /candidate_rank      - ?top=5&sort=dn_candidate -> top-N ranked

This service is additive to p29_pbp_api.py (v1) and p33_pbp_v2_api.py (v2).
It does not import those modules -- it is standalone and only depends on
sse_library.yaml and p37_candidate_score.py.

CLI:
  python src/p37_candidate_api.py          # serve on 0.0.0.0:8765
  python src/p37_candidate_api.py --port 9000
  python src/p37_candidate_api.py --reload
"""
from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import DATA_DIR  # noqa: E402

import p37_candidate_score as cs  # noqa: E402
import p37_enrich_candidates as ce  # noqa: E402


# --------------------------------------------------------------------------- #
# Startup (lifespan)
# --------------------------------------------------------------------------- #

def _build_state() -> dict:
    lib = cs.load_source_library()
    scored = cs.score_library(lib)
    by_id = {r["id"]: r for r in scored}
    by_formula = {r["formula"]: r for r in scored}
    return {
        "lib": lib,
        "scored": scored,
        "by_id": by_id,
        "by_formula": by_formula,
        "n": len(scored),
        "src": ("sse_library_enriched.yaml"
                if (DATA_DIR / "sse_library_enriched.yaml").exists()
                else "sse_library.yaml"),
    }


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _STATE.update(_build_state())
    yield


app = FastAPI(title="donor-screener-pbp candidate API",
              version="0.1.0",
              description="Score 14 SSE candidates from sse_library.yaml"
                          " with the v2 weighted-DN formula.",
              lifespan=_lifespan)

_STATE: dict = {}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class ScoreRequest(BaseModel):
    id: Optional[int] = Field(default=None, description="SSE library id 1..14")
    formula: Optional[str] = Field(default=None,
                                   description="Chemical formula (ad-hoc)")
    class_override: Optional[str] = Field(default=None,
                                          description="class for ad-hoc formula")


class EnrichRequest(BaseModel):
    id: int = Field(..., description="SSE library id 1..14")
    force: bool = Field(default=False,
                        description="Re-run even if enrichment already exists")


# --------------------------------------------------------------------------- #
# Startup (handled by the lifespan context above)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "n_candidates": _STATE.get("n", 0),
        "lib": _STATE.get("src", "sse_library.yaml"),
    }


@app.get("/candidates")
def list_candidates() -> List[dict]:
    return _STATE.get("scored", [])


@app.post("/candidate_score")
def candidate_score(req: ScoreRequest) -> dict:
    if req.id is not None:
        row = _STATE.get("by_id", {}).get(req.id)
        if row is None:
            raise HTTPException(status_code=404,
                                detail=f"id {req.id} not in library")
        return row

    if req.formula:
        # ad-hoc candidate: score against class defaults
        entry = {
            "id": 0,
            "name": f"adhoc:{req.formula}",
            "formula": req.formula,
            "class": (req.class_override or "other").lower(),
            "sigma_ion_S_cm": float("nan"),
        }
        filled = cs.fill_class_defaults(entry)
        comps = cs.compute_dn_candidate_v2(filled)
        return {**filled, **comps, "rank": None,
                "enrichment_source": "adhoc",
                "notes": "scored from class defaults only"}

    raise HTTPException(status_code=400,
                        detail="provide either `id` or `formula`")


@app.post("/candidate_enrich")
def candidate_enrich(req: EnrichRequest) -> dict:
    """Run p37_enrich_candidates on a single id. Updates state."""
    lib = _STATE.get("lib", [])
    target = next((e for e in lib if e.get("id") == req.id), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"id {req.id} not found")
    if "enrichment" in target and not req.force:
        return {"id": req.id, "status": "already-enriched",
                "entry": target}

    counter = [0]
    enriched = ce.enrich_entry(target, offline=False,
                               request_count=counter,
                               max_requests=10)
    # replace in lib, re-score
    new_lib = [enriched if e.get("id") == req.id else e for e in lib]
    scored = cs.score_library(new_lib)
    _STATE["lib"] = new_lib
    _STATE["scored"] = scored
    _STATE["by_id"] = {r["id"]: r for r in scored}
    _STATE["by_formula"] = {r["formula"]: r for r in scored}
    _STATE["n"] = len(scored)
    return {"id": req.id, "status": "enriched",
            "http_requests": counter[0],
            "enriched_entry": enriched,
            "new_score": _STATE["by_id"].get(req.id)}


@app.get("/candidate_rank")
def candidate_rank(top: int = Query(default=5, ge=1, le=50),
                   sort: str = Query(default="dn_candidate")) -> List[dict]:
    rows = _STATE.get("scored", [])
    if sort != "dn_candidate":
        rows = sorted(rows, key=lambda r: r.get(sort, 0), reverse=True)
    return rows[:top]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()
    import uvicorn
    uvicorn.run("p37_candidate_api:app",
                host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
