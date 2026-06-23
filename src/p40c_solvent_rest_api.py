"""p40c_solvent_rest_api.py - FastAPI endpoints for EEI solvent screening.

Extends the existing FastAPI service (p29_pbp_api.py on port 8001)
with DEER-specific screening endpoints.

New endpoints:
  POST /solvent_screen      - score one or more solvents by SMILES
  GET  /solvent_top_k      - return top-K from cached results
  GET  /solvent_pareto      - return the current Pareto front
  POST /solvent_compare    - compare two solvents side-by-side
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils import RESULTS_DIR, DATA_DIR  # noqa: E402

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# --------------------------------------------------------------------------- #
# Pydantic request/response models
# --------------------------------------------------------------------------- #

class SolventInput(BaseModel):
    smiles: str = Field(..., description="SMILES string of the solvent")
    name:   str | None = Field(None, description="Optional human-readable name")


class SolventListInput(BaseModel):
    solvents: list[SolventInput] = Field(..., description="List of solvents to screen")


class SolventResult(BaseModel):
    name:                  str
    smiles:                str
    dn:                    float
    eei_dissolution_score: float
    electrode_compat_score: float
    regeneration_potential_pct: float
    composite_score:       float
    tier:                  str | None = None


class CompareInput(BaseModel):
    smiles_a: str
    name_a:   str
    smiles_b: str
    name_b:   str


# --------------------------------------------------------------------------- #
# Shared scoring logic (same as p40_solvent_screening.py)
# --------------------------------------------------------------------------- #

DN_NORM_BOT = 14.0
DN_NORM_TOP = 29.0


def physics_score(dn: float, an: float, epsilon_r: float,
                  oxidation_V: float = 4.2, reduction_V: float = 0.8,
                  viscosity_cp: float = 2.0,
                  melting_point_c: float = -100.0) -> dict:
    dn_norm = max(0.0, min(1.0, (dn - DN_NORM_BOT) / (DN_NORM_TOP - DN_NORM_BOT)))

    _eps = max(0.0, epsilon_r - 35.0)
    eps_comp = max(0.0, 1.0 - 0.015 * _eps)
    if epsilon_r > 80.0:
        eps_comp = 0.0
    elif epsilon_r > 60.0:
        eps_comp = min(eps_comp, 0.2)

    ox_factor = min(1.0, max(0.0, (oxidation_V - 3.5) / 1.3))
    red_factor = min(1.0, max(0.0, (reduction_V - 0.1) / 0.9))
    visc_abs = max(abs(viscosity_cp), 0.1)
    visc_factor = min(1.2, 2.1 / visc_abs)

    raw = (
        dn_norm     * 0.50 +
        eps_comp    * 0.25 +
        ox_factor  * 0.20 +
        red_factor  * 0.05 +
        visc_factor * 0.05
    )

    eps_mult = 0.5 if epsilon_r > 80.0 else 1.0
    raw = raw * eps_mult

    eei_diss = max(0.0, min(1.0, raw))
    dn_penalty = 0.0 if dn <= 32.0 else (dn - 32.0) * 0.04
    compat = max(0.0, min(1.0, 0.90 - dn_penalty - (1.0 - ox_factor) * 0.10))
    regen = max(0.0, min(1.0, (eei_diss ** 0.7) * (compat ** 0.3) * 0.98))
    return {
        "eei_dissolution_score":     round(eei_diss, 4),
        "electrode_compat_score":    round(compat, 4),
        "regeneration_potential_pct": round(regen * 100.0, 1),
    }


def composite(row: dict) -> float:
    return (0.50 * row["eei_dissolution_score"] +
            0.30 * row["electrode_compat_score"] +
            0.20 * row["regeneration_potential_pct"] / 100.0)


def tier_label(dn: float, diss: float, compat: float) -> str:
    if diss >= 0.85 and compat >= 0.88:
        return "Tier-1: DEER Prime"
    elif diss >= 0.70 and compat >= 0.80:
        return "Tier-2: DEER Candidate"
    elif diss >= 0.50 and compat >= 0.75:
        return "Tier-3: Co-solvent"
    else:
        return "Tier-4: Low Priority"


def score_smiles(smiles: str, name: str | None = None) -> SolventResult:
    # Look up known solvents from the library CSV first
    lib_path = DATA_DIR / "solvent_eei_properties.csv"
    if lib_path.exists():
        lib = pd.read_csv(lib_path)
        match = lib[lib["smiles"].str.strip() == smiles.strip()]
        if not match.empty:
            row = match.iloc[0]
            dn      = float(row.get("dn", 20.0))
            an      = float(row.get("an", 10.0))
            eps     = float(row.get("epsilon_r", 30.0))
            ox      = float(row.get("oxidation_stability_V", 4.2))
            rd      = float(row.get("reduction_stability_V", 0.8))
            vc      = float(row.get("viscosity_cp", 2.0))
            scores  = physics_score(dn, an, eps, ox, rd, vc)
            composite_score = composite(scores)
            return SolventResult(
                name                  = name or str(row.get("name", smiles)),
                smiles                = smiles,
                dn                    = dn,
                eei_dissolution_score = scores["eei_dissolution_score"],
                electrode_compat_score = scores["electrode_compat_score"],
                regeneration_potential_pct = scores["regeneration_potential_pct"],
                composite_score       = round(composite_score, 4),
                tier                  = tier_label(dn, scores["eei_dissolution_score"],
                                                   scores["electrode_compat_score"]),
            )

    # Fallback: RDKit descriptors or pure physics
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        donors = Descriptors.NumHDonors(mol)
        _acceptors = Descriptors.NumHAcceptors(mol)
        logp = Descriptors.MolLogP(mol)
        # Rough DN estimate from functional groups
        n_n = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
        n_o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
        dn_est = 14.0 + 0.5 * n_n + 0.3 * n_o + 2.0 * (tpsa / 100)
        dn_est = min(35.0, max(5.0, dn_est))
        eps_est = min(100.0, max(5.0, 10.0 + logp * 2 + donors * 3))
        scores = physics_score(dn_est, 10.0, eps_est)
    except Exception:
        scores = physics_score(20.0, 10.0, 30.0)

    cs = composite(scores)
    return SolventResult(
        name                  = name or smiles,
        smiles                = smiles,
        dn                    = 0.0,
        eei_dissolution_score = scores["eei_dissolution_score"],
        electrode_compat_score = scores["electrode_compat_score"],
        regeneration_potential_pct = scores["regeneration_potential_pct"],
        composite_score       = round(cs, 4),
        tier                  = None,
    )


# --------------------------------------------------------------------------- #
# FastAPI app (only built if fastapi is available)
# --------------------------------------------------------------------------- #

if HAS_FASTAPI:
    app = FastAPI(
        title="DEER Solvent Screening API",
        description="EEI dissolution solvent screening for Direct Electrode-to-Electrode "
                    "Regeneration of end-of-life Li-ion batteries",
        version="1.0.0",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "deer_solvent_screening"}

    @app.post("/solvent_screen", response_model=list[SolventResult])
    async def solvent_screen(payload: SolventListInput):
        return [score_smiles(s.smiles, s.name) for s in payload.solvents]

    @app.get("/solvent_top_k", response_model=list[SolventResult])
    async def solvent_top_k(k: int = 20):
        path = RESULTS_DIR / "solvent_eei_predictions.csv"
        if not path.exists():
            raise HTTPException(status_code=404,
                               detail="Run p40_solvent_screening.py first")
        df = pd.read_csv(path).sort_values("composite_score", ascending=False).head(k)
        return [
            SolventResult(
                name                  = str(r.get("name", "")),
                smiles                = str(r.get("smiles", "")),
                dn                    = float(r.get("dn", 0)),
                eei_dissolution_score = float(r["eei_dissolution_score"]),
                electrode_compat_score = float(r["electrode_compat_score"]),
                regeneration_potential_pct = float(r["regeneration_potential_pct"]),
                composite_score       = float(r["composite_score"]),
                tier                  = tier_label(
                    float(r.get("dn", 0)),
                    float(r["eei_dissolution_score"]),
                    float(r["electrode_compat_score"]),
                ),
            )
            for _, r in df.iterrows()
        ]

    @app.get("/solvent_pareto", response_model=list[SolventResult])
    async def solvent_pareto():
        path = DATA_DIR / "solvent_pareto_front.csv"
        if not path.exists():
            raise HTTPException(status_code=404,
                               detail="Run p40b_solvent_pareto.py first")
        df = pd.read_csv(path)
        return [
            SolventResult(
                name                  = str(r.get("name", "")),
                smiles                = str(r.get("smiles", "")),
                dn                    = float(r.get("dn", 0)),
                eei_dissolution_score = float(r["eei_dissolution_score"]),
                electrode_compat_score = float(r["electrode_compat_score"]),
                regeneration_potential_pct = float(r["regeneration_potential_pct"]),
                composite_score       = float(r.get("desirability_score", r.get("composite_score", 0))),
                tier                  = str(r.get("tier", "")),
            )
            for _, r in df.iterrows()
        ]

    @app.post("/solvent_compare")
    async def solvent_compare(payload: CompareInput):
        a = score_smiles(payload.smiles_a, payload.name_a)
        b = score_smiles(payload.smiles_b, payload.name_b)
        return {
            "solvent_a": a.model_dump(),
            "solvent_b": b.model_dump(),
            "winner_by_dissolution": payload.name_a if
                a.eei_dissolution_score > b.eei_dissolution_score else payload.name_b,
            "winner_by_compat": payload.name_a if
                a.electrode_compat_score > b.electrode_compat_score else payload.name_b,
            "winner_by_composite": payload.name_a if
                a.composite_score > b.composite_score else payload.name_b,
        }

else:
    # Stub so `from src.p40c_solvent_rest_api import app` doesn't crash
    app = None   # type: ignore


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #

def _cli() -> int:
    """Quick smoke test when run directly: score DMI, DMSO, EC."""
    for smi, name in [
        ("O=C1N(C)CCC1", "DMI"),
        ("CS(C)=O",       "DMSO"),
        ("O=C1OCCO1",     "EC"),
    ]:
        r = score_smiles(smi, name)
        print(f"{name:8s} ({smi}): DN={r.dn:.1f}  diss={r.eei_dissolution_score:.3f}  "
              f"compat={r.electrode_compat_score:.3f}  regen={r.regeneration_potential_pct:.1f}%  "
              f"tier={r.tier or 'unknown'}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    if HAS_FASTAPI and len(_sys.argv) == 1:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
    else:
        _sys.exit(_cli())
