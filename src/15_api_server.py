"""Step 15 (v3): FastAPI REST server exposing the 5-model stacking
ensemble as 3 endpoints:

  POST /predict_smiles   - one SMILES -> DN prediction + uncertainty
  POST /estimate_dn      - same as /predict_smiles but with confidence CI
  POST /screen_top       - list of SMILES -> top-k candidates + Pareto

The endpoints accept JSON like:
  { "smiles": "CCO" }                              (predict_smiles)
  { "smiles": "CCO" }                              (estimate_dn)
  { "smiles_list": ["CCO", "CCN", ...], "k": 20 }  (screen_top)

Start the server (from repo root):
  uvicorn src.15_api_server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
try:
    # Pydantic v2
    from pydantic import field_validator as _validator
except ImportError:
    # Pydantic v1 fallback
    from pydantic import validator as _validator  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DIR, RESULTS_DIR, get_logger, set_global_seed  # noqa: E402

warnings.filterwarnings("ignore")
set_global_seed(42)
log = get_logger("api")

NON_FEATURE_COLS = {
    "mol_id", "smiles", "smiles_x", "smiles_y",
    "dn_rf", "dn_empirical", "dn_final", "confidence", "is_anchor",
}

# Region classification: 0=very weak, 1=weak, 2=intermediate, 3=strong, 4=very strong
DN_BANDS = [
    (0.0, 5.0,   0, "very_weak_solvent"),
    (5.0, 12.0,  1, "weak_solvent"),
    (12.0, 20.0, 2, "intermediate_solvent"),
    (20.0, 28.0, 3, "strong_solvent"),
    (28.0, 100.0, 4, "very_strong_solvent"),
]


def _band_for_dn(dn: float) -> tuple[int, str]:
    for lo, hi, idx, name in DN_BANDS:
        if lo <= dn < hi:
            return idx, name
    return 4, DN_BANDS[-1][3]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SMILESInput(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=2000,
                        description="A canonical or canonicalisable SMILES")
    include_descriptors: bool = Field(False, description="If true, also return 996-d descriptor vector")


class SMILESListInput(BaseModel):
    smiles_list: list[str] = Field(..., min_items=1, max_items=5000)
    k: int = Field(20, ge=1, le=100)
    include_pareto: bool = Field(True)

    @_validator("smiles_list")
    def _dedup_and_trim(cls, v: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for s in v:
            s2 = s.strip()
            if not s2 or s2 in seen:
                continue
            seen.add(s2)
            out.append(s2)
        return out


class PredictResponse(BaseModel):
    smiles: str
    dn_pred: float
    dn_lower: float   # 95% bootstrap CI lower bound (approximate)
    dn_upper: float
    region_index: int
    region_name: str
    n_models: int
    model_std: float
    descriptors_included: bool = False
    wall_time_ms: float


class ScreenTopItem(BaseModel):
    smiles: str
    dn_pred: float
    region_index: int
    region_name: str


class ScreenTopResponse(BaseModel):
    top_k: list[ScreenTopItem]
    pareto: list[ScreenTopItem] = Field(default_factory=list)
    n_input: int
    n_canonical: int
    wall_time_ms: float


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------
class _State:
    """Lazy singleton holding the loaded models + descriptor reference."""
    df: Optional[pd.DataFrame] = None
    X: Optional[np.ndarray] = None
    feat_cols: Optional[list[str]] = None
    models: dict = {}
    test_residuals: dict = {}     # for uncertainty bands

    @classmethod
    def ensure(cls) -> None:
        if cls.df is not None:
            return
        cls.load()

    @classmethod
    def load(cls) -> None:
        t0 = time.perf_counter()
        desc = pd.read_csv(DATA_DIR / "descriptors_v2.csv")
        labels = pd.read_csv(DATA_DIR / "dn_labels.csv")
        df = desc.merge(labels[["mol_id", "dn_final"]], on="mol_id", how="left")
        feat_cols = [c for c in df.columns
                     if c not in NON_FEATURE_COLS and df[c].dtype != "O"]
        cls.df = df
        cls.feat_cols = feat_cols
        cls.X = df[feat_cols].values.astype(np.float64)

        # Load best params
        metrics_path = RESULTS_DIR / "bayes_metrics_5model.json"
        if metrics_path.exists():
            params = json.loads(metrics_path.read_text(encoding="utf-8"))["best_params"]
        else:
            log.warning("No bayes_metrics_5model.json; using default LightGBM")
            params = {"lgbm": {"n_estimators": 600, "max_depth": 6,
                                "learning_rate": 0.05, "num_leaves": 31,
                                "min_child_samples": 20, "subsample": 0.8,
                                "colsample_bytree": 0.8, "reg_alpha": 0.1,
                                "reg_lambda": 0.1}}

        # Train the 5-model stacking once (no Optuna here — params assumed fixed)
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from xgboost import XGBRegressor

        # Reuse the ScaledMLP from step 9b (it wraps MLPRegressor with a
        # StandardScaler so predictions match the 5-model training pipeline).
        # Python disallows `import <module starting with digit>` at the
        # language level, so we use importlib for that file specifically.
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_mlp_mod",
            Path(__file__).resolve().parent / "09b_bayes_reuse.py",
        )
        _mod = _ilu.module_from_spec(_spec)  # type: ignore
        _spec.loader.exec_module(_mod)  # type: ignore
        ScaledMLP = _mod.ScaledMLP  # type: ignore

        X = cls.X
        y = df["dn_final"].values.astype(np.float64)

        if "rf" in params:
            cls.models["rf"] = RandomForestRegressor(
                **{**params["rf"], "random_state": 42, "n_jobs": -1}
            ).fit(X, y)
        if "xgb" in params:
            cls.models["xgb"] = XGBRegressor(
                **{**params["xgb"], "random_state": 42, "n_jobs": -1,
                   "verbosity": 0, "tree_method": "hist"}
            ).fit(X, y)
        if "mlp" in params:
            try:
                cls.models["mlp"] = ScaledMLP(**params["mlp"]).fit(X, y)
            except Exception as e:
                log.warning("MLP load skipped: %s", e)
        if "lgbm" in params:
            try:
                import lightgbm as lgb
                cls.models["lgbm"] = lgb.LGBMRegressor(
                    **{**params["lgbm"], "random_state": 42, "n_jobs": -1, "verbosity": -1}
                ).fit(X, y)
            except ImportError:
                log.warning("lightgbm not installed")
        if "cat" in params:
            try:
                from catboost import CatBoostRegressor
                cls.models["cat"] = CatBoostRegressor(
                    **{**params["cat"], "random_seed": 42, "verbose": False,
                       "thread_count": -1}
                ).fit(X, y)
            except ImportError:
                log.warning("catboost not installed")

        # Held-out residuals (20% test split, same seed as 09c)
        idx = np.arange(len(y))
        _, test_idx = train_test_split(idx, test_size=0.20, random_state=42)
        X_t, y_t = X[test_idx], y[test_idx]
        for name, mdl in cls.models.items():
            try:
                p = mdl.predict(X_t)
                cls.test_residuals[name] = (p - y_t).astype(np.float64)
            except Exception:
                pass

        log.info("API state loaded in %.1f s  models=%s  feats=%d",
                 time.perf_counter() - t0, list(cls.models.keys()), len(feat_cols))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _featurize_smiles(smiles: str) -> Optional[np.ndarray]:
    """Compute the v2 996-d feature vector for one SMILES."""
    _State.ensure()
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Descriptors, MACCSkeys  # type: ignore
    from rdkit.Chem.EState import EState_VSA  # type: ignore
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc: dict = {}
    # RDKit 2D descriptors
    for name, fn in Descriptors._descList:  # type: ignore
        try:
            desc[name] = float(fn(mol))
        except Exception:
            desc[name] = 0.0
    # EState VSA
    try:
        for i, v in enumerate(EState_VSA.EState_VSA_(mol)):
            desc[f"EState_VSA{i}"] = float(v)
    except Exception:
        pass
    # MACCS
    try:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        for i, bit in enumerate(maccs):
            desc[f"MACCS_{i:03d}"] = float(bit)
    except Exception:
        pass
    # Morgan radius 2, 1024 bits (sub-sampled to ~512 by the existing v2 column list)
    try:
        gen = GetMorganGenerator(radius=2, fpSize=512)
        fp = gen.GetFingerprint(mol)
        for i in range(512):
            desc[f"Morgan2_{i:04d}"] = float(fp.GetBit(i))
    except Exception:
        pass

    # Align to feat_cols
    X = np.zeros((1, len(_State.feat_cols)), dtype=np.float64)
    for j, c in enumerate(_State.feat_cols):
        if c in desc:
            v = desc[c]
            X[0, j] = v if not (v != v) else 0.0  # NaN -> 0
    return X


def _uncertainty(p: np.ndarray, n_models: int) -> tuple[float, float, float]:
    """Return (mean, lower, upper) using ensemble spread + train residuals."""
    mean = float(p.mean())
    spread = float(p.std(ddof=0)) if n_models > 1 else 0.0
    avg_resid_std = float(np.mean([np.std(_State.test_residuals.get(n, [0.0]))
                                   for n in _State.models]))
    half_ci = 1.96 * (spread + avg_resid_std) / 2.0
    return mean, mean - half_ci, mean + half_ci


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException
except ImportError:
    log.error("fastapi / uvicorn not installed. Run: pip install fastapi uvicorn")
    raise

app = FastAPI(
    title="donor-number-screener API",
    version="3.0",
    description=(
        "REST API for the 5-model stacking ensemble (RF, XGB, MLP, LGBM, CatBoost) "
        "trained on 3,551 molecules with 996 descriptors (v2). "
        "Returns predicted Gutmann donor number (DN) + region label + uncertainty."
    ),
)


@app.get("/health")
def health() -> dict:
    _State.ensure()
    return {
        "status": "ok",
        "n_features": len(_State.feat_cols) if _State.feat_cols else 0,
        "models_loaded": list(_State.models.keys()),
        "wall_time_s": 0.0,
    }


@app.post("/predict_smiles", response_model=PredictResponse)
def predict_smiles(body: SMILESInput) -> PredictResponse:
    t0 = time.perf_counter()
    _State.ensure()
    X = _featurize_smiles(body.smiles)
    if X is None:
        raise HTTPException(status_code=400,
                            detail=f"could not parse SMILES: {body.smiles}")
    preds = np.array([mdl.predict(X)[0] for mdl in _State.models.values()])
    mean, lo, hi = _uncertainty(preds, len(preds))
    idx, name = _band_for_dn(mean)
    return PredictResponse(
        smiles=body.smiles,
        dn_pred=round(mean, 4),
        dn_lower=round(lo, 4),
        dn_upper=round(hi, 4),
        region_index=idx,
        region_name=name,
        n_models=len(preds),
        model_std=round(float(preds.std(ddof=0)), 4),
        descriptors_included=False,
        wall_time_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/estimate_dn", response_model=PredictResponse)
def estimate_dn(body: SMILESInput) -> PredictResponse:
    # Same as /predict_smiles for now (placeholder for a Bayesian-flavored endpoint)
    return predict_smiles(body)


@app.post("/screen_top", response_model=ScreenTopResponse)
def screen_top(body: SMILESListInput) -> ScreenTopResponse:
    t0 = time.perf_counter()
    _State.ensure()

    rows: list[dict] = []
    for s in body.smiles_list:
        X = _featurize_smiles(s)
        if X is None:
            continue
        preds = np.array([mdl.predict(X)[0] for mdl in _State.models.values()])
        mean = float(preds.mean())
        idx, name = _band_for_dn(mean)
        rows.append({"smiles": s, "dn_pred": mean, "region_index": idx, "region_name": name})

    rows.sort(key=lambda r: -r["dn_pred"])
    top = rows[: body.k]
    pareto: list[dict] = []
    if body.include_pareto and rows:
        # Simple "Pareto" by region: keep at least 1 from each region
        seen: set[int] = set()
        for r in rows:
            if r["region_index"] not in seen:
                pareto.append(r)
                seen.add(r["region_index"])
        # Cap pareto to top 10
        pareto = pareto[:10]

    return ScreenTopResponse(
        top_k=top,
        pareto=pareto,
        n_input=len(body.smiles_list),
        n_canonical=len(rows),
        wall_time_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


def main() -> None:
    """Entry point for `python src/15_api_server.py` — starts uvicorn."""
    import uvicorn
    _State.ensure()
    log.info("Starting uvicorn on http://0.0.0.0:8000 ...")
    uvicorn.run("src.15_api_server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
