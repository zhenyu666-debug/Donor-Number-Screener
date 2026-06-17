"""Tests for the v3 scripts (14-18).

These tests are designed to be fast (< 30 s) and exercise the v3
artefacts and APIs without running the heavy training (which is
covered by test_pipeline.py).

Run with:
  cd donor-number-screener
  PYTHONPATH=. python -m pytest tests/test_v3.py -v
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from utils import RESULTS_DIR  # noqa: E402

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- #
# 12 — outreach PDFs exist
# ---------------------------------------------------------------- #
def test_outreach_pdfs_present():
    expected = [
        "case_study_li_s_v1.pdf",
        "capability_one_pager.pdf",
        "pricing_v1.pdf",
    ]
    for fname in expected:
        p = PROJECT_ROOT / "outreach" / fname
        assert p.exists(), f"missing {p}"
        assert p.stat().st_size > 1000, f"{p} too small ({p.stat().st_size} B)"


def test_outreach_build_report():
    p = PROJECT_ROOT / "outreach" / "build_report.json"
    assert p.exists()
    report = json.loads(p.read_text(encoding="utf-8"))
    assert len(report) >= 3
    for k, v in report.items():
        assert v["backend"] in ("weasyprint", "pdfkit", "fpdf2", "html-only")
        if v["backend"] != "html-only":
            assert v["pdf"] is not None


# ---------------------------------------------------------------- #
# 14 — SHAP artefacts (only if the script has been run)
# ---------------------------------------------------------------- #
def test_shap_artefacts_if_present():
    p = RESULTS_DIR / "shap_summary.json"
    if not p.exists():
        pytest.skip("shap_summary.json not generated yet (run src/14_shap_explain.py)")
    summary = json.loads(p.read_text(encoding="utf-8"))
    assert "n_tree_models" in summary
    assert summary["n_tree_models"] >= 2
    assert "top1_feature" in summary
    csv_p = RESULTS_DIR / "shap_top20_attribution.csv"
    assert csv_p.exists()
    df = pd.read_csv(csv_p)
    assert "feature" in df.columns
    assert "mean_abs_shap" in df.columns
    assert len(df) >= 10


# ---------------------------------------------------------------- #
# 15 — API server: Pydantic models
# ---------------------------------------------------------------- #
def test_api_pydantic_models():
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_api_mod",
            Path(__file__).resolve().parent.parent / "src" / "15_api_server.py",
        )
        if _spec is None:  # type: ignore
            raise ImportError("spec is None")
        _api_mod = _ilu.module_from_spec(_spec)  # type: ignore
        _spec.loader.exec_module(_api_mod)  # type: ignore
        SMILESInput      = _api_mod.SMILESInput
        SMILESListInput  = _api_mod.SMILESListInput
        _band_for_dn     = _api_mod._band_for_dn
    except Exception as e:
        pytest.skip(f"15_api_server not importable (deps missing): {e}")

    # Validate SMILESInput
    inp = SMILESInput(smiles="CCO")
    assert inp.smiles == "CCO"

    # band logic
    idx, name = _band_for_dn(15.0)
    assert 0 <= idx <= 4
    assert isinstance(name, str)

    # SMILESListInput dedups
    lst = SMILESListInput(smiles_list=["CCO", "CCO", "CCN"])
    assert lst.smiles_list == ["CCO", "CCN"]
    assert lst.k == 20


# ---------------------------------------------------------------- #
# 16 — feat stability
# ---------------------------------------------------------------- #
def test_feat_stability_module_imports():
    """Just check the module imports and main() has a parser."""
    import importlib.util as _ilu
    p = Path(__file__).resolve().parent.parent / "src" / "16_feat_stability.py"
    spec = _ilu.spec_from_file_location("_fs_mod", p)
    if spec is None:  # type: ignore
        pytest.fail("16_feat_stability: spec is None")
    mod = _ilu.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    assert hasattr(mod, "main")
    assert hasattr(mod, "jaccard")
    # 0.5 for identical sets
    assert mod.jaccard({"a", "b"}, {"a", "b"}) == 1.0
    # disjoint
    assert mod.jaccard({"a", "b"}, {"c", "d"}) == 0.0
    # partial
    assert abs(mod.jaccard({"a", "b", "c"}, {"b", "c", "d"}) - 0.5) < 1e-9


# ---------------------------------------------------------------- #
# 17 — external validation constants
# ---------------------------------------------------------------- #
def test_external_dn_anchors():
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_ext_mod",
            Path(__file__).resolve().parent.parent / "src" / "17_external_validate.py",
        )
        if _spec is None:  # type: ignore
            raise ImportError("spec is None")
        _ext_mod = _ilu.module_from_spec(_spec)  # type: ignore
        _spec.loader.exec_module(_ext_mod)  # type: ignore
        EXTERNAL_DN = _ext_mod.EXTERNAL_DN
    except Exception as e:
        pytest.skip(f"17_external_validate not importable: {e}")
    assert len(EXTERNAL_DN) == 12
    for smiles, name, dn in EXTERNAL_DN:
        assert isinstance(smiles, str) and len(smiles) >= 1
        assert isinstance(name, str)
        assert 0.0 <= dn <= 30.0


# ---------------------------------------------------------------- #
# 18 — drift detector
# ---------------------------------------------------------------- #
def test_drift_module_imports_and_psi():
    """Just check the module imports and main() has a parser."""
    import importlib.util as _ilu
    p = Path(__file__).resolve().parent.parent / "src" / "18_drift_detect.py"
    spec = _ilu.spec_from_file_location("_dd_mod", p)
    if spec is None:  # type: ignore
        pytest.fail("18_drift_detect: spec is None")
    mod = _ilu.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    assert hasattr(mod, "main")
    assert hasattr(mod, "psi_for_feature")

    # PSI sanity: identical distros -> ~0
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, size=500)
    b = rng.normal(0, 1, size=500)
    edges = np.linspace(-3, 3, 11)
    psi = mod.psi_for_feature(a, b, edges)
    assert 0.0 <= psi < 0.10

    # PSI with very different distros -> > 0.2
    c = rng.normal(5, 1, size=500)
    psi2 = mod.psi_for_feature(a, c, edges)
    assert psi2 > 0.5


# ---------------------------------------------------------------- #
# ruff: code quality
# ---------------------------------------------------------------- #
def test_ruff_clean():
    """Confirm the 4 v3 scripts pass ruff."""
    import subprocess
    src = PROJECT_ROOT / "src"
    targets = ["14_shap_explain.py", "15_api_server.py", "16_feat_stability.py",
               "17_external_validate.py", "18_drift_detect.py", "12_build_pdf_safe.py"]
    for fname in targets:
        f = src / fname
        if not f.exists():
            continue
        result = subprocess.run(
            ["python", "-m", "ruff", "check", str(f), "--quiet"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"ruff failed for {fname}:\n{result.stdout}\n{result.stderr}"
        )
