"""Smoke test for the donor-number-screener pipeline.

Designed to run fast (< 1 minute) on any modern laptop.  Verifies
that all the headline artefacts exist and that the model
performance is in the expected range.

Run with:
  cd donor-number-screener
  PYTHONPATH=. python -m pytest tests/ -v
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

from utils import DATA_DIR, FIGURES_DIR, RESULTS_DIR

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- #
# Existence tests
# ---------------------------------------------------------------- #

def test_candidate_library_exists():
    f = DATA_DIR / "candidate_library.csv"
    assert f.exists(), f"missing {f}"
    df = pd.read_csv(f)
    assert len(df) >= 100, f"too few candidates: {len(df)}"
    assert "smiles" in df.columns


def test_anchor_table_exists():
    f = DATA_DIR / "dn_anchor_table.csv"
    assert f.exists(), f"missing {f}"
    df = pd.read_csv(f)
    assert "smiles" in df.columns and "dn_expt" in df.columns
    assert len(df) >= 50, f"too few anchors: {len(df)}"


def test_v1_descriptors_exist():
    f = DATA_DIR / "descriptors.csv"
    assert f.exists()
    df = pd.read_csv(f)
    assert df.shape[1] >= 200, f"v1 should have ~236 cols, got {df.shape[1]}"


def test_v2_descriptors_exist():
    f = DATA_DIR / "descriptors_v2.csv"
    assert f.exists(), (
        "v2 descriptors missing. Run: python src/02b_compute_descriptors_v2.py"
    )
    df = pd.read_csv(f)
    assert df.shape[1] >= 500, (
        f"v2 should have ~996 cols (Morgan+MACCS+EState added), got {df.shape[1]}"
    )


def test_top20_exists():
    candidates = [
        RESULTS_DIR / "top20_candidates_5model.csv",
        RESULTS_DIR / "top20_candidates_bayes.csv",
        RESULTS_DIR / "top20_candidates.csv",
    ]
    f = next((c for c in candidates if c.exists()), None)
    assert f is not None, "no top-20 file found in results/"
    df = pd.read_csv(f)
    assert len(df) == 20, f"expected 20 rows, got {len(df)}"


def test_dashboard_exists():
    f = PROJECT_ROOT / "dashboard.html"
    assert f.exists(), "missing dashboard.html"
    # Don't check exact size — just non-empty
    assert f.stat().st_size > 10_000


# ---------------------------------------------------------------- #
# Performance tests
# ---------------------------------------------------------------- #

def test_bayes_metrics_3model():
    f = RESULTS_DIR / "bayes_metrics.json"
    if not f.exists():
        pytest.skip("bayes_metrics.json not found (run 09 first)")
    m = json.loads(f.read_text())
    cv = m.get("cv_metrics", {}).get("stack", {}).get("R2", 0)
    assert cv >= 0.985, f"3-model stack CV R² too low: {cv}"


def test_5model_metrics():
    f = RESULTS_DIR / "bayes_metrics_5model.json"
    if not f.exists():
        pytest.skip("bayes_metrics_5model.json not found (run 09c first)")
    m = json.loads(f.read_text())
    cv = m.get("cv_metrics", {}).get("stack", {}).get("R2", 0)
    assert cv >= 0.985, f"5-model stack CV R² too low: {cv}"


def test_clean_meta_reasonable():
    f = RESULTS_DIR / "clean_meta.json"
    if not f.exists():
        pytest.skip("clean_meta.json not found (run 04a first)")
    m = json.loads(f.read_text())
    assert m["clean_n_rows"] >= 3000
    assert m["clean_n_cols"] >= 200


# ---------------------------------------------------------------- #
# Linting-style sanity checks on outputs
# ---------------------------------------------------------------- #

def test_top20_smiles_are_valid():
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    f = RESULTS_DIR / "top20_candidates_5model.csv"
    if not f.exists():
        f = RESULTS_DIR / "top20_candidates_bayes.csv"
    if not f.exists():
        f = RESULTS_DIR / "top20_candidates.csv"
    if not f.exists():
        pytest.skip("no top-20 file")

    df = pd.read_csv(f)
    n_valid = sum(1 for s in df["smiles"]
                  if Chem.MolFromSmiles(s) is not None)
    assert n_valid == len(df), f"{len(df) - n_valid} invalid SMILES in top-20"


def test_landing_page_exists():
    f = PROJECT_ROOT / "LANDING_PAGE.html"
    if not f.exists():
        pytest.skip("LANDING_PAGE.html not generated yet")
    html = f.read_text(encoding="utf-8")
    assert "donor-number-screener" in html
    assert "Pricing" in html or "RMB" in html


def test_top20_svg_exists():
    f = FIGURES_DIR / "top20_color_graded.svg"
    if not f.exists():
        pytest.skip("top20 SVG not generated yet")
    svg = f.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "DN=" in svg
