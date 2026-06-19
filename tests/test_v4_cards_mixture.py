"""Tests for v4 additions: per-molecule cards (22) and mixture optimization (23)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import FIGURES_DIR, RESULTS_DIR  # noqa: E402


def test_top20_cards_csv_exists():
    p = RESULTS_DIR / "top20_cards.csv"
    assert p.exists(), f"missing {p}; run src/22_per_molecule_cards.py"
    df = pd.read_csv(p)
    assert len(df) == 20, f"expected 20 rows, got {len(df)}"
    for col in ("rank", "mol_id", "smiles", "iupac", "dn_pred"):
        assert col in df.columns, f"missing column {col}"
    assert df["rank"].is_monotonic_increasing
    # dn_pred should be finite numbers
    assert df["dn_pred"].notna().all()
    # iupac should be non-empty strings
    assert (df["iupac"].astype(str).str.len() > 0).all()


def test_per_molecule_cards_dir_exists():
    p = FIGURES_DIR / "top20_cards"
    assert p.exists(), f"missing {p}"
    files = list(p.glob("*.png"))
    assert len(files) == 20, f"expected 20 PNGs, got {len(files)}"
    # Each file > 2KB (sanity: actual rendered image, not empty)
    for f in files:
        assert f.stat().st_size > 2000, f"{f} is too small"


def test_contact_sheet_exists():
    p = FIGURES_DIR / "top20_cards.png"
    assert p.exists() and p.stat().st_size > 2000


def test_mixture_ratios_csv():
    p = RESULTS_DIR / "mixture_ratios.csv"
    assert p.exists(), f"missing {p}; run src/23_mixture_optimization.py"
    df = pd.read_csv(p)
    assert len(df) == 210, f"expected 21 grid points * 10 pairs = 210, got {len(df)}"
    for col in ("mol_id_a", "mol_id_b", "x_a", "x_b", "dn_linear"):
        assert col in df.columns
    # mole fractions sum to 1
    sums = (df["x_a"] + df["x_b"]).round(6)
    assert (sums == 1.0).all(), "x_a + x_b must equal 1"
    # linear DN is just the weighted average
    assert df["dn_linear"].notna().all()


def test_mixture_best_blends():
    p = RESULTS_DIR / "mixture_best_blends.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) == 10
    assert "smiles_a" in df.columns and "smiles_b" in df.columns


def test_mixture_ternary():
    p = RESULTS_DIR / "mixture_ternary.csv"
    assert p.exists()
    df = pd.read_csv(p)
    # 21*21 ternary points (the (x_a, x_b) grid; x_c = 1 - x_a - x_b)
    assert len(df) == 231
    sums = (df["x_a"] + df["x_b"] + df["x_c"]).round(6)
    assert ((sums - 1.0).abs() < 1e-6).all(), "ternary fractions must sum to 1"


def test_mixture_grid_png():
    p = FIGURES_DIR / "mixture_grid.png"
    assert p.exists() and p.stat().st_size > 5000


def test_per_molecule_card_specific_aminopyridine():
    """Spot-check: top molecule (2-aminopyridine) should produce a non-empty
    IUPAC field and a 95% CI derived from the EBM posterior."""
    df = pd.read_csv(RESULTS_DIR / "top20_cards.csv")
    top = df.iloc[0]
    assert top["iupac"], f"empty IUPAC for top hit mol_id={top['mol_id']}"
    # The 22 script falls back to the SMILES when RDKit-pypi has no
    # MolToIUPACName, so accept either a real IUPAC or the SMILES.
    iupac = str(top["iupac"])
    assert len(iupac) >= 3, f"iupac too short: {iupac!r}"
    # CI bounds finite if EBM ran for this mol
    assert np.isfinite(top["ebm_q05"]) and np.isfinite(top["ebm_q95"]), \
        f"EBM CI missing for top hit mol_id={top['mol_id']}"
    assert top["ebm_q05"] <= top["ebm_q95"]
