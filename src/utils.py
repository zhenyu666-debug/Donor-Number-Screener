"""Utility helpers for the Li-S additive screening reproduction.

Common functions: project paths, fingerprint seeding, log setup,
descriptor sanity checks, and deterministic RNG.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

for d in (DATA_DIR, FIGURES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and (when available) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
    except Exception:
        pass


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def normalize_smiles(smiles: str) -> str:
    """Canonical SMILES using RDKit (returns original if RDKit fails)."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


def report_dtypes(df, cols=None) -> None:
    """Print dtypes for quick debugging."""
    if cols is None:
        cols = df.columns.tolist()
    print(df[cols].dtypes)
