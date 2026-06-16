"""Step 9d: Optuna study caching for fast re-runs.

Wraps any Optuna study in a SQLite-backed RDB storage. On first
run it executes the search; on subsequent runs it loads the cached
best trial instantly.

This is a drop-in addition to `09_bayesian_optimization.py` and
`09c_5model_stacking.py`.  Use:

    from optuna_utils import cached_study

    study = cached_study("rf_v2", n_trials=60, seed=42, direction="maximize")
    study.optimize(objective, n_trials=n_trials)   # fast after first run

The SQLite DB lives at `results/optuna_cache.db`.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("pip install optuna")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DB = PROJECT_ROOT / "results" / "optuna_cache.db"


def cached_study(name: str, *, direction: str = "maximize",
                 seed: int = 42) -> optuna.Study:
    """Get (or load) an Optuna study backed by SQLite.

    The storage URL format is `sqlite:///abs/path/to/db`.
    """
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{CACHE_DB}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(
        study_name=name, storage=storage, direction=direction,
        sampler=sampler, load_if_exists=True,
    )


if __name__ == "__main__":
    # Smoke test
    s = cached_study("smoke_test")
    print(f"Cache DB: {CACHE_DB}")
    print(f"Loaded {len(s.trials)} prior trials for 'smoke_test'")
