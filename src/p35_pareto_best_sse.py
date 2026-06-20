"""35_pareto_best_sse.py - Multi-objective Pareto front over the merged SSE dataset.

5 objectives (all "higher is better" after sign conversion):
  f1 = log10(sigma_ion)        (ionic conductivity)
  f2 = E_g                     (band gap / electronic insulator)
  f3 = stability_window        (electrochemical window)
  f4 = -migration_barrier      (kinetics)
  f5 = -cost_index             (cheap)

Algorithm: pure numpy non-dominated sort (O(N^2) but N < 1000 is trivial).
We also report:
  - top-3 per single objective
  - one "balanced" representative (closest to ideal in normalized space)
  - one representative per family (sulfide / oxide / halide / polymer / ...)

Output:
  data/pareto_front.csv         - non-dominated SSEs (sorted by f1 desc)
  data/pareto_summary.json      - per-objective top-3 + per-family + balanced
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import DATA_DIR, write_csv, write_json, set_seed  # noqa: E402


# --------------------------------------------------------------------------- #
# Pareto primitives
# --------------------------------------------------------------------------- #

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a dominates b (all >= and at least one >)."""
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(values: np.ndarray) -> np.ndarray:
    """O(N^2) non-dominated-sort. Returns a boolean mask of size N."""
    n = len(values)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        for j in range(n):
            if i == j or not nd[j]:
                continue
            if dominates(values[i], values[j]):
                nd[j] = False
    return nd


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def build_objective_matrix(rows: List[Dict[str, Any]],
                           objectives: List[str]) -> np.ndarray:
    """Build the (N, K) matrix of objective values, all "higher is better"."""
    n = len(rows)
    k = len(objectives)
    M = np.zeros((n, k), dtype=float)
    for i, r in enumerate(rows):
        for j, (key, sign) in enumerate(objectives):
            v = r.get(key, float("nan"))
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                v = np.nanmedian([r.get(key, 0.0) for r in rows if
                                  isinstance(r.get(key), (int, float))
                                  and math.isfinite(r.get(key))]) if any(
                    isinstance(r.get(key), (int, float)) and math.isfinite(r.get(key))
                    for r in rows) else 0.0
            M[i, j] = sign * float(v)
    # Replace NaN with column min
    for j in range(k):
        col = M[:, j]
        if np.isnan(col).any():
            col_min = np.nanmin(col)
            M[np.isnan(col), j] = col_min
    return M


def normalize(M: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; if constant column, all -> 0.5."""
    out = np.zeros_like(M)
    for j in range(M.shape[1]):
        col = M[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            out[:, j] = 0.5
        else:
            out[:, j] = (col - lo) / (hi - lo)
    return out


def representative_by_distance(M_norm: np.ndarray, mask: np.ndarray) -> int:
    """Return index (within mask) of the row closest to the ideal (1,1,...,1)."""
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return -1
    ideal = np.ones(M_norm.shape[1])
    d = np.linalg.norm(M_norm[idxs] - ideal, axis=1)
    return int(idxs[np.argmin(d)])


def representative_per_family(rows: List[Dict[str, Any]], M_norm: np.ndarray,
                              mask: np.ndarray) -> Dict[str, int]:
    """For each family on the Pareto front, pick the balanced rep."""
    idxs = np.where(mask)[0]
    out: Dict[str, int] = {}
    for i in idxs:
        fam = rows[i].get("family", "other")
        if fam not in out:
            out[fam] = i
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", default=str(DATA_DIR / "sse_datasets_combined.csv"))
    p.add_argument("--out_csv", default=str(DATA_DIR / "pareto_front.csv"))
    p.add_argument("--out_json", default=str(DATA_DIR / "pareto_summary.json"))
    args = p.parse_args()
    set_seed(0)

    # Read merged CSV
    in_path = Path(args.in_csv)
    if not in_path.exists():
        print(f"[pareto] missing {in_path}; run src/34_fetch_sse_datasets.py first")
        return 1
    import csv as _csv
    rows: List[Dict[str, Any]] = []
    with in_path.open() as f:
        for r in _csv.DictReader(f):
            row = dict(r)
            for k in ("sigma_ion_S_cm", "E_g_eV", "stability_window_V",
                      "migration_barrier_eV", "cost_index", "dn_pbp_v2"):
                try:
                    row[k] = float(row.get(k, "nan"))
                except Exception:
                    row[k] = float("nan")
            rows.append(row)
    if not rows:
        print("[pareto] empty input")
        return 1
    print(f"[pareto] {len(rows)} SSEs loaded")

    # 5 objectives: (column, sign), all converted so higher == better
    objectives = [
        ("sigma_ion_S_cm", 1.0),
        ("E_g_eV", 1.0),
        ("stability_window_V", 1.0),
        ("migration_barrier_eV", -1.0),
        ("cost_index", -1.0),
    ]
    M = build_objective_matrix(rows, objectives)
    # log10 on sigma
    M[:, 0] = np.log10(np.maximum(M[:, 0], 1e-12))
    M_norm = normalize(M)
    mask = non_dominated_sort(M)

    # Build Pareto output
    front_rows: List[Dict[str, Any]] = []
    idxs = np.where(mask)[0]
    for rank_i, i in enumerate(idxs, 1):
        r = dict(rows[i])
        r["pareto_rank"] = rank_i
        for j, (k, _) in enumerate(objectives):
            r[f"f{j+1}_{k}"] = float(M[i, j])
        front_rows.append(r)
    # Sort by f1 desc (log sigma)
    front_rows.sort(key=lambda x: x.get("f1_sigma_ion_S_cm", 0), reverse=True)
    write_csv(Path(args.out_csv), front_rows)
    print(f"[pareto] {len(front_rows)} non-dominated SSEs in {args.out_csv}")

    # Summary
    top3_per_obj = {}
    obj_names = ["sigma_ion", "E_g", "stability", "low_migration", "low_cost"]
    for j, name in enumerate(obj_names):
        order = np.argsort(-M[:, j])
        top3 = []
        for k_i in order[:3]:
            top3.append({
                "rank": k_i + 1,
                "formula": rows[k_i].get("formula", ""),
                "name": rows[k_i].get("name", ""),
                "family": rows[k_i].get("family", ""),
                "value": float(M[k_i, j]),
            })
        top3_per_obj[name] = top3

    balanced_idx = representative_by_distance(M_norm, mask)
    fam_idxs = representative_per_family(rows, M_norm, mask)

    summary = {
        "n_input": len(rows),
        "n_pareto": int(mask.sum()),
        "objectives": [n for n in obj_names],
        "top3_per_objective": top3_per_obj,
        "balanced_representative": {
            "formula": rows[balanced_idx].get("formula", ""),
            "name": rows[balanced_idx].get("name", ""),
            "family": rows[balanced_idx].get("family", ""),
            "dn_pbp_v2": rows[balanced_idx].get("dn_pbp_v2"),
        } if balanced_idx >= 0 else None,
        "representative_per_family": {
            fam: {"formula": rows[i].get("formula", ""),
                  "name": rows[i].get("name", "")}
            for fam, i in fam_idxs.items()
        },
        "pareto_formulas": [r.get("formula", "") for r in front_rows],
    }
    write_json(Path(args.out_json), summary)
    print("[pareto] top-3 per objective:")
    for name, lst in top3_per_obj.items():
        s = ", ".join(f"{x['formula']}({x['value']:.2f})" for x in lst)
        print(f"  {name:14s}: {s}")
    if balanced_idx >= 0:
        print(f"[pareto] balanced rep: {rows[balanced_idx].get('formula','')} "
              f"({rows[balanced_idx].get('family','')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
