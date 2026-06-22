"""37_candidate_score.py - Score every SSE candidate in sse_library.yaml with
the candidate v2 formula (5-component weighted DN), and rank.

Reads (in order of preference):
  data/sse_library_enriched.yaml - if present (from p37_enrich_candidates)
  data/sse_library.yaml          - canonical 14-SSE library fallback

Writes:
  data/candidate_scores.csv  - 14 rows + 1 header (see COLUMNS below)
  data/candidate_scores.json - same data, JSON-encoded

Algorithm
---------
The v2 formula mirrors p32_sse_redn.rerank() and p34's compute_dn_pbp_v2:
    dn = w_langevin * em
       + w_particle * (anchor + 0.5 * (4 - coord) - 0.5 * (migration - 0.3) * 2)
       + w_sei     * em * max(0, 1 - 0.5 * migration)
       + w_aimd    * (8 + 5 * sqrt(sigma))
       + w_empirical * em_clamped
where em = clamp(1e3 * log10(sigma) + Eg + 5 - 2 * migration, 5, 40)
and em_clamped = em.

For fields missing from the input library we use the class-based default
table from p34 (sulfide/oxide/polymer/halide/...) so the formula still
produces a finite DN for any candidate.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import DATA_DIR, write_csv, write_json, set_seed  # noqa: E402


SRC_LIB = DATA_DIR / "sse_library.yaml"
SRC_ENRICHED = DATA_DIR / "sse_library_enriched.yaml"
OUT_CSV = DATA_DIR / "candidate_scores.csv"
OUT_JSON = DATA_DIR / "candidate_scores.json"

# Fixed weights matching p32_sse_redn.py and p34_fetch_sse_datasets.py.
WEIGHTS = {"langevin": 0.5, "particle": 0.1, "sei": 0.1,
           "aimd": 0.2, "empirical": 0.1}
ANCHOR_DN = 22.0

COLUMNS = [
    "id", "name", "formula", "class",
    "sigma_ion_S_cm", "E_g_eV", "migration_eV",
    "E_form_eV", "stability_window_V", "density_g_cm3", "molar_mass_g_mol",
    "li_coord_num", "dn_candidate", "rank",
    "enrichment_source", "notes",
]

# Class-based defaults for fields that may be missing after enrichment.
# Same values as p34.fill_missing_with_heuristic.
CLASS_DEFAULTS = {
    "sulfide":     {"E_g_eV": 2.5, "stability_window_V": 5.0,
                    "migration_eV": 0.30, "li_coord_num": 4.0,
                    "E_form_eV": -0.45, "density_g_cm3": 1.95},
    "oxide":       {"E_g_eV": 5.0, "stability_window_V": 5.5,
                    "migration_eV": 0.40, "li_coord_num": 4.5,
                    "E_form_eV": -3.00, "density_g_cm3": 3.5},
    "polymer":     {"E_g_eV": 4.0, "stability_window_V": 4.0,
                    "migration_eV": 0.50, "li_coord_num": 3.5,
                    "E_form_eV": -0.10, "density_g_cm3": 1.20},
    "halide":      {"E_g_eV": 5.0, "stability_window_V": 4.5,
                    "migration_eV": 0.35, "li_coord_num": 4.0,
                    "E_form_eV": -2.00, "density_g_cm3": 2.50},
    "other":       {"E_g_eV": 4.0, "stability_window_V": 4.5,
                    "migration_eV": 0.40, "li_coord_num": 4.0,
                    "E_form_eV": -0.50, "density_g_cm3": 2.00},
}


# --------------------------------------------------------------------------- #
# Field-level helpers
# --------------------------------------------------------------------------- #

def is_missing_field(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float):
        return not math.isfinite(v)
    return False


def fill_class_defaults(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Apply class-based defaults for any field still missing. Returns a copy."""
    out = dict(entry)
    cls = (out.get("class") or "other").lower()
    if cls not in CLASS_DEFAULTS:
        cls = "other"
    for k, v in CLASS_DEFAULTS[cls].items():
        if is_missing_field(out.get(k)):
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def compute_dn_candidate_v2(entry: Dict[str, Any],
                            weights: Optional[Dict[str, float]] = None,
                            anchor_dn: float = ANCHOR_DN) -> Dict[str, float]:
    """Return a dict of (dn_candidate, dn_langevin, dn_particle, dn_sei,
    dn_aimd, dn_empirical) for a single SSE entry.

    All inputs are assumed filled by fill_class_defaults().
    """
    w = weights or WEIGHTS
    sigma = max(float(entry.get("sigma_ion_S_cm", 1e-6)), 1e-12)
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = 1e-9
    Eg = float(entry.get("E_g_eV", 4.0))
    if not math.isfinite(Eg):
        Eg = 4.0
    migration = float(entry.get("migration_eV", 0.4))
    if not math.isfinite(migration):
        migration = 0.4
    coord = float(entry.get("li_coord_num", 4.0))
    if not math.isfinite(coord):
        coord = 4.0

    # Empirical (raw + clamped)
    raw_em = 1.0e3 * math.log10(sigma) + 1.0 * Eg + 5.0 - 2.0 * migration
    em = max(5.0, min(40.0, raw_em))

    # Components
    lang_dn = em
    p_corr = 0.5 * (4.0 - coord) - 0.5 * (migration - 0.3) * 2.0
    s = max(0.0, 1.0 - 0.5 * migration)
    a = 8.0 + 5.0 * math.sqrt(sigma)

    dn = (w["langevin"] * lang_dn
          + w["particle"] * (anchor_dn + p_corr)
          + w["sei"] * em * s
          + w["aimd"] * a
          + w["empirical"] * em)

    return {
        "dn_candidate": float(dn),
        "dn_langevin": float(lang_dn),
        "dn_particle": float(anchor_dn + p_corr),
        "dn_sei": float(em * s),
        "dn_aimd": float(a),
        "dn_empirical": float(em),
    }


def score_library(lib: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in lib:
        filled = fill_class_defaults(entry)
        comps = compute_dn_candidate_v2(filled)
        # Tiebreaker: existing dn_empirical field, if any, then id.
        notes = []
        if "enrichment" in entry:
            src = entry["enrichment"].get("sources", {}) or {}
            n_filled = sum(
                1 for k, v in entry["enrichment"].items()
                if isinstance(v, dict) and v.get("found")
            )
            notes.append(f"enrichment sources queried={list(src.keys())}"
                         f" found={n_filled}")
        rows.append({
            "id": entry.get("id"),
            "name": entry.get("name"),
            "formula": entry.get("formula"),
            "class": entry.get("class"),
            "sigma_ion_S_cm": filled.get("sigma_ion_S_cm"),
            "E_g_eV": filled.get("E_g_eV"),
            "migration_eV": filled.get("migration_eV"),
            "E_form_eV": filled.get("E_form_eV"),
            "stability_window_V": filled.get("stability_window_V"),
            "density_g_cm3": filled.get("density_g_cm3"),
            "molar_mass_g_mol": filled.get("molar_mass_g_mol"),
            "li_coord_num": filled.get("li_coord_num"),
            "dn_candidate": comps["dn_candidate"],
            "enrichment_source": ("enriched" if "enrichment" in entry
                                  else "library"),
            "notes": "; ".join(notes),
        })
    rows.sort(key=lambda r: (r["dn_candidate"], -float(r.get("E_form_eV") or 0)),
              reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def load_source_library() -> List[Dict[str, Any]]:
    if SRC_ENRICHED.exists():
        with SRC_ENRICHED.open(encoding="utf-8-sig") as f:
            d = yaml.safe_load(f) or {}
        return d.get("sse", [])
    if SRC_LIB.exists():
        with SRC_LIB.open(encoding="utf-8-sig") as f:
            d = yaml.safe_load(f) or {}
        return d.get("sse", [])
    return []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv", default=str(OUT_CSV))
    p.add_argument("--out_json", default=str(OUT_JSON))
    args = p.parse_args()
    set_seed(0)
    t0 = time.time()
    lib = load_source_library()
    if not lib:
        print("[score] no SSE library found; nothing to score.")
        return 1
    rows = score_library(lib)
    write_csv(Path(args.out_csv), rows, fieldnames=COLUMNS)
    payload = {
        "n_candidates": len(rows),
        "weights": WEIGHTS,
        "anchor_dn": ANCHOR_DN,
        "elapsed_s": round(time.time() - t0, 3),
        "top3": [{"rank": r["rank"], "name": r["name"],
                  "dn_candidate": round(r["dn_candidate"], 3)}
                 for r in rows[:3]],
        "rows": rows,
    }
    write_json(Path(args.out_json), payload)
    print(f"[score] {len(rows)} candidates -> {args.out_csv}")
    for r in rows[:3]:
        print(f"  #{r['rank']} {r['name'][:30]:30s}"
              f" dn_candidate={r['dn_candidate']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
