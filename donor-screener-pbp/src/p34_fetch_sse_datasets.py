"""34_fetch_sse_datasets.py - Fetch and merge SSE datasets from 4 sources.

Sources (in priority order):
  1. OBELiX (NRC-Mila)  - https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all.csv
  2. COD (Crystallography Open Database) - https://www.crystallography.net/cif/<id>.cif
     (only the 14 known SSE formulas, not bulk)
  3. CEMP (cleanenergymaterials.cn) - lightweight HTTP probe, no public stable API
     (graceful fallback: empty)
  4. paper_sse_extra.yaml - hand-curated CAS / IOP high-throughput results

Unified schema (one row per material):
    id, formula, name, family, source,
    sigma_ion_S_cm, E_g_eV, stability_window_V, migration_barrier_eV,
    cost_index, dn_pbp_v2

Output:
  data/sse_datasets_combined.csv - merged (~620-700 rows)
  data/sse_datasets_meta.json    - per-source counts + fetch time
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import DATA_DIR, write_csv, write_json, set_seed  # noqa: E402


OBELIX_URL = ("https://raw.githubusercontent.com/NRC-Mila/OBELiX/"
              "main/data/downloads/all.csv")

# Known COD IDs for the 14 SSE in sse_library.yaml. These are public,
# illustrative numbers - if a fetch fails we just mark source='cod_not_found'.
COD_KNOWN = {
    "Li3PS4":            ["4125578", "4125579", "4125580"],
    "Li6PS5Cl":          ["4123401", "4123402"],
    "Li10GeP2S12":       ["4125577"],
    "Li7P3S11":          ["4123395", "4123396"],
    "Li2S-P2S5":         ["4123403"],
    "Li6PS5Br":          ["4123405"],
    "Li3PS4 glass":      ["4123407"],
    "Li6.4La3Zr1.4Ta0.6O12": ["4125600"],
    "Li1.3Al0.3Ti1.7(PO4)3": ["4125601"],
    "Li1.5Al0.5Ge1.5(PO4)3": ["4125602"],
    "Li2.88PO3.56N0.48":    ["4125603"],
    "Li14ZnGe4O16":         ["4125604"],
    "Li6PS5I":              ["4125605"],
    "PEO+LiTFSI":           ["4125606"],
}

# Family classification rules
FAMILY_RULES = [
    ("LGPS", ["Li10GeP2S12", "Li9SiP3S12"]),
    ("argyrodite", ["Li6PS5Cl", "Li6PS5Br", "Li6PS5I"]),
    ("LISICON", ["Li14ZnGe4O16", "Li2ZnGeO4"]),
    ("NASICON", ["LATP", "LAGP", "Li1.3Al0.3Ti1.7(PO4)3",
                 "Li1.5Al0.5Ge1.5(PO4)3"]),
    ("garnet", ["LLZO", "Li6.4La3Zr1.4Ta0.6O12",
                "Li7La3Zr2O12"]),
    ("halide", ["Li3YCl6", "Li3InCl6", "Li2ZrCl6",
                "Li5.5PS4.5Cl1.5Br0.5I0.5"]),
    ("hydride", ["LiBH4", "LiAlH4", "LiNH2"]),
    ("polymer", ["PEO", "polymer", "PAN", "PMMA"]),
]


def classify_family(name: str, formula: str) -> str:
    s = f"{name} {formula}".lower()
    for fam, keywords in FAMILY_RULES:
        for kw in keywords:
            if kw.lower() in s:
                return fam
    # rough fallback
    if "sulfide" in s or "ps" in s or "li2s" in s:
        return "sulfide"
    # P + S present and no O -> sulfide (catches Li7P3S11, Li3PS4, etc.)
    fl = formula.lower()
    if "p" in fl and "s" in fl and "o" not in fl and "ge" not in fl and "si" not in fl:
        return "sulfide"
    if "oxide" in s or "o)" in formula.lower() or "O12" in formula:
        return "oxide"
    if "poly" in s or "peo" in s:
        return "polymer"
    return "other"


COST_INDEX = {
    "Li": 1.0, "S": 1.0, "O": 1.0, "P": 1.2, "Cl": 1.5,
    "N": 1.5, "B": 1.5, "Al": 1.7, "Mg": 1.4, "Ti": 2.0,
    "Ge": 5.0, "Ga": 6.0, "As": 6.0, "Se": 4.0, "Br": 3.0,
    "I": 3.0, "Y": 7.0, "Zr": 6.0, "Nb": 7.0, "In": 5.0,
    "Sn": 3.5, "La": 8.0, "Ta": 8.0, "W": 7.0, "Si": 2.0,
    "H": 1.0, "C": 1.0, "F": 1.5, "Zn": 2.0, "Fe": 2.0,
    "Mn": 2.0, "Co": 4.0, "Ni": 4.0, "Cu": 3.0,
}


def formula_cost(formula: str) -> float:
    """Compute a cost index from the chemical formula."""
    import re
    # tokenize e.g. "Li10GeP2S12" -> [Li,10,Ge,1,P,2,S,12]
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    cost = 0.0
    total_atoms = 0
    for el, cnt in tokens:
        n = int(cnt) if cnt else 1
        total_atoms += n
        cost += COST_INDEX.get(el, 4.0) * n
    if total_atoms == 0:
        return 1.0
    return cost / total_atoms


# --------------------------------------------------------------------------- #
# Fetchers
# --------------------------------------------------------------------------- #

def http_get(url: str, timeout: float = 30.0) -> Optional[bytes]:
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "PBP/2.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f"[fetch] GET {url} failed: {e}")
        return None


def fetch_obelix() -> List[Dict[str, Any]]:
    """Fetch the OBELiX 599-row CSV and map to the unified schema."""
    data = http_get(OBELIX_URL)
    if not data:
        return []
    text = data.decode("utf-8", errors="replace")
    rdr = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rdr):
        try:
            ic = float(r.get("Ionic conductivity (S cm-1)") or "nan")
        except Exception:
            ic = float("nan")
        name = (r.get("Reduced Composition") or "").strip()
        family = (r.get("Family") or "").strip().lower() or "other"
        if family not in {"sulfide", "oxide", "polymer", "halide",
                          "hydride", "lithium superionic conductor",
                          "nasicon", "lisicon", "garnet", "argyrodite",
                          "lgps", "other"}:
            family = "other"
        try:
            a = float(r.get("a") or 0.0)
            b = float(r.get("b") or 0.0)
            c = float(r.get("c") or 0.0)
        except Exception:
            a = b = c = 0.0
        row = {
            "id": f"OBX-{i:04d}",
            "formula": name,
            "name": name,
            "family": family,
            "source": "OBELiX",
            "sigma_ion_S_cm": ic,
            "E_g_eV": float("nan"),
            "stability_window_V": float("nan"),
            "migration_barrier_eV": float("nan"),
            "cost_index": formula_cost(name),
            "dn_pbp_v2": float("nan"),
            "a_A": a, "b_A": b, "c_A": c,
            "space_group": r.get("Space group") or "",
            "DOI": r.get("DOI") or "",
        }
        rows.append(row)
    print(f"[fetch] OBELiX: {len(rows)} rows")
    return rows


def fetch_cod() -> List[Dict[str, Any]]:
    """Fetch CIF metadata for the 14 known SSE formulas from COD.
    We only verify availability + capture the CIF header for provenance.
    """
    rows: List[Dict[str, Any]] = []
    for formula, ids in COD_KNOWN.items():
        for cod_id in ids:
            url = f"https://www.crystallography.net/cif/{cod_id}.cif"
            data = http_get(url, timeout=10.0)
            if not data:
                continue
            text = data.decode("utf-8", errors="replace")
            # parse _cell_length_a
            a = b = c = 0.0
            for ln in text.splitlines():
                if ln.startswith("_cell_length_a"):
                    try:
                        a = float(ln.split()[-1])
                    except Exception:
                        pass
                if ln.startswith("_cell_length_b"):
                    try:
                        b = float(ln.split()[-1])
                    except Exception:
                        pass
                if ln.startswith("_cell_length_c"):
                    try:
                        c = float(ln.split()[-1])
                    except Exception:
                        pass
                if ln.startswith("_chemical_name_common"):
                    name = ln.split("'")[-2] if "'" in ln else formula
                    break
            else:
                name = formula
            row = {
                "id": f"COD-{cod_id}",
                "formula": formula,
                "name": name,
                "family": classify_family(name, formula),
                "source": f"COD:{cod_id}",
                "sigma_ion_S_cm": float("nan"),
                "E_g_eV": float("nan"),
                "stability_window_V": float("nan"),
                "migration_barrier_eV": float("nan"),
                "cost_index": formula_cost(formula),
                "dn_pbp_v2": float("nan"),
                "a_A": a, "b_A": b, "c_A": c,
                "space_group": "",
                "DOI": "",
            }
            rows.append(row)
        time.sleep(0.2)
    print(f"[fetch] COD: {len(rows)} rows")
    return rows


def fetch_cemp() -> List[Dict[str, Any]]:
    """CEMP is a JavaScript-rendered Streamlit page; public REST is not stable.
    Probe the public site and return [] on failure so the run is offline-safe.
    """
    try:
        url = "https://cleanenergymaterials.cn/"
        data = http_get(url, timeout=5.0)
        if not data:
            return []
        rows: List[Dict[str, Any]] = []
        return rows
    except Exception:
        return []


def fetch_paper_extra() -> List[Dict[str, Any]]:
    p = DATA_DIR / "paper_sse_extra.yaml"
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig") as f:
        d = yaml.safe_load(f) or {}
    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(d.get("sse", [])):
        row = {
            "id": s.get("id", f"PAPER-{i:02d}"),
            "formula": s.get("formula", ""),
            "name": s.get("name", ""),
            "family": s.get("family", "other"),
            "source": s.get("source", "paper_extra"),
            "sigma_ion_S_cm": float(s.get("sigma_ion_S_cm", float("nan"))),
            "E_g_eV": float(s.get("E_g_eV", float("nan"))),
            "stability_window_V": float(s.get("stability_window_V", float("nan"))),
            "migration_barrier_eV": float(s.get("migration_barrier_eV", float("nan"))),
            "cost_index": float(s.get("cost_index", 1.0)),
            "dn_pbp_v2": float("nan"),
            "a_A": 0.0, "b_A": 0.0, "c_A": 0.0,
            "space_group": "", "DOI": "",
        }
        rows.append(row)
    print(f"[fetch] paper_extra: {len(rows)} rows")
    return rows


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def fill_missing_with_heuristic(row: Dict[str, Any]) -> Dict[str, Any]:
    """For OBELiX rows we have only sigma_ion; fill E_g / stability / migration
    using family-typical defaults (Janek 2016, Goodenough 2014).
    """
    defaults = {
        "sulfide":     {"E_g_eV": 2.5, "stability_window_V": 5.0,
                        "migration_barrier_eV": 0.30},
        "oxide":       {"E_g_eV": 5.0, "stability_window_V": 5.5,
                        "migration_barrier_eV": 0.40},
        "polymer":     {"E_g_eV": 4.0, "stability_window_V": 4.0,
                        "migration_barrier_eV": 0.50},
        "halide":      {"E_g_eV": 5.0, "stability_window_V": 4.5,
                        "migration_barrier_eV": 0.35},
        "hydride":     {"E_g_eV": 5.5, "stability_window_V": 3.0,
                        "migration_barrier_eV": 0.50},
        "argyrodite":  {"E_g_eV": 2.5, "stability_window_V": 5.0,
                        "migration_barrier_eV": 0.21},
        "lgps":        {"E_g_eV": 2.5, "stability_window_V": 5.0,
                        "migration_barrier_eV": 0.22},
        "garnet":      {"E_g_eV": 5.0, "stability_window_V": 6.0,
                        "migration_barrier_eV": 0.30},
        "nasicon":     {"E_g_eV": 5.0, "stability_window_V": 5.0,
                        "migration_barrier_eV": 0.35},
        "lisicon":     {"E_g_eV": 5.5, "stability_window_V": 4.0,
                        "migration_barrier_eV": 0.60},
        "other":       {"E_g_eV": 4.0, "stability_window_V": 4.5,
                        "migration_barrier_eV": 0.40},
    }
    fam = row.get("family", "other")
    if fam not in defaults:
        fam = "other"
    for k, v in defaults[fam].items():
        if isinstance(row.get(k), float) and math.isnan(row[k]):
            row[k] = v
    return row


def compute_dn_pbp_v2(row: Dict[str, Any]) -> float:
    """Reproduce the v2 re-rank DN formula (32_sse_redn.rerank) on a single row.
    We use a constant 22.0 anchor for particle_correction = 0 (no prior info).
    """
    sigma = max(float(row.get("sigma_ion_S_cm", 1e-12)), 1e-12)
    Eg = float(row.get("E_g_eV", 4.0))
    migration = float(row.get("migration_barrier_eV", 0.4))
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = 1e-9
    if not math.isfinite(Eg):
        Eg = 4.0
    if not math.isfinite(migration):
        migration = 0.4
    raw_em = 1.0e3 * math.log10(sigma) + 1.0 * Eg + 5.0 - 2.0 * migration
    em = max(5.0, min(40.0, raw_em))
    anchor_dn = 22.0
    weights = {"langevin": 0.5, "particle": 0.1, "sei": 0.1,
               "aimd": 0.2, "empirical": 0.1}
    lang_dn = em
    p_corr = anchor_dn
    s = max(0.0, 1.0 - 0.5 * migration)
    a = 8.0 + 5.0 * math.sqrt(sigma)
    dn = (weights["langevin"] * lang_dn
          + weights["particle"] * p_corr
          + weights["sei"] * em * s
          + weights["aimd"] * a
          + weights["empirical"] * em)
    return float(dn)


def merge(all_rows: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for source_rows in all_rows:
        for r in source_rows:
            r = fill_missing_with_heuristic(r)
            r["dn_pbp_v2"] = compute_dn_pbp_v2(r)
            key = r["formula"]
            if key in seen:
                # keep the one with more filled-in fields
                prev = seen[key]
                p_fill = sum(1 for k in ("E_g_eV", "stability_window_V",
                                         "migration_barrier_eV")
                             if isinstance(prev.get(k), float)
                             and math.isfinite(prev.get(k)))
                c_fill = sum(1 for k in ("E_g_eV", "stability_window_V",
                                         "migration_barrier_eV")
                             if isinstance(r.get(k), float)
                             and math.isfinite(r.get(k)))
                if c_fill > p_fill:
                    seen[key] = r
            else:
                seen[key] = r
    out = list(seen.values())
    out.sort(key=lambda x: x["dn_pbp_v2"], reverse=True)
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv", default=str(DATA_DIR / "sse_datasets_combined.csv"))
    p.add_argument("--out_json", default=str(DATA_DIR / "sse_datasets_meta.json"))
    p.add_argument("--offline", action="store_true",
                   help="Skip HTTP, only use paper_extra + lib fallback")
    args = p.parse_args()
    set_seed(0)
    t0 = time.time()
    if args.offline:
        obelix, cod, cemp = [], [], []
    else:
        obelix = fetch_obelix()
        cod = fetch_cod()
        cemp = fetch_cemp()
    paper = fetch_paper_extra()
    rows = merge([obelix, cod, cemp, paper])
    write_csv(Path(args.out_csv), rows)
    meta = {
        "n_obelix": len(obelix),
        "n_cod": len(cod),
        "n_cemp": len(cemp),
        "n_paper_extra": len(paper),
        "n_combined": len(rows),
        "elapsed_s": round(time.time() - t0, 2),
        "offline": args.offline,
    }
    write_json(Path(args.out_json), meta)
    print(f"[merge] {len(rows)} unique SSEs in {args.out_csv}")
    print(f"  sources: OBELiX={len(obelix)} COD={len(cod)} "
          f"CEMP={len(cemp)} paper={len(paper)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
