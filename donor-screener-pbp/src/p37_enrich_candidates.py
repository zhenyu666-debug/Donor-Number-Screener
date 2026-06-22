"""37_enrich_candidates.py - Backfill missing fields in sse_library.yaml via
external APIs (Crossref, Materials Project, PubChem).

Reads:
  data/sse_library.yaml - the canonical 14-SSE library (do not overwrite)
Writes:
  data/sse_library_enriched.yaml - same 14 entries + enrichment metadata
  data/sse_enrich_meta.json     - per-entry status / sources queried

Strategy
--------
For every SSE entry we compute a `missing_fields` set and only query APIs for
those fields. All three APIs are called via urllib (no extra deps), and every
failure is a warning -- the run never raises.

API precedence (per missing field, top-down):
  1. Crossref   - if the entry's "source" string contains a DOI/PMID/arXiv id,
                  the work's metadata is fetched. We mainly use it to update
                  the `source` field (year, journal); no field backfill.
  2. Materials Project - if MP_API_KEY is set, query the formula. We backfill
                  `E_g_eV` (band_gap), `density_g_cm3` (density), and
                  `E_form_eV` (formation_energy_per_atom). If the env var is
                  not set, this step is skipped silently.
  3. PubChem    - REST formula lookup. We backfill `molar_mass_g_mol` only.
                  PubChem is meant for organic molecules, so it is the
                  weakest signal and only used as a last resort.

CLI:
  --offline       : never call any HTTP endpoint, return [] for everything.
  --only id1,id2  : only enrich the entries whose id is in this list.
  --max-requests N: cap total HTTP requests (default 50).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import yaml

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils_pb import DATA_DIR, ensure_dir, write_json  # noqa: E402


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SRC_LIB = DATA_DIR / "sse_library.yaml"
OUT_LIB = DATA_DIR / "sse_library_enriched.yaml"
OUT_META = DATA_DIR / "sse_enrich_meta.json"

UA = "lithium-battery-processes-p37/1.0 (offline-safe)"
DEFAULT_TIMEOUT = 15.0

CROSSREF = "https://api.crossref.org/works/{id}"
MP_URL = "https://api.materialsproject.org/materials/{formula}/vasp/{mp_id}"
PUBCHEM = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/"
           "{formula}/property/MolecularWeight/JSON")

# Map: which enriched field corresponds to which Materials Project JSON key.
# We probe the vasp endpoint first (gives the full set), fall back to /summary.
MP_FIELD_MAP = {
    "E_g_eV": "band_gap",
    "density_g_cm3": "density",
    "E_form_eV": "formation_energy_per_atom",
}

# Fields the scoring formula cares about.  Anything missing here is a
# candidate for API enrichment.
SCORING_FIELDS = (
    "sigma_ion_S_cm",
    "E_g_eV",
    "E_form_eV",
    "stability_window_V",
    "density_g_cm3",
    "molar_mass_g_mol",
    "li_coord_num",
    "migration_eV",
)


# --------------------------------------------------------------------------- #
# HTTP helper
# --------------------------------------------------------------------------- #

def http_get_json(url: str, timeout: float = DEFAULT_TIMEOUT) -> Optional[Any]:
    """urllib JSON GET, return parsed object or None on any failure."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                   "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"[enrich] GET {url} failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# Field-level enrichment
# --------------------------------------------------------------------------- #

# Crossref pattern: any token in the source string that looks like a DOI/PMID/arXiv.
_RE_DOI = re.compile(r"\b10\.\d{4,9}/[^\s,;]+")
_RE_PMID = re.compile(r"\bPMID[:\s]*(\d+)\b", re.IGNORECASE)
_RE_ARXIV = re.compile(r"\barXiv[:\s]*(\d{4}\.\d{4,5}(v\d+)?)\b", re.IGNORECASE)


def parse_source_refs(source: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (doi, pmid, arxiv) extracted from the source string (first hit)."""
    if not source:
        return None, None, None
    doi_m = _RE_DOI.search(source)
    pmid_m = _RE_PMID.search(source)
    arxiv_m = _RE_ARXIV.search(source)
    return (
        doi_m.group(0).rstrip(".,;") if doi_m else None,
        pmid_m.group(1) if pmid_m else None,
        arxiv_m.group(1) if arxiv_m else None,
    )


def enrich_crossref(source: str) -> Dict[str, Any]:
    """Return what Crossref tells us about the source publication. Never raises."""
    doi, pmid, arxiv = parse_source_refs(source)
    if not (doi or pmid or arxiv):
        return {"queried": False, "reason": "no-doi-pmid-arxiv"}
    url_id = doi or f"pmid:{pmid}" if pmid else f"arxiv:{arxiv}"
    url = CROSSREF.format(id=quote(url_id, safe=""))
    data = http_get_json(url, timeout=10.0)
    if not data or "message" not in data:
        return {"queried": True, "found": False, "id": url_id}
    msg = data["message"]
    issued = (msg.get("issued") or {}).get("date-parts") or [[None]]
    year = None
    if issued and issued[0]:
        year = issued[0][0]
    journal = ((msg.get("container-title") or [None]) or [None])[0]
    return {
        "queried": True,
        "found": True,
        "id": url_id,
        "year": year,
        "journal": journal,
        "title": (msg.get("title") or [None])[0],
    }


def enrich_materials_project(formula: str,
                             missing: List[str]) -> Dict[str, Any]:
    """Use MP to fill E_g / density / formation_energy_per_atom. Requires key."""
    if not os.environ.get("MP_API_KEY"):
        return {"queried": False, "reason": "MP_API_KEY not set"}
    if not missing:
        return {"queried": False, "reason": "no MP-relevant missing fields"}
    # The public MP REST requires a formula id; for safety we use the summary
    # endpoint which accepts a chemsys / formula identifier.  For our 14 SSEs
    # the formula is stable enough.
    url = f"https://api.materialsproject.org/materials/{quote(formula, safe='')}/summary"
    headers = {"User-Agent": UA, "Accept": "application/json",
               "X-API-KEY": os.environ["MP_API_KEY"]}
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"[enrich] MP GET {url} failed: {e}")
        return {"queried": True, "found": False, "reason": str(e)}
    rows = (data or {}).get("data") or []
    if not rows:
        return {"queried": True, "found": False, "reason": "no data"}
    # take the first row; multiple polymorphs would need further filtering
    row = rows[0]
    out = {"queried": True, "found": True,
           "mp_id": row.get("material_id"),
           "values": {}}
    for sse_key, mp_key in MP_FIELD_MAP.items():
        if sse_key in missing and mp_key in row and row[mp_key] is not None:
            out["values"][sse_key] = float(row[mp_key])
    return out


def enrich_pubchem(formula: str, missing: List[str]) -> Dict[str, Any]:
    """Use PubChem to fill molar_mass.  Will only work for organic formulas."""
    if "molar_mass_g_mol" not in missing:
        return {"queried": False, "reason": "molar_mass not missing"}
    # PubChem can't handle subscripts/dots like Li6.4La3Zr1.4Ta0.6O12 cleanly;
    # for inorganic formulas we still try -- on failure it just returns None.
    # Strip charge/special chars to be safe.
    safe = re.sub(r"[^A-Za-z0-9]", "", formula)
    if not safe:
        return {"queried": False, "reason": "empty formula"}
    url = PUBCHEM.format(formula=quote(safe, safe=""))
    data = http_get_json(url, timeout=10.0)
    if not data or "PropertyTable" not in data:
        return {"queried": True, "found": False, "reason": "no PropertyTable"}
    props = (data["PropertyTable"].get("Properties") or [])
    if not props:
        return {"queried": True, "found": False, "reason": "no properties"}
    mw = props[0].get("MolecularWeight")
    if mw is None:
        return {"queried": True, "found": False, "reason": "no MolecularWeight"}
    return {"queried": True, "found": True, "values": {"molar_mass_g_mol": float(mw)}}


# --------------------------------------------------------------------------- #
# Main enrichment driver
# --------------------------------------------------------------------------- #

def load_library() -> List[Dict[str, Any]]:
    if not SRC_LIB.exists():
        return []
    with SRC_LIB.open(encoding="utf-8-sig") as f:
        d = yaml.safe_load(f) or {}
    return d.get("sse", [])


def is_missing(entry: Dict[str, Any], field: str) -> bool:
    v = entry.get(field)
    if v is None:
        return True
    if isinstance(v, float):
        try:
            import math
            return not math.isfinite(v)
        except Exception:
            return True
    return False


def missing_scoring_fields(entry: Dict[str, Any]) -> List[str]:
    return [f for f in SCORING_FIELDS if is_missing(entry, f)]


def enrich_entry(entry: Dict[str, Any], *, offline: bool,
                 request_count: List[int], max_requests: int) -> Dict[str, Any]:
    """Enrich a single SSE entry. Mutates a copy and returns it."""
    import copy
    out = copy.deepcopy(entry)
    missing = missing_scoring_fields(entry)
    enrich_record: Dict[str, Any] = {
        "id": entry.get("id"),
        "name": entry.get("name"),
        "missing_fields": missing,
        "sources": {},
    }
    if offline or not missing:
        out["enrichment"] = enrich_record
        return out

    # 1. Crossref: metadata only -- not a field source, but useful provenance.
    if request_count[0] < max_requests:
        request_count[0] += 1
        time.sleep(0.3)
        enrich_record["sources"]["crossref"] = enrich_crossref(entry.get("source", ""))

    # 2. Materials Project: E_g / density / E_form.
    mp_relevant = [f for f in missing if f in MP_FIELD_MAP]
    if mp_relevant and request_count[0] < max_requests:
        request_count[0] += 1
        time.sleep(0.3)
        mp = enrich_materials_project(entry.get("formula", ""), mp_relevant)
        enrich_record["sources"]["materials_project"] = mp
        for k, v in (mp.get("values") or {}).items():
            if is_missing(out, k):
                out[k] = v

    # 3. PubChem: molar_mass.
    if "molar_mass_g_mol" in missing and request_count[0] < max_requests:
        request_count[0] += 1
        time.sleep(0.3)
        pc = enrich_pubchem(entry.get("formula", ""), missing)
        enrich_record["sources"]["pubchem"] = pc
        for k, v in (pc.get("values") or {}).items():
            if is_missing(out, k):
                out[k] = v

    out["enrichment"] = enrich_record
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true",
                   help="Skip all HTTP, only carry over enrichment scaffold.")
    p.add_argument("--only", default="",
                   help="Comma-separated ids to enrich (default: all).")
    p.add_argument("--max-requests", type=int, default=50,
                   help="Cap on total HTTP requests (default 50).")
    args = p.parse_args()

    lib = load_library()
    if not lib:
        print("[enrich] source library empty or missing; nothing to do.")
        return 0

    only_ids = set()
    if args.only:
        for tok in args.only.split(","):
            tok = tok.strip()
            if tok:
                try:
                    only_ids.add(int(tok))
                except ValueError:
                    only_ids.add(tok)

    request_count = [0]
    enriched: List[Dict[str, Any]] = []
    for entry in lib:
        if only_ids and entry.get("id") not in only_ids:
            # pass-through unchanged (no enrichment block added)
            enriched.append(entry)
            continue
        out = enrich_entry(entry, offline=args.offline,
                           request_count=request_count,
                           max_requests=args.max_requests)
        enriched.append(out)
        m = out.get("enrichment", {})
        n_miss = len(m.get("missing_fields") or [])
        n_fix = sum(1 for k in SCORING_FIELDS
                    if not is_missing(out, k) and is_missing(entry, k))
        print(f"[enrich] id={entry.get('id'):>2} {entry.get('name', '')[:32]:32s}"
              f" missing={n_miss} filled={n_fix}")

    ensure_dir(OUT_LIB.parent)
    with OUT_LIB.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"sse": enriched, "enrichment_source": "p37"},
                       f, sort_keys=False, allow_unicode=True)
    meta = {
        "offline": args.offline,
        "n_entries": len(enriched),
        "n_enriched": sum(1 for e in enriched if "enrichment" in e),
        "http_requests": request_count[0],
        "max_requests": args.max_requests,
        "mp_api_key_set": bool(os.environ.get("MP_API_KEY")),
    }
    write_json(OUT_META, meta)
    print(f"[enrich] wrote {OUT_LIB}  ({len(enriched)} entries, "
          f"{request_count[0]} HTTP requests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
