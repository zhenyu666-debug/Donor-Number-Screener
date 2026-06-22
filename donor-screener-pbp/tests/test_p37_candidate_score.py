"""test_p37_candidate_score.py - tests for the p37 candidate scoring pipeline.

Covers:
  - library load (14 entries, required fields)
  - score determinism (no RNG, same input -> same DN)
  - class-based default fill
  - rank ordering
  - offline enrich run produces no HTTP traffic
  - FastAPI /health responds 200 with n_candidates == 14
  - FastAPI /candidate_score for LGPS (id=3) returns expected breakdown
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))

import p37_candidate_score as cs  # noqa: E402
import p37_enrich_candidates as ce  # noqa: E402


REQUIRED_FIELDS = (
    "id", "name", "formula", "class",
    "sigma_ion_S_cm", "E_g_eV", "migration_eV",
)


# --------------------------------------------------------------------------- #
# 1. Library load
# --------------------------------------------------------------------------- #

def test_library_loads():
    lib = cs.load_source_library()
    assert len(lib) == 14, f"expected 14 SSEs, got {len(lib)}"
    for e in lib:
        for f in REQUIRED_FIELDS:
            assert f in e, f"entry id={e.get('id')} missing field {f}"
            assert e[f] is not None, f"entry id={e.get('id')} has null {f}"


# --------------------------------------------------------------------------- #
# 2. Score determinism
# --------------------------------------------------------------------------- #

def test_score_deterministic():
    entry = {"id": 99, "name": "test", "formula": "LiX",
             "class": "sulfide", "sigma_ion_S_cm": 1e-3,
             "E_g_eV": 2.5, "migration_eV": 0.3,
             "li_coord_num": 4.0}
    a = cs.compute_dn_candidate_v2(entry)
    b = cs.compute_dn_candidate_v2(entry)
    assert a["dn_candidate"] == b["dn_candidate"]
    assert math.isfinite(a["dn_candidate"])
    # All component DNs must be finite
    for k, v in a.items():
        assert math.isfinite(v), f"{k} not finite: {v}"


# --------------------------------------------------------------------------- #
# 3. Class defaults applied
# --------------------------------------------------------------------------- #

def test_class_defaults_applied():
    # Strip fields -> after fill, missing ones should be non-NaN
    e = {"id": 0, "name": "x", "formula": "Xx", "class": "oxide",
         "sigma_ion_S_cm": 1e-6, "E_g_eV": float("nan"),
         "migration_eV": float("nan"), "li_coord_num": float("nan")}
    filled = cs.fill_class_defaults(e)
    for f in ("E_g_eV", "migration_eV", "li_coord_num"):
        v = filled.get(f)
        assert v is not None
        assert isinstance(v, (int, float))
        assert math.isfinite(float(v)), f"{f} not finite after fill"


# --------------------------------------------------------------------------- #
# 4. Rank is sorted descending
# --------------------------------------------------------------------------- #

def test_rank_is_sorted_desc():
    lib = cs.load_source_library()
    rows = cs.score_library(lib)
    dns = [r["dn_candidate"] for r in rows]
    assert dns == sorted(dns, reverse=True), f"not sorted desc: {dns}"
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))


# --------------------------------------------------------------------------- #
# 5. Offline enrich is offline (no HTTP)
# --------------------------------------------------------------------------- #

def test_enrich_dry_run():
    lib = cs.load_source_library()
    target = lib[2]  # LGPS, id=3
    counter = [0]
    out = ce.enrich_entry(target, offline=True,
                          request_count=counter, max_requests=99)
    assert counter[0] == 0, f"offline run made {counter[0]} HTTP calls"
    assert "enrichment" in out
    # The enrichment block records what was missing; offline run never queries.
    assert out["enrichment"]["sources"] == {}


# --------------------------------------------------------------------------- #
# 6. /health
# --------------------------------------------------------------------------- #

def test_api_health():
    from fastapi.testclient import TestClient
    import p37_candidate_api as api
    with TestClient(api.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_candidates"] == 14
    assert body["lib"] in ("sse_library.yaml", "sse_library_enriched.yaml")


# --------------------------------------------------------------------------- #
# 7. /candidate_score for LGPS
# --------------------------------------------------------------------------- #

def test_api_candidate_score_lgps():
    from fastapi.testclient import TestClient
    import p37_candidate_api as api
    with TestClient(api.app) as client:
        r = client.post("/candidate_score", json={"id": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 3
    assert "LGPS" in body["name"]
    assert math.isfinite(body["dn_candidate"])
    # LGPS has the highest sigma (1.2e-2) in the lib; should rank #1.
    assert body["rank"] == 1
