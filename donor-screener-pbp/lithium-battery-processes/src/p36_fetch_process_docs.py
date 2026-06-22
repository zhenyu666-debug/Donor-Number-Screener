"""p36_fetch_process_docs.py - offline-safe fetcher for the lithium-battery
process documentation mirror.

Reads ``manifest.json`` (hand-curated list of sources) and downloads each
entry into ``sources/<category>/<file>``, computing sha256 along the way.
Writes a unified bilingual ``index.csv`` at the repo root.

CLI:
  python src/p36_fetch_process_docs.py           # online, fetch everything
  python src/p36_fetch_process_docs.py --offline # verify only, no network
  python src/p36_fetch_process_docs.py --verify  # re-checksum, no network
  python src/p36_fetch_process_docs.py --only cat1,cat2  # filter categories

Offline mode is the default in CI: every HTTP failure is a warning, not an
error. The manifest entry is updated with the actual sha256 on a successful
download so subsequent runs can verify integrity.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from utils_lb import (  # noqa: E402
    DATA_DIR, INDEX_CSV, MANIFEST_JSON, REPO_ROOT, ensure_dir, http_get, load_yaml, sha256_file, write_csv, write_json,
)


REQUIRED_KEYS = (
    "id", "category", "title_en", "title_zh", "publisher", "year",
    "url_en", "url_zh", "license", "local_path", "sha256",
    "notes_en", "notes_zh",
)

CATEGORIES = ("process_summary", "electrode_manufacturing",
              "cell_assembly", "formation", "aging_calendar")


def load_manifest() -> Dict:
    if not MANIFEST_JSON.exists():
        return {c: [] for c in CATEGORIES}
    with MANIFEST_JSON.open(encoding="utf-8-sig") as f:
        return json.load(f)


def validate_entry(e: Dict) -> List[str]:
    return [k for k in REQUIRED_KEYS if k not in e]


def fetch_entry(e: Dict, *, offline: bool = False) -> Dict:
    """Download a single entry. Returns a copy with sha256 + status updated."""
    out = dict(e)
    local = REPO_ROOT / e["local_path"]
    ensure_dir(local.parent)

    if offline:
        if local.exists():
            actual = sha256_file(local)
            expected = (e.get("sha256") or "").strip()
            if expected and expected != actual:
                out["status"] = "MISMATCH"
                out["sha256"] = actual
                print(f"[offline] {e['id']}: SHA mismatch "
                      f"(expected {expected[:8]}, got {actual[:8]})")
            else:
                out["status"] = "OK"
                out["sha256"] = actual
        else:
            out["status"] = "MISSING"
            print(f"[offline] {e['id']}: missing local file {local}")
        return out

    url = e.get("url_en") or e.get("url_zh")
    if not url:
        out["status"] = "NO_URL"
        return out
    data = http_get(url, timeout=45.0)
    if data is None:
        out["status"] = "FETCH_FAIL"
        if local.exists():
            out["sha256"] = sha256_file(local)
            out["status"] = "FETCH_FAIL_KEPT"
        return out

    if url.endswith(".html") or e["local_path"].endswith(".html"):
        header = (f"<!-- mirrored from {url} on "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')} -->\n").encode("utf-8")
        local.write_bytes(header + data)
    else:
        local.write_bytes(data)
    out["sha256"] = sha256_file(local)
    out["status"] = "OK"
    print(f"[fetch] {e['id']}: {len(data)} bytes -> {local.name} "
          f"sha256={out['sha256'][:8]}")
    time.sleep(0.3)
    return out


def build_index_rows(manifest: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for cat in CATEGORIES:
        for e in manifest.get(cat, []):
            rows.append({
                "id": e.get("id", ""),
                "category": cat,
                "title_en": e.get("title_en", ""),
                "title_zh": e.get("title_zh", ""),
                "publisher": e.get("publisher", ""),
                "year": e.get("year", ""),
                "url_en": e.get("url_en") or "",
                "url_zh": e.get("url_zh") or "",
                "local_path": e.get("local_path", ""),
                "sha256": e.get("sha256", ""),
                "license": e.get("license", ""),
                "notes_en": e.get("notes_en", ""),
                "notes_zh": e.get("notes_zh", ""),
            })
    return rows


def update_manifest_with_status(manifest: Dict,
                                updated_by_id: Dict[str, Dict]) -> Dict:
    for cat in CATEGORIES:
        new_list = []
        for e in manifest.get(cat, []):
            upd = updated_by_id.get(e["id"])
            if upd:
                e = dict(e)
                e["sha256"] = upd.get("sha256", e.get("sha256", ""))
            new_list.append(e)
        manifest[cat] = new_list
    return manifest


def verify_param_hints() -> List[str]:
    steps = load_yaml("process_steps.yaml")
    ranges_path = DATA_DIR / "parameter_ranges.csv"
    keys: set = set()
    if ranges_path.exists():
        with ranges_path.open(encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                if row.get("key"):
                    keys.add(row["key"].strip())
    missing: List[str] = []
    for st in steps.get("steps", []):
        h = (st.get("param_hint") or "").strip()
        if h and h not in keys:
            missing.append(h)
    return missing


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true",
                   help="Verify existing local files only, no network.")
    p.add_argument("--verify", action="store_true",
                   help="Alias of --offline (kept for backwards compatibility).")
    p.add_argument("--only", default="",
                   help="Comma-separated category filter, e.g. "
                        "process_summary,formation")
    args = p.parse_args()
    offline = bool(args.offline or args.verify)

    t0 = time.time()
    manifest = load_manifest()
    only = {c.strip() for c in args.only.split(",") if c.strip()}

    updated_by_id: Dict[str, Dict] = {}
    n_ok = n_warn = 0
    for cat in CATEGORIES:
        if only and cat not in only:
            continue
        for e in manifest.get(cat, []):
            miss = validate_entry(e)
            if miss:
                print(f"[skip] {e.get('id','?')}: missing keys {miss}")
                n_warn += 1
                continue
            upd = fetch_entry(e, offline=offline)
            updated_by_id[e["id"]] = upd
            if upd.get("status") == "OK":
                n_ok += 1
            else:
                n_warn += 1

    rows = build_index_rows(manifest)
    if rows:
        write_csv(INDEX_CSV, rows, fieldnames=list(REQUIRED_KEYS))

    if not offline and n_ok > 0:
        manifest = update_manifest_with_status(manifest, updated_by_id)
        write_json(MANIFEST_JSON, manifest)

    missing_hints = verify_param_hints()
    summary = {
        "n_ok": n_ok,
        "n_warn": n_warn,
        "n_total": sum(len(manifest.get(c, [])) for c in CATEGORIES),
        "n_categories": len([c for c in CATEGORIES if not only or c in only]),
        "missing_param_hints": missing_hints,
        "elapsed_s": round(time.time() - t0, 2),
        "offline": offline,
    }
    print(f"[done] {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
