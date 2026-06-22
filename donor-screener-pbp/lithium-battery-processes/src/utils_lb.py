"""utils_lb.py - shared helpers for the lithium-battery-processes mirror.

Mirrors the role of donor-screener-pbp/src/utils_pb.py: CSV / JSON I/O,
sha256, urllib-only HTTP fetch, and a small YAML loader. No external
deps beyond pyyaml (already in the parent repo's requirements.txt).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DATA_DIR = REPO_ROOT / "data"
SOURCES_DIR = REPO_ROOT / "sources"
INDEX_CSV = REPO_ROOT / "index.csv"
MANIFEST_JSON = REPO_ROOT / "manifest.json"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: Path, rows: List[dict],
              fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fns = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fns})


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_yaml(name: str) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    p = DATA_DIR / name
    if not p.exists():
        return {}
    with p.open(encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, timeout: float = 30.0,
             user_agent: str = "lithium-battery-processes/1.0") -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f"[http_get] {url} failed: {e}")
        return None


def set_seed(_seed: int = 0) -> None:
    return None


def main() -> int:
    print(f"REPO_ROOT   = {REPO_ROOT}")
    print(f"DATA_DIR    = {DATA_DIR}")
    print(f"SOURCES_DIR = {SOURCES_DIR}")
    print(f"MANIFEST    = {MANIFEST_JSON}")
    print(f"INDEX_CSV   = {INDEX_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
