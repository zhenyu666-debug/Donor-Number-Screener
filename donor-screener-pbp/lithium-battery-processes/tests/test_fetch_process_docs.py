"""test_fetch_process_docs.py - tests for the process documentation fetcher."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))

from utils_lb import (  # noqa: E402
    INDEX_CSV, MANIFEST_JSON, REPO_ROOT,
    sha256_file, sha256_bytes, write_csv,
)
import p36_fetch_process_docs as p36  # noqa: E402


def test_manifest_loads():
    assert MANIFEST_JSON.exists(), f"missing {MANIFEST_JSON}"
    with MANIFEST_JSON.open(encoding="utf-8-sig") as f:
        m = json.load(f)
    total = 0
    for cat in p36.CATEGORIES:
        for e in m.get(cat, []):
            miss = p36.validate_entry(e)
            assert not miss, f"{cat}/{e.get('id','?')}: missing keys {miss}"
            total += 1
    assert total >= 12, f"expected >= 12 sources, got {total}"


def test_process_steps_has_14_steps():
    steps = p36.load_yaml("process_steps.yaml").get("steps", [])
    assert len(steps) == 14, f"expected 14 steps, got {len(steps)}"
    for st in steps:
        for k in ("name_en", "name_zh", "equipment_en", "equipment_zh",
                  "purpose_en", "purpose_zh"):
            assert k in st, f"step {st.get('id')}: missing {k}"
            assert st[k], f"step {st.get('id')}: empty {k}"


def test_parameter_ranges_keys_resolve():
    missing = p36.verify_param_hints()
    assert not missing, f"param_hint not found in parameter_ranges.csv: {missing}"


def test_index_csv_header_and_min_rows():
    if not INDEX_CSV.exists():
        m = json.loads(MANIFEST_JSON.read_text(encoding="utf-8-sig"))
        rows = p36.build_index_rows(m)
        write_csv(INDEX_CSV, rows, fieldnames=list(p36.REQUIRED_KEYS))
    with INDEX_CSV.open(encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        n_data = sum(1 for _ in rdr)
    assert header == list(p36.REQUIRED_KEYS), \
        f"unexpected header: {header}"
    assert n_data >= 12, f"expected >= 12 rows, got {n_data}"


def test_sha256_roundtrip():
    payload = b"hello lithium-battery-processes\n"
    h = sha256_bytes(payload)
    assert len(h) == 64
    p = (REPO_ROOT / "tests" / "_sha256_smoke.txt")
    p.write_bytes(payload)
    try:
        assert sha256_file(p) == h
    finally:
        p.unlink()


def test_offline_runs_clean():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "p36_fetch_process_docs.py"),
         "--offline"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, \
        f"offline run failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "elapsed_s" in proc.stdout


def test_required_keys_complete():
    for k in p36.REQUIRED_KEYS:
        assert k


if __name__ == "__main__":
    test_manifest_loads()
    test_process_steps_has_14_steps()
    test_parameter_ranges_keys_resolve()
    test_index_csv_header_and_min_rows()
    test_sha256_roundtrip()
    test_offline_runs_clean()
    test_required_keys_complete()
    print("OK: all fetch_process_docs tests passed")
