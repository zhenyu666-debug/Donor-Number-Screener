"""test_sse_redn.py - tests for the SSE re-ranking module."""
import math
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src"))
from p32_sse_redn import (  # noqa: E402
    empirical_dn, langevin_proxy_dn, sei_attenuation,
    load_sse_library, rerank,
)


def test_empirical_dn_increases_with_sigma():
    base = {"E_g_eV": 2.5, "migration_eV": 0.3, "sigma_ion_S_cm": 1e-6}
    w = {"alpha_sigma": 1e3, "beta_Eg": 1.0, "gamma_floor": 5.0, "delta_migration": -2.0}
    d_low = empirical_dn(base, w)
    d_high = empirical_dn({**base, "sigma_ion_S_cm": 1e-2}, w)
    assert d_high > d_low


def test_empirical_dn_decreases_with_migration():
    base = {"E_g_eV": 2.5, "sigma_ion_S_cm": 1e-3, "migration_eV": 0.3}
    w = {"alpha_sigma": 1e3, "beta_Eg": 1.0, "gamma_floor": 5.0, "delta_migration": -2.0}
    d_low = empirical_dn(base, w)
    d_high = empirical_dn({**base, "migration_eV": 0.6}, w)
    assert d_low > d_high


def test_sei_attenuation_in_0_1():
    e = {"migration_eV": 0.3}
    s = sei_attenuation(e)
    assert 0.0 <= s <= 1.0


def test_langevin_proxy_dn_finite():
    np.random.seed(0)
    e = {"sigma_ion_S_cm": 1e-3, "E_g_eV": 2.5, "migration_eV": 0.3}
    d = langevin_proxy_dn(e, anchor=22.0)
    assert math.isfinite(d)


def test_sse_library_has_14():
    lib, rk = load_sse_library()
    assert len(lib) == 14


def test_rerank_returns_14_rows_in_order():
    rows = rerank()
    assert len(rows) == 14
    dns = [r["dn_pbp_v2"] for r in rows]
    assert dns == sorted(dns, reverse=True)
    # Rank must be 1..14
    ranks = sorted(r["rank"] for r in rows)
    assert ranks == list(range(1, 15))


def test_lgps_in_top5():
    """LGPS has the highest sigma_ion -> should be near the top."""
    rows = rerank()
    by_name = {r["sse"]: r for r in rows}
    assert by_name["Li10GeP2S12 (LGPS)"]["rank"] <= 5


def test_peo_low_sigma_low_dn():
    """PEO+LiTFSI has 1e-5 S/cm -> near the bottom."""
    rows = rerank()
    by_name = {r["sse"]: r for r in rows}
    assert by_name["PEO+LiTFSI (polymer)"]["rank"] >= 10


if __name__ == "__main__":
    test_empirical_dn_increases_with_sigma()
    test_empirical_dn_decreases_with_migration()
    test_sei_attenuation_in_0_1()
    test_langevin_proxy_dn_finite()
    test_sse_library_has_14()
    test_rerank_returns_14_rows_in_order()
    test_lgps_in_top5()
    test_peo_low_sigma_low_dn()
    print("OK: all SSE re-rank tests passed")
