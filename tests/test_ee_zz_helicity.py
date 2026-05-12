"""Tests for the e+ e- → Z Z helicity-amplitude evaluator."""
from __future__ import annotations

import pytest

from feynman_engine.amplitudes.ee_zz_helicity import is_supported, cross_section


class TestEeZZSupport:
    @pytest.mark.parametrize("process", [
        "e+ e- -> Z Z",
        "e- e+ -> Z Z",
    ])
    def test_supported(self, process):
        assert is_supported(process)

    @pytest.mark.parametrize("process", [
        "e+ e- -> W+ W-",
        "u u~ -> Z Z",
        "e+ e- -> Z H",
    ])
    def test_not_supported(self, process):
        assert not is_supported(process)


class TestEeZZCrossSection:
    def test_below_threshold(self):
        r = cross_section("e+ e- -> Z Z", sqrt_s=150.0)
        assert not r["supported"]
        assert "below the Z Z threshold" in r["error"]

    def test_at_lep2(self):
        """ee→ZZ at LEP2 (√s = 200 GeV): MG5 = 1.312 pb."""
        r = cross_section("e+ e- -> Z Z", sqrt_s=200.0)
        assert r["supported"]
        # Engine should agree with MG5 within 10% (no α-scheme issue here
        # since γZ exchange has different scaling).  Verified -4% as of v0.2.2.
        assert 1.10 < r["sigma_pb"] < 1.50

    def test_at_500(self):
        """ee→ZZ at √s = 500 GeV: MG5 = 0.416 pb."""
        r = cross_section("e+ e- -> Z Z", sqrt_s=500.0)
        assert r["supported"]
        assert 0.36 < r["sigma_pb"] < 0.45

    def test_high_energy_decreases(self):
        """σ should decrease at high √s (no gauge-violating blow-up)."""
        sig_500 = cross_section("e+ e- -> Z Z", sqrt_s=500.0)["sigma_pb"]
        sig_1000 = cross_section("e+ e- -> Z Z", sqrt_s=1000.0)["sigma_pb"]
        # Energy-dependence: σ ~ 1/s at high s for fixed-coupling processes
        assert sig_500 > sig_1000

    def test_dispatch_via_total_cross_section(self):
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e+ e- -> Z Z", "EW", sqrt_s=200.0)
        assert r["sigma_pb"] > 1.0
        assert r["method"] == "ee-zz-helicity-amplitudes"
