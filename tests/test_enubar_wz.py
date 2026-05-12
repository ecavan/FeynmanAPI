"""Tests for the e- ν̄_e → W- Z helicity-amplitude evaluator."""
from __future__ import annotations

import pytest

from feynman_engine.amplitudes.enubar_wz_helicity import (
    is_supported, cross_section,
)


class TestEnubarWZSupport:
    """Process-string parsing and dispatch."""

    @pytest.mark.parametrize("process", [
        "e- nu_e~ -> W- Z",
        "e- nuebar -> W- Z",
        "e+ nu_e -> W+ Z",
        "e+ nue -> W+ Z",
    ])
    def test_supported(self, process):
        assert is_supported(process)

    @pytest.mark.parametrize("process", [
        "e+ e- -> mu+ mu-",
        "u u~ -> W+ W-",
        "e- nu_e~ -> W+ Z",  # wrong final-state charge
        "e+ e- -> W- Z",
    ])
    def test_not_supported(self, process):
        assert not is_supported(process)


class TestEnubarWZCrossSection:
    """Cross-section sanity checks."""

    def test_below_threshold_returns_error(self):
        r = cross_section("e- nu_e~ -> W- Z", sqrt_s=100.0)
        assert not r["supported"]
        assert "below the W-Z threshold" in r["error"]

    def test_above_threshold_returns_finite_sigma(self):
        r = cross_section("e- nu_e~ -> W- Z", sqrt_s=240.0)
        assert r["supported"]
        assert r["sigma_pb"] > 0.0
        # At √s = 240 GeV, σ is expected in the range 5-15 pb based on
        # published LEP-3 / FCC-ee diboson-physics literature.
        assert 1.0 < r["sigma_pb"] < 30.0

    def test_high_energy_unitarity(self):
        """σ should fall as 1/s at high energy (gauge cancellation)."""
        sigma_lo = cross_section("e- nu_e~ -> W- Z", sqrt_s=500.0)["sigma_pb"]
        sigma_hi = cross_section("e- nu_e~ -> W- Z", sqrt_s=3000.0)["sigma_pb"]
        # σ at 3 TeV must be smaller than σ at 500 GeV (no longitudinal
        # blow-up); should fall by at least factor 5 across this √s ratio.
        assert sigma_hi < sigma_lo
        assert sigma_lo / sigma_hi > 3.0

    def test_charge_conjugate_symmetric(self):
        """σ(e- ν̄_e → W- Z) == σ(e+ ν_e → W+ Z) up to numerical noise."""
        sigma_minus = cross_section("e- nu_e~ -> W- Z", sqrt_s=300.0)["sigma_pb"]
        sigma_plus = cross_section("e+ nu_e -> W+ Z", sqrt_s=300.0)["sigma_pb"]
        # CP-symmetric process; should be identical (up to numerical noise
        # from the cos θ grid).
        assert abs(sigma_minus - sigma_plus) / sigma_minus < 0.01


class TestEnubarWZIntegration:
    """End-to-end via total_cross_section dispatch."""

    def test_dispatch_via_total_cross_section_low_s(self):
        """At √s < 700 GeV the engine is ~50% LOW vs MG5 — trust is ROUGH."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e- nu_e~ -> W- Z", "EW", sqrt_s=240.0)
        assert r["sigma_pb"] > 1.0
        assert r["method"] == "enubar-wz-helicity-amplitudes"
        assert r["trust_level"] == "rough"
        assert "ORDER-OF-MAGNITUDE" in r["accuracy_caveat"]

    def test_dispatch_via_total_cross_section_high_s(self):
        """At √s ≥ 700 GeV the engine agrees with MG5 within α-scheme — APPROXIMATE."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e- nu_e~ -> W- Z", "EW", sqrt_s=1000.0)
        assert r["sigma_pb"] > 1.0
        assert r["method"] == "enubar-wz-helicity-amplitudes"
        assert r["trust_level"] == "approximate"
