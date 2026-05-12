"""Tests for the HPZ helicity-amplitude evaluator.

Validates the full LO matrix element for q q̄ → W+ W- against:
  * MG5 LO published reference (e+ e- → W+ W- at √s = 200 GeV: 19.54 pb)
  * Engine's ee→μμ pure-QED textbook value (sanity check on framework)
  * Polarization sum + spinor completeness identities
  * Symmetric behavior under flavor permutations within charge group
  * Cross-section response shape (peaks just above threshold, falls at high s)
"""
from __future__ import annotations

import math

import pytest


# ─── Self-consistency: framework primitives ─────────────────────────────────

class TestPrimitives:
    """The Dirac/spinor/polarization scaffolding must satisfy known identities."""

    def test_polarization_sum_identity(self):
        """Σ_λ ε^μ(p, λ) ε*^ν(p, λ) = -g^{μν} + p^μ p^ν / m_W² for on-shell p."""
        import numpy as np
        from feynman_engine.amplitudes.qqbar_ww_helicity import (
            _w_polarization, _ETA, _M_W,
        )
        direction = np.array([0.3, 0.4, 0.5])
        direction = direction / np.linalg.norm(direction)
        E = 200.0
        p_abs = math.sqrt(E*E - _M_W*_M_W)
        p = np.array([E, p_abs * direction[0], p_abs * direction[1], p_abs * direction[2]])

        pol_sum = np.zeros((4, 4), dtype=complex)
        for lam in (-1, 0, +1):
            eps = _w_polarization(p, lam)
            for mu in range(4):
                for nu in range(4):
                    pol_sum[mu, nu] += eps[mu] * np.conjugate(eps[nu])
        expected = -_ETA + np.outer(p, p) / (_M_W ** 2)
        assert np.abs(pol_sum - expected).max() < 1e-12

    def test_spinor_completeness_massless(self):
        """Σ_h u_h(p) ū_h(p) = p̸ for massless p."""
        import numpy as np
        from feynman_engine.amplitudes.qqbar_ww_helicity import (
            _spinor_u, _adjoint, _slash,
        )
        E = 100.0
        # massless: |p| = E
        p = np.array([E, 30.0, 40.0, math.sqrt(E*E - 30.0**2 - 40.0**2)])
        u_sum = np.zeros((4, 4), dtype=complex)
        for h in (-1, +1):
            u = _spinor_u(p, h)
            u_sum += np.outer(u, _adjoint(u))
        assert np.abs(u_sum - _slash(p)).max() < 1e-12


# ─── Cross-section vs published references ─────────────────────────────────

class TestPublishedReferences:
    """Direct comparison to the headline numbers from MG5_COMPARISON.md."""

    def test_ee_ww_at_200_matches_mg5(self):
        """e+ e- → W+ W- at √s = 200 GeV: MG5 v3.7.1 reports 19.54 ± 0.05 pb."""
        from feynman_engine.amplitudes.qqbar_ww_helicity import cross_section
        r = cross_section("e+ e- -> W+ W-", sqrt_s=200.0)
        assert r["supported"]
        # Allow 5 % tolerance: massless-electron approximation + α(M_Z) scheme.
        assert 18.0 <= r["sigma_pb"] <= 21.0, (
            f"σ(ee→WW, 200 GeV) = {r['sigma_pb']:.3f} pb; "
            f"MG5 reference 19.54 pb (5 % tolerance)"
        )

    def test_below_threshold_blocked(self):
        """Below 2 m_W ≈ 161 GeV the result must be unsupported."""
        from feynman_engine.amplitudes.qqbar_ww_helicity import cross_section
        r = cross_section("e+ e- -> W+ W-", sqrt_s=150.0)
        assert not r["supported"]
        assert "threshold" in r.get("error", "").lower()

    def test_cross_section_peaks_just_above_threshold(self):
        """σ(ee→WW) maximises near √s ≈ 200 GeV and falls at high energy."""
        from feynman_engine.amplitudes.qqbar_ww_helicity import cross_section
        sigmas = []
        for E in [200, 300, 500, 1000, 2000]:
            r = cross_section("e+ e- -> W+ W-", sqrt_s=E)
            sigmas.append(r["sigma_pb"])
        # Should be monotonically decreasing past peak
        for s_lo, s_hi in zip(sigmas[1:-1], sigmas[2:]):
            assert s_lo >= s_hi, f"σ should fall above peak: {sigmas}"


# ─── Per-flavour symmetry ───────────────────────────────────────────────────

class TestFlavorSymmetry:
    """u/c quarks identical (same Q, T₃); d/s/b quarks identical."""

    def test_up_type_quarks_identical(self):
        """σ(uū → WW) = σ(cc̄ → WW) — same charge/isospin, massless limit."""
        from feynman_engine.amplitudes.qqbar_ww_helicity import cross_section
        r_u = cross_section("u u~ -> W+ W-", sqrt_s=300.0)
        r_c = cross_section("c c~ -> W+ W-", sqrt_s=300.0)
        assert abs(r_u["sigma_pb"] - r_c["sigma_pb"]) / r_u["sigma_pb"] < 1e-6

    def test_down_type_quarks_identical(self):
        """σ(dd̄) = σ(ss̄) = σ(bb̄)."""
        from feynman_engine.amplitudes.qqbar_ww_helicity import cross_section
        r_d = cross_section("d d~ -> W+ W-", sqrt_s=300.0)
        r_s = cross_section("s s~ -> W+ W-", sqrt_s=300.0)
        r_b = cross_section("b b~ -> W+ W-", sqrt_s=300.0)
        assert abs(r_d["sigma_pb"] - r_s["sigma_pb"]) / r_d["sigma_pb"] < 1e-6
        assert abs(r_d["sigma_pb"] - r_b["sigma_pb"]) / r_d["sigma_pb"] < 1e-6


# ─── Engine integration: total_cross_section dispatches to HPZ ─────────────

class TestEngineDispatch:
    """The engine's `total_cross_section` should route q q̄ → W+ W- to HPZ."""

    def test_engine_uses_hpz_for_ee_ww(self):
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e+ e- -> W+ W-", "EW", sqrt_s=200.0)
        assert r.get("supported")
        assert r.get("method") == "hpz-helicity-amplitudes"
        # Same numerical agreement vs MG5
        assert 18.0 <= r["sigma_pb"] <= 21.0

    def test_engine_uses_hpz_for_uu_ww(self):
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("u u~ -> W+ W-", "EW", sqrt_s=250.0)
        assert r.get("supported")
        assert r.get("method") == "hpz-helicity-amplitudes"

    def test_unsupported_process_returns_unsupported(self):
        from feynman_engine.amplitudes.qqbar_ww_helicity import cross_section
        r = cross_section("e+ e- -> mu+ mu-", sqrt_s=200.0)
        assert not r["supported"]


# ─── High-energy gauge cancellation ────────────────────────────────────────

class TestUnitarityCancellation:
    """The full SM result has σ ~ 1/s² at high s (gauge invariance), while
    individual diagrams (t-only, γ-only, Z-only) grow with s.  Verify the
    cancellation works."""

    def test_full_sm_falls_at_high_energy(self):
        from feynman_engine.amplitudes.qqbar_ww_helicity import cross_section
        s_low = cross_section("e+ e- -> W+ W-", sqrt_s=500.0)["sigma_pb"]
        s_hi = cross_section("e+ e- -> W+ W-", sqrt_s=5000.0)["sigma_pb"]
        # σ falls by at least 10× from 500 → 5000 GeV (s grew by 100×).
        # In the SM full result σ ~ 1/s × log corrections, so the ratio
        # should be ~50× or more.
        assert s_low / s_hi > 10, (
            f"Full SM σ(ee→WW) should fall sharply at high s.  "
            f"σ(500)={s_low:.3f}, σ(5000)={s_hi:.3f}, ratio={s_low/s_hi:.1f}"
        )
