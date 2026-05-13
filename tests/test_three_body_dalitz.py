"""Tests for the general 1→3 Dalitz-plot integrator."""
from __future__ import annotations

import math

import pytest

from feynman_engine.amplitudes.three_body_dalitz import dalitz_partial_width


class TestDalitzKinematics:
    """The 1→3 phase-space volume must match the textbook form."""

    def test_massless_threebody_phase_space(self):
        """∫dΦ₃ for M → 0+0+0 should equal M²/(256 π³).

        From PDG 47.22 with |M̄|² = 1:
            Γ = 1/(256 π³ M³) · ∫ ds₁₂ ds₂₃
        The Dalitz area for three massless daughters is M⁴/2, so
            Γ = 1/(256 π³ M³) · M⁴/2 = M/(512 π³).
        """
        M = 10.0
        r = dalitz_partial_width(M, 0.0, 0.0, 0.0, lambda s12, s23: 1.0)
        assert r["supported"]
        expected = M / (512.0 * math.pi ** 3)
        assert abs(r["Gamma_gev"] / expected - 1.0) < 1e-3

    def test_below_threshold(self):
        """When ∑m_i ≥ M, the decay is forbidden."""
        r = dalitz_partial_width(1.0, 0.4, 0.4, 0.4, lambda s12, s23: 1.0)
        assert not r["supported"]
        assert "energetically forbidden" in r["error"]

    def test_two_body_at_threshold_degenerate(self):
        """At ∑m_i = M − ε the Dalitz area should vanish smoothly."""
        M = 1.0
        m = 0.33
        r = dalitz_partial_width(M, m, m, m, lambda s12, s23: 1.0)
        assert r["supported"]
        # Phase space at near-threshold is small but positive
        assert 0.0 < r["Gamma_gev"] < M / (512.0 * math.pi ** 3)


class TestSargentLimitNumeric:
    """For massless daughters with Dalitz slots 1 = e, 2 = ν̄_e, 3 = ν_μ, the
    pure-V-A matrix element is

        |M̄|² = 64 G_F² (p_μ · p_{ν̄_e})(p_e · p_{ν_μ})
             = 16 G_F² s_(13) (M² − s_(13))
             = 16 G_F² (M² − s₁₂ − s₂₃)(s₁₂ + s₂₃)   (s_(13) = M² − s₁₂ − s₂₃)

    where the identifications follow from energy-momentum conservation in
    the μ rest frame.  Sargent: Γ = G_F² m_μ⁵ / (192 π³).
    """

    def test_muon_sargent_limit(self):
        G_F = 1.1663787e-5  # GeV⁻²
        m_mu = 0.1056583755
        M_sq = m_mu ** 2

        def msq(s12, s23):
            s13 = M_sq - s12 - s23
            return 16.0 * G_F ** 2 * s13 * (M_sq - s13)

        r = dalitz_partial_width(m_mu, 0.0, 0.0, 0.0, msq)
        assert r["supported"]
        expected = G_F ** 2 * m_mu ** 5 / (192.0 * math.pi ** 3)
        # Generic Dalitz integrator should reproduce Sargent to 1%.
        assert abs(r["Gamma_gev"] / expected - 1.0) < 0.01, (
            f"Γ(μ→eνν̄) = {r['Gamma_gev']:.3e}, expected {expected:.3e}"
        )
