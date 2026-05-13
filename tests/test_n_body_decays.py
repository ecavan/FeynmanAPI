"""Tests for the N-body decay-width integrator."""
from __future__ import annotations

import math
import numpy as np
import pytest

from feynman_engine.amplitudes.n_body_decays import (
    n_body_partial_width, higgs_to_4l_BR,
)


class TestNBodyPhaseSpaceVolume:
    """For massless final states with |M̄|²=1, RAMBO should reproduce the
    analytic N-body phase-space volume.

    Φ_N = M^(2N-4) / [2 (4π)^(2N-3) (N-1)! (N-2)!]

    Then Γ = 1/(2M) × Φ_N for |M̄|²=1 → Γ_N(M) = M^(2N-5) / ...
    """

    def test_4body_phase_space_volume(self):
        """4-body massless: Φ_4 = M⁴ / [2 (4π)⁵ × 3! × 2!] = M⁴ / [(4π)⁵ × 24]."""
        M = 10.0
        N = 4
        r = n_body_partial_width(
            M, [0.0] * N,
            msq_callback=lambda mom: np.ones(mom.shape[0]),
            n_events=20_000, seed=42,
        )
        assert r["supported"]
        # Phase-space volume Φ_N for N massless daughters at parent mass M:
        # Φ_N = M^(2N-4) / [(4π)^(2N-3) × (N-1)! × (N-2)! × 2]
        Phi_analytic = M ** (2 * N - 4) / (
            (4.0 * math.pi) ** (2 * N - 3)
            * math.factorial(N - 1) * math.factorial(N - 2) * 2.0
        )
        gamma_analytic = Phi_analytic / (2.0 * M)
        # 5% Monte Carlo tolerance
        assert abs(r["Gamma_gev"] / gamma_analytic - 1.0) < 0.05, (
            f"Γ={r['Gamma_gev']:.3e} vs expected {gamma_analytic:.3e}"
        )

    def test_below_threshold_forbidden(self):
        r = n_body_partial_width(1.0, [0.4, 0.4, 0.4, 0.4], lambda m: np.ones(m.shape[0]))
        assert not r["supported"]
        assert "forbidden" in r["error"]

    def test_uncertainty_positive_definite(self):
        r = n_body_partial_width(
            10.0, [0.0, 0.0, 0.0, 0.0],
            lambda mom: np.ones(mom.shape[0]),
            n_events=5000, seed=42,
        )
        assert r["supported"]
        assert r["Gamma_uncertainty_gev"] > 0
        assert r["Gamma_uncertainty_gev"] < r["Gamma_gev"]    # rel. unc. < 1


class TestHiggsTo4lBR:
    """Narrow-width-approximation H → 4ℓ BR matches PDG order of magnitude."""

    def test_h_to_4e_BR_in_pdg_ballpark(self):
        r = higgs_to_4l_BR("e", "e")
        # PDG H → 4e is ~3e-5
        assert 1e-5 < r["BR"] < 1e-4

    def test_h_to_2e2mu_BR(self):
        r = higgs_to_4l_BR("e", "mu")
        # PDG H → 2e2μ is ~6e-5 (factor 2 over 4e due to distinguishable Z pairs)
        assert 3e-5 < r["BR"] < 2e-4

    def test_mixed_flavor_combinatoric(self):
        """BR(2e2μ) should be 2× BR(4e) by the combinatoric factor."""
        r_4e   = higgs_to_4l_BR("e", "e")
        r_2e2m = higgs_to_4l_BR("e", "mu")
        assert abs(r_2e2m["BR"] / r_4e["BR"] - 2.0) < 0.01
