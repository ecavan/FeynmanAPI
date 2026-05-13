"""Tests for τ resonance/hadronic mode lookups (PDG-tabulated)."""
from __future__ import annotations

import pytest

from feynman_engine.amplitudes.three_body_decays import (
    tau_resonance_width, tau_branching_summary,
)


class TestTauResonancePDG:
    """PDG 2024 BRs round-trip from BR × Γ_τ back to BR."""

    def test_tau_to_pi_nu_BR(self):
        r = tau_resonance_width("tau- -> pi- nu_tau")
        assert r["BR"] == pytest.approx(0.1082, abs=1e-4)
        # Γ_partial = BR × Γ_τ; Γ_τ ≈ 2.27e-12 GeV → Γ ≈ 2.46e-13 GeV
        assert 2.2e-13 < r["width_gev"] < 2.7e-13

    def test_tau_to_rho_dominant_2pi_BR(self):
        r = tau_resonance_width("tau- -> pi- pi0 nu_tau")
        assert r["BR"] == pytest.approx(0.2549, abs=1e-4)

    def test_unknown_channel_returns_none(self):
        r = tau_resonance_width("tau- -> e- nu_e~ nu_tau")
        assert r is None     # leptonic mode handled by Sargent, not resonance lookup


class TestTauBranchingSummary:
    """Verify the Sargent-vs-PDG gap is exposed correctly."""

    def test_sargent_inclusive_below_pdg_total_hadronic(self):
        s = tau_branching_summary()
        # Sargent inclusive should be 50-58%; PDG total hadronic ~65%; gap ~7-12%
        assert 0.45 < s["sargent_inclusive_BR"] < 0.60
        assert s["pdg_total_hadronic_BR"] == pytest.approx(0.6479, abs=1e-4)
        gap = s["pdg_total_hadronic_BR"] - s["sargent_inclusive_BR"]
        assert 0.05 < gap < 0.15

    def test_leptonic_plus_hadronic_sums_close_to_unity(self):
        s = tau_branching_summary()
        # 2 × BR(τ→ℓν̄ν) + BR(τ→hadrons) ≈ 1
        total = 2 * s["leptonic_BR_each"] + s["pdg_total_hadronic_BR"]
        assert 0.99 < total < 1.01

    def test_resonance_table_sums_close_to_total_hadronic(self):
        s = tau_branching_summary()
        # The tabulated resonance modes should cover most of the hadronic BR
        # (PDG: ~64% total hadronic, tabulated here ~57%; the remaining ~7%
        # is multi-meson high-mass channels not in our table).
        assert s["pdg_tabulated_resonance_BR"] > 0.50
        assert s["pdg_tabulated_resonance_BR"] <= s["pdg_total_hadronic_BR"]
