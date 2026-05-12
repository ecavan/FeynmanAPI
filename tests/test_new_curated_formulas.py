"""Smoke tests for the curated formulas added 2026-05-10.

Covers:
1. BSM Z' portal + scalar dark matter (5 entries × concrete flavours)
2. Concrete-flavour Z → ll̄ / νν̄ / qq̄ decays (11 entries)
3. Concrete-flavour H → ll̄ / qq̄ decays (5 entries)
4. Per-flavour QED ll → l'l' 1-loop VP (8 entries)
5. Per-flavour QCD qq̄ → gg / gg → qq̄ 1-loop (10 entries)

Numerical tolerances follow the engine convention:
- "exact" formula vs PDG: 5%
- 1-loop curated (no LoopTools eval): just confirm registration + LaTeX exists
"""
from __future__ import annotations

import math

import pytest


# ── BSM (Z′ + scalar DM) ─────────────────────────────────────────────────────

class TestBSMCurated:
    """BSM Z' portal + scalar dark matter."""

    @pytest.fixture(scope="class")
    def curated(self):
        from feynman_engine.physics.amplitude import _CURATED
        return {(p, t): a for (p, t), a in _CURATED.items() if t == "BSM"}

    def test_bsm_zp_decays_registered(self, curated):
        """Z′ → ll̄ for e/μ + Z′ → χχ̄ are registered."""
        assert ("Zp -> e+ e-", "BSM") in curated
        assert ("Zp -> mu+ mu-", "BSM") in curated
        assert ("Zp -> chi chi~", "BSM") in curated

    def test_bsm_scattering_registered(self, curated):
        """e+e- → χχ̄, μ+μ- → χχ̄, e+e- → μμ via Z' all registered."""
        assert ("e+ e- -> chi chi~", "BSM") in curated
        assert ("mu+ mu- -> chi chi~", "BSM") in curated
        assert ("e+ e- -> mu+ mu-", "BSM") in curated

    def test_bsm_dm_annihilation_registered(self, curated):
        """χχ̄ → e+e- and χχ̄ → μ+μ- both registered."""
        assert ("chi chi~ -> e+ e-", "BSM") in curated
        assert ("chi chi~ -> mu+ mu-", "BSM") in curated

    def test_zp_resonance_peak(self):
        """e+e- → Z' → μμ peaks on resonance and is sub-pb off-resonance."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        # Off-resonance (m_Zp = 1000 GeV by default)
        r_off = total_cross_section("e+ e- -> mu+ mu-", "BSM", sqrt_s=200.0)
        # On-resonance (m_Zp = 1000 GeV)
        r_on = total_cross_section("e+ e- -> mu+ mu-", "BSM", sqrt_s=1000.0)

        assert r_off["supported"] and r_on["supported"]
        # Resonance should give substantially larger σ
        assert r_on["sigma_pb"] > 100 * r_off["sigma_pb"], (
            f"Z' resonance peak too small: σ_on={r_on['sigma_pb']:.4e}, "
            f"σ_off={r_off['sigma_pb']:.4e}"
        )

    def test_dm_pair_below_threshold_blocked(self):
        """e+e- → χχ̄ below 2 m_χ = 200 GeV should be kinematically blocked."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e+ e- -> chi chi~", "BSM", sqrt_s=150.0)
        assert not r["supported"]
        assert "threshold" in r.get("error", "").lower()

    def test_dm_annihilation_crossing_factor(self):
        """σ(χχ̄ → ll̄) / σ(ll̄ → χχ̄) ≈ 4 × β_ratio (spin-sum vs spin-avg)."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        sqrt_s = 500.0
        r_ll = total_cross_section("e+ e- -> chi chi~", "BSM", sqrt_s=sqrt_s)
        r_dm = total_cross_section("chi chi~ -> e+ e-", "BSM", sqrt_s=sqrt_s)
        assert r_ll["supported"] and r_dm["supported"]

        ratio = r_dm["sigma_pb"] / r_ll["sigma_pb"]
        # Expected: 4 × (β_l/β_χ) ≈ 4 × (1.0 / 0.917) ≈ 4.36
        assert 3.0 < ratio < 5.5, (
            f"Crossing-symmetry factor off: {ratio:.3f}, expected ~4.36"
        )


# ── Concrete-flavour Z and H decays ──────────────────────────────────────────

class TestConcreteEWDecays:
    """Concrete-flavour Z → ff̄ and H → ff̄ partial widths vs PDG 2024."""

    @pytest.fixture(scope="class")
    def decay(self):
        """Helper: query the decay-width route through Python rather than HTTP."""
        from feynman_engine.api.routes import get_decay_width

        def _q(process: str, theory: str = "EW", order: str = "LO") -> dict:
            return get_decay_width(process=process, theory=theory, order=order)
        return _q

    @pytest.mark.parametrize("process, pdg_mev, tol", [
        ("Z -> e+ e-",        83.91, 0.05),  # PDG 2024
        ("Z -> mu+ mu-",      83.99, 0.05),
        ("Z -> tau+ tau-",    84.08, 0.05),
        ("Z -> nu_e nu_e~",   165.6, 0.05),
        ("Z -> nu_mu nu_mu~", 165.6, 0.05),
        ("Z -> u u~",         300.2, 0.07),
        ("Z -> d d~",         375.6, 0.05),
        ("Z -> b b~",         375.6, 0.05),
        ("Z -> c c~",         300.2, 0.07),
        ("Z -> s s~",         375.6, 0.05),
    ])
    def test_z_decay_vs_pdg(self, decay, process, pdg_mev, tol):
        r = decay(process)
        w = r.get("width_mev")
        assert w is not None, f"No width returned for {process}: {r}"
        rel = abs(w - pdg_mev) / pdg_mev
        assert rel < tol, (
            f"{process}: engine={w:.3f} MeV vs PDG {pdg_mev:.2f} MeV "
            f"(Δ={rel:.1%}, tol={tol:.0%})"
        )

    @pytest.mark.parametrize("process, pdg_mev, tol", [
        ("H -> tau+ tau-", 0.257,    0.05),  # PDG 2024
        ("H -> mu+ mu-",   0.000891, 0.10),  # 8.9e-4 MeV
        ("H -> b b~",      2.135,    0.05),  # LO; NLO K=1.13 brings to 2.41
        ("H -> c c~",      0.0945,   0.05),  # LO; NLO K=1.24 brings to 0.117
    ])
    def test_h_decay_lo_vs_textbook(self, decay, process, pdg_mev, tol):
        r = decay(process)
        w = r.get("width_lo_mev") or r.get("width_mev")
        assert w is not None, f"No width returned for {process}: {r}"
        rel = abs(w - pdg_mev) / pdg_mev
        assert rel < tol, (
            f"{process}: engine={w:.5g} MeV vs textbook LO {pdg_mev:.5g} MeV "
            f"(Δ={rel:.1%}, tol={tol:.0%})"
        )

    def test_h_to_bb_nlo_matches_pdg(self, decay):
        """H → bb̄ at NLO QCD K=1.13 should match PDG 2.41 MeV."""
        r = decay("H -> b b~", order="NLO")
        w = r.get("width_mev")
        assert w is not None
        rel = abs(w - 2.41) / 2.41
        assert rel < 0.05, f"H → bb NLO: engine={w:.3f} MeV vs PDG 2.41 MeV (Δ={rel:.1%})"


# ── Per-flavour QED 1-loop VP ────────────────────────────────────────────────

class TestQEDLoopFlavours:
    """Per-flavour 1-loop QED VP corrections."""

    def test_curated_loop_lookup_succeeds_for_every_flavour(self):
        """Every (initial, final) lepton pair has a curated 1-loop entry."""
        from feynman_engine.amplitudes.loop_curated import get_loop_curated_amplitude

        flavours = ["e", "mu", "tau"]
        registered_count = 0
        for i in flavours:
            for f in flavours:
                proc = f"{i}+ {i}- -> {f}+ {f}-"
                amp = get_loop_curated_amplitude(proc, "QED")
                if amp is not None:
                    registered_count += 1
                    assert amp.theory == "QED"
                    assert amp.backend == "curated-1loop"
        # Should cover at least 8 of the 9 (mu+mu-, tau+tau-, e+e-) × 3
        # combinations — e+e- → e+e- routes through the dedicated Bhabha entry.
        assert registered_count >= 8, (
            f"Expected >= 8 ll→l'l' 1-loop entries, got {registered_count}"
        )


# ── Per-flavour QCD 1-loop ───────────────────────────────────────────────────

class TestQCDLoopPerFlavour:
    """Per-flavour qq̄ → gg and gg → qq̄ 1-loop entries."""

    @pytest.mark.parametrize("quark", ["u", "d", "s", "c", "b"])
    def test_qqbar_to_gg_1loop_registered(self, quark):
        from feynman_engine.amplitudes.loop_curated import get_loop_curated_amplitude
        amp = get_loop_curated_amplitude(f"{quark} {quark}~ -> g g (1-loop full)", "QCD")
        assert amp is not None, f"q={quark} qq̄→gg 1-loop missing"
        assert amp.theory == "QCD"
        assert amp.backend == "curated-1loop"

    @pytest.mark.parametrize("quark", ["u", "d", "s", "c", "b"])
    def test_gg_to_qqbar_1loop_registered(self, quark):
        from feynman_engine.amplitudes.loop_curated import get_loop_curated_amplitude
        amp = get_loop_curated_amplitude(f"g g -> {quark} {quark}~ (1-loop full)", "QCD")
        assert amp is not None, f"q={quark} gg→qq̄ 1-loop missing"
        assert amp.theory == "QCD"
        assert amp.backend == "curated-1loop"


# ── Bug fixes verified ───────────────────────────────────────────────────────

class TestBugFixes20260510:
    """Regression coverage for the three bugs fixed on 2026-05-10."""

    def test_nlo_cross_section_returns_sigma_pb(self):
        """`order=NLO` cross-section must populate sigma_pb (was None)."""
        from feynman_engine.amplitudes.nlo_cross_section import nlo_cross_section
        r = nlo_cross_section("e+ e- -> mu+ mu-", "QED", sqrt_s=200.0)
        assert r["sigma_pb"] is not None
        assert r["sigma_lo_pb"] is not None
        assert r["sigma_nlo_pb"] is not None
        # σ_NLO ≈ σ_LO × K (K_QED ≈ 1.00174)
        assert abs(r["sigma_pb"] - r["sigma_lo_pb"] * r["k_factor"]) < 1e-9

    def test_pp_ww_now_uses_hpz(self):
        """`p p -> W+ W-` was BLOCKED (curated qq̄ formula t-channel-only).
        After the 2026-05-10 HPZ rewrite, the partonic σ̂ is the full SM
        tree-level result, so the hadronic process is APPROXIMATE — not BLOCKED.
        """
        from feynman_engine.physics.trust import classify, TrustLevel
        e = classify("p p -> W+ W-", "EW", "LO")
        assert e.trust_level == TrustLevel.APPROXIMATE, (
            "p p -> W+ W- should be APPROXIMATE since HPZ provides the full "
            "SM partonic σ̂ (97 % vs MG5 at √s=200 GeV)."
        )

    def test_h_to_bb_factor_of_two_fixed(self):
        """H → bb̄ at LO must be ~2.13 MeV (was 1.07 due to factor-of-2 trace bug)."""
        from feynman_engine.api.routes import get_decay_width
        r = get_decay_width(process="H -> b b~", theory="EW", order="LO")
        w = r["width_mev"]
        # PDG 2.41 / K_NLO 1.13 = 2.13 MeV at LO
        assert 2.0 < w < 2.3, (
            f"H → bb̄ LO width = {w} MeV, expected ~2.13 (factor-of-2 trace fix)"
        )
