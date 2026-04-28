"""Tests for PDF parametrization and hadronic cross-sections.

Validates:
- PDF sum rules (momentum, valence)
- PDF positivity and boundary behavior
- Parton luminosity ordering
- Drell-Yan cross-section (order of magnitude, channel fractions)
- Top pair cross-section (order of magnitude, gg dominance)
- NLO K-factors
- API endpoints
"""
from __future__ import annotations

import math
import pytest
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# PDF tests
# ---------------------------------------------------------------------------


class TestPDFSumRules:
    """Verify that the built-in PDF satisfies fundamental sum rules."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from feynman_engine.amplitudes.pdf import PDFSet
        self.pdf = PDFSet()
        self.Q0_sq = self.pdf.Q0_sq

    def test_momentum_sum_rule_q0(self):
        """∫₀¹ x·Σf dx = 1 at Q₀²."""
        flavors = [0, 1, -1, 2, -2, 3, -3]
        total = sum(
            quad(lambda x, f=f: self.pdf.xf(f, x, self.Q0_sq),
                 1e-6, 1 - 1e-6, limit=200)[0]
            for f in flavors
        )
        assert abs(total - 1.0) < 0.01, f"Momentum sum = {total}, expected 1.0"

    def test_momentum_sum_rule_mz(self):
        """∫₀¹ x·Σf dx = 1 at Q² = M_Z² (enforced by rescaling)."""
        Q2 = 91.2 ** 2
        flavors = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
        total = sum(
            quad(lambda x, f=f: self.pdf.xf(f, x, Q2),
                 1e-6, 1 - 1e-6, limit=200)[0]
            for f in flavors
        )
        assert abs(total - 1.0) < 0.05, f"Momentum sum = {total}"

    def test_valence_u_sum_rule(self):
        """∫₀¹ (u - ū) dx = 2 at Q₀²."""
        result, _ = quad(
            lambda x: self.pdf.f(2, x, self.Q0_sq) - self.pdf.f(-2, x, self.Q0_sq),
            1e-6, 1 - 1e-6, limit=200,
        )
        assert abs(result - 2.0) < 0.05, f"u-valence = {result}"

    def test_valence_d_sum_rule(self):
        """∫₀¹ (d - d̄) dx = 1 at Q₀²."""
        result, _ = quad(
            lambda x: self.pdf.f(1, x, self.Q0_sq) - self.pdf.f(-1, x, self.Q0_sq),
            1e-6, 1 - 1e-6, limit=200,
        )
        assert abs(result - 1.0) < 0.05, f"d-valence = {result}"


class TestPDFProperties:
    """Verify PDF shape, positivity, and boundary behavior."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from feynman_engine.amplitudes.pdf import PDFSet
        self.pdf = PDFSet()

    def test_positivity(self):
        """xf(x, Q²) ≥ 0 for all flavors, x, Q²."""
        for Q2 in [10.0, 100.0, 8000.0]:
            for x in [0.001, 0.01, 0.1, 0.3, 0.5, 0.9]:
                for flav in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                    val = self.pdf.xf(flav, x, Q2)
                    assert val >= 0, f"xf({flav}, {x}, {Q2}) = {val} < 0"

    def test_boundary_x0(self):
        """xf(0, Q²) = 0 (no singularity)."""
        for flav in [0, 2, -2, 1, 3]:
            assert self.pdf.xf(flav, 0.0, 100.0) == 0.0

    def test_boundary_x1(self):
        """xf(1, Q²) = 0 (kinematic limit)."""
        for flav in [0, 2, -2, 1, 3]:
            assert self.pdf.xf(flav, 1.0, 100.0) == 0.0

    def test_u_larger_than_d_at_moderate_x(self):
        """u(x) > d(x) at moderate x (more u valence quarks)."""
        Q2 = 100.0
        for x in [0.1, 0.2, 0.3]:
            xu = self.pdf.xf(2, x, Q2)
            xd = self.pdf.xf(1, x, Q2)
            assert xu > xd, f"xu={xu} <= xd={xd} at x={x}"

    def test_gluon_dominates_at_small_x(self):
        """Gluon PDF is larger than any quark at small x."""
        Q2 = 100.0
        for x in [0.001, 0.01]:
            xg = self.pdf.xf(0, x, Q2)
            xu = self.pdf.xf(2, x, Q2)
            assert xg > xu, f"xg={xg} <= xu={xu} at x={x}"

    def test_heavy_quark_threshold(self):
        """Charm/bottom PDFs are zero below their thresholds."""
        assert self.pdf.xf(4, 0.1, 1.0) == 0.0  # Q² = 1 < 4m_c²
        assert self.pdf.xf(5, 0.1, 10.0) == 0.0  # Q² = 10 < 4m_b²
        assert self.pdf.xf(4, 0.1, 100.0) > 0  # Q² = 100 > 4m_c²
        assert self.pdf.xf(5, 0.1, 1000.0) > 0  # Q² = 1000 > 4m_b²


class TestPartonLuminosity:
    """Verify parton luminosity ordering and properties."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from feynman_engine.amplitudes.pdf import PDFSet, parton_luminosity
        self.pdf = PDFSet()
        self.parton_luminosity = parton_luminosity
        # Z-pole at 14 TeV
        self.tau = 91.0 ** 2 / 14000.0 ** 2
        self.mu2 = 91.0 ** 2

    def test_luminosity_positive(self):
        """Parton luminosity is positive."""
        for fa, fb in [(2, -2), (1, -1), (0, 0)]:
            L = self.parton_luminosity(self.pdf, fa, fb, self.tau, self.mu2)
            assert L > 0, f"L({fa},{fb}) = {L} <= 0"

    def test_luminosity_uu_dominates_dd(self):
        """L_uū > L_dd̄ (u quarks carry more momentum)."""
        L_uu = self.parton_luminosity(self.pdf, 2, -2, self.tau, self.mu2)
        L_dd = self.parton_luminosity(self.pdf, 1, -1, self.tau, self.mu2)
        assert L_uu > L_dd, f"L_uu={L_uu} <= L_dd={L_dd}"

    def test_luminosity_decreasing_with_tau(self):
        """Luminosity decreases as τ increases (harder to find high-x partons)."""
        tau_low = 0.001
        tau_high = 0.01
        L_low = self.parton_luminosity(self.pdf, 2, -2, tau_low, self.mu2)
        L_high = self.parton_luminosity(self.pdf, 2, -2, tau_high, self.mu2)
        assert L_low > L_high, f"L(τ={tau_low})={L_low} <= L(τ={tau_high})={L_high}"

    def test_gluon_luminosity_large(self):
        """Gluon-gluon luminosity is large at small τ."""
        L_gg = self.parton_luminosity(self.pdf, 0, 0, self.tau, self.mu2)
        L_uu = self.parton_luminosity(self.pdf, 2, -2, self.tau, self.mu2)
        assert L_gg > L_uu, f"L_gg={L_gg} should exceed L_uu={L_uu}"


# ---------------------------------------------------------------------------
# Hadronic cross-section tests
# ---------------------------------------------------------------------------


class TestDrellYan:
    """Drell-Yan pp → l+l- cross-section tests."""

    def test_drell_yan_order_of_magnitude(self):
        """σ(pp → μ+μ-, 14 TeV) is O(100 pb) to O(nb)."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section(
            "p p -> mu+ mu-", sqrt_s=14000.0, order="LO",
            m_ll_min=60.0, m_ll_max=120.0,
        )
        assert result["supported"]
        sigma = result["sigma_pb"]
        # With simplified PDFs, expect ~100-2000 pb
        assert 50 < sigma < 5000, f"sigma = {sigma} pb"

    def test_drell_yan_u_channel_largest(self):
        """u-quark channel contributes the most (charge² × luminosity)."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section(
            "p p -> mu+ mu-", sqrt_s=14000.0, order="LO",
        )
        channels = result["channels"]
        u_sigma = next(ch["sigma_pb"] for ch in channels if "u u~" in ch["partonic"])
        # u-channel should be the largest
        assert u_sigma == max(ch["sigma_pb"] for ch in channels)

    def test_drell_yan_energy_scaling(self):
        """Cross-section increases with √s."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r7 = hadronic_cross_section("p p -> mu+ mu-", sqrt_s=7000.0, order="LO")
        r14 = hadronic_cross_section("p p -> mu+ mu-", sqrt_s=14000.0, order="LO")
        assert r14["sigma_pb"] > r7["sigma_pb"]

    def test_drell_yan_nlo_kfactor(self):
        """NLO K-factor > 1 for Drell-Yan (running α_em increases with Q²)."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section(
            "p p -> mu+ mu-", sqrt_s=14000.0, order="NLO",
        )
        assert result["k_factor"] > 1.0
        assert result["k_factor"] < 2.0  # shouldn't be huge

    def test_drell_yan_electron_channel(self):
        """pp → e+e- also works."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section(
            "p p -> e+ e-", sqrt_s=14000.0, order="LO",
        )
        assert result["supported"]
        assert result["sigma_pb"] > 0


class TestTopPairs:
    """Top pair pp → tt̄ cross-section tests."""

    def test_top_pair_order_of_magnitude(self):
        """σ(pp → tt̄, 13 TeV) is O(100-2000 pb) at LO."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section(
            "p p -> t t~", sqrt_s=13000.0, theory="QCD", order="LO",
        )
        assert result["supported"]
        sigma = result["sigma_pb"]
        assert 50 < sigma < 5000, f"sigma = {sigma} pb"

    def test_top_pair_gg_dominant(self):
        """gg → tt̄ channel dominates at LHC energies."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section(
            "p p -> t t~", sqrt_s=13000.0, theory="QCD", order="LO",
        )
        channels = result["channels"]
        gg_ch = next(ch for ch in channels if "g g" in ch["partonic"])
        assert gg_ch["fraction"] > 0.5, f"gg fraction = {gg_ch['fraction']}"

    def test_top_pair_energy_scaling(self):
        """Cross-section increases from 8 TeV to 14 TeV."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r8 = hadronic_cross_section("p p -> t t~", sqrt_s=8000.0, theory="QCD")
        r14 = hadronic_cross_section("p p -> t t~", sqrt_s=14000.0, theory="QCD")
        assert r14["sigma_pb"] > r8["sigma_pb"]

    def test_top_pair_below_threshold(self):
        """Below threshold √s < 2m_t, σ should be ~0."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section(
            "p p -> t t~", sqrt_s=300.0, theory="QCD",
        )
        # At pp √s = 300 GeV, √ŝ can still exceed threshold
        # but the luminosity at τ = (345)²/(300)² > 1 means no phase space
        # Actually τ_min > 1 so it should fail
        assert result["sigma_pb"] == 0 or not result["supported"]


class TestHadronicGeneral:
    """General hadronic cross-section tests."""

    def test_unsupported_process(self):
        """Unsupported process returns error.

        Use pp → HH (di-Higgs) which has no curated partonic amplitude
        for any (a, b) parton pair.  pp → W+W- IS now supported via the
        new qq̄→W+W- curated amplitude.
        """
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section("p p -> H H", sqrt_s=14000.0, theory="EW")
        assert not result["supported"]

    def test_non_pp_rejected(self):
        """Non-pp process is rejected."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section("e+ e- -> mu+ mu-", sqrt_s=91.0)
        assert not result["supported"]

    def test_result_structure(self):
        """Result dict has all expected keys."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section("p p -> mu+ mu-", sqrt_s=14000.0)
        assert "sigma_pb" in result
        assert "channels" in result
        assert "pdf" in result
        assert "order" in result
        assert result["hadronic"] is True


# ---------------------------------------------------------------------------
# Benchmark σ assertions — the kind of test that would have caught the
# ZZ sign bug, the missing photon-quark Q_f, and the missing τ_H factor.
# Each test compares a specific σ value to a published reference number
# with a tolerance band that's tight enough to flag wrong-by-construction
# results but loose enough to allow LO numerical precision.
# ---------------------------------------------------------------------------


class TestBenchmarkCrossSections:
    """Anchor σ to specific published numbers, not just supported=True."""

    def test_ee_to_mumu_at_91GeV(self):
        """σ(e+e-→μμ at √s=91 GeV) matches analytic 4πα²/(3s) within 1%."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        import math
        r = total_cross_section("e+ e- -> mu+ mu-", "QED", sqrt_s=91.0)
        sigma_analytic = 4 * math.pi * (1 / 137.036) ** 2 / (3 * 91.0 ** 2) * 3.8938e8
        assert abs(r["sigma_pb"] - sigma_analytic) / sigma_analytic < 0.01

    def test_ee_to_mumu_at_10GeV(self):
        """σ(e+e-→μμ at √s=10 GeV) ≈ 869 pb (analytic)."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e+ e- -> mu+ mu-", "QED", sqrt_s=10.0)
        assert 850 < r["sigma_pb"] < 880

    def test_ee_to_ZZ_at_200GeV_physical(self):
        """σ(e+e-→ZZ at √s=200 GeV) ≈ 1.5 pb LO (physical convention with 1/2!)."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e+ e- -> Z Z", "EW", sqrt_s=200.0)
        assert r["identical_particle_factor"] == 2  # 2 identical Z's
        # LO theoretical without EW corrections ≈ 1.5 pb; ours ≈ 1.39 pb (~8% LOW)
        assert 0.7 < r["sigma_pb"] < 2.5, (
            f"σ(e+e-→ZZ at 200 GeV) = {r['sigma_pb']} — should be 0.7-2.5 pb"
        )

    def test_ee_to_gammagamma_identical_factor(self):
        """e+e-→γγ has the 2-identical-photon factor, halving σ vs raw integration."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e+ e- -> gamma gamma", "QED", sqrt_s=10.0)
        assert r["identical_particle_factor"] == 2

    def test_ee_to_ee_distinguishable(self):
        """e+e-→e+e- (Bhabha) — final-state e+ and e- are distinguishable."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e+ e- -> e+ e-", "QED", sqrt_s=10.0)
        assert r["identical_particle_factor"] == 1

    def test_qcd_qqbar_to_gg_identical(self):
        """qq̄→gg has the 2-identical-gluon factor."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("u u~ -> g g", "QCD", sqrt_s=100.0)
        if r.get("supported"):
            assert r["identical_particle_factor"] == 2

    def test_pp_drell_yan_at_14TeV(self):
        """σ(pp→μ+μ-, M_ll∈[60,120]) at 14 TeV (engine, built-in PDF)."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section("p p -> mu+ mu-", sqrt_s=14000.0, order="LO")
        # Built-in PDF gives ~800 pb; LHC Run 2 measured ~2000 pb (2-3× higher with CT18LO).
        # Order-of-magnitude check.
        assert 200 < r["sigma_pb"] < 5000

    def test_pp_to_tt_at_13TeV(self):
        """σ(pp→tt̄) at 13 TeV — should be in the LHC-LO ballpark."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section("p p -> t t~", sqrt_s=13000.0, theory="QCD", order="LO")
        # LHC LO ~700 pb; engine's built-in PDF ~1600 pb (2× HIGH).
        # Allow a wide band reflecting built-in-PDF systematic.
        assert 200 < r["sigma_pb"] < 3000

    def test_pp_to_H_at_13TeV(self):
        """σ(pp→H, ggF) at 13 TeV is in LHC-LO ballpark (~16 pb)."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section("p p -> H", sqrt_s=13000.0, theory="EW")
        assert r["supported"]
        # LHC LO published ~16 pb; engine ~12 pb (within 25%).
        assert 5 < r["sigma_pb"] < 30

    def test_ee_to_mumu_EW_at_Z_pole(self):
        """σ(e+e- → μμ) at the Z peak should be ~2 nb (LEP measurement)."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e+ e- -> mu+ mu-", "EW", sqrt_s=91.2)
        sigma_pb = r["sigma_pb"]
        # LEP measured σ_peak ≈ 1.5-2 nb (depending on cuts).
        # Engine should be in [1, 3] nb.
        assert 1000 < sigma_pb < 3000, f"σ at Z pole = {sigma_pb} pb (expect 1-3 nb)"

    def test_ee_to_mumu_EW_above_Z_pole(self):
        """σ(e+e- → μμ) at √s=200 GeV should be ~2 pb (mostly QED-like)."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("e+ e- -> mu+ mu-", "EW", sqrt_s=200.0)
        # LEP-2 measurement ~ 2 pb.  Engine: 2.6 pb.
        assert 1.0 < r["sigma_pb"] < 5.0

    def test_no_silent_zero_for_EW_processes(self):
        """EW Z-mediated processes must NOT silently return σ=0 (regression test).

        Was a real bug: missing g_Z_X coupling defaults caused the integrator
        to silently catch float() errors and return σ=0 with supported=True.
        """
        from feynman_engine.amplitudes.cross_section import total_cross_section
        for proc, sqrts in [
            ("e+ e- -> mu+ mu-", 91.2),
            ("u u~ -> e+ e-", 200.0),
            ("d d~ -> mu+ mu-", 200.0),
        ]:
            r = total_cross_section(proc, "EW", sqrt_s=sqrts)
            if r.get("supported"):
                assert r["sigma_pb"] > 0, f"{proc} returned supported=True with σ=0"

    def test_pp_to_ZZ_at_13TeV(self):
        """σ(pp→ZZ) at 13 TeV (qq̄ initiated, LO ≈ 5-7 pb after 1/2!)."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section("p p -> Z Z", sqrt_s=13000.0, theory="EW")
        assert r["supported"]
        # LHC LO qq̄→ZZ ~ 5-8 pb (physical convention).
        assert 1.0 < r["sigma_pb"] < 20.0

    def test_pp_to_WW_at_13TeV(self):
        """σ(pp→W+W-) at 13 TeV using new qq̄→W+W- curated amplitude."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section("p p -> W+ W-", sqrt_s=13000.0, theory="EW")
        assert r["supported"]
        # LHC LO qq̄→W+W- ~ 30-60 pb. Built-in PDF + t-channel-only
        # approximation gives ~15 pb (factor of ~3 LOW). Allow wide band.
        assert 5.0 < r["sigma_pb"] < 100.0
        assert r["n_channels_evaluated"] >= 4

    def test_qqbar_to_ww_partonic(self):
        """σ̂(uū→W+W-) at √ŝ=500 GeV is positive."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("u u~ -> W+ W-", "EW", sqrt_s=500.0)
        assert r["supported"]
        assert r["sigma_pb"] > 0
        # No identical-particle factor for W+W- (distinguishable charges).
        assert r["identical_particle_factor"] == 1

    def test_udbar_to_wgamma_partonic(self):
        """σ̂(ud̄→W+γ) at √ŝ=300 GeV is positive."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("u d~ -> W+ gamma", "EW", sqrt_s=300.0)
        assert r["supported"]
        assert r["sigma_pb"] > 0

    def test_qqbar_to_tt_uses_massive_formula(self):
        """qq̄→tt̄ uses the massive Combridge formula, not the wrong massless one.

        Regression test: the previous massless gg→qq̄ formula applied to top
        gave σ̂(uū→tt̄, 500 GeV) ≈ 145 pb (5× HIGH).  The massive Combridge
        formula gives ~9 pb (textbook).
        """
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("u u~ -> t t~", "QCD", sqrt_s=500.0)
        assert r["supported"]
        # Textbook ~10 pb at 500 GeV, allow factor of 2 either way
        assert 4 < r["sigma_pb"] < 25

    def test_gg_to_tt_uses_massive_formula(self):
        """gg→tt̄ uses the massive Combridge formula, not the wrong massless one."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("g g -> t t~", "QCD", sqrt_s=500.0)
        assert r["supported"]
        # Textbook gg→tt̄ at 500 GeV ≈ 25-30 pb
        assert 10 < r["sigma_pb"] < 100

    def test_vbf_higgs_at_13TeV(self):
        """VBF Higgs (pp → H j j) at 13 TeV anchored to LHC HWG YR4 ≈ 3.78 pb."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section("p p -> H j j", sqrt_s=13000.0, theory="EW")
        assert r["supported"]
        # Anchored to LHC HWG YR4 LO at 13 TeV
        assert 3.5 < r["sigma_pb"] < 4.2
        assert r["method"] == "vbf-calibrated-ref"

    def test_vbf_higgs_energy_dependence(self):
        """VBF σ at 14 TeV should be larger than at 7 TeV."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r7 = hadronic_cross_section("p p -> H j j", sqrt_s=7000.0, theory="EW")
        r14 = hadronic_cross_section("p p -> H j j", sqrt_s=14000.0, theory="EW")
        assert r14["sigma_pb"] > r7["sigma_pb"]
        # 7 TeV: ~1.2 pb, 14 TeV: ~4.4 pb
        assert 1.0 < r7["sigma_pb"] < 1.5
        assert 4.0 < r14["sigma_pb"] < 5.0


class TestTrustSystem:
    """Trust-level enforcement: validated returns σ; blocked returns 422."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from fastapi.testclient import TestClient
        from feynman_engine.api.app import app
        self.client = TestClient(app)

    def test_validated_process_includes_trust_metadata(self):
        """A validated σ result has trust_level='validated' + reference."""
        r = self.client.get("/api/amplitude/cross-section", params={
            "process": "e+ e- -> mu+ mu-", "theory": "QED", "sqrt_s": 91,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["trust_level"] == "validated"
        assert d["trust_reference"]

    def test_approximate_process_includes_caveat(self):
        """An approximate σ result has trust_level='approximate' + caveat."""
        r = self.client.get("/api/amplitude/cross-section", params={
            "process": "p p -> mu+ mu-", "theory": "EW", "sqrt_s": 13000,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["trust_level"] == "approximate"
        assert "accuracy_caveat" in d

    def test_blocked_process_returns_422(self):
        """A blocked process returns 422 with a structured detail."""
        r = self.client.get("/api/amplitude/cross-section", params={
            "process": "p p -> W+ W-", "theory": "EW", "sqrt_s": 13000,
        })
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["trust_level"] == "blocked"
        assert detail["block_reason"]
        assert detail["workaround"]

    def test_blocked_partonic_qq_to_ll(self):
        """Standalone partonic q q~ → l+l- in EW is blocked (V-A approx)."""
        r = self.client.get("/api/amplitude/cross-section", params={
            "process": "u u~ -> e+ e-", "theory": "EW", "sqrt_s": 200,
        })
        assert r.status_code == 422

    def test_nlo_uses_tabulated_kfactor(self):
        """pp → H NLO returns σ_LO × tabulated K (1.7), not running-coupling."""
        r_lo = self.client.get("/api/amplitude/hadronic-cross-section", params={
            "process": "p p -> H", "sqrt_s": 13000, "order": "LO",
        })
        r_nlo = self.client.get("/api/amplitude/hadronic-cross-section", params={
            "process": "p p -> H", "sqrt_s": 13000, "order": "NLO",
        })
        assert r_lo.status_code == 200 and r_nlo.status_code == 200
        K = r_nlo.json()["sigma_pb"] / r_lo.json()["sigma_pb"]
        # K_NLO for ggH ≈ 1.7 from tabulated
        assert 1.6 < K < 1.8


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestHadronicAPI:
    """Test the hadronic cross-section API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from fastapi.testclient import TestClient
        from feynman_engine.api.routes import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_hadronic_endpoint_drell_yan(self):
        """GET /amplitude/hadronic-cross-section for Drell-Yan returns valid JSON."""
        resp = self.client.get(
            "/api/amplitude/hadronic-cross-section",
            params={
                "process": "p p -> mu+ mu-",
                "sqrt_s": 14000,
                "order": "LO",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"]
        assert data["sigma_pb"] > 0
        assert "channels" in data

    def test_hadronic_endpoint_top_pairs(self):
        """GET /amplitude/hadronic-cross-section for top pairs."""
        resp = self.client.get(
            "/api/amplitude/hadronic-cross-section",
            params={
                "process": "p p -> t t~",
                "sqrt_s": 13000,
                "theory": "QCD",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"]
        assert data["sigma_pb"] > 0

    def test_cross_section_endpoint_redirects_pp(self):
        """GET /amplitude/cross-section with 'p p ->' redirects to hadronic."""
        resp = self.client.get(
            "/api/amplitude/cross-section",
            params={
                "process": "p p -> mu+ mu-",
                "sqrt_s": 14000,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("hadronic") is True
        assert data["sigma_pb"] > 0

    def test_pp_to_ZZ_via_curated(self):
        """pp → ZZ now reachable via the (numerical) qq̄→ZZ curated amplitudes."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section("p p -> Z Z", sqrt_s=13000.0, theory="EW")
        assert r["supported"]
        # LHC LO σ(qq̄→ZZ) is ~8-12 pb; engine convention (no 1/2! for identical) doubles.
        assert 1.0 < r["sigma_pb"] < 30.0
        assert r["n_channels_evaluated"] >= 4

    def test_ee_to_ZZ_cross_section_shape(self):
        """e+e-→ZZ has the textbook rise-peak-fall shape across √s."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        sigmas = {}
        for sqrts in [185, 200, 250, 500, 1000]:
            r = total_cross_section("e+ e- -> Z Z", "EW", sqrt_s=float(sqrts))
            assert r["supported"]
            sigmas[sqrts] = r["sigma_pb"]
        # Threshold rise: 185 < 200
        assert sigmas[185] < sigmas[200]
        # Peak somewhere around 250 GeV: 200 < 250 > 500
        assert sigmas[200] < sigmas[250]
        assert sigmas[250] > sigmas[500]
        # High-energy fall: 500 > 1000
        assert sigmas[500] > sigmas[1000]
        # All values positive (the previous bug had σ clamped to 0)
        for s, v in sigmas.items():
            assert v > 0, f"σ at √s={s} was {v} (≤0 indicates regression)"
        # Physical convention (1/2! applied for identical Z's).
        # LEP-2 LO theoretical ~1.5 pb (without EW corrections);
        # measured ~0.7-1.0 pb (with NLO EW corrections).
        # Allow factor-of-2 either way for tree-level numerical precision.
        assert 0.7 < sigmas[200] < 3.0, f"σ(e+e-→ZZ, 200 GeV) = {sigmas[200]} pb"

    def test_pp_to_H_via_gg_fusion(self):
        """pp → H specialized path returns σ in the LHC-LO ballpark.

        Built-in PDF: σ ≈ 12 pb (LHC LO ~16, within 25%).
        CT18LO (LHAPDF): σ ≈ 22-24 pb (LHC LO with this PDF/scale ~18-22).
        Both are in the right order of magnitude.  The pdf_warning field is
        only populated for the built-in PDF.
        """
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        from feynman_engine.amplitudes.pdf import _lhapdf_available

        # Force built-in PDF for predictable warning behaviour
        r_b = hadronic_cross_section(
            "p p -> H", sqrt_s=13000.0, theory="EW", pdf_name="LO-simple",
        )
        assert r_b["supported"]
        assert r_b["method"] == "ggH-fusion-heavy-top-NWA"
        assert 5.0 < r_b["sigma_pb"] < 30.0
        assert r_b.get("pdf_warning") is not None  # built-in always warns

        # Auto path (uses LHAPDF if available)
        r_auto = hadronic_cross_section("p p -> H", sqrt_s=13000.0, theory="EW")
        assert r_auto["supported"]
        # Allow wider band (5-50 pb) — LHAPDF gives ~22 pb, built-in ~12 pb.
        assert 5.0 < r_auto["sigma_pb"] < 50.0

        # Higgs production rises with √s (both PDFs)
        r_low = hadronic_cross_section(
            "p p -> H", sqrt_s=7000.0, theory="EW", pdf_name="LO-simple",
        )
        assert r_b["sigma_pb"] > r_low["sigma_pb"]

    def test_hadronic_endpoint_unsupported(self):
        """Unsupported hadronic process returns 404.

        Use pp → HH (di-Higgs) which has no curated partonic amplitude.
        """
        resp = self.client.get(
            "/api/amplitude/hadronic-cross-section",
            params={
                "process": "p p -> H H",
                "sqrt_s": 14000,
                "theory": "EW",
            },
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# LHAPDF backend tests (skip-if-not-installed)
# ---------------------------------------------------------------------------


class TestLHAPDFBackend:
    """LHAPDFSet wrapper and get_pdf factory.

    Skipped when the LHAPDF Python bindings are not installed; the goal here
    is to validate the public surface (constructor, xf/f signatures, fallback)
    rather than re-test LHAPDF itself.
    """

    def test_factory_falls_back_when_lhapdf_missing(self):
        """get_pdf('auto') returns the built-in PDF when LHAPDF is unavailable."""
        from feynman_engine.amplitudes.pdf import (
            _lhapdf_available, get_pdf, PDFSet, LHAPDFSet,
        )
        pdf = get_pdf("auto")
        if _lhapdf_available():
            # LHAPDF installed → should prefer it (or fall back if CT18LO missing)
            assert isinstance(pdf, (LHAPDFSet, PDFSet))
        else:
            assert isinstance(pdf, PDFSet)
            assert pdf.backend == "builtin"

    def test_factory_explicit_lo_simple(self):
        """get_pdf('LO-simple') always returns the built-in regardless of LHAPDF."""
        from feynman_engine.amplitudes.pdf import get_pdf, PDFSet
        pdf = get_pdf("LO-simple")
        assert isinstance(pdf, PDFSet)
        assert pdf.backend == "builtin"

    def test_factory_unknown_lhapdf_raises_when_no_bindings(self):
        """Asking for an LHAPDF set when the bindings aren't installed raises ImportError."""
        from feynman_engine.amplitudes.pdf import _lhapdf_available, get_pdf
        if _lhapdf_available():
            pytest.skip("LHAPDF installed; this test only exercises the missing-bindings path")
        with pytest.raises(ImportError):
            get_pdf("CT18NLO")

    def test_lhapdf_interface_when_available(self):
        """When LHAPDF bindings exist, the wrapper exposes xf/f and matches the contract."""
        from feynman_engine.amplitudes.pdf import _lhapdf_available, LHAPDFSet
        if not _lhapdf_available():
            pytest.skip("LHAPDF bindings not installed")
        try:
            lh = LHAPDFSet("CT18LO")
        except RuntimeError:
            pytest.skip("CT18LO not installed locally")
        assert lh.backend == "lhapdf"
        # Sum rule: ∫ x·Σ_f f(x, Q²) dx ≈ 1 within a few percent
        # (LHAPDF grids carry the actual sum rules of the published set).
        flavors = [21, 1, -1, 2, -2, 3, -3, 4, -4]
        from scipy.integrate import quad
        total, _ = quad(
            lambda x: sum(lh.xf(f, x, 100.0) for f in flavors),
            1e-5, 1 - 1e-5, limit=300, epsrel=1e-3,
        )
        assert 0.9 < total < 1.1, f"LHAPDF momentum sum = {total}"


# ---------------------------------------------------------------------------
# Generic parton-enumeration path
# ---------------------------------------------------------------------------


class TestGenericHadronicEnumeration:
    """Hadronic cross-section by enumerating partonic (a, b) channels."""

    def test_pp_to_diphoton_basic(self):
        """pp → γγ runs through the generic path and returns a sensible σ."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section(
            "p p -> gamma gamma",
            sqrt_s=13000.0,
            min_partonic_cm=50.0,
        )
        assert r["supported"]
        assert r["method"] == "generic-parton-enumeration"
        # Born qq̄→γγ at LHC with M_γγ > 50 GeV is O(100–2000 pb) before
        # NLO/gg-box contributions; allow a wide order-of-magnitude band.
        assert 30.0 < r["sigma_pb"] < 5000.0
        assert r["n_channels_evaluated"] >= 4
        # u-channel must dominate over d-channel by ~|Q_u/Q_d|^4 ratio (~16x).
        chans = sorted(r["channels"], key=lambda c: -c["sigma_pb"])
        u_channel = chans[0]["sigma_pb"]
        d_channel = next(
            c["sigma_pb"] for c in chans
            if c["partonic"].split()[0] in ("d", "d~")
            and c["partonic"].split()[1] in ("d", "d~")
        )
        assert u_channel > d_channel * 5, (
            f"u-channel ({u_channel} pb) should dominate d-channel ({d_channel} pb) "
            f"after Q_f^4 weighting"
        )

    def test_partonic_charge_scaling(self):
        """σ̂(uū → γγ) / σ̂(dd̄ → γγ) ≈ 16 at fixed √ŝ (charge ratio (2/3)⁴/(1/3)⁴)."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        ru = total_cross_section("u u~ -> gamma gamma", "QCDQED", sqrt_s=200.0)
        rd = total_cross_section("d d~ -> gamma gamma", "QCDQED", sqrt_s=200.0)
        assert ru["supported"] and rd["supported"]
        ratio = ru["sigma_pb"] / rd["sigma_pb"]
        assert 14.0 < ratio < 18.0, f"u/d partonic ratio = {ratio}, expected ≈ 16"

    def test_generic_skip_unreachable(self):
        """Generic path never returns supported=True with sigma=0 for unreachable processes."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        # An obviously bogus final state — no parton channel can reach it.
        r = hadronic_cross_section("p p -> chi chi~", sqrt_s=13000.0, theory="QED")
        # Either supported=False with a clear error, or supported=True with σ>0.
        # The contract is specifically "never silently return 0 with supported=True".
        if r.get("supported"):
            assert r["sigma_pb"] > 0
        else:
            assert "error" in r

    def test_generic_default_cut_applied(self):
        """Massless final states get the default 50 GeV partonic cut applied."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section(
            "p p -> gamma gamma", sqrt_s=13000.0,
        )
        if r.get("supported") and r.get("method") == "generic-parton-enumeration":
            assert r["min_partonic_sqrts_gev"] == pytest.approx(50.0, rel=1e-6)
            assert r["partonic_cut_reason"] == "default-massless-cut"

    def test_generic_user_cut_honored(self):
        """A higher user cut yields a smaller σ (less phase space)."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r_low = hadronic_cross_section(
            "p p -> gamma gamma", sqrt_s=13000.0, min_partonic_cm=50.0,
        )
        r_high = hadronic_cross_section(
            "p p -> gamma gamma", sqrt_s=13000.0, min_partonic_cm=200.0,
        )
        assert r_low["supported"] and r_high["supported"]
        assert r_low["sigma_pb"] > r_high["sigma_pb"]

    def test_dy_uses_specialized_path(self):
        """pp → l⁺l⁻ keeps using the analytic γ+Z method, not the generic path."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section("p p -> mu+ mu-", sqrt_s=13000.0)
        assert r["supported"]
        assert r["method"] == "drell-yan-analytic-gamma-Z"

    def test_top_pair_uses_specialized_path(self):
        """pp → tt̄ keeps using the QCD-grid method, not the generic path."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section("p p -> t t~", sqrt_s=13000.0, theory="QCD")
        assert r["supported"]
        assert r["method"] == "qcd-partonic-grid"

    def test_pdf_label_includes_backend(self):
        """The result reports both the backend and set name."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section(
            "p p -> mu+ mu-", sqrt_s=13000.0, pdf_name="LO-simple",
        )
        assert ":" in r["pdf"]  # "builtin:LO-simple"
        assert r["pdf"].startswith("builtin:")
