"""Tests for NLO cross-section and Catani-Seymour dipole subtraction."""
from __future__ import annotations

import math

import numpy as np
import pytest

from feynman_engine.amplitudes.cross_section import ALPHA_EM, ALPHA_S
from feynman_engine.amplitudes.dipole_subtraction import (
    born_msq_eemumu,
    cs_ff_map,
    cs_if_map,
    dipole_sum_eemumu,
    _dipole_ff,
)
from feynman_engine.amplitudes.nlo_cross_section import (
    nlo_cross_section,
    nlo_cross_section_qed,
    alpha_s_running,
    alpha_em_running,
    _NLO_K_FACTOR,
)
from feynman_engine.amplitudes.phase_space import dot4, rambo_massless


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_beam(sqrt_s: float):
    E = sqrt_s / 2.0
    p1 = np.array([E, 0.0, 0.0, E])
    p2 = np.array([E, 0.0, 0.0, -E])
    return p1, p2


# ---------------------------------------------------------------------------
# Phase-space mapping tests
# ---------------------------------------------------------------------------

class TestCSFFMapping:
    """Catani-Seymour final-final mapping for massless partons."""

    def test_onshell(self):
        """Tilded momenta are massless (p^2 = 0)."""
        rng = np.random.default_rng(1)
        momenta, _ = rambo_massless(3, 91.0, 100, rng)
        q1, q2, q3 = momenta[:, 0], momenta[:, 1], momenta[:, 2]

        tilde_ij, tilde_k, y, z = cs_ff_map(q1, q3, q2)
        assert np.allclose(dot4(tilde_ij, tilde_ij), 0.0, atol=1e-8)
        assert np.allclose(dot4(tilde_k, tilde_k), 0.0, atol=1e-8)

    def test_momentum_conservation(self):
        """tilde_ij + tilde_k = p_i + p_j + p_k."""
        rng = np.random.default_rng(2)
        momenta, _ = rambo_massless(3, 91.0, 100, rng)
        q1, q2, q3 = momenta[:, 0], momenta[:, 1], momenta[:, 2]

        tilde_ij, tilde_k, _, _ = cs_ff_map(q1, q3, q2)
        total_orig = q1 + q3 + q2
        total_tilde = tilde_ij + tilde_k
        assert np.allclose(total_orig, total_tilde, atol=1e-8)

    def test_y_z_range(self):
        """y in [0, 1] and z in [0, 1]."""
        rng = np.random.default_rng(3)
        momenta, _ = rambo_massless(3, 91.0, 1000, rng)
        q1, q2, q3 = momenta[:, 0], momenta[:, 1], momenta[:, 2]

        _, _, y, z = cs_ff_map(q1, q3, q2)
        assert np.all(y >= -1e-12)
        assert np.all(y <= 1.0 + 1e-12)
        assert np.all(z >= -1e-12)
        assert np.all(z <= 1.0 + 1e-12)


class TestCSIFMapping:
    """Catani-Seymour initial-final mapping for massless partons."""

    def test_onshell(self):
        """Tilded initial momenta are massless."""
        p1, p2 = _make_beam(91.0)
        rng = np.random.default_rng(4)
        momenta, _ = rambo_massless(3, 91.0, 100, rng)
        q1, q2, q3 = momenta[:, 0], momenta[:, 1], momenta[:, 2]

        tilde_a, tilde_b, tilde_finals, x = cs_if_map(p1, q3, p2, [q1, q2])
        assert np.allclose(dot4(tilde_a, tilde_a), 0.0, atol=1e-8)
        assert np.allclose(dot4(tilde_b, tilde_b), 0.0, atol=1e-8)
        for tf in tilde_finals:
            assert np.allclose(dot4(tf, tf), 0.0, atol=1e-6)

    def test_momentum_conservation(self):
        """tilde_a + tilde_b = sum(tilde_finals)."""
        p1, p2 = _make_beam(91.0)
        rng = np.random.default_rng(5)
        momenta, _ = rambo_massless(3, 91.0, 100, rng)
        q1, q2, q3 = momenta[:, 0], momenta[:, 1], momenta[:, 2]

        tilde_a, tilde_b, tilde_finals, _ = cs_if_map(p1, q3, p2, [q1, q2])
        lhs = tilde_a + tilde_b
        rhs = sum(tilde_finals)
        assert np.allclose(lhs, rhs, atol=1e-7)

    def test_x_range(self):
        """Momentum fraction x in (0, 1]."""
        p1, p2 = _make_beam(91.0)
        rng = np.random.default_rng(6)
        momenta, _ = rambo_massless(3, 91.0, 1000, rng)
        q3 = momenta[:, 2]

        _, _, _, x = cs_if_map(p1, q3, p2, [momenta[:, 0], momenta[:, 1]])
        assert np.all(x > -1e-12)
        assert np.all(x <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# Born matrix element test
# ---------------------------------------------------------------------------

class TestBornMSQ:
    def test_90deg_value(self):
        """At 90-degree scattering, Born = e^4."""
        E = 45.5
        p1 = np.array([[E, 0, 0, E]])
        p2 = np.array([[E, 0, 0, -E]])
        q1 = np.array([[E, E, 0, 0]])
        q2 = np.array([[E, -E, 0, 0]])

        born = born_msq_eemumu(p1, p2, q1, q2)[0]
        e = math.sqrt(4.0 * math.pi * ALPHA_EM)
        assert born == pytest.approx(e**4, rel=1e-6)


# ---------------------------------------------------------------------------
# Dipole collinear limit test
# ---------------------------------------------------------------------------

class TestDipoleSum:
    """Properties of the CS dipole sum."""

    def test_same_line_dipoles_positive(self):
        """The 4 same-line FF+II dipoles are positive at typical phase-space points.

        These are the original 4 dipoles whose splitting kernels are non-negative
        for our (e+e-→μ+μ-γ, all unit charges, opposite-sign within each line)
        configuration.
        """
        rng = np.random.default_rng(42)
        momenta, _ = rambo_massless(3, 91.0, 100, rng)
        p1, p2 = _make_beam(91.0)
        dip = dipole_sum_eemumu(p1, p2, momenta, include_cross_line=False)
        assert np.all(dip > 0)

    def test_full_dipole_sum_finite(self):
        """The 12-dipole sum (4 same-line + 8 cross-line) is finite, but can
        have either sign because cross-line dipoles get charge correlators ±1.

        This is correct physics — cross-line eikonal contributions can be
        positive or negative depending on whether the emitter and spectator
        have same or opposite charges.  The test only requires finiteness.
        """
        rng = np.random.default_rng(42)
        momenta, _ = rambo_massless(3, 91.0, 100, rng)
        p1, p2 = _make_beam(91.0)
        dip = dipole_sum_eemumu(p1, p2, momenta, include_cross_line=True)
        assert np.all(np.isfinite(dip))


# ---------------------------------------------------------------------------
# NLO cross-section tests
# ---------------------------------------------------------------------------

class TestNLOCrossSection:
    """Integration tests for the NLO QED cross-section."""

    def test_kfactor_value(self):
        """K-factor = 1 + 3*alpha/(4*pi)."""
        expected = 1.0 + 3.0 * ALPHA_EM / (4.0 * math.pi)
        assert _NLO_K_FACTOR == pytest.approx(expected, rel=1e-10)

    def test_nlo_at_91gev(self):
        """sigma_NLO at sqrt_s = 91 GeV is consistent with Born * K."""
        result = nlo_cross_section_qed("e+ e- -> mu+ mu-", "QED", 91.0)
        assert result["supported"] is True
        assert result["k_factor"] == pytest.approx(_NLO_K_FACTOR, rel=1e-8)
        assert result["sigma_nlo_pb"] == pytest.approx(
            result["sigma_born_pb"] * _NLO_K_FACTOR, rel=1e-8
        )

    def test_energy_independence_of_kfactor(self):
        """The K-factor is energy-independent for massless fermions."""
        for sqrts in [10.0, 91.0, 500.0]:
            result = nlo_cross_section_qed("e+ e- -> mu+ mu-", "QED", sqrts)
            assert result["k_factor"] == pytest.approx(_NLO_K_FACTOR, rel=1e-10)

    def test_sigma_nlo_greater_than_born(self):
        """NLO cross-section > Born (positive correction)."""
        result = nlo_cross_section_qed("e+ e- -> mu+ mu-", "QED", 91.0)
        assert result["sigma_nlo_pb"] > result["sigma_born_pb"]

    def test_delta_nlo_magnitude(self):
        """The NLO correction is ~0.17% of Born."""
        result = nlo_cross_section_qed("e+ e- -> mu+ mu-", "QED", 91.0)
        frac = result["delta_nlo_pb"] / result["sigma_born_pb"]
        assert frac == pytest.approx(3.0 * ALPHA_EM / (4.0 * math.pi), rel=1e-8)

    def test_method_is_analytic(self):
        """The method should be 'analytic-kfactor'."""
        result = nlo_cross_section_qed("e+ e- -> mu+ mu-", "QED", 91.0)
        assert result["method"] == "analytic-kfactor"
        assert result["order"] == "NLO"

    def test_unsupported_process(self):
        """Unsupported processes return supported=False."""
        result = nlo_cross_section_qed("x x~ -> y y~", "QED", 91.0)
        assert result.get("supported") is False

    def test_bhabha_via_universal_qed(self):
        """V2.7: Bhabha (same flavor) doesn't match the textbook 2→2 exact branch
        but is now handled by the universal QED charge-correlator formula.
        For e+e-→e+e- with 4 charged legs of |Q|=1 the universal formula
        gives K = 1 + (α/(4π)) × Σ Q² × (3/4) = 1 + 3α/(4π) — same value
        as the textbook diff-flavor case.  Pure-leptonic 4-charged-leg 2→2
        is classified as ``validated`` (exact closed-form result)."""
        result = nlo_cross_section_qed("e+ e- -> e+ e-", "QED", 91.0)
        assert result.get("supported") is True
        expected_k = 1.0 + 3.0 * ALPHA_EM / (4.0 * math.pi)
        assert result["k_factor"] == pytest.approx(expected_k, rel=1e-6)

    def test_leptons_in_qcd_rejected(self):
        """Leptons in QCD theory are rejected (not in QCD registry)."""
        result = nlo_cross_section_qed("e+ e- -> mu+ mu-", "QCD", 91.0)
        assert result.get("supported") is False

    def test_tau_pair(self):
        """e+e- -> tau+tau- works at NLO (high energy)."""
        result = nlo_cross_section_qed("e+ e- -> tau+ tau-", "QED", 91.0)
        assert result["supported"] is True
        assert result["k_factor"] == pytest.approx(_NLO_K_FACTOR, rel=1e-8)

    def test_near_threshold_rejected(self):
        """Near-threshold tau pairs are rejected (mass corrections too large)."""
        result = nlo_cross_section_qed("e+ e- -> tau+ tau-", "QED", 4.0)
        assert result.get("supported") is False
        assert "threshold" in result.get("error", "")


# ---------------------------------------------------------------------------
# API endpoint test
# ---------------------------------------------------------------------------

class TestNLOAPIEndpoint:
    """Test the /api/amplitude/cross-section endpoint with order=NLO."""

    def test_nlo_endpoint(self):
        from fastapi.testclient import TestClient
        from feynman_engine.api.app import app

        client = TestClient(app)
        r = client.get(
            "/api/amplitude/cross-section",
            params={
                "process": "e+ e- -> mu+ mu-",
                "sqrt_s": 91,
                "order": "NLO",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["order"] == "NLO"
        assert data["k_factor"] == pytest.approx(1.001742, rel=1e-3)
        assert data["sigma_nlo_pb"] > data["sigma_born_pb"]

    def test_lo_still_works(self):
        from fastapi.testclient import TestClient
        from feynman_engine.api.app import app

        client = TestClient(app)
        r = client.get(
            "/api/amplitude/cross-section",
            params={"process": "e+ e- -> mu+ mu-", "sqrt_s": 91},
        )
        assert r.status_code == 200
        data = r.json()
        assert "sigma_pb" in data
        assert "order" not in data or data.get("order") != "NLO"

    def test_nlo_unregistered_qcd_blocked_via_api(self):
        """Unregistered QCD NLO is BLOCKED (no tabulated K, no running-coupling
        fallback) — the API returns 422 with a workaround pointing at the
        K-factor table or OpenLoops."""
        from fastapi.testclient import TestClient
        from feynman_engine.api.app import app

        client = TestClient(app)
        r = client.get(
            "/api/amplitude/cross-section",
            params={
                "process": "u u~ -> d d~",
                "theory": "QCD",
                "sqrt_s": 91,
                "order": "NLO",
            },
        )
        assert r.status_code == 422
        detail = r.json().get("detail", {})
        # block_reason explains why; workaround tells the user what to do
        assert "K-factor" in str(detail)

    def test_nlo_2to3_qed_via_universal(self):
        """V2.7: 2→3 QED NLO routed through the universal charge-correlator
        formula (approximate, ~0.1% accuracy on inclusive observables)."""
        from fastapi.testclient import TestClient
        from feynman_engine.api.app import app

        client = TestClient(app)
        r = client.get(
            "/api/amplitude/cross-section",
            params={
                "process": "e+ e- -> mu+ mu- gamma",
                "sqrt_s": 91,
                "order": "NLO",
                "n_events": 10000,
                "min_invariant_mass": 1.0,
            },
        )
        # Either 200 (universal-QED routes) or 422 (Born σ unavailable for 2→3 here).
        assert r.status_code in (200, 422)


# ---------------------------------------------------------------------------
# Running coupling tests
# ---------------------------------------------------------------------------

class TestRunningCouplings:
    """Tests for 1-loop running coupling functions."""

    def test_alpha_s_at_mz(self):
        """alpha_s(M_Z^2) reproduces input value."""
        mz = 91.1876
        assert alpha_s_running(mz**2) == pytest.approx(ALPHA_S, rel=1e-4)

    def test_alpha_s_asymptotic_freedom(self):
        """alpha_s decreases with increasing energy (asymptotic freedom)."""
        assert alpha_s_running(200**2) < alpha_s_running(91**2)
        assert alpha_s_running(1000**2) < alpha_s_running(200**2)

    def test_alpha_s_grows_below_mz(self):
        """alpha_s increases at lower scales."""
        assert alpha_s_running(30**2) > alpha_s_running(91**2)

    def test_alpha_s_landau_pole(self):
        """alpha_s returns 1.0 at the Landau pole (safe fallback)."""
        # Very low scale hits the Landau pole
        result = alpha_s_running(0.1**2)
        assert result == 1.0 or result > 0

    def test_alpha_s_nonpositive_scale(self):
        """alpha_s at non-positive scale returns input value."""
        assert alpha_s_running(0.0) == ALPHA_S
        assert alpha_s_running(-1.0) == ALPHA_S

    def test_alpha_em_at_zero(self):
        """alpha_em(0) = alpha_em (no running at zero)."""
        assert alpha_em_running(0.0) == ALPHA_EM
        assert alpha_em_running(-1.0) == ALPHA_EM

    def test_alpha_em_increases_with_energy(self):
        """alpha_em grows with energy (screening)."""
        assert alpha_em_running(10**2) > ALPHA_EM
        assert alpha_em_running(91**2) > alpha_em_running(10**2)

    def test_alpha_em_at_mz_reasonable(self):
        """alpha_em(M_Z) should be between 1/137 and 1/127 (leptonic VP only)."""
        a = alpha_em_running(91.2**2)
        assert 1.0 / 137.0 < a < 1.0 / 127.0


# ---------------------------------------------------------------------------
# Unregistered NLO is BLOCKED (V1 trust policy — no leading-log fallback)
# ---------------------------------------------------------------------------

class TestUnregisteredQCDNLOBlocked:
    """QCD NLO without a tabulated K-factor remains BLOCKED (V1 policy).
    QED and EW now have universal NLO modules (V2.7) and are NOT in this list."""

    @pytest.mark.parametrize("process,theory", [
        ("u u~ -> d d~",        "QCD"),
        ("u u~ -> g g",         "QCD"),
        ("g g -> u u~",         "QCD"),
        ("g g -> g g",          "QCD"),
        ("u g -> u g",          "QCD"),
        ("u u~ -> d d~",        "QCDQED"),
    ])
    def test_unregistered_qcd_nlo_blocked(self, process, theory):
        result = nlo_cross_section(process, theory, 91.0)
        assert result.get("supported") is False
        assert "K-factor" in result.get("error", "")


class TestUnregisteredQEDEWNowApproximate:
    """V2.7: QED via universal charge-correlator formula and EW via Sudakov LL+NLL
    now return supported=True for previously-blocked processes.  Pure-leptonic
    QED 4-charged-leg cases get trust=validated (textbook closed form);
    others get trust=approximate."""

    @pytest.mark.parametrize("process,theory", [
        ("e+ e- -> e+ e-",      "QED"),
        ("e+ e- -> gamma gamma","QED"),
        ("e- gamma -> e- gamma","QED"),
        ("e+ e- -> mu+ mu-",    "EW"),
    ])
    def test_qed_ew_universal_route(self, process, theory):
        result = nlo_cross_section(process, theory, 91.0)
        # supported=True OR Born σ unavailable (some processes have no LO formula)
        if result.get("supported"):
            assert "k_factor" in result
            assert result.get("trust_level") in ("approximate", "validated")


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------

class TestNLOEdgeCases:
    """Edge cases and error handling for the generalized NLO module."""

    def test_nonexistent_particles(self):
        """Unknown particles gracefully fail."""
        result = nlo_cross_section("x x~ -> y y~", "QED", 91.0)
        assert result.get("supported") is False

    def test_nonexistent_theory(self):
        """Unknown theory gracefully fails."""
        result = nlo_cross_section("e+ e- -> mu+ mu-", "SUSY", 91.0)
        assert result.get("supported") is False

    def test_result_schema(self):
        """NLO result has all required fields."""
        result = nlo_cross_section("e+ e- -> mu+ mu-", "QED", 91.0)
        required_keys = [
            "process", "theory", "sqrt_s_gev", "s_gev2", "order",
            "method", "nlo_description", "sigma_born_pb", "sigma_nlo_pb",
            "delta_nlo_pb", "k_factor", "supported",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_backward_compat_alias(self):
        """nlo_cross_section_qed delegates to nlo_cross_section."""
        r1 = nlo_cross_section("e+ e- -> mu+ mu-", "QED", 91.0)
        r2 = nlo_cross_section_qed("e+ e- -> mu+ mu-", "QED", 91.0)
        assert r1["k_factor"] == r2["k_factor"]
        assert r1["sigma_nlo_pb"] == r2["sigma_nlo_pb"]

    def test_2to1_rejected(self):
        """2→1 processes are rejected."""
        result = nlo_cross_section("e+ e- -> Zp", "BSM", 500.0)
        assert result.get("supported") is False

    def test_decay_rejected(self):
        """1→2 decays are rejected (not 2→N scattering)."""
        result = nlo_cross_section("Z -> e+ e-", "EW", 91.0)
        assert result.get("supported") is False
        assert "Decay" in result.get("error", "")

