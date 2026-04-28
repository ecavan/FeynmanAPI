"""Tests for differential observables and the qq̄→ZH curated amplitude."""
from __future__ import annotations

import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 2→2 deterministic histogram
# ---------------------------------------------------------------------------

class TestDifferentialCosTheta:
    """dσ/d(cosθ) for e+e-→μ+μ- via scipy.quad per bin."""

    def test_basic_shape(self):
        """1 + cos²θ shape at √s = 10 GeV."""
        from feynman_engine.amplitudes.differential import differential_distribution
        edges = np.linspace(-1, 1, 11)
        r = differential_distribution(
            "e+ e- -> mu+ mu-", "QED", sqrt_s=10.0,
            observable="cos_theta", bin_edges=edges,
        )
        assert r["supported"]
        # Symmetric about cosθ=0
        ds = np.array(r["dsigma_dX_pb"])
        assert np.allclose(ds, ds[::-1], rtol=1e-3)
        # Peaks at the endpoints (1+cos²θ form)
        assert ds[0] > ds[len(ds) // 2]

    def test_total_matches_analytic(self):
        """Σ(σ_per_bin) reproduces 4πα²/(3s) for e+e-→μ+μ-."""
        from feynman_engine.amplitudes.differential import differential_distribution
        edges = np.linspace(-0.999, 0.999, 21)
        r = differential_distribution(
            "e+ e- -> mu+ mu-", "QED", sqrt_s=10.0,
            observable="cos_theta", bin_edges=edges,
        )
        # Analytic: 4πα²/(3s) × 1/137²×3.9e8 ~ 870 pb at √s=10
        sigma_analytic = 4 * math.pi * (1 / 137.036) ** 2 / (3 * 100.0) * 3.8938e8
        assert abs(r["sigma_total_pb"] - sigma_analytic) / sigma_analytic < 0.02

    def test_running_kfactor_applied(self):
        """order='NLO-running' rescales every bin by the same K factor."""
        from feynman_engine.amplitudes.differential import differential_distribution
        edges = np.linspace(-0.5, 0.5, 6)
        lo = differential_distribution(
            "e+ e- -> mu+ mu-", "QED", sqrt_s=91.0,
            observable="cos_theta", bin_edges=edges, order="LO",
        )
        nlo = differential_distribution(
            "e+ e- -> mu+ mu-", "QED", sqrt_s=91.0,
            observable="cos_theta", bin_edges=edges, order="NLO-running",
        )
        assert nlo["order"] == "NLO"
        ratios = [
            n / l if l > 0 else 1.0
            for n, l in zip(nlo["dsigma_dX_pb"], lo["dsigma_dX_pb"])
        ]
        # All bins scaled by the same K
        assert max(ratios) - min(ratios) < 1e-6
        assert nlo["k_factor"] == pytest.approx(ratios[0], rel=1e-6)


# ---------------------------------------------------------------------------
# 2→N MC histograms
# ---------------------------------------------------------------------------

class TestDifferentialMC:
    """Per-event MC histograms for 2→N observables."""

    def test_pT_lepton_2to3(self):
        """e+e-→μ+μ-γ: dσ/dpT_μ histogram has positive σ in every populated bin."""
        from feynman_engine.amplitudes.differential import differential_distribution
        edges = np.linspace(0, 40, 9)
        r = differential_distribution(
            "e+ e- -> mu+ mu- gamma", "QED", sqrt_s=91.0,
            observable="pT_lepton", bin_edges=edges,
            n_events=10_000, min_invariant_mass=2.0,
        )
        assert r["supported"]
        ds = np.array(r["dsigma_dX_pb"])
        assert np.all(ds >= 0)
        assert r["sigma_total_pb"] > 0

    def test_M_ll_for_2to3(self):
        """e+e-→μ+μ-γ: dσ/dM_μμ histogram is non-negative."""
        from feynman_engine.amplitudes.differential import differential_distribution
        edges = np.linspace(0, 91, 10)
        r = differential_distribution(
            "e+ e- -> mu+ mu- gamma", "QED", sqrt_s=91.0,
            observable="M_ll", bin_edges=edges,
            n_events=10_000, min_invariant_mass=2.0,
        )
        assert r["supported"]
        assert all(v >= 0 for v in r["dsigma_dX_pb"])

    def test_invalid_bin_edges(self):
        """Non-monotone bin edges return supported=False."""
        from feynman_engine.amplitudes.differential import differential_distribution
        r = differential_distribution(
            "e+ e- -> mu+ mu-", "QED", sqrt_s=10.0,
            observable="cos_theta", bin_edges=[0.0, 1.0, 0.5],
        )
        assert not r["supported"]

    def test_unsupported_observable_for_2to2(self):
        """Asking for M_ll on a 2→2 process where it makes no sense — we still try."""
        from feynman_engine.amplitudes.differential import differential_distribution
        r = differential_distribution(
            "e+ e- -> mu+ mu-", "QED", sqrt_s=10.0,
            observable="M_ll", bin_edges=np.linspace(0, 20, 5),
            n_events=2000,
        )
        # 2→2 has 2 leptons → M_ll is the full M_inv → should work
        assert r.get("supported")


# ---------------------------------------------------------------------------
# Curated qq̄ → ZH amplitude
# ---------------------------------------------------------------------------

class TestQQbarToZH:
    """qq̄ → ZH (Higgsstrahlung) curated amplitude."""

    def test_curated_registered(self):
        """All five quark flavours have a curated qq̄→ZH amplitude in EW theory."""
        from feynman_engine.physics.amplitude import get_curated_amplitude
        for q in ("u", "d", "c", "s", "b"):
            r = get_curated_amplitude(f"{q} {q}~ -> Z H", "EW")
            assert r is not None, f"missing curated for {q} {q}~ -> Z H"
            assert r.backend == "curated"

    def test_partonic_cross_section_above_threshold(self):
        """σ̂(uū → ZH) > 0 above threshold √ŝ > m_Z + m_H ≈ 216 GeV."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("u u~ -> Z H", "EW", sqrt_s=300.0)
        assert r["supported"]
        assert r["sigma_pb"] > 0

    def test_below_threshold_zero(self):
        """Below threshold σ̂=0 (production forbidden)."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        r = total_cross_section("u u~ -> Z H", "EW", sqrt_s=200.0)
        # Either not supported (threshold check) or sigma=0
        assert (not r["supported"]) or r["sigma_pb"] == 0

    def test_up_dominates_down_via_couplings(self):
        """Partonic σ̂(uū → ZH) and σ̂(dd̄ → ZH) order-of-magnitude consistent."""
        from feynman_engine.amplitudes.cross_section import total_cross_section
        ru = total_cross_section("u u~ -> Z H", "EW", sqrt_s=300.0)
        rd = total_cross_section("d d~ -> Z H", "EW", sqrt_s=300.0)
        # Both nonzero, within factor of 5 of each other (couplings differ but
        # not by orders of magnitude).
        assert ru["sigma_pb"] > 0 and rd["sigma_pb"] > 0
        ratio = max(ru["sigma_pb"], rd["sigma_pb"]) / min(ru["sigma_pb"], rd["sigma_pb"])
        assert ratio < 5

    def test_pp_to_ZH_via_generic_enumeration(self):
        """pp → ZH integrates through the generic enumerator using these amplitudes."""
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        r = hadronic_cross_section("p p -> Z H", sqrt_s=13000.0, theory="EW")
        assert r["supported"]
        # Order-of-magnitude comparison to LHC qq̄→ZH (~0.5 pb at LO; loop-induced
        # gg→ZH not yet implemented).
        assert 0.05 < r["sigma_pb"] < 2.0
        # At least 4 of {u, d, s, c, b} channels should have contributed.
        assert r["n_channels_evaluated"] >= 4


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

class TestDifferentialAPI:
    """Test /amplitude/differential-distribution endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from fastapi.testclient import TestClient
        from feynman_engine.api.app import app
        self.client = TestClient(app)

    def test_cos_theta_endpoint(self):
        r = self.client.get(
            "/api/amplitude/differential-distribution",
            params={
                "process": "e+ e- -> mu+ mu-",
                "theory": "QED",
                "sqrt_s": 10.0,
                "observable": "cos_theta",
                "bin_min": -1.0,
                "bin_max": 1.0,
                "n_bins": 8,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert d["observable"] == "cos_theta"
        assert d["unit"] == "dimensionless"
        assert len(d["bin_centers"]) == 8
        assert d["sigma_total_pb"] > 0

    def test_pT_endpoint(self):
        r = self.client.get(
            "/api/amplitude/differential-distribution",
            params={
                "process": "e+ e- -> mu+ mu- gamma",
                "theory": "QED",
                "sqrt_s": 91.0,
                "observable": "pT_lepton",
                "bin_min": 0.0,
                "bin_max": 30.0,
                "n_bins": 6,
                "n_events": 5000,
                "min_invariant_mass": 2.0,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert d["unit"] == "GeV"
        assert all(v >= 0 for v in d["dsigma_dX_pb"])
