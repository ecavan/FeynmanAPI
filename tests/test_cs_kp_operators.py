"""Tests for the Catani-Seymour K and P operators (cs_kp_operators.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from feynman_engine.amplitudes.cs_kp_operators import (
    C_F, C_A, T_R, N_F, GAMMA_Q, GAMMA_G,
    K_qq_regular, K_gg_regular, K_qg, K_gq,
    P_qq_split, P_gg_split, P_qg_split, P_gq_split,
    cs_pdf_counterterm,
)


class TestSplittingFunctions:
    """The DGLAP splitting functions must reduce to known values at z=0, 1/2, 1."""

    def test_P_qq_at_half(self):
        """P_qq(1/2) = C_F · (1 + 1/4) / (1/2) = C_F · 5/2."""
        z = np.array([0.5])
        assert float(P_qq_split(z)[0]) == pytest.approx(2.5 * C_F)

    def test_P_qg_at_half(self):
        """P_qg(1/2) = T_R · (1/4 + 1/4) = T_R / 2."""
        z = np.array([0.5])
        assert float(P_qg_split(z)[0]) == pytest.approx(0.5 * T_R)

    def test_P_qg_at_zero(self):
        """P_qg(0) = T_R · (0 + 1) = T_R (gluon → qq̄ smooth at z=0)."""
        z = np.array([1e-8])
        assert float(P_qg_split(z)[0]) == pytest.approx(T_R, rel=1e-4)

    def test_P_gg_at_half(self):
        """P_gg(1/2) (regular) = 2 C_A · (1/2 / (1/2) + 1/2 · 1/2)
                                = 2 C_A · (1 + 1/4)  — wait, our P_gg^reg
        excludes the z/(1-z) plus part.  We have only (1-z)/z + z(1-z).
        At z=1/2: (1/2)/(1/2) + (1/2)(1/2) = 1 + 0.25 = 1.25
        So P_gg^reg(1/2) = 2 C_A · 1.25 = 2.5 C_A.
        """
        z = np.array([0.5])
        assert float(P_gg_split(z)[0]) == pytest.approx(2.5 * C_A)

    def test_P_gq_at_half(self):
        """P_gq(1/2) = C_F · (1 + 1/4) / (1/2) = C_F · 5/2."""
        z = np.array([0.5])
        assert float(P_gq_split(z)[0]) == pytest.approx(2.5 * C_F)


class TestKOperatorBoundary:
    """K operators are smooth functions; verify they're finite on (0, 1)."""

    @pytest.mark.parametrize("fn", [K_qg, K_gq, K_qq_regular, K_gg_regular])
    def test_finite_in_interior(self, fn):
        z = np.linspace(0.01, 0.99, 50)
        vals = fn(z)
        assert np.all(np.isfinite(vals)), f"{fn.__name__} not finite on (0.01, 0.99)"

    def test_K_qg_at_half(self):
        """K_qg(1/2) = P_qg(1/2) × log((1/2)²/(1/2)) + 2 T_R × 1/4
                   = (T_R/2) × log(1/2) + T_R/2 = T_R × (1 + log(1/2))/2
        """
        z = np.array([0.5])
        expected = T_R * 0.5 * math.log(0.5) + 2.0 * T_R * 0.25
        assert float(K_qg(z)[0]) == pytest.approx(expected, rel=1e-6)


class TestPDFCountertermInfrastructure:
    """The cs_pdf_counterterm function must execute without errors and return
    a structured result. Numerical validation against MCFM is out-of-scope for
    a smoke test; we verify the interface and the α_s/2π normalisation.
    """

    def test_qqbar_counterterm_runs(self):
        # σ̂_Born does not depend on z in this synthetic test
        def f_z(z):
            return np.ones_like(z) * 100.0  # arbitrary "g(z)" baseline

        r = cs_pdf_counterterm(
            initial_partons=("q", "qbar"),
            f_callback=f_z,
            alpha_s=0.118,
            mu_F_sq=91.1876 ** 2,
            mu_R_sq=91.1876 ** 2,   # μ_F = μ_R: P-operator log = 0
            n_z=200,
        )
        assert r["supported"]
        # log(μ_F²/μ_R²) = 0 for this choice
        assert r["log_muF2_muR2"] == pytest.approx(0.0, abs=1e-12)
        # P-part log coefficient should be zero, so all P_part_logFR = 0
        for c in r["contributions"]:
            assert c["P_part_logFR"] == pytest.approx(0.0, abs=1e-10)
        # K-part should be non-zero (the actual MS-bar counterterm)
        K_total = sum(c["K_part"] for c in r["contributions"])
        assert abs(K_total) > 0

    def test_gg_counterterm_runs(self):
        def f_z(z):
            return np.ones_like(z) * 50.0

        r = cs_pdf_counterterm(
            initial_partons=("g", "g"),
            f_callback=f_z,
            alpha_s=0.118,
            mu_F_sq=125.0 ** 2,
            mu_R_sq=91.1876 ** 2,
            n_z=200,
        )
        assert r["supported"]
        # log(μ_F²/μ_R²) ≠ 0 here
        assert r["log_muF2_muR2"] != 0.0
        # P-part should contribute
        P_total = sum(c["P_part_logFR"] for c in r["contributions"])
        assert abs(P_total) > 0

    def test_alpha_s_normalisation(self):
        """The prefactor α_s/(2π) must scale the result linearly."""
        def f_z(z):
            return np.ones_like(z) * 10.0

        r1 = cs_pdf_counterterm(("q", "qbar"), f_z, 0.1, 91.0**2, 91.0**2, n_z=200)
        r2 = cs_pdf_counterterm(("q", "qbar"), f_z, 0.2, 91.0**2, 91.0**2, n_z=200)
        # Doubling α_s should double σ_pdf_ct (linear in α_s)
        assert r2["sigma_pdf_ct_pb"] == pytest.approx(2.0 * r1["sigma_pdf_ct_pb"], rel=1e-6)

    def test_scale_log_linearity(self):
        """Doubling log(μ_F²/μ_R²) should double the P-part contribution."""
        def f_z(z):
            return np.ones_like(z) * 10.0

        # Reference: μ_F = μ_R
        r0 = cs_pdf_counterterm(("g", "g"), f_z, 0.118, 91.0**2, 91.0**2, n_z=200)
        # μ_F² / μ_R² = e (log = 1)
        r1 = cs_pdf_counterterm(("g", "g"), f_z, 0.118, math.e * 91.0**2, 91.0**2, n_z=200)
        # μ_F² / μ_R² = e² (log = 2)
        r2 = cs_pdf_counterterm(("g", "g"), f_z, 0.118, math.e**2 * 91.0**2, 91.0**2, n_z=200)
        # The P-part scales linearly with the log; K-part is invariant.
        P1 = sum(c["P_part_logFR"] for c in r1["contributions"])
        P2 = sum(c["P_part_logFR"] for c in r2["contributions"])
        K0 = sum(c["K_part"] for c in r0["contributions"])
        K1 = sum(c["K_part"] for c in r1["contributions"])
        assert K1 == pytest.approx(K0, rel=1e-8)
        # P2 / P1 should be 2 (since log doubles)
        if abs(P1) > 1e-20:
            assert P2 == pytest.approx(2.0 * P1, rel=1e-6)
