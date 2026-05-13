"""Tests for the colour-correlated I-operator (cs_dipoles.i_operator_color_correlated).

These tests require the OpenLoops library to be installed.  The colour
correlator path is what unlocks /api/amplitude/nlo-general for processes
with coloured final states (pp→tt̄, pp→jj, etc.).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Skip if OpenLoops not installed
ol_missing = False
try:
    from feynman_engine.amplitudes.openloops_bridge import _load_openloops
    if _load_openloops() is None:
        ol_missing = True
except Exception:
    ol_missing = True

if ol_missing:
    pytest.skip("OpenLoops not installed", allow_module_level=True)

from feynman_engine.amplitudes.cs_dipoles import (
    C_F, C_A,
    parton_type,
    i_operator_for_born,
    i_operator_color_correlated,
)


def _make_2to2_momenta(sqrt_s: float, m_out: float = 0.0) -> np.ndarray:
    """Build a 4-leg (5,4)-flattened momentum array for 2→2 kinematics."""
    E_in = sqrt_s / 2.0
    p_out = (sqrt_s / 2.0) * math.sqrt(max(1.0 - (2 * m_out / sqrt_s) ** 2, 0.0))
    return np.array([
        E_in, 0.0, 0.0,  E_in, 0.0,        # incoming a
        E_in, 0.0, 0.0, -E_in, 0.0,        # incoming b
        sqrt_s / 2, 0.0, 0.0,  p_out, m_out,  # outgoing 1
        sqrt_s / 2, 0.0, 0.0, -p_out, m_out,  # outgoing 2
    ], dtype=np.float64)


class TestColorCorrelatedQQtoTT:
    """q q̄ → t t̄ has 4 SU(3)-fundamental legs.  Colour-conservation gives
    Σ_i Σ_{k≠i} T_i·T_k / |B|² = -Σ_i T_i² = -4 C_F = -16/3."""

    def test_pole2_matches_color_conservation(self):
        born_in = [parton_type("u"), parton_type("u~")]
        born_out = [parton_type("t"), parton_type("t~")]
        mom = _make_2to2_momenta(500.0, m_out=172.69)
        r = i_operator_for_born(
            born_in, born_out, s=500.0 ** 2, mu_sq=91.1876 ** 2,
            process="u u~ -> t t~", momenta_5xn=mom,
        )
        # pole2 should equal -Σ T_i·T_k summed with the OL sign convention
        # we apply (+ sign in our cc_norm sum, see code).  For 4 quark legs
        # this is 4 × C_F = 16/3 ≈ 5.333.
        expected = 4.0 * C_F
        assert r.pole2 == pytest.approx(expected, rel=1e-2)


class TestColorCorrelatedGGtoTT:
    """g g → t t̄ has 2 adjoint + 2 fundamental colour legs.  Colour
    conservation: Σ T_i² = 2 C_A + 2 C_F = 6 + 8/3 = 26/3 ≈ 8.667."""

    def test_pole2_matches_color_conservation(self):
        born_in = [parton_type("g"), parton_type("g")]
        born_out = [parton_type("t"), parton_type("t~")]
        mom = _make_2to2_momenta(500.0, m_out=172.69)
        try:
            r = i_operator_for_born(
                born_in, born_out, s=500.0 ** 2, mu_sq=91.1876 ** 2,
                process="g g -> t t~", momenta_5xn=mom,
            )
        except RuntimeError as e:
            pytest.skip(f"OL register_process for g g → t t~ failed: {e}")
        expected = 2.0 * C_A + 2.0 * C_F
        assert r.pole2 == pytest.approx(expected, rel=1e-2)


class TestColorCorrelatedFallthrough:
    """When ``process`` and ``momenta_5xn`` are NOT supplied, the function
    should still work for colour-neutral final states (legacy path).
    """

    def test_qqbar_to_zz_no_extra_args(self):
        born_in = [parton_type("u"), parton_type("u~")]
        born_out = [parton_type("Z"), parton_type("Z")]
        r = i_operator_for_born(born_in, born_out, s=200.0 ** 2)
        # Standard q q̄ → V V I-operator: pole2 = 2 C_F = 8/3
        assert r.pole2 == pytest.approx(2.0 * C_F, rel=1e-6)
