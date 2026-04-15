"""Tests for 1→2 decay computations (V→ff, S→ff)."""
from __future__ import annotations

import pytest
from sympy import Symbol, Rational, simplify, pi as sym_pi, sqrt

from feynman_engine.amplitudes.form_trace import get_form_decay
from feynman_engine.physics.amplitude import get_amplitude


# ── Z → e+ e- (vector → lepton pair) ────────────────────────────────────────

def test_z_to_ee_massless_limit():
    """Z → e+e- in the massless-electron limit: |M̄|² = (4/3) g² m_Z²."""
    result = get_form_decay("Z -> e+ e-", "EW")
    assert result is not None
    assert result.backend == "form-decay"

    m_Z = Symbol("m_Z")
    # Coupling is flavour-specific: g_Z_e
    g = Symbol("g_Z_e")
    msq_massless = result.msq.subs(Symbol("m_e"), 0)
    expected = Rational(4, 3) * g**2 * m_Z**2
    assert simplify(msq_massless - expected) == 0


def test_z_to_ee_has_width():
    """Z → e+e- should include a decay width expression."""
    result = get_form_decay("Z -> e+ e-", "EW")
    assert result is not None
    assert result.integral_latex is not None
    assert "pi" in result.integral_latex or "\\pi" in result.integral_latex


def test_z_to_ee_numerical():
    """Z → e+e-: |M̄|² should be positive at physical values."""
    result = get_form_decay("Z -> e+ e-", "EW")
    assert result is not None
    # g_Z_e ≈ g_Z × (T3 - Q sin²θ_W) ≈ 0.7434 × (-0.5 + 0.2312) ≈ -0.200
    # But we just need positive |M|², so use |g_Z_e| = 0.2.
    vals = {
        Symbol("m_Z"): 91.1876,
        Symbol("g_Z_e"): 0.2,
        Symbol("m_e"): 0.000511,
    }
    num = float(result.msq.subs(vals))
    assert num > 0


# ── Z → mu+ mu- ─────────────────────────────────────────────────────────────

def test_z_to_mumu():
    """Z → μ+μ- should give same structure as Z → e+e- (lepton universality)."""
    result_ee = get_form_decay("Z -> e+ e-", "EW")
    result_mm = get_form_decay("Z -> mu+ mu-", "EW")
    assert result_ee is not None
    assert result_mm is not None
    m_Z = Symbol("m_Z")
    # Massless limit: both should be (4/3) g² m_Z² with their respective couplings.
    msq_ee = result_ee.msq.subs(Symbol("m_e"), 0)
    msq_mm = result_mm.msq.subs(Symbol("m_mu"), 0)
    expected_ee = Rational(4, 3) * Symbol("g_Z_e")**2 * m_Z**2
    expected_mm = Rational(4, 3) * Symbol("g_Z_mu")**2 * m_Z**2
    assert simplify(msq_ee - expected_ee) == 0
    assert simplify(msq_mm - expected_mm) == 0


# ── H → b b~ (scalar → quark pair) ──────────────────────────────────────────

def test_h_to_bb():
    """H → bb̄: |M̄|² = Nc × y_b² × 2(M_H² - 4m_b²)."""
    result = get_form_decay("H -> b b~", "EW")
    assert result is not None
    assert result.backend == "form-decay"

    m_H = Symbol("m_H")
    m_b = Symbol("m_b")
    y_b = Symbol("y_b")

    # Expected: Nc=3, trace = 4(q1·q2 - m²) = 2M² - 8m² = 2(M² - 4m²)
    expected = 3 * y_b**2 * (2 * m_H**2 - 8 * m_b**2)
    assert simplify(result.msq - expected) == 0


def test_h_to_bb_numerical_positive():
    """H → bb̄ should have positive |M̄|² at physical masses."""
    result = get_form_decay("H -> b b~", "EW")
    assert result is not None
    vals = {
        Symbol("m_H"): 125.20,
        Symbol("m_b"): 4.18,
        Symbol("y_b"): 0.024,  # approximate Yukawa
    }
    num = float(result.msq.subs(vals))
    assert num > 0


# ── H → tau+ tau- ───────────────────────────────────────────────────────────

def test_h_to_tautau():
    """H → ττ: scalar decay to lepton pair with color factor 1."""
    result = get_form_decay("H -> tau+ tau-", "EW")
    assert result is not None

    m_H = Symbol("m_H")
    m_tau = Symbol("m_tau")
    y_tau = Symbol("y_tau")

    # Color factor = 1 (leptons), trace = 2(M²-4m²)
    expected = y_tau**2 * (2 * m_H**2 - 8 * m_tau**2)
    assert simplify(result.msq - expected) == 0


# ── Pipeline integration ─────────────────────────────────────────────────────

def test_get_amplitude_routes_decay():
    """get_amplitude() should route 1→2 processes to the decay backend."""
    result = get_amplitude("Z -> e+ e-", "EW")
    assert result is not None
    assert result.backend == "form-decay"


def test_get_amplitude_routes_decay_higgs():
    """get_amplitude() should route H → bb̄ through decay backend."""
    result = get_amplitude("H -> b b~", "EW")
    assert result is not None
    assert result.backend == "form-decay"


# ── Edge cases ───────────────────────────────────────────────────────────────

def test_decay_returns_none_for_scattering():
    """get_form_decay() should return None for 2→2 processes."""
    result = get_form_decay("e+ e- -> mu+ mu-", "QED")
    assert result is None


def test_decay_returns_none_for_massless_parent():
    """Massless particles don't decay."""
    result = get_form_decay("gamma -> e+ e-", "QED")
    assert result is None
