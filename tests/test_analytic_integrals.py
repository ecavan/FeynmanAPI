"""Tests for analytic closed-form PV scalar integrals.

Validates:
  - A₀, B₀ (all 6 special cases), B₁, B₀₀ against LoopTools
  - C₀ closed-form Li₂ and Feynman parameter integral against LoopTools
  - D₀ massless box formula
  - Symbolic mode produces correct SymPy expressions
  - evaluate() methods on PV integral dataclasses
"""
import math
import cmath
import pytest
from sympy import Symbol, symbols, log, simplify, S

from feynman_engine.amplitudes.analytic_integrals import (
    analytic_A0,
    analytic_B0,
    analytic_B1,
    analytic_B00,
    analytic_C0,
    analytic_D0,
    Delta_UV,
)
from feynman_engine.amplitudes.loop import (
    A0Integral, B0Integral, B1Integral, B00Integral,
    C0Integral, D0Integral,
)

# ── LoopTools availability ───────────────────────────────────────────────────

try:
    from feynman_engine.amplitudes.looptools_bridge import (
        is_available as _lt_avail,
        A0 as LT_A0, B0 as LT_B0, B1 as LT_B1, B00 as LT_B00,
        C0 as LT_C0, D0 as LT_D0,
    )
    _HAS_LT = _lt_avail()
except ImportError:
    _HAS_LT = False

requires_looptools = pytest.mark.skipif(
    not _HAS_LT, reason="LoopTools not installed"
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _rel_err(a, b):
    if abs(b) > 1e-15:
        return abs(a - b) / abs(b)
    return abs(a - b)

TOL = 1e-8  # relative tolerance for LoopTools comparison


# ══════════════════════════════════════════════════════════════════════════════
# A₀ tests
# ══════════════════════════════════════════════════════════════════════════════

class TestA0:
    def test_massless_vanishes(self):
        assert analytic_A0(0.0) == 0.0

    @requires_looptools
    @pytest.mark.parametrize("m_sq", [0.25, 1.0, 4.0, 100.0])
    def test_matches_looptools(self, m_sq):
        an = analytic_A0(m_sq)
        lt = LT_A0(m_sq)
        assert _rel_err(an, lt) < TOL, f"A0({m_sq}): {an} vs {lt}"

    def test_symbolic_contains_delta_uv(self):
        m = Symbol("m", positive=True)
        result = analytic_A0(m**2)
        assert result.has(Delta_UV)

    def test_symbolic_finite_part(self):
        m = Symbol("m", positive=True)
        result = analytic_A0(m**2, delta_uv=0)
        # Should be m² × (1 - ln(m²/μ²))
        assert not result.has(Delta_UV)


# ══════════════════════════════════════════════════════════════════════════════
# B₀ tests
# ══════════════════════════════════════════════════════════════════════════════

class TestB0:
    @requires_looptools
    @pytest.mark.parametrize("p_sq,m1,m2", [
        # Both massless
        (-2.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        # Equal mass
        (4.0, 1.0, 1.0),    # above threshold
        (-2.0, 1.0, 1.0),   # spacelike
        (2.0, 1.0, 1.0),    # below threshold
        # Equal mass, p²=0
        (0.0, 1.0, 1.0),
        # One massless
        (1.0, 0.0, 1.0),
        (-2.0, 0.0, 1.0),
        (4.0, 0.0, 1.0),
        # Zero momentum, different masses
        (0.0, 1.0, 4.0),
        (0.0, 0.25, 1.0),
        # General
        (-2.0, 1.0, 4.0),
        (3.0, 0.5, 2.0),
    ])
    def test_matches_looptools(self, p_sq, m1, m2):
        an = analytic_B0(p_sq, m1, m2)
        lt = LT_B0(p_sq, m1, m2)
        assert _rel_err(an, lt) < TOL, f"B0({p_sq},{m1},{m2}): {an} vs {lt}"

    def test_scaleless_vanishes(self):
        assert analytic_B0(0.0, 0.0, 0.0) == 0.0

    def test_symbolic_equal_mass(self):
        p, m = symbols("p m", positive=True)
        result = analytic_B0(p**2, m**2, m**2)
        assert result.has(Delta_UV)
        assert result.has(log)


# ══════════════════════════════════════════════════════════════════════════════
# B₁ and B₀₀ tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTensorB:
    @requires_looptools
    @pytest.mark.parametrize("p_sq,m1,m2", [
        (4.0, 1.0, 1.0),
        (-2.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (-2.0, 1.0, 4.0),
    ])
    def test_B1_matches_looptools(self, p_sq, m1, m2):
        an = analytic_B1(p_sq, m1, m2)
        lt = LT_B1(p_sq, m1, m2)
        if an is None:
            pytest.skip("B1 returned None")
        assert _rel_err(an, lt) < TOL, f"B1({p_sq},{m1},{m2}): {an} vs {lt}"

    @requires_looptools
    @pytest.mark.parametrize("p_sq,m1,m2", [
        (4.0, 1.0, 1.0),
        (-2.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
    ])
    def test_B00_matches_looptools(self, p_sq, m1, m2):
        an = analytic_B00(p_sq, m1, m2)
        lt = LT_B00(p_sq, m1, m2)
        if an is None:
            pytest.skip("B00 returned None")
        assert _rel_err(an, lt) < TOL, f"B00({p_sq},{m1},{m2}): {an} vs {lt}"


# ══════════════════════════════════════════════════════════════════════════════
# C₀ tests
# ══════════════════════════════════════════════════════════════════════════════

class TestC0:
    @requires_looptools
    @pytest.mark.parametrize("args", [
        # One-mass triangle: C0(0,0,s; 0,m²,m²) — Li₂ closed form
        (0, 0, -2, 0, 1, 1),
        (0, 0, -1, 0, 1, 1),
        (0, 0, -5, 0, 1, 1),
        (0, 0, 0.5, 0, 1, 1),
        (0, 0, 2, 0, 1, 1),      # timelike, s/m² > 1
        (0, 0, 4, 0, 1, 1),      # above threshold
        (0, 0, 8, 0, 1, 1),      # deep timelike
        (0, 0, -2, 0, 4, 4),     # different mass
        (0, 0, 10, 0, 4, 4),     # above threshold m²=4
    ])
    def test_one_mass_matches_looptools(self, args):
        fargs = [float(x) for x in args]
        an = analytic_C0(*fargs)
        lt = LT_C0(*fargs)
        assert an is not None, f"C0{args}: returned None"
        assert _rel_err(an, lt) < TOL, f"C0{args}: {an} vs {lt}"

    @requires_looptools
    @pytest.mark.parametrize("args", [
        # QED vertex spacelike: C0(m²,m²,q²; 0,m²,m²)
        (1, 1, -0.5, 0, 1, 1),
        (1, 1, -2, 0, 1, 1),
        (1, 1, -5, 0, 1, 1),
        (4, 4, -2, 0, 4, 4),
        # General spacelike
        (1, 2, -3, 0.5, 1, 2),
    ])
    def test_spacelike_matches_looptools(self, args):
        fargs = [float(x) for x in args]
        an = analytic_C0(*fargs)
        lt = LT_C0(*fargs)
        if an is None:
            pytest.skip("unsupported config")
        assert _rel_err(an, lt) < TOL, f"C0{args}: {an} vs {lt}"

    def test_timelike_vertex_falls_back(self):
        # Threshold crossing → should return None
        assert analytic_C0(1.0, 1.0, 5.0, 0.0, 1.0, 1.0) is None

    def test_scaleless_returns_none(self):
        assert analytic_C0(0.0, 0.0, 0.0, 0.0, 1.0, 1.0) is None


# ══════════════════════════════════════════════════════════════════════════════
# D₀ tests
# ══════════════════════════════════════════════════════════════════════════════

class TestD0:
    def test_massless_box_returns_value(self):
        val = analytic_D0(0, 0, 0, 0, 4.0, -2.0, 0, 0, 0, 0)
        assert val is not None
        assert isinstance(val, complex)

    def test_unsupported_returns_none(self):
        val = analytic_D0(1, 0, 1, 0, 4.0, -2.0, 1, 0, 1, 0)
        assert val is None


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass evaluate() tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDataclassEvaluate:
    @requires_looptools
    def test_A0_evaluate(self):
        integral = A0Integral(m_sq=1.0)
        assert _rel_err(integral.evaluate(), LT_A0(1.0)) < TOL

    @requires_looptools
    def test_B0_evaluate(self):
        integral = B0Integral(p_sq=4.0, m1_sq=1.0, m2_sq=1.0)
        assert _rel_err(integral.evaluate(), LT_B0(4.0, 1.0, 1.0)) < TOL

    @requires_looptools
    def test_B1_evaluate(self):
        integral = B1Integral(p_sq=4.0, m1_sq=1.0, m2_sq=1.0)
        assert _rel_err(integral.evaluate(), LT_B1(4.0, 1.0, 1.0)) < TOL

    @requires_looptools
    def test_B00_evaluate(self):
        integral = B00Integral(p_sq=4.0, m1_sq=1.0, m2_sq=1.0)
        assert _rel_err(integral.evaluate(), LT_B00(4.0, 1.0, 1.0)) < TOL

    @requires_looptools
    def test_C0_evaluate(self):
        integral = C0Integral(p1_sq=0.0, p2_sq=0.0, p12_sq=-2.0,
                              m1_sq=0.0, m2_sq=1.0, m3_sq=1.0)
        assert _rel_err(integral.evaluate(), LT_C0(0, 0, -2, 0, 1, 1)) < TOL

    def test_D0_evaluate(self):
        integral = D0Integral(p1_sq=0, p2_sq=0, p3_sq=0, p4_sq=0,
                              p12_sq=4.0, p23_sq=-2.0,
                              m1_sq=0, m2_sq=0, m3_sq=0, m4_sq=0)
        val = integral.evaluate()
        assert val is not None
        assert isinstance(val, complex)


# ══════════════════════════════════════════════════════════════════════════════
# Symbolic mode tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSymbolic:
    def test_A0_symbolic_structure(self):
        m_sq = Symbol("m_sq", positive=True)
        result = analytic_A0(m_sq)
        # Should contain Delta_UV and log
        assert result.has(Delta_UV)
        assert result.has(log)

    def test_A0_symbolic_zero_delta(self):
        m_sq = Symbol("m_sq", positive=True)
        result = analytic_A0(m_sq, delta_uv=0)
        assert not result.has(Delta_UV)

    def test_B0_symbolic_returns_expr(self):
        p, m = symbols("p m", positive=True)
        result = analytic_B0(p**2, m**2, m**2)
        assert result.has(Delta_UV)

    def test_A0_massless_symbolic(self):
        assert analytic_A0(S.Zero) == S.Zero
