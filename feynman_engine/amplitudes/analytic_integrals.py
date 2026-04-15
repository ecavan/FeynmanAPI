"""Analytic closed-form evaluation of Passarino-Veltman scalar integrals.

Implements the standard 1-loop scalar integrals A₀, B₀, C₀, D₀ as explicit
functions of their kinematic arguments using logarithms, dilogarithms (Li₂),
and other special functions.  No external numerical library (LoopTools) is
required — everything is pure Python/SymPy/mpmath.

Each function supports two modes, auto-detected from input types:

  **Symbolic** — SymPy expression inputs → returns SymPy expression with
  ``Delta_UV``, ``log``, ``polylog``, ``pi``.

  **Numeric** — float/int inputs → returns ``complex`` with ``Delta_UV = 0``
  and ``mu² = 1.0`` (matching LoopTools defaults).

UV pole conventions (MS-bar)
----------------------------
``Delta_UV = 1/ε − γ_E + ln(4π)`` is a SymPy Symbol.  In numeric mode it is
set to 0, giving the LoopTools-compatible finite part.  In symbolic mode it
stays symbolic so the full dimensional-regularisation structure is visible.

iε prescription
---------------
For timelike momenta (p² > 0 above threshold), logarithms become complex.
In numeric mode we implement ``p² → p² + iε`` by adding a small imaginary
part (+1e-30j) before computing ``cmath.log``.

References
----------
- 't Hooft & Veltman, Nucl. Phys. B153 (1979) 365
- Denner, Fortschr. Phys. 41 (1993) 307
- Ellis & Zanderighi, JHEP 0802:002 (2008), arXiv:0712.1851
- Passarino & Veltman, Nucl. Phys. B160 (1979) 151
- Patel (Package-X), Comput. Phys. Commun. 197 (2015) 276
"""
from __future__ import annotations

import cmath
import math
from typing import Optional, Union

from sympy import (
    Expr, I, Rational, Symbol, latex, log, pi, polylog, sqrt, symbols,
    oo, S, simplify, Number,
)

# ── UV pole symbol ────────────────────────────────────────────────────────────

Delta_UV = Symbol("Delta_UV")
"""MS-bar UV pole: Δ_UV = 1/ε − γ_E + ln(4π).  Set to 0 for finite part."""

mu_sq_sym = Symbol("mu^2", positive=True)
"""Renormalisation scale μ² (symbolic)."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_numeric(*args) -> bool:
    """Return True if all arguments are numeric (float, int, complex)."""
    return all(isinstance(a, (int, float, complex)) for a in args)


def _is_zero(x, tol: float = 1e-30) -> bool:
    """Check if x is zero (numeric or symbolic)."""
    if isinstance(x, (int, float)):
        return abs(x) < tol
    if isinstance(x, complex):
        return abs(x) < tol
    # SymPy
    if hasattr(x, "is_zero"):
        if x.is_zero is True:
            return True
        if x.is_zero is False:
            return False
    try:
        return x == 0
    except Exception:
        return False


def _are_equal(a, b, tol: float = 1e-30) -> bool:
    """Check if a and b are equal (numeric or symbolic)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < tol * max(abs(a), abs(b), 1.0)
    # SymPy
    try:
        diff = simplify(a - b)
        if diff == 0:
            return True
        if hasattr(diff, "is_zero") and diff.is_zero is True:
            return True
    except Exception:
        pass
    return False


def _clog(z: complex) -> complex:
    """Complex logarithm with Feynman iε prescription.

    For real negative arguments, this gives ln|z| − iπ (approaching from above),
    matching the physical sheet for p² + iε.
    """
    if isinstance(z, (int, float)):
        if z > 0:
            return math.log(z) + 0j
        else:
            # z < 0: approach from above → ln|z| + iπ  (NOT −iπ)
            # But Feynman prescription p²+iε means we approach from above
            # log(x + iε) for x<0 → log|x| + iπ
            return cmath.log(complex(z, 1e-30))
    return cmath.log(z)


def _li2(z: complex) -> complex:
    """Numerical dilogarithm Li₂(z) = −∫₀ᶻ ln(1−t)/t dt."""
    import mpmath
    return complex(mpmath.polylog(2, z))


# ── A₀(m²): Scalar tadpole ──────────────────────────────────────────────────

def analytic_A0(
    m_sq,
    mu_sq=None,
    delta_uv=None,
):
    r"""Evaluate A₀(m²) = m² × (Δ_UV + 1 − ln(m²/μ²)).

    In dimensional regularisation (d = 4 − 2ε) with MS-bar subtraction:

        A₀(m²) = m² [1/ε − γ_E + ln(4π) + 1 − ln(m²/μ²)]
                = m² [Δ_UV + 1 − ln(m²/μ²)]

    A₀(0) = 0 in dim-reg (massless tadpole vanishes).

    Parameters
    ----------
    m_sq : float or Expr
        Internal mass squared m².
    mu_sq : float or Expr, optional
        Renormalisation scale μ².  Default: 1.0 (numeric) or ``mu^2`` (symbolic).
    delta_uv : float or Expr, optional
        UV pole term.  Default: 0 (numeric) or ``Delta_UV`` (symbolic).

    Returns
    -------
    complex or Expr
        The scalar tadpole integral.

    References
    ----------
    Denner (1993) eq. (A.1); 't Hooft-Veltman (1979).
    """
    if _is_zero(m_sq):
        return 0.0 if _is_numeric(m_sq) else S.Zero

    if _is_numeric(m_sq):
        mu2 = float(mu_sq) if mu_sq is not None else 1.0
        duv = float(delta_uv) if delta_uv is not None else 0.0
        return complex(m_sq) * (duv + 1.0 - _clog(m_sq / mu2))
    else:
        mu2 = mu_sq if mu_sq is not None else mu_sq_sym
        duv = delta_uv if delta_uv is not None else Delta_UV
        return m_sq * (duv + 1 - log(m_sq / mu2))


# ── B₀(p², m₁², m₂²): Scalar bubble ─────────────────────────────────────────

def _B0_both_massless_numeric(p_sq: float, mu2: float, duv: float) -> complex:
    r"""B₀(p²; 0, 0) = Δ_UV + 2 − ln(−p²/μ²).

    For p² > 0 (timelike): ln(−p²) = ln(p²) − iπ.
    """
    return duv + 2.0 - _clog(-complex(p_sq, 1e-30) / mu2)


def _B0_equal_mass_zero_p_numeric(m_sq: float, mu2: float, duv: float) -> complex:
    r"""B₀(0; m², m²) = Δ_UV − ln(m²/μ²)."""
    return duv - _clog(m_sq / mu2)


def _B0_equal_mass_numeric(p_sq: float, m_sq: float, mu2: float, duv: float) -> complex:
    r"""B₀(p²; m², m²) = Δ − ln(m²/μ²) + 2 − f(p²/m²).

    f(x) = β ln((β+1)/(β−1)) where β = √(1 − 4m²/p²) = √(1 − 4/x)

    Special regimes:
      • p² < 0 (spacelike): β is real > 1, result is real
      • 0 < p² < 4m² (below threshold): β = i|β|, use arctan form
      • p² > 4m² (above threshold): β is real 0<β<1, result is complex (absorptive part)
      • p² ≈ 4m² (threshold): Taylor expansion to avoid 0/0
    """
    ratio = p_sq / m_sq
    base = duv - _clog(m_sq / mu2) + 2.0

    # Near threshold: |4 − ratio| < 1e-8 → Taylor expand
    if abs(ratio - 4.0) < 1e-8:
        # At threshold β=0: B₀ → Δ − ln(m²) + 2   (the β·ln term → 0)
        return complex(base)

    beta_sq = 1.0 - 4.0 * m_sq / p_sq  # = 1 - 4/ratio

    if isinstance(beta_sq, complex) or beta_sq >= 0:
        # Above threshold (p² > 4m², β real 0<β<1) or spacelike (p² < 0, β > 1)
        beta = cmath.sqrt(beta_sq)
        if abs(beta) < 1e-15:
            return complex(base)
        return base - beta * _clog((beta + 1.0) / (beta - 1.0))
    else:
        # Below threshold: 0 < p² < 4m², β² < 0, β = i|β_im|
        beta_im = math.sqrt(-beta_sq)
        # β ln((β+1)/(β-1)) = i|β| × [ln((i|β|+1)/(i|β|-1))]
        # = i|β| × [iπ − 2i arctan(|β|)]   (for 0 < |β| < ∞)
        # = −|β| × (π − 2 arctan(|β|))
        # = −2|β| × arctan(1/|β|)     [using arctan(x) + arctan(1/x) = π/2]
        # Actually: 2|β| arctan(1/|β|) but with correct sign.
        # Let's use the complex formula directly for correctness:
        beta = 1j * beta_im
        return base - beta * _clog((beta + 1.0) / (beta - 1.0))


def _B0_one_massless_numeric(p_sq: float, m_sq: float, mu2: float, duv: float) -> complex:
    r"""B₀(p²; 0, m²) = Δ + 2 − ln(m²/μ²) + (m²/p² − 1)ln(1 − p²/m²).

    Special case B₀(m²; 0, m²) = Δ + 2 − ln(m²/μ²) (the log term vanishes).
    """
    base = duv + 2.0 - _clog(m_sq / mu2)
    if abs(p_sq - m_sq) < 1e-30 * max(abs(p_sq), abs(m_sq), 1.0):
        return complex(base)
    # General case
    coeff = m_sq / p_sq - 1.0
    arg = 1.0 - complex(p_sq, 1e-30) / m_sq
    return base + coeff * _clog(arg)


def _B0_zero_p_numeric(m1_sq: float, m2_sq: float, mu2: float, duv: float) -> complex:
    r"""B₀(0; m₁², m₂²) = Δ + 1 − [m₁² ln(m₁²/μ²) − m₂² ln(m₂²/μ²)] / (m₁² − m₂²).

    For m₁ = m₂: B₀(0; m², m²) = Δ − ln(m²/μ²).
    """
    if abs(m1_sq - m2_sq) < 1e-30 * max(abs(m1_sq), abs(m2_sq), 1.0):
        return duv - _clog(m1_sq / mu2)
    return (duv + 1.0
            - (m1_sq * _clog(m1_sq / mu2) - m2_sq * _clog(m2_sq / mu2))
            / (m1_sq - m2_sq))


def _B0_general_numeric(p_sq: float, m1_sq: float, m2_sq: float,
                         mu2: float, duv: float) -> complex:
    r"""General B₀(p²; m₁², m₂²) via Feynman parametrisation.

    B₀ = Δ − ∫₀¹ dx ln[D(x)/μ²]

    where D(x) = m₁²(1−x) + m₂²x − p²x(1−x) − iε

    This is the denominator of the Feynman-parametrised 1-loop integral.
    For D(x) > 0 over [0,1], the result is real.
    For D(x) < 0 somewhere (above threshold), ln(D−iε) = ln|D| − iπ.

    Ref: 't Hooft-Veltman (1979); Denner (1993) eq. (A.3).
    """
    from scipy.integrate import quad

    def _integrand_real(x):
        # D(x) = m₁²(1−x) + m₂²x − p²x(1−x)
        D = m1_sq * (1.0 - x) + m2_sq * x - p_sq * x * (1.0 - x)
        if D > 0:
            return math.log(D / mu2)
        elif D < 0:
            return math.log(abs(D) / mu2)
        else:
            return 0.0  # measure-zero point

    def _integrand_imag(x):
        D = m1_sq * (1.0 - x) + m2_sq * x - p_sq * x * (1.0 - x)
        if D < 0:
            return -math.pi  # Feynman iε: D−iε → ln|D| − iπ
        return 0.0

    real_part, _ = quad(_integrand_real, 0, 1, limit=100)
    imag_part, _ = quad(_integrand_imag, 0, 1, limit=100)
    integral = complex(real_part, imag_part)

    return duv - integral


# ── Symbolic B₀ helpers ──────────────────────────────────────────────────────

def _B0_both_massless_symbolic(p_sq, mu2, duv):
    """B₀(p²; 0, 0) = Δ + 2 − ln(−p²/μ²)."""
    return duv + 2 - log(-p_sq / mu2)


def _B0_equal_mass_zero_p_symbolic(m_sq, mu2, duv):
    """B₀(0; m², m²) = Δ − ln(m²/μ²)."""
    return duv - log(m_sq / mu2)


def _B0_equal_mass_symbolic(p_sq, m_sq, mu2, duv):
    """B₀(p²; m², m²) = Δ − ln(m²/μ²) + 2 − β ln((β+1)/(β−1))."""
    beta = sqrt(1 - 4 * m_sq / p_sq)
    return duv - log(m_sq / mu2) + 2 - beta * log((beta + 1) / (beta - 1))


def _B0_one_massless_symbolic(p_sq, m_sq, mu2, duv):
    """B₀(p²; 0, m²) = Δ + 2 − ln(m²/μ²) + (m²/p² − 1)ln(1 − p²/m²)."""
    return duv + 2 - log(m_sq / mu2) + (m_sq / p_sq - 1) * log(1 - p_sq / m_sq)


def _B0_zero_p_symbolic(m1_sq, m2_sq, mu2, duv):
    """B₀(0; m₁², m₂²) with m₁ ≠ m₂."""
    return (duv + 1
            - (m1_sq * log(m1_sq / mu2) - m2_sq * log(m2_sq / mu2))
            / (m1_sq - m2_sq))


def _B0_general_symbolic(p_sq, m1_sq, m2_sq, mu2, duv):
    """General B₀ via 't Hooft-Veltman: symbolic form."""
    # For symbolic, return the integral representation form
    # B₀ = Δ + 2 − ln(m₁m₂/μ²) − (m₁²−m₂²)/(2p²) ln(m₁²/m₂²)
    #      + (root terms involving sqrt of Källén function)
    # For now, express as the factored form
    lam = (p_sq - m1_sq - m2_sq)**2 - 4 * m1_sq * m2_sq
    sqrt_lam = sqrt(lam)
    x_plus = ((p_sq - m2_sq + m1_sq) + sqrt_lam) / (2 * p_sq)
    x_minus = ((p_sq - m2_sq + m1_sq) - sqrt_lam) / (2 * p_sq)

    # ∫₀¹ ln(x−a) dx = (1−a)ln(1−a) − (−a)ln(−a) − 1
    def _int_ln_sym(a):
        return (1 - a) * log(1 - a) - (-a) * log(-a) - 1

    integral = log(-p_sq / mu2) + _int_ln_sym(x_plus) + _int_ln_sym(x_minus)
    return duv - integral


# ── Public B₀ router ─────────────────────────────────────────────────────────

def analytic_B0(
    p_sq,
    m1_sq,
    m2_sq,
    mu_sq=None,
    delta_uv=None,
):
    r"""Evaluate B₀(p²; m₁², m₂²) analytically.

    Auto-detects the kinematic configuration and dispatches to the
    appropriate special-case formula.

    Parameters
    ----------
    p_sq : float or Expr
        External momentum squared.
    m1_sq, m2_sq : float or Expr
        Internal masses squared.
    mu_sq : float or Expr, optional
        Renormalisation scale μ².
    delta_uv : float or Expr, optional
        UV pole term (0 for finite part).

    Returns
    -------
    complex or Expr
        The scalar bubble integral.

    References
    ----------
    't Hooft-Veltman (1979) eqs. (B.4)–(B.8); Denner (1993) eq. (A.3).
    """
    numeric = _is_numeric(p_sq, m1_sq, m2_sq)

    if numeric:
        mu2 = float(mu_sq) if mu_sq is not None else 1.0
        duv = float(delta_uv) if delta_uv is not None else 0.0
        p, m1, m2 = float(p_sq), float(m1_sq), float(m2_sq)

        # Canonical ordering: put the smaller mass first for one-massless detection
        # (doesn't affect equal-mass or both-massless cases)

        # Case 1: Both massless
        if _is_zero(m1) and _is_zero(m2):
            if _is_zero(p):
                return 0.0 + 0j  # B₀(0;0,0) = 0 in dim-reg (scaleless)
            return _B0_both_massless_numeric(p, mu2, duv)

        # Case 2: p² = 0
        if _is_zero(p):
            if _is_zero(m1) and not _is_zero(m2):
                # B₀(0; 0, m²): limit of the zero-p formula
                return duv + 1.0 - _clog(m2 / mu2) + 0j
            if not _is_zero(m1) and _is_zero(m2):
                return duv + 1.0 - _clog(m1 / mu2) + 0j
            return _B0_zero_p_numeric(m1, m2, mu2, duv)

        # Case 3: Equal mass
        if _are_equal(m1, m2):
            if _is_zero(m1):
                return _B0_both_massless_numeric(p, mu2, duv)
            return _B0_equal_mass_numeric(p, m1, mu2, duv)

        # Case 4: One massless
        if _is_zero(m1) and not _is_zero(m2):
            return _B0_one_massless_numeric(p, m2, mu2, duv)
        if not _is_zero(m1) and _is_zero(m2):
            return _B0_one_massless_numeric(p, m1, mu2, duv)

        # Case 5: General (all nonzero, unequal masses)
        return _B0_general_numeric(p, m1, m2, mu2, duv)

    else:
        # Symbolic mode
        mu2 = mu_sq if mu_sq is not None else mu_sq_sym
        duv = delta_uv if delta_uv is not None else Delta_UV

        z1 = _is_zero(m1_sq)
        z2 = _is_zero(m2_sq)
        zp = _is_zero(p_sq)

        if z1 and z2:
            if zp:
                return S.Zero
            return _B0_both_massless_symbolic(p_sq, mu2, duv)

        if zp:
            if z1:
                return duv + 1 - log(m2_sq / mu2)
            if z2:
                return duv + 1 - log(m1_sq / mu2)
            if _are_equal(m1_sq, m2_sq):
                return _B0_equal_mass_zero_p_symbolic(m1_sq, mu2, duv)
            return _B0_zero_p_symbolic(m1_sq, m2_sq, mu2, duv)

        if _are_equal(m1_sq, m2_sq):
            if z1:
                return _B0_both_massless_symbolic(p_sq, mu2, duv)
            return _B0_equal_mass_symbolic(p_sq, m1_sq, mu2, duv)

        if z1:
            return _B0_one_massless_symbolic(p_sq, m2_sq, mu2, duv)
        if z2:
            return _B0_one_massless_symbolic(p_sq, m1_sq, mu2, duv)

        return _B0_general_symbolic(p_sq, m1_sq, m2_sq, mu2, duv)


# ── Tensor B-integral reductions ─────────────────────────────────────────────

def analytic_B1(
    p_sq,
    m1_sq,
    m2_sq,
    mu_sq=None,
    delta_uv=None,
):
    r"""B₁(p²; m₁², m₂²) via PV reduction identity.

    B₁ = 1/(2p²) × [A₀(m₁²) − A₀(m₂²) − (p² + m₁² − m₂²) B₀(p²; m₁², m₂²)]

    Ref: Passarino-Veltman (1979) eq. (4.7).
    """
    if _is_zero(p_sq):
        return None  # Need special limit; fall back to LoopTools

    a0_1 = analytic_A0(m1_sq, mu_sq=mu_sq, delta_uv=delta_uv)
    a0_2 = analytic_A0(m2_sq, mu_sq=mu_sq, delta_uv=delta_uv)
    b0 = analytic_B0(p_sq, m1_sq, m2_sq, mu_sq=mu_sq, delta_uv=delta_uv)
    if b0 is None:
        return None

    if _is_numeric(p_sq, m1_sq, m2_sq):
        p = float(p_sq)
        return (a0_1 - a0_2 - (p + float(m1_sq) - float(m2_sq)) * b0) / (2 * p)
    else:
        return (a0_1 - a0_2 - (p_sq + m1_sq - m2_sq) * b0) / (2 * p_sq)


def analytic_B00(
    p_sq,
    m1_sq,
    m2_sq,
    mu_sq=None,
    delta_uv=None,
):
    r"""B₀₀(p²; m₁², m₂²) via PV reduction identity.

    B₀₀ = 1/6 × [A₀(m₂²) + 2m₁² B₀ + (p² + m₁² − m₂²) B₁ + m₁² + m₂² − p²/3]

    In d=4 (the [d−1] in the denominator becomes 3):
    B₀₀ = (1/(2(d-1))) × [A₀(m₂²) + 2m₁²B₀ + f₁₂B₁ + m₁²+m₂²−p²/3]

    Simplified for d=4 (denominator = 6).

    Ref: Passarino-Veltman (1979) eq. (4.10).
    """
    a0_2 = analytic_A0(m2_sq, mu_sq=mu_sq, delta_uv=delta_uv)
    b0 = analytic_B0(p_sq, m1_sq, m2_sq, mu_sq=mu_sq, delta_uv=delta_uv)
    b1 = analytic_B1(p_sq, m1_sq, m2_sq, mu_sq=mu_sq, delta_uv=delta_uv)
    if b0 is None or b1 is None:
        return None

    if _is_numeric(p_sq, m1_sq, m2_sq):
        m1, m2, p = float(m1_sq), float(m2_sq), float(p_sq)
        f12 = p + m1 - m2
        return (a0_2 + 2 * m1 * b0 + f12 * b1 + m1 + m2 - p / 3.0) / 6.0
    else:
        f12 = p_sq + m1_sq - m2_sq
        return (a0_2 + 2 * m1_sq * b0 + f12 * b1 + m1_sq + m2_sq - p_sq / 3) / 6


# ── C₀: Scalar triangle ────────────────────────────────────────────────────

def _li2_feynman(z) -> complex:
    """Dilogarithm Li₂(z) with Feynman iε convention.

    mpmath evaluates Li₂ approaching the branch cut (z > 1) from below
    by default, while the Feynman prescription requires approach from above
    (corresponding to p² → p² + iε in the propagator).  For real z > 1,
    the two limits differ by a sign in the imaginary part, so we conjugate.
    """
    import mpmath
    result = complex(mpmath.polylog(2, z))
    if isinstance(z, (int, float)) and z > 1:
        return result.conjugate()
    return result


def _C0_all_massless_numeric(s: float, mu2: float) -> complex:
    r"""C₀(0,0,s; 0,0,0) = −1/s × [½ ln²(−s/μ²) + π²/6].

    Contains IR poles in dim-reg; this is the finite part at ε=0.
    """
    ln_s = _clog(-complex(s, 1e-30) / mu2)
    return -1.0 / complex(s) * (0.5 * ln_s**2 + math.pi**2 / 6.0)


def _C0_one_mass_analytic(s: float, m_sq: float) -> complex:
    r"""C₀(0, 0, s; 0, m², m²) = −1/s × Li₂(s/m²).

    Closed-form result obtained by performing the inner Feynman parameter
    integral analytically (yields −ln(1−sx/m²)/(sx)) and then recognising
    the outer integral as the dilogarithm.

    Valid for all s ≠ 0 and m² > 0, including timelike s > 0 (where the
    imaginary part arises from the branch cut of Li₂ for s/m² > 1).

    Ref: 't Hooft-Veltman (1979) §5; Denner (1993) eq. (A.6).
    """
    return -1.0 / complex(s) * _li2_feynman(s / m_sq)


def _C0_general_feynman_param(p1sq, p2sq, p12sq, m1sq, m2sq, m3sq) -> complex:
    r"""General C₀ via Feynman parameter double integral.

    C₀ = −∫₀¹ dx ∫₀^{1−x} dy  1/D(x,y)

    where the denominator from the standard simplex parametrisation is:

        D = −p₁²·xy − p₂²·yz − p₁₂²·xz + m₁²·x + m₂²·y + m₃²·z

    with z = 1−x−y.

    This formula is derived by Feynman-parametrising the three propagators
    D₁ = k²−m₁², D₂ = (k+p₁)²−m₂², D₃ = (k+p₁+p₂)²−m₃², completing
    the square over the loop momentum, and performing the d-dimensional
    integral.

    Gives machine-precision results for configurations where D > 0 over
    the entire simplex (spacelike momenta).  Returns ``None`` for timelike
    configurations where D passes through zero (the integral develops an
    imaginary part that requires special handling), or for IR-divergent
    configurations where D → 0 at the boundary causing a non-integrable
    singularity.

    Ref: 't Hooft-Veltman (1979); Denner (1993) eq. (A.6).
    """
    from scipy.integrate import dblquad
    from scipy.optimize import minimize

    def D_func(x, y):
        z = 1.0 - x - y
        return (-p1sq * x * y - p2sq * y * z - p12sq * x * z
                + m1sq * x + m2sq * y + m3sq * z)

    # Check whether D goes negative anywhere in the simplex (threshold
    # crossing).  D > 0 everywhere → spacelike, integral is real and dblquad
    # handles it.  D < 0 somewhere → timelike absorptive part that dblquad
    # cannot capture; fall back to LoopTools.
    #
    # D(x,y) is quadratic on the simplex, so we use scipy.optimize.minimize
    # from multiple starts plus explicit boundary/edge checks.
    from scipy.optimize import minimize

    min_D = float("inf")

    starts = [(0.33, 0.33), (0.1, 0.1), (0.7, 0.1), (0.1, 0.7),
              (0.5, 0.25), (0.25, 0.5), (0.05, 0.9), (0.9, 0.05)]
    for x0, y0 in starts:
        if x0 + y0 >= 1:
            continue
        try:
            res = minimize(
                lambda v: D_func(v[0], v[1]),
                [x0, y0],
                bounds=[(1e-14, 1 - 1e-14), (1e-14, 1 - 1e-14)],
                constraints={"type": "ineq", "fun": lambda v: 1 - v[0] - v[1] - 1e-14},
                method="SLSQP",
            )
            if res.success and res.fun < min_D:
                min_D = res.fun
        except Exception:
            pass

    # Dense grid on the simplex interior and near edges
    N = 20
    for i in range(N + 1):
        for j in range(N + 1 - i):
            tx = (i + 0.5) / (N + 1)
            ty = (j + 0.5) / (N + 1)
            if tx + ty >= 1:
                continue
            D_val = D_func(tx, ty)
            if D_val < min_D:
                min_D = D_val

    # Reject only if D actually goes negative (threshold crossing).
    # D approaching zero from above is fine — the boundary singularity
    # is integrable (at most logarithmic) and dblquad handles it.
    if min_D < -1e-12:
        return None

    def integrand(y, x):
        D = D_func(x, y)
        if abs(D) < 1e-30:
            return 0.0
        return 1.0 / D

    result, _ = dblquad(integrand, 0, 1, 0, lambda x: 1 - x)
    return complex(-result)


def analytic_C0(
    p1_sq,
    p2_sq,
    p12_sq,
    m1_sq,
    m2_sq,
    m3_sq,
    mu_sq=None,
    delta_uv=None,
) -> Optional[Union[complex, Expr]]:
    r"""Evaluate C₀(p₁², p₂², (p₁+p₂)²; m₁², m₂², m₃²) analytically.

    Returns ``None`` for kinematic configurations not yet implemented
    (caller should fall back to LoopTools).

    Supported cases (numeric):

      • **All massless** C₀(0, 0, s; 0, 0, 0)
            = −1/s × [½ ln²(−s/μ²) + π²/6]
            (finite part of IR-divergent integral)

      • **One-mass triangle** C₀(0, 0, s; 0, m², m²)
            = −1/s × Li₂(s/m²)
            Closed form valid for all s ≠ 0, including timelike.

      • **General spacelike** — any configuration where the Feynman
        parameter denominator D > 0 everywhere (no threshold crossing).
        Evaluated via ``scipy.integrate.dblquad`` to machine precision.

    Notes
    -----
    C₀(m², m², 0; 0, m², m²) — the Schwinger point — is **IR divergent**
    (two propagators go on-shell simultaneously at q² = 0).  LoopTools
    also returns NaN for this configuration.  Use the full vertex form
    factor extraction (which is finite) instead of the bare C₀.

    References
    ----------
    't Hooft-Veltman (1979) §5; Denner (1993) eq. (A.6);
    Ellis-Zanderighi (2008) §4.
    """
    numeric = _is_numeric(p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq)
    mu2 = float(mu_sq) if mu_sq is not None and numeric else (mu_sq or 1.0)

    if numeric:
        p1, p2, p12 = float(p1_sq), float(p2_sq), float(p12_sq)
        m1, m2, m3 = float(m1_sq), float(m2_sq), float(m3_sq)

        # All massless: C₀(0,0,s; 0,0,0)
        if (_is_zero(p1) and _is_zero(p2) and
                _is_zero(m1) and _is_zero(m2) and _is_zero(m3)):
            if _is_zero(p12):
                return None  # Scaleless
            return _C0_all_massless_numeric(p12, float(mu2))

        # One-mass triangle: C₀(0, 0, s; 0, m², m²)  — closed-form Li₂
        if (_is_zero(p1) and _is_zero(p2) and
                _is_zero(m1) and _are_equal(m2, m3) and not _is_zero(m2)):
            if _is_zero(p12):
                return None  # Degenerate (scaleless)
            return _C0_one_mass_analytic(p12, m2)

        # General: Feynman parameter integral (spacelike only)
        return _C0_general_feynman_param(p1, p2, p12, m1, m2, m3)

    else:
        # Symbolic mode: no closed-form expressions implemented yet
        return None


# ── D₀: Scalar box (special cases) ──────────────────────────────────────────

def analytic_D0(
    p1_sq, p2_sq, p3_sq, p4_sq, s, t,
    m1_sq, m2_sq, m3_sq, m4_sq,
    mu_sq=None,
    delta_uv=None,
) -> Optional[Union[complex, Expr]]:
    r"""Evaluate D₀ for supported special cases.

    Currently only the fully massless box:
        D₀(0,0,0,0,s,t; 0,0,0,0) = 2/(st) × [ln²(−s/μ²) + ln²(−t/μ²)
                                                − ln²(−(s+t)/μ²) − π²]

    where u = −s − t for massless external particles.

    Returns ``None`` for unsupported configurations.

    References
    ----------
    't Hooft-Veltman (1979) §6; Ellis-Zanderighi (2008) §5.
    """
    numeric = _is_numeric(p1_sq, p2_sq, p3_sq, p4_sq, s, t,
                          m1_sq, m2_sq, m3_sq, m4_sq)

    if not numeric:
        return None  # Only numeric for now

    p1, p2, p3, p4 = float(p1_sq), float(p2_sq), float(p3_sq), float(p4_sq)
    sv, tv = float(s), float(t)
    m1, m2, m3, m4 = float(m1_sq), float(m2_sq), float(m3_sq), float(m4_sq)
    mu2 = float(mu_sq) if mu_sq is not None else 1.0

    # Fully massless box
    all_ext_zero = all(_is_zero(x) for x in (p1, p2, p3, p4))
    all_int_zero = all(_is_zero(x) for x in (m1, m2, m3, m4))

    if all_ext_zero and all_int_zero:
        u = -sv - tv  # massless: s + t + u = 0
        ln_s = _clog(-complex(sv, 1e-30) / mu2)
        ln_t = _clog(-complex(tv, 1e-30) / mu2)
        ln_u = _clog(-complex(u, 1e-30) / mu2)
        return 2.0 / (complex(sv) * complex(tv)) * (
            ln_s**2 + ln_t**2 - ln_u**2 - math.pi**2
        )

    return None  # Unsupported
