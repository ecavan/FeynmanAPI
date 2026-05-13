"""Generic 1→3 Dalitz-plot phase-space integrator.

Computes the partial width Γ(P → 1 + 2 + 3) by integrating |M̄|² over the
Dalitz plot.  Complements ``three_body_decays.py`` (which uses the closed
Sargent-Cabibbo form for pure-Fermi V-A interactions) by handling decays
where the matrix element is *not* a simple V×A × V×A contraction —
e.g. radiative decays like Z → ℓ⁺ℓ⁻γ, H → bb̄γ.

Kinematics
----------
With four-momenta p_P → p₁ + p₂ + p₃ and masses M, m₁, m₂, m₃, define the
two independent Mandelstam-like invariants

    s₁₂ = (p₁ + p₂)²,   s₂₃ = (p₂ + p₃)²

The third invariant is fixed by momentum conservation:

    s₁₃ = M² + m₁² + m₂² + m₃² − s₁₂ − s₂₃

The partial width is

    Γ = 1/(2M) · (2π)⁻⁵ · (π²/4M²) · ∫ |M̄|²(s₁₂, s₂₃) ds₁₂ ds₂₃
      = 1/(256 π³ M³) · ∫ |M̄|²(s₁₂, s₂₃) ds₁₂ ds₂₃

(see PDG ``Review of Particle Physics: Kinematics'' Eq. 47.22.)

Dalitz boundary
---------------
For fixed s₁₂ ∈ [(m₁+m₂)², (M−m₃)²], the allowed s₂₃ range is

    s₂₃ ∈ [(E₂* + E₃*)² − (|p₂*| + |p₃*|)²,
           (E₂* + E₃*)² − (|p₂*| − |p₃*|)²]

with E_i* and |p_i*| evaluated in the (1,2) rest frame at s = s₁₂.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from scipy.integrate import dblquad


_HBAR_GEV_S = 6.582119569e-25     # ℏ in GeV·s — Γ → τ conversion


def _kallen(a: float, b: float, c: float) -> float:
    """Källén triangle function λ(a, b, c) = a² + b² + c² − 2(ab+bc+ca)."""
    return a * a + b * b + c * c - 2.0 * (a * b + b * c + c * a)


def _s23_bounds(
    s12: float, M_sq: float, m1_sq: float, m2_sq: float, m3_sq: float,
) -> Optional[tuple[float, float]]:
    """Allowed s₂₃ range for a given s₁₂ on the Dalitz plot."""
    if s12 <= 0.0:
        return None
    # |p₂*|² in (1,2) CM
    lam12 = _kallen(s12, m1_sq, m2_sq)
    lam12_M3 = _kallen(s12, M_sq, m3_sq)
    if lam12 <= 0.0 or lam12_M3 <= 0.0:
        return None
    E2_star = (s12 + m2_sq - m1_sq) / (2.0 * math.sqrt(s12))
    E3_star = (M_sq - s12 - m3_sq) / (2.0 * math.sqrt(s12))
    p2_star = math.sqrt(max(E2_star * E2_star - m2_sq, 0.0))
    p3_star = math.sqrt(max(E3_star * E3_star - m3_sq, 0.0))
    base = (E2_star + E3_star) ** 2
    s_min = base - (p2_star + p3_star) ** 2
    s_max = base - (p2_star - p3_star) ** 2
    if s_max <= s_min:
        return None
    return s_min, s_max


def dalitz_partial_width(
    M_parent: float,
    m1: float,
    m2: float,
    m3: float,
    msq_callback: Callable[[float, float], float],
    *,
    epsrel: float = 1e-3,
    epsabs: float = 0.0,
) -> dict:
    """Partial width Γ(P → 1 + 2 + 3) via Dalitz-plot integration.

    Parameters
    ----------
    M_parent : float
        Mass of the decaying parent in GeV.
    m1, m2, m3 : float
        Masses of the three daughters in GeV.
    msq_callback : callable
        ``msq_callback(s12, s23) → |M̄|²`` (spin-averaged for the parent,
        spin-summed for the daughters).  Dimensionless if M_parent and
        masses are in GeV.
    epsrel, epsabs : float
        Tolerances passed to ``scipy.integrate.dblquad``.

    Returns
    -------
    dict
        ``Gamma_gev`` (partial width in GeV),
        ``tau_seconds`` (corresponding mean lifetime if this were the sole
        channel),
        ``s12_range``, ``s23_range_at_midpoint``,
        ``supported`` (False if the kinematics make Γ → 0 trivially, e.g.
        ∑m_i ≥ M_parent).
    """
    if M_parent <= (m1 + m2 + m3):
        return {
            "Gamma_gev": 0.0,
            "supported": False,
            "error": (
                f"Kinematic boundary: m_parent = {M_parent} ≤ m1+m2+m3 = "
                f"{m1+m2+m3}; decay is energetically forbidden."
            ),
        }

    M_sq = M_parent ** 2
    m1_sq, m2_sq, m3_sq = m1 * m1, m2 * m2, m3 * m3

    s12_min = (m1 + m2) ** 2
    s12_max = (M_parent - m3) ** 2
    if s12_max <= s12_min:
        return {
            "Gamma_gev": 0.0,
            "supported": False,
            "error": "Degenerate Dalitz plot (s12_max ≤ s12_min).",
        }

    def s23_lower(s12: float) -> float:
        bounds = _s23_bounds(s12, M_sq, m1_sq, m2_sq, m3_sq)
        return bounds[0] if bounds else 0.0

    def s23_upper(s12: float) -> float:
        bounds = _s23_bounds(s12, M_sq, m1_sq, m2_sq, m3_sq)
        return bounds[1] if bounds else 0.0

    def integrand(s23: float, s12: float) -> float:
        try:
            return float(msq_callback(s12, s23))
        except Exception:
            return 0.0

    result, _ = dblquad(
        integrand,
        s12_min, s12_max,
        s23_lower, s23_upper,
        epsrel=epsrel, epsabs=epsabs,
    )
    # PDG 47.22: dΓ = 1/(2M) · |M̄|² · dΦ₃ ; dΦ₃ = (1/(2π)⁵)(π²/4M²) ds₁₂ ds₂₃
    # ⇒ Γ = 1/(256 π³ M³) · ∫|M̄|² ds₁₂ ds₂₃
    prefactor = 1.0 / (256.0 * math.pi ** 3 * M_parent ** 3)
    gamma_gev = prefactor * result

    s12_mid = 0.5 * (s12_min + s12_max)
    bounds_mid = _s23_bounds(s12_mid, M_sq, m1_sq, m2_sq, m3_sq) or (0.0, 0.0)

    return {
        "Gamma_gev": gamma_gev,
        "tau_seconds": (_HBAR_GEV_S / gamma_gev) if gamma_gev > 0 else float("inf"),
        "s12_range": (s12_min, s12_max),
        "s23_range_at_midpoint": bounds_mid,
        "M_parent_gev": M_parent,
        "daughter_masses_gev": (m1, m2, m3),
        "supported": True,
    }


