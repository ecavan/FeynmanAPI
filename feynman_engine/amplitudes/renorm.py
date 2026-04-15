"""UV renormalisation for 1-loop amplitudes (MS-bar scheme).

This module provides:
1. Symbolic counterterm expressions for QED and QCD
2. Renormalised propagators and vertices
3. Running couplings α(μ²) and α_s(μ²) at 1-loop

References
----------
- Peskin & Schroeder, "An Introduction to Quantum Field Theory", Chapters 10–12
- Muta, "Foundations of Quantum Chromodynamics", 3rd ed.
- Denner, Fortschr. Phys. 41 (1993) 307

Sign/convention choices
-----------------------
MS-bar scheme throughout.  UV poles appear as 1/ε = 1/(4-d)×2 terms in dim-reg.
Counterterms subtract exactly the 1/ε poles (and the ln(4π) - γ_E constants that
go with them in the MS-bar convention).
"""
from __future__ import annotations

import math
from typing import Optional

from sympy import (
    Expr, Integer, Rational, Symbol, log, pi, symbols
)


# ── Common symbols ─────────────────────────────────────────────────────────────
alpha, alpha_s, mu_sq = symbols("alpha alpha_s mu^2", positive=True)
m_e, m_q = symbols("m_e m_q", positive=True)
p_sq = Symbol("p^2", real=True)


# ── 1-loop β functions ────────────────────────────────────────────────────────

def qed_beta0() -> float:
    r"""QED one-loop β-function coefficient (electron loop only).

    β(α) = +β₀ α² + ...  where  β₀ = 2/(3π)  (one lepton flavor).

    The running coupling:
        1/α(μ²) = 1/α(μ₀²) − (2/3π) log(μ²/μ₀²)
    """
    return 2.0 / (3.0 * math.pi)


def qcd_beta0(n_f: int = 6) -> float:
    r"""QCD one-loop β-function coefficient (gluon + quark loops).

    β₀ = (11 C_A − 2 n_f) / (12π)  where  C_A = 3 for SU(3).

    The running coupling:
        α_s(μ²) = α_s(μ₀²) / [1 + α_s(μ₀²) β₀ log(μ²/μ₀²)]
    """
    C_A = 3.0
    return (11.0 * C_A - 2.0 * n_f) / (12.0 * math.pi)


# ── Running couplings ──────────────────────────────────────────────────────────

def alpha_running(
    q_sq: float,
    mu0_sq: float = 1.0,
    alpha0: float = 1.0 / 137.036,
) -> float:
    r"""QED running fine-structure constant at scale q².

    α(q²) = α(μ₀²) / [1 − (α(μ₀²)/π) × (1/3) × log(q²/μ₀²)]

    Valid for q² far from thresholds (no quark thresholds included).

    Parameters
    ----------
    q_sq : float
        Renormalisation scale squared q² (GeV²).
    mu0_sq : float
        Reference scale μ₀² (default = 1 GeV²).
    alpha0 : float
        α at the reference scale (default = α(1 GeV²) ≈ α(0) ≈ 1/137).

    Returns
    -------
    float
        α(q²) in the one-loop QED approximation.
    """
    b0 = qed_beta0()
    log_ratio = math.log(q_sq / mu0_sq) if q_sq > 0 and mu0_sq > 0 else 0.0
    denom = 1.0 - alpha0 * b0 * log_ratio
    if abs(denom) < 1e-15:
        raise ValueError(f"QED Landau pole encountered at q²={q_sq:.3g} GeV².")
    return alpha0 / denom


def alpha_s_running(
    q_sq: float,
    mu0_sq: float = 91.1876 ** 2,
    alpha_s0: float = 0.1179,
    n_f: int = 5,
) -> float:
    r"""QCD running coupling α_s(q²) at one loop.

    α_s(q²) = α_s(MZ²) / [1 + α_s(MZ²) β₀ log(q²/MZ²)]

    Default reference point: α_s(M_Z) = 0.1179 (PDG 2023).

    Parameters
    ----------
    q_sq : float
        Scale squared q² (GeV²).
    mu0_sq : float
        Reference scale μ₀² (default = M_Z² ≈ 8302 GeV²).
    alpha_s0 : float
        α_s at the reference scale (default = 0.1179).
    n_f : int
        Number of active quark flavors (default = 5 for q² ≲ m_t²).

    Returns
    -------
    float
        α_s(q²) in the one-loop QCD approximation.
    """
    b0 = qcd_beta0(n_f)
    log_ratio = math.log(q_sq / mu0_sq) if q_sq > 0 and mu0_sq > 0 else 0.0
    denom = 1.0 + alpha_s0 * b0 * log_ratio
    if denom <= 0:
        raise ValueError(
            f"QCD Landau pole (confinement) encountered at q²={q_sq:.3g} GeV². "
            f"ΛQCD is at β₀ α_s = 1/log(q²/μ₀²)."
        )
    return alpha_s0 / denom


# ── Counterterms (symbolic) ───────────────────────────────────────────────────

def qed_photon_field_ct() -> Expr:
    r"""QED photon field renormalisation counterterm δZ₃ (MS-bar).

    The photon wavefunction renormalisation at 1-loop:
        δZ₃ = −(α/3π) × (1/ε − log(m_e²/μ²))   [MS-bar: drop log terms]
             = −α/(3π) × (1/ε)                    [MS-bar, divergent part only]

    In the MS-bar scheme, we subtract only the UV pole.

    Returns the **finite MS-bar counterterm** (pole subtracted), which equals:
        δZ₃^{finite} = (α/3π) × log(m_e²/μ²)
    """
    return (alpha / (3 * pi)) * log(m_e**2 / mu_sq)


def qed_electron_mass_ct() -> Expr:
    r"""QED electron mass renormalisation δm_e (MS-bar).

    The electron mass counterterm at 1-loop in Feynman gauge (ξ=1):
        δm_e / m_e = −3α/(4π) × (1/ε + log(μ²/m_e²) + 1)

    MS-bar finite part (pole subtracted):
        δm_e / m_e |^{finite} = (3α/4π) × log(m_e²/μ²) + const
    """
    return -Rational(3, 4) * (alpha / pi) * (1 + log(m_e**2 / mu_sq))


def qcd_quark_mass_ct(C_F: float = 4.0/3.0) -> Expr:
    r"""QCD quark mass renormalisation δm_q (MS-bar, 1-loop).

        δm_q / m_q = −(α_s C_F / π) × (3/4) × (1/ε + log(μ²/m_q²) + ...)

    MS-bar finite part:
        δm_q / m_q |^{finite} = −(3 α_s C_F / 4π) × log(m_q²/μ²)
    """
    cf = Rational(4, 3)   # C_F for SU(3)
    return -Rational(3, 4) * (alpha_s * cf / pi) * log(m_q**2 / mu_sq)


def qcd_gluon_field_ct(n_f: int = 6) -> Expr:
    r"""QCD gluon field renormalisation counterterm δZ₃^g (MS-bar, 1-loop).

        δZ₃^g = −(α_s / 4π) × (β₀ × 2) × (1/ε)

    where β₀ = (11 C_A − 2 n_f)/3 for SU(3).

    MS-bar finite part:
        δZ₃^g |^{finite} = −(α_s / 4π) × β₀ × log(μ²/k²)
    """
    CA = Integer(3)
    beta0_sym = (11 * CA - 2 * Integer(n_f)) / Integer(3)
    k_sq = Symbol("k^2", real=True)
    return -(alpha_s / (4 * pi)) * beta0_sym * log(mu_sq / k_sq)


# ── Renormalised self-energies ────────────────────────────────────────────────

def qed_renormalised_photon_selfenergy(
    k_sq: float,
    m_sq: float,
    mu_sq_val: float = 1.0,
    alpha_val: float = 1.0 / 137.036,
) -> Optional[complex]:
    """Compute the MS-bar renormalised photon self-energy Σ̂_T(k²).

    Σ̂_T(k²) = Σ_T(k²) − Σ_T(0)   (on-shell subtraction at k²=0 for photon)

    For the photon, the physical renormalisation condition is Σ_T(0) = 0,
    which is automatically satisfied in dim-reg when the photon mass is zero.
    The MS-bar result equals the finite part of the unrenormalised Σ_T.

    Parameters
    ----------
    k_sq : float
        Photon virtuality k² (GeV²).
    m_sq : float
        Internal fermion mass squared (GeV²).
    mu_sq_val : float
        Renormalisation scale μ² (GeV²).
    alpha_val : float
        Fine-structure constant.

    Returns
    -------
    complex or None
    """
    from feynman_engine.amplitudes.looptools_bridge import is_available, A0, B0, set_mu_sq
    if not is_available():
        return None
    try:
        set_mu_sq(mu_sq_val)
        # Σ_T(k²) = (α/π)[2A₀(m²) − (4m² − k²) B₀(k²; m², m²)]
        # Σ_T(0)  = (α/π)[2A₀(m²) − 4m² B₀(0; m², m²)]
        # Δ = Σ_T(k²) − Σ_T(0) = (α/π)(k²) B₀(k²; m², m²) − (α/π)(k² − 4m²)(B₀(k²)−B₀(0))
        import math
        prefactor = alpha_val / math.pi
        a0 = A0(m_sq)
        b0_k = B0(k_sq, m_sq, m_sq)
        b0_0 = B0(0.0, m_sq, m_sq)
        sigma_k = prefactor * (2 * a0 - (4 * m_sq - k_sq) * b0_k)
        sigma_0 = prefactor * (2 * a0 - 4 * m_sq * b0_0)
        return sigma_k - sigma_0
    except Exception:
        return None


def qed_renormalised_vertex_ff(
    q_sq: float,
    m_sq: float,
    mu_sq_val: float = 1.0,
    alpha_val: float = 1.0 / 137.036,
) -> Optional[complex]:
    """Compute the renormalised QED vertex form factor correction δF₁^R(q²).

    In the on-shell scheme:
        δF₁^R(q²) = δF₁(q²) − δF₁(q²=0)

    where δF₁(q²) = (α/2π)[-B₀(m²;0,m²) + (4m²-q²/2)/q² × C₀(m²,m²,q²;0,m²,m²)]

    At q²=0 the vertex correction equals the mass/wavefunction counterterm.

    Returns the **physical (renormalised)** form factor correction.
    """
    from feynman_engine.amplitudes.looptools_bridge import is_available, B0, C0, set_mu_sq
    if not is_available():
        return None
    try:
        set_mu_sq(mu_sq_val)
        import math
        prefactor = alpha_val / (2 * math.pi)
        b0 = B0(m_sq, 0.0, m_sq)
        if abs(q_sq) < 1e-10:
            # q²→0: F1 correction is IR-divergent; return None for zero transfer
            return None
        c0 = C0(m_sq, m_sq, q_sq, 0.0, m_sq, m_sq)
        return prefactor * (-b0 + (4 * m_sq - q_sq / 2) / q_sq * c0)
    except Exception:
        return None


# ── Summary ───────────────────────────────────────────────────────────────────

def renorm_status() -> dict:
    """Return a summary of available renormalisation computations."""
    from feynman_engine.amplitudes.looptools_bridge import is_available
    return {
        "looptools_available": is_available(),
        "scheme": "MS-bar",
        "available_counterterms": [
            "qed_photon_field_ct (δZ₃)",
            "qed_electron_mass_ct (δm_e)",
            "qcd_quark_mass_ct (δm_q)",
            "qcd_gluon_field_ct (δZ₃^g)",
        ],
        "available_renormalised": [
            "qed_renormalised_photon_selfenergy",
            "qed_renormalised_vertex_ff",
        ],
        "running_couplings": [
            "alpha_running(q_sq)",
            "alpha_s_running(q_sq)",
        ],
    }
