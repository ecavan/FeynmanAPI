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
    r"""QED one-loop β-function coefficient per unit charge² × N_c.

    β(α) = +β₀ α² + ...  where  β₀ = 2/(3π) for ONE Dirac fermion of unit
    charge and one colour.  Multiply by Q_f² × N_c^f and sum over all
    fermion flavours active at the scale to get the total β₀.

    The running coupling for one fermion:
        1/α(μ²) = 1/α(μ₀²) − (2/3π) log(μ²/μ₀²)
    """
    return 2.0 / (3.0 * math.pi)


# ── Charged-fermion content for full α(q²) running ───────────────────────────
# (PDG 2024 pole masses; m_e is exact, others are effective MS-bar at low scale)
_LEPTON_MASSES_GEV = {
    "electron": 0.000511,  # m_e
    "muon":     0.10566,   # m_μ
    "tau":      1.77686,   # m_τ
}
# Light-quark contribution to α(q²) is dominated by non-perturbative QCD
# below ~m_τ.  At M_Z² the world-average leptonic + hadronic shift is
# Δα(M_Z²) ≈ 0.05912 (Jegerlehner 2017; PDG 2024).  Decomposed:
#   Δα_lep(M_Z²) ≈ 0.03150  (e + μ + τ)
#   Δα_had(M_Z²) ≈ 0.02768  (light + heavy quark hadronic, from R(s))
#   Δα_top(M_Z²) ≈ -0.00007 (negligible)
# We compute the leptonic piece exactly and add an empirical hadronic shift
# scaled to log(q²/(2 GeV)²) so that running below m_τ stays approximately
# leptonic and at q² ≈ M_Z² the total reproduces 1/α ≈ 128.95.
_DELTA_ALPHA_HAD_AT_MZ = 0.02768   # PDG 2024 / Jegerlehner


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
    mu0_sq: float = 0.0,
    alpha0: float = 1.0 / 137.036,
) -> float:
    r"""QED running fine-structure constant α(q²).

    Includes the one-loop contribution from every charged fermion above its
    threshold, plus an empirical hadronic shift Δα_had calibrated so that
    α(M_Z²) reproduces the PDG value 1/α(M_Z²) = 128.95.

    Δα(q²) = Σ_lep (α₀/(3π)) × Q_l² × log(q²/m_l²) × θ(q² − 4 m_l²)
            + Δα_had(M_Z²) × log(q²/(2 GeV)²) / log(M_Z²/(2 GeV)²)

    1/α(q²) = 1/α₀ − Δα(q²)

    The leptonic part is the analytic Schwinger-Dyson resummed answer; the
    hadronic part interpolates linearly in log(q²) from 0 (at the
    light-hadron threshold ~2 GeV) to the dispersion-relation result at M_Z²
    — a standard low-budget approximation that's good to ~10⁻⁴ in α at
    LEP energies.

    Parameters
    ----------
    q_sq : float
        Scale squared q² (GeV²).
    mu0_sq : float
        Reference scale.  ``0`` (default) means α₀ is taken as the
        Thomson-limit value 1/137.036; non-zero values keep the legacy
        single-flavour running for backward compatibility.
    alpha0 : float
        α at q² = 0 (default = 1/137.036).
    """
    if q_sq <= 0:
        return alpha0

    # Legacy single-flavour mode: keep behaviour for callers that pass a
    # non-zero reference scale (notably the unit-test sweep).
    if mu0_sq > 0:
        b0 = qed_beta0()
        log_ratio = math.log(q_sq / mu0_sq)
        denom = 1.0 - alpha0 * b0 * log_ratio
        if abs(denom) < 1e-15:
            raise ValueError(f"QED Landau pole encountered at q²={q_sq:.3g} GeV².")
        return alpha0 / denom

    # Full Δα(q²) running — leptonic loops + empirical hadronic shift.
    # Lepton contribution at the "-5/3" subtraction (Jegerlehner 2017
    # Eq. 2.18): Δα_f = (α/(3π))[log(q²/m_f²) − 5/3] for q² ≫ m_f².
    delta = 0.0
    for _, m_l in _LEPTON_MASSES_GEV.items():
        threshold = 4.0 * m_l ** 2
        if q_sq <= threshold:
            continue
        delta += (alpha0 / (3.0 * math.pi)) * (
            math.log(q_sq / m_l ** 2) - 5.0 / 3.0
        )

    # Empirical hadronic shift (Δα_had at M_Z² ≈ 0.02768).  Below ~2 GeV
    # the perturbative QCD calc breaks down, so we linearly ramp Δα_had
    # in log(q²) from 0 at q² = 4 GeV² up to the M_Z² value.
    Q_HAD_LOW_SQ = 4.0  # (2 GeV)²
    M_Z_SQ = 91.1876 ** 2
    if q_sq > Q_HAD_LOW_SQ:
        log_ratio_had = math.log(q_sq / Q_HAD_LOW_SQ)
        log_ratio_max = math.log(M_Z_SQ / Q_HAD_LOW_SQ)
        delta += _DELTA_ALPHA_HAD_AT_MZ * (log_ratio_had / log_ratio_max)

    inv_alpha = 1.0 / alpha0 - delta / alpha0
    if inv_alpha <= 0:
        raise ValueError(f"QED Landau pole encountered at q²={q_sq:.3g} GeV².")
    return 1.0 / inv_alpha


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
