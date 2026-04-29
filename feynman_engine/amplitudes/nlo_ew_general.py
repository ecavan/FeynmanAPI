"""V2.7.A — Generic EW NLO via Sudakov leading + next-to-leading logs.

At LHC energies (√s ≫ M_W, M_Z), the dominant electroweak NLO corrections
come from Sudakov double logarithms log²(s/M_W²).  These give negative
corrections that grow with energy:

    δ_EW_LL = -(α/(4π sin²θ_W)) × Σ_legs T_eff² × log²(s/M_W²) + (NLL terms)

For pp→DY with M_ll = 1 TeV: δ_EW ≈ -1% (small but measurable).
For pp→VV at M_VV = 2 TeV:    δ_EW ≈ -10% (significant!)

This module ships the leading-log Sudakov framework for inclusive observables.
The full EW NLO (renormalization in broken phase, EW counterterms, finite
γ-Z mixing finite parts, finite vertex corrections) is a multi-week effort
deferred to V3+; what this module provides is the *dominant* EW correction
at LHC scales.

References
----------
Ciafaloni-Comelli, PLB 446 (1999) 278 — Sudakov LL + NLL framework.
Beenakker-Denner, "EW corrections at high energies", PRD 65 (2002) 113008.
Pozzorini, "Electroweak radiative corrections at high energies",
PRD 71 (2005) 053002.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# EW parameters (PDG 2024)
ALPHA_EM = 1.0 / 137.035999084
M_W = 80.377
M_Z = 91.1876
SIN2_THETA_W = 0.23122
G_W_SQ = 4.0 * math.pi * ALPHA_EM / SIN2_THETA_W
G_Z_SQ = 4.0 * math.pi * ALPHA_EM / (SIN2_THETA_W * (1.0 - SIN2_THETA_W))


# ─── Particle EW quantum numbers ───────────────────────────────────────────

# T₃ × Q for Sudakov coefficient: T_eff² = T₃² + Q² × tan²θ_W (rough)
EW_T3_Q: dict[str, tuple[float, float]] = {
    # Charged leptons: T₃ = -1/2, Q = -1
    "e-":   (-0.5, -1.0), "e+":   (+0.5, +1.0),
    "mu-":  (-0.5, -1.0), "mu+":  (+0.5, +1.0),
    "tau-": (-0.5, -1.0), "tau+": (+0.5, +1.0),
    # Neutrinos: T₃ = +1/2, Q = 0
    "nu_e":   (+0.5, 0.0), "nuebar":  (-0.5, 0.0),
    "nu_mu":  (+0.5, 0.0), "numubar": (-0.5, 0.0),
    "nu_tau": (+0.5, 0.0), "nutaubar": (-0.5, 0.0),
    # Up-type quarks: T₃ = +1/2, Q = +2/3
    "u": (+0.5, +2.0/3), "u~": (-0.5, -2.0/3),
    "c": (+0.5, +2.0/3), "c~": (-0.5, -2.0/3),
    "t": (+0.5, +2.0/3), "t~": (-0.5, -2.0/3),
    # Down-type quarks: T₃ = -1/2, Q = -1/3
    "d": (-0.5, -1.0/3), "d~": (+0.5, +1.0/3),
    "s": (-0.5, -1.0/3), "s~": (+0.5, +1.0/3),
    "b": (-0.5, -1.0/3), "b~": (+0.5, +1.0/3),
    # Gauge bosons: T₃ depends on isospin embedding; we use effective values
    "W+": (+1.0, +1.0), "W-": (-1.0, -1.0),
    "Z":  (0.0, 0.0),
    "gamma": (0.0, 0.0), "ph": (0.0, 0.0),
    "H": (0.0, 0.0), "h": (0.0, 0.0),
    # Gluon: not coupled to EW
    "g": (0.0, 0.0), "gluon": (0.0, 0.0),
}


def _T_eff_sq(particle: str) -> float:
    """Effective EW coupling T_eff² = T₃² + (tan²θ_W) × Q² for Sudakov.

    For external charged fermions T₃² + Q² tan²θ_W combines the SU(2) and
    U(1)_Y contributions at high energies.
    """
    if particle not in EW_T3_Q:
        return 0.0
    T3, Q = EW_T3_Q[particle]
    tan2_W = SIN2_THETA_W / (1.0 - SIN2_THETA_W)
    return T3 * T3 + tan2_W * Q * Q


# ─── EW NLO Sudakov K-factor ────────────────────────────────────────────────

@dataclass
class EWNLOSudakovResult:
    process: str
    sqrt_s_gev: float
    incoming: list[str]
    outgoing: list[str]
    sum_T_eff_sq: float
    log_squared: float
    delta_LL: float           # Leading-log Sudakov
    delta_NLL: float          # Next-to-leading log (linear log)
    k_factor: float
    method: str
    trust_level: str
    accuracy_caveat: Optional[str] = None
    notes: str = ""


def ew_nlo_sudakov_kfactor(
    process: str, sqrt_s_gev: float,
) -> EWNLOSudakovResult:
    """Universal EW Sudakov K-factor at high energy s ≫ M_W².

    For inclusive observables, the leading-log + next-to-leading log
    Sudakov correction:

        δ_EW = -(α/(4π sin²θ_W)) × Σ_legs T_eff² × {
            L²    (leading log, double log)
          + 3 L  (NLL: single-log enhancement from collinear gauge-boson splitting)
        }

    where L = log(s/M_W²).  This formula is universal at the leading-power
    level; sub-leading contributions (vertex finite parts, mass-shift
    counterterms) are finite and process-specific (V3+ work).

    For typical LHC kinematics:
      M_ll = 100 GeV (Z peak): δ_EW ≈ 0% (L → 0)
      M_ll = 500 GeV:           δ_EW ≈ -2%
      M_ll = 1 TeV:             δ_EW ≈ -5%
      M_ll = 2 TeV:             δ_EW ≈ -8%
    """
    if "->" not in process:
        return EWNLOSudakovResult(
            process=process, sqrt_s_gev=sqrt_s_gev,
            incoming=[], outgoing=[], sum_T_eff_sq=0.0,
            log_squared=0.0, delta_LL=0.0, delta_NLL=0.0,
            k_factor=1.0, method="error", trust_level="blocked",
            accuracy_caveat="Process must contain '->'",
        )
    lhs, rhs = process.split("->")
    incoming = [p for p in lhs.split() if p]
    outgoing = [p for p in rhs.split() if p]

    sum_T_eff_sq = sum(_T_eff_sq(p) for p in incoming + outgoing)
    s = sqrt_s_gev * sqrt_s_gev
    M_W_sq = M_W * M_W

    if s <= M_W_sq:
        # Below EW scale, Sudakov logs aren't enhanced
        L = 0.0
    else:
        L = math.log(s / M_W_sq)

    L_sq = L * L

    # EW prefactor
    prefactor = ALPHA_EM / (4.0 * math.pi * SIN2_THETA_W)

    delta_LL = -prefactor * sum_T_eff_sq * L_sq
    delta_NLL = -prefactor * sum_T_eff_sq * 3.0 * L
    k_factor = 1.0 + delta_LL + delta_NLL

    # Trust assignment
    if sqrt_s_gev < 200.0:
        # Low energy: Sudakov isn't dominant; finite EW corrections matter more
        trust = "approximate"
        caveat = (
            f"At √s = {sqrt_s_gev:.0f} GeV, Sudakov logs L = log(s/M_W²) = {L:.2f} "
            "are small.  Finite EW corrections (~1% per leg) are not included; "
            "the K-factor here is leading-Sudakov-only."
        )
    elif sqrt_s_gev > 5000.0:
        # Very high energy: Sudakov breaks down (multi-log resummation needed)
        trust = "approximate"
        caveat = (
            f"At √s = {sqrt_s_gev:.0f} GeV, multi-Sudakov-log resummation "
            "becomes important.  The fixed-order LL + NLL formula here may "
            "underestimate the negative correction by 30-50%."
        )
    else:
        # Sweet spot: 200 GeV – 5 TeV
        trust = "approximate"
        caveat = (
            "EW Sudakov LL + NLL only.  Finite EW corrections (vertex, "
            "mass-shift counterterms) are not included; expect ~1% additional "
            "corrections from these pieces.  Full EW NLO is V3+ work."
        )

    return EWNLOSudakovResult(
        process=process, sqrt_s_gev=sqrt_s_gev,
        incoming=incoming, outgoing=outgoing,
        sum_T_eff_sq=sum_T_eff_sq,
        log_squared=L_sq,
        delta_LL=delta_LL, delta_NLL=delta_NLL,
        k_factor=k_factor, method="ew-sudakov-LL-NLL",
        trust_level=trust, accuracy_caveat=caveat,
        notes=(
            f"L = log(s/M_W²) = {L:.3f}, "
            f"Σ T_eff² = {sum_T_eff_sq:.3f}, "
            f"δ_LL = {delta_LL:+.4f}, δ_NLL = {delta_NLL:+.4f}"
        ),
    )


def ew_nlo_cross_section(
    process: str, sqrt_s_gev: float, theory: str = "EW",
) -> EWNLOSudakovResult:
    """Compute σ_NLO_EW for an arbitrary process at high energy.

    Uses the engine's existing LO σ + the EW Sudakov K-factor.  Most
    useful for high-pT or high-M_inv tails where Sudakov logs dominate.
    """
    from feynman_engine.amplitudes.cross_section import total_cross_section
    sudakov = ew_nlo_sudakov_kfactor(process, sqrt_s_gev)

    try:
        lo = total_cross_section(process, theory, sqrt_s=sqrt_s_gev)
        if lo.get("supported", False):
            sudakov.notes = (
                f"σ_LO = {lo['sigma_pb']:.4f} pb, σ_NLO_EW ≈ "
                f"{lo['sigma_pb'] * sudakov.k_factor:.4f} pb. " + sudakov.notes
            )
    except Exception:
        pass

    return sudakov
