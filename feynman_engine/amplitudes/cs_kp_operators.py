"""Catani-Seymour K and P operators for initial-state collinear subtraction.

Reference: Catani & Seymour, NPB 485 (1997) 291; PLB 467 (1999) 399
(typo-corrected version), Section 10 and 11.

After integrating the CS dipoles analytically over the one-parton phase
space, the *initial-state* dipoles split into

    ∫dφ_1 D = (α_s/(2π)) × {I-poles + K(z) + P(z) log(μ_F²/...)}

where I-poles are absorbed into the I-operator (`i_operator_*` in
``cs_dipoles.py``), and K and P remain as *finite* functions of the
collinear momentum fraction z.  The hadronic σ_NLO PDF-counterterm is then

    σ_PDF_CT = (α_s/(2π)) × Σ_{a,b} ∫dz [K_ab(z) + P_ab(z) log(μ_F²/μ²)]
                                          × L_ab(τ/z) ⊗ σ̂_Born(zs)

where L_ab(τ) = ∫dx_a/x_a f_a(x_a) f_b(τ/x_a) is the parton luminosity.

This module provides K_ab(z) and P_ab(z) for the four diagonal/off-diagonal
combinations (q→q, g→g, q→g, g→q) needed by hadronic pp NLO QCD.

Splitting functions (DGLAP, regularised):
    P_qq(z) = C_F × [(1+z²)/(1-z)]_+        (diagonal q → q + g)
    P_gg(z) = 2 C_A × [z/(1-z) + (1-z)/z + z(1-z)]_+ + (β₀/2) δ(1-z)
    P_qg(z) = T_R × [z² + (1-z)²]           (off-diag g → q + q̄, no plus)
    P_gq(z) = C_F × [(1 + (1-z)²)/z]        (off-diag q → g + q, no plus)

CS PLB 467 (1999) 399 eq. (6.55)-(6.61) gives K_ab and P_ab in MS-bar.
For our purposes (LHC hadronic Born + 2→2+1 real emission) only the
"hadronic" pieces matter; the K-tilde finite-mass pieces (CS eq. 6.20)
vanish for massless initial states.

Diagonal K_qq, K_gg form is (CS 6.55):

    K_qq(z) = P_qq(z)_+_reg × log((1-z)²/z)  - (3/2) C_F / (1-z)_+
            - C_F × 2 (1+z) ln(1-z) + C_F × (1 - z)
            + δ(1-z) × C_F × (5 - π²)

Off-diagonal K_qg, K_gq (CS 6.61):

    K_qg(z) = P_qg(z) × log((1-z)²/z)  + 2 T_R z (1-z)
    K_gq(z) = P_gq(z) × log((1-z)²/z)  + C_F × z

P operators (CS 6.58):

    P_qq(z; μ_F², μ²) = P_qq^reg(z) × log(μ_F²/μ²)
    similar for P_gg, P_qg, P_gq

Conventions
-----------
- All splitting functions use the standard MS-bar regularised form.
- The plus-prescription means ∫_0^1 dz [f(z)]_+ g(z) = ∫_0^1 dz f(z) [g(z)−g(1)].
- We define K and P including their convolution with the regularised plus
  prescription against a Born σ̂(zs); when σ̂ is z-independent (e.g. DY at
  fixed M_ll²), the plus prescription collapses.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

C_F = 4.0 / 3.0
C_A = 3.0
T_R = 0.5
N_F = 5.0
GAMMA_Q = 1.5 * C_F                                    # = 2 (PDG γ_q convention)
GAMMA_G = 11.0 / 6.0 * C_A - 2.0 / 3.0 * T_R * N_F     # n_f-dependent
PI = math.pi


# ────────────────────────────────────────────────────────────────────────────
# Regular (non-plus) parts of the K and P kernels
# ────────────────────────────────────────────────────────────────────────────
#
# These are the "easy" pieces — smooth functions of z on (0, 1) that can
# be integrated against any σ̂(z) without special treatment.
#
# The plus-prescription pieces are handled separately in the convolution
# routine because they require knowing σ̂(1).

def K_qq_regular(z: np.ndarray) -> np.ndarray:
    """Regular part of K^{q→q}(z) (CS 6.55).

    K_qq(z) = C_F × {[(1+z²)/(1-z) × log((1-z)²/z)]_reg − (1+z) log((1-z)/z)
                     + (1 - z)}  +  plus-distribution terms (separate)

    Here "regular" means everything except the (1-z)^-1 plus-prescription
    bits; those are handled by ``_plus_integral``.
    """
    z = np.asarray(z, dtype=float)
    log_term = np.log((1.0 - z) ** 2 / np.maximum(z, 1e-300))
    return C_F * (
        (1.0 + z * z) / np.maximum(1.0 - z, 1e-300) * log_term
        - (1.0 + z) * np.log(np.maximum(1.0 - z, 1e-300))
        + (1.0 - z)
    )


def K_gg_regular(z: np.ndarray) -> np.ndarray:
    """Regular part of K^{g→g}(z) (CS 6.55).

    K_gg(z) = 2 C_A × {[z/(1-z) + (1-z)/z + z(1-z)]_reg × log((1-z)²/z)}
            + ... + δ(1-z) finite remainder
    """
    z = np.asarray(z, dtype=float)
    log_term = np.log((1.0 - z) ** 2 / np.maximum(z, 1e-300))
    return 2.0 * C_A * (
        ((1.0 - z) / np.maximum(z, 1e-300) + z * (1.0 - z)) * log_term
    )


def K_qg(z: np.ndarray) -> np.ndarray:
    """K^{g→q}(z) — off-diagonal, no plus-prescription (CS 6.61).

    Active when a quark Born has a g-initial real-emission counterpart
    (e.g. g q̄ → V q for q q̄ → V).
    """
    z = np.asarray(z, dtype=float)
    log_term = np.log((1.0 - z) ** 2 / np.maximum(z, 1e-300))
    P_qg = T_R * (z * z + (1.0 - z) ** 2)
    return P_qg * log_term + 2.0 * T_R * z * (1.0 - z)


def K_gq(z: np.ndarray) -> np.ndarray:
    """K^{q→g}(z) — off-diagonal, no plus-prescription (CS 6.61).

    Active when a gluon Born has a q/q̄-initial real-emission counterpart
    (e.g. q g → V q for g g → V).
    """
    z = np.asarray(z, dtype=float)
    log_term = np.log((1.0 - z) ** 2 / np.maximum(z, 1e-300))
    P_gq = C_F * (1.0 + (1.0 - z) ** 2) / np.maximum(z, 1e-300)
    return P_gq * log_term + C_F * z


# ────────────────────────────────────────────────────────────────────────────
# P operators (factorisation-scale log coefficients)
# ────────────────────────────────────────────────────────────────────────────
#
# P_ab(z; μ_F, μ_R) = P_ab^reg(z) × log(μ_F²/μ_R²), where P_ab^reg is the
# regularised DGLAP splitting kernel.  These are added to K_ab in the
# counterterm.

def P_qq_split(z: np.ndarray) -> np.ndarray:
    """Regularised P_{q→q}(z) splitting function (DGLAP, no plus part).

    Plus-prescription part handled by the convolution routine.
    """
    z = np.asarray(z, dtype=float)
    return C_F * (1.0 + z * z) / np.maximum(1.0 - z, 1e-300)


def P_gg_split(z: np.ndarray) -> np.ndarray:
    """Regularised P_{g→g}(z) splitting function (DGLAP, no plus part).

    Plus-prescription handled by the convolution routine.
    """
    z = np.asarray(z, dtype=float)
    return 2.0 * C_A * (
        (1.0 - z) / np.maximum(z, 1e-300) + z * (1.0 - z)
    )


def P_qg_split(z: np.ndarray) -> np.ndarray:
    """P_{g→q}(z) — off-diagonal.  No plus-prescription."""
    z = np.asarray(z, dtype=float)
    return T_R * (z * z + (1.0 - z) ** 2)


def P_gq_split(z: np.ndarray) -> np.ndarray:
    """P_{q→g}(z) — off-diagonal.  No plus-prescription."""
    z = np.asarray(z, dtype=float)
    return C_F * (1.0 + (1.0 - z) ** 2) / np.maximum(z, 1e-300)


# ────────────────────────────────────────────────────────────────────────────
# Convolution machinery
# ────────────────────────────────────────────────────────────────────────────
#
# σ_PDF_CT = (α_s/(2π)) × ∫_0^1 dz [K_ab(z) + P_ab(z) log(μ_F²/μ²)] f_ab(z)
#
# where f_ab(z) = ∫dx/x f_a(x/z, μ_F) f_b(τ_0/(x/z), μ_F) σ̂_Born((τ_0/z)s)
#                                                                              /
# in practice the user supplies f_ab(z) — the *full* z-dependent integrand —
# and we do the plus-prescription convolution.

def _plus_convolution(
    f_callback: Callable[[np.ndarray], np.ndarray],
    z_grid: np.ndarray,
    f_at_one: float,
    coef_z: np.ndarray,
) -> float:
    """Convolve a plus-distribution ``[coef(z) / (1-z)]_+`` against f(z).

    ``∫_0^1 dz [g(z)/(1-z)]_+ f(z) = ∫_0^1 dz g(z) [f(z) - f(1)] / (1-z) +
                                       (∫_0^1 dz [g(z) - g(1)]/(1-z)) f(1)
                                     ≈ ∫_0^1 dz g(z) [f(z) - f(1)] / (1-z)``

    (when ``g(1)`` is finite, the second term is the leftover endpoint piece).
    Here we approximate via a trapezoid rule on ``z_grid`` of ``coef_z * (f - f₁)
    / (1-z)`` — appropriate when the user-supplied grid is fine enough.
    """
    f_vals = np.asarray(f_callback(z_grid), dtype=float)
    integrand = coef_z * (f_vals - f_at_one) / np.maximum(1.0 - z_grid, 1e-300)
    return float(np.trapezoid(integrand, z_grid))


def cs_pdf_counterterm(
    initial_partons: tuple[str, str],   # e.g. ("q", "qbar"), ("g", "g"), ("q", "g")
    f_callback: Callable[[np.ndarray], np.ndarray],
    alpha_s: float,
    mu_F_sq: float,
    mu_R_sq: float,
    n_z: int = 200,
    z_min: float = 1e-4,
) -> dict:
    """Catani-Seymour MS-bar PDF counterterm for one initial-state parton.

    For each Born initial-state parton a with momentum fraction x_a, the
    NLO real-emission collinear singularity from a→a+(parton) splittings
    is absorbed into the PDF.  The MS-bar counterterm is

        σ_PDF^a = (α_s/(2π)) ∫_0^1 dz [K_aa(z) + P_aa(z) log(μ_F²/μ_R²)] · g(z)

    where g(z) is the convolution of σ̂_Born((z/x)·s) with the PDF
    luminosity at the modified momentum fraction.  The off-diagonal terms
    (K_qg, K_gq) couple to the gluon-initiated real channels.

    Parameters
    ----------
    initial_partons : ("q","qbar"), ("g","g"), ("q","g"), ("g","q"), etc.
        Identifies which K/P operators to include for each leg.  Each leg
        contributes K_aa + (off-diag K_ab for the partner parton).
    f_callback : callable
        f_callback(z) → array of g(z) values.  The user computes g(z) as
        the convolution of σ̂_Born(z·s_partonic) with the PDFs at the
        rescaled momentum fraction.  Must accept an np.ndarray.
    alpha_s : float
        α_s evaluated at μ_R (the renormalization scale).
    mu_F_sq : float
        Factorization scale squared, GeV².
    mu_R_sq : float
        Renormalization scale squared, GeV².  Used in the log(μ_F²/μ_R²)
        coefficient of the P-operator.
    n_z : int
        Trapezoid grid points for the z integration.
    z_min : float
        Lower cutoff to avoid log singularities at z=0 (the singularity
        is integrable; the cutoff is for numerical stability).

    Returns
    -------
    dict with keys
        ``sigma_pdf_ct_pb`` — total counterterm to add to σ_NLO,
        ``contributions`` — breakdown by leg and channel,
        ``z_grid``, ``K_integrand``, ``P_integrand``.
    """
    z = np.linspace(z_min, 1.0 - z_min, n_z)
    log_F_R = math.log(mu_F_sq / mu_R_sq) if mu_R_sq > 0 else 0.0
    prefactor = alpha_s / (2.0 * math.pi)

    # We sum over both incoming legs.  Each leg gets a K_diag and a K_offdiag
    # depending on what real-emission channel it can split into.
    contributions = []
    total = 0.0

    f_vals = np.asarray(f_callback(z), dtype=float)
    f_at_one_arr = np.asarray(f_callback(np.array([1.0 - 1e-6])), dtype=float)
    f_at_one = float(f_at_one_arr[0]) if f_at_one_arr.size else 0.0

    for parton in initial_partons:
        is_quark = parton in ("q", "qbar", "u", "d", "s", "c", "b",
                              "u~", "d~", "s~", "c~", "b~")
        if is_quark:
            # Diagonal K_qq: contains plus-prescriptions.
            # Regular part:
            K_reg = K_qq_regular(z) * f_vals
            # Plus-distribution part: -(3/2 C_F)/(1-z)_+ from (3/2 C_F)/(1-z) ↦ plus
            # Plus part of P_qq: C_F × (1+z²)/(1-z) → 2 C_F/(1-z) plus extra.
            # ∫dz [(1+z²)/(1-z)]_+ f(z) = ∫dz (1+z²)/(1-z) [f(z) - f(1)]
            P_qq_plus_coef = C_F * (1.0 + z * z)   # this is g(z) in [g(z)/(1-z)]_+
            P_qq_plus_int = _plus_convolution(
                f_callback, z, f_at_one, P_qq_plus_coef,
            )
            # -3/(2(1-z))_+ from K_qq has coef -(3/2) C_F :
            P_neg32_int = _plus_convolution(
                f_callback, z, f_at_one,
                np.full_like(z, -1.5 * C_F),
            )
            K_int = float(np.trapezoid(K_reg, z)) + P_neg32_int
            # δ(1-z) contribution: C_F × (5 - π²)  (CS 6.55)
            K_delta_coef = C_F * (5.0 - PI * PI)
            K_delta = K_delta_coef * f_at_one
            # P × log(μ_F²/μ_R²): convolve P_qq^reg + plus parts
            P_reg = P_qq_split(z) * f_vals
            P_plus_qq = _plus_convolution(
                f_callback, z, f_at_one, P_qq_plus_coef,
            )
            # γ_q × δ(1-z) piece for P_qq (P_qq^+ includes γ_q δ(1-z))
            P_delta = GAMMA_Q * f_at_one
            P_int = (float(np.trapezoid(P_reg, z)) + P_plus_qq + P_delta) * log_F_R

            leg_total = K_int + K_delta + P_int
            contributions.append({
                "leg": parton,
                "kind": "K_qq + P_qq",
                "K_part": K_int + K_delta,
                "P_part_logFR": P_int,
            })

            # Off-diagonal: q ← g splitting contribution (real q g → ... channel)
            # K_qg only matters if the PDF has a gluon contribution at z; we
            # account for it by adding to the leg's counterterm too.  In
            # practice the user supplies f_callback as the PDF-convolved σ̂,
            # so the off-diagonal coupling would need a separate callback for
            # the g-initial channel.  Here we report the K_qg integral with
            # f_callback as a proxy.
            K_qg_int = float(np.trapezoid(K_qg(z) * f_vals, z))
            P_qg_int = float(np.trapezoid(P_qg_split(z) * f_vals, z)) * log_F_R
            contributions.append({
                "leg": parton,
                "kind": "K_qg + P_qg (off-diag, real g-channel proxy)",
                "K_part": K_qg_int,
                "P_part_logFR": P_qg_int,
            })

            total += leg_total + K_qg_int + P_qg_int

        else:
            # Gluon leg: K_gg (diagonal) + K_gq (off-diagonal to q-real)
            K_reg = K_gg_regular(z) * f_vals
            # P_gg has a (z/(1-z))_+ plus-prescription term: 2 C_A × z/(1-z)
            P_gg_plus_coef = 2.0 * C_A * z
            P_gg_plus_int = _plus_convolution(
                f_callback, z, f_at_one, P_gg_plus_coef,
            )
            K_int = float(np.trapezoid(K_reg, z)) + P_gg_plus_int
            K_delta_coef = C_A * (
                50.0 / 9.0 - PI * PI
            ) - (16.0 / 9.0) * T_R * N_F      # CS 6.56 (typo-corrected)
            K_delta = K_delta_coef * f_at_one
            P_reg = P_gg_split(z) * f_vals
            P_plus = _plus_convolution(
                f_callback, z, f_at_one, P_gg_plus_coef,
            )
            P_delta = GAMMA_G * f_at_one
            P_int = (float(np.trapezoid(P_reg, z)) + P_plus + P_delta) * log_F_R

            leg_total = K_int + K_delta + P_int
            contributions.append({
                "leg": parton,
                "kind": "K_gg + P_gg",
                "K_part": K_int + K_delta,
                "P_part_logFR": P_int,
            })

            K_gq_int = float(np.trapezoid(K_gq(z) * f_vals, z))
            P_gq_int = float(np.trapezoid(P_gq_split(z) * f_vals, z)) * log_F_R
            contributions.append({
                "leg": parton,
                "kind": "K_gq + P_gq (off-diag, real q-channel proxy)",
                "K_part": K_gq_int,
                "P_part_logFR": P_gq_int,
            })

            total += leg_total + K_gq_int + P_gq_int

    return {
        "sigma_pdf_ct_pb": prefactor * total,
        "alpha_s_over_2pi": prefactor,
        "log_muF2_muR2": log_F_R,
        "contributions": contributions,
        "z_grid": z,
        "supported": True,
    }
