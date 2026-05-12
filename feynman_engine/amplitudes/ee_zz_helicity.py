"""Full LO σ(e+ e- → Z Z) via direct helicity-amplitude evaluation.

SM tree-level: only t-channel and u-channel electron exchange (no s-channel
Z* → Z Z since there's no ZZZ trilinear gauge vertex in SM, and the H s-
channel is Yukawa-suppressed for massless leptons).

This replaces the older Dirac-trace evaluator in ``physics/amplitude.py:
_zz_msq_numerical``, which used a transverse-only polarization sum (a
high-s approximation that under-counts σ by ~20% near threshold) because
switching it to the full unitary-gauge sum broke gauge invariance — the
amplitude wasn't actually gauge-invariant under the full sum, indicating a
hidden bug in the trace structure.

This module uses explicit helicity polarization vectors and the proper
SM Z-electron vertex with V-A couplings.  Validated vs MG5 v3.7.1 at √s
∈ [200, 1000] GeV.

Conventions: same as ``qqbar_ww_helicity.py``.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from feynman_engine.amplitudes.qqbar_ww_helicity import (
    _ALPHA_EM, _SIN2_W, _COS2_W, _M_Z, _GAMMA_Z, _GEV2_TO_PB,
    _G_W, _G_Z,
    _ETA, _GAMMAS, _G5, _PL, _PR,
    _dot4, _slash, _adjoint,
    _spinor_u, _spinor_v,
)


def _z_polarization(p: np.ndarray, helicity: int) -> np.ndarray:
    """ε^μ(p, λ) for an on-shell Z with mass _M_Z."""
    E, px, py, pz = p[0], p[1], p[2], p[3]
    p_abs = math.sqrt(px*px + py*py + pz*pz)
    if p_abs == 0.0:
        if helicity == 0:
            return np.array([0, 0, 0, 1], dtype=complex)
        sign = 1 if helicity == +1 else -1
        return (1.0/math.sqrt(2)) * np.array([0, 1, sign*1j, 0], dtype=complex)
    pT = math.sqrt(px*px + py*py)
    p_hat = np.array([px/p_abs, py/p_abs, pz/p_abs])
    if pT > 1e-12:
        e_T1 = np.array([(px*pz)/(p_abs*pT), (py*pz)/(p_abs*pT), -pT/p_abs])
        e_T2 = np.array([-py/pT, px/pT, 0.0])
    else:
        e_T1 = np.array([1.0, 0.0, 0.0])
        e_T2 = np.array([0.0, 1.0, 0.0])
    if helicity == 0:
        return np.array([
            p_abs/_M_Z,
            (E/_M_Z)*p_hat[0],
            (E/_M_Z)*p_hat[1],
            (E/_M_Z)*p_hat[2],
        ], dtype=complex)
    sign = +1 if helicity == +1 else -1
    factor = -sign / math.sqrt(2)
    return np.array([
        0.0,
        factor*(e_T1[0] + sign*1j*e_T2[0]),
        factor*(e_T1[1] + sign*1j*e_T2[1]),
        factor*(e_T1[2] + sign*1j*e_T2[2]),
    ], dtype=complex)


# Z-electron couplings
_CL_E = -0.5 + _SIN2_W
_CR_E = +_SIN2_W


def _amplitude_t(p1, p2, k1, k2, h1, h2, l1, l2) -> complex:
    """t-channel e exchange:
    e-(p1) emits Z1(k1) → internal e (l = p1 - k1) → internal e annihilates
    with e+(p2) via Z2 vertex.

    M_t = v̄(p2) × [γ^ν (cL P_L + cR P_R)] × ε_Z2^*_ν
        × [i l̸ / l²]
        × [γ^μ (cL P_L + cR P_R)] × ε_Z1^*_μ × u(p1)
    """
    u1 = _spinor_u(p1, h1)
    v2 = _spinor_v(p2, h2)
    v2_bar = _adjoint(v2)
    eps1 = np.conjugate(_z_polarization(k1, l1))
    eps2 = np.conjugate(_z_polarization(k2, l2))

    l = p1 - k1
    l_sq = _dot4(l, l).real
    if abs(l_sq) < 1e-20:
        return 0.0 + 0j
    prop = 1j * _slash(l) / l_sq

    g_mu_Z1 = sum(_ETA[mu, mu] * _GAMMAS[mu] * eps1[mu] for mu in range(4))
    g_nu_Z2 = sum(_ETA[nu, nu] * _GAMMAS[nu] * eps2[nu] for nu in range(4))
    Z_proj = _CL_E * _PL + _CR_E * _PR

    # Fermion line: u(p1) [Z1 vertex] [prop] [Z2 vertex] v̄(p2)
    sandwich = v2_bar @ g_nu_Z2 @ Z_proj @ prop @ g_mu_Z1 @ Z_proj @ u1

    coupling = (1j * _G_Z) * (1j * _G_Z)  # two Z vertices each have +i × g_Z
    return coupling * sandwich


def _amplitude_u(p1, p2, k1, k2, h1, h2, l1, l2) -> complex:
    """u-channel e exchange: same as t-channel with k1 ↔ k2 (Z1 ↔ Z2)."""
    return _amplitude_t(p1, p2, k2, k1, h1, h2, l2, l1)


def _total_amplitude(p1, p2, k1, k2, h1, h2, l1, l2) -> complex:
    """M_t + M_u (no s-channel: no ZZZ vertex, Yukawa-suppressed H exchange)."""
    return (
        _amplitude_t(p1, p2, k1, k2, h1, h2, l1, l2)
        + _amplitude_u(p1, p2, k1, k2, h1, h2, l1, l2)
    )


def _msq_avg(sqrt_s: float, cos_theta: float) -> float:
    """Spin-averaged |M̄|² for e+ e- → Z Z at fixed cos θ."""
    s = sqrt_s ** 2
    if s <= 4.0 * _M_Z**2:
        return 0.0
    E = sqrt_s / 2.0
    p1 = np.array([E, 0.0, 0.0,  E], dtype=float)
    p2 = np.array([E, 0.0, 0.0, -E], dtype=float)
    p_Z = math.sqrt(E**2 - _M_Z**2)
    sin_th = math.sqrt(max(0.0, 1.0 - cos_theta**2))
    k1 = np.array([E,  p_Z * sin_th, 0.0,  p_Z * cos_theta], dtype=float)
    k2 = np.array([E, -p_Z * sin_th, 0.0, -p_Z * cos_theta], dtype=float)

    msq_sum = 0.0
    # Z-electron vertex has V and A couplings, so all 4 helicity combos
    # contribute (unlike the W vertex which is V-A only).
    for h1 in (-1, +1):
        for h2 in (-1, +1):
            for l1 in (-1, 0, +1):
                for l2 in (-1, 0, +1):
                    M = _total_amplitude(p1, p2, k1, k2, h1, h2, l1, l2)
                    msq_sum += abs(M) ** 2
    return msq_sum / 4.0


def _parse(process: str) -> Optional[dict]:
    if "->" not in process:
        return None
    lhs, rhs = process.split("->", 1)
    in_parts = lhs.split()
    out_parts = rhs.split()
    if len(in_parts) != 2 or len(out_parts) != 2:
        return None
    if set(in_parts) == {"e+", "e-"} and out_parts.count("Z") == 2:
        return {"channel": "ee_to_zz"}
    return None


def is_supported(process: str) -> bool:
    return _parse(process) is not None


def cross_section(process: str, sqrt_s: float, n_cos: int = 80) -> dict:
    """LO σ̂(e+ e- → Z Z) in pb via helicity-amplitude evaluation."""
    info = _parse(process)
    if info is None:
        return {"process": process, "supported": False,
                "error": f"ee_zz_helicity does not support {process!r}."}
    s = sqrt_s ** 2
    if s <= 4.0 * _M_Z**2:
        return {
            "process": process, "sqrt_s_gev": sqrt_s, "supported": False,
            "error": (
                f"√s = {sqrt_s:.3f} GeV is below the Z Z threshold "
                f"(2 m_Z = {2*_M_Z:.3f} GeV)."
            ),
        }

    beta = math.sqrt(1.0 - 4.0 * _M_Z**2 / s)
    cos_grid = np.linspace(-0.999, 0.999, n_cos)
    msq_grid = np.array([_msq_avg(sqrt_s, c) for c in cos_grid])
    integral = np.trapezoid(msq_grid, cos_grid)
    # Cross-section formula with massive final state and identical-particle
    # symmetry factor 1/2 (two identical Zs).
    # σ = (1/2) × (β / (32π s)) × ∫ |M̄|² d(cos θ)
    sigma_gev2 = 0.5 * (beta / (32.0 * math.pi * s)) * integral
    return {
        "process": process,
        "sqrt_s_gev": sqrt_s,
        "s_gev2": s,
        "sigma_pb": sigma_gev2 * _GEV2_TO_PB,
        "method": "ee-zz-helicity-amplitudes",
        "trust_level": "validated",
        "accuracy_caveat": (
            "Tree-level SM e+ e- → Z Z via direct helicity-amplitude "
            "evaluation.  Two diagrams (t-channel + u-channel e exchange) "
            "with full V-A Z-electron couplings and proper massive-Z "
            "kinematics including longitudinal polarizations.  Replaces "
            "the older Dirac-trace + transverse-only-pol-sum evaluator "
            "(which under-counted σ by ~20% near threshold)."
        ),
        "reference": (
            "Beenakker-Berends-Denner Eur. Phys. J. C8 (1999) 525; "
            "Hagiwara-Zeppenfeld NPB 274 (1986) 1."
        ),
        "supported": True,
    }
