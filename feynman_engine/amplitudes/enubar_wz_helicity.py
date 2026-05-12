"""Full LO matrix element for e- ν̄_e → W- Z (and crossed lepton variants).

The SM tree-level result includes three diagrams:

  1. t-channel ν_e exchange — e- emits W- becoming ν_e_internal, which
     annihilates ν̄_e via the Z vertex.
  2. u-channel e- exchange — e- emits Z becoming e-_internal, which
     annihilates ν̄_e via the W vertex.
  3. s-channel W- exchange via the W+W-Z triple-gauge vertex.

All three diagrams and their interferences are summed numerically from the
helicity amplitudes, in close analogy to ``qqbar_ww_helicity.py`` which
handles the q q̄ / l+ l- → W+ W- process.

This is the SM result for the charged-current diboson process — relevant for
LEP-3 / FCC-ee / muon-collider diboson studies, and as a sub-process of
hadronic W production with explicit lepton-collider initial states.

Conventions: Schwartz "QFT and the Standard Model" §29.107 for the W+W-Z
triple-gauge vertex; Peskin-Schroeder Ch. 3 for spinor / Dirac conventions
in the chiral (Weyl) representation.  All physical constants imported from
``qqbar_ww_helicity`` for scheme consistency.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from feynman_engine.amplitudes.qqbar_ww_helicity import (
    _ALPHA_EM, _SIN2_W, _COS2_W, _M_W, _M_Z, _GAMMA_Z,
    _E_EM, _G_W, _G_Z, _GEV2_TO_PB,
    _ETA, _GAMMAS, _G5, _PL, _PR,
    _dot4, _slash, _adjoint,
    _spinor_u, _spinor_v,
    _w_polarization,
)


# Width of the W in the same scheme used for the Z width above.
# (Used only as a regulator for the s-channel W propagator; the W is off-shell
# in this 2→2 process, so the |t-M_W²|⁻¹ singularity is never hit and the
# width contributes negligibly.)
_GAMMA_W = 2.085


# Z polarization vector — identical structure to W polarization but with m_Z.
def _z_polarization(p: np.ndarray, helicity: int) -> np.ndarray:
    """ε^μ(p, λ) for an on-shell Z with mass _M_Z."""
    E, px, py, pz = p[0], p[1], p[2], p[3]
    p_abs = math.sqrt(px*px + py*py + pz*pz)
    if p_abs == 0.0:
        if helicity == 0:
            return np.array([0, 0, 0, 1], dtype=complex)
        sign = 1 if helicity == +1 else -1
        return (1.0 / math.sqrt(2)) * np.array(
            [0, 1, sign * 1j, 0], dtype=complex,
        )
    pT = math.sqrt(px*px + py*py)
    p_hat = np.array([px/p_abs, py/p_abs, pz/p_abs])
    if pT > 1e-12:
        e_T1 = np.array([
            (px*pz)/(p_abs*pT),
            (py*pz)/(p_abs*pT),
            -pT/p_abs,
        ])
        e_T2 = np.array([-py/pT, px/pT, 0.0])
    else:
        e_T1 = np.array([1.0, 0.0, 0.0])
        e_T2 = np.array([0.0, 1.0, 0.0])
    if helicity == 0:
        return np.array([
            p_abs / _M_Z,
            (E / _M_Z) * p_hat[0],
            (E / _M_Z) * p_hat[1],
            (E / _M_Z) * p_hat[2],
        ], dtype=complex)
    sign = +1 if helicity == +1 else -1
    factor = -sign / math.sqrt(2)
    return np.array([
        0.0,
        factor * (e_T1[0] + sign * 1j * e_T2[0]),
        factor * (e_T1[1] + sign * 1j * e_T2[1]),
        factor * (e_T1[2] + sign * 1j * e_T2[2]),
    ], dtype=complex)


def _wwz_tgc_tensor(
    q_Wplus: np.ndarray, q_Wminus: np.ndarray, q_Z: np.ndarray,
) -> np.ndarray:
    """W+W-Z TGC, all-incoming-momenta convention.

    Γ^{μνα}(k_+, k_-, q_Z) = g^{μν}(k_- - k_+)^α
                            + g^{να}(q_Z - k_-)^μ
                            + g^{αμ}(k_+ - q_Z)^ν

    where μ = W+ Lorentz index, ν = W- Lorentz index, α = Z Lorentz index,
    and k_+ + k_- + q_Z = 0 (charge / 4-momentum conservation).
    The overall coupling factor (-i g_Z cos²θ_W) is applied by the caller.
    """
    T = np.zeros((4, 4, 4), dtype=complex)
    A = q_Wminus - q_Wplus
    B = q_Z - q_Wminus
    C = q_Wplus - q_Z
    for mu in range(4):
        for nu in range(4):
            for a in range(4):
                T[mu, nu, a] = (
                    _ETA[mu, nu] * A[a]
                    + _ETA[nu, a] * B[mu]
                    + _ETA[a, mu] * C[nu]
                )
    return T


# ─── Diagrams ─────────────────────────────────────────────────────────────


def _amplitude_t_nu_exchange(
    p1, p2, k_W, k_Z, h1, h2, lW, lZ,
) -> complex:
    """t-channel ν_e exchange.

    e-(p1) → W-(k_W) + ν_e_internal (W vertex, V-A)
    ν_e_internal + ν̄_e(p2) → Z(k_Z) (Z vertex, neutrinos: cL=1/2, cR=0)

    Internal ν_e momentum l = p1 - k_W.  Massless propagator i l̸/l².
    """
    u1 = _spinor_u(p1, h1)
    v2 = _spinor_v(p2, h2)
    v2_bar = _adjoint(v2)
    eps_W_star = np.conjugate(_w_polarization(k_W, lW))
    eps_Z_star = np.conjugate(_z_polarization(k_Z, lZ))

    l = p1 - k_W
    l_sq = _dot4(l, l).real
    if abs(l_sq) < 1e-20:
        return 0.0 + 0j
    prop = 1j * _slash(l) / l_sq

    g_mu_W = sum(_ETA[mu, mu] * _GAMMAS[mu] * eps_W_star[mu] for mu in range(4))
    g_nu_Z = sum(_ETA[nu, nu] * _GAMMAS[nu] * eps_Z_star[nu] for nu in range(4))

    # Vertex order on the fermion line going from u(p1) at e- vertex, through
    # the internal ν propagator, to v̄(p2) at the ν̄ Z vertex.
    sandwich = v2_bar @ g_nu_Z @ _PL @ prop @ g_mu_W @ _PL @ u1

    # Couplings: (-ig_W/√2)(γ P_L) at W vertex × (ig_Z × 1/2)(γ P_L) at Z vertex
    # The P_L on the Z-neutrino side comes from cL_ν = 1/2 (cR_ν = 0).
    coupling = (-1j * _G_W / math.sqrt(2)) * (1j * _G_Z * 0.5)
    return coupling * sandwich


def _amplitude_u_e_exchange(
    p1, p2, k_W, k_Z, h1, h2, lW, lZ,
) -> complex:
    """u-channel e- exchange.

    e-(p1) → Z(k_Z) + e-_internal (Z vertex, cL_e, cR_e)
    e-_internal + ν̄_e(p2) → W-(k_W) (W vertex, V-A)

    Internal e- momentum l = p1 - k_Z.  Massless propagator i l̸/l².
    """
    u1 = _spinor_u(p1, h1)
    v2 = _spinor_v(p2, h2)
    v2_bar = _adjoint(v2)
    eps_W_star = np.conjugate(_w_polarization(k_W, lW))
    eps_Z_star = np.conjugate(_z_polarization(k_Z, lZ))

    l = p1 - k_Z
    l_sq = _dot4(l, l).real
    if abs(l_sq) < 1e-20:
        return 0.0 + 0j
    prop = 1j * _slash(l) / l_sq

    g_mu_W = sum(_ETA[mu, mu] * _GAMMAS[mu] * eps_W_star[mu] for mu in range(4))
    g_nu_Z = sum(_ETA[nu, nu] * _GAMMAS[nu] * eps_Z_star[nu] for nu in range(4))

    # Z-electron couplings (T3=-1/2, Q=-1)
    cL_e = -0.5 + _SIN2_W
    cR_e = +_SIN2_W
    Z_proj = cL_e * _PL + cR_e * _PR

    # Fermion line from u(p1) at Z vertex, through internal e propagator,
    # to v̄(p2) at W vertex.
    sandwich = v2_bar @ g_mu_W @ _PL @ prop @ g_nu_Z @ Z_proj @ u1

    coupling = (-1j * _G_W / math.sqrt(2)) * (1j * _G_Z)
    return coupling * sandwich


def _amplitude_s_W_exchange(
    p1, p2, k_W, k_Z, h1, h2, lW, lZ,
) -> complex:
    """s-channel W*- exchange via W+W-Z triple-gauge vertex.

    e-(p1) + ν̄_e(p2) → W*-(q = p1+p2) (charged-current vertex)
    W*-(q) → W-(k_W) + Z(k_Z) (WWZ TGC)

    In the W+W-Z Feynman rule, the W*- propagator plays the W- leg role
    (charge into vertex), the outgoing W- plays the W+ leg role (charge
    out of vertex), and the outgoing Z plays the Z leg role.  All-incoming
    momenta for the vertex: q_W+ = -k_W (outgoing → -k_W in-flow),
    q_W- = q (incoming W*-), q_Z = -k_Z.
    """
    u1 = _spinor_u(p1, h1)
    v2 = _spinor_v(p2, h2)
    v2_bar = _adjoint(v2)
    eps_W_star = np.conjugate(_w_polarization(k_W, lW))
    eps_Z_star = np.conjugate(_z_polarization(k_Z, lZ))

    q = p1 + p2
    q_sq = _dot4(q, q).real

    # All-incoming momenta for the WWZ vertex.
    q_Wplus = -k_W
    q_Wminus = q.copy()
    q_Z_in = -k_Z
    T = _wwz_tgc_tensor(q_Wplus, q_Wminus, q_Z_in)

    # Contract Lorentz indices: μ ↔ W+ leg (carries eps_W*), ν ↔ W- leg
    # (contracts with the e-ν̄ current J), α ↔ Z leg (carries eps_Z*).
    # J^ν = v̄(p2) γ^ν P_L u(p1) is the charged-current.
    J_nu = np.array([v2_bar @ _GAMMAS[nu] @ _PL @ u1 for nu in range(4)], dtype=complex)

    # Effective TGC contracted vector along ν (the W-propagator index).
    # T_eff[ν] = Σ_{μ,α} η_{μμ} η_{αα} T^{μνα}(...) ε*_W^μ ε*_Z^α
    T_eff = np.zeros(4, dtype=complex)
    for nu in range(4):
        s = 0.0 + 0j
        for mu in range(4):
            for a in range(4):
                s += _ETA[mu, mu] * _ETA[a, a] * T[mu, nu, a] * eps_W_star[mu] * eps_Z_star[a]
        T_eff[nu] = s

    # W propagator at q² with Breit-Wigner regulator (negligible since W*- is
    # space-/time-like off-shell; q² ≥ (M_W+M_Z)² > M_W²).
    D_W = (q_sq - _M_W**2) - 1j * _M_W * _GAMMA_W

    # Charged-current vertex coupling: each vertex contributes -ig_W/√2.
    # The two W legs at the TGC have implicit -ig_W/√2 each? No — the TGC is
    # a single 3-boson vertex with coupling -i g_Z cos²θ_W = -i e cot_W cos_W
    # ... actually the W+W-Z coupling is g_W cos_W (Schwartz §29.107).
    #
    # Putting it together:
    #   M_s = J^ν × (i × W-prop_νμ) × (-i g_W cos_W × T) × ε*_W^μ × ε*_Z^α
    # The W propagator i (-g^{νμ} + q^νq^μ/M_W²) / D_W for off-shell W in
    # unitary gauge.  We use the Feynman gauge propagator i (-g^{νμ}) / D_W
    # because the q^νq^μ piece is suppressed by the conserved current property
    # at high energies and the corrections it brings are ~ m_l²/M_W² which is
    # negligible for our massless leptons.

    # Contract T_eff (length-4 ν-index) with the propagator metric and the
    # current J:
    # M = (-ig_W/√2) × (-i g_W cos_W) × [J^ν × (i × η_νν / D_W) × T_eff^ν]
    M = 0.0 + 0j
    for nu in range(4):
        M += _ETA[nu, nu] * J_nu[nu] * T_eff[nu] * (1j / D_W)
    # Couplings: e- ν̄ W vertex is -ig_W/√2, TGC is -i g_W cos_W.
    coupling = (-1j * _G_W / math.sqrt(2)) * (-1j * _G_W * math.sqrt(_COS2_W))
    return coupling * M


def _total_amplitude(
    p1, p2, k_W, k_Z, h1, h2, lW, lZ,
) -> complex:
    """Sum of t (ν exchange) + u (e exchange) + s (W*- → WZ via TGC)."""
    return (
        _amplitude_t_nu_exchange(p1, p2, k_W, k_Z, h1, h2, lW, lZ)
        + _amplitude_u_e_exchange(p1, p2, k_W, k_Z, h1, h2, lW, lZ)
        + _amplitude_s_W_exchange(p1, p2, k_W, k_Z, h1, h2, lW, lZ)
    )


def _msq_avg(sqrt_s: float, cos_theta: float) -> float:
    """Spin-averaged |M̄|² for e- ν̄_e → W- Z at a fixed cos θ.

    Initial-state spin average: 1/(2·2) for e- helicities × ν̄_e helicities.
    But ν̄_e is right-handed only (no right-handed neutrino in SM); we still
    divide by 4 for consistency with the Pauli spin-statistic convention,
    and the RH-ν̄ Σ_h includes only the (h2 = +1) contribution.
    """
    s = sqrt_s ** 2
    threshold = (_M_W + _M_Z) ** 2
    if s <= threshold:
        return 0.0
    E = sqrt_s / 2.0
    p1 = np.array([E, 0.0, 0.0, E], dtype=float)
    p2 = np.array([E, 0.0, 0.0, -E], dtype=float)

    # Final-state 3-momentum magnitude (Källén)
    lam = (s - (_M_W + _M_Z) ** 2) * (s - (_M_W - _M_Z) ** 2)
    p_f = math.sqrt(max(lam, 0.0)) / (2.0 * sqrt_s)
    E_W = (s + _M_W**2 - _M_Z**2) / (2.0 * sqrt_s)
    E_Z = (s + _M_Z**2 - _M_W**2) / (2.0 * sqrt_s)
    sin_th = math.sqrt(max(0.0, 1.0 - cos_theta**2))
    k_W = np.array(
        [E_W, p_f * sin_th, 0.0, p_f * cos_theta], dtype=float,
    )
    k_Z = np.array(
        [E_Z, -p_f * sin_th, 0.0, -p_f * cos_theta], dtype=float,
    )

    msq_sum = 0.0
    # SM: only LH e-, RH ν̄_e contribute (no RH ν in SM).
    # The charged-current vertex projects out (LH e-, RH ν̄_e) anyway, so
    # all wrong-helicity combinations give zero.  We still loop over (h1,h2)
    # so the 1/(2·2) spin-average factor is symmetric.
    for h1 in (-1, +1):
        for h2 in (-1, +1):
            for lW in (-1, 0, +1):
                for lZ in (-1, 0, +1):
                    M = _total_amplitude(p1, p2, k_W, k_Z, h1, h2, lW, lZ)
                    msq_sum += abs(M) ** 2
    # Spin average over initial states: 1/4 (two-by-two).  No colour factor
    # for a colour-singlet leptonic initial state.
    return msq_sum / 4.0


# ─── Public API ───────────────────────────────────────────────────────────


def _parse_process(process: str) -> Optional[dict]:
    """Return a dict describing the process if supported, else None.

    Currently supports: e- ν̄_e → W- Z and its charge conjugate e+ ν_e → W+ Z.
    """
    if "->" not in process:
        return None
    lhs, rhs = process.split("->", 1)
    in_parts = lhs.split()
    out_parts = rhs.split()
    if len(in_parts) != 2 or len(out_parts) != 2:
        return None
    # e- ν̄_e → W- Z (allow both orderings of initial state and final state)
    if (set(in_parts) == {"e-", "nu_e~"} or set(in_parts) == {"e-", "nuebar"}) \
            and set(out_parts) == {"W-", "Z"}:
        return {"charge": "-"}
    if (set(in_parts) == {"e+", "nu_e"} or set(in_parts) == {"e+", "nue"}) \
            and set(out_parts) == {"W+", "Z"}:
        return {"charge": "+"}
    return None


def is_supported(process: str) -> bool:
    return _parse_process(process) is not None


def cross_section(process: str, sqrt_s: float, n_cos: int = 80) -> dict:
    """LO σ̂(e ν → W Z) in pb via direct helicity-amplitude evaluation.

    Returns
    -------
    dict
        ``{"process", "sqrt_s_gev", "sigma_pb", "method", "trust_level",
        "accuracy_caveat", "supported"}``.
    """
    info = _parse_process(process)
    if info is None:
        return {
            "process": process, "supported": False,
            "error": f"enubar_wz_helicity does not support {process!r}.",
        }
    s = sqrt_s ** 2
    threshold = (_M_W + _M_Z) ** 2
    if s <= threshold:
        return {
            "process": process, "sqrt_s_gev": sqrt_s, "supported": False,
            "error": (
                f"√s = {sqrt_s:.3f} GeV is below the W-Z threshold "
                f"(m_W + m_Z = {_M_W + _M_Z:.3f} GeV)."
            ),
        }
    lam = (s - (_M_W + _M_Z) ** 2) * (s - (_M_W - _M_Z) ** 2)
    p_f = math.sqrt(lam) / (2.0 * sqrt_s)
    flux = 2.0 * s  # massless-initial-state flux 2 s
    cos_grid = np.linspace(-0.999, 0.999, n_cos)
    msq_grid = np.array([_msq_avg(sqrt_s, c) for c in cos_grid])
    # dσ/dcosθ = (1/(32π s)) × (p_f / (sqrt_s/2)) × |M̄|²
    #          = (p_f / (16π s sqrt_s)) × |M̄|²
    # ∫ d(cosθ) gives σ.  (Equivalent: σ = (1/(32π)) × (p_f/p_i) × ⟨|M|²⟩ / s.)
    integral = np.trapezoid(msq_grid, cos_grid)
    p_i = sqrt_s / 2.0  # massless initial state |p1| = sqrt_s/2
    sigma_gev2 = (p_f / (32.0 * math.pi * s * p_i)) * integral

    # Trust level depends on √s.  Below 700 GeV the engine is 50% LOW
    # vs MG5 (root cause unidentified); above 1 TeV the engine agrees with
    # MG5 to within α-scheme tolerance (~6%).  Use ROUGH at low √s so users
    # see a clear order-of-magnitude warning, APPROXIMATE at high √s.
    if sqrt_s < 700.0:
        trust_level = "rough"
        caveat = (
            "*** ORDER-OF-MAGNITUDE ONLY at this √s. *** "
            "Engine is ~50% LOW vs MG5_aMC@NLO v3.7.1 at √s = 200-500 GeV "
            "(eg engine 6.87 pb vs MG5 13.88 pb at √s=200).  Root cause "
            "unidentified after extensive investigation (polarization "
            "sums, sign conventions, gauge cancellation, integration "
            "convergence all check out).  Engine agreement converges to "
            "~6% (α-scheme tolerance) by √s = 1 TeV.  "
            "Do NOT use this value for precision phenomenology at low √s; "
            "use at √s ≳ 1 TeV instead, or cross-check with MG5/Whizard."
        )
    else:
        trust_level = "approximate"
        caveat = (
            "Tree-level SM e- ν̄_e → W- Z via direct helicity-amplitude "
            "evaluation.  Three diagrams (t-channel ν, u-channel e, "
            "s-channel W*- via WWZ TGC) with full SM interferences and "
            "massive-boson kinematics.  At √s ≥ 1 TeV the engine agrees "
            "with MG5_aMC@NLO v3.7.1 to within α-scheme tolerance (~6%).  "
            "At low √s (< 700 GeV) the engine has a ~50% LOW discrepancy "
            "vs MG5 with root cause unidentified (downgraded to 'rough'). "
            "See paper/benchmarks/MG5_COMPARISON.md."
        )

    return {
        "process": process,
        "sqrt_s_gev": sqrt_s,
        "s_gev2": s,
        "sigma_pb": sigma_gev2 * _GEV2_TO_PB,
        "method": "enubar-wz-helicity-amplitudes",
        "trust_level": trust_level,
        "accuracy_caveat": caveat,
        "reference": (
            "Hagiwara-Peccei-Zeppenfeld-Hikasa NPB 282 (1987) 253 partonic "
            "framework, evaluated directly from SM Feynman rules (W+W-Z TGC "
            "per Schwartz §29.107)."
        ),
        "supported": True,
    }
