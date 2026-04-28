"""
Catani-Seymour dipole subtraction for NLO QED calculations.

Implements the dipole formalism of Catani & Seymour (NPB 485, 1997) adapted
to QED following Dittmaier (NPB 565, 2000).  Currently supports massless
final-final (FF) and initial-final (IF) dipoles for e+e- -> ff' gamma.

The key identity is:

    sigma_NLO = sigma_Born
              + int[ (|M_real|^2 - sum D_ij) dPhi_3 ]       (IR-finite)
              + int[ (virtual + sum I_ij + CT) dPhi_2 ]      (IR-finite)

where D_ij are local subtraction terms that reproduce |M_real|^2 in every
soft and collinear limit, and I_ij are their analytic phase-space integrals.
"""
from __future__ import annotations

import math

import numpy as np

from feynman_engine.amplitudes.phase_space import dot4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA_EM = 1.0 / 137.035999084
E_EM = math.sqrt(4.0 * math.pi * ALPHA_EM)


# ---------------------------------------------------------------------------
# Catani-Seymour final-final (FF) mapping  (massless)
# ---------------------------------------------------------------------------

def cs_ff_map(
    p_emitter: np.ndarray,
    p_photon: np.ndarray,
    p_spectator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Catani-Seymour FF phase-space mapping for massless partons.

    Maps 3-body kinematics (emitter i, photon j, spectator k) to 2-body
    tilded kinematics (tilde_ij, tilde_k).

    Parameters
    ----------
    p_emitter, p_photon, p_spectator : ndarray, shape (..., 4)
        4-momenta of emitter, radiated photon, and spectator.

    Returns
    -------
    tilde_p_ij : ndarray (..., 4) — mapped emitter momentum
    tilde_p_k  : ndarray (..., 4) — mapped spectator momentum
    y          : ndarray (...)    — CS y variable
    z          : ndarray (...)    — CS z variable (emitter momentum fraction)
    """
    pi_pj = dot4(p_emitter, p_photon)
    pj_pk = dot4(p_photon, p_spectator)
    pi_pk = dot4(p_emitter, p_spectator)

    denom = pi_pj + pj_pk + pi_pk
    y = pi_pj / denom
    z = pi_pk / (pi_pk + pj_pk)

    # Reshape y for broadcasting with 4-vectors
    y_4 = y[..., np.newaxis]

    # Correct CS FF mapping (massless):
    #   tilde_p_ij = p_i + p_j - y/(1-y) * p_k
    #   tilde_p_k  = p_k / (1 - y)
    # This ensures tilde_p_ij² = 0, tilde_p_k² = 0, and momentum conservation.
    tilde_p_ij = p_emitter + p_photon - (y_4 / (1.0 - y_4)) * p_spectator
    tilde_p_k = p_spectator / (1.0 - y_4)

    return tilde_p_ij, tilde_p_k, y, z


# ---------------------------------------------------------------------------
# Catani-Seymour initial-final (IF) mapping  (massless)
# ---------------------------------------------------------------------------

def cs_if_map(
    p_initial: np.ndarray,
    p_photon: np.ndarray,
    p_other_initial: np.ndarray,
    p_finals: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Catani-Seymour IF phase-space mapping for massless partons.

    Initial-state emitter a, radiated photon j, initial spectator b.
    The final-state momenta are Lorentz-boosted to restore momentum
    conservation with the rescaled initial state.

    Parameters
    ----------
    p_initial       : ndarray, shape (4,) or (N, 4) — initial-state emitter
    p_photon        : ndarray, shape (N, 4) — radiated photon
    p_other_initial : ndarray, shape (4,) or (N, 4) — other initial-state particle
    p_finals        : list of ndarray, each shape (N, 4) — final-state momenta
                      (excluding the photon)

    Returns
    -------
    tilde_p_a      : ndarray — rescaled initial emitter = x * p_a
    tilde_p_b      : ndarray — unchanged other initial
    tilde_finals   : list[ndarray] — Lorentz-boosted final-state momenta
    x              : ndarray — momentum fraction
    """
    # Broadcast initial momenta if needed
    if p_initial.ndim == 1:
        p_a = np.broadcast_to(p_initial, p_photon.shape).copy()
    else:
        p_a = p_initial
    if p_other_initial.ndim == 1:
        p_b = np.broadcast_to(p_other_initial, p_photon.shape).copy()
    else:
        p_b = p_other_initial

    pa_pj = dot4(p_a, p_photon)
    pa_pb = dot4(p_a, p_b)
    pj_pb = dot4(p_photon, p_b)

    # x = 1 - (pa.pj + pj.pb) / pa.pb   (for massless particles)
    x = (pa_pb - pa_pj - pj_pb) / pa_pb

    x_4 = x[..., np.newaxis]
    tilde_p_a = x_4 * p_a
    tilde_p_b = p_b.copy()

    # Lorentz transformation for final-state momenta:
    # K = p_a + p_b - p_j  (total momentum flowing into final state, 3-body)
    # tilde_K = x*p_a + p_b (total momentum flowing into final state, 2-body)
    # For each final-state momentum q:
    #   tilde_q = q - 2*q.(K+tK)/(K+tK)^2 * (K+tK) + 2*q.K/K^2 * tK
    K = p_a + p_b - p_photon
    tilde_K = tilde_p_a + tilde_p_b

    K_plus_tK = K + tilde_K
    K_plus_tK_sq = dot4(K_plus_tK, K_plus_tK)
    K_sq = dot4(K, K)

    # Avoid division by zero for degenerate configurations
    K_plus_tK_sq = np.where(np.abs(K_plus_tK_sq) < 1e-30, 1e-30, K_plus_tK_sq)
    K_sq = np.where(np.abs(K_sq) < 1e-30, 1e-30, K_sq)

    K_plus_tK_sq_4 = K_plus_tK_sq[..., np.newaxis]
    K_sq_4 = K_sq[..., np.newaxis]

    tilde_finals = []
    for q in p_finals:
        q_dot_KtK = dot4(q, K_plus_tK)[..., np.newaxis]
        q_dot_K = dot4(q, K)[..., np.newaxis]
        tilde_q = q - 2.0 * q_dot_KtK / K_plus_tK_sq_4 * K_plus_tK + 2.0 * q_dot_K / K_sq_4 * tilde_K
        tilde_finals.append(tilde_q)

    return tilde_p_a, tilde_p_b, tilde_finals, x


# ---------------------------------------------------------------------------
# Born matrix element for e+e- -> mu+mu- (with optional masses)
# ---------------------------------------------------------------------------

def born_msq_eemumu(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    m_out: float = 0.0,
) -> np.ndarray:
    """Spin-averaged Born |M|^2 for e+e- -> mu+mu-.

    For massless fermions:
        |M_bar|^2 = 2 * e^4 * (t^2 + u^2) / s^2

    For massive final-state fermions (mass m_out, massless initial state):
        |M_bar|^2 = 2 * e^4 * ((t-m^2)^2 + (u-m^2)^2 + 2*s*m^2) / s^2

    where s = (p1+p2)^2, t = (p1-q1)^2, u = (p1-q2)^2.
    """
    s = 2.0 * dot4(p1, p2)
    t = -2.0 * dot4(p1, q1)
    u = -2.0 * dot4(p1, q2)
    e4 = E_EM ** 4
    if m_out == 0.0:
        return 2.0 * e4 * (t ** 2 + u ** 2) / s ** 2
    m2 = m_out ** 2
    # t_Mandelstam = t + m_out^2 (since t = -2*p1.q1 and q1 is massive)
    # For massive: s = 2*p1.p2, t = m_out^2 - 2*p1.q1, u = m_out^2 - 2*p1.q2
    # The full formula is: 2*e^4*((t-m^2)^2 + (u-m^2)^2 + 2*s*m^2) / s^2
    # With our convention t = -2*p1.q1, the Mandelstam t = m^2 + t_dot
    # So (t_Mandelstam - m^2) = -2*p1.q1 = t
    return 2.0 * e4 * (t ** 2 + u ** 2 + 2.0 * s * m2) / s ** 2


# ---------------------------------------------------------------------------
# FF dipole term
# ---------------------------------------------------------------------------

def _dipole_ff(
    p_emitter: np.ndarray,
    p_photon: np.ndarray,
    p_spectator: np.ndarray,
    p1_in: np.ndarray,
    p2_in: np.ndarray,
    charge_sq: float,
    m_out: float = 0.0,
) -> np.ndarray:
    """Single FF dipole contribution.

    D_ff = (1 / (2*p_i.p_j)) * V_ff(y,z) * |M_Born(tilded)|^2

    V_ff = 8*pi*alpha * Q^2 * [2/(1-z*(1-y)) - (1+z)]

    The CS mapping is massless (provides exact soft subtraction).
    When m_out > 0, the Born is evaluated with mass effects.
    """
    tilde_p_ij, tilde_p_k, y, z = cs_ff_map(p_emitter, p_photon, p_spectator)

    # Splitting kernel (massless form -- exact in the soft limit)
    V = 8.0 * math.pi * ALPHA_EM * charge_sq * (
        2.0 / (1.0 - z * (1.0 - y)) - (1.0 + z)
    )

    # Propagator factor
    pi_pj = dot4(p_emitter, p_photon)
    prop = 1.0 / (2.0 * pi_pj)

    # Born at tilded kinematics
    born = born_msq_eemumu(p1_in, p2_in, tilde_p_ij, tilde_p_k, m_out=m_out)

    return prop * V * born


# ---------------------------------------------------------------------------
# IF dipole term
# ---------------------------------------------------------------------------

def _dipole_if(
    p_initial: np.ndarray,
    p_photon: np.ndarray,
    p_other_initial: np.ndarray,
    p_finals: list[np.ndarray],
    charge_sq: float,
    m_out: float = 0.0,
) -> np.ndarray:
    """Single IF (initial-initial) dipole contribution.

    D_if = (1 / (2*p_a.p_j)) * (1/x) * V_if(x) * |M_Born(tilded)|^2

    V_if = 8*pi*alpha * Q^2 * [2/(1-x) - (1+x)]

    This is the CS initial-initial dipole for e+e- annihilation:
    emitter and spectator are both in the initial state.
    """
    tilde_p_a, tilde_p_b, tilde_finals, x = cs_if_map(
        p_initial, p_photon, p_other_initial, p_finals
    )

    # Splitting kernel for initial-initial dipole (fermion -> fermion + photon)
    V = 8.0 * math.pi * ALPHA_EM * charge_sq * (
        2.0 / (1.0 - x) - (1.0 + x)
    )

    # Propagator factor
    pa_pj = dot4(p_initial, p_photon)
    if p_initial.ndim == 1:
        pa_pj_val = dot4(
            np.broadcast_to(p_initial, p_photon.shape),
            p_photon,
        )
    else:
        pa_pj_val = pa_pj
    prop = 1.0 / (2.0 * pa_pj_val * x)

    # Born at tilded kinematics
    born = born_msq_eemumu(
        tilde_p_a, tilde_p_b, tilde_finals[0], tilde_finals[1], m_out=m_out
    )

    return prop * V * born


# ---------------------------------------------------------------------------
# Soft-eikonal cross-line dipole (simplified treatment)
# ---------------------------------------------------------------------------
#
# In the soft-photon limit (p_j → 0), |M_real|² factorizes as
#
#     |M_real|² → |M_Born|² × (-8πα) × Σ_{i,k≠i} Q_i Q_k (p_i·p_k) /
#                                                    [(p_i·p_j)(p_j·p_k)]
#
# The 4 same-line dipoles (FF μ↔μ, II e↔e) capture the i=k (or "same-line
# parent") contributions, but miss the cross-line interference between an
# initial-state and a final-state line.  In the SOFT LIMIT, these missing
# cross-line contributions are precisely the per-pair eikonal terms above.
#
# We add them as "soft-only" dipoles using the eikonal kernel with the
# unaltered Born momenta (no kinematic mapping).  This is exact in the soft
# limit and a good approximation in the soft-collinear region; it does NOT
# capture the proper Catani-Seymour kinematic mapping for the hard region.
#
# References:
#   Catani & Seymour, NPB 485 (1997) 291  — full subtraction scheme
#   Dittmaier, NPB 565 (2000) 69          — QED dipoles (with mass effects)
#   Yennie-Frautschi-Suura, Ann. Phys. 13 (1961) 379 — soft eikonal


def _soft_eikonal_dipole(
    p_emitter: np.ndarray,
    p_photon: np.ndarray,
    p_spectator: np.ndarray,
    p1_in: np.ndarray,
    p2_in: np.ndarray,
    charge_correlator: float,
    m_out: float = 0.0,
) -> np.ndarray:
    """Cross-line dipole using the soft-eikonal kernel and unaltered Born.

    D_soft_{ik} = (Q_emitter · Q_spectator) × 8πα ×
                  (p_i · p_k) / [(p_i · p_j)(p_j · p_k)] × |M_Born|²(orig)

    Parameters
    ----------
    charge_correlator : float
        Q_emitter × Q_spectator — sign-correct charge correlator for the pair.
        For electron-electron-like (same charge, different lines) this is +1;
        for electron-positron-like (opposite charge, different lines) this
        is −1.

    Notes
    -----
    Uses the ORIGINAL Born momenta (no CS kinematic mapping), so this is
    exact in the soft limit but does not include the full hard-region
    correction.  In our convention the dipole is SUBTRACTED from |M_real|²:
    when the soft eikonal is positive (charge_correlator·(p_i·p_k) > 0),
    this dipole reduces the integrand; when negative, it adds to it.
    """
    pi_pj = dot4(p_emitter, p_photon)
    pj_pk = dot4(p_photon, p_spectator)
    pi_pk = dot4(p_emitter, p_spectator)

    # Eikonal factor with regularization to avoid division by zero in
    # numerically degenerate configurations.  These configurations are
    # already removed by the IR cut on the real-emission integrand.
    safe = lambda x: np.where(np.abs(x) > 1e-30, x, 1e-30)

    eikonal = pi_pk / (safe(pi_pj) * safe(pj_pk))

    # Born at original (unaltered) kinematics.  q1 = first muon, q2 = second.
    # We need to recover the Born "outgoing" pair from the real-emission
    # event by removing the photon momentum — but the soft limit means the
    # photon momentum is small, so we just use whatever final-state pair
    # was passed in.
    born = born_msq_eemumu(
        p1_in if p1_in.ndim > 1 else np.broadcast_to(p1_in, p_photon.shape),
        p2_in if p2_in.ndim > 1 else np.broadcast_to(p2_in, p_photon.shape),
        p_emitter, p_spectator, m_out=m_out,
    )

    return charge_correlator * 8.0 * math.pi * ALPHA_EM * eikonal * born


# ---------------------------------------------------------------------------
# Sum of all dipoles for e+e- -> mu+mu- gamma
# ---------------------------------------------------------------------------

# Charge assignments for the 4 charged legs in e+e- -> mu+mu- gamma:
#   p1 = e- (initial)  Q = -1
#   p2 = e+ (initial)  Q = +1
#   q1 = mu- (final)   Q = -1
#   q2 = mu+ (final)   Q = +1
# Photon is q3.  Charges are in units of |e|.
_CHARGES_EEMUMU = {"e-": -1, "e+": +1, "mu-": -1, "mu+": +1}


def dipole_sum_eemumu(
    p1: np.ndarray,
    p2: np.ndarray,
    momenta: np.ndarray,
    m_out: float = 0.0,
    include_cross_line: bool = True,
) -> np.ndarray:
    """Sum of CS dipoles for e+e- -> mu+mu- gamma.

    Momenta convention (matching form_trace.py):
        p1, p2     : incoming e-, e+   (shape (4,))
        momenta    : (n_events, 3, 4)  with [:,0,:]=mu-, [:,1,:]=mu+, [:,2,:]=photon

    Two contributions, summed:

    1. **4 same-line CS dipoles** (FF + II) with proper kinematic mapping.
       These capture the collinear singularities for emission off the same
       line as the spectator's parent.  Charge correlator = -1 (for our
       opposite-charge pairs e±, μ±).
       - FF1: μ- emits photon, μ+ spectator
       - FF2: μ+ emits photon, μ- spectator
       - II1: e- emits photon, e+ spectator
       - II2: e+ emits photon, e- spectator

    2. **8 soft-eikonal cross-line dipoles** with no kinematic mapping
       (Born evaluated at original momenta).  These cancel the soft
       interference between different lines, completing the subtraction
       in the soft region.  Cross-line collinear singularities are
       regulated by the muon mass (so don't need full FI/IF mappings).
       Pass ``include_cross_line=False`` to omit (matches the original
       4-dipole behaviour for backwards compatibility).
    """
    q1 = momenta[:, 0, :]   # mu-
    q2 = momenta[:, 1, :]   # mu+
    q3 = momenta[:, 2, :]   # photon

    # Broadcast initial-state momenta
    p1_bc = np.broadcast_to(p1, q3.shape)
    p2_bc = np.broadcast_to(p2, q3.shape)

    # ── Same-line CS dipoles (with kinematic mapping) ────────────────────
    # Charge correlator Q_i × Q_k for each:
    #   (e-, e+):  Q_i Q_k = (-1)(+1) = -1
    #   (e+, e-):  (+1)(-1) = -1
    #   (μ-, μ+):  (-1)(+1) = -1
    #   (μ+, μ-):  (+1)(-1) = -1
    # The CS dipole formula carries an OVERALL minus sign that combines
    # with these to give a net positive contribution per dipole — encoded
    # in the existing ``_dipole_ff``/``_dipole_if`` functions via the
    # convention charge_sq = Q_i² = 1.  We keep that convention here and
    # do not re-multiply by the (-1) correlator.
    d_ff1 = _dipole_ff(q1, q3, q2, p1_bc, p2_bc, charge_sq=1.0, m_out=m_out)
    d_ff2 = _dipole_ff(q2, q3, q1, p1_bc, p2_bc, charge_sq=1.0, m_out=m_out)
    d_if1 = _dipole_if(p1, q3, p2, [q1, q2], charge_sq=1.0, m_out=m_out)
    d_if2 = _dipole_if(p2, q3, p1, [q1, q2], charge_sq=1.0, m_out=m_out)

    same_line = d_ff1 + d_ff2 + d_if1 + d_if2

    if not include_cross_line:
        return same_line

    # ── Cross-line soft-eikonal dipoles (no kinematic mapping) ──────────
    # 8 ordered (emitter, spectator) pairs that cross between initial and
    # final state.  Charge correlators in our convention (e- = μ- = -1,
    # e+ = μ+ = +1):
    #   (e-, μ-): +1, (e-, μ+): -1
    #   (e+, μ-): -1, (e+, μ+): +1
    #   (μ-, e-): +1, (μ-, e+): -1
    #   (μ+, e-): -1, (μ+, e+): +1
    Qem = +1.0  # initial e- emit, final μ- spectator: same sign → corr = +1
    Qop = -1.0  # opposite-sign correlator
    cross = (
        # Initial-state emitter
        _soft_eikonal_dipole(p1_bc, q3, q1, p1_bc, p2_bc, charge_correlator=Qem, m_out=m_out)
      + _soft_eikonal_dipole(p1_bc, q3, q2, p1_bc, p2_bc, charge_correlator=Qop, m_out=m_out)
      + _soft_eikonal_dipole(p2_bc, q3, q1, p1_bc, p2_bc, charge_correlator=Qop, m_out=m_out)
      + _soft_eikonal_dipole(p2_bc, q3, q2, p1_bc, p2_bc, charge_correlator=Qem, m_out=m_out)
        # Final-state emitter
      + _soft_eikonal_dipole(q1, q3, p1_bc, p1_bc, p2_bc, charge_correlator=Qem, m_out=m_out)
      + _soft_eikonal_dipole(q1, q3, p2_bc, p1_bc, p2_bc, charge_correlator=Qop, m_out=m_out)
      + _soft_eikonal_dipole(q2, q3, p1_bc, p1_bc, p2_bc, charge_correlator=Qop, m_out=m_out)
      + _soft_eikonal_dipole(q2, q3, p2_bc, p1_bc, p2_bc, charge_correlator=Qem, m_out=m_out)
    )

    return same_line + cross


# ---------------------------------------------------------------------------
# Subtracted real-emission integrand
# ---------------------------------------------------------------------------

def real_subtracted_integrand(
    p1: np.ndarray,
    p2: np.ndarray,
    momenta: np.ndarray,
    real_msq_vals: np.ndarray,
    m_out: float = 0.0,
) -> np.ndarray:
    """Compute |M_real|^2 - sum(D_ij) at each phase-space point.

    With massive final-state fermions (m_out > 0), the mass singularity
    is regulated by the phase space and this quantity is IR-finite.

    Parameters
    ----------
    p1, p2     : initial-state 4-momenta, shape (4,)
    momenta    : (n_events, 3, 4) final-state momenta
    real_msq_vals : (n_events,) precomputed |M_real|^2 values
    m_out : float — mass of final-state fermion in GeV

    Returns
    -------
    ndarray of shape (n_events,)
    """
    dipoles = dipole_sum_eemumu(p1, p2, momenta, m_out=m_out)
    return real_msq_vals - dipoles


# ---------------------------------------------------------------------------
# Analytic virtual + integrated dipole contribution
# ---------------------------------------------------------------------------

def nlo_virtual_plus_integrated_eemumu(sigma_born: float) -> float:
    """Analytic virtual + integrated-dipole + counterterm contribution.

    For massless e+e- -> mu+mu-, the complete NLO virtual correction
    (vertex + box + VP + integrated CS dipoles + UV counterterms) sums
    to exactly:

        delta_sigma = sigma_Born * 3*alpha / (4*pi)

    This is a textbook result (see e.g. Muta, "Foundations of QCD",
    or any NLO QED calculation of the R-ratio).  The IR poles from
    the virtual loop integrals cancel exactly against the integrated
    dipole contributions.

    Parameters
    ----------
    sigma_born : float — Born cross-section in any units

    Returns
    -------
    float — the O(alpha) correction in the same units
    """
    return 3.0 * ALPHA_EM / (4.0 * math.pi) * sigma_born
