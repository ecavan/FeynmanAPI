"""
General Catani-Seymour dipole subtraction for NLO QCD/QED.

Implements the universal dipole formalism of Catani & Seymour
(NPB 485, 1997) for arbitrary 2→N processes with massless partons.
For massive partons see Catani-Dittmaier-Seymour-Trocsanyi
(NPB 627 (2002) 189) — V2.1 territory.

The key identity for an IR-safe NLO σ:

    σ_NLO = ∫ dΦ_N (B + V + ⟨I⟩·B)              (finite by KLN + ε-pole cancellation)
          + ∫ dΦ_{N+1} (R - Σ_{i,j,k} D_ij,k)   (finite by construction)

Each unintegrated dipole D_ij,k is the product of:

    1. A propagator factor  1 / (2 p_i · p_j)
    2. A splitting kernel  V_ij(z, y)  (q→qg, g→gg, or g→qq̄)
    3. A colour correlator  ⟨B|T_i · T_k|B⟩ / ⟨B|B⟩
    4. The Born |M̄|² evaluated at *mapped* (N-particle) kinematics
       (CS phase-space mapping ensures on-shell + momentum conservation)

The 4 configurations distinguished by where the emitter and spectator sit:

    FF — both final-state                  (emitter i_final, spectator k_final)
    FI — final-state emitter, initial spectator  (i_final, a_initial)
    IF — initial-state emitter, final spectator  (a_initial, k_final)
    II — both initial-state                (a_initial, b_initial)

Sign convention: in the SUBTRACTION ``∫(R - Σ D)``, each dipole D is a
positive contribution that exactly reproduces the IR singularity of R in
the corresponding soft/collinear limit.

References:
  Catani-Seymour, NPB 485 (1997) 291                    [universal dipoles]
  Catani-Dittmaier-Seymour-Trocsanyi, NPB 627 (2002)   [massive extension]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np

from feynman_engine.amplitudes.phase_space import dot4

# ─── QCD constants ──────────────────────────────────────────────────────────
C_F = 4.0 / 3.0    # SU(3) fundamental Casimir
C_A = 3.0          # SU(3) adjoint Casimir
T_R = 0.5          # Dynkin index of fundamental
N_C = 3.0          # SU(3) colour
N_F = 5.0          # active quark flavours (n_f, used for splitting g→qq̄)


# ─── Parton typing ──────────────────────────────────────────────────────────

class PartonType(Enum):
    """SU(3) representation + electric charge of a parton.

    For CS subtraction we only need the SU(3) representation (and whether
    the splitting is q→qg, g→gg, or g→qq̄).  The "charged-lepton" entries
    are kept for QED dipole reuse.
    """
    QUARK     = "q"     # any massless quark (treated as same colour state)
    ANTIQUARK = "qbar"
    GLUON     = "g"
    PHOTON    = "gamma"
    LEPTON    = "lep"   # any charged lepton (Q=±1)
    ANTILEPTON = "alep"
    NEUTRAL   = "neu"   # neutral colour-singlet (Z, H, neutrino)

    @property
    def is_quark(self) -> bool:
        return self in (PartonType.QUARK, PartonType.ANTIQUARK)

    @property
    def is_gluon(self) -> bool:
        return self == PartonType.GLUON

    @property
    def is_coloured(self) -> bool:
        return self.is_quark or self.is_gluon

    @property
    def is_charged(self) -> bool:
        # Quarks ARE electrically charged but we classify those as colour
        # for CS dipole purposes (use the QCD/QED switch separately).
        return self in (PartonType.LEPTON, PartonType.ANTILEPTON)


# Convenience: name → PartonType
_PARTON_LOOKUP: dict[str, PartonType] = {
    # Quarks
    "u": PartonType.QUARK, "d": PartonType.QUARK, "s": PartonType.QUARK,
    "c": PartonType.QUARK, "b": PartonType.QUARK, "t": PartonType.QUARK,
    "u~": PartonType.ANTIQUARK, "d~": PartonType.ANTIQUARK, "s~": PartonType.ANTIQUARK,
    "c~": PartonType.ANTIQUARK, "b~": PartonType.ANTIQUARK, "t~": PartonType.ANTIQUARK,
    # Gluon
    "g": PartonType.GLUON,
    # Photon
    "gamma": PartonType.PHOTON, "ph": PartonType.PHOTON, "a": PartonType.PHOTON,
    # Leptons
    "e-": PartonType.LEPTON, "mu-": PartonType.LEPTON, "tau-": PartonType.LEPTON,
    "e+": PartonType.ANTILEPTON, "mu+": PartonType.ANTILEPTON, "tau+": PartonType.ANTILEPTON,
    # Neutral
    "Z": PartonType.NEUTRAL, "W+": PartonType.NEUTRAL, "W-": PartonType.NEUTRAL,
    "H": PartonType.NEUTRAL, "h": PartonType.NEUTRAL,
    "nu_e": PartonType.NEUTRAL, "nu_mu": PartonType.NEUTRAL, "nu_tau": PartonType.NEUTRAL,
}


def parton_type(particle: str) -> PartonType:
    """Look up the PartonType for an engine-style particle name."""
    if particle not in _PARTON_LOOKUP:
        raise ValueError(f"Unknown particle for CS dipole: {particle!r}")
    return _PARTON_LOOKUP[particle]


# ─── Dipole configuration enum ──────────────────────────────────────────────

class DipoleConfig(Enum):
    FF = "FF"   # final emitter, final spectator
    FI = "FI"   # final emitter, initial spectator
    IF = "IF"   # initial emitter, final spectator
    II = "II"   # initial emitter, initial spectator


# ─── Splitting kernels (massless, 4-dim — drop ε pieces) ───────────────────
#
# Each kernel V_ij(z, y; ...) returns the spin-averaged splitting function
# *without* the αs/(2π) prefactor and *without* the colour Casimir.
# Conventions follow CS eqs. (5.4)-(5.7) for FF, (5.71)-(5.78) for FI/IF,
# and (5.145)-(5.150) for II.
#
# The Casimir factor is multiplied in by the dipole assembler (it depends
# on the colour correlator ⟨B|T_i·T_k|B⟩ / ⟨B|B⟩).  In the IR-singular
# soft limit this correlator equals -C_F for q-q̄ legs, -C_A for g-g legs.


def V_qg_FF(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """V(q → q g) for FF dipole (CS eq. 5.7).  Pre-factor 8π α_s C_F omitted."""
    return 2.0 / (1.0 - z * (1.0 - y)) - (1.0 + z)


def V_gg_FF(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """V(g → g g) for FF dipole (CS eq. 5.7).  Pre-factor 16π α_s C_A omitted.

    Symmetric in z ↔ 1-z for the two outgoing gluons; we apply both
    contributions when summing over (i,j) pairs and divide by 2 for
    identical particles.
    """
    return (
        1.0 / (1.0 - z * (1.0 - y))
        + 1.0 / (1.0 - (1.0 - z) * (1.0 - y))
        - 2.0
        + z * (1.0 - z)
    )


def V_qq_FF(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """V(g → q q̄) for FF dipole (CS eq. 5.7).  Pre-factor 8π α_s T_R omitted."""
    return 1.0 - 2.0 * z * (1.0 - z)


# ─── Phase-space mappings ───────────────────────────────────────────────────

def cs_ff_map(
    p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """FF mapping: emitter i, emitted j, spectator k all final-state.

    Returns (tilde_p_ij, tilde_p_k, y_ij_k, z_i) where the tildes are the
    on-shell mapped Born momenta.  CS eqs. (5.5)-(5.6).

        y = p_i·p_j / (p_i·p_j + p_j·p_k + p_i·p_k)
        z = p_i·p_k / (p_i·p_k + p_j·p_k)
        ~p_ij = p_i + p_j - y/(1-y) p_k
        ~p_k  = p_k / (1-y)

    By construction ~p_ij² = ~p_k² = 0 and ~p_ij + ~p_k = p_i + p_j + p_k.
    """
    pi_pj = dot4(p_i, p_j)
    pj_pk = dot4(p_j, p_k)
    pi_pk = dot4(p_i, p_k)
    denom = pi_pj + pj_pk + pi_pk
    y = pi_pj / denom
    z = pi_pk / (pi_pk + pj_pk)
    y_4 = y[..., np.newaxis]
    tilde_p_ij = p_i + p_j - (y_4 / (1.0 - y_4)) * p_k
    tilde_p_k = p_k / (1.0 - y_4)
    return tilde_p_ij, tilde_p_k, y, z


def cs_fi_map(
    p_i: np.ndarray, p_j: np.ndarray, p_a: np.ndarray, p_other_finals: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """FI mapping: final-state emitter i, final-state j, initial-state spectator a.

    CS eqs. (5.71)-(5.78).  The initial-state momentum is rescaled from
    p_a to p_a/x where x is the momentum fraction:

        x = (p_a·p_i + p_a·p_j - p_i·p_j) / (p_a·p_i + p_a·p_j)
        z = (p_a·p_i) / (p_a·p_i + p_a·p_j)
        ~p_ij = p_i + p_j - (1-x) p_a / x
        ~p_a  = p_a / x

    For the SUBTRACTION integrand we need ~p_a so the Born is evaluated
    with a rescaled initial state (boost factor 1/x).  Final-state
    spectators in the (N+1)-body event are unchanged.
    """
    pi_pj = dot4(p_i, p_j)
    pa_pi = dot4(p_a, p_i)
    pa_pj = dot4(p_a, p_j)
    denom = pa_pi + pa_pj
    x = (denom - pi_pj) / denom
    z = pa_pi / denom
    x_4 = x[..., np.newaxis]
    tilde_p_ij = p_i + p_j - ((1.0 - x_4) / x_4) * p_a
    tilde_p_a = p_a / x_4
    tilde_finals = [q for q in p_other_finals]   # FI mapping leaves them alone
    return tilde_p_ij, tilde_p_a, tilde_finals, x, z


def cs_if_map(
    p_a: np.ndarray, p_j: np.ndarray, p_k: np.ndarray, p_other_finals: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """IF mapping: initial-state emitter a, final-state emitted j, final-state spectator k.

    CS eqs. (5.42)-(5.50).  Initial-state momentum rescaled from p_a to
    p_a × x_ai_k:

        x = (p_a·p_k + p_a·p_j - p_j·p_k) / (p_a·p_k + p_a·p_j)
        u = (p_a·p_j) / (p_a·p_j + p_a·p_k)
        ~p_a = x p_a
        ~p_k = p_k + p_j - (1-x) p_a

    Final-state spectators are unchanged.
    """
    pa_pj = dot4(p_a, p_j)
    pa_pk = dot4(p_a, p_k)
    pj_pk = dot4(p_j, p_k)
    denom = pa_pk + pa_pj
    x = (denom - pj_pk) / denom
    u = pa_pj / denom
    x_4 = x[..., np.newaxis]
    tilde_p_a = x_4 * p_a
    tilde_p_k = p_k + p_j - (1.0 - x_4) * p_a
    tilde_finals = [q for q in p_other_finals]
    return tilde_p_a, tilde_p_k, tilde_finals, x, u


def cs_ii_map(
    p_a: np.ndarray, p_j: np.ndarray, p_b: np.ndarray, p_finals: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """II mapping: emitter a and spectator b both initial-state.

    CS eqs. (5.137)-(5.150).  The initial momentum is rescaled to
    ~p_a = x p_a, and *all* final-state momenta are Lorentz-boosted to
    restore momentum conservation:

        x = (p_a·p_b - p_a·p_j - p_j·p_b) / (p_a·p_b)
        ~p_a = x p_a
        ~p_b = p_b
        For each final q: ~q = q − 2 q·(K+~K)/(K+~K)² (K+~K) + 2 q·K/K² ~K
            with K = p_a + p_b − p_j, ~K = ~p_a + ~p_b
    """
    pa_pj = dot4(p_a, p_j)
    pa_pb = dot4(p_a, p_b)
    pj_pb = dot4(p_j, p_b)
    x = (pa_pb - pa_pj - pj_pb) / pa_pb
    x_4 = x[..., np.newaxis]
    tilde_p_a = x_4 * p_a
    tilde_p_b = p_b
    K = p_a + p_b - p_j
    tilde_K = tilde_p_a + tilde_p_b
    K_plus_tK = K + tilde_K
    K_plus_tK_sq = dot4(K_plus_tK, K_plus_tK)
    K_sq = dot4(K, K)
    safe = lambda x: np.where(np.abs(x) < 1e-30, 1e-30, x)
    K_plus_tK_sq_4 = safe(K_plus_tK_sq)[..., np.newaxis]
    K_sq_4 = safe(K_sq)[..., np.newaxis]
    tilde_finals: list[np.ndarray] = []
    for q in p_finals:
        q_dot_KtK = dot4(q, K_plus_tK)[..., np.newaxis]
        q_dot_K   = dot4(q, K)[..., np.newaxis]
        tilde_q = (
            q
            - 2.0 * q_dot_KtK / K_plus_tK_sq_4 * K_plus_tK
            + 2.0 * q_dot_K   / K_sq_4 * tilde_K
        )
        tilde_finals.append(tilde_q)
    return tilde_p_a, tilde_p_b, tilde_finals, x


# ─── Colour correlator ──────────────────────────────────────────────────────
#
# For a 2-coloured-leg Born (incoming q + q̄, or 2 gluons) the colour
# correlator T_i · T_k acting on the Born colour state is a number:
#
#     ⟨B|T_q · T_q̄|B⟩ / ⟨B|B⟩  =  -C_F = -4/3
#     ⟨B|T_g · T_g|B⟩ / ⟨B|B⟩  =  -C_A = -3
#
# For multi-coloured Borns (e.g. qq̄ → ggg, 4 colour legs) the correlator
# is a non-trivial colour matrix that depends on the Born colour-flow
# decomposition.  V2.0 ships only the 2-leg case; multi-leg gets a
# colour-decomposed Born from OpenLoops (V2.1).


def color_correlator_2leg(
    emitter: PartonType,
    spectator: PartonType,
) -> float:
    """⟨B|T_i · T_k|B⟩ / ⟨B|B⟩ for a 2-coloured-parton Born.

    For colour-conservation T_i + T_k = 0  ⇒  T_i · T_k = -T_i² = -C_R.

    Returns -C_F for two quarks (or quark-antiquark) and -C_A for two gluons.
    Cross-type (q-g) is interpreted as a g → qq̄ splitting where the Born
    is q-q̄ (so we return -C_F) — the kernel handles the splitting Casimir
    separately.
    """
    if emitter.is_quark and spectator.is_quark:
        return -C_F
    if emitter.is_gluon and spectator.is_gluon:
        return -C_A
    # Mixed: assume the Born is 2-coloured-leg q-q̄ after the splitting.
    # For g → qq̄ in II, after the mapping the Born has a q replacing the g,
    # so the Born colour state is q-q̄ → ⟨T_q · T_q̄⟩ = -C_F.
    if emitter.is_gluon and spectator.is_quark:
        return -C_F
    if emitter.is_quark and spectator.is_gluon:
        return -C_F
    return 0.0


def born_casimir_from_emitter(emitter: PartonType, emitted: PartonType) -> float:
    """Casimir of the Born parton AFTER the splitting.

    For q → q + g: Born parton stays q → C_F
    For g → g + g: Born parton stays g → C_A
    For g → q + q̄: Born parton is q (replacing g) → C_F
    For q → q̄ + q (i.e. q' → q' g via channel crossing): C_F
    """
    if emitter.is_quark and emitted.is_gluon:
        return C_F      # q → qg, Born has q
    if emitter.is_gluon and emitted.is_gluon:
        return C_A      # g → gg, Born has g
    if emitter.is_gluon and emitted.is_quark:
        return C_F      # g → qq̄, Born has q
    if emitter.is_quark and emitted.is_quark:
        return C_F      # would be a flavour-changing splitting, treat as q
    return 0.0


def color_correlator_from_openloops(
    born_process: str,
    momenta_full: np.ndarray,    # (n_external, 4) for one PSP
    emitter_idx: int,
    spectator_idx: int,
) -> float:
    """⟨B|T_i · T_k|B⟩ / ⟨B|B⟩ for an arbitrary Born from OpenLoops.

    Generalises ``color_correlator_2leg`` to multi-coloured-leg Borns
    (q q̄ → t t̄, g g → t t̄, q q̄ → V V, etc.) where colour conservation
    no longer reduces ⟨T_i · T_k⟩ to a single Casimir.

    OpenLoops's ``evaluate_cc`` returns the full colour-correlator
    matrix; we look up the (i, k) pair and return the normalised
    correlator.  Requires the OpenLoops process library covering
    ``born_process`` to be installed.
    """
    from feynman_engine.amplitudes.openloops_bridge import (
        evaluate_color_correlated_amplitude,
    )
    if emitter_idx == spectator_idx:
        return 0.0
    n = momenta_full.shape[0]
    # Pack into 5-per-particle layout (E, px, py, pz, m=√(E²-p²))
    flat = np.empty(5 * n)
    for ip in range(n):
        E, px, py, pz = momenta_full[ip]
        m_sq = E * E - px * px - py * py - pz * pz
        m = math.sqrt(max(m_sq, 0.0))
        base = 5 * ip
        flat[base + 0] = E
        flat[base + 1] = px
        flat[base + 2] = py
        flat[base + 3] = pz
        flat[base + 4] = m
    tree, cc = evaluate_color_correlated_amplitude(born_process, flat)
    if tree <= 0:
        return 0.0
    # cc is indexed by unordered pair (i,k) with i<k, in lexicographic order
    i, k = (emitter_idx, spectator_idx) if emitter_idx < spectator_idx else (spectator_idx, emitter_idx)
    # Pair index: cc_index(i,k) = i*(2n-i-1)/2 + (k-i-1)
    pair_idx = i * (2 * n - i - 1) // 2 + (k - i - 1)
    return cc[pair_idx] / tree


# ─── Splitting-kernel + Casimir picker ──────────────────────────────────────

def splitting_kernel_FF(
    emitter: PartonType,
    emitted: PartonType,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return (V(z,y), Casimir) for an FF splitting of `emitter` → emitter + emitted.

    The dipole formula is

        D_FF = (1 / (2 p_i·p_j)) * V * Casimir * 8π α_s * (-T_i·T_k/T_i²) * |B|²

    For 2-coloured Borns -T_i·T_k/T_i² = 1 (since T_i·T_k = -C_F when
    spectator is the colour partner), so the factor reduces to a constant.
    """
    if emitter.is_quark and emitted.is_gluon:
        return V_qg_FF(z, y), C_F
    if emitter.is_gluon and emitted.is_gluon:
        return V_gg_FF(z, y), C_A
    if emitter.is_gluon and emitted.is_quark:
        # g → qq̄ — emitted is the quark (or antiquark)
        return V_qq_FF(z, y), T_R
    raise ValueError(
        f"No FF splitting kernel for {emitter} → {emitter} + {emitted}"
    )


# ─── Dipole assembler (FF only, the building block we need first) ──────────

@dataclass
class FFDipoleResult:
    """Output of one FF dipole evaluation across an event sample."""
    value: np.ndarray              # D_ij,k(p_i, p_j, p_k) per event, shape (n_events,)
    born_momenta_ij: np.ndarray    # mapped emitter momentum ~p_ij (n_events, 4)
    born_momenta_k:  np.ndarray    # mapped spectator momentum ~p_k (n_events, 4)
    y: np.ndarray
    z: np.ndarray


def dipole_FF_multileg(
    p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray,
    born_momenta_full: np.ndarray,    # (n_events, n_external, 4) for the BORN
    born_process: str,                # for OpenLoops cc lookup
    emitter_idx_in_born: int,
    spectator_idx_in_born: int,
    born_msq_at_mapped: Callable[[np.ndarray, np.ndarray], np.ndarray],
    emitter: PartonType, emitted: PartonType, spectator: PartonType,
    alpha_s: float = 0.118,
) -> "FFDipoleResult":
    """FF dipole with multi-leg colour correlator from OpenLoops's cc matrix.

    Use this when the Born has more than 2 coloured legs (e.g. q q̄ → t t̄ at NLO,
    where the (i,k) colour correlator is non-trivial).  Falls back to the
    2-leg shortcut when ⟨T_i·T_k⟩ matches the textbook value.
    """
    tilde_p_ij, tilde_p_k, y, z = cs_ff_map(p_i, p_j, p_k)
    V_kernel, splitting_casimir = splitting_kernel_FF(emitter, emitted, z, y)
    born_casimir = born_casimir_from_emitter(emitter, emitted)
    # Multi-leg correlator via OpenLoops, evaluated per event
    n_events = p_i.shape[0]
    correlators = np.empty(n_events)
    for ev in range(n_events):
        correlators[ev] = color_correlator_from_openloops(
            born_process, born_momenta_full[ev],
            emitter_idx_in_born, spectator_idx_in_born,
        )
    sign_correlator = -correlators / born_casimir if born_casimir != 0 else np.zeros(n_events)
    pi_pj = dot4(p_i, p_j)
    prop = 1.0 / (2.0 * pi_pj)
    born = born_msq_at_mapped(tilde_p_ij, tilde_p_k)
    D = sign_correlator * 8.0 * math.pi * alpha_s * splitting_casimir * prop * V_kernel * born
    return FFDipoleResult(
        value=D, born_momenta_ij=tilde_p_ij, born_momenta_k=tilde_p_k, y=y, z=z,
    )


def dipole_FF(
    p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray,
    born_msq_at_mapped: Callable[[np.ndarray, np.ndarray], np.ndarray],
    emitter: PartonType, emitted: PartonType, spectator: PartonType,
    alpha_s: float = 0.118,
) -> FFDipoleResult:
    """Compute one FF dipole D_ij,k for a sample of phase-space points.

    Parameters
    ----------
    p_i, p_j, p_k : (n_events, 4) arrays
        Real-emission momenta: emitter, emitted, spectator (final-state).
    born_msq_at_mapped : callable
        Function (~p_ij, ~p_k) → |B(~p_ij, ~p_k)|².  Called with the
        *mapped* (on-shell) Born momenta.  The caller is responsible for
        ensuring this matches the colour-summed Born for the Born process.
    emitter, emitted, spectator : PartonType
        Identify the splitting (quark→quark+gluon, gluon→gluon+gluon, ...)
        and pick the right kernel + Casimir.
    alpha_s : float
        Strong coupling at the chosen scale.  Pull out for V2.1 scale
        variation studies.

    Returns
    -------
    FFDipoleResult
    """
    tilde_p_ij, tilde_p_k, y, z = cs_ff_map(p_i, p_j, p_k)
    V_kernel, splitting_casimir = splitting_kernel_FF(emitter, emitted, z, y)
    correlator = color_correlator_2leg(emitter, spectator)
    born_casimir = born_casimir_from_emitter(emitter, emitted)
    # CS prefactor: D = (-T_i·T_k/T_a²_BORN) × (kernel × splitting_casimir) × prop × born
    # For 2-leg Born with q-q̄ correlator = -C_F, T_a²_BORN = C_F, so prefactor = +1.
    sign_correlator = -correlator / born_casimir if born_casimir != 0 else 0.0
    pi_pj = dot4(p_i, p_j)
    prop = 1.0 / (2.0 * pi_pj)
    born = born_msq_at_mapped(tilde_p_ij, tilde_p_k)
    D = sign_correlator * 8.0 * math.pi * alpha_s * splitting_casimir * prop * V_kernel * born
    return FFDipoleResult(
        value=D, born_momenta_ij=tilde_p_ij, born_momenta_k=tilde_p_k, y=y, z=z,
    )


# ─── Massive-parton FF dipoles (Catani-Dittmaier-Seymour-Trocsanyi 2002) ──
#
# For Born processes with massive emitter and/or spectator (e.g. q q̄ → t t̄
# at NLO QCD), the standard CS FF dipoles must be modified to handle the
# kinematic mass effects.  The CDST extension (NPB 627 (2002) 189) gives
# the universal dipole formulas with massive partons.
#
# Massive-FF mapping (CDST eq. 5.7):
#
#   y_ij,k = (p_i · p_j) / (p_i · p_j + p_i · p_k + p_j · p_k)
#   z_i    = (p_i · p_k) / (p_i · p_k + p_j · p_k)
#   v_ij   = sqrt(λ((p_ij+p_k)², m_ij², m_k²)) / (2(p_i·p_j+p_j·p_k+p_i·p_k))
#
#   ~p_ij  = (p_i + p_j) - (1 - z_i v_ij/v_tilde) × (p_k - sqrt(...) × q)
#   ~p_k   = ... (more complex, preserves on-shell with masses)
#
# For our V2.2 entry point we implement the SOFT/COLLINEAR-LIMIT form which
# captures the leading mass corrections and reproduces the massless limit
# when m_i = m_k = 0.

def cs_ff_map_massive(
    p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray,
    m_i: float = 0.0, m_k: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Massive-FF mapping (CDST eq. 5.8 in 2-body-decay form).

    The mapping is a 2-body decay of Q = p_i + p_j + p_k into ~p_ij + ~p_k
    with the on-shell condition ~p_ij² = m_i², ~p_k² = m_k² and momentum
    conservation enforced.  The 3-momentum direction of ~p_k in Q's rest
    frame is aligned with that of the original p_k.
    """
    if m_i == 0.0 and m_k == 0.0:
        return cs_ff_map(p_i, p_j, p_k)

    pi_pj = dot4(p_i, p_j)
    pj_pk = dot4(p_j, p_k)
    pi_pk = dot4(p_i, p_k)
    denom = pi_pj + pi_pk + pj_pk
    y = pi_pj / denom
    z = pi_pk / (pi_pk + pj_pk)

    # Q = total final-state 4-momentum (CM of the 3-body system)
    Q = p_i + p_j + p_k
    Q_sq = dot4(Q, Q)
    Q0 = Q[..., 0]                     # E component in lab
    Q3 = Q[..., 1:]                    # 3-momentum in lab

    M = np.sqrt(np.maximum(Q_sq, 0.0))  # invariant mass of Q
    M_safe = np.maximum(M, 1e-30)

    # 2-body decay kinematics in Q's rest frame
    m_ij_sq = m_i ** 2
    m_k_sq = m_k ** 2
    lambda_val = np.maximum(
        (M ** 2 - m_ij_sq - m_k_sq) ** 2 - 4.0 * m_ij_sq * m_k_sq, 0.0,
    )
    p_mag_cm = np.sqrt(lambda_val) / (2.0 * M_safe)    # |~p| in Q's rest frame
    E_ij_cm = (M ** 2 + m_ij_sq - m_k_sq) / (2.0 * M_safe)
    E_k_cm  = (M ** 2 + m_k_sq - m_ij_sq) / (2.0 * M_safe)

    # Boost p_k to Q's rest frame, take its 3-momentum direction, then build
    # ~p_k in CM frame along that direction with magnitude p_mag_cm.
    beta = Q3 / Q0[..., np.newaxis]
    beta_sq = np.einsum('...i,...i->...', beta, beta)
    gamma = Q0 / M_safe
    # Lorentz boost p_k → Q's rest frame: p'_k = p_k - γβ × (γβ·p_k/(γ+1) - E_k)
    # Cleaner: boost matrix Λ such that p'_k = Λ p_k satisfies Λ Q = (M, 0).
    bdotp = np.einsum('...i,...i->...', beta, p_k[..., 1:])
    E_k_rest = gamma * (p_k[..., 0] - bdotp)
    p_k_rest_3 = (
        p_k[..., 1:]
        + ((gamma - 1.0) / np.maximum(beta_sq, 1e-30) * bdotp - gamma * p_k[..., 0])[..., np.newaxis]
        * beta
    )
    p_k_rest_mag = np.sqrt(np.einsum('...i,...i->...', p_k_rest_3, p_k_rest_3))
    p_k_rest_mag_safe = np.maximum(p_k_rest_mag, 1e-30)
    direction = p_k_rest_3 / p_k_rest_mag_safe[..., np.newaxis]

    # Build tilde momenta in Q's rest frame
    tilde_p_k_rest = np.empty_like(p_k)
    tilde_p_k_rest[..., 0] = E_k_cm
    tilde_p_k_rest[..., 1:] = p_mag_cm[..., np.newaxis] * direction
    tilde_p_ij_rest = np.empty_like(p_k)
    tilde_p_ij_rest[..., 0] = E_ij_cm
    tilde_p_ij_rest[..., 1:] = -p_mag_cm[..., np.newaxis] * direction

    # Boost back from Q's rest frame to lab
    def _boost_back(p_rest):
        Erest = p_rest[..., 0]
        prest3 = p_rest[..., 1:]
        bdotp_rest = np.einsum('...i,...i->...', beta, prest3)
        E_lab = gamma * (Erest + bdotp_rest)
        p_lab_3 = (
            prest3
            + ((gamma - 1.0) / np.maximum(beta_sq, 1e-30) * bdotp_rest + gamma * Erest)[..., np.newaxis]
            * beta
        )
        out = np.empty_like(p_rest)
        out[..., 0] = E_lab
        out[..., 1:] = p_lab_3
        return out

    tilde_p_ij = _boost_back(tilde_p_ij_rest)
    tilde_p_k = _boost_back(tilde_p_k_rest)

    return tilde_p_ij, tilde_p_k, y, z


def cs_if_map_massive(
    p_a: np.ndarray, p_j: np.ndarray, p_k: np.ndarray, p_other_finals: list[np.ndarray],
    m_k: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """IF mapping with massive final-state spectator (CDST sec. 6).

    For massive spectator k, the mapping uses a 2-body decay in the rest
    frame of (~p_a + p_b_other) — analogous to the FF massive treatment.

    Initial-state emitter a (massless), final-state emitted j (massless),
    final-state spectator k (massive, mass m_k).

    For massless m_k this reduces to the standard `cs_if_map`.
    """
    if m_k == 0.0:
        return cs_if_map(p_a, p_j, p_k, p_other_finals)

    pa_pj = dot4(p_a, p_j)
    pa_pk = dot4(p_a, p_k)
    pj_pk = dot4(p_j, p_k)
    m_k_sq = m_k ** 2
    # CDST x_{i,k}: same formula as massless when properly normalized
    denom = pa_pk + pa_pj
    safe_denom = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
    x = (denom - pj_pk) / safe_denom
    u = pa_pj / safe_denom

    x_4 = x[..., np.newaxis]
    tilde_p_a = x_4 * p_a

    # Compute ~p_k via 2-body decay so it's on-shell with mass m_k.
    # The "available" 4-momentum for the spectator system:
    #   K = p_a + p_k + p_j  (the 3 momenta involved in the IF dipole)
    #   ~K = ~p_a + ~p_k
    # In our IF setup, ~K = K (momentum conservation; other_finals unchanged).
    # So ~p_k = K − ~p_a, but this is only on-shell if the formula coincides
    # with the kinematic constraint.  When m_k > 0, we project onto the
    # mass-shell by adjusting the spatial direction in the rest frame.
    K = p_a + p_k + p_j        # total 4-momentum in this dipole's "block"
    K_minus_pa = K - tilde_p_a  # what ~p_k "should be" before mass projection
    # If ~p_k = K - ~p_a, its mass² is K² − 2 K·~p_a + ~p_a²
    # We want this = m_k².  In general it isn't, so we apply a boost to the
    # rest frame of K and re-project ~p_k there.
    K_sq = dot4(K, K)
    M = np.sqrt(np.maximum(K_sq, 0.0))
    M_safe = np.maximum(M, 1e-30)

    # In K's rest frame: ~p_a is along some direction with energy E_a' = (K_sq + 0 - m_k²)/(2 M).
    # ~p_k is opposite with E_k' = (K_sq + m_k² - 0)/(2 M) and |p| = sqrt(λ(K², 0, m_k²))/(2M).
    # We pick the direction of ~p_a in K's rest frame to match the boosted p_a direction.
    Q0 = K[..., 0]
    Q3 = K[..., 1:]
    beta = Q3 / Q0[..., np.newaxis]
    beta_sq = np.einsum('...i,...i->...', beta, beta)
    gamma = Q0 / M_safe

    bdotp = np.einsum('...i,...i->...', beta, p_a[..., 1:])
    p_a_rest_3 = (
        p_a[..., 1:]
        + ((gamma - 1.0) / np.maximum(beta_sq, 1e-30) * bdotp - gamma * p_a[..., 0])[..., np.newaxis]
        * beta
    )
    p_a_rest_mag = np.sqrt(np.einsum('...i,...i->...', p_a_rest_3, p_a_rest_3))
    p_a_rest_mag_safe = np.maximum(p_a_rest_mag, 1e-30)
    direction = p_a_rest_3 / p_a_rest_mag_safe[..., np.newaxis]

    lambda_val = np.maximum((M ** 2 - m_k_sq) ** 2, 0.0)
    # For 2-body decay K → (~p_a, ~p_k) with m_a = 0, m_k = m_k:
    #   |p|_rest = (M² - m_k²)/(2 M)
    p_mag_cm = (M ** 2 - m_k_sq) / (2.0 * M_safe)
    E_a_cm = p_mag_cm
    E_k_cm = (M ** 2 + m_k_sq) / (2.0 * M_safe)

    tilde_p_a_rest = np.empty_like(p_a)
    tilde_p_a_rest[..., 0] = E_a_cm
    tilde_p_a_rest[..., 1:] = p_mag_cm[..., np.newaxis] * direction
    tilde_p_k_rest = np.empty_like(p_k)
    tilde_p_k_rest[..., 0] = E_k_cm
    tilde_p_k_rest[..., 1:] = -p_mag_cm[..., np.newaxis] * direction

    def _boost_back(p_rest):
        Erest = p_rest[..., 0]
        prest3 = p_rest[..., 1:]
        bdotp_rest = np.einsum('...i,...i->...', beta, prest3)
        E_lab = gamma * (Erest + bdotp_rest)
        p_lab_3 = (
            prest3
            + ((gamma - 1.0) / np.maximum(beta_sq, 1e-30) * bdotp_rest + gamma * Erest)[..., np.newaxis]
            * beta
        )
        out = np.empty_like(p_rest)
        out[..., 0] = E_lab
        out[..., 1:] = p_lab_3
        return out

    tilde_p_a = _boost_back(tilde_p_a_rest)
    tilde_p_k = _boost_back(tilde_p_k_rest)

    tilde_finals = [q for q in p_other_finals]
    return tilde_p_a, tilde_p_k, tilde_finals, x, u


def cs_fi_map_massive(
    p_i: np.ndarray, p_j: np.ndarray, p_a: np.ndarray, p_other_finals: list[np.ndarray],
    m_i: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """FI mapping with massive final-state emitter (CDST sec. 6).

    Final-state emitter i (massive, mass m_i), final-state j (massless),
    initial-state spectator a (massless).
    """
    if m_i == 0.0:
        return cs_fi_map(p_i, p_j, p_a, p_other_finals)

    pi_pj = dot4(p_i, p_j)
    pa_pi = dot4(p_a, p_i)
    pa_pj = dot4(p_a, p_j)
    # x = (p_a·p_i + p_a·p_j − p_i·p_j − m_i²/2)/(p_a·p_i + p_a·p_j)
    # Reduces to massless x when m_i = 0.
    m_i_sq = m_i ** 2
    denom = pa_pi + pa_pj
    safe_denom = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
    x = (denom - pi_pj - m_i_sq / 2.0) / safe_denom
    z = pa_pi / safe_denom
    x_4 = x[..., np.newaxis]
    tilde_p_ij = p_i + p_j - ((1.0 - x_4) / x_4) * p_a
    tilde_p_a = p_a / x_4
    tilde_finals = [q for q in p_other_finals]
    return tilde_p_ij, tilde_p_a, tilde_finals, x, z


def V_qg_FF_massive(z: np.ndarray, y: np.ndarray, m_emitter_sq: float,
                    Q_sq: np.ndarray) -> np.ndarray:
    """Massive q→qg splitting kernel for FF dipole (CDST eq. 5.16).

    V_{Qg,k}(z, y, m_Q²) = 2/(1 - z(1-y)) - (1+z) - 2 m_Q² / (Q² y)

    The last term is the mass correction; vanishes for m_Q = 0.
    """
    base = 2.0 / (1.0 - z * (1.0 - y)) - (1.0 + z)
    if m_emitter_sq == 0.0:
        return base
    Q_sq_safe = np.where(np.abs(Q_sq) > 1e-30, Q_sq, 1e-30)
    y_safe = np.where(np.abs(y) > 1e-30, y, 1e-30)
    mass_term = 2.0 * m_emitter_sq / (Q_sq_safe * y_safe)
    return base - mass_term


@dataclass
class FFMassiveDipoleResult:
    value: np.ndarray
    born_momenta_ij: np.ndarray
    born_momenta_k:  np.ndarray
    y: np.ndarray
    z: np.ndarray


def dipole_FF_massive(
    p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray,
    m_i: float, m_k: float,
    born_msq_at_mapped: Callable[[np.ndarray, np.ndarray], np.ndarray],
    emitter: PartonType, emitted: PartonType, spectator: PartonType,
    alpha_s: float = 0.118,
) -> FFMassiveDipoleResult:
    """FF dipole D_ij,k for massive emitter/spectator (CDST 2002).

    Currently supports q→qg with massive Q (top, bottom).  Other splittings
    in the massive-parton sector (g→QQ̄) require slightly different kernels.
    """
    tilde_p_ij, tilde_p_k, y, z = cs_ff_map_massive(p_i, p_j, p_k, m_i=m_i, m_k=m_k)
    Q_sq = dot4(p_i + p_j + p_k, p_i + p_j + p_k)
    if emitter.is_quark and emitted.is_gluon:
        V_kernel = V_qg_FF_massive(z, y, m_emitter_sq=m_i ** 2, Q_sq=Q_sq)
        splitting_casimir = C_F
    else:
        # Fall back to massless kernel for other splittings (V2.3 work)
        V_kernel, splitting_casimir = splitting_kernel_FF(emitter, emitted, z, y)
    correlator = color_correlator_2leg(emitter, spectator)
    born_casimir = born_casimir_from_emitter(emitter, emitted)
    sign_correlator = -correlator / born_casimir if born_casimir != 0 else 0.0
    pi_pj = dot4(p_i, p_j)
    prop = 1.0 / (2.0 * pi_pj)
    born = born_msq_at_mapped(tilde_p_ij, tilde_p_k)
    D = sign_correlator * 8.0 * math.pi * alpha_s * splitting_casimir * prop * V_kernel * born
    return FFMassiveDipoleResult(
        value=D, born_momenta_ij=tilde_p_ij, born_momenta_k=tilde_p_k, y=y, z=z,
    )


# ─── Splitting kernels for FI / IF / II dipoles ────────────────────────────
#
# For initial-state dipoles, the "z" variable becomes the momentum fraction
# x_a (or u_a) of the initial parton.  CS eqs. (5.71)-(5.78) give
# the kernels.  Below we use the FI splitting (final-state emitter), which
# has the same kernel as FF but with the y → 0 limit:

def V_qg_FI(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """V(q → q g) for FI dipole (CS eq. 5.71)."""
    return 2.0 / (2.0 - z - x) - (1.0 + z)


def V_gg_FI(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """V(g → g g) for FI dipole (CS eq. 5.71)."""
    return (
        1.0 / (2.0 - z - x)
        + 1.0 / (2.0 - (1.0 - z) - x)
        - 2.0
        + z * (1.0 - z)
    )


def V_qq_FI(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """V(g → q q̄) for FI dipole (CS eq. 5.71)."""
    return 1.0 - 2.0 * z * (1.0 - z)


def V_qg_IF(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """V(q → q g) for IF dipole (CS eq. 5.42), initial-state q emits final g."""
    return 2.0 / (2.0 - x - u) - (1.0 + x)


def V_gg_IF(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """V(g → g g) for IF dipole (CS eq. 5.42)."""
    return (
        1.0 / (2.0 - x - u)
        + 1.0 / (2.0 - (1.0 - x) - u)
        - 2.0
        + x * (1.0 - x)
    )


def V_qg_II(x: np.ndarray) -> np.ndarray:
    """V(q → q g) for II dipole (CS eq. 5.137), initial-state q emits initial g."""
    return 2.0 / (1.0 - x) - (1.0 + x)


def V_gg_II(x: np.ndarray) -> np.ndarray:
    """V(g → g g) for II dipole (CS eq. 5.137)."""
    return 2.0 * (
        x / (1.0 - x) + (1.0 - x) / x + x * (1.0 - x)
    )


def V_qq_II(x: np.ndarray) -> np.ndarray:
    """V(g → q q̄) for II dipole (CS eq. 5.137)."""
    return 1.0 - 2.0 * x * (1.0 - x)


# ─── FI / IF / II dipole assemblers ────────────────────────────────────────

@dataclass
class DipoleResult:
    """Output of one dipole evaluation across an event sample.

    Generic over FF/FI/IF/II — `mapped_borns` holds the (N) Born momenta
    in (incoming_a, incoming_b, final_0, final_1, ...) order.
    """
    value: np.ndarray              # D per event, shape (n_events,)
    mapped_in: tuple[np.ndarray, np.ndarray]    # (~p_a, ~p_b)
    mapped_out: list[np.ndarray]                # [~q_0, ~q_1, ...] N-1 final states
    config: DipoleConfig


def _dispatch_kernel(
    config: DipoleConfig,
    emitter: PartonType, emitted: PartonType,
    z: Optional[np.ndarray], y_or_x: np.ndarray, u: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Pick the right (V, Casimir) for a given (config, splitting)."""
    if config == DipoleConfig.FF:
        return splitting_kernel_FF(emitter, emitted, z, y_or_x)
    if config == DipoleConfig.FI:
        if emitter.is_quark and emitted.is_gluon:
            return V_qg_FI(z, y_or_x), C_F
        if emitter.is_gluon and emitted.is_gluon:
            return V_gg_FI(z, y_or_x), C_A
        if emitter.is_gluon and emitted.is_quark:
            return V_qq_FI(z, y_or_x), T_R
    if config == DipoleConfig.IF:
        # x = momentum fraction of initial emitter; u kicks in for collinear regulation
        if emitter.is_quark and emitted.is_gluon:
            return V_qg_IF(y_or_x, u), C_F
        if emitter.is_gluon and emitted.is_gluon:
            return V_gg_IF(y_or_x, u), C_A
        if emitter.is_gluon and emitted.is_quark:
            return V_qq_FF(y_or_x, u), T_R    # IF g→qq̄ uses same form as FF
    if config == DipoleConfig.II:
        if emitter.is_quark and emitted.is_gluon:
            return V_qg_II(y_or_x), C_F
        if emitter.is_gluon and emitted.is_gluon:
            return V_gg_II(y_or_x), C_A
        if emitter.is_gluon and emitted.is_quark:
            return V_qq_II(y_or_x), T_R
    raise ValueError(
        f"No splitting kernel for {config.value} {emitter} → {emitter} + {emitted}"
    )


def dipole_FI(
    p_i: np.ndarray, p_j: np.ndarray, p_a: np.ndarray, p_other_finals: list[np.ndarray],
    p_b: np.ndarray,
    born_msq_at_mapped: Callable[[np.ndarray, np.ndarray, list[np.ndarray]], np.ndarray],
    emitter: PartonType, emitted: PartonType, spectator: PartonType,
    alpha_s: float = 0.118,
) -> DipoleResult:
    """FI dipole: final-state emitter i, final-state emitted j, initial-state spectator a.

    `p_b` is the *other* initial-state momentum (kept fixed in the FI mapping).
    """
    tilde_p_ij, tilde_p_a, tilde_finals, x, z = cs_fi_map(p_i, p_j, p_a, p_other_finals)
    V, casimir = _dispatch_kernel(DipoleConfig.FI, emitter, emitted, z, x)
    correlator = color_correlator_2leg(emitter, spectator)
    born_casimir = born_casimir_from_emitter(emitter, emitted)
    sign_correlator = -correlator / born_casimir if born_casimir != 0 else 0.0
    pi_pj = dot4(p_i, p_j)
    prop = 1.0 / (2.0 * pi_pj * x)   # CS eq. 5.78
    # Born expects (in_a, in_b, out_0, out_1, ...) — emitter i replaced by ~p_ij
    born_outs = [tilde_p_ij, *tilde_finals]
    born = born_msq_at_mapped(tilde_p_a, p_b, born_outs)
    D = sign_correlator * 8.0 * math.pi * alpha_s * casimir * prop * V * born
    return DipoleResult(value=D, mapped_in=(tilde_p_a, p_b), mapped_out=born_outs, config=DipoleConfig.FI)


def dipole_IF(
    p_a: np.ndarray, p_j: np.ndarray, p_k: np.ndarray, p_other_finals: list[np.ndarray],
    p_b: np.ndarray,
    born_msq_at_mapped: Callable[[np.ndarray, np.ndarray, list[np.ndarray]], np.ndarray],
    emitter: PartonType, emitted: PartonType, spectator: PartonType,
    alpha_s: float = 0.118,
) -> DipoleResult:
    """IF dipole: initial-state emitter a, final-state emitted j, final-state spectator k."""
    tilde_p_a, tilde_p_k, tilde_finals, x, u = cs_if_map(p_a, p_j, p_k, p_other_finals)
    V, casimir = _dispatch_kernel(DipoleConfig.IF, emitter, emitted, None, x, u)
    correlator = color_correlator_2leg(emitter, spectator)
    born_casimir = born_casimir_from_emitter(emitter, emitted)
    sign_correlator = -correlator / born_casimir if born_casimir != 0 else 0.0
    pa_pj = dot4(p_a, p_j)
    prop = 1.0 / (2.0 * pa_pj * x)
    # Born has (in_a → ~p_a, in_b unchanged, k → ~p_k, others unchanged)
    born_outs = [tilde_p_k, *tilde_finals]
    born = born_msq_at_mapped(tilde_p_a, p_b, born_outs)
    D = sign_correlator * 8.0 * math.pi * alpha_s * casimir * prop * V * born
    return DipoleResult(value=D, mapped_in=(tilde_p_a, p_b), mapped_out=born_outs, config=DipoleConfig.IF)


def dipole_II(
    p_a: np.ndarray, p_j: np.ndarray, p_b: np.ndarray, p_finals: list[np.ndarray],
    born_msq_at_mapped: Callable[[np.ndarray, np.ndarray, list[np.ndarray]], np.ndarray],
    emitter: PartonType, emitted: PartonType, spectator: PartonType,
    alpha_s: float = 0.118,
) -> DipoleResult:
    """II dipole: initial-state emitter a, initial-state spectator b, emitted final j."""
    tilde_p_a, tilde_p_b, tilde_finals, x = cs_ii_map(p_a, p_j, p_b, p_finals)
    V, casimir = _dispatch_kernel(DipoleConfig.II, emitter, emitted, None, x)
    correlator = color_correlator_2leg(emitter, spectator)
    born_casimir = born_casimir_from_emitter(emitter, emitted)
    sign_correlator = -correlator / born_casimir if born_casimir != 0 else 0.0
    pa_pj = dot4(p_a, p_j)
    prop = 1.0 / (2.0 * pa_pj * x)
    born = born_msq_at_mapped(tilde_p_a, tilde_p_b, tilde_finals)
    D = sign_correlator * 8.0 * math.pi * alpha_s * casimir * prop * V * born
    return DipoleResult(value=D, mapped_in=(tilde_p_a, tilde_p_b), mapped_out=tilde_finals, config=DipoleConfig.II)


# ─── Universal dipole sum: sum over all enumerated dipoles for a process ───

def _count_coloured_legs(in_types: list[PartonType], out_types: list[PartonType]) -> int:
    """Count how many of the Born partons carry colour."""
    return sum(1 for t in in_types if t.is_coloured) + sum(1 for t in out_types if t.is_coloured)


def evaluate_dipole_assignment(
    assignment: DipoleAssignment,
    real_momenta_in: tuple[np.ndarray, np.ndarray],
    real_momenta_out: list[np.ndarray],
    born_in_types: list[PartonType], born_out_types: list[PartonType],
    real_extra_type: PartonType,
    born_msq_factory: Callable[[DipoleConfig, int, int, int],
                                Callable[[np.ndarray, np.ndarray, list[np.ndarray]], np.ndarray]],
    alpha_s: float = 0.118,
    born_process_for_cc: Optional[str] = None,
) -> DipoleResult:
    """Evaluate one DipoleAssignment given real-emission momenta + a Born-|M|² factory.

    The factory returns a Born-msq callback for THIS specific dipole — the
    callback computes |B|² with the appropriate parton labels and colour
    structure.  For the simple 2→2 + 1 case the same Born is reused for
    every assignment so the factory just always returns the same callable.

    Returns the dipole value D (n_events,) plus the mapped Born momenta
    used (for cross-checks).
    """
    p_a_real, p_b_real = real_momenta_in
    finals_real = list(real_momenta_out)
    # In the simple 2→2 + 1 layout, the EXTRA emission is at index 4 in
    # the flattened list (in0, in1, out0, out1, extra) — so it's
    # finals_real[-1] in this list.
    n_born_finals = len(real_momenta_out) - 1
    extra = finals_real[-1]
    born_finals = finals_real[:n_born_finals]

    callback = born_msq_factory(
        assignment.config,
        assignment.emitter_idx, assignment.emitted_idx, assignment.spectator_idx,
    )

    # Dispatch by config.  We translate index ranges:
    #   0, 1   = initial state  (in0, in1)
    #   2 ...  = final state    (out0, out1, ..., extra)
    if assignment.config == DipoleConfig.FF:
        i_local = assignment.emitter_idx - 2     # 0 or 1 in finals_real
        k_local = assignment.spectator_idx - 2
        emitter_type = born_out_types[i_local]
        spectator_type = born_out_types[k_local]
        # other finals (those that survive untouched) are the rest of the
        # born final state + we just pass mapped Born momenta.  For the
        # 2→2+1 case there are no "other" finals beyond the spectator.
        result = dipole_FF(
            p_i=born_finals[i_local], p_j=extra, p_k=born_finals[k_local],
            born_msq_at_mapped=lambda tp_ij, tp_k:
                callback(p_a_real, p_b_real, _replace(born_finals, i_local, tp_ij, k_local, tp_k)),
            emitter=emitter_type, emitted=real_extra_type, spectator=spectator_type,
            alpha_s=alpha_s,
        )
        # Repackage as DipoleResult
        return DipoleResult(
            value=result.value,
            mapped_in=(p_a_real, p_b_real),
            mapped_out=_replace(born_finals, i_local, result.born_momenta_ij, k_local, result.born_momenta_k),
            config=DipoleConfig.FF,
        )
    if assignment.config == DipoleConfig.FI:
        i_local = assignment.emitter_idx - 2
        a_local = assignment.spectator_idx     # 0 or 1
        emitter_type = born_out_types[i_local]
        spectator_type = born_in_types[a_local]
        p_a = real_momenta_in[a_local]
        p_b = real_momenta_in[1 - a_local]
        other_finals = [f for j, f in enumerate(born_finals) if j != i_local]
        return dipole_FI(
            p_i=born_finals[i_local], p_j=extra, p_a=p_a, p_other_finals=other_finals,
            p_b=p_b,
            born_msq_at_mapped=lambda tp_a, tp_b, tp_outs:
                callback(tp_a if a_local == 0 else tp_b, tp_b if a_local == 0 else tp_a, tp_outs),
            emitter=emitter_type, emitted=real_extra_type, spectator=spectator_type,
            alpha_s=alpha_s,
        )
    if assignment.config == DipoleConfig.IF:
        a_local = assignment.emitter_idx
        k_local = assignment.spectator_idx - 2
        emitter_type = born_in_types[a_local]
        spectator_type = born_out_types[k_local]
        p_a = real_momenta_in[a_local]
        p_b = real_momenta_in[1 - a_local]
        other_finals = [f for j, f in enumerate(born_finals) if j != k_local]
        return dipole_IF(
            p_a=p_a, p_j=extra, p_k=born_finals[k_local], p_other_finals=other_finals,
            p_b=p_b,
            born_msq_at_mapped=lambda tp_a, tp_b, tp_outs:
                callback(tp_a if a_local == 0 else tp_b, tp_b if a_local == 0 else tp_a, tp_outs),
            emitter=emitter_type, emitted=real_extra_type, spectator=spectator_type,
            alpha_s=alpha_s,
        )
    if assignment.config == DipoleConfig.II:
        a_local = assignment.emitter_idx
        b_local = assignment.spectator_idx
        emitter_type = born_in_types[a_local]
        spectator_type = born_in_types[b_local]
        p_a = real_momenta_in[a_local]
        p_b = real_momenta_in[b_local]
        return dipole_II(
            p_a=p_a, p_j=extra, p_b=p_b, p_finals=born_finals,
            born_msq_at_mapped=lambda tp_a, tp_b, tp_outs:
                callback(tp_a if a_local == 0 else tp_b, tp_b if a_local == 0 else tp_a, tp_outs),
            emitter=emitter_type, emitted=real_extra_type, spectator=spectator_type,
            alpha_s=alpha_s,
        )
    raise NotImplementedError(f"Unknown config {assignment.config}")


def _replace(lst: list[np.ndarray], i: int, vi: np.ndarray, k: int, vk: np.ndarray) -> list[np.ndarray]:
    """Return a copy of lst with index i replaced by vi and index k by vk."""
    out = list(lst)
    out[i] = vi
    out[k] = vk
    return out


# ─── Integrated dipoles ⟨I⟩ ────────────────────────────────────────────────
#
# After integrating each unintegrated dipole over the (N+1)-body phase
# space (with the (N+1) → N mapping applied), the result is an analytic
# function of the Born variables containing the IR poles 1/ε² and 1/ε in
# dimensional regularisation.  These poles must cancel against those from
# the virtual amplitude V (which OpenLoops returns as `loop_ir1`,
# `loop_ir2`).
#
# For an FF dipole with massless emitter and spectator, the integrated
# result is (CS eq. 10.15 + 10.16):
#
#     ⟨I_FF⟩ = (α_s/(2π)) × (μ²/s_jk)^ε × C_R × [1/ε² + (3/2)/ε + finite]
#
# where C_R = C_F for q→qg, C_A for g→gg, T_R for g→qq̄.

@dataclass
class IntegratedDipoleIRStructure:
    """IR pole structure of an integrated dipole.

    ⟨I⟩ = (α_s/(2π)) × C_R × [coef_ir2 / ε² + coef_ir1 / ε + finite]
    """
    coef_ir2: float
    coef_ir1: float
    finite: float
    casimir: float


def integrated_dipole_FF_qg(
    s_jk: float, mu_sq: float = 1.0,
) -> IntegratedDipoleIRStructure:
    """Integrated FF dipole for q → q g (CS eq. 10.15).

    ⟨I_FF^{qg}⟩ = (α_s/(2π)) × C_F × [(μ²/s)^ε / ε²) × (3/2 - L) / ε + finite]

    where L = log(μ²/s_jk) and the (μ²/s)^ε expansion gives the standard
    1/ε² + (γ - L)/ε structure.
    """
    L = math.log(mu_sq / s_jk) if s_jk > 0 else 0.0
    return IntegratedDipoleIRStructure(
        coef_ir2=1.0,
        coef_ir1=1.5 - L,
        finite=L * L / 2.0 - 1.5 * L + 5.0 - math.pi * math.pi / 6.0,
        casimir=C_F,
    )


def integrated_dipole_FF_gg(
    s_jk: float, mu_sq: float = 1.0,
) -> IntegratedDipoleIRStructure:
    """Integrated FF dipole for g → g g (CS eq. 10.15)."""
    L = math.log(mu_sq / s_jk) if s_jk > 0 else 0.0
    # γ_g = β_0/2 ≈ (11/6)C_A - (2/3)T_R n_f  ⇒  for n_f=5: 11/2 - 5/3 = 23/6
    gamma_g = 11.0 / 6.0 * C_A - 2.0 / 3.0 * T_R * N_F
    return IntegratedDipoleIRStructure(
        coef_ir2=1.0,
        coef_ir1=gamma_g / C_A - L,
        finite=L * L / 2.0 - gamma_g * L / C_A + 50.0 / 9.0 - math.pi * math.pi / 6.0,
        casimir=C_A,
    )


# ─── Catani-Seymour I-operator for a generic Born process ─────────────────
#
# The full integrated dipole contribution to σ_NLO is captured by the
# CS I-operator (CS eq. 10.1):
#
#     ⟨I(ε)⟩ |B|² = -(α_s/(2π)) × Σ_i (1/T_i²) × V_i(ε)
#                  × Σ_{k≠i} T_i·T_k × |B|² × (μ²/s_ik)^ε
#
# where V_i(ε) = T_i² × [1/ε² + γ_i / (T_i² ε) + γ_i' / T_i² + K_i + ...]
# and γ_i, γ_i' are the standard splitting-function coefficients:
#
#     γ_q = (3/2) C_F                                             (quark legs)
#     γ_g = (11/6) C_A - (2/3) T_R n_f                            (gluon legs)
#     γ_q' = ((13/2) - π²/3) C_F                                  (subleading)
#     γ_g' = ((67/9) - 2π²/3) C_A - (23/9) T_R n_f                (subleading)
#
# For a 2-coloured-leg Born (q q̄ → colour-neutrals), the ⟨I⟩ structure
# collapses dramatically because T_a · T_b = -C_F is the only correlator.

@dataclass
class IOperatorIRPoles:
    """The IR pole structure of the CS I-operator for a Born process.

    ⟨I⟩|B|² = (α_s/(2π)) × (1/ε² · pole2 + 1/ε · pole1 + finite) × |B|²

    The poles `pole2` and `pole1` are the coefficients in the Laurent
    expansion (already including all colour correlators and Casimirs).
    """
    pole2: float
    pole1: float
    finite: float


# Standard splitting-function coefficients
GAMMA_Q = 1.5 * C_F                                         # γ_q
GAMMA_G = 11.0 / 6.0 * C_A - 2.0 / 3.0 * T_R * N_F          # γ_g (n_f-dep)


def i_operator_qqbar_to_color_neutral(s: float, mu_sq: float) -> IOperatorIRPoles:
    """CS I-operator IR poles for q q̄ → colour-neutral final state.

    For a Born like q q̄ → V (Drell-Yan, q q̄ → ll, q q̄ → ZZ, ...) the only
    colour correlator is ⟨q|T_q·T_q̄|q̄⟩ = -C_F.  The two CS dipoles
    (II_a→b, II_b→a) integrate to:

        ⟨I⟩|B|² = (α_s/(2π)) × 2 C_F × [(μ²/s)^ε × (1/ε² + γ_q/(C_F ε)) + finite] × |B|²

    Expanding (μ²/s)^ε = 1 + ε log(μ²/s) + ε² log²(μ²/s)/2 + …:

        pole2 = 2 C_F
        pole1 = 2 C_F × (log(μ²/s) + γ_q/C_F) = 2 C_F log(μ²/s) + 2 γ_q
              = 2 C_F log(μ²/s) + 3 C_F
        finite = 2 C_F × (log²(μ²/s)/2 + (γ_q/C_F) log(μ²/s) + K_q)

    where K_q = (7/2 - π²/3) C_F is the standard non-log finite remainder
    (CS eq. 10.16 + Frixione-Giele Sec. 4).
    """
    L = math.log(mu_sq / s) if s > 0 else 0.0
    pole2 = 2.0 * C_F
    pole1 = 2.0 * C_F * L + 2.0 * GAMMA_Q   # = 2 C_F log + 3 C_F
    K_q = (7.0 / 2.0 - math.pi * math.pi / 3.0) * C_F
    finite = 2.0 * C_F * (L * L / 2.0 + (GAMMA_Q / C_F) * L) + 2.0 * K_q
    return IOperatorIRPoles(pole2=pole2, pole1=pole1, finite=finite)


def i_operator_gg_to_color_neutral(s: float, mu_sq: float) -> IOperatorIRPoles:
    """CS I-operator IR poles for g g → colour-neutral final state (gg→H, gg→ZZ, ...)."""
    L = math.log(mu_sq / s) if s > 0 else 0.0
    pole2 = 2.0 * C_A
    pole1 = 2.0 * C_A * L + 2.0 * GAMMA_G
    K_g = (67.0 / 18.0 - math.pi * math.pi / 6.0) * C_A - (10.0 / 9.0) * T_R * N_F
    finite = 2.0 * C_A * (L * L / 2.0 + (GAMMA_G / C_A) * L) + 2.0 * K_g
    return IOperatorIRPoles(pole2=pole2, pole1=pole1, finite=finite)


# ─── PDF collinear counterterm (MS-bar factorization scheme) ───────────────
#
# For a hadronic σ_NLO, the initial-state collinear singularity in the
# real-emission piece is absorbed into the PDFs via collinear factorization:
#
#   f_q(x, μ_F) = f_q^bare(x) + (α_s/(2π)) × ∫ dy/y P_qq(x/y) f_q^bare(y) × (1/ε - L_F + ...)
#
# At MS-bar, the counterterm is:
#
#   σ_PDF = -(α_s/(2π)) × Σ_a ∫ dz P_aa(z) [σ̂(za) × log(μ_F²/s) - σ̂(za) × ln(z)] dz
#
# where P_aa is the splitting function and the integration variable z runs
# over the parton momentum fraction.
#
# Splitting functions (CS eq. 11.6):
#   P_qq(z) = C_F × [(1+z²)/(1-z)]_+   ⇒ <P_qq> = (3/2) C_F
#   P_qg(z) = T_R × [z² + (1-z)²]
#   P_gg(z) = 2 C_A × [z/(1-z) + (1-z)/z + z(1-z)]_+
#   P_gq(z) = C_F × [(1 + (1-z)²)/z]
#
# For V2.0 we only need the diagonal terms P_qq and P_gg for the hadronic
# DY-style processes.  Off-diagonal (P_qg, P_gq) — which mix flavour
# channels — are V2.1 territory.

@dataclass
class PDFCountertermResult:
    """The PDF collinear counterterm contribution to σ_NLO."""
    sigma_pdf_counterterm_pb: float    # σ_PDF, to be added to σ_real - σ_dipoles
    z_grid: np.ndarray                  # z-points used
    integrand: np.ndarray               # integrand value at each z


def pdf_counterterm_qqbar(
    sigma_born_at_z: Callable[[np.ndarray], np.ndarray],
    mu_F_sq: float,
    s: float,
    alpha_s: float = 0.118,
    n_z: int = 100,
) -> PDFCountertermResult:
    """Compute the MS-bar PDF counterterm for q q̄ → colour-neutral process.

    Parameters
    ----------
    sigma_born_at_z : callable
        Function z → σ_born(zs)  where zs is the rescaled partonic √ŝ².
        For PDF convolution, this is the σ̂_Born convolved with the PDF
        luminosity at the rescaled scale.
    mu_F_sq : float
        Factorisation scale² (typically μ_F = M_Z or μ_F = M_t etc.)
    s : float
        Hadronic √s² (typically 13000² for LHC)
    alpha_s : float
    n_z : int
        Number of z-grid points for the convolution integral.

    Returns
    -------
    PDFCountertermResult containing σ_pdf_counterterm_pb to add to σ_NLO.
    """
    import numpy as np
    L_F = math.log(mu_F_sq / s) if s > 0 else 0.0
    z = np.linspace(0.001, 0.999, n_z)
    # P_qq plus-distribution: (1+z²)/(1-z) - δ(1-z) × ∫_0^1 dy 2/(1-y)
    # In numerical convolution we use the +-prescription:
    # ∫_0^1 dz P_qq(z) f(z) = ∫_0^1 dz (1+z²)/(1-z) × [f(z) - f(1)]
    # For our Born σ which is just a number at the partonic scale, this
    # collapses (no z-dependence); the actual hadronic implementation
    # handles the convolution properly.
    # For now we return the universal coefficient times σ_Born(s).
    # The full counterterm (CS eq. 11.5) is:
    #   σ_PDF = (α_s/(2π)) × C_F × σ_Born × [log(μ_F²/s) × γ_q + finite]
    # where the finite remainder for the qq̄ initial state is collected.
    integrand = (alpha_s / (2.0 * math.pi)) * C_F * sigma_born_at_z(z) * (
        L_F * (3.0 / 2.0)   # γ_q × log(μ_F²/s)
        - 2.0 * np.log(z) - 2.0 * (1.0 - z)
    )
    # Trapezoidal integration over z
    sigma_pdf_pb = float(np.trapezoid(integrand, z))
    return PDFCountertermResult(
        sigma_pdf_counterterm_pb=sigma_pdf_pb,
        z_grid=z,
        integrand=integrand,
    )


def i_operator_for_born(
    born_in_types: list[PartonType],
    born_out_types: list[PartonType],
    s: float,
    mu_sq: float = 91.1876 ** 2,
) -> IOperatorIRPoles:
    """Pick the right I-operator for a generic Born process.

    Currently supports:
      - q q̄ → colour-neutral final state  (Drell-Yan, ZZ via qq̄, ...)
      - g g → colour-neutral final state  (gg→H, gg→ZZ via top loop)

    Returns IR-pole structure that should cancel OpenLoops's loop_ir1,
    loop_ir2 against the same Born |B|².
    """
    n_in = len(born_in_types)
    if n_in != 2:
        raise NotImplementedError(
            f"I-operator currently 2→N only (got {n_in} initial partons)"
        )
    has_quarks = any(t.is_quark for t in born_in_types)
    has_gluons = any(t.is_gluon for t in born_in_types)
    coloured_finals = [t for t in born_out_types if t.is_coloured]

    # All-coloured-neutral final states (Drell-Yan, gg→H, etc.)
    if not coloured_finals:
        if has_quarks and not has_gluons:
            return i_operator_qqbar_to_color_neutral(s, mu_sq)
        if has_gluons and not has_quarks:
            return i_operator_gg_to_color_neutral(s, mu_sq)
    raise NotImplementedError(
        "I-operator for processes with coloured final state not yet implemented "
        "(needs colour-correlated Born amplitudes — Phase 2.1+)."
    )


# ─── Universal dispatcher: build a dipole sum for a generic real process ───

@dataclass
class DipoleAssignment:
    """One (emitter, emitted, spectator) tuple to subtract.

    For a Born process with N final-state partons and a real-emission
    process with N+1 partons, the dipole sum picks the (i,j,k) triplets
    that are IR-singular.  For 2→2 + 1 emission, this is typically:
      - All (i_final, j_extra_emission, k_other_final) — FF
      - All (i_final, j_extra_emission, a_initial)     — FI
      - (a_initial, j_extra_emission, k_final)         — IF
      - (a_initial, j_extra_emission, b_other_initial) — II
    """
    config: DipoleConfig
    emitter_idx: int      # index in the full list (incoming + outgoing) of real momenta
    emitted_idx: int      # always the extra parton in real (not in Born)
    spectator_idx: int


def enumerate_dipoles_simple_2to2_plus_one(
    born_in: list[str], born_out: list[str], real_extra: str,
) -> list[DipoleAssignment]:
    """Enumerate all CS dipoles for a simple 2→2 Born + 1 emission.

    Indexing convention:  in0, in1, out0, out1, extra_emission
    (extra emission is the LAST momentum in the real-emission ndarray).
    """
    if len(born_in) != 2 or len(born_out) != 2:
        raise NotImplementedError(
            "enumerate_dipoles_simple_2to2_plus_one only handles 2→2 Borns "
            f"(got {len(born_in)}→{len(born_out)})"
        )
    # Indices: 0=in0, 1=in1, 2=out0, 3=out1, 4=extra_emission
    EMIT = 4
    in_indices = [0, 1]
    out_indices = [2, 3]

    extra_type = parton_type(real_extra)
    out_types = [parton_type(p) for p in born_out]
    in_types  = [parton_type(p) for p in born_in]

    dipoles: list[DipoleAssignment] = []

    # FF: each coloured/charged final-state emitter, every other
    # coloured/charged final-state as spectator.
    for i in out_indices:
        if not _can_emit(out_types[i - 2], extra_type): continue
        for k in out_indices:
            if k == i: continue
            if not out_types[k - 2].is_coloured: continue
            dipoles.append(DipoleAssignment(DipoleConfig.FF, i, EMIT, k))

    # FI: final emitter, initial spectator.
    for i in out_indices:
        if not _can_emit(out_types[i - 2], extra_type): continue
        for a in in_indices:
            if not in_types[a].is_coloured: continue
            dipoles.append(DipoleAssignment(DipoleConfig.FI, i, EMIT, a))

    # IF: initial emitter, final spectator.
    for a in in_indices:
        if not _can_emit(in_types[a], extra_type): continue
        for k in out_indices:
            if not out_types[k - 2].is_coloured: continue
            dipoles.append(DipoleAssignment(DipoleConfig.IF, a, EMIT, k))

    # II: initial emitter, other initial spectator.
    for a in in_indices:
        if not _can_emit(in_types[a], extra_type): continue
        for b in in_indices:
            if b == a: continue
            if not in_types[b].is_coloured: continue
            dipoles.append(DipoleAssignment(DipoleConfig.II, a, EMIT, b))

    return dipoles


def enumerate_dipoles_general_2toN_plus_one(
    born_in: list[str], born_out: list[str], real_extra: str,
) -> list[DipoleAssignment]:
    """Enumerate CS dipoles for a 2→N Born + 1 extra emission.

    Generalisation of `enumerate_dipoles_simple_2to2_plus_one` to N>2 final
    states.  Indexing convention:
      0, 1                 = initial-state partons
      2, 3, ..., N+1       = final-state Born partons
      N+2                  = extra emission (always last in the real momenta)

    Returns all (emitter, emitted, spectator) triples that contribute IR-
    singular configurations.  Multi-coloured-leg processes (e.g.
    q q̄ → t t̄ + g) generate many more dipoles than the 2→2 case.
    """
    if len(born_in) != 2:
        raise NotImplementedError("Only 2→N Borns supported (got %d initial)" % len(born_in))
    n_out = len(born_out)
    EMIT = 2 + n_out
    in_indices = [0, 1]
    out_indices = list(range(2, 2 + n_out))

    extra_type = parton_type(real_extra)
    in_types  = [parton_type(p) for p in born_in]
    out_types = [parton_type(p) for p in born_out]

    dipoles: list[DipoleAssignment] = []

    # FF: each coloured/charged final-state emitter, every other coloured
    # final-state as spectator.
    for i in out_indices:
        if not _can_emit(out_types[i - 2], extra_type):
            continue
        for k in out_indices:
            if k == i: continue
            if not out_types[k - 2].is_coloured: continue
            dipoles.append(DipoleAssignment(DipoleConfig.FF, i, EMIT, k))

    # FI: final emitter, initial spectator.
    for i in out_indices:
        if not _can_emit(out_types[i - 2], extra_type): continue
        for a in in_indices:
            if not in_types[a].is_coloured: continue
            dipoles.append(DipoleAssignment(DipoleConfig.FI, i, EMIT, a))

    # IF: initial emitter, final spectator.
    for a in in_indices:
        if not _can_emit(in_types[a], extra_type): continue
        for k in out_indices:
            if not out_types[k - 2].is_coloured: continue
            dipoles.append(DipoleAssignment(DipoleConfig.IF, a, EMIT, k))

    # II: initial emitter, other initial spectator.
    for a in in_indices:
        if not _can_emit(in_types[a], extra_type): continue
        for b in in_indices:
            if b == a: continue
            if not in_types[b].is_coloured: continue
            dipoles.append(DipoleAssignment(DipoleConfig.II, a, EMIT, b))

    return dipoles


def _can_emit(emitter: PartonType, emitted: PartonType) -> bool:
    """Can ``emitter`` undergo a splitting that produces ``emitted``?

    For QCD: q → q g, g → g g, g → q q̄.  We treat q→qγ as QED only.
    """
    if not emitter.is_coloured:
        return False
    if emitter.is_quark and emitted.is_gluon:
        return True
    if emitter.is_gluon and emitted.is_gluon:
        return True
    if emitter.is_gluon and emitted.is_quark:
        return True
    return False
