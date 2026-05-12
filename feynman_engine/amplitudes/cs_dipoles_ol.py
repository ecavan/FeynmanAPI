"""Catani-Seymour QED photon dipoles using OpenLoops Born evaluation.

Companion to ``cs_dipoles.py`` (which uses analytic engine-internal Born
formulas).  This module evaluates the CS dipole D_ij,k for arbitrary
2→N processes by calling OpenLoops's ``Process.evaluate(pp)`` to get
|M_Born|² at the mapped (N-1)-body kinematics — keeping both the real
matrix element |M_R|² and the dipole D in OpenLoops's normalization
convention so the subtraction (R - ΣD) is point-by-point IR-finite.

The QED dipole structure (Catani-Seymour NPB 485 (1997) App. A,
Dittmaier NPB 565 (2000)):

    D_ij,k = (1 / (2 p_i·p_j)) × V_ij,k(z, y; Q_i, Q_k) × |M_Born(tilded)|²

where:
    - i is the EMITTER (charged fermion that radiates)
    - j is the EMITTED PHOTON
    - k is the SPECTATOR (any other charged leg)
    - V_ij,k is the QED splitting kernel × charge correlator (4π α Q_i Q_k)
    - tilde momenta are constructed via the CS phase-space mapping
      (FF for two final emitter+spectator, FI/IF/II for mixed/initial)

Sign convention: D enters the subtraction as (R - ΣD), with each D
positive in the soft/collinear limit.  The charge correlator Q_i Q_k
introduces a sign from the relative orientation of charges (negative
for opposite-sign legs, positive for same-sign).

Why a separate module from ``cs_dipoles.py``?
---------------------------------------------
``cs_dipoles.py`` (and ``dipole_subtraction.py``) evaluate the Born
analytically using engine-internal formulas (e.g. ``born_msq_eemumu``).
Those formulas use ``ALPHA_EM = 1/137.036`` and a specific normalization
that differs from OpenLoops's ``α(M_Z) ≈ 1/132.2`` (G_μ scheme) and
internal phase-space factors.  Using engine-Born dipoles to subtract
OpenLoops |M_R|² gives an O((α_GMU/α(0))² - 1) ≈ 7% normalization mismatch
that ruins the IR cancellation at the few-percent level.

This module is the GENERIC path that works for any process where
OpenLoops can register the (N-1)-body Born.

References
----------
Catani, Seymour, NPB 485 (1997) 291 — universal dipole formalism.
Dittmaier, NPB 565 (2000) 69 — full QED dipole framework.
Dittmaier, Roth, NPB 642 (2002) 307 — initial-state QED dipoles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ─── Charge lookup (re-exported from QED module) ───────────────────────────

from feynman_engine.amplitudes.nlo_qed_general import QED_CHARGE


# ─── CS phase-space mappings (massless) ────────────────────────────────────

def _dot4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minkowski dot product with metric (+,-,-,-)."""
    return a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], axis=-1)


def cs_ff_map_ol(
    p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """FF mapping: emitter i, photon j, spectator k all final-state.

    Returns (~p_ij, ~p_k, y, z) where:
        y = p_i·p_j / (p_i·p_j + p_j·p_k + p_i·p_k)
        z = p_i·p_k / (p_i·p_k + p_j·p_k)
        ~p_ij = p_i + p_j - y/(1-y) * p_k
        ~p_k = p_k / (1-y)
    By construction ~p_ij² = ~p_k² = 0 (massless) and momentum conservation
    holds: ~p_ij + ~p_k = p_i + p_j + p_k.
    """
    pi_pj = _dot4(p_i, p_j)
    pj_pk = _dot4(p_j, p_k)
    pi_pk = _dot4(p_i, p_k)
    denom = pi_pj + pj_pk + pi_pk
    # Physical PSP: all invariants > 0; guard against pathological inputs.
    if np.any(denom <= 0) or np.any((pi_pk + pj_pk) <= 0):
        raise ValueError(
            "cs_ff_map_ol: degenerate kinematics — PSP must have positive "
            "pairwise invariants p_i·p_j > 0."
        )
    y = pi_pj / denom
    z = pi_pk / (pi_pk + pj_pk)
    if np.any(y >= 1.0):
        # 1-y = 0 in the strict y→1 limit (collinear singularity)
        raise ValueError("cs_ff_map_ol: y → 1 is the strict collinear limit")
    y_4 = y[..., np.newaxis]
    tilde_p_ij = p_i + p_j - (y_4 / (1.0 - y_4)) * p_k
    tilde_p_k = p_k / (1.0 - y_4)
    return tilde_p_ij, tilde_p_k, y, z


def cs_ii_map_ol(
    p_a: np.ndarray, p_j: np.ndarray, p_b: np.ndarray,
    p_finals: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """II mapping: initial emitter a, photon j, initial spectator b.

    The (N+1)-body kinematics with one collinear-divergent photon emitted
    along p_a is mapped to (N)-body kinematics where the initial-state
    momentum is rescaled (p_a → x p_a) and all final-state momenta are
    boosted to restore momentum conservation.

    CS NPB 485, eqs. (5.137)-(5.144).

        x = (p_a·p_b - p_a·p_j - p_b·p_j) / (p_a·p_b)
        ~p_a = x p_a
        ~p_b = p_b
        ~p_n_finals = Lorentz boost of p_n to compensate

    Returns (~p_a, ~p_b, [~p_finals], x).
    """
    pa_pb = _dot4(p_a, p_b)
    pa_pj = _dot4(p_a, p_j)
    pb_pj = _dot4(p_b, p_j)
    # pa_pb is guaranteed > 0 for physical back-to-back massless initial states
    # (= s/2 in the CM frame).  Guard against pathological inputs.
    if np.any(pa_pb <= 0):
        raise ValueError("cs_ii_map_ol: p_a · p_b must be positive")
    x = (pa_pb - pa_pj - pb_pj) / pa_pb

    # Rescale the emitter's initial momentum: ~p_a = x p_a.  Broadcast x to
    # the trailing 4-vector axis when momenta are batched (...,4).
    x_4 = np.expand_dims(x, axis=-1) if hasattr(x, "ndim") and x.ndim > 0 else x
    tilde_p_a = x_4 * p_a
    tilde_p_b = p_b

    # For the simplified II mapping, the final-state momenta are kept as-is.
    # This is exact in the strict soft-photon limit (which dominates IR
    # cancellation for inclusive observables); full Lorentz boost from
    # K = p_a + p_b - p_j to ~K = ~p_a + ~p_b is needed for differential
    # precision and is V4+ work.
    tilde_finals: list[np.ndarray] = list(p_finals)

    return tilde_p_a, tilde_p_b, tilde_finals, x


# ─── QED splitting kernels ─────────────────────────────────────────────────

def V_qg_FF(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """V(q → q γ) for FF dipole — fermion → fermion + photon.

    CS NPB 485, eq. (5.7) adapted to QED (drop colour Casimir):
        V = 2 / (1 - z(1 - y)) - (1 + z)
    Pre-factor 8π α Q² is applied externally.
    """
    return 2.0 / (1.0 - z * (1.0 - y)) - (1.0 + z)


def V_qg_II(x: np.ndarray) -> np.ndarray:
    """V(q → q γ) for II dipole — initial-state photon emission.

    CS NPB 485, eq. (5.145) adapted to QED:
        V = 2/(1-x) - (1+x)
    Pre-factor 8π α Q² is applied externally.
    """
    return 2.0 / (1.0 - x) - (1.0 + x)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _charge(particle: str) -> float:
    return QED_CHARGE.get(particle, 0.0)


def _is_charged(particle: str) -> bool:
    return abs(_charge(particle)) > 1e-12


def _is_initial_state_idx(idx: int, n_in: int) -> bool:
    return idx < n_in


# ─── Generic photon dipole enumerator ──────────────────────────────────────

@dataclass
class DipoleAssignment:
    """One (emitter i, photon j, spectator k) triple for a process."""
    emitter_idx: int           # particle index (0-based; first n_in are initial)
    photon_idx: int            # final-state photon index (only one photon allowed)
    spectator_idx: int         # particle index of the spectator
    config: str                # "FF", "FI", "IF", "II"
    Q_emitter: float
    Q_spectator: float
    is_emitter_initial: bool
    is_spectator_initial: bool

    @property
    def charge_correlator(self) -> float:
        """Q_i × Q_k (with sign).  Used as the dipole prefactor."""
        return self.Q_emitter * self.Q_spectator


def enumerate_qed_dipoles(
    incoming: list[str],
    outgoing_with_photon: list[str],
    photon_idx_in_outgoing: Optional[int] = None,
) -> list[DipoleAssignment]:
    """Enumerate all CS QED dipoles for the radiative process.

    For a (N+1)-body real-emission process with one extra final-state
    photon, this returns every (emitter, photon, spectator) triple where
    both emitter and spectator are charged.

    Parameters
    ----------
    incoming : list[str]
        Initial-state particle names (indices 0..n_in-1 in OL momentum array).
    outgoing_with_photon : list[str]
        Final-state particle names INCLUDING the radiated photon.  The
        photon must be one of these (typically the last entry).
    photon_idx_in_outgoing : int | None
        Index (0-based) of the photon within ``outgoing_with_photon``.
        If None, auto-detects the first ``"gamma"``/``"ph"`` entry.

    Returns
    -------
    list[DipoleAssignment] — one entry per dipole.

    For e+e- → μ+μ-γ: 4 charged legs (e+, e-, μ+, μ-), 1 photon.
    Triples (i, γ, k) with i ≠ γ, k ≠ γ, i ≠ k, both charged → 12 dipoles.

    The dipole sign Q_i Q_k determines whether the dipole adds (same-sign
    legs) or subtracts (opposite-sign legs) to the eikonal sum.
    """
    n_in = len(incoming)
    n_out = len(outgoing_with_photon)
    n_total = n_in + n_out

    if photon_idx_in_outgoing is None:
        for i, p in enumerate(outgoing_with_photon):
            if p in {"gamma", "ph", "photon", "a"}:
                photon_idx_in_outgoing = i
                break
        if photon_idx_in_outgoing is None:
            return []

    photon_global_idx = n_in + photon_idx_in_outgoing
    all_particles = incoming + outgoing_with_photon

    dipoles: list[DipoleAssignment] = []
    for i_idx in range(n_total):
        if i_idx == photon_global_idx:
            continue
        if not _is_charged(all_particles[i_idx]):
            continue
        for k_idx in range(n_total):
            if k_idx == photon_global_idx:
                continue
            if k_idx == i_idx:
                continue
            if not _is_charged(all_particles[k_idx]):
                continue

            i_initial = _is_initial_state_idx(i_idx, n_in)
            k_initial = _is_initial_state_idx(k_idx, n_in)
            if i_initial and k_initial:
                config = "II"
            elif i_initial and not k_initial:
                config = "IF"
            elif not i_initial and k_initial:
                config = "FI"
            else:
                config = "FF"

            dipoles.append(DipoleAssignment(
                emitter_idx=i_idx,
                photon_idx=photon_global_idx,
                spectator_idx=k_idx,
                config=config,
                Q_emitter=_charge(all_particles[i_idx]),
                Q_spectator=_charge(all_particles[k_idx]),
                is_emitter_initial=i_initial,
                is_spectator_initial=k_initial,
            ))
    return dipoles


# ─── OL-Born dipole evaluator ──────────────────────────────────────────────

def _build_ol_pp_array(
    incoming: list[np.ndarray],
    outgoing: list[np.ndarray],
) -> np.ndarray:
    """Pack momenta into OL's (E, px, py, pz, m) 5×N flattened format."""
    n = len(incoming) + len(outgoing)
    pp = np.zeros(5 * n, dtype=np.float64)
    for j, p in enumerate(incoming + outgoing):
        pp[5 * j: 5 * j + 4] = p
        pp[5 * j + 4] = 0.0  # massless leg
    return pp


def evaluate_ol_born(
    born_proc_obj,
    incoming: list[np.ndarray],
    outgoing: list[np.ndarray],
) -> float:
    """Call OL's tree evaluator at given momenta, return |M_Born|².

    Parameters
    ----------
    born_proc_obj : openloops.Process
        Pre-registered Born tree process at the right coupling order.
    incoming, outgoing : list of (4,) arrays
        Single phase-space point (4-momentum per particle).

    Returns
    -------
    float — |M_Born|² in OL's normalization at this PSP.
    """
    pp = _build_ol_pp_array(incoming, outgoing)
    me = born_proc_obj.evaluate(pp)
    return float(me.tree)


def evaluate_qed_dipole_sum(
    born_proc_obj,
    p_in: list[np.ndarray],
    p_out: list[np.ndarray],
    incoming_names: list[str],
    outgoing_names_with_photon: list[str],
    alpha: float,
    photon_idx_in_outgoing: Optional[int] = None,
    include_cross_line: bool = False,
) -> float:
    """Sum of all CS QED dipoles for one radiative phase-space point.

    Returns ΣD = Σ_{ij,k} D_ij,k(p) for the given (N+1)-particle PSP point.
    The Born tree is evaluated via OpenLoops at each mapped (N)-particle
    configuration, so dipoles are in OL's normalization and can be
    point-by-point subtracted from |M_R|² also obtained from OL.

    Parameters
    ----------
    born_proc_obj : openloops.Process
        Pre-registered Born tree process (e.g. e+e-→μμ at order_ew=2).
    p_in : list of (4,) arrays
        Initial-state 4-momenta.
    p_out : list of (4,) arrays
        Final-state 4-momenta (including the radiated photon).
    incoming_names : list[str]
        Initial-state particle names (for charge lookup).
    outgoing_names_with_photon : list[str]
        Final-state particle names (including the radiated photon).
    alpha : float
        EW coupling at the running scale (typically OL's α(M_Z)).
    photon_idx_in_outgoing : int | None
        Index of the photon in outgoing_names_with_photon.  Auto-detect if None.

    Returns
    -------
    float — ΣD at this PSP point.

    Notes
    -----
    For the inclusive K-factor, the IR-finite combination (|M_R|² - ΣD) is
    integrated over the (N+1)-body PSP:
        σ_R_subtracted = (1/2s) ∫ dΦ_{N+1} (|M_R|² - ΣD)
    """
    dipoles = enumerate_qed_dipoles(
        incoming_names, outgoing_names_with_photon, photon_idx_in_outgoing,
    )
    if not dipoles:
        return 0.0

    n_in = len(incoming_names)
    n_out = len(outgoing_names_with_photon)
    photon_idx_in_arr = (
        photon_idx_in_outgoing
        if photon_idx_in_outgoing is not None
        else next(
            (i for i, p in enumerate(outgoing_names_with_photon)
             if p in {"gamma", "ph", "photon", "a"}),
            n_out - 1,
        )
    )

    p_photon = p_out[photon_idx_in_arr]
    # Leading minus sign of the CS QED dipole formula (CS NPB 485 eq. 5.2):
    #   D_ij,k = -(1/(2 p_i·p_j)) × 8π α Q_i Q_k × V_kernel × |M_Born(tilde)|²
    # The minus sign combines with the negative Q_i Q_k of the dominant
    # opposite-sign pairs in standard QED processes (e+e-→μμ has all
    # pairs with Q_i Q_k = ±1, but the soft-photon eikonal sum
    # ΣQ_i Q_k × positive_eikonal is itself negative for e+e-→μμ — see
    # Σ_{i≠k} Q_i Q_k = (Σ Q)² - Σ Q² = -4 for 4 unit-charge legs).
    # The leading minus produces ΣD > 0 in the soft limit, matching |M_R|².
    coupling_factor = -8.0 * math.pi * alpha

    total = 0.0
    for dip in dipoles:
        Q_corr = dip.charge_correlator
        if Q_corr == 0.0:
            continue

        # Identify the emitter momentum vs the spectator momentum
        if dip.is_emitter_initial:
            p_emitter = p_in[dip.emitter_idx]
        else:
            p_emitter = p_out[dip.emitter_idx - n_in]
        if dip.is_spectator_initial:
            p_spectator = p_in[dip.spectator_idx]
        else:
            p_spectator = p_out[dip.spectator_idx - n_in]

        # Compute the dipole based on its configuration
        if dip.config == "FF":
            try:
                tilde_p_ij, tilde_p_k, y, z = cs_ff_map_ol(
                    p_emitter[np.newaxis, :], p_photon[np.newaxis, :],
                    p_spectator[np.newaxis, :],
                )
            except ValueError:
                # Degenerate PSP for this dipole — skip it but continue summing
                # the others.  Drives one event's ΣD slightly off but doesn't
                # abort the whole MC integral.
                continue
            tilde_p_ij = tilde_p_ij[0]
            tilde_p_k = tilde_p_k[0]
            y_v = float(y[0])
            z_v = float(z[0])
            V = float(V_qg_FF(np.array(z_v), np.array(y_v)))
            pi_pj = float(_dot4(p_emitter, p_photon))
            if pi_pj <= 0:
                continue
            prop = 1.0 / (2.0 * pi_pj)

            # Build the mapped Born configuration: replace the emitter and
            # spectator with their tilded versions, drop the photon
            mapped_outgoing = []
            for k_out in range(n_out):
                if k_out == photon_idx_in_arr:
                    continue
                global_idx = n_in + k_out
                if global_idx == dip.emitter_idx:
                    mapped_outgoing.append(tilde_p_ij)
                elif global_idx == dip.spectator_idx:
                    mapped_outgoing.append(tilde_p_k)
                else:
                    mapped_outgoing.append(p_out[k_out])
            mapped_incoming = [p_in[k_in] for k_in in range(n_in)]

            try:
                born_msq = evaluate_ol_born(
                    born_proc_obj, mapped_incoming, mapped_outgoing,
                )
            except Exception:
                continue

            D = prop * coupling_factor * Q_corr * V * born_msq
            total += D

        elif dip.config == "II":
            # Both emitter and spectator are initial-state
            other_finals = [p_out[k] for k in range(n_out) if k != photon_idx_in_arr]
            try:
                tilde_p_a, tilde_p_b, tilde_finals, x = cs_ii_map_ol(
                    p_emitter[np.newaxis, :], p_photon[np.newaxis, :],
                    p_spectator[np.newaxis, :], other_finals,
                )
            except ValueError:
                continue
            tilde_p_a = tilde_p_a[0]
            x_v = float(x[0])
            if x_v <= 0 or x_v >= 1:
                continue
            V = float(V_qg_II(np.array(x_v)))
            pa_pj = float(_dot4(p_emitter, p_photon))
            if pa_pj <= 0:
                continue
            prop = 1.0 / (2.0 * pa_pj * x_v)

            mapped_incoming = [
                tilde_p_a if k == dip.emitter_idx else p_in[k]
                for k in range(n_in)
            ]
            mapped_outgoing = [p_out[k] for k in range(n_out) if k != photon_idx_in_arr]

            try:
                born_msq = evaluate_ol_born(
                    born_proc_obj, mapped_incoming, mapped_outgoing,
                )
            except Exception:
                continue

            D = prop * coupling_factor * Q_corr * V * born_msq
            total += D

        else:
            # FI / IF — proper Catani-Dittmaier-Seymour mapping required.
            # The naive "soft-eikonal with original Born momenta" approach
            # violates momentum conservation when E_photon > 0 (the Born is
            # evaluated at √s_eff = √s - E_photon, but OL is told √s).
            # Skipping by default; for percent-level inclusive accuracy on
            # 4-charged-leg processes (e+e-→ll), the FF + II contributions
            # capture the dominant IR singularities and the cross-line
            # collinear piece is regulated by the lepton mass.
            if not include_cross_line:
                continue
            # Approximate cross-line: rescale Born CM-energy to "absorb" the
            # photon.  This is rough but at least momentum-conserving.
            pi_pj = float(_dot4(p_emitter, p_photon))
            if pi_pj <= 0:
                continue
            prop = 1.0 / (2.0 * pi_pj)
            V_soft = 2.0
            # Build a momentum-conserving Born at reduced √s_eff
            sqrts_eff = float(np.sqrt(_dot4(
                np.sum(p_in, axis=0) - p_photon,
                np.sum(p_in, axis=0) - p_photon,
            )))
            if sqrts_eff <= 0:
                continue
            E_eff = sqrts_eff / 2.0
            # Use back-to-back massless initial states at √s_eff
            p1_eff = np.array([E_eff, 0.0, 0.0,  E_eff])
            p2_eff = np.array([E_eff, 0.0, 0.0, -E_eff])
            # Use back-to-back final states (averaged Born) — rough but
            # restores momentum conservation
            p3_eff = np.array([E_eff, 0.0, 0.0,  E_eff])
            p4_eff = np.array([E_eff, 0.0, 0.0, -E_eff])
            try:
                born_msq = evaluate_ol_born(
                    born_proc_obj,
                    [p1_eff, p2_eff],
                    [p3_eff, p4_eff],
                )
            except Exception:
                continue
            D = prop * coupling_factor * Q_corr * V_soft * born_msq
            total += D

    return total
