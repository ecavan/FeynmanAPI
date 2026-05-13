"""Pure-Python anti-kT jet clustering for low-multiplicity final states.

Reference: Cacciari, Salam, Soyez, JHEP 04 (2008) 063 (arXiv:0802.1189).

This implementation is suitable for NLO Monte Carlo with up to ~10 final-
state partons.  For higher multiplicities (parton showering, NNLO+jets),
use the proper FastJet library.

Anti-kT distance measure:
    d_ij = min(p_T,i⁻², p_T,j⁻²) · ΔR_ij² / R²
    d_iB = p_T,i⁻²
where ΔR_ij² = (Δη_ij)² + (Δφ_ij)².  Recursion: find min{d_ij, d_iB}; if
d_ij wins, merge i and j into a pseudojet by summing 4-momenta; if d_iB
wins, declare i a final jet and remove from the list.

Public API
----------
- ``anti_kT(four_momenta, R=0.4)`` → list of jet 4-momenta (sorted by pT desc.)
- ``pseudo_rapidity(p)``, ``rapidity(p)``, ``phi(p)``, ``pT(p)`` — helpers
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def pT(p: np.ndarray) -> float:
    """Transverse momentum of a 4-vector (E, px, py, pz)."""
    return math.sqrt(p[1] * p[1] + p[2] * p[2])


def pseudo_rapidity(p: np.ndarray) -> float:
    """η = (1/2) log((|p| + p_z) / (|p| − p_z)).  Use rapidity for masses."""
    p_abs = math.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2)
    if p_abs <= abs(p[3]) + 1e-30:
        return math.copysign(50.0, p[3])     # very forward — cap
    return 0.5 * math.log((p_abs + p[3]) / (p_abs - p[3]))


def rapidity(p: np.ndarray) -> float:
    """y = (1/2) log((E + p_z) / (E − p_z))."""
    if p[0] <= abs(p[3]) + 1e-30:
        return math.copysign(50.0, p[3])
    return 0.5 * math.log((p[0] + p[3]) / (p[0] - p[3]))


def phi(p: np.ndarray) -> float:
    """Azimuthal angle, principal branch in (-π, π]."""
    return math.atan2(p[2], p[1])


def _delta_R_sq(p1: np.ndarray, p2: np.ndarray) -> float:
    """ΔR² = (Δy)² + (Δφ)², using rapidity (handles massive particles)."""
    dy = rapidity(p1) - rapidity(p2)
    dphi = phi(p1) - phi(p2)
    # Wrap dphi to (-π, π]
    if dphi > math.pi:
        dphi -= 2.0 * math.pi
    elif dphi <= -math.pi:
        dphi += 2.0 * math.pi
    return dy * dy + dphi * dphi


def anti_kT(four_momenta: Sequence[np.ndarray], R: float = 0.4) -> list[np.ndarray]:
    """Cluster a list of 4-momenta with the anti-kT algorithm.

    Parameters
    ----------
    four_momenta : list of (4,) ndarrays
        Input particles in (E, px, py, pz) Minkowski convention.
    R : float
        Jet radius (typical LHC values: 0.4 for ATLAS, 0.5 for CMS Run 1,
        0.4 for CMS Run 2).

    Returns
    -------
    list of (4,) ndarrays
        Jet 4-momenta, sorted by descending pT.
    """
    if R <= 0.0:
        raise ValueError(f"Jet radius R must be positive (got {R}).")

    particles: list[np.ndarray] = [np.asarray(p, dtype=float).copy() for p in four_momenta]
    jets: list[np.ndarray] = []
    R2_inv = 1.0 / (R * R)

    while particles:
        n = len(particles)
        # Compute all distances
        min_d = float("inf")
        min_pair = None    # (i, j) or (i, None) for beam
        for i in range(n):
            pTi_sq = particles[i][1] ** 2 + particles[i][2] ** 2
            if pTi_sq < 1e-30:
                # Beam direction — kT⁻² blows up; treat as beam distance 0
                # so it gets removed first.
                d_iB = 0.0
            else:
                d_iB = 1.0 / pTi_sq
            if d_iB < min_d:
                min_d = d_iB
                min_pair = (i, None)
            for j in range(i + 1, n):
                pTj_sq = particles[j][1] ** 2 + particles[j][2] ** 2
                if pTj_sq < 1e-30:
                    continue
                inv_pTi_sq = 1.0 / max(pTi_sq, 1e-30)
                inv_pTj_sq = 1.0 / pTj_sq
                min_inv_pT = min(inv_pTi_sq, inv_pTj_sq)
                d_ij = min_inv_pT * _delta_R_sq(particles[i], particles[j]) * R2_inv
                if d_ij < min_d:
                    min_d = d_ij
                    min_pair = (i, j)
        if min_pair is None:
            break
        i, j = min_pair
        if j is None:
            # d_iB minimum — promote particle i to a jet
            jets.append(particles.pop(i))
        else:
            # d_ij minimum — merge i and j (j > i so pop j first)
            merged = particles[i] + particles[j]
            # Remove higher index first
            del particles[j]
            del particles[i]
            particles.append(merged)

    jets.sort(key=lambda p: -pT(p))
    return jets


def jets_passing_cuts(
    four_momenta: Sequence[np.ndarray],
    R: float = 0.4,
    pT_min: float = 25.0,
    eta_max: float = 4.5,
) -> list[np.ndarray]:
    """Cluster with anti-kT and keep jets passing pT and |η| cuts.

    Default cuts: pT > 25 GeV, |η| < 4.5 (ATLAS Run 2 inclusive jets).
    """
    return [
        j for j in anti_kT(four_momenta, R)
        if pT(j) > pT_min and abs(pseudo_rapidity(j)) < eta_max
    ]
