"""Generic N-body decay-width integrator (N ≥ 4) via Monte Carlo.

Complements ``three_body_dalitz.py`` (1→3 deterministic Dalitz) for processes
with more than three final-state particles where the Dalitz parametrisation
is no longer adequate.  Uses RAMBO uniform-phase-space sampling and
multiplies by a user-supplied |M̄|².

Covers physics applications:
- H → 4ℓ (golden Higgs channel via Z*Z*)
- H → ℓℓγγ (resonance + photon emission)
- t → b ℓ ν γ (radiative top decay)
- B → π π ℓ ν (semileptonic with multi-meson final state)

Public API
----------
- ``n_body_partial_width(M_parent, masses, msq_callback, n_events=...)``
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from feynman_engine.amplitudes.phase_space import rambo_massless, rambo_massive


_HBAR_GEV_S = 6.582119569e-25


def n_body_partial_width(
    M_parent: float,
    masses: Sequence[float],
    msq_callback: Callable[[np.ndarray], np.ndarray],
    *,
    n_events: int = 50_000,
    seed: int = 42,
) -> dict:
    """Γ(P → 1+2+...+N) by RAMBO Monte Carlo over N-body phase space.

    Γ = 1/(2 M) × ⟨|M̄|²(p₁,…,p_N) × w_RAMBO⟩

    where the average is over n_events uniformly-distributed phase-space
    points and w_RAMBO is the per-event RAMBO weight.

    Parameters
    ----------
    M_parent : float
        Mass of the decaying parent in GeV.
    masses : sequence of float
        Final-state masses in GeV.  Length N ≥ 2.
    msq_callback : callable
        ``msq(momenta) → array of |M̄|² values`` where ``momenta`` has
        shape ``(n_events, N, 4)``.  Spin-averaged for the parent,
        spin-summed for the daughters.
    n_events : int
        RAMBO MC sample count.
    seed : int
        Random seed.

    Returns
    -------
    dict with ``Gamma_gev`` (partial width), ``Gamma_uncertainty_gev``,
    ``tau_seconds`` (if Gamma > 0), ``n_events``, ``method='rambo-mc'``.
    """
    N = len(masses)
    if N < 2:
        return {"supported": False, "error": "Need at least 2 daughters."}
    sum_m = sum(masses)
    if M_parent <= sum_m:
        return {
            "supported": False,
            "error": (
                f"Kinematic boundary: M_parent={M_parent} ≤ Σ m_i={sum_m}; "
                "decay is energetically forbidden."
            ),
        }

    rng = np.random.default_rng(seed)
    # Use massive RAMBO if any daughter has mass
    if any(m > 0.0 for m in masses):
        momenta, weights = rambo_massive(
            N, M_parent, list(masses), n_events, rng,
        )
    else:
        momenta, weights = rambo_massless(N, M_parent, n_events, rng)

    # Per-event |M̄|² × RAMBO weight
    msq_vals = np.asarray(msq_callback(momenta), dtype=float)
    per_event = msq_vals * weights
    integral_mean = float(np.mean(per_event))
    integral_std = float(np.std(per_event, ddof=1) / math.sqrt(n_events))

    # Γ = 1/(2 M) × integral
    gamma_gev = integral_mean / (2.0 * M_parent)
    gamma_err = integral_std / (2.0 * M_parent)

    return {
        "Gamma_gev":              gamma_gev,
        "Gamma_uncertainty_gev":  gamma_err,
        "tau_seconds":            (_HBAR_GEV_S / gamma_gev) if gamma_gev > 0 else float("inf"),
        "M_parent_gev":           M_parent,
        "daughter_masses_gev":    list(masses),
        "n_events":               n_events,
        "n_daughters":            N,
        "method":                 "rambo-mc",
        "supported":              True,
    }


# ─── Convenience: H → 4ℓ (Higgs golden channel via Z*Z* → 4ℓ) ─────────────
#
# For a quick H → 4ℓ test, we use the narrow-width approximation:
#   Γ(H → ZZ* → 4ℓ) ≈ Γ(H→ZZ*) × BR(Z→ℓℓ)² × (combinatoric factor)
#
# This is not a full Dalitz-plot integration — that requires the full
# Z propagator structure.  We expose it as a curated convenience.

_M_H = 125.25
_M_Z = 91.1876
_GAMMA_Z = 2.4952
_GAMMA_H = 0.00407           # PDG H total width
_BR_Z_TO_LL_PER_FLAVOR = 0.03366    # PDG BR(Z → ℓ+ℓ-)


def higgs_to_4l_BR(flavor_a: str = "e", flavor_b: str = "mu") -> dict:
    """Estimate BR(H → ZZ* → ℓ_a ℓ_a ℓ_b ℓ_b) via narrow-width approximation.

    For flavor_a = flavor_b = "e" or "mu" you get BR(H → 4e) or BR(H → 4μ).
    Mixed flavors give the BR(H → 2e2μ) "golden" channel.
    PDG 2024:  BR(H → 4ℓ, any combination ≥ 2e + 2μ + …) ≈ 2.7e-4

    Returns the BR plus the partial Γ; does NOT do a full 4-body Dalitz
    integration of the Z propagators (deferred — needs OL).
    """
    BR_HZZ = 0.0262   # PDG BR(H → ZZ*)
    BR_Z_ll = _BR_Z_TO_LL_PER_FLAVOR
    # Combinatoric: identical-flavor 4ℓ has indistinguishable permutations
    same_flavor = (flavor_a == flavor_b)
    combo = 1.0 if same_flavor else 2.0  # 2 for distinct flavors (Z1↔Z2)
    BR_h_4l = BR_HZZ * BR_Z_ll * BR_Z_ll * combo
    return {
        "process":    f"H -> Z Z* -> {flavor_a}+ {flavor_a}- {flavor_b}+ {flavor_b}-",
        "BR":         BR_h_4l,
        "Gamma_gev":  BR_h_4l * _GAMMA_H,
        "method":     "narrow-width-approximation",
        "reference":  "PDG 2024: BR(H→ZZ*)=2.62%, BR(Z→ℓℓ)=3.366%",
        "supported":  True,
    }
