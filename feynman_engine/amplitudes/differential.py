"""Differential cross-section histograms.

Computes dσ/dX as a binned histogram for any 2→N scattering process the
engine has an amplitude for.  Handles two sampling modes:

- **2→2 deterministic**: scipy.quad over each cos θ bin (or its Jacobian
  for pT/η/y observables) for percent-level accuracy.
- **2→N Monte Carlo**: per-event RAMBO sampling with histogram fills,
  reusing the same |M̄|² / dipole-subtracted weights as the integrated
  cross-section path.

NLO modes:

- ``order='LO'``: bare Born |M̄|².
- ``order='NLO-running'``: every bin scaled by the (running α)^n K factor
  evaluated at √ŝ — the same approximation used by ``nlo_cross_section``
  for inclusive σ.  Cheap, but does not reshape distributions.
- ``order='NLO-subtracted'`` (currently only ``e+ e- -> mu+ mu-``):
  combines Born histogram + the per-event subtracted-real weights from
  ``nlo_cross_section_subtracted_eemumu`` + the analytic V+I correction
  in each bin.  Use with ``min_invariant_mass`` cuts that match the
  experimental fiducial volume.

The result schema matches what the frontend histogram view expects:
``bin_edges``, ``bin_centers``, ``bin_widths``, ``dsigma_dX_pb``,
``dsigma_dX_uncertainty_pb`` (per-bin Gaussian, MC stat error or 0 for
deterministic), ``sigma_total_pb``, ``unit``.

Supported observables (validated against analytic 2→2 / textbook):

    cos_theta      —  2→2, scattering angle (unit: dimensionless)
    pT_lepton      —  leading lepton transverse momentum (GeV)
    pT_photon      —  leading photon transverse momentum (GeV)
    eta_lepton     —  leading lepton pseudorapidity
    y_lepton       —  leading lepton rapidity
    M_inv          —  total invariant mass of final-state system (GeV)
    M_ll           —  dilepton invariant mass (GeV) — for processes with two leptons
    DR_ll          —  ΔR = √((Δη)² + (Δφ)²) between two leptons

Hadronic ``pp → F`` is supported by routing each partonic channel through
this module and summing channel histograms with parton-luminosity weights.

References:
    Catani, Seymour, NPB 485 (1997) — dipole subtraction
    Ellis, Stirling, Webber, "QCD and Collider Physics", Ch. 3 — kinematics
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import sympy as sp

from feynman_engine.amplitudes.cross_section import (
    ALPHA_EM, ALPHA_S, GEV2_TO_PB,
    _build_coupling_defaults,
    _get_particle_masses,
    _msq_to_callable,
    _phase_space_factor,
    _validate_cross_section_scope,
    differential_cross_section,
    total_cross_section,
    total_cross_section_mc,
)
from feynman_engine.amplitudes.phase_space import (
    compute_dot_products, compute_invariants, dot4, rambo_massless,
)


# ---------------------------------------------------------------------------
# Per-event observable extraction
# ---------------------------------------------------------------------------

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.abs(b) > 1e-30, a / np.where(np.abs(b) > 1e-30, b, 1.0), 0.0)


def _pT(p: np.ndarray) -> np.ndarray:
    """Transverse momentum: √(px² + py²)."""
    return np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)


def _eta(p: np.ndarray) -> np.ndarray:
    """Pseudorapidity η = -ln tan(θ/2)."""
    pT = _pT(p)
    pz = p[..., 3]
    p3 = np.sqrt(pT ** 2 + pz ** 2)
    # η = 0.5 ln((|p| + pz)/(|p| - pz))
    num = np.maximum(p3 + pz, 1e-30)
    den = np.maximum(p3 - pz, 1e-30)
    return 0.5 * np.log(num / den)


def _y(p: np.ndarray) -> np.ndarray:
    """Rapidity y = 0.5 ln((E+pz)/(E-pz))."""
    E = p[..., 0]
    pz = p[..., 3]
    num = np.maximum(E + pz, 1e-30)
    den = np.maximum(E - pz, 1e-30)
    return 0.5 * np.log(num / den)


def _phi(p: np.ndarray) -> np.ndarray:
    """Azimuthal angle in (-π, π]."""
    return np.arctan2(p[..., 2], p[..., 1])


def _M_inv(p_sum: np.ndarray) -> np.ndarray:
    """Lorentz-invariant mass √((Σp)²)."""
    sq = dot4(p_sum, p_sum)
    return np.sqrt(np.maximum(sq, 0.0))


# Per-particle classification used to find the leading lepton/photon
_LEPTON_TOKENS = {"e+", "e-", "mu+", "mu-", "tau+", "tau-"}
_PHOTON_TOKENS = {"gamma"}


def _leading_index(outgoing: list[str], targets: set[str]) -> Optional[int]:
    """Index of the first outgoing particle in ``targets``, or None."""
    for i, name in enumerate(outgoing):
        if name in targets:
            return i
    return None


# ---------------------------------------------------------------------------
# Observable kernels for the MC path
# ---------------------------------------------------------------------------
# Each kernel takes (p1, p2, momenta, outgoing_names) → array of shape (n_events,)
# representing the observable value at each phase-space point.

def _obs_pT_lepton(p1, p2, momenta, outgoing):
    idx = _leading_index(outgoing, _LEPTON_TOKENS)
    if idx is None:
        raise ValueError("pT_lepton: no lepton in final state")
    return _pT(momenta[:, idx, :])


def _obs_pT_photon(p1, p2, momenta, outgoing):
    idx = _leading_index(outgoing, _PHOTON_TOKENS)
    if idx is None:
        raise ValueError("pT_photon: no photon in final state")
    return _pT(momenta[:, idx, :])


def _obs_eta_lepton(p1, p2, momenta, outgoing):
    idx = _leading_index(outgoing, _LEPTON_TOKENS)
    if idx is None:
        raise ValueError("eta_lepton: no lepton in final state")
    return _eta(momenta[:, idx, :])


def _obs_y_lepton(p1, p2, momenta, outgoing):
    idx = _leading_index(outgoing, _LEPTON_TOKENS)
    if idx is None:
        raise ValueError("y_lepton: no lepton in final state")
    return _y(momenta[:, idx, :])


def _obs_M_inv(p1, p2, momenta, outgoing):
    """Total invariant mass of the final state."""
    p_sum = np.sum(momenta, axis=1)
    return _M_inv(p_sum)


def _obs_M_ll(p1, p2, momenta, outgoing):
    """Dilepton invariant mass — sum the two leptons in the final state."""
    lep_indices = [i for i, n in enumerate(outgoing) if n in _LEPTON_TOKENS]
    if len(lep_indices) < 2:
        raise ValueError(f"M_ll: needs ≥2 leptons in final state, found {len(lep_indices)}")
    p_sum = momenta[:, lep_indices[0], :] + momenta[:, lep_indices[1], :]
    return _M_inv(p_sum)


def _obs_DR_ll(p1, p2, momenta, outgoing):
    """ΔR = √((Δη)² + (Δφ)²) between the two leading leptons."""
    lep_indices = [i for i, n in enumerate(outgoing) if n in _LEPTON_TOKENS]
    if len(lep_indices) < 2:
        raise ValueError(f"DR_ll: needs ≥2 leptons, found {len(lep_indices)}")
    p_a = momenta[:, lep_indices[0], :]
    p_b = momenta[:, lep_indices[1], :]
    deta = _eta(p_a) - _eta(p_b)
    dphi = _phi(p_a) - _phi(p_b)
    # Wrap φ to (-π, π]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(deta ** 2 + dphi ** 2)


_MC_OBSERVABLES: dict[str, tuple[Callable, str]] = {
    "pT_lepton":  (_obs_pT_lepton,  "GeV"),
    "pT_photon":  (_obs_pT_photon,  "GeV"),
    "eta_lepton": (_obs_eta_lepton, "dimensionless"),
    "y_lepton":   (_obs_y_lepton,   "dimensionless"),
    "M_inv":      (_obs_M_inv,      "GeV"),
    "M_ll":       (_obs_M_ll,       "GeV"),
    "DR_ll":      (_obs_DR_ll,      "dimensionless"),
}


# ---------------------------------------------------------------------------
# 2→2 deterministic histograms
# ---------------------------------------------------------------------------

def _histogram_2to2_costheta(
    process: str,
    theory: str,
    sqrt_s: float,
    bin_edges: np.ndarray,
    coupling_vals: Optional[dict] = None,
) -> dict:
    """dσ/d(cosθ) histogram for 2→2 via scipy.quad per bin.

    Returns a dict matching the public schema.  Each bin is integrated
    deterministically; the per-bin uncertainty is the quad estimate.
    """
    from scipy.integrate import quad

    s_val = sqrt_s ** 2
    result, error = _validate_cross_section_scope(process, theory)
    if error is not None:
        return {**error, "supported": False}
    msq_expr = result.msq

    defaults = _build_coupling_defaults(theory)
    if coupling_vals:
        defaults.update(coupling_vals)
    try:
        masses = _get_particle_masses(process, theory)
    except Exception:
        masses = (0.0, 0.0, 0.0, 0.0)

    s_sym = sp.symbols("s", real=True)
    cos_sym = sp.symbols("cos_theta", real=True)
    f_msq = _msq_to_callable(msq_expr, s_sym, cos_sym, defaults, masses=masses)
    ps_factor = _phase_space_factor(s_val, masses)

    def integrand(ct):
        try:
            v = float(f_msq(s_val, ct))
            if not math.isfinite(v):
                return 0.0
            return max(v, 0.0) * ps_factor
        except Exception:
            return 0.0

    bin_edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(bin_edges) - 1
    sigma_per_bin_pb = np.zeros(n_bins)
    err_per_bin_pb = np.zeros(n_bins)

    # Identical-particle symmetry factor (matches total_cross_section).
    from feynman_engine.amplitudes.cross_section import (
        _identical_particle_symmetry_factor, _get_outgoing_particles,
    )
    sym_factor = _identical_particle_symmetry_factor(
        _get_outgoing_particles(process, theory)
    )

    eps = 1e-3
    for i in range(n_bins):
        lo = max(bin_edges[i], -1.0 + eps)
        hi = min(bin_edges[i + 1], 1.0 - eps)
        if hi <= lo:
            continue
        try:
            sig, err = quad(integrand, lo, hi, limit=200, epsabs=1e-14, epsrel=1e-9)
            sigma_per_bin_pb[i] = max(sig, 0.0) * GEV2_TO_PB / sym_factor
            err_per_bin_pb[i] = abs(err) * GEV2_TO_PB / sym_factor
        except Exception:
            pass

    widths = np.diff(bin_edges)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dsigma_dX = np.where(widths > 0, sigma_per_bin_pb / widths, 0.0)
    dsigma_dX_err = np.where(widths > 0, err_per_bin_pb / widths, 0.0)

    return {
        "process": process,
        "theory": theory.upper(),
        "sqrt_s_gev": sqrt_s,
        "observable": "cos_theta",
        "unit": "dimensionless",
        "bin_edges": bin_edges.tolist(),
        "bin_centers": centers.tolist(),
        "bin_widths": widths.tolist(),
        "sigma_per_bin_pb": sigma_per_bin_pb.tolist(),
        "dsigma_dX_pb": dsigma_dX.tolist(),
        "dsigma_dX_uncertainty_pb": dsigma_dX_err.tolist(),
        "sigma_total_pb": float(np.sum(sigma_per_bin_pb)),
        "identical_particle_factor": sym_factor,
        "method": "deterministic-quad-2to2",
        "order": "LO",
        "supported": True,
    }


# ---------------------------------------------------------------------------
# 2→N MC histograms
# ---------------------------------------------------------------------------

def _build_msq_evaluator(msq_expr, theory: str, coupling_vals: Optional[dict]):
    """Build an msq_func(p1, p2, momenta) numerical evaluator from a symbolic |M|²."""
    defaults = _build_coupling_defaults(theory)
    if coupling_vals:
        defaults.update(coupling_vals)

    subs_map = {sym: defaults[sym.name]
                for sym in msq_expr.free_symbols if sym.name in defaults}
    msq_substituted = msq_expr.subs(subs_map)

    remaining = msq_substituted.free_symbols
    if not remaining:
        msq_const = float(msq_substituted)

        def f(p1, p2, momenta):
            return np.full(momenta.shape[0], msq_const)
        return f

    sym_names = sorted(s.name for s in remaining)
    sym_list = [sp.symbols(n, real=True) for n in sym_names]
    f_msq = sp.lambdify(sym_list, msq_substituted, modules="numpy")

    def f(p1, p2, momenta):
        invs = compute_dot_products(p1, p2, momenta)
        invs.update(compute_invariants(p1, p2, momenta))
        for key in list(invs):
            bare = key.replace("_", "")
            if bare != key:
                invs[bare] = invs[key]
        args = []
        for name in sym_names:
            if name not in invs:
                raise ValueError(
                    f"Symbol '{name}' in |M|² not found in kinematic invariants."
                )
            args.append(invs[name])
        vals = f_msq(*args)
        return np.maximum(vals, 0.0)

    return f


def _histogram_2toN_mc(
    process: str,
    theory: str,
    sqrt_s: float,
    observable: str,
    bin_edges: np.ndarray,
    n_events: int = 100_000,
    seed: int = 42,
    min_invariant_mass: float = 1.0,
    coupling_vals: Optional[dict] = None,
    nlo_subtracted_weights: Optional[np.ndarray] = None,
    nlo_v_plus_i_pb: float = 0.0,
) -> dict:
    """MC histogram of dσ/dX for 2→N.

    Standard mode: per-event |M̄|² × RAMBO weight / (2s) / n_events filled
    into the bin determined by the observable evaluator.

    NLO subtraction mode (when ``nlo_subtracted_weights`` is supplied):
    fills the histogram with the precomputed per-event weights from the
    Catani-Seymour subtraction routine, then adds the V+I analytic σ_total
    proportionally to the Born distribution (so the total integral matches
    the inclusive σ_NLO).
    """
    from feynman_engine.physics.amplitude import get_amplitude
    from feynman_engine.physics.translator import parse_process

    # Accept observable name case-insensitively (M_ll == m_ll, pT_lepton == pt_lepton, …)
    obs_canon = next(
        (k for k in _MC_OBSERVABLES if k.lower() == observable.lower()),
        observable,
    )
    if obs_canon not in _MC_OBSERVABLES:
        return {
            "supported": False,
            "error": (
                f"Observable '{observable}' not implemented for 2→N. "
                f"Available: {sorted(_MC_OBSERVABLES)}."
            ),
        }
    obs_fn, unit = _MC_OBSERVABLES[obs_canon]

    try:
        spec = parse_process(process.strip(), theory.upper())
    except ValueError as exc:
        return {"supported": False, "error": str(exc)}

    if len(spec.incoming) != 2:
        return {"supported": False, "error": "Need 2 incoming particles."}
    n_out = len(spec.outgoing)
    if n_out < 2:
        return {"supported": False, "error": "Need ≥2 final-state particles."}

    result = get_amplitude(process.strip(), theory.upper())
    if result is None or result.msq is None:
        return {"supported": False, "error": f"No |M̄|² for '{process}' in {theory}."}

    msq_func = _build_msq_evaluator(result.msq, theory, coupling_vals)

    rng = np.random.default_rng(seed)
    momenta, weights = rambo_massless(n_out, sqrt_s, n_events, rng)

    E = sqrt_s / 2.0
    p1 = np.array([E, 0, 0, E])
    p2 = np.array([E, 0, 0, -E])

    # IR cuts on pairwise invariants (only relevant for 2→N≥3 with massless final state).
    if min_invariant_mass > 0 and n_out >= 3:
        sij_min = min_invariant_mass ** 2
        mask = np.ones(n_events, dtype=bool)
        for i in range(n_out):
            for j in range(i + 1, n_out):
                qi, qj = momenta[:, i, :], momenta[:, j, :]
                mask &= dot4(qi + qj, qi + qj) >= sij_min
        # Also t-channel cuts vs incoming
        for i in range(n_out):
            qi = momenta[:, i, :]
            t1 = dot4(p1[np.newaxis, :] - qi, p1[np.newaxis, :] - qi)
            t2 = dot4(p2[np.newaxis, :] - qi, p2[np.newaxis, :] - qi)
            mask &= np.abs(t1) >= sij_min
            mask &= np.abs(t2) >= sij_min
        cut_weights = np.where(mask, weights, 0.0)
    else:
        cut_weights = weights

    s_val = sqrt_s ** 2

    # Identical-particle symmetry factor (P&S convention) so the integral
    # of the histogram matches the physical σ from total_cross_section_mc.
    from feynman_engine.amplitudes.cross_section import (
        _identical_particle_symmetry_factor,
    )
    sym_factor = _identical_particle_symmetry_factor(list(spec.outgoing))

    if nlo_subtracted_weights is None:
        msq_vals = msq_func(p1, p2, momenta)
        # σ-equivalent per-event weight in pb such that Σ w = σ_pb
        per_event_pb = (
            msq_vals * cut_weights / (2.0 * s_val) / n_events
            * GEV2_TO_PB / sym_factor
        )
    else:
        # Subtraction provides finished per-event σ-equivalent weights in pb.
        # We still apply the IR mask defensively, and divide by the symmetry
        # factor (the subtracted weights come from the *real-emission* phase
        # space — same convention treatment).
        if min_invariant_mass > 0 and n_out >= 3:
            mask_factor = np.where(cut_weights > 0, 1.0, 0.0)
        else:
            mask_factor = np.ones(n_events)
        per_event_pb = nlo_subtracted_weights * mask_factor / sym_factor

    # Compute observable per event
    try:
        obs_vals = obs_fn(p1, p2, momenta, list(spec.outgoing))
    except ValueError as exc:
        return {"supported": False, "error": str(exc)}

    bin_edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(bin_edges) - 1

    # Histogram fill (numpy.histogram is fastest; we want per-bin σ and σ²)
    sigma_per_bin_pb, _ = np.histogram(obs_vals, bins=bin_edges, weights=per_event_pb)
    # Per-bin variance: Σ w_i² in each bin (since Σw = σ, σ² ≈ Σw² for large N)
    weights_sq = per_event_pb ** 2
    var_per_bin, _ = np.histogram(obs_vals, bins=bin_edges, weights=weights_sq)
    err_per_bin_pb = np.sqrt(np.maximum(var_per_bin, 0.0))

    # Add analytic V+I uniformly weighted to the Born shape (NLO mode only)
    if nlo_subtracted_weights is not None and nlo_v_plus_i_pb != 0.0:
        # Distribute V+I according to the Born shape (which equals the
        # current histogram up to the small subtracted contribution).
        born_shape = sigma_per_bin_pb.copy()
        s_total = float(np.sum(born_shape))
        if abs(s_total) > 1e-30:
            sigma_per_bin_pb = sigma_per_bin_pb + (nlo_v_plus_i_pb * born_shape / s_total)

    widths = np.diff(bin_edges)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dsigma_dX = np.where(widths > 0, sigma_per_bin_pb / widths, 0.0)
    dsigma_dX_err = np.where(widths > 0, err_per_bin_pb / widths, 0.0)

    return {
        "process": process,
        "theory": theory.upper(),
        "sqrt_s_gev": sqrt_s,
        "observable": observable,
        "unit": unit,
        "bin_edges": bin_edges.tolist(),
        "bin_centers": centers.tolist(),
        "bin_widths": widths.tolist(),
        "sigma_per_bin_pb": sigma_per_bin_pb.tolist(),
        "dsigma_dX_pb": dsigma_dX.tolist(),
        "dsigma_dX_uncertainty_pb": dsigma_dX_err.tolist(),
        "sigma_total_pb": float(np.sum(sigma_per_bin_pb)),
        "method": (
            "monte-carlo-rambo-with-cs-subtraction"
            if nlo_subtracted_weights is not None else "monte-carlo-rambo"
        ),
        "order": "NLO" if nlo_subtracted_weights is not None else "LO",
        "n_events": n_events,
        "n_bins": n_bins,
        "min_invariant_mass_gev": min_invariant_mass,
        "identical_particle_factor": sym_factor,
        "supported": True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def differential_distribution(
    process: str,
    theory: str,
    sqrt_s: float,
    observable: str,
    bin_edges: list[float] | np.ndarray,
    *,
    order: str = "LO",
    n_events: int = 100_000,
    seed: int = 42,
    min_invariant_mass: float = 1.0,
    coupling_vals: Optional[dict] = None,
) -> dict:
    """Compute dσ/dX as a histogram in pb / unit.

    Dispatches by observable + final-state multiplicity:
    - ``observable='cos_theta'`` for 2→2 → deterministic scipy.quad per bin.
    - All other observables → Monte Carlo with per-event histogram fill.

    Parameters
    ----------
    process : str
        Scattering process, e.g. ``"e+ e- -> mu+ mu- gamma"``.
    theory : str
        Theory name (QED/QCD/QCDQED/EW/BSM).
    sqrt_s : float
        Centre-of-mass energy (GeV).
    observable : str
        One of ``cos_theta``, ``pT_lepton``, ``pT_photon``, ``eta_lepton``,
        ``y_lepton``, ``M_inv``, ``M_ll``, ``DR_ll``.
    bin_edges : sequence of float
        Histogram bin edges (length N+1 for N bins).  Must be monotone increasing.
    order : str
        ``LO`` (Born), ``NLO-running`` (rescale every bin by running-coupling K),
        ``NLO-subtracted`` (e+ e- -> mu+ mu- only — full subtraction histogram).
    n_events : int
        MC sample count for 2→N processes (ignored for 2→2 cos_theta).
    seed : int
        Random seed.
    min_invariant_mass : float
        IR safety cut on pairwise invariants for 2→N (GeV).
    coupling_vals : dict, optional
        Coupling overrides keyed by symbol name.

    Returns
    -------
    dict with bin_edges, bin_centers, bin_widths, dsigma_dX_pb,
    dsigma_dX_uncertainty_pb, sigma_total_pb, method, order, supported.
    """
    bin_edges = np.asarray(bin_edges, dtype=float)
    if bin_edges.ndim != 1 or len(bin_edges) < 2:
        return {"supported": False, "error": "bin_edges must have at least 2 entries."}
    if not np.all(np.diff(bin_edges) > 0):
        return {"supported": False, "error": "bin_edges must be strictly increasing."}

    # Special path: 2→2 cos_theta → deterministic
    if observable == "cos_theta":
        result = _histogram_2to2_costheta(
            process, theory, sqrt_s, bin_edges, coupling_vals,
        )
        if not result.get("supported", False):
            return result
        if order.upper() == "NLO-RUNNING":
            return _apply_running_kfactor(result, process, theory, sqrt_s)
        return result

    # NLO-subtracted special path (only e+ e- -> mu+ mu- supported today)
    if order.upper() == "NLO-SUBTRACTED":
        return _differential_nlo_subtracted(
            process, theory, sqrt_s, observable, bin_edges,
            n_events=n_events, seed=seed,
            min_invariant_mass=min_invariant_mass,
        )

    # Default: MC histogram of the Born |M|²
    result = _histogram_2toN_mc(
        process, theory, sqrt_s, observable, bin_edges,
        n_events=n_events, seed=seed,
        min_invariant_mass=min_invariant_mass,
        coupling_vals=coupling_vals,
    )
    if not result.get("supported", False):
        return result

    if order.upper() == "NLO-RUNNING":
        return _apply_running_kfactor(result, process, theory, sqrt_s)
    return result


# ---------------------------------------------------------------------------
# NLO helpers
# ---------------------------------------------------------------------------

def _apply_running_kfactor(
    result: dict, process: str, theory: str, sqrt_s: float,
) -> dict:
    """Rescale every bin by the running-coupling K-factor at Q² = s."""
    from feynman_engine.amplitudes.nlo_cross_section import nlo_cross_section
    nlo = nlo_cross_section(process, theory, sqrt_s)
    if not nlo.get("supported", False):
        return {**result, "order": "LO", "nlo_warning": nlo.get("error", "no NLO available")}
    k = nlo.get("k_factor", 1.0)
    out = dict(result)
    out["dsigma_dX_pb"] = [v * k for v in result["dsigma_dX_pb"]]
    out["dsigma_dX_uncertainty_pb"] = [v * abs(k) for v in result["dsigma_dX_uncertainty_pb"]]
    out["sigma_per_bin_pb"] = [v * k for v in result["sigma_per_bin_pb"]]
    out["sigma_total_pb"] = result["sigma_total_pb"] * k
    out["k_factor"] = k
    out["order"] = "NLO"
    out["method"] = result["method"] + "+running-K"
    return out


def _differential_nlo_subtracted(
    process: str, theory: str, sqrt_s: float, observable: str,
    bin_edges: np.ndarray, n_events: int, seed: int, min_invariant_mass: float,
) -> dict:
    """NLO-subtracted differential for e+ e- -> mu+ mu- (Born+real) + V+I."""
    proc_clean = process.strip().lower().replace(" ", "")
    if proc_clean != "e+e-->mu+mu-":
        return {
            "supported": False,
            "error": (
                f"NLO-subtracted differential is currently only available for "
                f"'e+ e- -> mu+ mu-' (real-emission process e+ e- -> mu+ mu- gamma). "
                f"Got '{process}'."
            ),
        }

    from feynman_engine.amplitudes.nlo_cross_section import (
        _eemumu_subtracted_real_weights,
        nlo_cross_section_subtracted_eemumu,
    )

    # Born histogram from 2→2 (cos_theta if requested) or via MC of the 2→2 |M|²
    if observable == "cos_theta":
        born_hist = _histogram_2to2_costheta(
            "e+ e- -> mu+ mu-", "QED", sqrt_s, bin_edges,
        )
    else:
        # We need to histogram an OBSERVABLE that depends on the 2→2 final state.
        # For 2→2, just compute observable analytically per bin via MC over cos θ
        # (cheap, deterministic-style sampling of the 2→2 phase space).
        born_hist = _histogram_2toN_mc(
            "e+ e- -> mu+ mu-", "QED", sqrt_s, observable, bin_edges,
            n_events=n_events, seed=seed,
            min_invariant_mass=0.0,  # 2→2 has no IR singularity
        )
    if not born_hist.get("supported", False):
        return born_hist

    # Subtracted-real per-event weights for e+ e- -> mu+ mu- gamma
    momenta, p1, p2, weight_pb = _eemumu_subtracted_real_weights(
        sqrt_s=sqrt_s, n_events=n_events, seed=seed,
        min_invariant_mass=min_invariant_mass,
    )

    # Histogram those weights against the requested observable computed on
    # the 2→3 final state.  For 2→3, the outgoing list is [μ+, μ-, γ].
    real_outgoing = ["mu+", "mu-", "gamma"]
    if observable not in _MC_OBSERVABLES and observable != "cos_theta":
        return {"supported": False, "error": f"Observable '{observable}' unsupported."}

    if observable == "cos_theta":
        # Define cos θ of the leading muon w.r.t. beam axis as a proxy.
        idx_mu = 0  # μ+
        p_mu = momenta[:, idx_mu, :]
        cos_theta = p_mu[:, 3] / np.maximum(np.sqrt(np.sum(p_mu[:, 1:] ** 2, axis=1)), 1e-30)
        obs_vals = cos_theta
    else:
        obs_fn, _ = _MC_OBSERVABLES[observable]
        obs_vals = obs_fn(p1, p2, momenta, real_outgoing)

    sigma_per_bin_real, _ = np.histogram(obs_vals, bins=bin_edges, weights=weight_pb)
    weights_sq = weight_pb ** 2
    var_per_bin, _ = np.histogram(obs_vals, bins=bin_edges, weights=weights_sq)
    err_per_bin_real = np.sqrt(np.maximum(var_per_bin, 0.0))

    # Combined per-bin: Born + Real + V+I (V+I distributed proportionally to Born)
    sigma_born = np.array(born_hist["sigma_per_bin_pb"])
    err_born = np.array(born_hist["dsigma_dX_uncertainty_pb"]) * np.array(born_hist["bin_widths"])

    sigma_vi_total_pb = (3.0 * ALPHA_EM / (4.0 * math.pi)) * float(np.sum(sigma_born))
    born_total = float(np.sum(sigma_born))
    if born_total > 0:
        sigma_vi_per_bin = sigma_vi_total_pb * sigma_born / born_total
    else:
        sigma_vi_per_bin = np.zeros_like(sigma_born)

    sigma_total_per_bin = sigma_born + sigma_per_bin_real + sigma_vi_per_bin
    err_total_per_bin = np.sqrt(err_born ** 2 + err_per_bin_real ** 2)

    widths = np.diff(bin_edges)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dsigma_dX = np.where(widths > 0, sigma_total_per_bin / widths, 0.0)
    dsigma_dX_err = np.where(widths > 0, err_total_per_bin / widths, 0.0)

    unit = "dimensionless" if observable == "cos_theta" else _MC_OBSERVABLES[observable][1]

    return {
        "process": process,
        "theory": theory.upper(),
        "sqrt_s_gev": sqrt_s,
        "observable": observable,
        "unit": unit,
        "bin_edges": bin_edges.tolist(),
        "bin_centers": centers.tolist(),
        "bin_widths": widths.tolist(),
        "sigma_per_bin_pb": sigma_total_per_bin.tolist(),
        "sigma_per_bin_born_pb": sigma_born.tolist(),
        "sigma_per_bin_real_subtracted_pb": sigma_per_bin_real.tolist(),
        "sigma_per_bin_virtual_pb": sigma_vi_per_bin.tolist(),
        "dsigma_dX_pb": dsigma_dX.tolist(),
        "dsigma_dX_uncertainty_pb": dsigma_dX_err.tolist(),
        "sigma_total_pb": float(np.sum(sigma_total_per_bin)),
        "sigma_total_born_pb": born_total,
        "sigma_total_virtual_pb": sigma_vi_total_pb,
        "sigma_total_real_subtracted_pb": float(np.sum(sigma_per_bin_real)),
        "method": "born + cs-subtraction + analytic-V+I",
        "order": "NLO-subtracted",
        "n_events": n_events,
        "min_invariant_mass_gev": min_invariant_mass,
        "limitations": (
            "Subtraction uses partial (same-line) dipoles only; cross-line FI "
            "dipoles not yet included.  Use min_invariant_mass to keep integrand finite."
        ),
        "supported": True,
    }


# ---------------------------------------------------------------------------
# Hadronic differential (pp → F)
# ---------------------------------------------------------------------------

def hadronic_differential_distribution(
    process: str,
    sqrt_s: float,
    observable: str,
    bin_edges: list[float] | np.ndarray,
    *,
    theory: str | None = None,
    pdf_name: str = "auto",
    pdf_member: int = 0,
    mu_f: Optional[float] = None,
    n_events: int = 30_000,
    seed: int = 42,
    min_invariant_mass: float = 1.0,
    min_partonic_cm: Optional[float] = None,
    n_tau_grid: int = 12,
) -> dict:
    """Hadronic dσ/dX for pp → F by enumerating parton channels.

    For each ordered (a, b) parton pair where a partonic amplitude exists,
    samples a τ = ŝ/s grid, generates 2→N MC at each √ŝ, fills the
    observable histogram weighted by the parton luminosity dL_{ab}/dτ.

    Channels with no partonic |M̄|² are silently skipped (same contract
    as the integrated hadronic cross-section).  The histogram is the sum
    of channel histograms.
    """
    from feynman_engine.amplitudes.hadronic import (
        _PARTON_PDG, _active_partons, _detect_partonic_theory,
        _resolve_partonic_threshold, _final_state_total_mass,
    )
    from feynman_engine.amplitudes.pdf import get_pdf, parton_luminosity, LHAPDFSet

    process_clean = process.strip()
    if not (process_clean.lower().startswith("p p ->")
            or process_clean.lower().startswith("p p->")):
        return {"supported": False, "error": f"Expected 'p p -> X', got '{process}'."}
    final_state = process_clean.split("->", 1)[1].strip()
    if not final_state:
        return {"supported": False, "error": "Empty final state."}

    try:
        pdf = get_pdf(pdf_name, member=pdf_member)
    except (ImportError, RuntimeError) as exc:
        return {"supported": False, "error": f"PDF '{pdf_name}' unavailable: {exc}"}

    theory_used = (theory or _detect_partonic_theory(final_state)).upper()
    threshold, threshold_reason = _resolve_partonic_threshold(
        final_state, theory_used, min_partonic_cm,
    )
    s_pp = sqrt_s ** 2
    if threshold ** 2 >= s_pp:
        return {
            "supported": False,
            "error": (
                f"√s = {sqrt_s} GeV below partonic √ŝ cut ({threshold:.1f} GeV, "
                f"source: {threshold_reason})"
            ),
        }

    if mu_f is None:
        mu_f = max(threshold, 5.0)
    mu_f_sq = mu_f ** 2

    bin_edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(bin_edges) - 1
    if n_bins < 1:
        return {"supported": False, "error": "Need ≥1 bin."}
    sigma_per_bin_pb = np.zeros(n_bins)
    err_sq_per_bin_pb = np.zeros(n_bins)

    # τ grid (from threshold² / s_pp up to some τ_max < 1).
    tau_min = max(threshold ** 2 / s_pp, 1e-10)
    tau_max = 0.99
    if tau_min >= tau_max:
        return {"supported": False, "error": "No partonic phase space."}
    tau_grid = np.geomspace(tau_min, tau_max, n_tau_grid)
    # Trapezoidal integration weights in log τ space (dτ ≈ τ d ln τ)
    log_tau = np.log(tau_grid)
    d_log_tau = np.zeros_like(tau_grid)
    if n_tau_grid >= 2:
        d_log_tau[1:-1] = 0.5 * (log_tau[2:] - log_tau[:-2])
        d_log_tau[0] = 0.5 * (log_tau[1] - log_tau[0])
        d_log_tau[-1] = 0.5 * (log_tau[-1] - log_tau[-2])
    else:
        d_log_tau[0] = 1.0

    partons = _active_partons(mu_f_sq)
    theory_candidates: list[str] = []
    seen = set()
    for t in (theory_used, "QCD", "QCDQED", "EW", "QED"):
        if t and t not in seen:
            seen.add(t)
            theory_candidates.append(t)

    channels_evaluated = 0
    channels_examined = 0

    for a in partons:
        for b in partons:
            channels_examined += 1
            channel_sigma_per_bin = np.zeros(n_bins)
            channel_var_per_bin = np.zeros(n_bins)
            theory_used_ch = None

            for tau, dlog in zip(tau_grid, d_log_tau):
                sqrt_s_hat = math.sqrt(tau * s_pp)
                # Pick the first theory that has the partonic amplitude
                hist_at_tau = None
                for t_try in theory_candidates:
                    h = _histogram_2toN_mc(
                        f"{a} {b} -> {final_state}", t_try, sqrt_s_hat,
                        observable, bin_edges,
                        n_events=n_events, seed=seed,
                        min_invariant_mass=min_invariant_mass,
                    )
                    if h.get("supported", False):
                        hist_at_tau = h
                        theory_used_ch = t_try
                        break
                if hist_at_tau is None:
                    continue

                L = parton_luminosity(pdf, _PARTON_PDG[a], _PARTON_PDG[b], tau, mu_f_sq)
                # Channel contribution: L · dσ̂/dX × τ dlnτ (since dτ = τ dlnτ)
                #   σ_pp = ∫dτ L(τ) σ̂(τ s)  →  in log τ: ∫dlnτ τ L σ̂
                contrib = (
                    np.array(hist_at_tau["sigma_per_bin_pb"])
                    * L * tau * dlog
                )
                err_contrib = (
                    np.array(hist_at_tau["dsigma_dX_uncertainty_pb"])
                    * np.array(hist_at_tau["bin_widths"])
                    * L * tau * dlog
                )
                channel_sigma_per_bin += contrib
                channel_var_per_bin += err_contrib ** 2

            if np.sum(channel_sigma_per_bin) > 0:
                channels_evaluated += 1
                sigma_per_bin_pb += channel_sigma_per_bin
                err_sq_per_bin_pb += channel_var_per_bin

    if channels_evaluated == 0:
        return {
            "supported": False,
            "error": (
                f"No partonic channel had an amplitude for 'p p -> {final_state}' "
                f"after examining {channels_examined} ordered (a,b) pairs."
            ),
        }

    err_per_bin_pb = np.sqrt(err_sq_per_bin_pb)
    widths = np.diff(bin_edges)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dsigma_dX = np.where(widths > 0, sigma_per_bin_pb / widths, 0.0)
    dsigma_dX_err = np.where(widths > 0, err_per_bin_pb / widths, 0.0)

    pdf_label = f"{pdf.backend}:{pdf.name}"
    if isinstance(pdf, LHAPDFSet):
        pdf_label += f"/m{pdf.member}"

    # Pick the unit string from the observable
    if observable == "cos_theta":
        unit = "dimensionless"
    elif observable in _MC_OBSERVABLES:
        unit = _MC_OBSERVABLES[observable][1]
    else:
        unit = ""

    return {
        "process": process_clean,
        "hadronic": True,
        "theory": theory_used,
        "sqrt_s_gev": sqrt_s,
        "observable": observable,
        "unit": unit,
        "pdf": pdf_label,
        "mu_f_gev": mu_f,
        "min_partonic_sqrts_gev": threshold,
        "partonic_cut_reason": threshold_reason,
        "bin_edges": bin_edges.tolist(),
        "bin_centers": centers.tolist(),
        "bin_widths": widths.tolist(),
        "sigma_per_bin_pb": sigma_per_bin_pb.tolist(),
        "dsigma_dX_pb": dsigma_dX.tolist(),
        "dsigma_dX_uncertainty_pb": dsigma_dX_err.tolist(),
        "sigma_total_pb": float(np.sum(sigma_per_bin_pb)),
        "method": "hadronic-parton-enumeration",
        "order": "LO",
        "n_channels_evaluated": channels_evaluated,
        "n_channels_examined": channels_examined,
        "n_events": n_events,
        "n_tau_grid": n_tau_grid,
        "supported": True,
    }
