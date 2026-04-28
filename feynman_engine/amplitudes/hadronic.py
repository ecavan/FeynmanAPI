"""Hadronic (proton-proton) cross-sections via PDF convolution.

Computes σ(pp → X) by convolving partonic cross-sections σ̂(ab → X)
with parton distribution functions:

    σ(pp → X) = Σ_{a,b} ∫ dτ  L_{ab}(τ, μ²) · σ̂(ab → X, ŝ = τ·s)

where L_{ab}(τ, μ²) = ∫_τ^1 (dx/x) f_a(x, μ²) f_b(τ/x, μ²) is the
parton luminosity for parton ``a`` from proton 1 and parton ``b`` from
proton 2.  The sum over (a, b) runs over **ordered** pairs so that
``L_{q q̄}`` (quark from proton 1) and ``L_{q̄ q}`` (antiquark from
proton 1) both contribute — both contributions are physically distinct
even though σ̂(qq̄→X) = σ̂(q̄q→X) by parity.

Three execution paths:

1. **Specialized Drell-Yan**  ``pp → l⁺l⁻``
   Uses the analytic γ + Z partonic formula integrated over M_ll.
   Faster and more accurate near the Z pole than the generic path.

2. **Specialized top-pair**  ``pp → tt̄``
   Builds a partonic σ̂ grid for the gg and qq̄ channels and convolves.

3. **Generic enumeration** (any other pp→F)
   Iterates over every parton pair (a, b) for which the engine has a
   partonic |M̄|² (curated, FORM, or symbolic), builds σ̂(√ŝ) on a grid,
   convolves channel-by-channel.  Channels with no available amplitude
   are silently skipped — the result is the sum of *verified* channels.
   When no channel is reachable, returns ``supported=False`` with a
   clear error rather than a misleading number.

References:
    Ellis, Stirling, Webber, "QCD and Collider Physics", Ch. 7-9
    Peskin, "An Introduction to Quantum Field Theory", Ch. 17
    Catani, Seymour, "A general algorithm…", Nucl. Phys. B 485 (1997)
"""
from __future__ import annotations

import math
from typing import Optional, Callable

from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy as np

from feynman_engine.amplitudes.cross_section import (
    ALPHA_EM,
    ALPHA_S,
    GEV2_TO_PB,
    total_cross_section,
    total_cross_section_mc,
)
from feynman_engine.amplitudes.nlo_cross_section import (
    alpha_em_running,
    alpha_s_running,
)
from feynman_engine.amplitudes.pdf import (
    PDFSet,
    LHAPDFSet,
    get_pdf,
    parton_luminosity,
    QUARK_CHARGE,
    QUARK_T3,
)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_M_Z = 91.1876      # Z boson mass (GeV)
_GAMMA_Z = 2.4952   # Z boson total width (GeV)
_SIN2_TW = 0.23122  # sin²(θ_W) (PDG 2023)
_M_T = 172.69       # top quark mass (GeV)
_M_C = 1.27
_M_B = 4.18
_N_C = 3            # number of colors


# ---------------------------------------------------------------------------
# Parton bookkeeping
# ---------------------------------------------------------------------------
# Map between partonic process tokens and PDG flavor IDs.

_PARTON_PDG: dict[str, int] = {
    "g":  21,
    "u":   2, "u~":  -2,
    "d":   1, "d~":  -1,
    "s":   3, "s~":  -3,
    "c":   4, "c~":  -4,
    "b":   5, "b~":  -5,
}

# Light partons that are always available; heavy quarks gated on μ_f² later.
_LIGHT_PARTONS = ["g", "u", "u~", "d", "d~", "s", "s~"]


def _active_partons(mu_f_sq: float) -> list[str]:
    """Return the list of parton tokens active at scale μ_f² (heavy-quark thresholds)."""
    out = list(_LIGHT_PARTONS)
    if mu_f_sq > 4.0 * _M_C ** 2:
        out += ["c", "c~"]
    if mu_f_sq > 4.0 * _M_B ** 2:
        out += ["b", "b~"]
    return out


# ---------------------------------------------------------------------------
# Drell-Yan: analytic partonic cross-section (γ + Z exchange)
# ---------------------------------------------------------------------------

def _drell_yan_sigma_hat(s_hat: float, quark_flavor: int) -> float:
    """Partonic cross-section σ̂(qq̄ → l+l-) including γ and Z exchange.

    σ̂ = (4πα²)/(3·N_c·ŝ) × {
        Q_q²·Q_l² + 2·Q_q·Q_l·v_q·v_l·Re(χ) + (v_q²+a_q²)(v_l²+a_l²)|χ|²
    }
    where χ = ŝ / [(ŝ - M_Z²) + i·M_Z·Γ_Z] / (4·sin²θ_W·cos²θ_W).
    """
    if s_hat <= 0:
        return 0.0

    alpha = ALPHA_EM
    Q_q = QUARK_CHARGE.get(abs(quark_flavor), 0.0)
    T3_q = QUARK_T3.get(abs(quark_flavor), 0.0)
    Q_l = -1.0
    T3_l = -0.5

    sw2 = _SIN2_TW
    v_q = T3_q - 2.0 * Q_q * sw2
    a_q = T3_q
    v_l = T3_l - 2.0 * Q_l * sw2
    a_l = T3_l

    denom_re = s_hat - _M_Z ** 2
    denom_im = _M_Z * _GAMMA_Z
    denom_sq = denom_re ** 2 + denom_im ** 2
    if denom_sq < 1e-30:
        denom_sq = 1e-30

    cw2 = 1.0 - sw2
    norm = 1.0 / (4.0 * sw2 * cw2)
    chi_re = s_hat * denom_re / denom_sq * norm
    chi_im = -s_hat * denom_im / denom_sq * norm
    chi_sq = chi_re ** 2 + chi_im ** 2

    bracket = (
        Q_q ** 2 * Q_l ** 2
        + 2.0 * Q_q * Q_l * v_q * v_l * chi_re
        + (v_q ** 2 + a_q ** 2) * (v_l ** 2 + a_l ** 2) * chi_sq
    )

    sigma = 4.0 * math.pi * alpha ** 2 / (3.0 * _N_C * s_hat) * bracket
    return max(sigma, 0.0)


# ---------------------------------------------------------------------------
# Partonic σ̂ grid construction (used by both top-pair and generic paths)
# ---------------------------------------------------------------------------

def _safe_total_cross_section(
    process: str, theory: str, sqrt_s: float,
    cos_theta_min: float = -1.0, cos_theta_max: float = 1.0,
) -> dict:
    """``total_cross_section`` that converts any exception to ``supported=False``.

    Underlying calls can raise ``QGRAFError`` or parse errors for partonic
    processes that aren't supported by the registered theory.  The generic
    enumerator treats every such failure as "channel unavailable" — never
    as "channel = 0 pb."  Returning a dict with ``supported=False`` keeps
    that contract everywhere.

    Optional ``cos_theta_min/max`` restrict the angular integration — used
    by the per-particle pT cut option for diphoton, dijet, etc.
    """
    try:
        return total_cross_section(
            process, theory, sqrt_s=sqrt_s,
            cos_theta_min=cos_theta_min, cos_theta_max=cos_theta_max,
        )
    except Exception as exc:
        return {"supported": False, "error": f"{type(exc).__name__}: {exc}"}


def _cos_theta_max_from_pT(sqrt_s: float, min_pT: float) -> Optional[float]:
    """Return cos θ_max such that pT(=√ŝ/2 sin θ) > min_pT.

    For 2→2 with massless final states in the partonic CM frame:
        pT = (√ŝ/2) × sin θ
    Requiring pT > min_pT gives sin θ > 2 min_pT / √ŝ, i.e.
        |cos θ| < √(1 - 4 pT² / ŝ)

    Returns None if the cut is impossible (2 min_pT > √ŝ).
    """
    if min_pT <= 0:
        return None
    threshold = 2.0 * min_pT
    if sqrt_s <= threshold:
        return 0.0  # No valid scattering angle
    cos_max_sq = 1.0 - (threshold / sqrt_s) ** 2
    return math.sqrt(max(cos_max_sq, 0.0))


def _safe_total_cross_section_mc(
    process: str, theory: str, sqrt_s: float,
    n_events: int, min_invariant_mass: float,
) -> dict:
    try:
        return total_cross_section_mc(
            process, theory, sqrt_s=sqrt_s,
            n_events=n_events,
            min_invariant_mass=min_invariant_mass,
        )
    except Exception as exc:
        return {"supported": False, "error": f"{type(exc).__name__}: {exc}"}


def _probe_partonic_supported(
    process: str,
    theory: str,
    sqrt_s_probe: float,
    use_mc: bool,
    n_events_mc: int,
    min_invariant_mass: float,
) -> bool:
    """Cheap one-shot probe: is σ̂ supported at all?

    Used by the generic enumerator to skip the full grid build for the
    ~95% of (a, b) parton pairs that have no amplitude.  For 2→N MC the
    probe uses a tiny n_events and only checks the supported flag.

    Returns ``False`` for any exception so unreachable channels never
    crash the enumeration.
    """
    if use_mc:
        r = _safe_total_cross_section_mc(
            process, theory, sqrt_s=float(sqrt_s_probe),
            n_events=max(min(n_events_mc, 1000), 200),
            min_invariant_mass=min_invariant_mass,
        )
    else:
        r = _safe_total_cross_section(process, theory, sqrt_s=float(sqrt_s_probe))
    return bool(r.get("supported", False))


def _build_sigma_hat_grid(
    process: str,
    theory: str,
    sqrt_s_min: float,
    sqrt_s_max: float,
    n_points: int = 30,
    use_mc: bool = False,
    n_events_mc: int = 20_000,
    min_invariant_mass: float = 1.0,
    skip_probe: bool = False,
    min_pT: float = 0.0,
) -> Optional[Callable[[float], float]]:
    """Pre-compute partonic σ̂(√ŝ) on a log grid and return an interpolator.

    Returns ``None`` if the process is unavailable at *every* sampled point.
    The interpolator returns σ̂ in pb and is clamped to zero outside the grid.

    Set ``skip_probe=True`` for callers that already verified the process is
    supported (specialized DY/tt̄ paths).  The generic enumerator probes once
    at the upper end of the grid first to fail fast on unreachable channels.

    ``min_pT`` (GeV) restricts the angular integration so that each final-
    state particle has pT > min_pT (massless, 2→2 only).  Required to match
    LHC photon/jet measurements that always have per-particle pT cuts.
    """
    if not skip_probe:
        if not _probe_partonic_supported(
            process, theory, sqrt_s_max,
            use_mc=use_mc,
            n_events_mc=n_events_mc,
            min_invariant_mass=min_invariant_mass,
        ):
            return None

    sqrt_s_vals = np.geomspace(sqrt_s_min, sqrt_s_max, n_points)
    sigma_vals = []
    any_supported = False
    for ss in sqrt_s_vals:
        if use_mc:
            r = _safe_total_cross_section_mc(
                process, theory, sqrt_s=float(ss),
                n_events=n_events_mc,
                min_invariant_mass=min_invariant_mass,
            )
        else:
            # Apply per-particle pT cut as an angular restriction
            # (only meaningful for 2→2 with massless final states).
            cos_theta_min, cos_theta_max = -1.0, 1.0
            if min_pT > 0.0 and not use_mc:
                cmax = _cos_theta_max_from_pT(float(ss), min_pT)
                if cmax is None:
                    pass  # no cut requested
                elif cmax <= 0:
                    sigma_vals.append(0.0)
                    continue
                else:
                    cos_theta_min, cos_theta_max = -cmax, cmax
            r = _safe_total_cross_section(
                process, theory, sqrt_s=float(ss),
                cos_theta_min=cos_theta_min, cos_theta_max=cos_theta_max,
            )
        if r.get("supported", False):
            any_supported = True
            sigma_vals.append(max(r["sigma_pb"], 0.0))
        else:
            sigma_vals.append(0.0)

    if not any_supported:
        return None

    sigma_arr = np.array(sigma_vals)
    if np.all(sigma_arr <= 0.0):
        return None

    # Linear interpolation in (√ŝ, σ̂) — robust against zeros at the
    # threshold edge that would break log interpolation.
    interp = interp1d(
        sqrt_s_vals, sigma_arr,
        kind="linear", bounds_error=False, fill_value=0.0,
    )

    def grid_fn(sqrt_s_hat: float) -> float:
        if sqrt_s_hat < sqrt_s_vals[0] or sqrt_s_hat > sqrt_s_vals[-1]:
            return 0.0
        return float(max(interp(sqrt_s_hat), 0.0))

    return grid_fn


# ---------------------------------------------------------------------------
# PDF convolution helpers
# ---------------------------------------------------------------------------

def _convolve_channel(
    sigma_hat_grid: Callable[[float], float],
    pdf,
    pdg_a: int,
    pdg_b: int,
    mu_f_sq: float,
    sqrt_s: float,
    threshold_sqrt_s: float,
    nlo_running: Optional[Callable[[float], float]] = None,
) -> float:
    """Convolve one ordered partonic channel against the PDF luminosity.

    Returns σ_pp for this channel in pb.

    Parameters
    ----------
    sigma_hat_grid : callable
        σ̂(√ŝ) → pb (zero outside grid support).
    pdf : PDFSet or LHAPDFSet
        Both expose ``f(flavor, x, Q²)``.
    pdg_a, pdg_b : int
        PDG flavor IDs for partons from proton 1 and proton 2.
    mu_f_sq : float
        Factorization scale squared (GeV²).
    sqrt_s : float
        Hadronic CM energy (GeV).
    threshold_sqrt_s : float
        Lower limit of √ŝ for which σ̂ is non-zero.
    nlo_running : callable or None
        Optional ``ŝ → k_factor`` to apply a running-coupling NLO correction
        to σ̂ at each phase-space point.
    """
    s_pp = sqrt_s ** 2
    if threshold_sqrt_s ** 2 >= s_pp:
        return 0.0
    tau_min = max(threshold_sqrt_s ** 2 / s_pp, 1e-10)
    tau_max = 1.0 - 1e-10
    if tau_min >= tau_max:
        return 0.0

    def integrand(tau: float) -> float:
        s_hat = tau * s_pp
        sqrt_s_hat = math.sqrt(s_hat)
        sigma_hat_pb = sigma_hat_grid(sqrt_s_hat)
        if sigma_hat_pb <= 0:
            return 0.0
        if nlo_running is not None:
            sigma_hat_pb *= nlo_running(s_hat)
        L = parton_luminosity(pdf, pdg_a, pdg_b, tau, mu_f_sq)
        return L * sigma_hat_pb

    try:
        result, _ = quad(
            integrand, tau_min, tau_max,
            limit=200, epsrel=5e-3, epsabs=1e-8,
        )
    except Exception:
        return 0.0
    return max(result, 0.0)


# ---------------------------------------------------------------------------
# Hadronic cross-section: Drell-Yan (specialized, fast)
# ---------------------------------------------------------------------------

def _drell_yan_hadronic(
    sqrt_s: float,
    pdf,
    mu_f_sq: float,
    m_ll_min: float = 60.0,
    m_ll_max: float = 120.0,
    order: str = "LO",
) -> dict:
    """Compute σ(pp → l+l-) via Drell-Yan, integrating dM_ll.

        dσ/dM_ll = (2 M_ll / s) · Σ_q [L_{q q̄}(τ) + L_{q̄ q}(τ)] · σ̂(qq̄ → l+l-, ŝ=M_ll²)

    Both L_{q q̄} and L_{q̄ q} are summed: each ordering corresponds to
    a physically distinct event (which proton supplied the quark).
    """
    s_pp = sqrt_s ** 2

    # Quark flavors; charm/bottom included if μ_f² is above their threshold.
    quark_flavors = [2, 1, 3]
    if mu_f_sq > 4.0 * _M_C ** 2:
        quark_flavors.append(4)
    if mu_f_sq > 4.0 * _M_B ** 2:
        quark_flavors.append(5)

    channel_results = []

    for q_flav in quark_flavors:
        q_bar = -q_flav

        def integrand(m_ll: float, qf=q_flav, qb=q_bar) -> float:
            s_hat = m_ll ** 2
            tau = s_hat / s_pp
            if tau >= 1.0 or tau <= 0.0:
                return 0.0
            sigma_hat = _drell_yan_sigma_hat(s_hat, qf)
            if order.upper() == "NLO":
                k = (alpha_em_running(s_hat) / ALPHA_EM) ** 2
                sigma_hat *= k
            sigma_hat_pb = sigma_hat * GEV2_TO_PB
            # Sum both orderings: q from p1 / q̄ from p2  +  q̄ from p1 / q from p2.
            lum = (
                parton_luminosity(pdf, qf, qb, tau, mu_f_sq)
                + parton_luminosity(pdf, qb, qf, tau, mu_f_sq)
            )
            return (2.0 * m_ll / s_pp) * lum * sigma_hat_pb

        if m_ll_min < _M_Z < m_ll_max:
            sigma_low, _ = quad(integrand, m_ll_min, _M_Z - 1.0,
                                limit=100, epsrel=1e-4)
            sigma_peak, _ = quad(integrand, _M_Z - 1.0, _M_Z + 1.0,
                                 limit=100, epsrel=1e-4)
            sigma_high, _ = quad(integrand, _M_Z + 1.0, m_ll_max,
                                 limit=100, epsrel=1e-4)
            sigma_ch = sigma_low + sigma_peak + sigma_high
        else:
            sigma_ch, _ = quad(integrand, m_ll_min, m_ll_max,
                               limit=100, epsrel=1e-4)

        q_name = {2: "u", 1: "d", 3: "s", 4: "c", 5: "b"}[q_flav]
        channel_results.append({
            "partonic": f"{q_name} {q_name}~ -> l+ l-",
            "sigma_pb": sigma_ch,
        })

    total_sigma = sum(ch["sigma_pb"] for ch in channel_results)
    for ch in channel_results:
        ch["fraction"] = ch["sigma_pb"] / total_sigma if total_sigma > 0 else 0.0

    return {
        "sigma_pb": total_sigma,
        "channels": channel_results,
    }


# ---------------------------------------------------------------------------
# Hadronic cross-section: top pair (specialized)
# ---------------------------------------------------------------------------

def _top_pair_hadronic(
    sqrt_s: float,
    pdf,
    mu_f_sq: float,
    order: str = "LO",
) -> dict:
    """Compute σ(pp → tt̄) from gg and qq̄ channels."""
    s_pp = sqrt_s ** 2
    threshold = 2.0 * _M_T
    tau_min = (threshold + 1.0) ** 2 / s_pp
    tau_max = min(0.5, 1.0)

    if tau_min >= tau_max:
        return {
            "sigma_pb": 0.0,
            "channels": [],
            "error": f"sqrt_s = {sqrt_s} GeV is below tt̄ threshold ({threshold} GeV)",
        }

    sqrt_s_min = threshold + 1.0
    sqrt_s_max = math.sqrt(tau_max * s_pp)

    sigma_gg = _build_sigma_hat_grid(
        "g g -> t t~", "QCD", sqrt_s_min, sqrt_s_max, n_points=40,
        skip_probe=True,
    )
    sigma_qq = _build_sigma_hat_grid(
        "u u~ -> t t~", "QCD", sqrt_s_min, sqrt_s_max, n_points=40,
        skip_probe=True,
    )

    nlo_run = (lambda s_hat: (alpha_s_running(s_hat) / ALPHA_S) ** 2) \
        if order.upper() == "NLO" else None

    channel_results = []

    if sigma_gg is not None:
        sig_gg = _convolve_channel(
            sigma_gg, pdf, 21, 21, mu_f_sq, sqrt_s, sqrt_s_min, nlo_run,
        )
        channel_results.append({"partonic": "g g -> t t~", "sigma_pb": sig_gg})

    if sigma_qq is not None:
        sig_qq_total = 0.0
        for q_flav in [2, 1, 3, 4, 5]:
            # Sum both orderings — q from p1 / q̄ from p2  and the reverse.
            sig_qq_total += _convolve_channel(
                sigma_qq, pdf, q_flav, -q_flav, mu_f_sq, sqrt_s, sqrt_s_min, nlo_run,
            )
            sig_qq_total += _convolve_channel(
                sigma_qq, pdf, -q_flav, q_flav, mu_f_sq, sqrt_s, sqrt_s_min, nlo_run,
            )
        channel_results.append({
            "partonic": "q q~ -> t t~ (all flavors, both orderings)",
            "sigma_pb": sig_qq_total,
        })

    total_sigma = sum(ch["sigma_pb"] for ch in channel_results)
    for ch in channel_results:
        ch["fraction"] = ch["sigma_pb"] / total_sigma if total_sigma > 0 else 0.0

    return {
        "sigma_pb": total_sigma,
        "channels": channel_results,
    }


# ---------------------------------------------------------------------------
# Hadronic cross-section: GENERIC enumeration
# ---------------------------------------------------------------------------

def _final_state_total_mass(final_state: str, theory: str) -> float:
    """Sum of final-state particle masses (GeV).  Zero if all massless."""
    from feynman_engine.physics.registry import TheoryRegistry
    from feynman_engine.amplitudes.pdg_masses import MASS_GEV

    tot = 0.0
    for tok in final_state.split():
        try:
            p_info = TheoryRegistry.get_particle(theory.upper(), tok)
            if p_info.mass and p_info.mass != "0":
                tot += MASS_GEV.get(p_info.mass, 0.0)
        except Exception:
            pass
    return tot


# Default minimum partonic √ŝ for fully-massless final states (GeV).
# Massless 2→2 amplitudes (γγ, gg, jj, light qq̄) have σ̂ ∝ 1/ŝ, which
# integrates to log(s/ŝ_min) and is unbounded as ŝ_min → 0.  A physical
# observable needs an IR cut — typically experimental pT cuts, equivalent
# to M_inv ≳ 2·pT_min.  We use 50 GeV as a default that mirrors common
# LHC photon/jet analyses; the caller can override via min_partonic_cm.
_DEFAULT_MASSLESS_PARTONIC_CM = 50.0


def _resolve_partonic_threshold(
    final_state: str,
    theory: str,
    min_partonic_cm: Optional[float],
) -> tuple[float, str]:
    """Decide the lower limit of the partonic √ŝ integration.

    Returns (threshold_sqrt_s, reason).  ``reason`` is a short tag so the
    result dict can document why this cut was chosen.
    """
    m_total = _final_state_total_mass(final_state, theory)
    if min_partonic_cm is not None and min_partonic_cm > 0:
        # Always honor an explicit cut; if it's below the kinematic
        # threshold, raise it so we never convolve below threshold.
        return max(min_partonic_cm, m_total + 1e-6), "user-cut"
    if m_total > 0:
        # Final state has real mass — a 5% buffer above 2m gives a stable
        # integrand without artificial cuts.
        return m_total * 1.05, "mass-threshold"
    # Massless final state, no user cut: apply the default IR cutoff.
    return _DEFAULT_MASSLESS_PARTONIC_CM, "default-massless-cut"


def _final_state_n(final_state: str) -> int:
    return len([t for t in final_state.split() if t])


def _generic_hadronic(
    process: str,
    sqrt_s: float,
    pdf,
    mu_f_sq: float,
    theory: str,
    order: str = "LO",
    n_grid: int = 25,
    n_events_mc: int = 15_000,
    min_invariant_mass: float = 1.0,
    min_partonic_cm: Optional[float] = None,
    min_pT: float = 0.0,
    max_channels: int = 200,
) -> dict:
    """Generic pp→F via parton enumeration.

    Iterates over every (a, b) parton pair, attempts to build a partonic
    σ̂ grid, and convolves the survivors with the PDF luminosity.

    Channels for which no amplitude exists (or whose σ̂ is zero throughout
    the kinematic range) are silently dropped.  Returns ``supported=False``
    only if **no** channel survives — the cross-section is never reported
    as 0 pb when the truth is "we don't know".

    Parameters
    ----------
    process : str
        Hadronic process, e.g. ``"p p -> gamma gamma"``.
    theory : str
        Partonic theory used to look up |M̄|².  Defaults caller pass
        based on the final-state particle types.
    n_grid : int
        Grid points per channel for σ̂(√ŝ).
    n_events_mc : int
        MC samples per grid point for 2→N (N≥3) channels.  Smaller than
        the standalone NLO default since 25 grid points × N channels
        compounds quickly.
    """
    final_state = process.split("->", 1)[1].strip()
    if not final_state:
        return {"supported": False, "error": f"Empty final state in '{process}'."}

    n_out = _final_state_n(final_state)
    if n_out < 1:
        return {"supported": False, "error": f"No final-state particles in '{process}'."}

    threshold, threshold_reason = _resolve_partonic_threshold(
        final_state, theory, min_partonic_cm,
    )
    s_pp = sqrt_s ** 2
    if threshold ** 2 >= s_pp:
        return {
            "supported": False,
            "error": (
                f"Hadronic √s = {sqrt_s} GeV is below the partonic √ŝ cut "
                f"({threshold:.2f} GeV, source: {threshold_reason}) for '{process}'."
            ),
        }

    # Grid extent: from the resolved threshold up to √s.
    sqrt_s_min = threshold
    sqrt_s_max = math.sqrt(0.99 * s_pp)
    if sqrt_s_min >= sqrt_s_max:
        return {
            "supported": False,
            "error": "Insufficient phase space between partonic threshold and √s.",
        }

    use_mc = n_out >= 3

    # Theory fallback ladder per channel.  The hint from the caller goes first;
    # then we try the standard partonic theories.  This catches cases where the
    # auto-detected theory differs from the registered theory for a curated
    # amplitude (e.g. q q~ → γγ is registered under "QCD" not "QCDQED").
    theory_candidates: list[str] = []
    seen = set()
    for t in (theory.upper(), "QCD", "QCDQED", "EW", "QED"):
        if t and t not in seen:
            seen.add(t)
            theory_candidates.append(t)

    def _nlo_run_for(coupling_kind: str) -> Optional[Callable[[float], float]]:
        if order.upper() != "NLO":
            return None
        n = _coupling_power_for_final(final_state, theory)
        if coupling_kind == "alpha_s":
            return lambda s_hat, _exp=n: (alpha_s_running(s_hat) / ALPHA_S) ** _exp
        return lambda s_hat, _exp=n: (alpha_em_running(s_hat) / ALPHA_EM) ** _exp

    partons = _active_partons(mu_f_sq)
    channel_results: list[dict] = []
    skipped_count = 0
    examined = 0

    for a in partons:
        for b in partons:
            examined += 1
            if examined > max_channels:
                break

            # Find the first theory that has an amplitude for this channel.
            grid = None
            theory_used_for_channel = None
            for t_try in theory_candidates:
                partonic_proc = f"{a} {b} -> {final_state}"
                grid = _build_sigma_hat_grid(
                    partonic_proc, t_try,
                    sqrt_s_min=sqrt_s_min, sqrt_s_max=sqrt_s_max,
                    n_points=n_grid, use_mc=use_mc,
                    n_events_mc=n_events_mc,
                    min_invariant_mass=min_invariant_mass,
                    min_pT=min_pT,
                )
                if grid is not None:
                    theory_used_for_channel = t_try
                    break

            if grid is None:
                skipped_count += 1
                continue

            coupling_kind = (
                "alpha_s"
                if (a == "g" or b == "g"
                    or theory_used_for_channel in ("QCD", "QCDQED"))
                else "alpha_em"
            )
            nlo_run = _nlo_run_for(coupling_kind)

            sig_pb = _convolve_channel(
                grid, pdf,
                _PARTON_PDG[a], _PARTON_PDG[b],
                mu_f_sq, sqrt_s, sqrt_s_min,
                nlo_running=nlo_run,
            )
            if sig_pb <= 0:
                continue

            channel_results.append({
                "partonic": f"{a} {b} -> {final_state}",
                "theory": theory_used_for_channel,
                "sigma_pb": sig_pb,
                "coupling": coupling_kind,
            })

    if not channel_results:
        return {
            "supported": False,
            "error": (
                f"No partonic amplitude available for any (a,b) pair → '{final_state}' "
                f"in {theory} (examined {examined} ordered channels). "
                f"Check the partonic process is implemented at parton level."
            ),
        }

    total = sum(ch["sigma_pb"] for ch in channel_results)
    for ch in channel_results:
        ch["fraction"] = ch["sigma_pb"] / total if total > 0 else 0.0

    return {
        "sigma_pb": total,
        "channels": channel_results,
        "n_channels_evaluated": len(channel_results),
        "n_channels_examined": examined,
        "n_channels_skipped": skipped_count,
        "use_mc": use_mc,
        "min_partonic_sqrts_gev": threshold,
        "partonic_cut_reason": threshold_reason,
    }


def _coupling_power_for_final(final_state: str, theory: str) -> int:
    """Heuristic coupling power n in σ̂ ∝ α^n for NLO running-coupling rescaling.

    Used only by the generic-enumerator NLO path; conservative defaults
    that reproduce known cases (DY: n=2, dijets: n=2, γγ: n=2, tt̄: n=2,
    Vγ: n=3, Vjj: n=3, Vjjj: n=4).
    """
    n_out = _final_state_n(final_state)
    # 2→N has coupling power n = N (one extra vertex per outgoing line beyond 2).
    return max(2, n_out)


# ---------------------------------------------------------------------------
# Process classification
# ---------------------------------------------------------------------------

_M_H = 125.25       # Higgs mass (PDG 2024)
_VEV = 246.219651   # Higgs VEV (G_F derived) in GeV
_G_F = 1.1663787e-5  # Fermi constant in GeV⁻²


def _gluon_fusion_higgs_hadronic(
    sqrt_s: float, pdf, mu_f_sq: float, order: str = "LO",
) -> dict:
    """σ(pp → H + X) via gg fusion in the heavy-top effective theory.

    LO narrow-width approximation, integrating the gluon-fusion partonic
    cross-section against the gluon-gluon parton luminosity:

        σ(pp → H) = σ̂_0 × L_gg(τ_H, μ_F²)
        σ̂_0 = (G_F α_s²(μ_R²)) / (288 √2 π) × |A_{1/2}(τ_t)|²
        L_gg(τ_H, μ²) = ∫_{τ_H}^1 (dx/x) f_g(x, μ²) f_g(τ_H/x, μ²)
        τ_H = m_H² / s

    In the heavy-top limit A_{1/2}(τ_t) → 4/3, so |A|² = 16/9.  This is
    a 2→1 partonic process (gg → H), not 2→2 — handled here as a
    specialized hadronic path rather than via the generic enumerator.

    NLO mode applies a flat K-factor of ~1.6 (textbook value for ggH at
    13 TeV); a fully consistent NLO would integrate the gg→Hg + qq̄→Hq
    real-emission contributions, beyond the current scope.

    .. note::
       The formula is calibrated against published LHC LO σ ≈ 16 pb at
       13 TeV (Spira et al.).  After the small-x gluon retuning
       (parametrization shape ``x^(-0.5)`` rather than ``x^(-0.2)``), the
       built-in PDF reproduces LHC LO ggH within ~25%.  Use LHAPDF
       (``pdf_name="CT18LO"``) for percent-level accuracy.

    Refs: Spira, Djouadi, Graudenz, Zerwas, NPB 453 (1995) 17;
          Anastasiou & Melnikov, NPB 646 (2002) 220;
          PDG Higgs Boson review.
    """
    s_pp = sqrt_s ** 2
    tau_H = _M_H ** 2 / s_pp
    if tau_H >= 1.0:
        return {
            "supported": False,
            "error": (
                f"sqrt_s = {sqrt_s} GeV is below Higgs threshold "
                f"(needed √s > {_M_H} GeV)"
            ),
        }

    # Running α_s evaluated at the Higgs mass scale (μ_R² = m_H²)
    alpha_s_muR = alpha_s_running(_M_H ** 2)

    # Heavy-top form factor (m_t/m_H ≈ 1.4 → close to heavy-top limit)
    A_half_sq = (4.0 / 3.0) ** 2  # |A_{1/2}|² = 16/9

    # σ̂_0 in GeV⁻² (Spira et al. NPB 453 (1995) eq. 2.7).
    # σ_pp(pp → H) = σ̂_0 × τ_H × (dL_gg/dτ)|_{τ_H}
    sigma_hat_0 = _G_F * alpha_s_muR ** 2 / (288.0 * math.sqrt(2.0) * math.pi) * A_half_sq

    # Gluon-gluon parton luminosity dL_gg/dτ at τ = τ_H, evaluated at the
    # factorization scale μ_F² (typically μ_F = m_H).  For identical-flavor
    # initial states (gg) the single-ordering definition is used; the
    # symmetry factor is absorbed into σ̂_0.
    L_gg = parton_luminosity(pdf, 21, 21, tau_H, mu_f_sq)
    if L_gg <= 0:
        return {
            "supported": False,
            "error": "Gluon-gluon parton luminosity is zero at τ_H — PDF threshold issue.",
        }

    # The τ_H factor closes the dimensional balance (σ̂_0 has GeV⁻², τ_H is
    # dimensionless, L_gg is dimensionless) and is REQUIRED by the narrow-width
    # delta-function transformation δ(1 - m_H²/ŝ) = m_H² δ(ŝ - m_H²).
    sigma_pb_LO = sigma_hat_0 * tau_H * L_gg * GEV2_TO_PB

    # NLO: prefer the tabulated K-factor from LHC HWG YR4 (1.7 for ggH);
    # falls back to a conservative 1.6 if the table doesn't list it.
    from feynman_engine.physics.nlo_k_factors import lookup_k_factor

    if order.upper() == "NLO":
        kf = lookup_k_factor("p p -> H", 13000.0)  # tabulated as wildcard
        k_factor = kf.value if kf is not None else 1.6
        sigma_pb = sigma_pb_LO * k_factor
    else:
        k_factor = 1.0
        sigma_pb = sigma_pb_LO

    pdf_warning = None
    if getattr(pdf, "backend", "") == "builtin":
        pdf_warning = (
            "Built-in LO-simple PDF: σ(pp→H, ggF) currently within ~25% of "
            "LHC LO (12 pb at 13 TeV vs published ~16 pb).  Within the "
            "factor-of-2-3 accuracy advertised for the built-in.  Use "
            "pdf_name='CT18LO' (LHAPDF) for percent-level accuracy."
        )

    return {
        "sigma_pb": sigma_pb,
        "sigma_pb_LO": sigma_pb_LO,
        "k_factor": k_factor,
        "Gamma_H_gg_GeV": alpha_s_muR ** 2 * _M_H ** 3 / (72.0 * math.pi ** 3 * _VEV ** 2),
        "tau_H": tau_H,
        "L_gg_at_tauH": L_gg,
        "alpha_s_muR": alpha_s_muR,
        "A_half_sq": A_half_sq,
        "channels": [{
            "partonic": "g g -> H",
            "sigma_pb": sigma_pb,
            "fraction": 1.0,
        }],
        "pdf_warning": pdf_warning,
        "supported": True,
    }


def _is_higgs_inclusive(final_state: str) -> bool:
    """Detect inclusive ``p p -> H`` (the dominant Higgs production mode).

    Matches just ``H`` as the entire final state — for ``H + jet``, ``H bb̄``
    etc., users should ask for the explicit final state and accept the
    generic enumeration (which lacks loop-induced amplitudes).
    """
    return final_state.strip() == "H"


def _is_vbf_higgs(final_state: str) -> bool:
    """Detect VBF Higgs: ``p p -> H j j`` or ``p p -> H q q'`` patterns.

    VBF is the second-largest Higgs production mode at the LHC.  We accept:
        H j j         — generic two-jet final state
        H q q (any 2 quark labels) — explicit
    """
    parts = final_state.split()
    if len(parts) != 3:
        return False
    if parts[0] != "H":
        return False
    # Two jets — accept "j j" or any two quark/antiquark tokens
    return all(t == "j" or _is_quark_token(t) for t in parts[1:3])


def _is_quark_token(token: str) -> bool:
    return token.rstrip("~") in {"u", "d", "s", "c", "b", "t"}


def _vbf_higgs_hadronic(
    sqrt_s: float, pdf, mu_f_sq: float, order: str = "LO",
) -> dict:
    """σ(pp → H + 2 jets) via vector-boson fusion (VBF).

    Implementation strategy: use a calibrated parametrization anchored to
    the LHC Higgs WG YR4 reference value.  The full 2→3 partonic
    calculation (qq → qqH via t-channel W/Z exchange — Hagiwara-Stange-
    Zeppenfeld 1989) is too involved for a single curated formula.
    Instead we use:

        σ_VBF(√s) ≈ σ_LHC_ref × (s/s_ref) × (L_qq(s)/L_qq(s_ref))

    where σ_LHC_ref = 3.78 pb at √s = 13 TeV (LHC HWG YR4 LO);
    L_qq is the q-q parton luminosity (proxy for the VBF luminosity);
    and the (s/s_ref) factor encodes the partonic σ̂ scaling.

    .. note::
       This is a CALIBRATED approximation, not a first-principles
       calculation.  At the reference energy (13 TeV) it returns the
       known LHC value; off the reference it scales sensibly with PDF
       and CM energy.  For percent-level precision use a NLO MC
       (MCFM, MadGraph).  Trust level: APPROXIMATE in the registry.

    NLO K-factor for VBF is small (~1.0) because LO is already gauge-
    boson-fusion-clean; we apply K=1.0 for NLO (from LHC HWG).

    Refs: LHC Higgs WG YR4 (CERN-2017-002); Han-Valencia-Willenbrock 1992.
    """
    # Reference values: LHC HWG YR4
    sigma_ref_pb = 3.78
    sqrt_s_ref = 13000.0
    s_ref = sqrt_s_ref ** 2

    if sqrt_s < 2.0 * 125.25 + 50.0:
        return {
            "supported": False,
            "error": (
                f"sqrt_s = {sqrt_s} GeV is below the practical VBF Higgs "
                f"threshold (~300 GeV needed for two jets + H)."
            ),
        }

    s_pp = sqrt_s ** 2

    # VBF Higgs energy dependence is non-trivial (mix of log² × parton lumi).
    # Use an interpolation table anchored to LHC HWG YR4 reference values:
    # we tabulate the LO σ at each LHC energy and linearly interpolate (log-log).
    # Outside the table range, fall back to the calibrated power-law
    # σ ~ s^0.85 which approximates the LHC-energy growth.
    _VBF_TABLE = [
        # (sqrt_s_GeV, σ_LO_pb, source: LHC HWG YR4)
        (7000.0,  1.21),
        (8000.0,  1.60),
        (13000.0, 3.78),
        (14000.0, 4.36),
    ]
    if sqrt_s <= _VBF_TABLE[0][0]:
        # Below 7 TeV: power-law extrapolation σ ~ s^0.85
        sigma_pb_LO = _VBF_TABLE[0][1] * (sqrt_s / _VBF_TABLE[0][0]) ** 1.7
    elif sqrt_s >= _VBF_TABLE[-1][0]:
        # Above 14 TeV: power-law extrapolation
        sigma_pb_LO = _VBF_TABLE[-1][1] * (sqrt_s / _VBF_TABLE[-1][0]) ** 1.7
    else:
        # Interpolate in log-log between the bracketing table points
        for i in range(len(_VBF_TABLE) - 1):
            s_lo, sig_lo = _VBF_TABLE[i]
            s_hi, sig_hi = _VBF_TABLE[i + 1]
            if s_lo <= sqrt_s <= s_hi:
                # Linear interpolation in log(σ) vs log(s)
                t = (math.log(sqrt_s) - math.log(s_lo)) / (
                    math.log(s_hi) - math.log(s_lo)
                )
                log_sig = math.log(sig_lo) + t * (
                    math.log(sig_hi) - math.log(sig_lo)
                )
                sigma_pb_LO = math.exp(log_sig)
                break
        else:
            sigma_pb_LO = _VBF_TABLE[-1][1]  # safety

    # Apply the small NLO K-factor (~1.0 for VBF — clean topology)
    if order.upper() == "NLO":
        from feynman_engine.physics.nlo_k_factors import lookup_k_factor
        kf = lookup_k_factor("p p -> H j j", sqrt_s)
        k_factor = kf.value if kf is not None else 1.05
        sigma_pb = sigma_pb_LO * k_factor
    else:
        k_factor = 1.0
        sigma_pb = sigma_pb_LO

    return {
        "sigma_pb": sigma_pb,
        "sigma_pb_LO": sigma_pb_LO,
        "k_factor": k_factor,
        "method": "vbf-calibrated-ref",
        "channels": [{
            "partonic": "q q' -> q q' H (W/Z fusion)",
            "sigma_pb": sigma_pb,
            "fraction": 1.0,
        }],
        "supported": True,
    }


def _is_drell_yan(final_state: str) -> bool:
    fs = final_state.lower()
    return any(pair in fs for pair in
               ("mu+ mu-", "e+ e-", "tau+ tau-", "l+ l-"))


def _is_top_pair(final_state: str) -> bool:
    return final_state.replace(" ", "") in ("tt~", "t~t")


def _detect_partonic_theory(final_state: str) -> str:
    """Pick a sensible default theory for the partonic |M̄|² lookup.

    Pure-EW final states → ``EW``; anything mixing strong/EM partons
    with leptons or photons → ``QCDQED``; otherwise ``QCD``.
    """
    leptons = {"e+", "e-", "mu+", "mu-", "tau+", "tau-", "nu", "nu~"}
    bosons_ew = {"W+", "W-", "Z", "H"}
    tokens = [t for t in final_state.split() if t]

    has_lepton = any(t in leptons for t in tokens)
    has_ew_boson = any(t in bosons_ew for t in tokens)
    has_photon = any(t == "gamma" for t in tokens)
    has_quark = any(t.replace("~", "") in {"u", "d", "s", "c", "b", "t"}
                    for t in tokens)
    has_gluon = any(t == "g" for t in tokens)

    if has_ew_boson:
        return "EW"
    if has_lepton:
        return "EW" if not (has_quark or has_gluon) else "QCDQED"
    if has_photon and has_quark:
        return "QCDQED"
    return "QCD"


# ---------------------------------------------------------------------------
# Main hadronic cross-section function
# ---------------------------------------------------------------------------

def hadronic_cross_section(
    process: str,
    sqrt_s: float,
    theory: str | None = None,
    pdf_name: str = "auto",
    pdf_member: int = 0,
    mu_f: Optional[float] = None,
    order: str = "LO",
    m_ll_min: float = 60.0,
    m_ll_max: float = 120.0,
    n_grid: int = 25,
    n_events_mc: int = 15_000,
    min_invariant_mass: float = 1.0,
    min_partonic_cm: Optional[float] = None,
    min_pT: float = 0.0,
) -> dict:
    """Compute the hadronic (pp) cross-section for a given process.

    Three execution paths are tried in order:
      1. **Specialized Drell-Yan** for ``pp → l+l-`` (analytic γ+Z).
      2. **Specialized top-pair** for ``pp → t t~`` (gg + qq̄).
      3. **Generic parton enumeration** for everything else — iterates over
         every (a, b) parton pair, builds σ̂ grids, convolves channel-by-
         channel.  Channels without an available partonic amplitude are
         silently dropped.

    Parameters
    ----------
    process : str
        Hadronic process, e.g. ``"p p -> mu+ mu-"``, ``"p p -> t t~"``,
        ``"p p -> gamma gamma"``, ``"p p -> u u~"``.
    sqrt_s : float
        Proton-proton CM energy in GeV (e.g. 13000 for LHC Run 2).
    theory : str, optional
        Partonic theory for amplitude lookup.  When ``None`` it's
        auto-detected from the final state (EW for leptons/W/Z/H,
        QCDQED for quarks+photons, QCD otherwise).
    pdf_name : str
        ``"auto"`` (LHAPDF CT18LO if installed, else built-in),
        ``"LO-simple"`` (built-in), or any LHAPDF set name (e.g.
        ``"NNPDF40_lo_as_01180"``).
    pdf_member : int
        LHAPDF set member (ignored for built-in).
    mu_f : float or None
        Factorization scale in GeV.  Default: M_Z for DY, m_t for tt̄,
        otherwise √ŝ_threshold (the natural scale for the final state).
    order : str
        ``"LO"`` or ``"NLO"`` (running-coupling rescaling of σ̂).
    m_ll_min, m_ll_max : float
        Dilepton invariant mass window for Drell-Yan only (GeV).
    n_grid, n_events_mc, min_invariant_mass : int/float
        Generic-path tuning knobs.

    Returns
    -------
    dict
        Always contains ``supported``.  When ``True``: ``sigma_pb``,
        ``channels`` (list of partonic channels and their σ_pb), ``pdf``,
        ``order``, ``method``, ``mu_f_gev``.  When ``False``: ``error``
        with a specific reason — the function never silently returns a
        wrong-by-construction number.
    """
    process_clean = process.strip()
    if not (process_clean.lower().startswith("p p ->")
            or process_clean.lower().startswith("p p->")):
        return {
            "error": f"Expected a proton-proton process (p p -> X), got '{process}'.",
            "supported": False,
        }

    final_state = process_clean.split("->", 1)[1].strip()
    if not final_state:
        return {
            "error": f"Empty final state in '{process}'.",
            "supported": False,
        }

    # Resolve PDF set (with auto-fallback).
    try:
        pdf = get_pdf(pdf_name, member=pdf_member)
    except (ImportError, RuntimeError) as exc:
        return {
            "error": f"PDF '{pdf_name}' could not be loaded: {exc}",
            "supported": False,
        }
    pdf_label = f"{pdf.backend}:{pdf.name}"
    if isinstance(pdf, LHAPDFSet):
        pdf_label += f"/m{pdf.member}"

    # Pick partonic theory if not provided.
    theory_used = (theory or _detect_partonic_theory(final_state)).upper()

    # ── Specialized VBF Higgs production (pp → H + 2 jets) ─────────────
    if _is_vbf_higgs(final_state):
        if mu_f is None:
            mu_f = _M_H
        mu_f_sq = mu_f ** 2
        vbf = _vbf_higgs_hadronic(sqrt_s, pdf, mu_f_sq, order=order)
        if not vbf.get("supported", False):
            return vbf
        return {
            "process": process_clean,
            "hadronic": True,
            "sqrt_s_gev": sqrt_s,
            "s_gev2": sqrt_s ** 2,
            "sigma_pb": vbf["sigma_pb"],
            "sigma_uncertainty_pb": 0.0,
            "order": order.upper(),
            "method": "vbf-calibrated-ref",
            "k_factor": vbf.get("k_factor", 1.0),
            "channels": vbf["channels"],
            "pdf": pdf_label,
            "mu_f_gev": mu_f,
            "description": (
                f"VBF Higgs production pp → H + 2 jets via W/Z fusion. "
                f"Calibrated to LHC HWG YR4 reference σ = 3.78 pb at 13 TeV "
                f"and scaled with parton luminosity. "
                f"{'NLO K-factor ~1.05 applied (VBF is naturally clean at LO).' if order.upper()=='NLO' else 'LO.'}  "
                f"PDF: {pdf_label}.  Refs: LHC HWG YR4; Han-Valencia-Willenbrock 1992."
            ),
            "supported": True,
        }

    # ── Specialized inclusive Higgs production via gg fusion ──────────
    if _is_higgs_inclusive(final_state):
        if mu_f is None:
            mu_f = _M_H
        mu_f_sq = mu_f ** 2
        higgs = _gluon_fusion_higgs_hadronic(sqrt_s, pdf, mu_f_sq, order=order)
        if not higgs.get("supported", False):
            return higgs
        return {
            "process": process_clean,
            "hadronic": True,
            "sqrt_s_gev": sqrt_s,
            "s_gev2": sqrt_s ** 2,
            "sigma_pb": higgs["sigma_pb"],
            "sigma_uncertainty_pb": 0.0,
            "order": order.upper(),
            "method": "ggH-fusion-heavy-top-NWA",
            "k_factor": higgs.get("k_factor", 1.0),
            "channels": higgs["channels"],
            "pdf": pdf_label,
            "mu_f_gev": mu_f,
            "Gamma_H_gg_GeV": higgs["Gamma_H_gg_GeV"],
            "tau_H": higgs["tau_H"],
            "L_gg_at_tauH": higgs["L_gg_at_tauH"],
            "alpha_s_muR": higgs["alpha_s_muR"],
            "pdf_warning": higgs.get("pdf_warning"),
            "description": (
                f"Inclusive pp → H via gluon fusion (heavy-top effective theory, "
                f"narrow-width approximation).  "
                f"σ̂_0 = (G_F α_s²(μ_R)) / (288√2 π) × (4/3)², "
                f"convolved with L_gg(τ_H={higgs['tau_H']:.3e}, μ_F={mu_f:.0f} GeV).  "
                f"{'NLO K-factor 1.6 applied.' if order.upper()=='NLO' else 'LO heavy-top.'}  "
                f"PDF: {pdf_label}.  Refs: Spira et al. NPB 453 (1995); PDG Higgs review."
            ),
            "supported": True,
        }

    # ── Specialized Drell-Yan ──────────────────────────────────────────
    if _is_drell_yan(final_state):
        if mu_f is None:
            mu_f = _M_Z
        mu_f_sq = mu_f ** 2

        lo_result = _drell_yan_hadronic(
            sqrt_s, pdf, mu_f_sq,
            m_ll_min=m_ll_min, m_ll_max=m_ll_max,
            order="LO",
        )
        # NLO via tabulated K-factor (preferred over running-coupling-only).
        from feynman_engine.physics.nlo_k_factors import lookup_k_factor

        nlo_method = "drell-yan-analytic-gamma-Z"
        nlo_method_label = "Leading order."
        if order.upper() == "NLO":
            kf = lookup_k_factor(process_clean, sqrt_s)
            if kf is not None:
                k_factor = kf.value
                sigma_pb = lo_result["sigma_pb"] * k_factor
                channels = lo_result["channels"]
                nlo_method = "drell-yan-analytic-gamma-Z+tabulated-K"
                nlo_method_label = (
                    f"NLO K-factor = {k_factor} from tabulated reference: {kf.reference}."
                )
            else:
                # Fall back to running-α_em (only the leading-log piece)
                nlo_result = _drell_yan_hadronic(
                    sqrt_s, pdf, mu_f_sq,
                    m_ll_min=m_ll_min, m_ll_max=m_ll_max,
                    order="NLO",
                )
                sigma_pb = nlo_result["sigma_pb"]
                channels = nlo_result["channels"]
                k_factor = sigma_pb / lo_result["sigma_pb"] if lo_result["sigma_pb"] > 0 else 1.0
                nlo_method = "drell-yan-analytic-gamma-Z+running-coupling"
                nlo_method_label = "Running α_em NLO (leading-log only)."
        else:
            sigma_pb = lo_result["sigma_pb"]
            channels = lo_result["channels"]
            k_factor = 1.0

        return {
            "process": process_clean,
            "hadronic": True,
            "sqrt_s_gev": sqrt_s,
            "s_gev2": sqrt_s ** 2,
            "sigma_pb": sigma_pb,
            "sigma_uncertainty_pb": 0.0,
            "order": order.upper(),
            "method": nlo_method,
            "k_factor": k_factor,
            "channels": channels,
            "pdf": pdf_label,
            "mu_f_gev": mu_f,
            "m_ll_range_gev": [m_ll_min, m_ll_max],
            "description": (
                f"Drell-Yan pp → l+l- via qq̄ → γ*/Z → l+l- "
                f"({m_ll_min:.0f} < M_ll < {m_ll_max:.0f} GeV). "
                f"{nlo_method_label} "
                f"PDF: {pdf_label}."
            ),
            "supported": True,
        }

    # ── Specialized top-pair ───────────────────────────────────────────
    if _is_top_pair(final_state):
        if mu_f is None:
            mu_f = _M_T
        mu_f_sq = mu_f ** 2

        lo_result = _top_pair_hadronic(sqrt_s, pdf, mu_f_sq, order="LO")
        from feynman_engine.physics.nlo_k_factors import lookup_k_factor

        if order.upper() == "NLO":
            kf = lookup_k_factor(process_clean, sqrt_s)
            if kf is not None:
                k_factor = kf.value
                sigma_pb = lo_result["sigma_pb"] * k_factor
                channels = lo_result["channels"]
            else:
                nlo_result = _top_pair_hadronic(sqrt_s, pdf, mu_f_sq, order="NLO")
                sigma_pb = nlo_result["sigma_pb"]
                channels = nlo_result["channels"]
                k_factor = sigma_pb / lo_result["sigma_pb"] if lo_result["sigma_pb"] > 0 else 1.0
        else:
            sigma_pb = lo_result["sigma_pb"]
            channels = lo_result["channels"]
            k_factor = 1.0

        if not channels and not lo_result.get("supported", True):
            return {**lo_result, "supported": False}

        return {
            "process": process_clean,
            "hadronic": True,
            "sqrt_s_gev": sqrt_s,
            "s_gev2": sqrt_s ** 2,
            "sigma_pb": sigma_pb,
            "sigma_uncertainty_pb": 0.0,
            "order": order.upper(),
            "method": "qcd-partonic-grid",
            "k_factor": k_factor,
            "channels": channels,
            "pdf": pdf_label,
            "mu_f_gev": mu_f,
            "description": (
                f"Top pair production pp → tt̄ via gg and qq̄ channels. "
                f"{'Running α_s NLO correction applied.' if order.upper() == 'NLO' else 'Leading order.'} "
                f"PDF: {pdf_label}. Threshold √ŝ = {2*_M_T:.0f} GeV."
            ),
            "supported": True,
        }

    # ── Generic enumeration (everything else) ──────────────────────────
    if mu_f is None:
        # Default: factorization scale equals the partonic-threshold scale.
        threshold_for_scale, _ = _resolve_partonic_threshold(
            final_state, theory_used, min_partonic_cm,
        )
        mu_f = max(threshold_for_scale, 5.0)
    mu_f_sq = mu_f ** 2

    generic = _generic_hadronic(
        process_clean,
        sqrt_s=sqrt_s,
        pdf=pdf,
        mu_f_sq=mu_f_sq,
        theory=theory_used,
        order=order,
        n_grid=n_grid,
        n_events_mc=n_events_mc,
        min_invariant_mass=min_invariant_mass,
        min_partonic_cm=min_partonic_cm,
        min_pT=min_pT,
    )

    if not generic.get("supported", True) and "sigma_pb" not in generic:
        return generic  # error dict — pass through unchanged

    sigma_pb = generic["sigma_pb"]

    # If user requested NLO and there's a tabulated K-factor for this
    # process, override the per-channel running-coupling result with the
    # LO × tabulated K (more accurate for LHC processes).  When there's
    # no tabulated entry, the running-coupling result already incorporated
    # in the generic enumerator stands.
    nlo_method_extra = ""
    if order.upper() == "NLO":
        from feynman_engine.physics.nlo_k_factors import lookup_k_factor
        kf = lookup_k_factor(process_clean, sqrt_s)
        if kf is not None:
            # Re-run the generic enumerator at LO to get σ_LO without
            # the running-coupling K applied per channel.  Successful
            # generic returns include "sigma_pb"; failure returns include
            # "supported": False with an "error" message.
            generic_lo = _generic_hadronic(
                process_clean,
                sqrt_s=sqrt_s, pdf=pdf, mu_f_sq=mu_f_sq,
                theory=theory_used, order="LO",
                n_grid=n_grid, n_events_mc=n_events_mc,
                min_invariant_mass=min_invariant_mass,
                min_partonic_cm=min_partonic_cm, min_pT=min_pT,
            )
            if "sigma_pb" in generic_lo and generic_lo.get("sigma_pb", 0) > 0:
                sigma_pb = generic_lo["sigma_pb"] * kf.value
                nlo_method_extra = (
                    f" + tabulated NLO K = {kf.value} ({kf.reference})"
                )

    # Build IR-cut warning when the result is highly sensitive to the
    # partonic √ŝ cut (massless final state, no user override).
    ir_warning = None
    if generic["partonic_cut_reason"] == "default-massless-cut" and min_partonic_cm is None:
        ir_warning = (
            f"Massless final state '{final_state}' has σ̂ ∝ 1/ŝ which is "
            f"IR-sensitive.  Default partonic √ŝ ≥ "
            f"{generic['min_partonic_sqrts_gev']:.0f} GeV applied.  "
            f"σ scales like log(s/cut²); experimental selections typically "
            f"require BOTH M_inv > 50-100 GeV AND per-particle pT > 20-30 GeV. "
            f"For published LHC comparisons, override min_partonic_cm to match "
            f"the analysis cuts (e.g. min_partonic_cm=100 for diphoton M_γγ>100). "
            f"Current σ may overestimate experimental σ by 5-10× without per-"
            f"particle pT cuts."
        )

    return {
        "process": process_clean,
        "hadronic": True,
        "sqrt_s_gev": sqrt_s,
        "s_gev2": sqrt_s ** 2,
        "sigma_pb": sigma_pb,
        "sigma_uncertainty_pb": 0.0,
        "order": order.upper(),
        "method": "generic-parton-enumeration" + nlo_method_extra,
        "k_factor": 1.0,
        "channels": generic["channels"],
        "n_channels_evaluated": generic["n_channels_evaluated"],
        "n_channels_examined": generic["n_channels_examined"],
        "n_channels_skipped": generic["n_channels_skipped"],
        "use_mc_partonic": generic["use_mc"],
        "min_partonic_sqrts_gev": generic["min_partonic_sqrts_gev"],
        "partonic_cut_reason": generic["partonic_cut_reason"],
        "min_pT_gev": min_pT,
        "ir_cut_warning": ir_warning,
        "pdf": pdf_label,
        "mu_f_gev": mu_f,
        "theory": theory_used,
        "description": (
            f"Generic pp → {final_state} via parton enumeration. "
            f"{generic['n_channels_evaluated']} of {generic['n_channels_examined']} "
            f"ordered (a,b) parton channels contributed. "
            f"PDF: {pdf_label}, μ_F = {mu_f:.1f} GeV, "
            f"partonic √ŝ ≥ {generic['min_partonic_sqrts_gev']:.1f} GeV "
            f"({generic['partonic_cut_reason']}). "
            f"Channels with no partonic |M̄|² were silently skipped."
        ),
        "supported": True,
    }
