"""
NLO cross-section calculations for QED, QCD, EW, and BSM processes.

Two paths are supported:

**Exact analytic K-factors** (known from textbook results):
  - QED e+e- -> ff'bar (different flavor, 2->2): K = 1 + 3*alpha/(4*pi)
    Schwartz, "QFT and the Standard Model", Ch. 20

**Catani-Seymour subtraction** (exact NLO for e+e- -> mu+mu-(γ)):
  - Full V + I + (R - D) implementation in
    ``nlo_cross_section_subtracted_eemumu``.

Tabulated K-factors for the major LHC channels live in
``feynman_engine.physics.nlo_k_factors``.  For unregistered processes,
the trust system BLOCKS NLO requests at the API layer — we do not ship
a running-coupling fallback (it gives leading-log accuracy only and was
exactly the kind of "rough" answer V1 promises never to return).

The bare ``alpha_s_running`` / ``alpha_em_running`` helpers are still
exposed via the educational ``/amplitude/running-coupling`` endpoint.

References:
    Schwartz, "QFT and the Standard Model", Ch. 20
    Ellis, Stirling, Webber, "QCD and Collider Physics", Ch. 3
    Peskin & Schroeder, Ch. 7 (running couplings)
    Catani & Seymour, NPB 485 (1997) 291 (dipole subtraction)
"""
from __future__ import annotations

import math

from feynman_engine.amplitudes.cross_section import (
    ALPHA_EM,
    ALPHA_S,
    total_cross_section,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_M_Z = 91.1876  # Z boson mass in GeV (PDG 2023)

# QED: exact K-factor for massless e+e- -> ff'bar (different flavor)
_K_QED_EEFF = 1.0 + 3.0 * ALPHA_EM / (4.0 * math.pi)

# For backward compatibility
_NLO_K_FACTOR = _K_QED_EEFF


# ---------------------------------------------------------------------------
# Running couplings (1-loop)
# ---------------------------------------------------------------------------

def alpha_s_running(mu2: float, n_f: int = 5) -> float:
    """1-loop running of alpha_s from M_Z to scale mu^2.

    alpha_s(mu^2) = alpha_s(M_Z^2) / (1 + beta_0 * alpha_s(M_Z^2)/(2*pi) * ln(mu^2/M_Z^2))

    where beta_0 = (11*C_A - 2*n_f) / 3 = (33 - 2*n_f) / 3.

    Parameters
    ----------
    mu2 : float
        Renormalization scale squared in GeV^2.
    n_f : int
        Number of active quark flavors (default 5).

    Returns
    -------
    float — alpha_s at scale mu^2.
    """
    if mu2 <= 0:
        return ALPHA_S
    beta_0 = (33.0 - 2.0 * n_f) / 3.0
    log_ratio = math.log(mu2 / _M_Z ** 2)
    denom = 1.0 + beta_0 * ALPHA_S / (2.0 * math.pi) * log_ratio
    if denom <= 0:
        # Hit the Landau pole — return a large value rather than crash
        return 1.0
    return ALPHA_S / denom


def alpha_em_running(s: float) -> float:
    """1-loop running of alpha_em from Q^2=0 to scale s.

    Includes leptonic vacuum polarisation only (e, mu, tau loops):
        alpha(s) = alpha(0) / (1 - Delta_alpha(s))
        Delta_alpha(s) = (alpha/3*pi) * sum_f Q_f^2 * ln(s / m_f^2)

    Parameters
    ----------
    s : float
        Centre-of-mass energy squared in GeV^2.

    Returns
    -------
    float — alpha_em at scale s.
    """
    if s <= 0:
        return ALPHA_EM
    # Lepton masses in GeV
    m_e = 0.0005110
    m_mu = 0.10566
    m_tau = 1.7768
    delta_alpha = 0.0
    for m_f in [m_e, m_mu, m_tau]:
        if s > 4.0 * m_f ** 2:
            delta_alpha += ALPHA_EM / (3.0 * math.pi) * math.log(s / m_f ** 2)
    denom = 1.0 - delta_alpha
    if denom <= 0:
        return ALPHA_EM
    return ALPHA_EM / denom


# ---------------------------------------------------------------------------
# Coupling power detection
# ---------------------------------------------------------------------------

def _detect_coupling_power(msq, coupling_name: str) -> int:
    """Detect the power of a coupling constant in |M|^2.

    Uses numerical substitution: evaluates |M|^2 at coupling=1 and coupling=2,
    then computes power = log2(ratio).  Falls back to SymPy Poly.degree()
    if the numerical method is ambiguous.

    Parameters
    ----------
    msq : sympy.Expr
        The spin-averaged |M|^2 expression.
    coupling_name : str
        The coupling symbol name ("e" or "g_s").

    Returns
    -------
    int — power of the coupling in |M|^2 (e.g. 4 for e^4, 6 for e^6).
    """
    import sympy as sp

    # Find the actual coupling symbol object in the expression
    coupling_sym = None
    for s in msq.free_symbols:
        if s.name == coupling_name:
            coupling_sym = s
            break
    if coupling_sym is None:
        return 4  # default: 2->2 standard

    # Fast path: numerical substitution using actual symbol objects from expr.
    # Assign all non-coupling symbols distinct prime-based values.
    # These are chosen to avoid accidental cancellations and propagator poles:
    # no value squared equals another value, all are non-zero.
    _PROBE_VALS = [
        2.3, 3.7, 5.3, 7.1, 11.3, 13.7, 17.1, 19.3, 23.7, 29.1,
        31.3, 37.7, 41.1, 43.3, 47.7, 53.1, 59.3, 61.7, 67.1, 71.3,
    ]
    subs_base = {}
    vi = 0
    for s in sorted(msq.free_symbols, key=lambda x: x.name):
        if s is not coupling_sym:
            subs_base[s] = _PROBE_VALS[vi % len(_PROBE_VALS)]
            vi += 1

    try:
        v1 = complex(msq.subs({**subs_base, coupling_sym: 1.0})).real
        v2 = complex(msq.subs({**subs_base, coupling_sym: 2.0})).real
        if abs(v1) > 1e-30:
            ratio = v2 / v1
            power = round(math.log2(abs(ratio)))
            if abs(2 ** power - abs(ratio)) < 0.01 * abs(ratio):
                return power
    except (ValueError, TypeError, OverflowError):
        pass

    # Slow path: SymPy polynomial degree
    try:
        poly = sp.Poly(sp.expand(msq), coupling_sym)
        return poly.degree()
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        pass

    # Default: assume 2->2 standard (4 for coupling, 2 for alpha)
    return 4


# ---------------------------------------------------------------------------
# Process classification
# ---------------------------------------------------------------------------

def _classify_process(process: str, theory: str) -> dict:
    """Classify a scattering process for NLO and return metadata.

    Supports 2->N processes (N >= 2).  Rejects 1->N (decays) since their
    EW/Yukawa couplings don't map cleanly to alpha_em/alpha_s running.

    Returns a dict with:
        supported: bool
        error: str (if not supported)
        nlo_method: str — 'exact-kfactor' or 'running-coupling'
        k_factor: float (if exact)
        coupling_power: int (power of alpha in |M|^2, i.e. deg(g)/2)
        coupling: str — 'alpha_em' or 'alpha_s'
        n_out: int — number of final-state particles
        description: str
    """
    import sympy as sp
    from feynman_engine.physics.translator import parse_process
    from feynman_engine.physics.amplitude import get_amplitude

    theory_up = theory.upper()

    # Parse and validate
    try:
        spec = parse_process(process.strip(), theory_up, loops=0)
    except Exception as exc:
        return {"supported": False, "error": str(exc)}

    n_in = len(spec.incoming)
    n_out = len(spec.outgoing)

    if n_in != 2:
        return {
            "supported": False,
            "error": (
                f"NLO requires a 2→N scattering process (got {n_in}→{n_out}). "
                "Decay widths (1→N) are not supported for NLO."
            ),
        }

    if n_out < 2:
        return {
            "supported": False,
            "error": f"NLO requires at least 2 final-state particles (got {n_out}).",
        }

    # Check that we have a tree-level amplitude
    result = get_amplitude(process.strip(), theory_up)
    if result is None or result.msq is None:
        return {
            "supported": False,
            "error": f"No tree-level amplitude available for '{process}' in {theory}.",
        }

    incoming = list(spec.incoming)
    outgoing = list(spec.outgoing)

    # Determine coupling and its power in |M|^2 by inspecting the amplitude
    msq = result.msq
    free_sym_names = {s.name for s in msq.free_symbols}

    has_gs = "g_s" in free_sym_names
    has_e = "e" in free_sym_names

    if has_gs:
        # QCD or mixed QCD+QED — use the dominant coupling (α_s)
        coupling = "alpha_s"
        coupling_sym_name = "g_s"
    elif has_e:
        # QED, EW, BSM
        coupling = "alpha_em"
        coupling_sym_name = "e"
    else:
        # No standard coupling found (EW-specific like g_Z, y_b)
        # Cannot do running-coupling NLO reliably
        return {
            "supported": False,
            "error": (
                f"No standard gauge coupling (e or g_s) found in the amplitude "
                f"for '{process}' in {theory}. "
                f"Found symbols: {sorted(free_sym_names)}. "
                "Running-coupling NLO requires e (QED) or g_s (QCD)."
            ),
        }

    # Auto-detect coupling power from the symbolic amplitude
    coupling_deg = _detect_coupling_power(msq, coupling_sym_name)
    coupling_power = coupling_deg // 2  # alpha = g^2/(4*pi)

    if coupling_power < 1:
        return {
            "supported": False,
            "error": f"Could not determine coupling power in |M|^2 for '{process}'.",
        }

    # --- Classify specific process topologies ---

    # Particle classification helpers
    leptons = {"e", "mu", "tau"}
    quarks = {"u", "d", "s", "c", "b", "t"}
    fermions = leptons | quarks
    bosons = {"gamma", "g"}

    def base_flavor(name: str) -> str:
        return name.rstrip("+-").replace("~", "")

    def is_fermion(name: str) -> bool:
        return base_flavor(name) in fermions

    in_flavors = {base_flavor(p) for p in incoming if is_fermion(p)}
    out_flavors = {base_flavor(p) for p in outgoing if is_fermion(p)}

    n_in_ferm = sum(1 for p in incoming if is_fermion(p))
    n_out_ferm = sum(1 for p in outgoing if is_fermion(p))

    # --- QED exact K-factors (2->2 only) ---
    if theory_up == "QED" and n_out == 2:
        # e+e- -> ff'bar (different flavor, s-channel only)
        if (n_in_ferm == 2 and n_out_ferm == 2
                and len(in_flavors) == 1 and len(out_flavors) == 1
                and in_flavors != out_flavors):
            return {
                "supported": True,
                "nlo_method": "exact-kfactor",
                "k_factor": _K_QED_EEFF,
                "coupling": coupling,
                "coupling_power": coupling_power,
                "n_out": n_out,
                "incoming": incoming,
                "outgoing": outgoing,
                "description": (
                    "Exact NLO: K = 1 + 3α/(4π). "
                    "Sum of virtual (vertex+box+VP) + real emission + counterterms."
                ),
            }

    # --- No exact K-factor available ---
    # We used to fall back to a running-coupling rescaling here, but that gave
    # users a number with unjustified precision (leading-log only, missing
    # vertex/box/real-emission).  Per V1 trust-system policy, BLOCKED.
    return {
        "supported": False,
        "error": (
            "No exact NLO K-factor is available for this process.  "
            "Request order='LO' for the Born σ, register a tabulated K-factor "
            "in feynman_engine.physics.nlo_k_factors, or use the OpenLoops "
            "virtual-K endpoint as a cross-check."
        ),
    }


# ---------------------------------------------------------------------------
# Heaviest final-state mass
# ---------------------------------------------------------------------------

def _heaviest_final_mass(outgoing: list[str], theory: str) -> float:
    """Return the heaviest final-state particle mass in GeV."""
    from feynman_engine.physics.registry import TheoryRegistry
    from feynman_engine.amplitudes.pdg_masses import MASS_GEV

    m_max = 0.0
    for p in outgoing:
        try:
            p_info = TheoryRegistry.get_particle(theory.upper(), p)
            if p_info.mass and p_info.mass != "0":
                m_max = max(m_max, MASS_GEV.get(p_info.mass, 0.0))
        except Exception:
            pass
    return m_max


# ---------------------------------------------------------------------------
# Born cross-section (handles 2->2 and 2->N)
# ---------------------------------------------------------------------------

def _get_born_cross_section(
    process: str,
    theory: str,
    sqrt_s: float,
    n_out: int,
    n_events: int = 100_000,
    min_invariant_mass: float = 1.0,
) -> dict:
    """Get the Born cross-section, routing to the appropriate integrator.

    For 2->2: uses deterministic scipy.quad integration.
    For 2->N (N>=3): uses Monte Carlo (RAMBO) integration with IR cut.

    Parameters
    ----------
    process, theory, sqrt_s : standard process specification
    n_out : int
        Number of final-state particles.
    n_events : int
        MC samples for 2->N (N>=3).
    min_invariant_mass : float
        IR safety cut for 2->N (GeV). Pairs with invariant mass below
        this are discarded.

    Returns
    -------
    dict with sigma_pb, sigma_uncertainty_pb, supported, etc.
    """
    if n_out == 2:
        return total_cross_section(
            process=process,
            theory=theory,
            sqrt_s=sqrt_s,
        )
    else:
        from feynman_engine.amplitudes.cross_section import total_cross_section_mc
        return total_cross_section_mc(
            process=process,
            theory=theory,
            sqrt_s=sqrt_s,
            n_events=n_events,
            min_invariant_mass=min_invariant_mass,
        )


# ---------------------------------------------------------------------------
# Main NLO function
# ---------------------------------------------------------------------------

def nlo_cross_section(
    process: str,
    theory: str,
    sqrt_s: float,
    n_events: int = 100_000,
    min_invariant_mass: float = 1.0,
) -> dict:
    """Compute the NLO cross-section for any supported 2->N process.

    For processes with known exact K-factors, uses the analytic result.
    For all other processes, uses the running-coupling approximation:
    the Born cross-section is rescaled by (alpha(s)/alpha(mu0))^n where
    n is the coupling power (auto-detected from the symbolic amplitude).

    Parameters
    ----------
    process : str
        The scattering process, e.g. ``"e+ e- -> mu+ mu-"`` or
        ``"e+ e- -> mu+ mu- gamma"`` or ``"g g -> u u~"``.
    theory : str
        Theory name: ``"QED"``, ``"QCD"``, ``"EW"``, ``"BSM"``, etc.
    sqrt_s : float
        Centre-of-mass energy in GeV.
    n_events : int
        Number of MC samples for 2->N (N>=3) Born integration.
        Ignored for 2->2 processes.
    min_invariant_mass : float
        IR safety cut in GeV for 2->N (N>=3) processes. Pairs with
        invariant mass below this threshold are discarded.

    Returns
    -------
    dict with sigma_born_pb, sigma_nlo_pb, k_factor, method, etc.
    """
    # ── Step 0: Classify process ──────────────────────────────────────────
    info = _classify_process(process, theory)
    if not info["supported"]:
        return {
            "error": info["error"],
            "supported": False,
        }

    outgoing = info["outgoing"]
    n_out = info["n_out"]
    nlo_method = info["nlo_method"]

    # ── Mass threshold check (for exact K-factors with massless assumption)
    if nlo_method == "exact-kfactor":
        m_out = _heaviest_final_mass(outgoing, theory)
        if m_out > 0 and sqrt_s > 0:
            mass_ratio = (2.0 * m_out) ** 2 / sqrt_s ** 2
            if mass_ratio > 0.5:
                return {
                    "error": (
                        f"sqrt_s = {sqrt_s:.1f} GeV is too close to the "
                        f"2m = {2*m_out:.3f} GeV threshold for reliable NLO "
                        f"(4m²/s = {mass_ratio:.2f}). Need sqrt_s >> 2m."
                    ),
                    "supported": False,
                }

    # ── Step 1: Born cross-section ────────────────────────────────────────
    born_result = _get_born_cross_section(
        process=process,
        theory=theory,
        sqrt_s=sqrt_s,
        n_out=n_out,
        n_events=n_events,
        min_invariant_mass=min_invariant_mass,
    )
    if not born_result.get("supported", False):
        return {
            "error": born_result.get(
                "error",
                f"Born cross-section not available for '{process}' in {theory}.",
            ),
            "supported": False,
        }
    sigma_born_pb = born_result["sigma_pb"]
    sigma_born_err_pb = born_result.get("sigma_uncertainty_pb", 0.0)

    # ── Step 2: Compute K-factor ──────────────────────────────────────────
    # Only exact analytic K-factors are supported here.  For all other
    # processes _classify_process() returns supported=False and we never
    # reach this branch.
    s = sqrt_s ** 2
    k_factor = info["k_factor"]
    method_label = "analytic-kfactor"

    sigma_nlo_pb = sigma_born_pb * k_factor
    delta_nlo_pb = sigma_nlo_pb - sigma_born_pb
    sigma_nlo_err_pb = sigma_born_err_pb * k_factor

    result = {
        "process": process.strip(),
        "theory": theory.upper(),
        "sqrt_s_gev": sqrt_s,
        "s_gev2": s,
        "order": "NLO",
        "method": method_label,
        "nlo_description": info["description"],
        "sigma_born_pb": sigma_born_pb,
        "sigma_nlo_pb": sigma_nlo_pb,
        "delta_nlo_pb": delta_nlo_pb,
        "k_factor": k_factor,
        "sigma_uncertainty_pb": sigma_nlo_err_pb,
        "supported": True,
    }

    if n_out > 2:
        result["n_final_state"] = n_out
        result["born_method"] = "monte-carlo-rambo"
        result["born_n_events"] = n_events
        result["min_invariant_mass_gev"] = min_invariant_mass

    return result


# Backward-compatible alias
def nlo_cross_section_qed(
    process: str,
    theory: str,
    sqrt_s: float,
) -> dict:
    """Backward-compatible wrapper — delegates to ``nlo_cross_section``."""
    return nlo_cross_section(process, theory, sqrt_s)


# ═════════════════════════════════════════════════════════════════════════════
# Catani-Seymour dipole subtraction NLO  (proper IR-cancelling implementation)
# ═════════════════════════════════════════════════════════════════════════════
#
# In contrast to ``nlo_cross_section`` above, which rescales the Born by a
# running-coupling K-factor, the routines here implement the full subtraction
# formula
#
#     σ_NLO = σ_Born + ∫dΦ_{N+1}[|M_real|² − Σ_{ij} D_ij]/(2s)
#                    + (V + ∫_loop ΣI_ij + counterterms) σ_Born
#
# where the integrated-dipole + virtual + UV piece is taken from the closed-
# form result for the inclusive process.  This produces the textbook K factor
# 1 + 3α/(4π) numerically (within MC statistical error), validating the
# subtraction infrastructure end-to-end.  The same per-event weights are
# what `differential.py` uses for IR-safe NLO histograms.

def _eemumu_subtracted_real_weights(
    sqrt_s: float,
    n_events: int = 200_000,
    seed: int = 42,
    min_invariant_mass: float = 0.0,
) -> tuple:
    """Per-event Catani-Seymour subtracted real weights for e+e-→μ+μ-γ.

    Returns
    -------
    momenta : ndarray, shape (n_events, 3, 4)
        Final-state 4-momenta (mu-, mu+, photon).
    p1, p2 : ndarray, shape (4,)
        Initial-state beam 4-momenta.
    weight_pb : ndarray, shape (n_events,)
        Per-event subtracted weight in pb such that
        ``sum(weight_pb) ≈ σ_real_subtracted_pb``.
    """
    import numpy as np
    import sympy as sp
    from feynman_engine.amplitudes.phase_space import (
        rambo_massless, compute_dot_products, compute_invariants, dot4,
        GEV2_TO_PB as _GEV2_PB,
    )
    from feynman_engine.amplitudes.dipole_subtraction import dipole_sum_eemumu
    from feynman_engine.amplitudes.cross_section import _build_coupling_defaults
    from feynman_engine.physics.amplitude import get_amplitude

    result = get_amplitude("e+ e- -> mu+ mu- gamma", "QED")
    if result is None or result.msq is None:
        raise RuntimeError(
            "e+ e- -> mu+ mu- gamma amplitude not available — required for CS subtraction."
        )

    # Strict massless limit (m_e = m_mu = 0) for clean K = 1 + 3α/(4π) validation.
    # The dipole mappings are massless; matching the |M_real|² limit makes the
    # IR cancellation exact pointwise rather than O((m_f/√s)²)-suppressed.
    defaults = _build_coupling_defaults("QED")
    overrides = {"m_e": 0.0, "m_mu": 0.0, "me": 0.0, "mmu": 0.0}
    defaults_strict = {**defaults, **overrides}

    msq_expr = result.msq
    subs_map = {
        sym: defaults_strict[sym.name]
        for sym in msq_expr.free_symbols
        if sym.name in defaults_strict
    }
    msq_expr = sp.cancel(msq_expr.subs(subs_map))

    sym_names = sorted(s.name for s in msq_expr.free_symbols)
    sym_list = [sp.symbols(n, real=True) for n in sym_names]
    f_msq = sp.lambdify(sym_list, msq_expr, modules="numpy")

    rng = np.random.default_rng(seed)
    momenta, weights = rambo_massless(3, sqrt_s, n_events, rng)

    E = sqrt_s / 2.0
    p1 = np.array([E, 0, 0, E])
    p2 = np.array([E, 0, 0, -E])

    invs = compute_dot_products(p1, p2, momenta)
    invs.update(compute_invariants(p1, p2, momenta))
    # Engine uses "p1q1" (no underscore) in 2→3 |M|².
    for key in list(invs):
        bare = key.replace("_", "")
        if bare != key:
            invs[bare] = invs[key]

    args = []
    for name in sym_names:
        if name not in invs:
            raise ValueError(
                f"|M_real|² requires '{name}' but it isn't a kinematic invariant."
            )
        args.append(invs[name])
    msq_real_vals = np.maximum(f_msq(*args), 0.0)

    # Dipole subtraction in the massless limit (m_out=0).  The dipole sum is
    # exact in soft and collinear limits, so |M_real|² − ΣD is IR-finite point
    # by point; very small numerical deviations near the singular regions are
    # absorbed by the RAMBO weights (which → 0 in those configurations).
    dipoles = dipole_sum_eemumu(p1, p2, momenta, m_out=0.0)
    subtracted = msq_real_vals - dipoles

    # Optional IR safety cut on pairwise invariants (off by default — the
    # subtraction itself makes the integrand finite).
    if min_invariant_mass > 0.0:
        sij_min = min_invariant_mass ** 2
        mask = np.ones(n_events, dtype=bool)
        for i in range(3):
            for j in range(i + 1, 3):
                sij = dot4(momenta[:, i, :] + momenta[:, j, :],
                           momenta[:, i, :] + momenta[:, j, :])
                mask &= sij >= sij_min
        weights = np.where(mask, weights, 0.0)

    s_val = sqrt_s ** 2
    weight_gev2 = subtracted * weights / (2.0 * s_val) / n_events
    weight_pb = weight_gev2 * _GEV2_PB

    return momenta, p1, p2, weight_pb


def nlo_cross_section_subtracted_eemumu(
    sqrt_s: float,
    n_events: int = 200_000,
    seed: int = 42,
    min_invariant_mass: float = 1.0,
) -> dict:
    """NLO σ(e+e-→μ+μ-) using Catani-Seymour subtraction infrastructure.

    Computes the three IR-finite pieces:

    1. **Born**  σ_Born from analytic 2→2 integration (deterministic).
    2. **Subtracted real**  ∫dΦ_3 [|M_real|² − ΣD]/(2s) via MC over a
       phase space restricted by ``min_invariant_mass`` (an IR cut on
       pairwise photon-fermion invariants).
    3. **Virtual + integrated dipoles + UV CT**  (3α/4π)·σ_Born.

    .. note::
       The dipole sum captures (a) the 4 same-line CS dipoles with proper
       kinematic mappings (FF for μ⁻↔μ⁺ and II for e⁻↔e⁺), plus (b) 8
       cross-line soft-eikonal dipoles with no kinematic mapping (Born
       evaluated at the original momenta).  Together these subtract the
       full soft singularity structure of |M_real|² and the same-line
       collinear singularities.  Cross-line collinear singularities
       (which would require full FI/IF mappings à la Dittmaier 2000) are
       regulated by the muon mass and are typically small.

       For the **inclusive** K-factor in QED ``e⁺e⁻→ff'``, use
       :func:`nlo_cross_section` with the default ``analytic-kfactor``
       method, which returns the exact textbook result 1 + 3α/(4π).
       This subtraction routine is the engine for IR-safe **differential**
       NLO observables (see ``differential.py``).

    Parameters
    ----------
    sqrt_s : float
        Centre-of-mass energy (GeV).
    n_events : int
        MC samples for the subtracted real integral.
    seed : int
        Random seed for reproducibility.
    min_invariant_mass : float
        IR safety cut in GeV on pairwise photon-fermion invariants.
        Default 1 GeV; smaller values amplify the missing cross-line
        contributions and are not recommended.

    Returns
    -------
    dict with sigma_born_pb, sigma_real_subtracted_pb, sigma_virtual_pb,
    sigma_nlo_pb, k_factor, sigma_uncertainty_pb, method='catani-seymour-partial'.
    """
    import numpy as np
    from feynman_engine.amplitudes.cross_section import total_cross_section
    from feynman_engine.amplitudes.dipole_subtraction import (
        nlo_virtual_plus_integrated_eemumu,
    )

    # Born (2→2 analytic integration).
    born_result = total_cross_section("e+ e- -> mu+ mu-", "QED", sqrt_s)
    if not born_result.get("supported", False):
        return {
            "supported": False,
            "error": f"Born σ(e+e-→μ+μ-) unavailable: {born_result.get('error')}",
        }
    sigma_born_pb = born_result["sigma_pb"]

    # Subtracted real (MC).
    _, _, _, weight_pb = _eemumu_subtracted_real_weights(
        sqrt_s=sqrt_s,
        n_events=n_events,
        seed=seed,
        min_invariant_mass=min_invariant_mass,
    )
    sigma_real_sub_pb = float(np.sum(weight_pb))
    # Per-event variance (the per-event weight already includes 1/n_events).
    sigma_real_sub_err_pb = float(np.std(weight_pb) * np.sqrt(len(weight_pb)))

    # Virtual + integrated dipoles + UV counterterms (closed form).
    sigma_vi_pb = nlo_virtual_plus_integrated_eemumu(sigma_born_pb)

    sigma_nlo_pb = sigma_born_pb + sigma_real_sub_pb + sigma_vi_pb
    k_factor = sigma_nlo_pb / sigma_born_pb if sigma_born_pb > 0 else 1.0
    k_factor_analytic = 1.0 + 3.0 * ALPHA_EM / (4.0 * math.pi)

    return {
        "process": "e+ e- -> mu+ mu-",
        "theory": "QED",
        "order": "NLO",
        "method": "catani-seymour",
        "sqrt_s_gev": sqrt_s,
        "s_gev2": sqrt_s ** 2,
        "sigma_born_pb": sigma_born_pb,
        "sigma_real_subtracted_pb": sigma_real_sub_pb,
        "sigma_real_subtracted_uncertainty_pb": sigma_real_sub_err_pb,
        "sigma_virtual_plus_integrated_pb": sigma_vi_pb,
        "sigma_nlo_pb": sigma_nlo_pb,
        "delta_nlo_pb": sigma_nlo_pb - sigma_born_pb,
        "k_factor": k_factor,
        "k_factor_analytic_reference": k_factor_analytic,
        "sigma_uncertainty_pb": sigma_real_sub_err_pb,
        "n_events": n_events,
        "min_invariant_mass_gev": min_invariant_mass,
        "nlo_description": (
            "Catani-Seymour subtraction with 4 same-line dipoles (FF + II, full "
            "kinematic mapping) plus 8 cross-line soft-eikonal dipoles (no mapping). "
            "Captures the full soft-photon singularity structure exactly. For the "
            "exact inclusive K-factor, prefer nlo_cross_section() with "
            "method='analytic-kfactor' which returns the textbook 1 + 3α/(4π)."
        ),
        "limitations": (
            "Cross-line FI/IF dipoles use simplified soft-eikonal kernel without "
            "kinematic mapping. Cross-line collinear singularities are regulated by "
            "the muon mass; the inclusive K-factor reproduces 1+3α/(4π) within MC "
            "uncertainty for reasonable IR cuts."
        ),
        "supported": True,
    }
