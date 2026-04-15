"""Cross-section integration via scipy.

Integrates the spin-averaged |M|² from get_amplitude() over solid angle
using full massive 2→2 kinematics:

    dσ/d(cosθ) = (p_f / (32π s · p_i)) × |M̄|²

where p_i, p_f are the initial/final CM momenta derived from the Källén
function λ(s, m_a², m_b²), and the Mandelstam variables are:

    t = m_a² + m_c² − 2(E_a E_c − p_i p_f cosθ)
    u = m_a² + m_d² − 2(E_a E_d + p_i p_f cosθ)

In the massless limit these reduce to the familiar t = −s/2(1−cosθ) etc.

The scipy.integrate.quad routine handles t-channel poles (Bhabha, Møller)
by backing off from cosθ = ±1 by a small cutoff ε.

Result is in pb (pico-barns).  Conversion: 1 GeV⁻² = 3.8938×10⁸ pb.
"""
from __future__ import annotations

import math
from typing import Optional

from sympy import lambdify, symbols, pi as sp_pi

# Physical constants
GEV2_TO_PB = 3.8938e8    # 1 GeV⁻² → pb  (PDG 2023)
ALPHA_EM   = 1.0 / 137.036
ALPHA_S    = 0.118
_E_EM      = math.sqrt(4 * math.pi * ALPHA_EM)   # electric charge |e|
_G_S       = math.sqrt(4 * math.pi * ALPHA_S)     # QCD strong coupling


from feynman_engine.amplitudes.pdg_masses import MASS_GEV as _MASS_GEV


def _build_coupling_defaults(theory: str) -> dict[str, float]:
    """Default coupling-constant substitution map for a given theory.

    Mass symbols (m_e, m_mu, …) are substituted with the particle mass in GeV,
    NOT mass-squared.  SymPy expressions use ``m_e**2`` when they mean m².
    """
    defaults: dict[str, float] = {
        "alpha":   ALPHA_EM,
        "alpha_s": ALPHA_S,
        "e":       _E_EM,
        "e_em":    _E_EM,
        "g_s":     _G_S,
        "g":       _G_S,
    }
    defaults.update(_MASS_GEV)
    return defaults


def _kallen(a: float, b: float, c: float) -> float:
    """Källén triangle function λ(a, b, c) = a² + b² + c² − 2(ab + ac + bc)."""
    return a * a + b * b + c * c - 2.0 * (a * b + a * c + b * c)


def _get_particle_masses(process: str, theory: str) -> tuple[float, float, float, float]:
    """Extract (m_a, m_b, m_c, m_d) in GeV for a 2→2 process.

    Uses PDG masses from the theory's particle registry when available,
    falling back to the symbol-name mass table.
    """
    from feynman_engine.physics.translator import parse_process
    from feynman_engine.physics.registry import TheoryRegistry

    spec = parse_process(process.strip(), theory.upper())
    particles = TheoryRegistry.get_particles(theory.upper())

    masses: list[float] = []
    for name in list(spec.incoming) + list(spec.outgoing):
        p = particles.get(name)
        if p is not None and p.mass_mev is not None:
            masses.append(float(p.mass_mev) / 1000.0)  # MeV → GeV
        elif p is not None and p.mass is not None:
            # mass is a symbol name like "m_e" — look it up
            masses.append(_MASS_GEV.get(p.mass, 0.0))
        else:
            masses.append(0.0)

    # Pad to 4 if fewer (shouldn't happen for 2→2)
    while len(masses) < 4:
        masses.append(0.0)
    return tuple(masses[:4])


def _has_tchannel_pole(msq_expr) -> bool:
    """Check if the amplitude contains a 1/t or t⁻ⁿ singularity."""
    from sympy import Symbol, Pow, Wild
    t_sym = None
    for sym in msq_expr.free_symbols:
        if sym.name == "t":
            t_sym = sym
            break
    if t_sym is None:
        return False
    # Detect 1/t or 1/t² in the expression
    for sub in msq_expr.atoms(Pow):
        if sub.base == t_sym and sub.exp < 0:
            return True
    return False


def _validate_cross_section_scope(process: str, theory: str):
    """Return ``None`` when the cross-section helpers can handle ``process``.

    The current implementation only supports 2->2 scattering with a genuine
    kinematic |M|^2 expression. Representative-point proxies are useful for the
    UI, but they should not be integrated as if they were full differential
    cross sections.
    """
    from feynman_engine.physics.amplitude import get_amplitude
    from feynman_engine.physics.translator import parse_process

    try:
        spec = parse_process(process.strip(), theory.upper())
    except ValueError as exc:
        return None, {"error": str(exc), "supported": False}

    if len(spec.incoming) != 2 or len(spec.outgoing) != 2:
        return None, {
            "error": (
                "Cross sections are currently only implemented for 2->2 scattering "
                f"processes; received {len(spec.incoming)}->{len(spec.outgoing)}."
            ),
            "supported": False,
        }

    result = get_amplitude(process.strip(), theory.upper())
    if result is None:
        return None, {
            "error": f"No amplitude available for '{process}' in {theory}.",
            "supported": False,
        }

    if getattr(result, "approximation_level", "") == "approximate-pointwise":
        return None, {
            "error": (
                "Only a representative-point |M|^2 proxy is available for this process, "
                "so a differential cross section cannot be integrated reliably yet."
            ),
            "supported": False,
        }

    return result, None


def _msq_to_callable(
    msq_expr,
    s_sym,
    cos_sym,
    coupling_vals: dict,
    masses: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
):
    """Convert SymPy |M̄|² to a fast numpy callable f(s_val, cos_theta).

    Performs:
        1. Substitute coupling constants and masses by name.
        2. Replace t and u with full massive 2→2 kinematics:
               t = m_a² + m_c² − 2(E_a E_c − p_i p_f cosθ)
               u = m_a² + m_d² − 2(E_a E_d + p_i p_f cosθ)
           In the massless limit these reduce to t = −s/2(1−cosθ) etc.
        3. lambdify the result over (s, cos_theta).

    Parameters
    ----------
    masses : tuple of 4 floats
        (m_a, m_b, m_c, m_d) in GeV — incoming a,b and outgoing c,d.
    """
    from sympy import Symbol, sqrt as sp_sqrt

    # Substitute coupling constants / masses by symbol name
    subs_map: dict = {}
    for sym in list(msq_expr.free_symbols):
        name = sym.name
        if name in coupling_vals:
            subs_map[sym] = coupling_vals[name]

    expr = msq_expr.subs(subs_map)

    m_a, m_b, m_c, m_d = masses
    is_massless = all(m < 1e-10 for m in masses)

    if is_massless:
        # Fast path: massless kinematics (no square roots needed)
        t_sub = -(s_sym / 2) * (1 - cos_sym)
        u_sub = -(s_sym / 2) * (1 + cos_sym)
    else:
        # Full massive 2→2 CM kinematics via Källén function
        ma2, mb2, mc2, md2 = m_a**2, m_b**2, m_c**2, m_d**2
        # CM momenta: p = λ^{1/2}(s, m₁², m₂²) / (2√s)
        lam_i = s_sym**2 + ma2**2 + mb2**2 - 2*s_sym*ma2 - 2*s_sym*mb2 - 2*ma2*mb2
        lam_f = s_sym**2 + mc2**2 + md2**2 - 2*s_sym*mc2 - 2*s_sym*md2 - 2*mc2*md2
        p_i = sp_sqrt(lam_i) / (2 * sp_sqrt(s_sym))
        p_f = sp_sqrt(lam_f) / (2 * sp_sqrt(s_sym))
        # CM energies: E_x = (s + m_x² − m_partner²) / (2√s)
        E_a = (s_sym + ma2 - mb2) / (2 * sp_sqrt(s_sym))
        E_c = (s_sym + mc2 - md2) / (2 * sp_sqrt(s_sym))
        E_d = (s_sym + md2 - mc2) / (2 * sp_sqrt(s_sym))
        # Mandelstam variables
        t_sub = ma2 + mc2 - 2 * (E_a * E_c - p_i * p_f * cos_sym)
        u_sub = ma2 + md2 - 2 * (E_a * E_d + p_i * p_f * cos_sym)

    t_sym_in_expr = None
    u_sym_in_expr = None
    for sym in expr.free_symbols:
        if sym.name == "t":
            t_sym_in_expr = sym
        if sym.name == "u":
            u_sym_in_expr = sym

    if t_sym_in_expr is not None:
        expr = expr.subs(t_sym_in_expr, t_sub)
    if u_sym_in_expr is not None:
        expr = expr.subs(u_sym_in_expr, u_sub)

    return lambdify([s_sym, cos_sym], expr, modules="numpy")


def _phase_space_factor(s_val: float, masses: tuple[float, float, float, float]) -> float:
    """Phase-space prefactor p_f / (32π s p_i) for massive 2→2 kinematics.

    Returns 1/(32πs) in the massless limit.
    """
    m_a, m_b, m_c, m_d = masses
    lam_i = _kallen(s_val, m_a**2, m_b**2)
    lam_f = _kallen(s_val, m_c**2, m_d**2)
    if lam_i <= 0.0 or lam_f < 0.0:
        return 0.0
    p_i = math.sqrt(lam_i) / (2.0 * math.sqrt(s_val))
    p_f = math.sqrt(lam_f) / (2.0 * math.sqrt(s_val))
    return p_f / (32.0 * math.pi * s_val * p_i)


def differential_cross_section(
    process: str,
    theory: str,
    s_val: float,
    cos_theta: float,
    coupling_vals: Optional[dict] = None,
) -> Optional[float]:
    """Compute dσ/d(cosθ) in pb at given s [GeV²] and cosθ.

    Uses full massive 2→2 kinematics when particle masses are non-zero.

    Parameters
    ----------
    process : str
        Scattering process, e.g. ``"e+ e- -> mu+ mu-"``.
    theory : str
        Theory name: ``"QED"``, ``"QCD"``, ``"EW"``, ``"BSM"``.
    s_val : float
        Mandelstam s in GeV².
    cos_theta : float
        Scattering angle.
    coupling_vals : dict, optional
        Override default coupling constants.  Keys are symbol names
        (``"alpha"``, ``"e"``, ``"alpha_s"``, etc.), values are floats.

    Returns
    -------
    float or None
        dσ/d(cosθ) in pb, or None if the amplitude is unavailable.
    """
    result, error = _validate_cross_section_scope(process, theory)
    if error is not None:
        return None

    msq_expr = result.msq
    if msq_expr is None:
        return None

    defaults = _build_coupling_defaults(theory)
    if coupling_vals:
        defaults.update(coupling_vals)

    try:
        masses = _get_particle_masses(process, theory)
    except Exception:
        masses = (0.0, 0.0, 0.0, 0.0)

    s_sym   = symbols("s", real=True)
    cos_sym = symbols("cos_theta", real=True)

    try:
        f = _msq_to_callable(msq_expr, s_sym, cos_sym, defaults, masses=masses)
        msq_val = float(f(s_val, cos_theta))
    except Exception:
        return None

    if not math.isfinite(msq_val) or msq_val < 0.0:
        return None

    ps = _phase_space_factor(s_val, masses)
    dsigma = msq_val * ps   # GeV⁻²
    return dsigma * GEV2_TO_PB


def total_cross_section(
    process: str,
    theory: str,
    sqrt_s: float,
    coupling_vals: Optional[dict] = None,
    cos_theta_min: float = -1.0,
    cos_theta_max: float =  1.0,
    eps: float = 1e-3,
) -> dict:
    """Integrate dσ/d(cosθ) over cosθ to obtain total σ.

    Parameters
    ----------
    process : str
        Scattering process string.
    theory : str
        Theory name.
    sqrt_s : float
        Centre-of-mass energy √s in GeV.
    coupling_vals : dict, optional
        Coupling overrides.
    cos_theta_min, cos_theta_max : float
        Integration range (default full range −1 to 1).
    eps : float
        Small cutoff pulled in from each endpoint to handle t-channel poles.
        Default 1e-3 (safe for Bhabha / Møller).

    Returns
    -------
    dict with keys:
        process, theory, sqrt_s_gev, s_gev2,
        sigma_pb, sigma_uncertainty_pb,
        dsigma_at_cos0_pb,  (dσ/d(cosθ) at cosθ=0)
        has_tchannel_pole, cos_theta_range, converged, formula_latex
    """
    from scipy.integrate import quad

    s_val = sqrt_s ** 2
    result, error = _validate_cross_section_scope(process, theory)
    if error is not None:
        return error

    msq_expr = result.msq
    if msq_expr is None:
        return {"error": "Amplitude has no closed-form |M̄|² expression.", "supported": False}

    defaults = _build_coupling_defaults(theory)
    if coupling_vals:
        defaults.update(coupling_vals)

    try:
        masses = _get_particle_masses(process, theory)
    except Exception:
        masses = (0.0, 0.0, 0.0, 0.0)

    # Check threshold: √s must exceed sum of final-state masses
    m_c, m_d = masses[2], masses[3]
    if sqrt_s <= m_c + m_d:
        return {
            "error": (
                f"√s = {sqrt_s:.3f} GeV is below the production threshold "
                f"({m_c + m_d:.3f} GeV) for this process."
            ),
            "supported": False,
        }

    s_sym   = symbols("s", real=True)
    cos_sym = symbols("cos_theta", real=True)

    try:
        f_msq = _msq_to_callable(msq_expr, s_sym, cos_sym, defaults, masses=masses)
    except Exception as exc:
        return {"error": f"Failed to lambdify amplitude: {exc}", "supported": False}

    has_pole = _has_tchannel_pole(msq_expr)
    ps_factor = _phase_space_factor(s_val, masses)
    is_massive = any(m > 1e-10 for m in masses)

    # Define integrand: dσ/d(cosθ) = (p_f / 32πs·p_i) × |M̄|²
    def integrand(ct: float) -> float:
        try:
            val = float(f_msq(s_val, ct))
            if not math.isfinite(val):
                return 0.0
            return max(val, 0.0) * ps_factor
        except Exception:
            return 0.0

    lo = cos_theta_min + eps
    hi = cos_theta_max - eps

    try:
        sigma_gev2, err_gev2 = quad(integrand, lo, hi, limit=200, epsabs=1e-10, epsrel=1e-8)
        converged = True
    except Exception:
        sigma_gev2, err_gev2 = 0.0, float("nan")
        converged = False

    sigma_pb = max(sigma_gev2, 0.0) * GEV2_TO_PB
    err_pb   = abs(err_gev2) * GEV2_TO_PB

    # dσ/d(cosθ) at cosθ = 0
    try:
        dsigma_cos0 = float(integrand(0.0)) * GEV2_TO_PB
    except Exception:
        dsigma_cos0 = None

    if is_massive:
        formula_latex = (
            r"\sigma = \int_{-1+\varepsilon}^{1-\varepsilon} "
            r"\frac{p_f}{32\pi s\, p_i}\,"
            r"|\overline{\mathcal{M}}|^2 \,\mathrm{d}(\cos\theta)"
        )
    else:
        formula_latex = (
            r"\sigma = \int_{-1+\varepsilon}^{1-\varepsilon} "
            r"\frac{|\overline{\mathcal{M}}|^2}{32\pi s} \,\mathrm{d}(\cos\theta)"
        )

    return {
        "process":               process.strip(),
        "theory":                theory.upper(),
        "sqrt_s_gev":            sqrt_s,
        "s_gev2":                s_val,
        "sigma_pb":              sigma_pb,
        "sigma_uncertainty_pb":  err_pb,
        "dsigma_at_cos0_pb":     dsigma_cos0,
        "has_tchannel_pole":     has_pole,
        "cos_theta_range":       [lo, hi],
        "eps":                   eps,
        "converged":             converged,
        "formula_latex":         formula_latex,
        "massive_kinematics":    is_massive,
        "masses_gev":            list(masses),
        "supported":             True,
    }


def total_cross_section_mc(
    process: str,
    theory: str,
    sqrt_s: float,
    coupling_vals: Optional[dict] = None,
    n_events: int = 100_000,
    seed: int = 42,
    min_invariant_mass: float = 0.0,
) -> dict:
    """Monte Carlo cross section for 2→N processes (N ≥ 2).

    Uses RAMBO phase space sampling and evaluates |M̄|² numerically at
    each phase space point.  Works for any multiplicity.

    Parameters
    ----------
    process : str
        Scattering process string, e.g. ``"e+ e- -> mu+ mu- gamma"``.
    theory : str
        Theory name.
    sqrt_s : float
        Centre-of-mass energy √s in GeV.
    coupling_vals : dict, optional
        Coupling overrides.
    n_events : int
        Number of Monte Carlo samples.
    seed : int
        Random seed.
    min_invariant_mass : float
        Minimum invariant mass (GeV) for any pair of final-state particles.
        Events where any s_{ij} < min_invariant_mass² are discarded.
        Required for IR-safe cross sections in processes with massless
        radiation (e.g. e+e-→μ+μ-γ).

    Returns
    -------
    dict with sigma_pb, sigma_uncertainty_pb, etc.
    """
    import numpy as np
    from feynman_engine.amplitudes.phase_space import (
        rambo_massless, compute_dot_products, GEV2_TO_PB as _GEV2_PB,
    )
    from feynman_engine.physics.amplitude import get_amplitude
    from feynman_engine.physics.translator import parse_process

    try:
        spec = parse_process(process.strip(), theory.upper())
    except ValueError as exc:
        return {"error": str(exc), "supported": False}

    n_in = len(spec.incoming)
    n_out = len(spec.outgoing)
    if n_in != 2:
        return {"error": "Only 2→N processes are supported.", "supported": False}

    result = get_amplitude(process.strip(), theory.upper())
    if result is None or result.msq is None:
        return {
            "error": f"No amplitude available for '{process}' in {theory}.",
            "supported": False,
        }

    msq_expr = result.msq

    defaults = _build_coupling_defaults(theory)
    if coupling_vals:
        defaults.update(coupling_vals)

    # Substitute coupling constants by symbol name.
    subs_map = {}
    for sym in list(msq_expr.free_symbols):
        name = sym.name
        if name in defaults:
            subs_map[sym] = defaults[name]
    msq_substituted = msq_expr.subs(subs_map)

    # Build a numerical evaluator.
    # Identify which invariant symbols remain (s, t, u for 2→2; s, s12, ... for 2→3).
    remaining = msq_substituted.free_symbols
    if not remaining:
        # Constant matrix element — just a number.
        msq_const = float(msq_substituted)

        def msq_func(p1, p2, momenta):
            return np.full(momenta.shape[0], msq_const)
    else:
        sym_names = sorted(s.name for s in remaining)
        sym_list = [symbols(n, real=True) for n in sym_names]
        f_msq = lambdify(sym_list, msq_substituted, modules="numpy")

        def msq_func(p1, p2, momenta):
            invs = compute_dot_products(p1, p2, momenta)
            # Also compute standard invariants.
            from feynman_engine.amplitudes.phase_space import compute_invariants
            std_invs = compute_invariants(p1, p2, momenta)
            invs.update(std_invs)
            # Dot-product keys use underscores ("p1_q1") but 2→3 SymPy
            # symbols omit them ("p1q1").  Add underscore-free aliases.
            for key in list(invs):
                bare = key.replace("_", "")
                if bare != key:
                    invs[bare] = invs[key]
            args = []
            for name in sym_names:
                if name in invs:
                    args.append(invs[name])
                else:
                    raise ValueError(
                        f"Symbol '{name}' in |M|² not found in kinematic invariants."
                    )
            vals = f_msq(*args)
            # Clamp negative values to zero (numerical noise).
            return np.maximum(vals, 0.0)

    s_val = sqrt_s ** 2
    rng = np.random.default_rng(seed)
    momenta, weights = rambo_massless(n_out, sqrt_s, n_events, rng)

    E_beam = sqrt_s / 2
    p1 = np.array([E_beam, 0, 0, E_beam])
    p2 = np.array([E_beam, 0, 0, -E_beam])

    # Apply invariant mass cut for IR safety.
    if min_invariant_mass > 0.0 and n_out >= 3:
        from feynman_engine.amplitudes.phase_space import dot4
        sij_min = min_invariant_mass ** 2
        mask = np.ones(n_events, dtype=bool)
        for i in range(n_out):
            for j in range(i + 1, n_out):
                qi = momenta[:, i, :]
                qj = momenta[:, j, :]
                sij = dot4(qi + qj, qi + qj)
                mask &= sij >= sij_min
        # Also cut on (incoming, outgoing) pairs for t-channel poles.
        for i in range(n_out):
            qi = momenta[:, i, :]
            t1 = dot4(p1[np.newaxis, :] - qi, p1[np.newaxis, :] - qi)
            t2 = dot4(p2[np.newaxis, :] - qi, p2[np.newaxis, :] - qi)
            mask &= np.abs(t1) >= sij_min
            mask &= np.abs(t2) >= sij_min
        # Zero out events failing the cut (keeps RAMBO weight normalization correct).
        cut_weights = np.where(mask, weights, 0.0)
    else:
        cut_weights = weights

    msq_vals = msq_func(p1, p2, momenta)
    integrand = msq_vals * cut_weights / (2 * s_val)

    sigma_gev2 = float(np.mean(integrand))
    sigma_err_gev2 = float(np.std(integrand) / np.sqrt(n_events))

    sigma_pb = sigma_gev2 * _GEV2_PB
    sigma_err_pb = sigma_err_gev2 * _GEV2_PB

    n_passed = int(np.sum(cut_weights > 0)) if min_invariant_mass > 0.0 else n_events

    return {
        "process": process.strip(),
        "theory": theory.upper(),
        "sqrt_s_gev": sqrt_s,
        "s_gev2": s_val,
        "sigma_pb": sigma_pb,
        "sigma_uncertainty_pb": sigma_err_pb,
        "n_events": n_events,
        "n_passed_cut": n_passed,
        "min_invariant_mass_gev": min_invariant_mass,
        "method": "monte-carlo-rambo",
        "converged": True,
        "supported": True,
    }


def total_cross_section_vegas(
    process: str,
    theory: str,
    sqrt_s: float,
    coupling_vals: Optional[dict] = None,
    n_iter: int = 10,
    n_eval_per_iter: int = 50_000,
    n_adapt: int = 5,
    n_bins: int = 50,
    seed: int = 42,
    min_invariant_mass: float = 0.0,
) -> dict:
    """Vegas adaptive MC cross section for 2→N processes (N ≥ 2).

    Uses the VEGAS importance-sampling algorithm on top of RAMBO phase-space
    generation.  The VEGAS grid learns the peaked structure of |M|² (t-channel
    poles, resonances) and concentrates samples where the integrand is largest.

    For 2→2 processes, this is usually not needed (scipy.quad is faster and
    more precise).  For 2→3+ processes with sharp kinematic features this
    converges significantly faster than flat RAMBO sampling.

    Parameters
    ----------
    process : str
        Scattering process string, e.g. ``"e+ e- -> mu+ mu- gamma"``.
    theory : str
        Theory name.
    sqrt_s : float
        Centre-of-mass energy √s in GeV.
    coupling_vals : dict, optional
        Coupling overrides.
    n_iter : int
        Total Vegas iterations (adaptation + accumulation).
    n_eval_per_iter : int
        Integrand evaluations per iteration.
    n_adapt : int
        Number of initial adaptation-only iterations.
    n_bins : int
        Bins per axis in the Vegas grid.
    seed : int
        Random seed.
    min_invariant_mass : float
        Minimum invariant mass cut (GeV) for IR safety.

    Returns
    -------
    dict with sigma_pb, sigma_uncertainty_pb, chi2_per_dof, etc.
    """
    import numpy as np
    from feynman_engine.amplitudes.phase_space import (
        rambo_massless, compute_dot_products, vegas_integrate,
        dot4, GEV2_TO_PB as _GEV2_PB,
    )
    from feynman_engine.physics.amplitude import get_amplitude
    from feynman_engine.physics.translator import parse_process

    try:
        spec = parse_process(process.strip(), theory.upper())
    except ValueError as exc:
        return {"error": str(exc), "supported": False}

    n_in = len(spec.incoming)
    n_out = len(spec.outgoing)
    if n_in != 2:
        return {"error": "Only 2→N processes are supported.", "supported": False}

    result = get_amplitude(process.strip(), theory.upper())
    if result is None or result.msq is None:
        return {
            "error": f"No amplitude available for '{process}' in {theory}.",
            "supported": False,
        }

    msq_expr = result.msq

    defaults = _build_coupling_defaults(theory)
    if coupling_vals:
        defaults.update(coupling_vals)

    # Substitute coupling constants by symbol name.
    subs_map = {}
    for sym in list(msq_expr.free_symbols):
        name = sym.name
        if name in defaults:
            subs_map[sym] = defaults[name]
    msq_substituted = msq_expr.subs(subs_map)

    # Build a numerical evaluator.
    remaining = msq_substituted.free_symbols
    if not remaining:
        msq_const = float(msq_substituted)

        def msq_func(p1, p2, momenta):
            return np.full(momenta.shape[0], msq_const)
    else:
        sym_names = sorted(s.name for s in remaining)
        sym_list = [symbols(n, real=True) for n in sym_names]
        f_msq = lambdify(sym_list, msq_substituted, modules="numpy")

        def msq_func(p1, p2, momenta):
            from feynman_engine.amplitudes.phase_space import (
                compute_dot_products, compute_invariants,
            )
            invs = compute_dot_products(p1, p2, momenta)
            std_invs = compute_invariants(p1, p2, momenta)
            invs.update(std_invs)
            for key in list(invs):
                bare = key.replace("_", "")
                if bare != key:
                    invs[bare] = invs[key]
            args = []
            for name in sym_names:
                if name in invs:
                    args.append(invs[name])
                else:
                    raise ValueError(
                        f"Symbol '{name}' in |M|² not found in kinematic invariants."
                    )
            vals = f_msq(*args)
            return np.maximum(vals, 0.0)

    s_val = sqrt_s ** 2
    E_beam = sqrt_s / 2
    p1 = np.array([E_beam, 0, 0, E_beam])
    p2 = np.array([E_beam, 0, 0, -E_beam])

    sij_min = min_invariant_mass ** 2 if min_invariant_mass > 0.0 else 0.0

    # RAMBO uses 4*n_out random numbers per event, but they're correlated —
    # the effective dimensionality is 3*n_out - 4 (massless N-body phase space).
    # We use 4*n_out as the VEGAS dimension since that's what rambo_massless
    # consumes (4 randoms per particle: rho1..rho4).
    n_rambo_randoms = 4 * n_out

    # Precompute the RAMBO phase-space volume (same for all events, massless).
    import math as _math
    log_phase_vol = ((n_out - 1) * _math.log(_math.pi / 2)
                     + (n_out - 2) * _math.log(s_val)
                     - (3 * n_out - 4) * _math.log(2 * _math.pi)
                     - _math.lgamma(n_out) - _math.lgamma(n_out - 1))
    phase_vol = _math.exp(log_phase_vol)

    def _vegas_integrand(random_block: np.ndarray) -> np.ndarray:
        """Map VEGAS-adapted random numbers → RAMBO momenta → σ integrand.

        Parameters
        ----------
        random_block : ndarray, shape (n_events, n_rambo_randoms)
            Random numbers in [0,1], adapted by the VEGAS grid.

        Returns
        -------
        ndarray, shape (n_events,)
            The integrand |M|² × phase_vol / (2s) for each event.
        """
        n_ev = random_block.shape[0]

        # Reshape into the 4 RAMBO randoms per particle.
        rho = random_block.reshape(n_ev, n_out, 4)
        rho1 = rho[:, :, 0]
        rho2 = rho[:, :, 1]
        rho3 = rho[:, :, 2]
        rho4 = rho[:, :, 3]

        # Clip to avoid log(0) or division by zero
        rho1 = np.clip(rho1, 1e-15, 1 - 1e-15)
        rho4 = np.clip(rho4, 1e-15, 1 - 1e-15)
        rho2 = np.clip(rho2, 1e-15, 1 - 1e-15)

        cos_theta = 2 * rho2 - 1
        sin_theta = np.sqrt(np.maximum(1 - cos_theta ** 2, 0.0))
        phi = 2 * np.pi * rho3

        E_q = -np.log(rho1 * rho4 + 1e-300)

        q = np.zeros((n_ev, n_out, 4))
        q[:, :, 0] = E_q
        q[:, :, 1] = E_q * sin_theta * np.cos(phi)
        q[:, :, 2] = E_q * sin_theta * np.sin(phi)
        q[:, :, 3] = E_q * cos_theta

        # RAMBO: boost to CM frame and rescale
        Q = np.sum(q, axis=1)
        Q_mass = np.sqrt(np.abs(
            Q[:, 0] ** 2 - Q[:, 1] ** 2 - Q[:, 2] ** 2 - Q[:, 3] ** 2
        ))
        Q_mass = np.maximum(Q_mass, 1e-30)

        b = -Q[:, 1:4] / Q[:, 0:1]
        b_sq = np.sum(b ** 2, axis=1)
        gamma = Q[:, 0] / Q_mass

        p = np.zeros_like(q)
        for i in range(n_out):
            q_i = q[:, i, :]
            b_dot_q = np.sum(b * q_i[:, 1:4], axis=1)
            factor = np.where(
                b_sq > 1e-30,
                (gamma - 1) * b_dot_q / b_sq + gamma * q_i[:, 0],
                q_i[:, 0],
            )
            p[:, i, 0] = gamma * (q_i[:, 0] + b_dot_q)
            p[:, i, 1] = q_i[:, 1] + b[:, 0] * factor
            p[:, i, 2] = q_i[:, 2] + b[:, 1] * factor
            p[:, i, 3] = q_i[:, 3] + b[:, 2] * factor

        x_scale = sqrt_s / Q_mass
        p *= x_scale[:, np.newaxis, np.newaxis]

        # Apply invariant mass cut
        mask = np.ones(n_ev, dtype=bool)
        if sij_min > 0.0 and n_out >= 3:
            for i in range(n_out):
                for j in range(i + 1, n_out):
                    qi = p[:, i, :]
                    qj = p[:, j, :]
                    sij = dot4(qi + qj, qi + qj)
                    mask &= sij >= sij_min
            for i in range(n_out):
                qi = p[:, i, :]
                t1 = dot4(p1[np.newaxis, :] - qi, p1[np.newaxis, :] - qi)
                t2 = dot4(p2[np.newaxis, :] - qi, p2[np.newaxis, :] - qi)
                mask &= np.abs(t1) >= sij_min
                mask &= np.abs(t2) >= sij_min

        # Evaluate |M|²
        msq_vals = msq_func(p1, p2, p)

        # σ integrand = |M|² × phase_vol / (2s), zeroed where cut fails
        integrand_vals = np.where(mask, msq_vals * phase_vol / (2 * s_val), 0.0)
        return integrand_vals

    # Run VEGAS
    vresult = vegas_integrate(
        integrand=_vegas_integrand,
        n_dim=n_rambo_randoms,
        n_iter=n_iter,
        n_eval_per_iter=n_eval_per_iter,
        n_bins=n_bins,
        alpha=1.5,
        n_adapt=n_adapt,
        seed=seed,
    )

    sigma_gev2 = vresult["integral"]
    error_gev2 = vresult["error"]
    sigma_pb = sigma_gev2 * _GEV2_PB
    error_pb = error_gev2 * _GEV2_PB

    return {
        "process": process.strip(),
        "theory": theory.upper(),
        "sqrt_s_gev": sqrt_s,
        "s_gev2": s_val,
        "sigma_pb": sigma_pb,
        "sigma_uncertainty_pb": error_pb,
        "chi2_per_dof": vresult["chi2_per_dof"],
        "n_eval": vresult["n_eval"],
        "n_iter": n_iter,
        "n_adapt": n_adapt,
        "n_eval_per_iter": n_eval_per_iter,
        "min_invariant_mass_gev": min_invariant_mass,
        "method": "vegas-adaptive",
        "converged": vresult["converged"],
        "supported": True,
    }
