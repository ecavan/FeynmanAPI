"""Generic NLO virtuals via OpenLoops 2.

This module provides a thin path to NLO K-factors for *any* process whose
OpenLoops process library is installed.  OpenLoops returns the spin-summed,
colour-summed pair (|M_tree|², 2·Re(M*_tree · M_loop)) at a phase-space
point in dimensional regularisation, with the IR poles delivered separately
so we can verify cancellation against the universal Catani-Seymour I-operator.

What this module computes
-------------------------
The NLO virtual K-factor evaluated at a *fixed* phase-space point at √s::

    K_virt(√s) = 1  +  (α/(2π)) · loop_finite / tree

This is the leading piece of a full NLO correction — it is exact for the
virtual contribution but *misses* the integrated dipoles and the genuine
real-emission piece needed for an IR-safe physical NLO σ.  Use it as:

- a precision cross-check on the running-coupling NLO K-factor for any
  process where the latter is the only fallback;
- a starting point for a future full NLO integrator (real + virtual +
  PDF counterterms + dipole subtraction).

Result is flagged ``method = "openloops-virtual-only"`` and
``trust_level = "approximate"`` so callers know it is not a complete NLO.
For the LHC channels where we have a tabulated K-factor (pp→tt̄, pp→H,
pp→DY, pp→ZZ, pp→ZH, pp→Hjj, …) the tabulated value still wins.

References
----------
Buccioni, Lang, Lindert, Maierhöfer, Pozzorini, Zhang, Zoller,
"OpenLoops 2," EPJ C 79 (2019) 866, arXiv:1907.13071.
"""
from __future__ import annotations

import math

from feynman_engine.amplitudes.cross_section import ALPHA_EM, ALPHA_S


def _coupling_for_theory(theory: str) -> tuple[str, float]:
    """Return (coupling-name, value) for the renormalisation-scheme expansion."""
    t = theory.upper()
    if t in ("QCD", "QCDQED"):
        return "alpha_s", ALPHA_S
    if t in ("QED", "EW"):
        return "alpha_em", ALPHA_EM
    return "alpha_s", ALPHA_S


def virtual_k_factor_openloops(
    process: str,
    sqrt_s_gev: float,
    theory: str = "QCD",
) -> dict:
    """Evaluate the NLO virtual K-factor for ``process`` via OpenLoops.

    Returns the pair (tree, 2·Re tree⋆ loop)/tree at a single random
    phase-space point at √s, plus the K-factor::

        K_virt = 1 + (α/(2π)) · loop_finite/tree

    Caller is responsible for combining this with a Born σ to get a virtual
    NLO σ; the result is *not* a physical full NLO.

    Parameters
    ----------
    process : str
        Engine process string, e.g. ``"u u~ -> e+ e-"``.  Translated to
        OpenLoops PDG via ``openloops_bridge.to_pdg_string``.
    sqrt_s_gev : float
        Centre-of-mass energy in GeV.
    theory : str
        Used only to pick the expansion coupling (α_s for QCD, α_em for QED/EW).

    Returns
    -------
    dict
        ``tree``, ``loop_finite``, ``loop_ir1``, ``loop_ir2``, ``k_factor``,
        ``coupling``, ``method``, ``trust_level``, ``reference``.
    """
    from feynman_engine.amplitudes.openloops_bridge import (
        evaluate_loop_squared,
        is_available,
    )

    if not is_available():
        return {
            "supported": False,
            "error": (
                "OpenLoops bindings unavailable.  Run "
                "`feynman install-openloops` and `feynman install-process <name>` "
                "for the relevant process library."
            ),
        }

    raw = evaluate_loop_squared(process, float(sqrt_s_gev))
    coupling_name, alpha = _coupling_for_theory(theory)

    if raw["tree"] == 0.0:
        # Pure loop-induced process (gg→H, gg→ZZ): no Born from tree —
        # the K-factor concept doesn't apply, return raw pieces only.
        return {
            "supported": True,
            "process": process.strip(),
            "theory": theory.upper(),
            "sqrt_s_gev": float(sqrt_s_gev),
            "tree": raw["tree"],
            "loop_finite": raw["loop_finite"],
            "loop_ir1": raw["loop_ir1"],
            "loop_ir2": raw["loop_ir2"],
            "k_factor": None,
            "coupling": coupling_name,
            "method": "openloops-loop-induced",
            "trust_level": "approximate",
            "reference": "OpenLoops 2 (Buccioni et al., arXiv:1907.13071)",
            "accuracy_caveat": (
                "Loop-induced process: tree amplitude is zero, so the "
                "K-factor formalism does not apply.  Quoted loop² is the "
                "leading contribution; for a hadronic σ wrap with dipole "
                "subtraction + PDF convolution."
            ),
        }

    # K_virt = 1 + (α/(2π)) · 2·Re(M*_tree · M_loop) / |M_tree|²
    k_virt = 1.0 + (alpha / (2.0 * math.pi)) * raw["loop_finite"] / raw["tree"]

    return {
        "supported": True,
        "process": process.strip(),
        "theory": theory.upper(),
        "sqrt_s_gev": float(sqrt_s_gev),
        "tree": raw["tree"],
        "loop_finite": raw["loop_finite"],
        "loop_ir1": raw["loop_ir1"],
        "loop_ir2": raw["loop_ir2"],
        "k_factor": k_virt,
        "coupling": coupling_name,
        "alpha": alpha,
        "method": "openloops-virtual-only",
        "trust_level": "approximate",
        "reference": "OpenLoops 2 (Buccioni et al., EPJ C 79 (2019) 866)",
        "accuracy_caveat": (
            "Virtual-only NLO K-factor at a single random phase-space point. "
            "Missing integrated Catani-Seymour dipoles + real-emission piece + "
            "PDF counterterms.  Use only as a cross-check; for hadron-collider "
            "predictions prefer the tabulated K-factors in nlo_k_factors.py."
        ),
    }
