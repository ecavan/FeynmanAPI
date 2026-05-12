"""OpenLoops-backed amplitude factory — Tier 4 fallback in get_amplitude.

When curated/FORM/SymPy backends all return None for a process, this module
checks whether OpenLoops has a process library installed for it.  If so,
returns an :class:`AmplitudeResult` with::

    backend = "openloops"
    approximation_level = "openloops-numerical"
    msq = None  (no symbolic expression)
    msq_latex = "<numerical OpenLoops evaluator>"
    evaluation_point = {"sqrt_s_gev": ..., "sigma_at_point_pb": ..., ...}

The :func:`feynman_engine.amplitudes.cross_section.total_cross_section` path
recognises ``backend="openloops"`` and routes σ_LO through OL Born + RAMBO
instead of the symbolic SymPy lambdify path.
"""
from __future__ import annotations

from typing import Optional

from feynman_engine.amplitudes.types import AmplitudeResult


def infer_leading_orders(process: str) -> tuple[int, int]:
    """Infer the leading-order (qcd, ew) coupling powers for a process.

    Used by both ``get_openloops_amplitude`` and the cross-section OL
    fallback to pick OL's lowest-order tree (not loop² or higher EW
    orders that may be present in `*_ew` libraries).

    Heuristic based on incoming/outgoing particle types:
      - charged-lepton initial → ew=2 (+1 per radiated γ)
      - gg initial → qcd=2
      - qq̄ initial → qcd=2 (QCD final) or ew=2 (EW final)
    """
    from feynman_engine.amplitudes.openloops_bridge import to_pdg_string
    try:
        pdg = to_pdg_string(process)
    except Exception:
        return (-1, -1)
    if "->" not in pdg:
        return (-1, -1)
    incoming = pdg.split("->", 1)[0].split()
    outgoing = pdg.split("->", 1)[1].split()
    n_g_in = sum(1 for p in incoming if p == "21")
    has_charged_lep_in = any(p in {"11","-11","13","-13","15","-15"} for p in incoming)
    has_gamma_out = any(p == "22" for p in outgoing)
    has_ew_boson_out = any(p in {"22","23","24","-24","25"} for p in outgoing)
    if has_charged_lep_in:
        return (0, 2 + (1 if has_gamma_out else 0))
    if n_g_in == 2:
        return (2, 0)
    # qq̄ initiated
    if has_ew_boson_out or any(p in {"11","-11","13","-13","15","-15"} for p in outgoing):
        return (0, 2)
    return (2, 0)


def get_openloops_amplitude(
    process: str, theory: str = "QED",
) -> Optional[AmplitudeResult]:
    """Return an OL-backed AmplitudeResult if a library is installed.

    Returns None if (a) OpenLoops bindings are unavailable, (b) the process
    cannot be translated to PDG codes, or (c) no installed OL library covers
    this channel (caller should fall through to None → API 422).
    """
    from feynman_engine.amplitudes.openloops_bridge import (
        is_available, to_pdg_string, installed_processes,
        register_process_with_orders, OpenLoopsRegistrationError,
        ew_nlo_library_for, has_ew_nlo_library,
    )

    if not is_available():
        return None

    try:
        pdg = to_pdg_string(process)
    except ValueError:
        return None

    # Probe whether ANY installed library covers this process.  OL doesn't
    # publish a process→library lookup, so we try registration first with
    # default orders, then fall through to specific (qcd, ew) order
    # combinations that match common library configurations:
    #   - ppllj      : QCD NLO at α²α_s¹ → tree at order_ew=2, order_qcd=0
    #   - eell_ew    : EW NLO at α²α¹  → tree at order_ew=2 (no QCD)
    #   - eevvjj    : 2→4 EW at α⁴
    # We probe each combination cheaply; the underlying lru_cache de-dupes.
    # Infer the LEADING coupling order from the process structure so we
    # always pick OL's lowest-order tree (not loop² or higher EW orders).
    leading_qcd, leading_ew = infer_leading_orders(process)

    # Try library defaults FIRST (lets OL pick canonical tree channel),
    # then inferred leading orders.  This avoids summing over multiple
    # variants in EW NLO libraries (e.g. eell_ew has 4 channels for
    # eeexex; explicit order_ew=2 sums them all).
    proc = None
    order_attempts = [
        (-1, -1),                           # library default (canonical tree)
        (leading_qcd, leading_ew),          # inferred leading
    ]
    # Add common variants only if leading didn't already cover them
    seen = {(-1, -1), (leading_qcd, leading_ew)}
    for combo in [(0, 2), (0, 3), (2, 0), (1, 2)]:
        if combo not in seen:
            order_attempts.append(combo)
            seen.add(combo)

    for oqcd, oew in order_attempts:
        try:
            proc = register_process_with_orders(
                process, amptype="tree",
                order_qcd=oqcd, order_ew=oew,
                loop_order_qcd=-1, loop_order_ew=-1,
            )
            break
        except OpenLoopsRegistrationError:
            continue
        except Exception:
            continue
    if proc is None:
        return None

    # Sample a single PSP point to get a representative |M|² number for the
    # evaluation_point field.  Pick √s above the kinematic threshold so OL's
    # internal RAMBO doesn't fail.
    incoming_pdgs = pdg.split("->")[0].split()
    outgoing_pdgs = pdg.split("->")[1].split() if "->" in pdg else []
    is_pp = any(p in {"21", "1", "-1", "2", "-2", "3", "-3", "4", "-4", "5", "-5"}
                for p in incoming_pdgs)

    # Compute sum of final-state masses to set a safe √s sample.
    _PDG_MASS_GEV: dict[str, float] = {
        "6": 172.69, "-6": 172.69,             # top
        "23": 91.19, "24": 80.38, "-24": 80.38,  # Z, W±
        "25": 125.25,                           # Higgs
    }
    sum_final_masses = sum(_PDG_MASS_GEV.get(p, 0.0) for p in outgoing_pdgs)
    threshold = max(sum_final_masses * 1.5, 100.0)  # 50% above threshold

    if is_pp:
        sqrt_s_sample = max(13000.0, threshold)
    else:
        sqrt_s_sample = max(200.0, threshold)

    sample_msq: Optional[float] = None
    try:
        from feynman_engine.amplitudes.openloops_bridge import (
            _CwdInPrefix, _register_lock,
        )
        with _register_lock, _CwdInPrefix():
            me = proc.evaluate(float(sqrt_s_sample))
        sample_msq = float(me.tree)
    except Exception:
        sample_msq = None

    # Determine which OL library covered this (best-effort metadata)
    library_hint = ew_nlo_library_for(process) or "(installed)"

    return AmplitudeResult(
        process=process,
        theory=theory,
        msq=None,
        msq_latex=(
            r"\text{numerical OpenLoops evaluator (library: " + library_hint + r")}"
        ),
        description=(
            f"OpenLoops 2 numerical |M|² evaluator (no symbolic form).  "
            f"Sample at √s = {sqrt_s_sample} GeV: |M|² ≈ "
            f"{sample_msq:.4e}" if sample_msq is not None else "(unavailable)"
            + ".  Use the cross-section endpoint for σ; no closed-form expression available."
        ),
        integral_latex=None,
        notes=(
            "Backend: OpenLoops 2.  This process has no curated symbolic |M|², "
            "so the engine routes σ computation through OL's compiled process "
            "library.  The user-facing cross-section, decay-width, and differential "
            "endpoints work; the symbolic |M|² display is unavailable for OL-backed "
            "processes."
        ),
        backend="openloops",
        approximation_level="openloops-numerical",
        evaluation_point={
            "sqrt_s_gev": sqrt_s_sample,
            "sigma_msq_at_psp": sample_msq,
            "library_hint": library_hint,
        },
    )
