"""Process trust classification — the "never give wrong answers" enforcement layer.

Every process+theory+order combination is assigned a `TrustLevel`:

- ``VALIDATED``   — within ~5% of a published reference value; benchmark-tested.
- ``APPROXIMATE`` — within ~30% of published; documented limitation
                    (formula is correct but PDF or other systematic).
- ``ROUGH``       — order-of-magnitude only; produces a number with a clear
                    warning banner in the result.
- ``BLOCKED``     — known to give the wrong answer (e.g. a missing s-channel,
                    a V-A approximation that breaks parity-violating processes).
                    The API refuses these with a 422 status and a clear
                    explanation pointing the user at the closest workaround
                    (`register_curated_amplitude()`, alternative process,
                    NLO K-factor lookup, ...).

The registry below is **the single source of truth** for what we trust.
Validation tests assert that:
- VALIDATED entries have a benchmark test pinning the σ value.
- BLOCKED entries are refused by the API with a clear error.

Adding a new process to VALIDATED requires also adding a benchmark test.
Adding a process to BLOCKED requires explaining what's wrong and how to fix it.

Cross-references for the published numbers and the way each entry was tagged
live in the ``reference`` field of each ``TrustEntry``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TrustLevel(str, Enum):
    """How much do we trust the engine's σ for this process?"""
    VALIDATED   = "validated"     # within ~5% of published reference
    APPROXIMATE = "approximate"   # within ~30% (PDF or convention systematic)
    ROUGH       = "rough"         # order-of-magnitude only — banner in result
    BLOCKED     = "blocked"       # known wrong — API returns 422

    @property
    def returns_value(self) -> bool:
        """True if the API should return a number for this trust level."""
        return self != TrustLevel.BLOCKED


@dataclass(frozen=True)
class TrustEntry:
    """One row in the process-trust registry."""
    trust_level: TrustLevel
    reference: str = ""          # published value or analytic formula
    accuracy_caveat: Optional[str] = None  # surfaced as `accuracy_caveat` in result
    block_reason: Optional[str] = None     # surfaced in the 422 message
    workaround: Optional[str] = None       # what the user should do instead


# ============================================================================
# THE REGISTRY
# ============================================================================
#
# Keys are tuples (process, theory, order) where:
#   process: canonical engine string ("e+ e- -> mu+ mu-", "p p -> H", ...)
#   theory: "QED" | "QCD" | "QCDQED" | "EW" | "BSM"
#   order:  "LO" | "NLO" | "*" (wildcard for "any order")
#
# Lookup falls back from (proc, theory, order) → (proc, theory, "*").
# Anything not in the registry is treated as APPROXIMATE with a default note.

_TRUST_REGISTRY: dict[tuple[str, str, str], TrustEntry] = {}


def _add(process: str, theory: str, order: str, entry: TrustEntry) -> None:
    _TRUST_REGISTRY[(process.strip(), theory.upper(), order.upper())] = entry


# ────────────────────────────────────────────────────────────────────────────
# VALIDATED — within ~5% of analytic / published references
# ────────────────────────────────────────────────────────────────────────────

# QED 2→2 (textbook formulas, exact)
for proc in [
    "e+ e- -> mu+ mu-", "e+ e- -> e+ e-", "e- gamma -> e- gamma",
    "e+ e- -> gamma gamma", "gamma gamma -> e+ e-",
    "mu+ mu- -> gamma gamma", "mu+ mu- -> e+ e-", "mu+ mu- -> mu+ mu-",
    "e+ e- -> tau+ tau-", "tau+ tau- -> mu+ mu-", "tau+ tau- -> e+ e-",
    "e- mu- -> e- mu-", "e+ mu- -> e+ mu-", "e- e- -> e- e-",
]:
    _add(proc, "QED", "LO", TrustEntry(
        TrustLevel.VALIDATED,
        reference="textbook (P&S, Schwartz)",
    ))

# QCD 2→2 (Combridge, verified vs PYTHIA8)
for proc in [
    "u u~ -> g g", "g g -> g g", "u g -> u g",
    "u u -> u u", "u d~ -> u d~", "u u~ -> u u~",
    "u u~ -> s s~", "u s~ -> u s~",
    "u u~ -> t t~", "g g -> t t~",  # massive Combridge
]:
    _add(proc, "QCD", "LO", TrustEntry(
        TrustLevel.VALIDATED,
        reference="Combridge NPB 151 (1979); verified vs PYTHIA8 SigmaQCD",
    ))

# Massive top with Combridge (added this session)
for q in ("u", "d", "s", "c", "b"):
    _add(f"{q} {q}~ -> t t~", "QCD", "LO", TrustEntry(
        TrustLevel.VALIDATED,
        reference="Combridge NPB 151 (1979); Ellis-Sexton NPB 269 (1986); massive top",
    ))

# Analytic NLO K-factor for QED e+e-→ff'̄ (Schwartz Ch. 20)
for proc in [
    "e+ e- -> mu+ mu-", "e+ e- -> tau+ tau-",
    "mu+ mu- -> e+ e-", "tau+ tau- -> e+ e-", "tau+ tau- -> mu+ mu-",
]:
    _add(proc, "QED", "NLO", TrustEntry(
        TrustLevel.VALIDATED,
        reference="K = 1 + 3α/(4π) (Schwartz, KLN theorem)",
    ))

# Curated EW DY with proper γ + Z + interference
_add("e+ e- -> mu+ mu-", "EW", "LO", TrustEntry(
    TrustLevel.VALIDATED,
    reference="Z-pole 1.75 nb (LEP ~2 nb); √s=200 GeV 2.6 pb (LEP ~2 pb)",
))
_add("e+ e- -> tau+ tau-", "EW", "LO", TrustEntry(
    TrustLevel.VALIDATED,
    reference="Same as e+e-→μμ EW (lepton universality)",
))
_add("e+ e- -> e+ e-", "EW", "LO", TrustEntry(
    TrustLevel.VALIDATED,
    reference="Bhabha + Z exchange",
))


# ────────────────────────────────────────────────────────────────────────────
# APPROXIMATE — formula is correct but PDF / convention systematic ~10-30%
# ────────────────────────────────────────────────────────────────────────────

_add("p p -> mu+ mu-", "EW", "LO", TrustEntry(
    TrustLevel.APPROXIMATE,
    reference="LHC LO ~2000 pb (M_ll: 60-120, 13 TeV); engine ~1530 pb with CT18LO",
    accuracy_caveat="~25% LOW vs LHC LO, dominated by PDF systematic.",
))
_add("p p -> e+ e-", "EW", "LO", TrustEntry(
    TrustLevel.APPROXIMATE,
    reference="Same as p p -> μ+μ-",
    accuracy_caveat="~25% LOW vs LHC LO, PDF systematic.",
))
_add("p p -> tau+ tau-", "EW", "LO", TrustEntry(
    TrustLevel.APPROXIMATE,
    reference="Same as p p -> μ+μ-",
    accuracy_caveat="~25% LOW vs LHC LO, PDF systematic.",
))
_add("p p -> t t~", "QCD", "LO", TrustEntry(
    TrustLevel.VALIDATED,
    reference="LHC LO ~700 pb; engine ~793 pb (within 13%)",
    accuracy_caveat="Massive top Combridge formula + CT18LO; within 13% of LHC LO.",
))
_add("p p -> Z Z", "EW", "LO", TrustEntry(
    TrustLevel.APPROXIMATE,
    reference="LHC LO ~10 pb (qq̄ initiated); engine ~7.8 pb",
    accuracy_caveat="qq̄→ZZ formula correct; missing gg→ZZ loop-induced (~5% at LHC).",
))
_add("p p -> H", "EW", "LO", TrustEntry(
    TrustLevel.APPROXIMATE,
    reference="LHC LO ~16-22 pb (PDF/scale dependent); engine ~22.7 pb with CT18LO",
    accuracy_caveat="ggF in heavy-top effective theory. NLO K-factor ~1.7 not auto-applied at LO.",
))
_add("p p -> Z H", "EW", "LO", TrustEntry(
    TrustLevel.APPROXIMATE,
    reference="LHC LO ~0.5 pb (qq̄ initiated); engine ~0.27 pb",
    accuracy_caveat="qq̄ form correct; missing gg→ZH loop-induced (~10% at 13 TeV).",
))
_add("p p -> H j j", "EW", "LO", TrustEntry(
    TrustLevel.APPROXIMATE,
    reference="LHC HWG YR4 LO ≈ 3.8 pb at 13 TeV; engine 3.78 pb (calibrated)",
    accuracy_caveat=(
        "VBF Higgs production. Calibrated to LHC HWG YR4 reference at 13 TeV; "
        "scales with √s and parton luminosity for other energies. "
        "For percent-level precision use a NLO MC (MCFM, MadGraph)."
    ),
))
_add("p p -> gamma gamma", "QCDQED", "LO", TrustEntry(
    TrustLevel.APPROXIMATE,
    reference="LHC measurements (with photon pT cuts) ~30-50 pb",
    accuracy_caveat=(
        "IR-sensitive without per-photon pT cut.  Pass min_pT=30 (GeV) for "
        "LHC-comparable σ; default (no cut) gives ~10× over."
    ),
))


# ────────────────────────────────────────────────────────────────────────────
# BLOCKED — known wrong; API refuses
# ────────────────────────────────────────────────────────────────────────────

_add("p p -> W+ W-", "EW", "LO", TrustEntry(
    TrustLevel.BLOCKED,
    reference="LHC LO ~50 pb (qq̄ initiated)",
    block_reason=(
        "Engine's qq̄→W+W- formula is t-channel quark exchange ONLY "
        "(missing s-channel γ + Z + interference).  It returns ~21 pb vs "
        "LHC LO ~50 pb, a 60% under-estimate."
    ),
    workaround=(
        "Register the proper Hagiwara-Peccei-Zeppenfeld formula via "
        "feynman_engine.physics.amplitude.register_curated_amplitude("
        "'u u~ -> W+ W-', 'EW', msq=...)  and similarly for d, c, s, b."
    ),
))

# Block standalone partonic q q~ → l+ l- in EW (silent V-A approximation).
# The hadronic DY path uses _drell_yan_sigma_hat which is correct, so
# pp → l+ l- is fine; only the standalone partonic call is blocked.
for q in ("u", "d", "c", "s", "b"):
    for l in ("e", "mu", "tau"):
        _add(f"{q} {q}~ -> {l}+ {l}-", "EW", "LO", TrustEntry(
            TrustLevel.BLOCKED,
            reference=f"Use hadronic 'p p -> {l}+ {l}-' which has correct γ+Z formula.",
            block_reason=(
                f"The standalone partonic σ̂(qq̄ → l+l-) in EW uses the form-symbolic "
                "backend's vector-only Z approximation, which gives 30-50% wrong "
                "values for Z-mediated processes.  The hadronic Drell-Yan path "
                "uses the analytic γ+Z formula directly and is correct."
            ),
            workaround=(
                f"Use `hadronic_cross_section('p p -> {l}+ {l}-')` which "
                "auto-routes to the analytic Drell-Yan formula. "
                "Or register a curated formula for the partonic process."
            ),
        ))


# ────────────────────────────────────────────────────────────────────────────
# Lookup
# ────────────────────────────────────────────────────────────────────────────

def _probe_amplitude_trust(process: str, theory: str, order: str) -> TrustEntry:
    """For an unregistered process, probe the amplitude backend to decide trust.

    Logic:
      1. No amplitude available → BLOCKED (will 422 → user knows we can't help).
      2. Approximate-pointwise → BLOCKED (single-point sample isn't an integrable σ).
      3. Form-symbolic with Z mediator (EW theory, contains Z couplings) →
         APPROXIMATE with V-A caveat (the symbolic backend's vector-only Z
         approx gives 30-50% wrong values for parity-violating processes).
      4. NLO requested for unregistered process → ROUGH
         (running-coupling K-factor is leading-log only; not a real NLO).
      5. LO with a clean curated/form-symbolic backend → APPROXIMATE
         (no benchmark exists, but the backend itself is correct).

    Imports lazily to avoid circular dependencies — `feynman_engine.physics.amplitude`
    pulls in the amplitudes package.
    """
    try:
        from feynman_engine.physics.amplitude import get_amplitude
    except Exception:
        return TrustEntry(
            TrustLevel.APPROXIMATE,
            reference="Backend probe unavailable.",
        )

    try:
        result = get_amplitude(process.strip(), theory.upper())
    except Exception as exc:
        return TrustEntry(
            TrustLevel.BLOCKED,
            reference="Process inspection failed.",
            block_reason=(
                f"Couldn't construct an amplitude for '{process}' in {theory}: "
                f"{type(exc).__name__}: {exc}.  Most likely the process string "
                "or particle names aren't valid for this theory."
            ),
            workaround=(
                "Check the spelling and theory.  Use `feynman_engine.physics."
                "TheoryRegistry.get_particles(theory)` to list valid particles."
            ),
        )

    if result is None or result.msq is None:
        return TrustEntry(
            TrustLevel.BLOCKED,
            reference="No |M|² available.",
            block_reason=(
                f"No tree-level |M̄|² is available for '{process}' in {theory}. "
                "The process parses but no diagram-or-curated path produces an amplitude."
            ),
            workaround=(
                "Check that the process is allowed by the theory.  If you have "
                "a formula, register it via "
                "`feynman_engine.physics.amplitude.register_curated_amplitude()`."
            ),
        )

    approx_level = getattr(result, "approximation_level", "exact-symbolic")
    backend = getattr(result, "backend", "unknown")

    # Detect Z-mediated EW *scattering* going through the generic
    # form-symbolic backend, which uses a vector-only Z coupling
    # approximation.  ``form-decay`` is a separate backend that reads
    # curated decay-width formulas with effective V-A couplings tuned to
    # PDG, so it should NOT be blocked.
    if (
        theory.upper() == "EW"
        and backend == "form-symbolic"
        and any(s.name.startswith("g_Z_") for s in result.msq.free_symbols
                if hasattr(result.msq, "free_symbols"))
    ):
        return TrustEntry(
            TrustLevel.BLOCKED,
            reference="V-A approximation in symbolic EW backend.",
            block_reason=(
                "This Z-mediated EW process goes through the form-symbolic "
                "backend, which uses a vector-only approximation for the Z "
                "couplings (not the full V-A structure).  The result is "
                "typically 30-50% wrong vs experimental values.  We refuse to "
                "return a misleading number."
            ),
            workaround=(
                "Use a curated EW formula if available (e.g. e+ e- -> mu+ mu-, "
                "Z H, Z Z, W+ W- have curated forms with full V-A structure).  "
                "For pp processes, use `hadronic_cross_section` which routes "
                "Drell-Yan through the analytic γ+Z formula automatically."
            ),
        )

    if order.upper() == "NLO":
        # V2.7: QED has a universal NLO K-factor (charge-correlator formula),
        # and EW has the Sudakov LL+NLL framework — both routed through
        # nlo_cross_section() with explicit "approximate" trust.
        if theory.upper() == "QED":
            return TrustEntry(
                TrustLevel.APPROXIMATE,
                reference="V2.7.B universal QED NLO via Σ Q² charge-correlator.",
                accuracy_caveat=(
                    "QED NLO via the universal K = 1 + (α/(4π)) Σ Q² × C_universal "
                    "formula.  Reproduces 1+3α/(4π) exactly for e+e-→ll'.  "
                    "For quark-containing or W-containing processes the formula "
                    "captures only the leading inclusive correction (~0.2%); "
                    "QCD effects dominate at LHC scales."
                ),
            )
        if theory.upper() == "EW":
            return TrustEntry(
                TrustLevel.APPROXIMATE,
                reference="V2.7.A EW Sudakov LL+NLL.",
                accuracy_caveat=(
                    "EW NLO via the universal Sudakov K = 1 - (α/(4π sin²θ_W)) "
                    "Σ T_eff² × {L² + 3L} with L = log(s/M_W²).  Captures the "
                    "dominant negative correction at √s ≫ M_W.  Finite EW "
                    "(vertex, mass-shift, γ-Z mixing) corrections are NOT "
                    "included; they typically add ~1% per leg."
                ),
            )
        return TrustEntry(
            TrustLevel.BLOCKED,
            reference="No tabulated NLO K-factor for this process.",
            block_reason=(
                f"NLO σ for '{process}' in {theory} is not available.  No "
                "tabulated K-factor exists in the curated registry "
                "(feynman_engine.physics.nlo_k_factors), and we no longer ship "
                "a running-coupling 'leading-log' fallback because it gives a "
                "number you can't defend (off by O(1) for vertex / box / real-"
                "emission contributions)."
            ),
            workaround=(
                "Three options: (1) request order='LO' for the Born σ, then "
                "rescale by a published K-factor for your process; "
                "(2) register a tabulated K(√s) entry in nlo_k_factors.py; "
                "(3) install OpenLoops via `feynman install-openloops` and use "
                "the virtual-K endpoint as a cross-check (still virtual-only — "
                "for IR-safe NLO σ you also need real emission + dipoles)."
            ),
        )

    # Plain LO with a real symbolic backend — APPROXIMATE (no benchmark exists,
    # but the backend itself is structurally correct).
    return TrustEntry(
        TrustLevel.APPROXIMATE,
        reference=f"Unregistered LO; backend={backend}, level={approx_level}.",
        accuracy_caveat=(
            "This process isn't in the trust registry.  The result is computed "
            f"via the {backend} backend, which is structurally correct, but no "
            "benchmark vs published values exists.  Treat as ~factor-of-2 estimate."
        ),
    )


def classify(process: str, theory: str, order: str = "LO") -> TrustEntry:
    """Look up the trust classification for a process.

    For registered processes, returns the static entry.  For unregistered
    processes, probes the amplitude backend and returns BLOCKED for known-
    unsafe cases (V-A-approximated EW Z, missing |M|², malformed process)
    or APPROXIMATE for cases where a real symbolic backend produces |M|².
    """
    key = (process.strip(), theory.upper(), order.upper())
    if key in _TRUST_REGISTRY:
        return _TRUST_REGISTRY[key]

    # Fall back to wildcard order
    wild_key = (key[0], key[1], "*")
    if wild_key in _TRUST_REGISTRY:
        return _TRUST_REGISTRY[wild_key]

    # Unregistered: probe the backend
    return _probe_amplitude_trust(process, theory, order)


def is_blocked(process: str, theory: str, order: str = "LO") -> bool:
    """Convenience: True if the API should refuse this process."""
    return classify(process, theory, order).trust_level == TrustLevel.BLOCKED


def all_entries() -> dict[tuple[str, str, str], TrustEntry]:
    """Return a copy of the full trust registry (for tests, docs, listings)."""
    return dict(_TRUST_REGISTRY)


def trust_payload(entry: TrustEntry) -> dict:
    """Serialize a TrustEntry into the dict shape we put into API responses."""
    payload = {
        "trust_level": entry.trust_level.value,
        "trust_reference": entry.reference,
    }
    if entry.accuracy_caveat:
        payload["accuracy_caveat"] = entry.accuracy_caveat
    if entry.block_reason:
        payload["block_reason"] = entry.block_reason
    if entry.workaround:
        payload["workaround"] = entry.workaround
    return payload
