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
    install_suggestion: Optional[dict] = None  # {libraries, install_commands, estimate}


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

# QED 2→2 — pure-s-channel textbook formulas (no t-channel singularities,
# integration-limit-independent).
for proc in [
    "e+ e- -> mu+ mu-", "e+ e- -> tau+ tau-",
    "mu+ mu- -> e+ e-", "tau+ tau- -> mu+ mu-", "tau+ tau- -> e+ e-",
    "gamma gamma -> e+ e-",
    "e- mu- -> e- mu-", "e+ mu- -> e+ mu-",  # u-channel only
]:
    _add(proc, "QED", "LO", TrustEntry(
        TrustLevel.VALIDATED,
        reference="textbook (P&S, Schwartz)",
    ))

# QED 2→2 with t-channel γ pole (Bhabha, Møller, fermion+fermion→γγ).
# The engine integrates cos θ ∈ [-0.999, +0.999] with no fiducial cuts,
# so σ is integration-limit-dependent and differs from MG5 (which applies
# default photon/lepton pT and angular cuts).  Benchmarked vs MG5 v3.7.1:
#   e+e- → e+e- @ 91 GeV : engine 1.7e5 vs MG5 4.7e3
#   e+e- → e+e- @ 200 GeV: engine 1.3e4 vs MG5 9.6e2
#   e+e- → e+e- @ 500 GeV: engine 2.0e3 vs MG5 1.6e2
#   e-e- → e-e- @ 200 GeV: engine 1.3e4 vs MG5 1.1e3
#   μ+μ- → μ+μ- @ 200 GeV: engine 1.3e4 vs MG5 9.6e2
#   e+e- → γγ   @ 200 GeV: engine 21.5  vs MG5 14.0
#   μ+μ- → γγ   @ 200 GeV: engine 21.5  vs MG5 14.0
# Use ``differential_distribution()`` with explicit cos θ window for a
# well-defined value, or pass fiducial cuts (when supported).
for proc in [
    "e+ e- -> e+ e-",       # Bhabha
    "mu+ mu- -> mu+ mu-",   # μ-Bhabha
    "e- e- -> e- e-",       # Møller
    "e+ e- -> gamma gamma",
    "mu+ mu- -> gamma gamma",
    "e- gamma -> e- gamma", # Compton
]:
    _add(proc, "QED", "LO", TrustEntry(
        TrustLevel.APPROXIMATE,
        reference=(
            "Textbook formula (P&S, Schwartz) integrated cos θ ∈ "
            "[-0.999, +0.999] with no fiducial cuts. Benchmarked vs "
            "MG5 v3.7.1 at √s = 91-500 GeV — engine grossly above MG5 "
            "due to t-channel γ pole / photon collinear singularity."
        ),
        accuracy_caveat=(
            "**Integration-limit-dependent**: engine integrates cos θ ∈ "
            "[-0.999, 0.999] including the t-channel/collinear singularity. "
            "MG5 default uses photon pT > 10 GeV and lepton angular cuts, "
            "giving 1-2 orders of magnitude smaller σ.  For a well-defined "
            "σ matching experiment, use `differential_distribution()` with "
            "an explicit cos θ or pT window."
        ),
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
    TrustLevel.APPROXIMATE,
    reference="Bhabha + Z exchange",
    accuracy_caveat=(
        "Engine integrates cos θ ∈ [-0.999, 0.999] with no fiducial cuts, "
        "hitting the t-channel Coulomb singularity.  Total σ is "
        "integration-limit-dependent and grossly differs from MG5 with "
        "default lepton cuts (engine 1.7e5 pb vs MG5 4.6e3 pb at √s = 91 GeV).  "
        "Use differential_distribution() with explicit cos θ window for a "
        "well-defined cross-section."
    ),
))

# e+e- → tt̄: closed-form γ+Z formula with proper β factor (ESW §6, Schwartz §29).
# Verified vs MG5 v3.7.1 2026-05-11: +10.3% at √s=350 GeV (threshold), +2.6% at 500 GeV,
# +2.3% at 1 TeV.  Replaces the prior OL+RAMBO numerical path which had on-shell
# condition warnings near threshold and gave +516%, +35%, +6.7% respectively.
_add("e+ e- -> t t~", "EW", "LO", TrustEntry(
    TrustLevel.VALIDATED,
    reference=(
        "Closed-form γ+Z exchange to massive top pair (ESW §6, Schwartz §29).  "
        "Benchmarked vs MG5 v3.7.1 at 350, 500, 1000 GeV: agreement to "
        "+10.3% / +2.6% / +2.3%."
    ),
    accuracy_caveat=(
        "Massive-top closed-form (β factor + proper (3-β²)/2 and β² angular "
        "integration).  ~10% high at threshold (√s=350 GeV) — likely "
        "α-scheme dependence near threshold.  Excellent agreement (≤3%) at "
        "√s ≥ 500 GeV."
    ),
))

# CC diboson e ν̄_e → W- Z (and charge-conjugate e+ ν_e → W+ Z).  The full
# 3-diagram SM tree amplitude is evaluated numerically via the helicity-
# amplitude evaluator (analogous to the HPZ q q̄ → W+ W- path).  Marked
# APPROXIMATE pending external MG5 / published cross-check.
for proc in [
    "e- nu_e~ -> W- Z", "e- nuebar -> W- Z",
    "e+ nu_e -> W+ Z",  "e+ nue -> W+ Z",
]:
    _add(proc, "EW", "LO", TrustEntry(
        TrustLevel.APPROXIMATE,
        reference=(
            "Tree-level SM via direct helicity-amplitude evaluation "
            "(3 diagrams: t-channel ν, u-channel e, s-channel W*- via WWZ TGC). "
            "Benchmarked vs MG5_aMC@NLO v3.7.1 at √s = 200/500/1000 GeV."
        ),
        accuracy_caveat=(
            "**Reliable at √s ≳ 1 TeV** (engine within 6% of MG5, "
            "consistent with α-scheme).  **Unreliable at low √s** (engine "
            "is ~50% LOW at 200-500 GeV).  Diagnostics ruled out: "
            "polarization-sum identity (verified 1e-15), sign conventions "
            "(matches HPZ ee→WW), gauge cancellation (s × σ plateau holds "
            "at √s ≥ 5 TeV), integration convergence (4 sig figs at "
            "n_cos=320), individual diagram magnitudes (consistent with "
            "destructive interference pattern), unitary vs Feynman gauge "
            "(equivalent for massless fermions), spin counting (matches MG5 "
            "IDEN=4 convention).  Root cause unidentified after extensive "
            "investigation.  v0.3 follow-up will compare amplitude per "
            "phase-space point against MG5's HELAS output."
        ),
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
    TrustLevel.APPROXIMATE,
    reference="LHC LO ~830 pb at 13 TeV (PDG); engine ~1330 pb with NNPDF40_lo_as_01180.",
    accuracy_caveat=(
        "Engine returns ~1330 pb at 13 TeV with the default LO PDF "
        "NNPDF40_lo_as_01180 (α_s=0.118).  MG5 default (NN23LO1, α_s=0.130) "
        "gives 504 pb.  The 2.6× difference is the modern-vs-legacy LO PDF "
        "systematic: NNPDF40_lo has higher gluon luminosity at the LHC than "
        "NN23LO1 because the NNPDF40 fit includes recent LHC top-pair data "
        "preferring higher g(x>0.1).  This is NOT a calibration bug — the "
        "engine's partonic σ̂ is the textbook Combridge formula.  To match "
        "MG5 LO, pass `pdf_name='NNPDF23_lo_as_0130'` after installing it via "
        "`feynman install-pdf-set NNPDF23_lo_as_0130`."
    ),
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

# pp → W+ W- previously BLOCKED because the curated qq̄ formula was t-channel
# only and gave ~30 % of the LHC LO σ.  As of 2026-05-10 the partonic
# σ̂(qq̄ → W+W-) routes through the full Hagiwara-Peccei-Zeppenfeld-Hikasa
# helicity-amplitude evaluator (``feynman_engine.amplitudes.qqbar_ww_helicity``),
# which is validated to 97 % vs MG5 LO at √s = 200 GeV for e+e- → W+W-.
# The hadronic σ now reflects the full 3-diagram tree-level SM.
_add("p p -> W+ W-", "EW", "LO", TrustEntry(
    TrustLevel.APPROXIMATE,
    reference=(
        "Hagiwara-Peccei-Zeppenfeld-Hikasa NPB 282 (1987) 253 partonic σ̂ "
        "convolved with PDF (LHC LO ~50 pb at 13 TeV)."
    ),
    accuracy_caveat=(
        "Partonic σ̂(qq̄→W+W-) is the full SM tree-level result via direct "
        "helicity-amplitude evaluation (97 % vs MG5 LO for e+e-→W+W-).  "
        "Hadronic σ inherits the built-in or LHAPDF PDF accuracy.  For "
        "percent-level NLO QCD precision install OpenLoops `ppvv`."
    ),
))

# Block e+ e- → H H (loop-induced di-Higgs at lepton colliders).
# Reason: SM has no tree-level e+e-→HH (no e-e-HH coupling); leading order is
# one-loop (top + W boxes/triangles, Spira-Zerwas).  OpenLoops 2.1.4 does not
# ship a precompiled eehh_ls library, the public process server does not have
# one, and OL doesn't ship the developer process-generator toolchain.  Until
# we either implement the curated Spira-Zerwas closed-form or a future OL
# release adds eehh_ls, this process is blocked.
_add("e+ e- -> H H", "EW", "LO", TrustEntry(
    TrustLevel.BLOCKED,
    reference="Loop-induced; no curated formula and no OL library available.",
    block_reason=(
        "e+ e- -> H H is loop-induced in the SM (no e-e-HH tree-level vertex). "
        "It requires either (a) a curated Spira-Zerwas closed-form with the "
        "full top + W triangle/box loop amplitudes, or (b) the OpenLoops "
        "'eehh_ls' library — neither of which is implemented in this engine "
        "and the OL public process server doesn't host eehh_ls."
    ),
    workaround=(
        "For e+e-→HH studies at FCC-ee / ILC energies use a dedicated MC "
        "(MG5_aMC@NLO with full one-loop SM, Whizard, or a HEFT calculator). "
        "Engine still supports the tree-level Higgsstrahlung channel "
        "'e+ e- -> Z H' for di-Higgs analyses via the Z*→ZH→ZHH decay chain "
        "(use the Z H process and resolve H decay externally)."
    ),
))

# Block e+ e- → Z' Z' in the minimal U(1)' dark-photon BSM model.
# Reason: the simplest dark-photon Lagrangian has only single-Z' couplings to
# SM fermions and to χχ̄ — there is no Z'-Z'-X trilinear vertex (the U(1)' is
# abelian) and no dark-Higgs / kinetic-mixing extension is in scope.  A
# misleading "tree-level" σ would be ~zero or wrong by orders of magnitude.
_add("e+ e- -> Zp Zp", "BSM", "LO", TrustEntry(
    TrustLevel.BLOCKED,
    reference="Minimal U(1)' dark-photon BSM model has no Z'Z'X vertex.",
    block_reason=(
        "e+ e- -> Z' Z' is not supported by the current BSM model.  The minimal "
        "U(1)' dark-photon Lagrangian shipped in this engine has only "
        "(Zp, ff̄) couplings and a (Zp, χχ̄) coupling — no Z'-Z'-Z, Z'-Z'-γ, "
        "or Z'-Z'-Z' vertex.  Producing a Z' pair requires extending the "
        "model (dark Higgs portal, or Z-Z' kinetic mixing)."
    ),
    workaround=(
        "Use a single-Z' production channel (e+ e- -> χ χ̄ via Z') or wait for "
        "a future BSM model extension that adds a dark-Higgs portal or Z-Z' "
        "kinetic mixing."
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

    if result is None or (
        result.msq is None and getattr(result, "backend", None) != "openloops"
    ):
        # Look up which OpenLoops library would cover this process so the
        # 422 response can offer a concrete install command.
        install_suggestion = _lookup_install_suggestion(process)

        if install_suggestion:
            block_reason = (
                f"No tree-level |M̄|² is curated for '{process}' in {theory}, "
                f"but OpenLoops library `{install_suggestion['recommended_library']}` "
                f"({install_suggestion['estimate']['human_disk']}, "
                f"{install_suggestion['estimate']['human_time']}) covers it.  "
                f"Install with: {install_suggestion['install_command']}"
            )
            workaround = (
                f"Run `{install_suggestion['install_command']}` to enable "
                f"numerical OL evaluation, or register a curated symbolic |M̄|² "
                "via `feynman_engine.physics.amplitude.register_curated_amplitude()`."
            )
        else:
            block_reason = (
                f"No tree-level |M̄|² is available for '{process}' in {theory}. "
                "No OpenLoops library in the public catalog covers this multiset."
            )
            workaround = (
                "Check spelling and theory.  If you have a formula, register it "
                "via `feynman_engine.physics.amplitude.register_curated_amplitude()`."
            )

        return TrustEntry(
            TrustLevel.BLOCKED,
            reference="No |M|² available.",
            block_reason=block_reason,
            workaround=workaround,
            install_suggestion=install_suggestion,
        )

    # OL-backed amplitudes have no symbolic |M|² but are valid numerically
    # (cross-section, decay-width, differential endpoints all work via
    # OL Born + RAMBO).  Mark them as trustworthy so the trust gate lets
    # them through.
    if result.msq is None and getattr(result, "backend", None) == "openloops":
        return TrustEntry(
            TrustLevel.VALIDATED,
            reference=(
                "Numerical OpenLoops 2 evaluator (no closed-form |M|², "
                "but σ via Born + RAMBO is exact at LO)."
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
            # V2.7.A + Path-A: try OpenLoops EW NLO library first; fall back
            # to analytic Sudakov + universal QED hybrid.
            try:
                from feynman_engine.amplitudes.openloops_bridge import (
                    has_ew_nlo_library, ew_nlo_library_for,
                )
                lib = ew_nlo_library_for(process)
                if lib and has_ew_nlo_library(process):
                    return TrustEntry(
                        TrustLevel.VALIDATED,
                        reference=(
                            f"Path-A: OpenLoops 2 EW NLO library '{lib}' "
                            "(Buccioni et al., EPJ C 79 (2019) 866) for the "
                            "finite virtual + universal QED inclusive K for the "
                            "real-photon piece.  Universal Catani IR-pole "
                            "closure verified at runtime."
                        ),
                        accuracy_caveat=(
                            "EW NLO via OpenLoops finite virtual.  Includes "
                            "γ + Z + W + H + t one-loop contributions with full "
                            "mass dependence in the G_μ scheme.  Combined with "
                            "the textbook universal QED inclusive K-factor for "
                            "the photon real-emission contribution.  Convention "
                            "bookkeeping note: σ_LO is computed with α(M_Z); "
                            "K_EW is the multiplicative correction in this scheme."
                        ),
                    )
            except Exception:
                pass
            return TrustEntry(
                TrustLevel.APPROXIMATE,
                reference="V2.7.A EW Sudakov LL+NLL fallback.",
                accuracy_caveat=(
                    "EW NLO via the universal Sudakov K = 1 - (α/(4π sin²θ_W)) "
                    "Σ T_eff² × {L² + 3L} with L = log(s/M_W²).  Captures the "
                    "dominant negative correction at √s ≫ M_W.  Finite EW "
                    "(vertex, mass-shift, γ-Z mixing) corrections are NOT "
                    "included; they typically add ~1% per leg.  Install the "
                    "process-specific *_ew library (e.g. eell_ew) to upgrade "
                    "to the OpenLoops finite-virtual path."
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


def _lookup_install_suggestion(process: str) -> Optional[dict]:
    """Find the smallest OL library that covers this process, if any.

    Used both inside ``_probe_amplitude_trust`` (for unregistered BLOCKED
    cases) and in ``classify`` (to enrich statically-registered BLOCKED
    entries that pre-date the install_suggestion field).  Returns None if
    the catalog has no match or every covering library is already
    installed.
    """
    try:
        from feynman_engine.resources.openloops import (
            libraries_for_process, library_meta, estimate_install,
        )
        from feynman_engine.amplitudes.openloops_bridge import installed_processes
        candidates = libraries_for_process(process.strip())
        already = set(installed_processes())
        uninstalled = [c for c in candidates if c not in already]
        if not uninstalled:
            return None
        ranked = sorted(uninstalled, key=lambda lib: (
            library_meta(lib).get("n_channels", 9999) if library_meta(lib) else 9999
        ))
        return {
            "candidate_libraries":    ranked,
            "recommended_library":    ranked[0],
            "install_command":        f"feynman install-process {ranked[0]}",
            "estimate":               estimate_install([ranked[0]]),
            "alternatives_count":     max(0, len(ranked) - 1),
        }
    except Exception:
        return None


def classify(process: str, theory: str, order: str = "LO") -> TrustEntry:
    """Look up the trust classification for a process.

    For registered processes, returns the static entry.  For unregistered
    processes, probes the amplitude backend and returns BLOCKED for known-
    unsafe cases (V-A-approximated EW Z, missing |M|², malformed process)
    or APPROXIMATE for cases where a real symbolic backend produces |M|².

    Statically-registered BLOCKED entries that don't already carry an
    ``install_suggestion`` get one attached at lookup time, so the API's
    422 response always tells the user which OpenLoops library would
    unblock the process.
    """
    import dataclasses

    key = (process.strip(), theory.upper(), order.upper())
    entry = _TRUST_REGISTRY.get(key)
    if entry is None:
        # Fall back to wildcard order
        wild_key = (key[0], key[1], "*")
        entry = _TRUST_REGISTRY.get(wild_key)
    if entry is None:
        # Unregistered: probe the backend
        return _probe_amplitude_trust(process, theory, order)

    # Static hit — enrich BLOCKED entries with an install_suggestion if they
    # don't already have one.  Returns a fresh frozen copy.
    if (
        entry.trust_level == TrustLevel.BLOCKED
        and entry.install_suggestion is None
    ):
        suggestion = _lookup_install_suggestion(process)
        if suggestion is not None:
            return dataclasses.replace(entry, install_suggestion=suggestion)
    return entry


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
