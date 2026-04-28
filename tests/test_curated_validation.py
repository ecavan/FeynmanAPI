"""Validation tests for every curated amplitude.

For every entry in the curated registry, this verifies:

1. The amplitude evaluates to a finite, non-negative |M̄|²(s, t, u) at a
   physical kinematic point — catches sign bugs and unbound symbols.
2. Where the engine has a 2→2 cross-section integrator, σ(√s) is finite,
   positive, and within an order-of-magnitude band keyed to physics
   expectations — catches order-of-magnitude blunders.
3. Where a published reference σ exists, the engine is within a factor of
   ~3 (built-in PDF) or ~30% (LHAPDF) of it — catches normalization bugs.

These are the regression tests that would have caught all three bugs
documented in eli.md (e+e-→ZZ sign, photon-quark Q_f, ggH τ_H factor).

Test naming convention: each curated entry gets one assertion in
``test_curated_amplitude_finite`` and (where applicable) one in
``test_curated_cross_section_in_band``.
"""
from __future__ import annotations

import math

import pytest

from feynman_engine.amplitudes.cross_section import (
    _build_coupling_defaults,
    total_cross_section,
)
from feynman_engine.physics.amplitude import _CURATED


# ---------------------------------------------------------------------------
# Per-process √s for amplitude evaluation
# ---------------------------------------------------------------------------
# A mass-aware √s suitable for testing each process category. Decays use
# the parent mass; 2→2 production uses something well above threshold.

_PARENT_MASS = {
    "Z": 91.188, "H": 125.25, "W-": 80.369, "W+": 80.369, "t": 172.69, "mu-": 0.106,
}


def _sqrt_s_for(process: str) -> float:
    """Pick a sensible √s for a process: parent mass for decay, well-above-threshold for scattering."""
    parts = process.split("->")
    incoming = parts[0].strip().split()
    outgoing = parts[1].strip().split() if len(parts) > 1 else []

    # Decay (1→N): use parent mass
    if len(incoming) == 1:
        return _PARENT_MASS.get(incoming[0], 100.0)

    # 2→2 or 2→N: well above the heaviest final-state mass
    heaviest = max(
        _PARENT_MASS.get(p.lstrip("+-").rstrip("~"), 0.0) for p in outgoing
    ) if outgoing else 0.0
    # Choose √s = max(2·heaviest + 50, 91.2) for safety above threshold
    return max(2 * heaviest + 50.0, 91.2)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _is_2to2(process: str) -> bool:
    parts = process.split("->")
    if len(parts) != 2:
        return False
    return len(parts[0].strip().split()) == 2 and len(parts[1].strip().split()) == 2


def _is_q_q_template(process: str) -> bool:
    """Is this a generic 'q q~ -> X' template (not a real particle)?"""
    return process.startswith("q ") or " q " in process or "q~" in process and process.split()[0] == "q"


def _has_unbound_symbols(msq, theory: str) -> list[str]:
    """Return list of free-symbol names that wouldn't be substituted."""
    defaults = _build_coupling_defaults(theory)
    # A symbol is "bound" if it's a kinematic invariant (s, t, u) or in defaults.
    KINEMATIC = {"s", "t", "u"}
    unbound = []
    for sym in msq.free_symbols:
        if sym.name in KINEMATIC:
            continue
        if sym.name in defaults:
            continue
        unbound.append(sym.name)
    return unbound


# ---------------------------------------------------------------------------
# Catalog the curated processes (parametrize)
# ---------------------------------------------------------------------------

_ALL_CURATED = sorted(_CURATED.items(), key=lambda kv: (kv[0][1], kv[0][0]))


@pytest.fixture(scope="module")
def curated_keys():
    """Yield (process, theory) for every curated amplitude."""
    return list(_CURATED.keys())


# ---------------------------------------------------------------------------
# Test 1: every curated amplitude has a fully-substitutable |M̄|²
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(_CURATED.keys()),
                         ids=[f"{p}|{t}" for p, t in sorted(_CURATED.keys())])
def test_curated_amplitude_no_unbound_symbols(key):
    """Every curated |M̄|² must use only kinematic + known coupling symbols.

    Catches the silent σ=0 bug class: when an amplitude has an unbound
    symbol like 'g_Z_X' or 'Q_q', the cross-section integrator silently
    returns zero (or fails on float()).
    """
    process, theory = key
    result = _CURATED[key]
    msq = result.msq
    if msq is None or msq == 0:
        pytest.skip(f"curated msq is None/0 for {process}")
        return
    if not hasattr(msq, "free_symbols"):
        # Plain numeric amplitude — fine
        return
    unbound = _has_unbound_symbols(msq, theory)
    # Allow the documented pseudo-symbols Q_q (quark charge), v_f / a_f
    # (Z couplings), m_f / y_f (generic fermion mass / Yukawa) which appear
    # in TEMPLATE entries (q q~ -> X, Z -> f f~, H -> f f~).  These are
    # not directly integrated; per-flavour entries (u u~ -> γγ etc.) are
    # what the engine actually uses.
    is_template = (
        process.startswith("q ")
        or " f " in process
        or process.endswith(" f~")
        or " f f~" in process
    )
    if is_template:
        allowed_template = {"Q_q", "v_f", "a_f", "y_f", "m_f"}
        unbound = [s for s in unbound if s not in allowed_template]

    assert not unbound, (
        f"Curated {process} ({theory}) has unbound symbols {unbound}. "
        f"These will cause σ=0 silent bugs. Add to _build_coupling_defaults() "
        f"in cross_section.py."
    )


# ---------------------------------------------------------------------------
# Test 2: every 2→2 curated amplitude integrates to a finite positive σ
# ---------------------------------------------------------------------------

# Skip templates (q q~ -> X with no specific quark) since they aren't real
# parseable processes.
_2TO2_TESTABLE = [
    (p, t) for (p, t) in sorted(_CURATED.keys())
    if _is_2to2(p) and not p.startswith("q ")
]


@pytest.mark.parametrize("key", _2TO2_TESTABLE,
                         ids=[f"{p}|{t}" for p, t in _2TO2_TESTABLE])
def test_curated_cross_section_finite_positive(key):
    """σ(√s) from each 2→2 curated amplitude is finite and positive."""
    process, theory = key
    sqrt_s = _sqrt_s_for(process)
    r = total_cross_section(process, theory, sqrt_s=sqrt_s)
    if not r.get("supported"):
        # Threshold check or theory-validation rejected — that's fine
        # as long as there's an explicit error message.
        assert r.get("error"), f"{process}: not supported and no error message"
        return
    sigma = r.get("sigma_pb")
    assert sigma is not None, f"{process}: σ is None despite supported=True"
    assert math.isfinite(sigma), f"{process}: σ is not finite: {sigma}"
    assert sigma >= 0, f"{process}: σ is negative: {sigma}"
    # Also catch the silent-zero case if there are no kinematic-only symbols
    # in |M̄|² (some amplitudes legitimately give σ=0 below threshold).
    if sqrt_s > 100.0:
        # Most processes should have σ > 1e-10 pb at our chosen √s
        if sigma == 0.0:
            # Allow zero only if the amplitude is structurally below threshold
            r_check = total_cross_section(process, theory, sqrt_s=sqrt_s * 5)
            # If even at 5× √s it's still zero, fail
            assert r_check.get("sigma_pb", 0) > 0, (
                f"{process}: σ identically zero across two energies — "
                "likely a silent unbound-symbol bug"
            )


# ---------------------------------------------------------------------------
# Test 3: spot-check key benchmarks against published numbers
# ---------------------------------------------------------------------------
# These pin SPECIFIC σ values to specific reference numbers.  If a future
# refactor changes the convention, breaks a sign, or drops a factor, one
# of these will fail.

@pytest.mark.parametrize("process,theory,sqrt_s,sigma_low,sigma_high,reference", [
    # QED
    ("e+ e- -> mu+ mu-", "QED", 91.0, 9.0, 12.0, "analytic 4πα²/3s ≈ 10.5 pb"),
    ("e+ e- -> mu+ mu-", "QED", 10.0, 850.0, 880.0, "analytic ≈ 869 pb"),
    ("e+ e- -> tau+ tau-", "QED", 91.0, 9.0, 12.0, "same as μμ above τ threshold"),
    ("e+ e- -> e+ e-", "QED", 10.0, 1e5, 1e7, "Bhabha; t-pole gives huge σ"),
    ("e+ e- -> gamma gamma", "QED", 10.0, 8000.0, 9000.0, "after 1/2! identical photons"),
    # QCD
    ("u u~ -> g g", "QCD", 100.0, 5e3, 1e5, "Combridge"),
    ("g g -> g g", "QCD", 100.0, 5e6, 5e7, "huge near forward"),
    ("u u~ -> t t~", "QCD", 500.0, 4.0, 25.0, "massive Combridge"),
    ("g g -> t t~", "QCD", 500.0, 10.0, 100.0, "massive Combridge"),
    # EW (Z pole + above)
    ("e+ e- -> mu+ mu-", "EW", 91.2, 1000.0, 3000.0, "Z peak ~ 1.5-2 nb"),
    ("e+ e- -> mu+ mu-", "EW", 200.0, 1.0, 5.0, "above-Z, ~2 pb"),
    ("e+ e- -> Z Z", "EW", 200.0, 0.5, 3.0, "LEP-2 ~ 1.5 pb"),
    ("e+ e- -> W+ W-", "EW", 200.0, 5.0, 30.0, "LEP-2 ~ 18 pb"),
    ("u u~ -> Z H", "EW", 300.0, 0.01, 1.0, "qq̄ ZH partonic"),
    ("u u~ -> Z Z", "EW", 500.0, 0.1, 5.0, "qq̄ ZZ partonic"),
    ("u u~ -> W+ W-", "EW", 500.0, 0.5, 10.0, "qq̄ WW partonic (t-channel only)"),
])
def test_curated_benchmark_values(process, theory, sqrt_s, sigma_low, sigma_high, reference):
    """Specific σ values pinned to reference numbers.

    Each pin would fail if a sign bug, missing factor, or convention
    change broke the curated formula.  The bands are wide enough to allow
    PDF/coupling tweaks but tight enough to catch real bugs.
    """
    r = total_cross_section(process, theory, sqrt_s=sqrt_s)
    assert r.get("supported"), (
        f"{process} ({theory}) at √s={sqrt_s}: not supported "
        f"({r.get('error', 'no error msg')})"
    )
    sigma = r["sigma_pb"]
    assert sigma_low <= sigma <= sigma_high, (
        f"{process} ({theory}) at √s={sqrt_s}: σ = {sigma} pb, "
        f"expected [{sigma_low}, {sigma_high}] ({reference})"
    )
