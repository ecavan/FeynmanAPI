"""
Amplitude facade.

This module prefers the generic QGRAF-driven symbolic backend and falls back to
the small curated library for processes/topologies not supported by the generic
backend yet.
"""
from __future__ import annotations

from typing import Optional

from sympy import Rational, Symbol, latex, symbols

from feynman_engine.amplitudes import AmplitudeResult, get_symbolic_amplitude


# ── Shared symbols ────────────────────────────────────────────────────────────

s, t, u = symbols("s t u", real=True)
e_em = symbols("e", positive=True)
g_s = symbols("g_s", positive=True)
cos_theta = symbols("cos_theta", real=True)


# ── Curated fallback amplitudes ───────────────────────────────────────────────

def _qed_ee_to_mumu() -> AmplitudeResult:
    msq = 2 * e_em**4 * (t**2 + u**2) / s**2
    return AmplitudeResult(
        process="e+ e- -> mu+ mu-",
        theory="QED",
        msq=msq,
        msq_latex=latex(msq),
        description="Single s-channel photon exchange",
        notes="Curated massless-limit QED result. Mandelstam: s+t+u=0.",
        backend="curated",
    )


def _qed_bhabha() -> AmplitudeResult:
    msq = 2 * e_em**4 * (
        (s**2 + u**2) / t**2
        + 2 * s * u / (s * t)
        + (t**2 + u**2) / s**2
    )
    return AmplitudeResult(
        process="e+ e- -> e+ e-",
        theory="QED",
        msq=msq,
        msq_latex=latex(msq),
        description="Bhabha scattering: s-channel + t-channel photon exchange",
        notes="Curated massless-limit result including interference.",
        backend="curated",
    )


def _qed_compton() -> AmplitudeResult:
    msq = -2 * e_em**4 * (s / u + u / s)
    return AmplitudeResult(
        process="e- gamma -> e- gamma",
        theory="QED",
        msq=msq,
        msq_latex=latex(msq),
        description="Compton scattering: s-channel + u-channel electron exchange",
        notes="Curated massless-electron limit result.",
        backend="curated",
    )


def _qed_ee_to_ee() -> AmplitudeResult:
    msq = 2 * e_em**4 * (
        (s**2 + u**2) / t**2
        - 2 * s**2 / (t * u)
        + (s**2 + t**2) / u**2
    )
    return AmplitudeResult(
        process="e- e- -> e- e-",
        theory="QED",
        msq=msq,
        msq_latex=latex(msq),
        description="Møller scattering: t-channel + u-channel photon exchange",
        notes="Curated massless-limit result.",
        backend="curated",
    )


def _qcd_qqbar_to_gg() -> AmplitudeResult:
    msq = Rational(32, 3) * g_s**4 * (
        t / u + u / t - Rational(9, 4) * (t**2 + u**2) / s**2
    )
    return AmplitudeResult(
        process="q q~ -> g g",
        theory="QCD",
        msq=msq,
        msq_latex=latex(msq),
        description="qq̄ → gg: t-channel + u-channel quark + s-channel gluon",
        notes="Curated SU(3) color-averaged result for massless quarks.",
        backend="curated",
    )


_CURATED: dict[tuple[str, str], AmplitudeResult] = {}


def _build_curated() -> None:
    results = [
        _qed_ee_to_mumu(),
        _qed_bhabha(),
        _qed_compton(),
        _qed_ee_to_ee(),
        _qcd_qqbar_to_gg(),
    ]
    for result in results:
        _CURATED[(result.process, result.theory)] = result


_build_curated()


def get_curated_amplitude(process: str, theory: str = "QED") -> Optional[AmplitudeResult]:
    return _CURATED.get((process.strip(), theory.upper()))


def get_amplitude(process: str, theory: str = "QED") -> Optional[AmplitudeResult]:
    """
    Return the best available spin-averaged |M|² for a process.

    Order of preference:
      1. Generic QGRAF-driven symbolic backend
      2. Curated fallback results
    """
    symbolic = get_symbolic_amplitude(process.strip(), theory.upper())
    if symbolic is not None:
        return symbolic
    return get_curated_amplitude(process, theory)


def list_supported_processes() -> list[dict]:
    curated = [
        {"process": result.process, "theory": result.theory, "description": result.description}
        for result in _CURATED.values()
    ]
    generic = [
        {
            "process": "e+ e- -> mu+ mu-",
            "theory": "QED",
            "description": "Generic symbolic s-channel fermion current exchange via QGRAF",
        },
        {
            "process": "u u~ -> d d~",
            "theory": "QCD",
            "description": "Generic symbolic s-channel qqbar annihilation through gluon exchange",
        },
        {
            "process": "e+ e- -> mu+ mu-",
            "theory": "EW",
            "description": "Generic symbolic neutral-current EW exchange (vector/scalar couplings)",
        },
        {
            "process": "e+ e- -> chi chi~",
            "theory": "BSM",
            "description": "Generic symbolic fermion-to-scalar annihilation through Zp exchange",
        },
    ]
    seen: set[tuple[str, str, str]] = set()
    results: list[dict] = []
    for item in generic + curated:
        key = (item["process"], item["theory"], item["description"])
        if key not in seen:
            seen.add(key)
            results.append(item)
    return results


def mandelstam_from_cm(s_val: float, cos_theta_val: float) -> dict[str, float]:
    """Convert CM energy² and scattering angle to Mandelstam s, t, u."""
    t_val = -s_val / 2.0 * (1.0 - cos_theta_val)
    u_val = -s_val / 2.0 * (1.0 + cos_theta_val)
    return {"s": s_val, "t": t_val, "u": u_val}
