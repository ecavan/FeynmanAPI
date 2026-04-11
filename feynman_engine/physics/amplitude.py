"""
Tree-level amplitude computation using symbolic math (sympy).

No Mathematica required — all calculations use pure Python.

Supported calculations
----------------------
* QED spin-averaged |M|² for standard 2→2 processes in the massless limit
* QCD tree-level colour factors
* Symbolic Mandelstam kinematics helpers

Usage
-----
    from feynman_engine.physics.amplitude import qed_amplitude, mandelstam

    result = qed_amplitude("e+ e- -> mu+ mu-")
    print(result.msq)          # sympy expression for |M|²
    print(result.msq_numeric)  # numerical value at s=100 GeV², t=-30 GeV²
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from sympy import (
        symbols, Rational, sqrt, pi, simplify, latex,
        cos, sin, oo, Symbol,
    )
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False


# ── Symbols ───────────────────────────────────────────────────────────────────

if _SYMPY_AVAILABLE:
    s, t, u       = symbols("s t u", real=True)          # Mandelstam
    e_em          = symbols("e", positive=True)           # QED coupling
    g_s           = symbols("g_s", positive=True)         # QCD coupling
    alpha_em      = symbols("alpha", positive=True)       # α = e²/4π
    m_e, m_mu, m_q = symbols("m_e m_mu m_q", nonneg=True)
    cos_theta     = symbols("cos_theta", real=True)       # CM scattering angle


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class AmplitudeResult:
    process: str
    theory: str
    msq: object                          # sympy expression (or str if no sympy)
    msq_latex: str                       # LaTeX rendering
    description: str
    notes: str = ""

    def msq_at(self, s_val: float, t_val: float, u_val: float,
               e_val: float = 0.3028) -> Optional[float]:
        """Evaluate |M|² numerically at given kinematics (e ≈ √(4πα), α≈1/137)."""
        if not _SYMPY_AVAILABLE:
            return None
        try:
            expr = self.msq.subs({
                s: s_val, t: t_val, u: u_val, e_em: e_val,
            })
            return float(expr)
        except Exception:
            return None


# ── QED amplitudes ────────────────────────────────────────────────────────────

def _qed_ee_to_mumu() -> AmplitudeResult:
    """
    e⁺e⁻ → μ⁺μ⁻ via single photon exchange (massless limit, mₑ = mμ = 0).

    |M|² = 2e⁴(t² + u²)/s²   (spin-averaged over initial, summed over final)

    In terms of the CM scattering angle θ:
      t = -s/2(1-cosθ),  u = -s/2(1+cosθ)
      |M|² = e⁴(1 + cos²θ)
    """
    msq = 2 * e_em**4 * (t**2 + u**2) / s**2
    return AmplitudeResult(
        process="e+ e- -> mu+ mu-",
        theory="QED",
        msq=msq,
        msq_latex=latex(msq),
        description="Single s-channel photon exchange",
        notes="Massless limit (mₑ = mμ = 0). Mandelstam: s+t+u=0.",
    )


def _qed_bhabha() -> AmplitudeResult:
    """
    e⁺e⁻ → e⁺e⁻ (Bhabha scattering) — s-channel + t-channel photon.

    |M|² = 2e⁴[(s²+u²)/t² + 2su/(st) + (t²+u²)/s²]  (massless)
    """
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
        notes="Massless limit. Interference term included.",
    )


def _qed_compton() -> AmplitudeResult:
    """
    Compton scattering e⁻γ → e⁻γ (Klein-Nishina).

    |M|² = -2e⁴[s/u + u/s]  (massless electron limit, summed over polarisations)
    """
    msq = -2 * e_em**4 * (s / u + u / s)
    return AmplitudeResult(
        process="e- gamma -> e- gamma",
        theory="QED",
        msq=msq,
        msq_latex=latex(msq),
        description="Compton scattering: s-channel + u-channel electron exchange",
        notes="Massless electron limit. Mandelstam t = 0 for forward scattering.",
    )


def _qed_ee_to_ee() -> AmplitudeResult:
    """
    Møller scattering e⁻e⁻ → e⁻e⁻ — t-channel + u-channel photon.

    |M|² = 2e⁴[(s²+u²)/t² - 2s²/(tu) + (s²+t²)/u²]  (massless)
    """
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
        description="Møller scattering: t-channel + u-channel photon (identical fermions)",
        notes="Massless limit. Fermi statistics gives minus sign for exchange diagram.",
    )


# ── QCD amplitudes ────────────────────────────────────────────────────────────

def _qcd_qqbar_to_gg() -> AmplitudeResult:
    """
    qq̄ → gg at tree level (massless quarks).

    |M|²/Nc = (32/3)g_s⁴ [t/u + u/t - 9tu/(4s²)] × colour factor

    Colour-averaged, polarisation-summed result for SU(3):
    |M|² = (32/3) g_s⁴ [t/u + u/t - (9/4)(t² + u²)/s²]
    """
    msq = Rational(32, 3) * g_s**4 * (
        t / u + u / t - Rational(9, 4) * (t**2 + u**2) / s**2
    )
    return AmplitudeResult(
        process="q q~ -> g g",
        theory="QCD",
        msq=msq,
        msq_latex=latex(msq),
        description="qq̄ → gg: t-channel + u-channel quark + s-channel gluon",
        notes="SU(3) colour factors averaged over initial colours (Nq=3, Ng=8). Massless quarks.",
    )


# ── Dispatch table ────────────────────────────────────────────────────────────

_KNOWN: dict[tuple[str, str], AmplitudeResult] = {}


def _build_known():
    if not _SYMPY_AVAILABLE:
        return
    results = [
        _qed_ee_to_mumu(),
        _qed_bhabha(),
        _qed_compton(),
        _qed_ee_to_ee(),
        _qcd_qqbar_to_gg(),
    ]
    for r in results:
        _KNOWN[(r.process, r.theory)] = r
        # also register without particle-type arrows  e.g. "e+ e- -> mu+ mu-"
        key_bare = (r.process.replace("~", "~"), r.theory)
        _KNOWN[key_bare] = r


_build_known()


def get_amplitude(process: str, theory: str = "QED") -> Optional[AmplitudeResult]:
    """
    Return the spin-averaged |M|² for a known process, or None.

    Parameters
    ----------
    process : str   e.g. "e+ e- -> mu+ mu-"
    theory  : str   "QED" or "QCD"
    """
    if not _SYMPY_AVAILABLE:
        return None
    key = (process.strip(), theory.upper())
    return _KNOWN.get(key)


def list_supported_processes() -> list[dict]:
    """Return a list of processes for which amplitudes are pre-computed."""
    return [
        {"process": r.process, "theory": r.theory, "description": r.description}
        for r in _KNOWN.values()
    ]


# ── Kinematics helpers ────────────────────────────────────────────────────────

def mandelstam_from_cm(s_val: float, cos_theta_val: float) -> dict[str, float]:
    """
    Convert CM energy² and scattering angle to Mandelstam s, t, u
    in the massless limit.

    Parameters
    ----------
    s_val       : float  CM energy squared (GeV²), e.g. (100)² = 10000
    cos_theta_val : float  cosine of scattering angle

    Returns
    -------
    dict with keys 's', 't', 'u' (all in GeV²)
    """
    t_val = -s_val / 2.0 * (1.0 - cos_theta_val)
    u_val = -s_val / 2.0 * (1.0 + cos_theta_val)
    return {"s": s_val, "t": t_val, "u": u_val}
