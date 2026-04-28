"""Shared amplitude result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    from sympy import Symbol
    _SYMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is required in normal operation
    _SYMPY_AVAILABLE = False


# Capability flags — what *can* a downstream consumer trust this amplitude for?
# All surviving backends (curated formulas, FORM traces, SymPy symbolic, and
# LoopTools-numerical loop amplitudes) produce real |M|²(s,t,u,...) functions
# that can be integrated over phase space.  The single-point "approximate-
# pointwise" backend was removed — its outputs were not safe to integrate.
_FEATURES_BY_LEVEL: dict[str, dict[str, bool]] = {
    "exact-symbolic": {
        "is_function_of_kinematics": True,
        "cross_section_integration":  True,
        "differential_distribution":  True,
        "monte_carlo":                True,
        "nlo_running_kfactor":        True,
        "trustworthy_value":          True,
    },
    "looptools-numerical": {
        "is_function_of_kinematics": True,
        "cross_section_integration":  True,
        "differential_distribution":  True,
        "monte_carlo":                True,
        "nlo_running_kfactor":        True,
        "trustworthy_value":          True,
    },
}


@dataclass
class AmplitudeResult:
    process: str
    theory: str
    msq: object
    msq_latex: str
    description: str
    integral_latex: str | None = None
    notes: str = ""
    backend: str = "unknown"
    approximation_level: str = "exact-symbolic"
    evaluation_point: dict | None = None

    @property
    def is_symbolic_function(self) -> bool:
        """True for every surviving backend — kept for backwards compatibility."""
        return True

    @property
    def features(self) -> dict[str, bool]:
        """What this amplitude can be used for.  See _FEATURES_BY_LEVEL above."""
        return dict(_FEATURES_BY_LEVEL.get(
            self.approximation_level,
            _FEATURES_BY_LEVEL["exact-symbolic"],
        ))

    @property
    def trustworthy_for_cross_section(self) -> bool:
        """One-line check used by σ integrators to refuse unsafe inputs."""
        return self.features["cross_section_integration"]

    def to_api_dict(self) -> dict:
        """Convert to a dict suitable for API/frontend consumption."""
        return {
            "process": self.process,
            "theory": self.theory,
            "description": self.description,
            "notes": self.notes,
            "backend": self.backend,
            "approximation_level": self.approximation_level,
            "evaluation_point": self.evaluation_point,
            "features": self.features,
            "is_symbolic_function": True,
            "msq": self.msq,
            "msq_latex": self.msq_latex,
            "integral_latex": self.integral_latex,
        }

    def msq_at(
        self,
        s_val: float,
        t_val: float,
        u_val: float,
        e_val: float = 0.3028,
        g_s_val: float = 1.0,
        g_Z_val: float = 0.7434,      # g_W/cos(θ_W) at m_Z
        sin2_W_val: float = 0.2312,   # sin²(θ_W) PDG 2023
        m_Z_val: float = 91.1876,     # GeV
        m_H_val: float = 125.20,      # GeV
        m_W_val: float = 80.377,      # GeV
    ) -> Optional[float]:
        """Evaluate |M|² numerically at given kinematics.

        EW parameter defaults are PDG 2023 central values.
        For QED/QCD processes the EW symbols do not appear and the defaults are ignored.
        """
        if not _SYMPY_AVAILABLE:
            return None
        try:
            # Build a name→value map, then match against actual free symbols
            # (SymPy treats symbols with different assumptions as distinct objects,
            # so we resolve by name rather than by identity).
            # Particle masses in GeV (PDG 2024).
            # Light quark masses are current-quark MS-bar values but only
            # matter for massive-kinematics expressions; for most 2→2
            # processes they are negligible and set to ≈0 anyway.
            name_map: dict[str, float] = {
                "s":      s_val,
                "t":      t_val,
                "u":      u_val,
                "e":      e_val,
                "g_s":    g_s_val,
                "g_Z":    g_Z_val,
                "sin2_W": sin2_W_val,
                "m_Z":    m_Z_val,
                "m_H":    m_H_val,
                "m_W":    m_W_val,
                # Lepton masses
                "m_e":    0.000511,
                "m_mu":   0.10566,
                "m_tau":  1.7768,
                # Quark masses (PDG current-quark, MS-bar at 2 GeV)
                "m_u":    0.00216,
                "m_d":    0.00467,
                "m_s":    0.0934,
                "m_c":    1.27,
                "m_b":    4.18,
                "m_t":    172.69,
            }
            substitutions = {
                sym: name_map[sym.name]
                for sym in self.msq.free_symbols
                if sym.name in name_map
            }
            return float(self.msq.subs(substitutions))
        except Exception:
            return None
