"""Shared amplitude result types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from sympy import Symbol
    _SYMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is required in normal operation
    _SYMPY_AVAILABLE = False


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
