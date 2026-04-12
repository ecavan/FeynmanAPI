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

    def msq_at(
        self,
        s_val: float,
        t_val: float,
        u_val: float,
        e_val: float = 0.3028,
        g_s_val: float = 1.0,
    ) -> Optional[float]:
        """Evaluate |M|² numerically at given kinematics."""
        if not _SYMPY_AVAILABLE:
            return None
        try:
            substitutions = {
                Symbol("s"): s_val,
                Symbol("t"): t_val,
                Symbol("u"): u_val,
                Symbol("e"): e_val,
                Symbol("g_s"): g_s_val,
            }
            return float(self.msq.subs(substitutions))
        except Exception:
            return None
