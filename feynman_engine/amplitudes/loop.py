"""Loop amplitude evaluation via Passarino-Veltman reduction.

Architecture
------------
1. A 1-loop diagram produced by QGRAF carries a ``topology`` field set by the
   graph classifier (``'self-energy'``, ``'triangle'``, ``'box'``, etc.).
2. ``pv_reduce()`` reads the topology + internal particle content and maps the
   diagram to the correct PV scalar integral (B₀, C₀, or D₀) with the known
   kinematic arguments and analytic coefficients from standard QED/QCD results.
3. Each scalar integral is evaluated numerically by the LoopTools bridge.

Coefficient sources
-------------------
self-energy (photon VP):
    Σ_T(s) = (α/π)[2A₀(m²) − (4m²−s) B₀(s; m², m²)]   [Denner 1993, eq. C.2]
vertex correction (QED):
    δF₁(s) = (α/2π)[−B₀(m²;0,m²) + (4m²−s/2)/s C₀(m²,m²,s;0,m²,m²)]
box (QED):
    Involves D₀(0,0,0,0,s,t; 0,m_e²,0,m_μ²) + C₀ sub-diagrams.

References
----------
- Passarino & Veltman, Nucl. Phys. B160 (1979) 151
- Denner, Fortschr. Phys. 41 (1993) 307
- Hahn & Pérez-Victoria, Comput. Phys. Commun. 118 (1999) 153
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from sympy import Expr, Integer, Rational, Symbol, latex, pi, symbols

from feynman_engine.amplitudes.types import AmplitudeResult
from feynman_engine.core.models import Diagram


# ── Loop topology ─────────────────────────────────────────────────────────────

class LoopTopology(Enum):
    """PV scalar integral topology, determined by the number of internal propagators."""
    TADPOLE    = "tadpole"     # 0 external momenta, A₀  (rarely appears with no-tadpole filter)
    SELF_ENERGY = "self-energy" # 1 external momentum,  B₀
    TRIANGLE   = "triangle"    # 2 external momenta,    C₀
    BOX        = "box"         # 3 external momenta,    D₀
    PENTAGON   = "pentagon"    # 4 external momenta,    E₀  (rare at 1-loop)


def classify_loop_topology(n_internal_propagators: int) -> LoopTopology:
    """Map number of internal lines to PV topology for a 1-loop diagram."""
    if n_internal_propagators <= 1:
        return LoopTopology.TADPOLE
    elif n_internal_propagators == 2:
        return LoopTopology.SELF_ENERGY
    elif n_internal_propagators == 3:
        return LoopTopology.TRIANGLE
    elif n_internal_propagators == 4:
        return LoopTopology.BOX
    else:
        return LoopTopology.PENTAGON


# ── Scalar integral types ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class A0Integral:
    """Scalar tadpole: A₀(m²) = 16π²μ^{4-d} ∫ d^d l / [(2π)^d (l²-m²)]."""
    m_sq: object   # mass squared (SymPy expression or float)

    def __str__(self) -> str:
        return f"A0({self.m_sq})"

    def latex(self) -> str:
        return r"A_0\!\left(" + latex(self.m_sq) + r"\right)"

    def evaluate(self, mu_sq=1.0, delta_uv=0):
        """Evaluate using analytic closed-form formula."""
        from feynman_engine.amplitudes.analytic_integrals import analytic_A0
        return analytic_A0(self.m_sq, mu_sq=mu_sq, delta_uv=delta_uv)


@dataclass(frozen=True)
class B0Integral:
    """Scalar bubble: B₀(p², m₁², m₂²)."""
    p_sq: object   # external momentum squared
    m1_sq: object
    m2_sq: object

    def __str__(self) -> str:
        return f"B0({self.p_sq}, {self.m1_sq}, {self.m2_sq})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p_sq, self.m1_sq, self.m2_sq))
        return r"B_0\!\left(" + args + r"\right)"

    def evaluate(self, mu_sq=1.0, delta_uv=0):
        """Evaluate using analytic closed-form formula."""
        from feynman_engine.amplitudes.analytic_integrals import analytic_B0
        return analytic_B0(self.p_sq, self.m1_sq, self.m2_sq,
                           mu_sq=mu_sq, delta_uv=delta_uv)


@dataclass(frozen=True)
class C0Integral:
    """Scalar triangle: C₀(p₁², p₂², (p₁+p₂)², m₁², m₂², m₃²)."""
    p1_sq: object
    p2_sq: object
    p12_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in
                         (self.p1_sq, self.p2_sq, self.p12_sq,
                          self.m1_sq, self.m2_sq, self.m3_sq))
        return f"C0({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in
                         (self.p1_sq, self.p2_sq, self.p12_sq,
                          self.m1_sq, self.m2_sq, self.m3_sq))
        return r"C_0\!\left(" + args + r"\right)"

    def evaluate(self, mu_sq=1.0, delta_uv=0):
        """Evaluate using analytic formula.  Returns None if unsupported."""
        from feynman_engine.amplitudes.analytic_integrals import analytic_C0
        return analytic_C0(self.p1_sq, self.p2_sq, self.p12_sq,
                           self.m1_sq, self.m2_sq, self.m3_sq,
                           mu_sq=mu_sq, delta_uv=delta_uv)


@dataclass(frozen=True)
class B1Integral:
    """Coefficient of p^μ in B^μ = B₁ p^μ (2-point vector integral)."""
    p_sq: object
    m1_sq: object
    m2_sq: object

    def __str__(self) -> str:
        return f"B1({self.p_sq}, {self.m1_sq}, {self.m2_sq})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p_sq, self.m1_sq, self.m2_sq))
        return r"B_1\!\left(" + args + r"\right)"

    def evaluate(self, mu_sq=1.0, delta_uv=0):
        """Evaluate via PV reduction identity using A₀ and B₀."""
        from feynman_engine.amplitudes.analytic_integrals import analytic_B1
        return analytic_B1(self.p_sq, self.m1_sq, self.m2_sq,
                           mu_sq=mu_sq, delta_uv=delta_uv)


@dataclass(frozen=True)
class B00Integral:
    """Coefficient of g^μν in B^μν = B₀₀ g^μν + B₁₁ p^μp^ν."""
    p_sq: object
    m1_sq: object
    m2_sq: object

    def __str__(self) -> str:
        return f"B00({self.p_sq}, {self.m1_sq}, {self.m2_sq})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p_sq, self.m1_sq, self.m2_sq))
        return r"B_{00}\!\left(" + args + r"\right)"

    def evaluate(self, mu_sq=1.0, delta_uv=0):
        """Evaluate via PV reduction identity using A₀, B₀, and B₁."""
        from feynman_engine.amplitudes.analytic_integrals import analytic_B00
        return analytic_B00(self.p_sq, self.m1_sq, self.m2_sq,
                            mu_sq=mu_sq, delta_uv=delta_uv)


@dataclass(frozen=True)
class B11Integral:
    """Coefficient of p^μp^ν in B^μν = B₀₀ g^μν + B₁₁ p^μp^ν."""
    p_sq: object
    m1_sq: object
    m2_sq: object

    def __str__(self) -> str:
        return f"B11({self.p_sq}, {self.m1_sq}, {self.m2_sq})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p_sq, self.m1_sq, self.m2_sq))
        return r"B_{11}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class C1Integral:
    """Coefficient of p₁^μ in the triangle vector integral."""
    p1_sq: object
    p2_sq: object
    p12_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq))
        return f"C1({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq))
        return r"C_1\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class C2Integral:
    """Coefficient of p₂^μ in the triangle vector integral."""
    p1_sq: object
    p2_sq: object
    p12_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq))
        return f"C2({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq))
        return r"C_2\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class C00Integral:
    """Coefficient of g^μν in the rank-2 triangle C^μν."""
    p1_sq: object
    p2_sq: object
    p12_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq))
        return f"C00({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq))
        return r"C_{00}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class C11Integral:
    """Coefficient of p₁^μp₁^ν in the rank-2 triangle."""
    p1_sq: object
    p2_sq: object
    p12_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq))
        return f"C11({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq))
        return r"C_{11}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class C12Integral:
    """Coefficient of p₁^μp₂^ν + p₂^μp₁^ν in the rank-2 triangle."""
    p1_sq: object
    p2_sq: object
    p12_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq))
        return f"C12({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq))
        return r"C_{12}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class C22Integral:
    """Coefficient of p₂^μp₂^ν in the rank-2 triangle."""
    p1_sq: object
    p2_sq: object
    p12_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq))
        return f"C22({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p12_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq))
        return r"C_{22}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class D0Integral:
    """Scalar box: D₀(p₁², p₂², p₃², p₄², p₁₂², p₂₃², m₁², m₂², m₃², m₄²)."""
    p1_sq: object
    p2_sq: object
    p3_sq: object
    p4_sq: object
    p12_sq: object
    p23_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object
    m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in
                         (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                          self.p12_sq, self.p23_sq,
                          self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D0({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in
                         (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                          self.p12_sq, self.p23_sq,
                          self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_0\!\left(" + args + r"\right)"

    def evaluate(self, mu_sq=1.0, delta_uv=0):
        """Evaluate using analytic formula.  Returns None if unsupported."""
        from feynman_engine.amplitudes.analytic_integrals import analytic_D0
        return analytic_D0(self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                           self.p12_sq, self.p23_sq,
                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq,
                           mu_sq=mu_sq, delta_uv=delta_uv)


@dataclass(frozen=True)
class D00Integral:
    """Coefficient of g^μν in the rank-2 box D^μν."""
    p1_sq: object
    p2_sq: object
    p3_sq: object
    p4_sq: object
    p12_sq: object
    p23_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object
    m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                           self.p12_sq, self.p23_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D00({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                              self.p12_sq, self.p23_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_{00}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class D1Integral:
    """Coefficient of p₁^μ in the box vector integral."""
    p1_sq: object
    p2_sq: object
    p3_sq: object
    p4_sq: object
    p12_sq: object
    p23_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object
    m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                           self.p12_sq, self.p23_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D1({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                              self.p12_sq, self.p23_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_1\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class D2Integral:
    """Coefficient of p₂^μ in the box vector integral."""
    p1_sq: object
    p2_sq: object
    p3_sq: object
    p4_sq: object
    p12_sq: object
    p23_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object
    m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                           self.p12_sq, self.p23_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D2({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                              self.p12_sq, self.p23_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_2\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class D3Integral:
    """Coefficient of p₃^μ in the box vector integral."""
    p1_sq: object
    p2_sq: object
    p3_sq: object
    p4_sq: object
    p12_sq: object
    p23_sq: object
    m1_sq: object
    m2_sq: object
    m3_sq: object
    m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                           self.p12_sq, self.p23_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D3({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                              self.p12_sq, self.p23_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_3\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class D11Integral:
    """Coefficient of p₁^μp₁^ν in the rank-2 box integral."""
    p1_sq: object; p2_sq: object; p3_sq: object; p4_sq: object
    p12_sq: object; p23_sq: object
    m1_sq: object; m2_sq: object; m3_sq: object; m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                           self.p12_sq, self.p23_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D11({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                              self.p12_sq, self.p23_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_{11}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class D12Integral:
    """Coefficient of (p₁^μp₂^ν + p₂^μp₁^ν)/2 in the rank-2 box integral."""
    p1_sq: object; p2_sq: object; p3_sq: object; p4_sq: object
    p12_sq: object; p23_sq: object
    m1_sq: object; m2_sq: object; m3_sq: object; m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                           self.p12_sq, self.p23_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D12({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                              self.p12_sq, self.p23_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_{12}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class D13Integral:
    """Coefficient of (p₁^μp₃^ν + p₃^μp₁^ν)/2 in the rank-2 box integral."""
    p1_sq: object; p2_sq: object; p3_sq: object; p4_sq: object
    p12_sq: object; p23_sq: object
    m1_sq: object; m2_sq: object; m3_sq: object; m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                           self.p12_sq, self.p23_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D13({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                              self.p12_sq, self.p23_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_{13}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class D22Integral:
    """Coefficient of p₂^μp₂^ν in the rank-2 box integral."""
    p1_sq: object; p2_sq: object; p3_sq: object; p4_sq: object
    p12_sq: object; p23_sq: object
    m1_sq: object; m2_sq: object; m3_sq: object; m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                           self.p12_sq, self.p23_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D22({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                              self.p12_sq, self.p23_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_{22}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class D23Integral:
    """Coefficient of (p₂^μp₃^ν + p₃^μp₂^ν)/2 in the rank-2 box integral."""
    p1_sq: object; p2_sq: object; p3_sq: object; p4_sq: object
    p12_sq: object; p23_sq: object
    m1_sq: object; m2_sq: object; m3_sq: object; m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                           self.p12_sq, self.p23_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D23({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                              self.p12_sq, self.p23_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_{23}\!\left(" + args + r"\right)"


@dataclass(frozen=True)
class D33Integral:
    """Coefficient of p₃^μp₃^ν in the rank-2 box integral."""
    p1_sq: object; p2_sq: object; p3_sq: object; p4_sq: object
    p12_sq: object; p23_sq: object
    m1_sq: object; m2_sq: object; m3_sq: object; m4_sq: object

    def __str__(self) -> str:
        args = ", ".join(str(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                           self.p12_sq, self.p23_sq,
                                           self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return f"D33({args})"

    def latex(self) -> str:
        args = ", ".join(latex(x) for x in (self.p1_sq, self.p2_sq, self.p3_sq, self.p4_sq,
                                              self.p12_sq, self.p23_sq,
                                              self.m1_sq, self.m2_sq, self.m3_sq, self.m4_sq))
        return r"D_{33}\!\left(" + args + r"\right)"


# ── PV expansion ──────────────────────────────────────────────────────────────

@dataclass
class PVExpansion:
    """A loop amplitude expressed as a sum of PV scalar integrals.

    ``terms`` maps each scalar integral to its rational coefficient
    (a SymPy expression in external kinematics s, t, u, masses).
    """
    process: str
    diagram_id: int
    topology: LoopTopology
    terms: dict = field(default_factory=dict)   # {A0Integral|B0Integral|...: Expr}
    uv_divergent: bool = True
    ir_divergent: bool = False
    notes: list[str] = field(default_factory=list)

    def to_latex(self) -> str:
        if not self.terms:
            return r"\mathcal{M}_{\text{loop}} = 0"
        parts = []
        for integral, coeff in self.terms.items():
            coeff_str = latex(coeff)
            parts.append(rf"\left({coeff_str}\right) \cdot {integral.latex()}")
        body = r" \\ + ".join(parts)
        return rf"i\mathcal{{M}}_{{1\text{{-loop}}}} = {body}"

    def to_terms_list(self) -> list[dict]:
        """Return each PV term as a structured dict for API responses.

        Each entry contains:
          - integral_type: e.g. "B0", "C0", "D0"
          - integral_args: human-readable arguments string
          - integral_latex: LaTeX for the integral alone
          - coefficient_latex: LaTeX for the symbolic coefficient
          - coefficient_sympy: str representation of the SymPy coefficient
          - has_analytic_form: whether analytic evaluation is available
        """
        result = []
        for integral, coeff in self.terms.items():
            # Determine integral type name from class
            cls_name = type(integral).__name__  # e.g. "B0Integral"
            int_type = cls_name.replace("Integral", "")

            has_analytic = hasattr(integral, "evaluate")

            result.append({
                "integral_type": int_type,
                "integral_args": str(integral),
                "integral_latex": integral.latex(),
                "coefficient_latex": latex(coeff),
                "coefficient_sympy": str(coeff),
                "has_analytic_form": has_analytic,
            })
        return result

    def evaluate_analytic(self, kinematics: dict | None = None,
                          mu_sq: float = 1.0, delta_uv: float = 0) -> complex | None:
        """Evaluate the PV expansion using analytic integral formulas.

        Parameters
        ----------
        kinematics : dict, optional
            Substitution dict for symbolic coefficients, e.g.
            ``{s: 4.0, m_e**2: 0.000511**2}``.  If ``None``, coefficients
            must already be numeric.
        mu_sq : float
            Renormalisation scale μ².
        delta_uv : float
            UV pole value (0 for finite part).

        Returns
        -------
        complex or None
            The evaluated amplitude, or ``None`` if any integral lacks an
            analytic form (caller should fall back to LoopTools).
        """
        total = 0.0 + 0j
        for integral, coeff in self.terms.items():
            # Evaluate the coefficient
            if kinematics:
                coeff_val = complex(coeff.subs(kinematics).evalf())
            else:
                coeff_val = complex(coeff.evalf())

            # Evaluate the integral
            if not hasattr(integral, "evaluate"):
                return None  # No analytic form for this integral type
            int_val = integral.evaluate(mu_sq=mu_sq, delta_uv=delta_uv)
            if int_val is None:
                return None  # Unsupported kinematic configuration
            total += coeff_val * complex(int_val)
        return total

    def amplitude_result(self, theory: str) -> AmplitudeResult:
        """Wrap this expansion as an AmplitudeResult for the API response."""
        return AmplitudeResult(
            process=self.process,
            theory=theory,
            msq=Integer(0),          # numerical msq not yet computed
            msq_latex="",
            integral_latex=self.to_latex(),
            description=(
                f"1-loop {self.topology.value} diagram; "
                "PV-reduced to scalar integrals (numerical evaluation requires LoopTools)"
            ),
            notes=" | ".join(self.notes) if self.notes else
                  "UV renormalisation not yet applied; result is bare.",
            backend="pv-reduction",
        )


# ── Helper: particle classification ──────────────────────────────────────────

def _particle_is_fermion(particle_name: str, theory: str) -> bool:
    """Return True if the particle is a fermion/antifermion."""
    from feynman_engine.physics.registry import TheoryRegistry
    from feynman_engine.core.models import ParticleType
    try:
        p = TheoryRegistry.get_particle(theory, particle_name)
        return p.particle_type in (ParticleType.FERMION, ParticleType.ANTIFERMION)
    except Exception:
        name = particle_name.lower().rstrip("~+- ")
        return name in ("e", "mu", "tau", "u", "d", "s", "c", "b", "t",
                        "e-", "e+", "mu-", "mu+")


def _mass_val(particle_name: str, theory: str) -> object:
    """Return a SymPy symbol m_{name}² (positive) or 0 for massless particles."""
    from feynman_engine.physics.registry import TheoryRegistry
    try:
        p = TheoryRegistry.get_particle(theory, particle_name)
        if p.mass and p.mass not in ("0", ""):
            clean = particle_name.rstrip("+-~").replace("-", "_")
            return symbols(f"m_{clean}", positive=True) ** 2
    except Exception:
        pass
    return Integer(0)


# ── PV reduction with correct kinematic coefficients ──────────────────────────

def pv_reduce(diagram: Diagram, theory: str) -> Optional[PVExpansion]:
    """Reduce a 1-loop diagram to a linear combination of PV scalar integrals.

    Uses the QGRAF ``topology`` field to identify the physical loop structure
    (self-energy insertion, vertex correction, or box), then applies the known
    analytic PV coefficients for QED/QCD from standard references.

    Coefficient sources
    -------------------
    - Photon self-energy:   Σ_T = (α/π)[2A₀(m²)−(4m²−s)B₀(s;m²,m²)]  [Denner 1993]
    - QED vertex correction: δF₁ = (α/2π)[−B₀(m²;0,m²)+(4m²−s/2)/s×C₀]
    - QCD quark SE:          Σ_q  = (α_s C_F/4π)[A₀(m²)+(s+m²)B₀(s;0,m²)]
    - Box (QED):             D₀(0,0,0,0,s,t;0,m_e²,0,m_μ²) with coefficient −8α²tu
                             from Tr[p/₁γ^μp/₂γ^ρ]×Tr[p/₃γ_μp/₄γ_ρ]=−8tu (massless)

    Parameters
    ----------
    diagram : Diagram
        A 1-loop diagram from the QGRAF generator (``diagram.loop_order == 1``).
    theory : str
        Theory name (QED, QCD, EW, BSM).

    Returns
    -------
    PVExpansion or None
        None if the diagram is tree-level or has unsupported topology.
    """
    if diagram.loop_order != 1:
        return None

    # Try tensor reduction from Feynman rules first (Phase 3a).
    try:
        from feynman_engine.amplitudes.loop_tensor_reduction import auto_pv_reduce
        tensor_result = auto_pv_reduce(diagram, theory)
        if tensor_result is not None:
            return tensor_result
    except Exception:
        pass  # Fall through to scalar forms below

    s_sym, t_sym, u_sym = symbols("s t u", real=True)
    alpha_sym = symbols("alpha", positive=True)
    alpha_s_sym = symbols("alpha_s", positive=True)

    # QGRAF topology field: 'self-energy', 'triangle', 'box', etc.
    qgraf_topo = (diagram.topology or "unknown").lower()
    internals = diagram.internal_edges
    internal_names = [e.particle for e in internals]

    # Count fermions and gauge bosons in the internal lines to determine loop type.
    n_fermions = sum(1 for p in internal_names if _particle_is_fermion(p, theory))
    fermion_names = [p for p in internal_names if _particle_is_fermion(p, theory)]
    boson_names   = [p for p in internal_names if not _particle_is_fermion(p, theory)]

    # ── Self-energy diagrams ──────────────────────────────────────────────
    if "self-energy" in qgraf_topo or "selfenergy" in qgraf_topo:
        # Determine if it's a photon/gluon self-energy (fermion loop)
        # or a fermion self-energy (boson+fermion loop).
        if n_fermions >= 2 and len(boson_names) >= 1:
            # Likely a gauge-boson self-energy with a fermion loop.
            # Get the fermion mass (first fermion in the loop).
            m2 = _mass_val(fermion_names[0], theory)

            if theory in ("QED", "EW"):
                # Photon VP: Σ_T(s) = (α/π)[2A₀(m²) − (4m²−s) B₀(s;m²,m²)]
                a0_int = A0Integral(m_sq=m2)
                b0_int = B0Integral(p_sq=s_sym, m1_sq=m2, m2_sq=m2)
                coeff_a0 = 2 * alpha_sym / pi
                coeff_b0 = -(4 * m2 - s_sym) * alpha_sym / pi
                terms = {a0_int: coeff_a0, b0_int: coeff_b0}
                uv_div = True
                notes = [
                    "Photon vacuum polarisation: Σ_T(s) = (α/π)[2A₀(m²) − (4m²−s)B₀(s;m²,m²)]",
                    "From Denner (1993) eq. C.2; UV-divergent, renormalised by δZ₃ in MS-bar.",
                ]
            elif theory == "QCD":
                # Gluon self-energy (quark loop): similar formula with α_s C_F
                CF = Rational(4, 3)
                a0_int = A0Integral(m_sq=m2)
                b0_int = B0Integral(p_sq=s_sym, m1_sq=m2, m2_sq=m2)
                coeff_a0 = 2 * alpha_s_sym * CF / pi
                coeff_b0 = -(4 * m2 - s_sym) * alpha_s_sym * CF / pi
                terms = {a0_int: coeff_a0, b0_int: coeff_b0}
                uv_div = True
                notes = ["Gluon self-energy (quark loop), C_F=4/3, SU(3)"]
            else:
                terms = {B0Integral(p_sq=s_sym,
                                     m1_sq=_mass_val(internal_names[0], theory),
                                     m2_sq=_mass_val(internal_names[1], theory)): Integer(1)}
                uv_div = True
                notes = ["Unknown theory; coefficient placeholder = 1"]

            return PVExpansion(
                process=f"{diagram.process} (diagram {diagram.id})",
                diagram_id=diagram.id,
                topology=LoopTopology.SELF_ENERGY,
                terms=terms,
                uv_divergent=uv_div,
                notes=notes,
            )

        else:
            # Fermion self-energy: boson + fermion internal lines
            # Σ_f(p²) = (α/4π)[A₀(m²) + (p² + m²) B₀(p²; 0, m²)]
            m2 = _mass_val(fermion_names[0] if fermion_names else internal_names[0], theory)
            if theory == "QCD":
                CF = Rational(4, 3)
                prefactor = alpha_s_sym * CF / (4 * pi)
                notes = ["QCD quark self-energy (gluon loop); C_F=4/3"]
            else:
                prefactor = alpha_sym / (4 * pi)
                notes = ["QED fermion self-energy (photon loop); Feynman gauge"]
            a0_int = A0Integral(m_sq=m2)
            b0_int = B0Integral(p_sq=s_sym, m1_sq=Integer(0), m2_sq=m2)
            terms = {a0_int: prefactor, b0_int: prefactor * (s_sym + m2)}
            return PVExpansion(
                process=f"{diagram.process} (diagram {diagram.id})",
                diagram_id=diagram.id,
                topology=LoopTopology.SELF_ENERGY,
                terms=terms,
                uv_divergent=True,
                notes=notes,
            )

    # ── Triangle / vertex correction diagrams ─────────────────────────────
    elif "triangle" in qgraf_topo:
        # Vertex correction: 2 fermions + 1 photon in loop
        # δF₁(s) = (α/2π)[−B₀(m²;0,m²) + (4m²−s/2)/s × C₀(m²,m²,s;0,m²,m²)]
        m2 = _mass_val(fermion_names[0] if fermion_names else internal_names[0], theory)
        if theory == "QCD":
            CF = Rational(4, 3)
            prefactor = alpha_s_sym * CF / (2 * pi)
            notes = ["QCD quark-gluon vertex correction; C_F=4/3"]
        else:
            prefactor = alpha_sym / (2 * pi)
            notes = [
                "QED vertex form factor: δF₁(s) = (α/2π)[−B₀(m²;0,m²)+(4m²−s/2)/s×C₀(m²,m²,s;0,m²,m²)]",
                "IR-divergent when photon mass → 0; C₀ evaluated by LoopTools.",
            ]
        b0_int = B0Integral(p_sq=m2, m1_sq=Integer(0), m2_sq=m2)
        c0_int = C0Integral(p1_sq=m2, p2_sq=m2, p12_sq=s_sym,
                            m1_sq=Integer(0), m2_sq=m2, m3_sq=m2)
        coeff_b0 = -prefactor
        coeff_c0 = prefactor * (4 * m2 - s_sym / 2) / s_sym
        terms = {b0_int: coeff_b0, c0_int: coeff_c0}
        return PVExpansion(
            process=f"{diagram.process} (diagram {diagram.id})",
            diagram_id=diagram.id,
            topology=LoopTopology.TRIANGLE,
            terms=terms,
            uv_divergent=False,
            ir_divergent=True,
            notes=notes,
        )

    # ── Box diagrams ──────────────────────────────────────────────────────
    elif "box" in qgraf_topo:
        # Full box: D₀(0,0,0,0,s,t; 0,m_e²,0,m_μ²) for e+e-→μ+μ-
        #
        # Coefficient from Dirac algebra (massless external fermions):
        #   Tr[p/₁γ^μp/₂γ^ρ] × Tr[p/₃γ_μp/₄γ_ρ] = −8tu
        #   Loop normalisation: e⁴ = 16π²α² cancels the 1/(16π²) loop factor
        #   → c_D₀ = −8α²tu
        #
        # C₀ sub-diagrams from rank-2 PV reduction of the box tensor
        # (triangle topologies that arise from reducing the box numerator):
        #   c_C₀(s) = +4α²s,  c_C₀(t) = +4α²t
        #
        # Ref: Passarino & Veltman (1979) §5; Bern, Dixon & Kosower (1994)
        f_masses = (fermion_names + internal_names)[:4]
        m_sqs = [_mass_val(p, theory) for p in f_masses[:4]]
        m_sqs = (m_sqs + [Integer(0)] * 4)[:4]
        d0_int = D0Integral(
            p1_sq=Integer(0), p2_sq=Integer(0),
            p3_sq=Integer(0), p4_sq=Integer(0),
            p12_sq=s_sym, p23_sq=t_sym,
            m1_sq=m_sqs[0], m2_sq=m_sqs[1],
            m3_sq=m_sqs[2], m4_sq=m_sqs[3],
        )
        # C₀ sub-topologies from the rank-2 PV reduction
        c0_s = C0Integral(p1_sq=Integer(0), p2_sq=Integer(0), p12_sq=s_sym,
                          m1_sq=m_sqs[0], m2_sq=m_sqs[1], m3_sq=m_sqs[2])
        c0_t = C0Integral(p1_sq=Integer(0), p2_sq=Integer(0), p12_sq=t_sym,
                          m1_sq=m_sqs[0], m2_sq=m_sqs[1], m3_sq=m_sqs[2])

        if theory in ("QED", "EW"):
            # Full Dirac trace result for massless QED box
            c_d0 = -8 * alpha_sym**2 * t_sym * u_sym
            c_c0_s = 4 * alpha_sym**2 * s_sym
            c_c0_t = 4 * alpha_sym**2 * t_sym
            notes = [
                r"Box topology: D₀(0,0,0,0,s,t;m₁²,m₂²,m₃²,m₄²) with Dirac trace coefficient −8α²tu.",
                r"C₀ sub-diagrams from rank-2 PV reduction: +4α²s (s-channel) and +4α²t (t-channel).",
                r"Ref: Tr[p/₁γ^μp/₂γ^ρ]×Tr[p/₃γ_μp/₄γ_ρ] = −8tu (massless); α²=e⁴/(16π²).",
            ]
        elif theory == "QCD":
            # QCD box: same Dirac structure but with α_s² and SU(3) colour factor C_F²
            CF = Rational(4, 3)
            c_d0 = -8 * alpha_s_sym**2 * CF**2 * t_sym * u_sym
            c_c0_s = 4 * alpha_s_sym**2 * CF**2 * s_sym
            c_c0_t = 4 * alpha_s_sym**2 * CF**2 * t_sym
            notes = [
                r"QCD box: same Dirac structure as QED with α_s²C_F² colour factor (C_F=4/3).",
            ]
        else:
            c_d0 = Integer(1)
            c_c0_s = Integer(-1)
            c_c0_t = Integer(-1)
            notes = ["Box topology; coefficient placeholder=1 (unknown theory)."]

        terms = {
            d0_int: c_d0,
            c0_s:   c_c0_s,
            c0_t:   c_c0_t,
        }
        return PVExpansion(
            process=f"{diagram.process} (diagram {diagram.id})",
            diagram_id=diagram.id,
            topology=LoopTopology.BOX,
            terms=terms,
            uv_divergent=False,
            ir_divergent=True,
            notes=notes,
        )

    # ── Unknown topology ──────────────────────────────────────────────────
    else:
        # Use internal edge count as fallback
        topo = classify_loop_topology(len(internals))
        m_sqs = [_mass_val(e.particle, theory) for e in internals]

        if topo == LoopTopology.SELF_ENERGY:
            integral = B0Integral(p_sq=s_sym,
                                  m1_sq=m_sqs[0] if m_sqs else Integer(0),
                                  m2_sq=m_sqs[1] if len(m_sqs) > 1 else Integer(0))
        elif topo == LoopTopology.TRIANGLE:
            m_sqs = (m_sqs + [Integer(0)] * 3)[:3]
            integral = C0Integral(p1_sq=s_sym, p2_sq=t_sym, p12_sq=u_sym,
                                  m1_sq=m_sqs[0], m2_sq=m_sqs[1], m3_sq=m_sqs[2])
        else:
            m_sqs = (m_sqs + [Integer(0)] * 4)[:4]
            integral = D0Integral(
                p1_sq=Integer(0), p2_sq=Integer(0),
                p3_sq=Integer(0), p4_sq=Integer(0),
                p12_sq=s_sym, p23_sq=t_sym,
                m1_sq=m_sqs[0], m2_sq=m_sqs[1],
                m3_sq=m_sqs[2], m4_sq=m_sqs[3],
            )
        return PVExpansion(
            process=f"{diagram.process} (diagram {diagram.id})",
            diagram_id=diagram.id,
            topology=topo,
            terms={integral: Integer(1)},
            notes=[f"Fallback: topology='{diagram.topology}'; coefficient placeholder=1."],
        )


# ── Convenience entry point ───────────────────────────────────────────────────

def get_loop_amplitude(process: str, theory: str = "QED",
                       loops: int = 1) -> Optional[AmplitudeResult]:
    """PV-reduce 1-loop diagrams for ``process`` and return the first as a
    symbolic AmplitudeResult (in terms of A₀/B₀/C₀/D₀ scalar integrals).

    For numerical evaluation, use the /amplitude/loop-pv endpoint or call
    ``_evaluate_expansion_at()`` directly with explicit kinematics.
    """
    from feynman_engine.core.generator import generate_diagrams
    from feynman_engine.physics.translator import parse_process

    theory = theory.upper()
    spec = parse_process(process.strip(), theory=theory, loops=loops)
    diagrams = generate_diagrams(spec)
    loop_diags = [d for d in diagrams if d.loop_order == loops]
    if not loop_diags:
        return None

    # PV-reduce all diagrams; collect the first successful expansion.
    expansions: list[PVExpansion] = []
    for diag in loop_diags:
        exp = pv_reduce(diag, theory)
        if exp is not None:
            expansions.append(exp)

    if not expansions:
        return None

    # Return the symbolic PV-reduced expansion — exact as a function of
    # kinematic invariants (A₀/B₀/C₀/D₀ symbols).  Numerical evaluation at a
    # representative point used to be wrapped here as the result, but that
    # produced a single number that wasn't safe to feed into σ integration;
    # the /amplitude/loop-pv endpoint exposes the decomposition for users
    # who want to evaluate it themselves.
    first = expansions[0]
    return first.amplitude_result(theory)


def get_loop_pv_decomposition(
    process: str, theory: str = "QED", loops: int = 1,
) -> Optional[PVExpansion]:
    """Return the raw PVExpansion for the first 1-loop diagram of ``process``.

    Unlike ``get_loop_amplitude()`` which wraps the result as an
    ``AmplitudeResult``, this returns the full ``PVExpansion`` with its
    structured ``terms`` dict — useful for extracting individual PV integrals
    and their symbolic coefficients.
    """
    from feynman_engine.core.generator import generate_diagrams
    from feynman_engine.physics.translator import parse_process

    theory = theory.upper()
    spec = parse_process(process.strip(), theory=theory, loops=loops)
    diagrams = generate_diagrams(spec)
    loop_diags = [d for d in diagrams if d.loop_order == loops]
    if not loop_diags:
        return None

    for diag in loop_diags:
        exp = pv_reduce(diag, theory)
        if exp is not None:
            return exp
    return None


def _evaluate_expansion_at(
    expansion: PVExpansion,
    s_val: float = 10.0,
    t_val: float = -3.0,
    alpha_val: float = 1.0 / 137.036,
    alpha_s_val: float = 0.118,
) -> Optional[complex]:
    """Numerically evaluate a PVExpansion at a given kinematic point via LoopTools.

    All SymPy symbols in the coefficients are substituted before evaluation:
    s → s_val, t → t_val, u → -(s_val+t_val) (massless constraint),
    alpha → alpha_val, alpha_s → alpha_s_val.
    Symbolic mass² terms use their PDG values (m_e=0.511 MeV, m_mu=105.7 MeV, etc.).
    """
    from feynman_engine.amplitudes.looptools_bridge import A0, B0, C0, D0
    import math

    u_val = -(s_val + t_val)

    mass_defaults: dict[str, float] = {
        "m_e": (0.000511) ** 2,
        "m_mu": (0.10566) ** 2,
        "m_tau": (1.777) ** 2,
        "m_u": (0.0022) ** 2,
        "m_d": (0.0047) ** 2,
        "m_s": (0.096) ** 2,
        "m_c": (1.274) ** 2,
        "m_b": (4.183) ** 2,
        "m_t": (172.76) ** 2,
        "m_W": (80.377) ** 2,
        "m_Z": (91.1876) ** 2,
        "m_H": (125.20) ** 2,
    }

    # Build substitution dict (match by symbol name)
    from sympy import Symbol as Sym
    subs: dict = {
        "s": s_val, "t": t_val, "u": u_val,
        "alpha": alpha_val, "alpha_s": alpha_s_val,
    }
    subs.update(mass_defaults)

    def _eval_coeff(coeff) -> Optional[float]:
        """Evaluate a SymPy coefficient expression to float."""
        try:
            from sympy import Expr
            if isinstance(coeff, (int, float)):
                return float(coeff)
            # Substitute known symbols
            substitutions = {}
            for sym in coeff.free_symbols:
                name = sym.name
                if name in subs:
                    substitutions[sym] = subs[name]
                elif name.endswith("**2") or "^2" in name:
                    # Handle m_e^2-style symbol names
                    base = name.rstrip("**2").rstrip("^2")
                    if base in subs:
                        substitutions[sym] = subs[base]
            val = float(coeff.subs(substitutions))
            return val if math.isfinite(val) else None
        except Exception:
            return None

    def _eval_integral(integral) -> Optional[complex]:
        """Evaluate a PV integral using LoopTools."""
        from feynman_engine.amplitudes.looptools_bridge import (
            A0, B0, B1, B00, B11, C0, C1, C2, C00, C11, C12, C22,
            D0, D00, D1, D2, D3,
        )
        try:
            # ── 1-point ──
            if isinstance(integral, A0Integral):
                m = _eval_coeff(integral.m_sq)
                return A0(m if m is not None else 0.0)
            # ── 2-point ──
            elif isinstance(integral, B0Integral):
                p = _eval_coeff(integral.p_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                return B0(p or 0.0, m1 or 0.0, m2 or 0.0)
            elif isinstance(integral, B1Integral):
                p = _eval_coeff(integral.p_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                return B1(p or 0.0, m1 or 0.0, m2 or 0.0)
            elif isinstance(integral, B00Integral):
                p = _eval_coeff(integral.p_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                return B00(p or 0.0, m1 or 0.0, m2 or 0.0)
            elif isinstance(integral, B11Integral):
                p = _eval_coeff(integral.p_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                return B11(p or 0.0, m1 or 0.0, m2 or 0.0)
            # ── 3-point ──
            elif isinstance(integral, C0Integral):
                p1 = _eval_coeff(integral.p1_sq)
                p2 = _eval_coeff(integral.p2_sq)
                p12 = _eval_coeff(integral.p12_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq)
                return C0(p1 or 0.0, p2 or 0.0, p12 or 0.0, m1 or 0.0, m2 or 0.0, m3 or 0.0)
            elif isinstance(integral, C1Integral):
                p1 = _eval_coeff(integral.p1_sq)
                p2 = _eval_coeff(integral.p2_sq)
                p12 = _eval_coeff(integral.p12_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq)
                return C1(p1 or 0.0, p2 or 0.0, p12 or 0.0, m1 or 0.0, m2 or 0.0, m3 or 0.0)
            elif isinstance(integral, C2Integral):
                p1 = _eval_coeff(integral.p1_sq)
                p2 = _eval_coeff(integral.p2_sq)
                p12 = _eval_coeff(integral.p12_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq)
                return C2(p1 or 0.0, p2 or 0.0, p12 or 0.0, m1 or 0.0, m2 or 0.0, m3 or 0.0)
            elif isinstance(integral, C00Integral):
                p1 = _eval_coeff(integral.p1_sq)
                p2 = _eval_coeff(integral.p2_sq)
                p12 = _eval_coeff(integral.p12_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq)
                return C00(p1 or 0.0, p2 or 0.0, p12 or 0.0, m1 or 0.0, m2 or 0.0, m3 or 0.0)
            elif isinstance(integral, C11Integral):
                p1 = _eval_coeff(integral.p1_sq)
                p2 = _eval_coeff(integral.p2_sq)
                p12 = _eval_coeff(integral.p12_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq)
                return C11(p1 or 0.0, p2 or 0.0, p12 or 0.0, m1 or 0.0, m2 or 0.0, m3 or 0.0)
            elif isinstance(integral, C12Integral):
                p1 = _eval_coeff(integral.p1_sq)
                p2 = _eval_coeff(integral.p2_sq)
                p12 = _eval_coeff(integral.p12_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq)
                return C12(p1 or 0.0, p2 or 0.0, p12 or 0.0, m1 or 0.0, m2 or 0.0, m3 or 0.0)
            elif isinstance(integral, C22Integral):
                p1 = _eval_coeff(integral.p1_sq)
                p2 = _eval_coeff(integral.p2_sq)
                p12 = _eval_coeff(integral.p12_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq)
                return C22(p1 or 0.0, p2 or 0.0, p12 or 0.0, m1 or 0.0, m2 or 0.0, m3 or 0.0)
            # ── 4-point ──
            elif isinstance(integral, D0Integral):
                p1 = _eval_coeff(integral.p1_sq)
                p2 = _eval_coeff(integral.p2_sq)
                p3 = _eval_coeff(integral.p3_sq)
                p4 = _eval_coeff(integral.p4_sq)
                p12 = _eval_coeff(integral.p12_sq)
                p23 = _eval_coeff(integral.p23_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq)
                m4 = _eval_coeff(integral.m4_sq)
                return D0(p1 or 0.0, p2 or 0.0, p3 or 0.0, p4 or 0.0,
                          p12 or 0.0, p23 or 0.0,
                          m1 or 0.0, m2 or 0.0, m3 or 0.0, m4 or 0.0)
            elif isinstance(integral, D00Integral):
                p1 = _eval_coeff(integral.p1_sq)
                p2 = _eval_coeff(integral.p2_sq)
                p3 = _eval_coeff(integral.p3_sq)
                p4 = _eval_coeff(integral.p4_sq)
                p12 = _eval_coeff(integral.p12_sq)
                p23 = _eval_coeff(integral.p23_sq)
                m1 = _eval_coeff(integral.m1_sq)
                m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq)
                m4 = _eval_coeff(integral.m4_sq)
                return D00(p1 or 0.0, p2 or 0.0, p3 or 0.0, p4 or 0.0,
                           p12 or 0.0, p23 or 0.0,
                           m1 or 0.0, m2 or 0.0, m3 or 0.0, m4 or 0.0)
            elif isinstance(integral, D1Integral):
                p1 = _eval_coeff(integral.p1_sq); p2 = _eval_coeff(integral.p2_sq)
                p3 = _eval_coeff(integral.p3_sq); p4 = _eval_coeff(integral.p4_sq)
                p12 = _eval_coeff(integral.p12_sq); p23 = _eval_coeff(integral.p23_sq)
                m1 = _eval_coeff(integral.m1_sq); m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq); m4 = _eval_coeff(integral.m4_sq)
                return D1(p1 or 0.0, p2 or 0.0, p3 or 0.0, p4 or 0.0,
                          p12 or 0.0, p23 or 0.0,
                          m1 or 0.0, m2 or 0.0, m3 or 0.0, m4 or 0.0)
            elif isinstance(integral, D2Integral):
                p1 = _eval_coeff(integral.p1_sq); p2 = _eval_coeff(integral.p2_sq)
                p3 = _eval_coeff(integral.p3_sq); p4 = _eval_coeff(integral.p4_sq)
                p12 = _eval_coeff(integral.p12_sq); p23 = _eval_coeff(integral.p23_sq)
                m1 = _eval_coeff(integral.m1_sq); m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq); m4 = _eval_coeff(integral.m4_sq)
                return D2(p1 or 0.0, p2 or 0.0, p3 or 0.0, p4 or 0.0,
                          p12 or 0.0, p23 or 0.0,
                          m1 or 0.0, m2 or 0.0, m3 or 0.0, m4 or 0.0)
            elif isinstance(integral, D3Integral):
                p1 = _eval_coeff(integral.p1_sq); p2 = _eval_coeff(integral.p2_sq)
                p3 = _eval_coeff(integral.p3_sq); p4 = _eval_coeff(integral.p4_sq)
                p12 = _eval_coeff(integral.p12_sq); p23 = _eval_coeff(integral.p23_sq)
                m1 = _eval_coeff(integral.m1_sq); m2 = _eval_coeff(integral.m2_sq)
                m3 = _eval_coeff(integral.m3_sq); m4 = _eval_coeff(integral.m4_sq)
                return D3(p1 or 0.0, p2 or 0.0, p3 or 0.0, p4 or 0.0,
                          p12 or 0.0, p23 or 0.0,
                          m1 or 0.0, m2 or 0.0, m3 or 0.0, m4 or 0.0)
        except Exception:
            return None
        return None

    total = complex(0.0)
    for integral, coeff in expansion.terms.items():
        coeff_val = _eval_coeff(coeff)
        int_val = _eval_integral(integral)
        if coeff_val is None or int_val is None:
            return None
        total += coeff_val * int_val

    return total
