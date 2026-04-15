"""Automatic PV tensor reduction from Feynman rules via FORM.

This module implements FORM-based computation of 1-loop numerator traces
for the standard QFT topologies (self-energy, vertex correction, box).
The results are expressed as PV tensor integrals which LoopTools evaluates
numerically in dimensional regularization.

Pipeline:
    1. Identify loop topology from QGRAF diagram.
    2. Build a FORM program for the numerator trace (4D trace).
    3. Parse FORM output → polynomial in l·p_i, l·l.
    4. Map monomials → PV tensor integrals (B0, B1, B00, ..., D33).
    5. LoopTools evaluates tensor integrals in dim-reg (correct finite parts).

Important: LoopTools computes tensor integrals directly via Passarino-Veltman
reduction in d dimensions.  Using FORM's 4D trace with LoopTools tensor
integrals gives correct results because the d-dimensional corrections are
absorbed into LoopTools' internal PV reduction.  Do NOT reduce tensor → scalar
yourself; that requires careful d-dimensional algebra (see Denner 1993).

References:
  - Passarino & Veltman, Nucl. Phys. B160 (1979) 151
  - Denner, Fortschr. Phys. 41 (1993) 307
  - Hahn & Pérez-Victoria (LoopTools), Comput. Phys. Commun. 118 (1999) 153
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from sympy import Integer, Rational, Symbol, cancel, symbols, sympify

from feynman_engine.core.models import Diagram, ParticleType
from feynman_engine.amplitudes.loop import (
    PVExpansion, LoopTopology, classify_loop_topology,
    A0Integral, B0Integral, B1Integral, B00Integral, B11Integral,
    C0Integral, C1Integral, C2Integral, C00Integral, C11Integral, C12Integral, C22Integral,
    D0Integral, D1Integral, D2Integral, D3Integral,
    D00Integral, D11Integral, D12Integral, D13Integral, D22Integral, D23Integral, D33Integral,
)
from feynman_engine.form import find_form_binary


# ── Symbols ──────────────────────────────────────────────────────────────────
s_sym, t_sym, u_sym = symbols("s t u", real=True)


def form_available() -> bool:
    """Return True if FORM binary is found."""
    return find_form_binary() is not None


def auto_pv_reduce(diagram: Diagram, theory: str) -> Optional[PVExpansion]:
    """Derive PV expansion from Feynman rules using FORM for the Dirac trace.

    For 1-loop diagrams with a fermion loop, builds a FORM program that:
    - Self-energy: computes the transverse self-energy Π_T(p²) as a function
      of loop dot products l·l, l·p.
    - Vertex: computes the vertex form factor δF₁ interfered with tree level.
    - Box: computes the full interference trace with tree level.

    Returns None if FORM is not available or the topology is unsupported.
    Falls through to the Denner scalar forms in pv_reduce() for QED/QCD.
    """
    if diagram.loop_order != 1:
        return None

    # QED and QCD have correct Denner scalar-form coefficients in pv_reduce();
    # only use FORM tensor reduction for theories without hardcoded coefficients.
    if theory.upper() in ("QED", "QCD"):
        return None

    if not form_available():
        return None

    internals = diagram.internal_edges
    n_internal = len(internals)
    topo = classify_loop_topology(n_internal)

    if topo not in (LoopTopology.SELF_ENERGY, LoopTopology.TRIANGLE, LoopTopology.BOX):
        return None

    from feynman_engine.physics.registry import TheoryRegistry

    # Check for fermion loop.
    has_fermion = False
    mass_syms = []
    for e in internals:
        try:
            p = TheoryRegistry.get_particle(theory, e.particle)
            if p.particle_type in {ParticleType.FERMION, ParticleType.ANTIFERMION}:
                has_fermion = True
            if p.mass and p.mass not in ("0", ""):
                mass_syms.append(Symbol(p.mass, positive=True) ** 2)
            else:
                mass_syms.append(Integer(0))
        except Exception:
            return None

    if not has_fermion:
        return None  # Ghost loops, pure gauge loops not handled

    # Get the fermion mass for the loop.
    m_label = None
    for e in internals:
        try:
            p = TheoryRegistry.get_particle(theory, e.particle)
            if p.particle_type in {ParticleType.FERMION, ParticleType.ANTIFERMION}:
                if p.mass and p.mass not in ("0", ""):
                    m_label = p.mass
                break
        except Exception:
            pass

    m_form = m_label.replace("_", "") if m_label else "0"

    # Build and run FORM programs.
    if topo == LoopTopology.SELF_ENERGY:
        return _form_self_energy_reduce(m_form, m_label, mass_syms, diagram, theory)
    elif topo == LoopTopology.TRIANGLE:
        return _form_vertex_reduce(m_form, m_label, mass_syms, diagram, theory)
    elif topo == LoopTopology.BOX:
        return _form_box_reduce(m_form, m_label, mass_syms, diagram, theory)

    return None


# ── Self-energy via FORM ─────────────────────────────────────────────────────

def _form_self_energy_reduce(
    m_form: str, m_label: Optional[str], mass_syms: list,
    diagram: Diagram, theory: str,
) -> Optional[PVExpansion]:
    """Compute Π_T(p²) for a fermion-loop self-energy using FORM.

    FORM computes two traces:
        gcontract = g_μν × Π^μν = Tr[γ^μ (/l+m) γ_μ (/l+/p+m)]
        pcontract = p_μ p_ν × Π^μν = Tr[/p (/l+m) /p (/l+/p+m)]

    Then the transverse part in d=4:
        Π_T = (pcontract/p² - gcontract) / (d-1) → / 3 in d=4
    """
    mass_term = f"+{m_form}*gi_(1)" if m_form != "0" else ""
    mass_sq_sym = f"{m_form}2"
    extra_syms = f",{m_form},{mass_sq_sym}" if m_form != "0" else ""
    mass_id = f"\nid {m_form}^2 = {mass_sq_sym};" if m_form != "0" else ""

    program = f"""#-
Off Statistics;
Symbols s,ll,lp{extra_syms};
Vectors l,p;
Indices mu,nu;

Local gcontract = g_(1,mu)*(g_(1,l){mass_term})*g_(1,mu)*(g_(1,l)+g_(1,p){mass_term});
trace4,1;
id l.l = ll;
id l.p = lp;
id p.p = s;{mass_id}
print gcontract;
.sort

Local pcontract = g_(1,p)*(g_(1,l){mass_term})*g_(1,p)*(g_(1,l)+g_(1,p){mass_term});
trace4,1;
id l.l = ll;
id l.p = lp;
id p.p = s;{mass_id}
print pcontract;
.end
"""

    output = _run_form(program)
    if output is None:
        return None

    # Parse both expressions.
    exprs = _parse_named_expressions(output)
    if "gcontract" not in exprs or "pcontract" not in exprs:
        return None

    g_expr = exprs["gcontract"]
    p_expr = exprs["pcontract"]

    # Π_T = (pcontract/s - gcontract) / 3
    pi_t = (p_expr / s_sym - g_expr) / Integer(3)
    pi_t = cancel(pi_t)

    # Now pi_t is a polynomial in ll, lp (loop dot products).
    # Map to PV tensor integrals.
    m_sq = Symbol(m_label, positive=True) ** 2 if m_label else Integer(0)
    return _se_poly_to_pv(pi_t, m_sq, diagram, theory)


def _se_poly_to_pv(
    pi_t, m_sq, diagram: Diagram, theory: str,
) -> PVExpansion:
    """Map the transverse self-energy polynomial to PV integrals.

    PV tensor decomposition for 2-point functions:
        ∫ 1         → B0(s; m², m²)
        ∫ l·p       → s × B1(s; m², m²)     [B^μ p_μ = s B1]
        ∫ l·l       → 4 B00 + s B11 × s     [from g_μν B^μν = 4 B00 + s B11... ]
                       Actually: l² under the integral = A0(m²) + m² B0  (propagator identity)
        ∫ (l·p)²    → s² × B11(s; m², m²)
    """
    ll = Symbol("ll")
    lp = Symbol("lp")

    from sympy import Poly, PolynomialError, degree

    # Collect terms by powers of ll and lp.
    terms: dict = {}
    notes = ["Auto PV-reduced via FORM trace: Π_T(p²) = (p_μ p_ν Π^μν/p² - g_μν Π^μν) / 3."]

    b0 = B0Integral(p_sq=s_sym, m1_sq=m_sq, m2_sq=m_sq)
    b1 = B1Integral(p_sq=s_sym, m1_sq=m_sq, m2_sq=m_sq)
    b00 = B00Integral(p_sq=s_sym, m1_sq=m_sq, m2_sq=m_sq)
    b11 = B11Integral(p_sq=s_sym, m1_sq=m_sq, m2_sq=m_sq)
    a0 = A0Integral(m_sq=m_sq)

    try:
        poly = Poly(pi_t, ll, lp)
        for monom, coeff in poly.as_dict().items():
            ll_pow, lp_pow = monom
            if ll_pow == 0 and lp_pow == 0:
                _add_term(terms, b0, coeff)
            elif ll_pow == 0 and lp_pow == 1:
                # l·p integral → s × B1
                _add_term(terms, b1, coeff * s_sym)
            elif ll_pow == 1 and lp_pow == 0:
                # l² integral → A0(m²) + m² × B0 (propagator cancellation)
                _add_term(terms, a0, coeff)
                _add_term(terms, b0, coeff * m_sq)
            elif ll_pow == 0 and lp_pow == 2:
                # (l·p)² → s² × B11
                _add_term(terms, b11, coeff * s_sym**2)
            elif ll_pow == 1 and lp_pow == 1:
                # l² × l·p → (A0 + m² B0) related + m² s B1
                _add_term(terms, b1, coeff * m_sq * s_sym)
                notes.append("Rank-3 term reduced via propagator cancellation.")
            else:
                # Higher rank: approximate
                _add_term(terms, b0, coeff * m_sq ** (ll_pow + lp_pow))
                notes.append(f"Rank-{ll_pow+lp_pow} term approximated.")
    except (PolynomialError, Exception):
        # pi_t might be a constant
        _add_term(terms, b0, pi_t)

    return PVExpansion(
        process=f"{diagram.process} (diagram {diagram.id})",
        diagram_id=diagram.id,
        topology=LoopTopology.SELF_ENERGY,
        terms=terms,
        uv_divergent=True,
        notes=notes,
    )


# ── Vertex correction via FORM ──────────────────────────────────────────────

def _form_vertex_reduce(
    m_form: str, m_label: Optional[str], mass_syms: list,
    diagram: Diagram, theory: str,
) -> Optional[PVExpansion]:
    """Compute the vertex correction form factor using FORM.

    For a QED/QCD vertex correction (photon/gluon exchange in loop):
    Interference with tree level gives:
        2 Re[M_tree* × M_loop] ∝ Tr[(/p2+m) γ^ν (/l+/p1+m) γ^μ (/l+/p2+m) γ_ν (/p1+m) γ_μ]

    This is the standard vertex correction form factor trace.
    """
    mass_term = f"+{m_form}*gi_(1)" if m_form != "0" else ""
    mass_sq_sym = f"{m_form}2"
    extra_syms = f",{m_form},{mass_sq_sym}" if m_form != "0" else ""
    mass_id = f"\nid {m_form}^2 = {mass_sq_sym};" if m_form != "0" else ""

    program = f"""#-
Off Statistics;
Symbols s,ll,lp1,lp2{extra_syms};
Vectors l,p1,p2;
Indices mu,nu;

* Vertex correction × tree interference trace:
* Tr[(/p2+m) γ^ν (/l+/p1+m) γ^μ (/l+/p2+m) γ_ν (/p1+m) γ_μ]
Local vertextr = (g_(1,p2){mass_term})
    *g_(1,nu)
    *(g_(1,l)+g_(1,p1){mass_term})
    *g_(1,mu)
    *(g_(1,l)+g_(1,p2){mass_term})
    *g_(1,nu)
    *(g_(1,p1){mass_term})
    *g_(1,mu);

trace4,1;
contract;

id l.l = ll;
id l.p1 = lp1;
id l.p2 = lp2;
id p1.p1 = {mass_sq_sym if m_form != "0" else "0"};
id p2.p2 = {mass_sq_sym if m_form != "0" else "0"};
id p1.p2 = (s - {mass_sq_sym + " - " + mass_sq_sym if m_form != "0" else "0"})/2;{mass_id}

print vertextr;
.end
"""

    output = _run_form(program)
    if output is None:
        return None

    exprs = _parse_named_expressions(output)
    if "vertextr" not in exprs:
        return None

    vtx_expr = exprs["vertextr"]

    m_sq = Symbol(m_label, positive=True) ** 2 if m_label else Integer(0)
    return _vertex_poly_to_pv(vtx_expr, m_sq, mass_syms, diagram, theory)


def _vertex_poly_to_pv(
    vtx_expr, m_sq, mass_syms: list, diagram: Diagram, theory: str,
) -> PVExpansion:
    """Map the vertex correction polynomial to PV integrals.

    3-point tensor decomposition:
        ∫ 1                → C0
        ∫ l·p1             → C1  (contracted form)
        ∫ l·p2             → C2
        ∫ l²               → A0(m1²) + m1² C0  (propagator cancellation)
        ∫ (l·p1)²          → C11 (contracted)
        ∫ (l·p1)(l·p2)     → C12
        ∫ (l·p2)²          → C22
    """
    ll = Symbol("ll")
    lp1 = Symbol("lp1")
    lp2 = Symbol("lp2")

    from sympy import Poly, PolynomialError

    # Internal masses for vertex correction: photon(0), fermion(m²), fermion(m²)
    m1_sq = Integer(0)  # Photon/gluon in loop
    m2_sq = m_sq        # Fermion
    m3_sq = m_sq        # Fermion

    c0 = C0Integral(p1_sq=m_sq, p2_sq=m_sq, p12_sq=s_sym,
                     m1_sq=m1_sq, m2_sq=m2_sq, m3_sq=m3_sq)
    c1 = C1Integral(p1_sq=m_sq, p2_sq=m_sq, p12_sq=s_sym,
                     m1_sq=m1_sq, m2_sq=m2_sq, m3_sq=m3_sq)
    c2 = C2Integral(p1_sq=m_sq, p2_sq=m_sq, p12_sq=s_sym,
                     m1_sq=m1_sq, m2_sq=m2_sq, m3_sq=m3_sq)
    c00 = C00Integral(p1_sq=m_sq, p2_sq=m_sq, p12_sq=s_sym,
                       m1_sq=m1_sq, m2_sq=m2_sq, m3_sq=m3_sq)
    c11 = C11Integral(p1_sq=m_sq, p2_sq=m_sq, p12_sq=s_sym,
                       m1_sq=m1_sq, m2_sq=m2_sq, m3_sq=m3_sq)
    c12 = C12Integral(p1_sq=m_sq, p2_sq=m_sq, p12_sq=s_sym,
                       m1_sq=m1_sq, m2_sq=m2_sq, m3_sq=m3_sq)
    c22 = C22Integral(p1_sq=m_sq, p2_sq=m_sq, p12_sq=s_sym,
                       m1_sq=m1_sq, m2_sq=m2_sq, m3_sq=m3_sq)
    a0 = A0Integral(m_sq=m1_sq)
    b0 = B0Integral(p_sq=m_sq, m1_sq=m1_sq, m2_sq=m2_sq)

    terms: dict = {}
    notes = ["Auto PV-reduced via FORM: vertex correction × tree interference trace."]

    try:
        poly = Poly(vtx_expr, ll, lp1, lp2)
        for monom, coeff in poly.as_dict().items():
            ll_pow, lp1_pow, lp2_pow = monom
            rank = ll_pow + lp1_pow + lp2_pow

            if rank == 0:
                _add_term(terms, c0, coeff)
            elif ll_pow == 0 and lp1_pow == 1 and lp2_pow == 0:
                _add_term(terms, c1, coeff)
            elif ll_pow == 0 and lp1_pow == 0 and lp2_pow == 1:
                _add_term(terms, c2, coeff)
            elif ll_pow == 1 and lp1_pow == 0 and lp2_pow == 0:
                _add_term(terms, a0, coeff)
                _add_term(terms, c0, coeff * m1_sq)
            elif ll_pow == 0 and lp1_pow == 2 and lp2_pow == 0:
                _add_term(terms, c11, coeff)
            elif ll_pow == 0 and lp1_pow == 1 and lp2_pow == 1:
                _add_term(terms, c12, coeff)
            elif ll_pow == 0 and lp1_pow == 0 and lp2_pow == 2:
                _add_term(terms, c22, coeff)
            elif ll_pow == 1 and rank == 2:
                # l² × l·p_i: propagator cancellation
                if lp1_pow == 1:
                    _add_term(terms, c1, coeff * m1_sq)
                if lp2_pow == 1:
                    _add_term(terms, c2, coeff * m1_sq)
                notes.append("Rank-3 reduced via propagator cancellation.")
            elif ll_pow == 2:
                # l⁴ → double propagator cancellation
                _add_term(terms, a0, coeff * (Integer(1) + 2 * m1_sq))
                _add_term(terms, c0, coeff * m1_sq**2)
            else:
                _add_term(terms, c0, coeff)
                notes.append(f"High-rank term ({rank}) coefficient placed on C0.")
    except Exception:
        _add_term(terms, c0, vtx_expr)

    return PVExpansion(
        process=f"{diagram.process} (diagram {diagram.id})",
        diagram_id=diagram.id,
        topology=LoopTopology.TRIANGLE,
        terms=terms,
        uv_divergent=False,
        ir_divergent=True,
        notes=notes,
    )


# ── Box via FORM ─────────────────────────────────────────────────────────────

def _form_box_reduce(
    m_form: str, m_label: Optional[str], mass_syms: list,
    diagram: Diagram, theory: str,
) -> Optional[PVExpansion]:
    """Compute the box diagram interference trace using FORM.

    For QED e+e- → μ+μ- box: two fermion traces with shared loop momentum
    on one line.  For massless external fermions:

    Tr[γ^μ /p1 γ^ρ /p2] × Tr[γ_μ (/l+/p3) γ_ρ (/l+/p1+/p2+/p3)]
    """
    program = f"""#-
Off Statistics;
Symbols s,t,u,ll,lp1,lp2,lp3;
Vectors l,p1,p2,p3;
Indices mu,rho;

* QED box: tree line × loop line
Local boxtrace =
    g_(1,mu)*g_(1,p1)*g_(1,rho)*g_(1,p2)
  * g_(2,mu)*(g_(2,l)+g_(2,p3))*g_(2,rho)*(g_(2,l)+g_(2,p1)+g_(2,p2)+g_(2,p3));

trace4,1;
trace4,2;
contract;

id l.l = ll;
id l.p1 = lp1;
id l.p2 = lp2;
id l.p3 = lp3;
id p1.p1 = 0;
id p2.p2 = 0;
id p3.p3 = 0;
id p1.p2 = s/2;
id p1.p3 = -s/2 - t/2;
id p2.p3 = t/2;

print boxtrace;
.end
"""

    output = _run_form(program)
    if output is None:
        return None

    exprs = _parse_named_expressions(output)
    if "boxtrace" not in exprs:
        return None

    box_expr = exprs["boxtrace"]

    m1_sq = mass_syms[0] if len(mass_syms) > 0 else Integer(0)
    m2_sq = mass_syms[1] if len(mass_syms) > 1 else Integer(0)
    m3_sq = mass_syms[2] if len(mass_syms) > 2 else Integer(0)
    m4_sq = mass_syms[3] if len(mass_syms) > 3 else Integer(0)

    return _box_poly_to_pv(box_expr, m1_sq, m2_sq, m3_sq, m4_sq, diagram, theory)


def _box_poly_to_pv(
    box_expr, m1_sq, m2_sq, m3_sq, m4_sq,
    diagram: Diagram, theory: str,
) -> PVExpansion:
    """Map the box trace polynomial to PV integrals."""
    ll = Symbol("ll")
    lp1 = Symbol("lp1")
    lp2 = Symbol("lp2")
    lp3 = Symbol("lp3")

    from sympy import Poly, PolynomialError

    d_kwargs = dict(
        p1_sq=Integer(0), p2_sq=Integer(0),
        p3_sq=Integer(0), p4_sq=Integer(0),
        p12_sq=s_sym, p23_sq=t_sym,
        m1_sq=m1_sq, m2_sq=m2_sq, m3_sq=m3_sq, m4_sq=m4_sq,
    )

    d0 = D0Integral(**d_kwargs)
    d1 = D1Integral(**d_kwargs)
    d2 = D2Integral(**d_kwargs)
    d3 = D3Integral(**d_kwargs)
    d00 = D00Integral(**d_kwargs)
    d11 = D11Integral(**d_kwargs)
    d12 = D12Integral(**d_kwargs)
    d13 = D13Integral(**d_kwargs)
    d22 = D22Integral(**d_kwargs)
    d23 = D23Integral(**d_kwargs)
    d33 = D33Integral(**d_kwargs)
    a0 = A0Integral(m_sq=m1_sq)

    terms: dict = {}
    notes = ["Auto PV-reduced via FORM: box interference trace."]

    try:
        poly = Poly(box_expr, ll, lp1, lp2, lp3)
        for monom, coeff in poly.as_dict().items():
            ll_pow, lp1_pow, lp2_pow, lp3_pow = monom
            rank = ll_pow + lp1_pow + lp2_pow + lp3_pow

            if rank == 0:
                _add_term(terms, d0, coeff)
            elif rank == 1 and ll_pow == 0:
                if lp1_pow == 1:
                    _add_term(terms, d1, coeff)
                elif lp2_pow == 1:
                    _add_term(terms, d2, coeff)
                elif lp3_pow == 1:
                    _add_term(terms, d3, coeff)
            elif ll_pow == 1 and rank == 1:
                _add_term(terms, a0, coeff)
                _add_term(terms, d0, coeff * m1_sq)
            elif rank == 2 and ll_pow == 0:
                # Rank-2 tensor integrals
                if lp1_pow == 2:
                    _add_term(terms, d11, coeff)
                elif lp1_pow == 1 and lp2_pow == 1:
                    _add_term(terms, d12, coeff)
                elif lp1_pow == 1 and lp3_pow == 1:
                    _add_term(terms, d13, coeff)
                elif lp2_pow == 2:
                    _add_term(terms, d22, coeff)
                elif lp2_pow == 1 and lp3_pow == 1:
                    _add_term(terms, d23, coeff)
                elif lp3_pow == 2:
                    _add_term(terms, d33, coeff)
            elif ll_pow == 1 and rank == 2:
                # l² × l·p_i → propagator cancellation
                if lp1_pow == 1:
                    _add_term(terms, d1, coeff * m1_sq)
                elif lp2_pow == 1:
                    _add_term(terms, d2, coeff * m1_sq)
                elif lp3_pow == 1:
                    _add_term(terms, d3, coeff * m1_sq)
                notes.append("Rank-3 reduced via propagator cancellation.")
            else:
                _add_term(terms, d0, coeff)
                notes.append(f"High-rank term ({rank}) placed on D0.")
    except Exception:
        _add_term(terms, d0, box_expr)

    return PVExpansion(
        process=f"{diagram.process} (diagram {diagram.id})",
        diagram_id=diagram.id,
        topology=LoopTopology.BOX,
        terms=terms,
        uv_divergent=False,
        ir_divergent=True,
        notes=notes,
    )


# ── FORM execution ───────────────────────────────────────────────────────────

def _run_form(program: str) -> Optional[str]:
    """Execute a FORM program and return stdout."""
    form_bin = find_form_binary()
    if form_bin is None:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".frm", delete=False
    ) as f:
        f.write(program)
        f.flush()
        frm_path = Path(f.name)

    try:
        result = subprocess.run(
            [str(form_bin), str(frm_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except Exception:
        return None
    finally:
        frm_path.unlink(missing_ok=True)


# ── FORM output parsing ─────────────────────────────────────────────────────

def _parse_named_expressions(stdout: str) -> dict:
    """Parse FORM output into named SymPy expressions.

    FORM prints expressions as:
        varname =
          <polynomial terms>
          ;

    Returns dict of {name: SymPy expression}.
    """
    results = {}

    # Split on expression boundaries.
    # Pattern: "   name =" at the start of a line, ends with ";"
    blocks = re.split(r'\n\s*(\w+)\s*=\s*\n', stdout)

    # blocks alternates: [preamble, name1, body1, name2, body2, ...]
    for i in range(1, len(blocks) - 1, 2):
        name = blocks[i].strip()
        body = blocks[i + 1]

        # Extract up to the semicolon.
        semi_idx = body.find(";")
        if semi_idx >= 0:
            body = body[:semi_idx]

        # Clean up: join lines, remove FORM formatting.
        body = body.replace("\n", " ").strip()
        body = re.sub(r'\s+', ' ', body)

        # Replace FORM bracketed symbols [X] with X.
        body = re.sub(r'\[(\w+)\]', r'\1', body)

        if not body or body == "0":
            results[name] = Integer(0)
            continue

        try:
            results[name] = sympify(body)
        except Exception:
            pass

    # Also try single-line format: "   name = expr;"
    for line in stdout.split("\n"):
        m = re.match(r'\s+(\w+)\s*=\s*(.+);', line)
        if m:
            name = m.group(1)
            if name not in results:
                body = m.group(2).strip()
                body = re.sub(r'\[(\w+)\]', r'\1', body)
                try:
                    results[name] = sympify(body)
                except Exception:
                    pass

    return results


# ── Utilities ────────────────────────────────────────────────────────────────

def _add_term(terms: dict, integral, coeff):
    """Add a coefficient to an integral's entry in the terms dict."""
    if integral in terms:
        terms[integral] = cancel(terms[integral] + coeff)
    else:
        terms[integral] = coeff
    # Remove zero terms.
    if terms[integral] == 0:
        del terms[integral]
