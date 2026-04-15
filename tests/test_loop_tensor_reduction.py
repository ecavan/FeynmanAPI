"""Tests for automatic PV tensor reduction via FORM."""
from __future__ import annotations

import pytest
from sympy import Symbol, Integer, symbols

from feynman_engine.amplitudes.loop_tensor_reduction import (
    form_available,
    _run_form,
    _parse_named_expressions,
    _form_self_energy_reduce,
)
from feynman_engine.amplitudes.loop import (
    A0Integral, B0Integral, B1Integral, B00Integral, B11Integral,
    PVExpansion, LoopTopology,
)


pytestmark = pytest.mark.skipif(
    not form_available(), reason="FORM not installed"
)


# ── FORM execution ───────────────────────────────────────────────────────────

def test_form_available():
    """FORM binary should be found."""
    assert form_available()


def test_run_form_basic_trace():
    """FORM can compute a basic gamma trace."""
    program = """#-
Off Statistics;
Symbols s;
Vectors p1,p2;

Local tr = g_(1,p1)*g_(1,p2);
trace4,1;
id p1.p2 = s/2;
print tr;
.end
"""
    output = _run_form(program)
    assert output is not None
    exprs = _parse_named_expressions(output)
    assert "tr" in exprs
    assert exprs["tr"] == 2 * Symbol("s")


# ── FORM output parsing ─────────────────────────────────────────────────────

def test_parse_single_expression():
    """Parse a single named expression from FORM output."""
    form_output = """FORM 5.0.0
    #-

   result =
      8*s + 4*t;

  0.00 sec
"""
    exprs = _parse_named_expressions(form_output)
    assert "result" in exprs
    s, t = Symbol("s"), Symbol("t")
    assert exprs["result"] == 8 * s + 4 * t


def test_parse_multiple_expressions():
    """Parse multiple named expressions from FORM output."""
    form_output = """FORM 5.0.0
    #-

   expr1 =
      4*s;


   expr2 =
      -8*t + 2*s;

  0.00 sec
"""
    exprs = _parse_named_expressions(form_output)
    assert "expr1" in exprs
    assert "expr2" in exprs
    s, t = Symbol("s"), Symbol("t")
    assert exprs["expr1"] == 4 * s
    assert exprs["expr2"] == -8 * t + 2 * s


# ── Self-energy trace ────────────────────────────────────────────────────────

def test_self_energy_form_trace_massless():
    """Self-energy trace for massless fermions gives correct structure."""
    program = """#-
Off Statistics;
Symbols s,ll,lp;
Vectors l,p;
Indices mu;

Local gcontract = g_(1,mu)*(g_(1,l))*g_(1,mu)*(g_(1,l)+g_(1,p));
trace4,1;
id l.l = ll;
id l.p = lp;
id p.p = s;
print gcontract;
.sort

Local pcontract = g_(1,p)*(g_(1,l))*g_(1,p)*(g_(1,l)+g_(1,p));
trace4,1;
id l.l = ll;
id l.p = lp;
id p.p = s;
print pcontract;
.end
"""
    output = _run_form(program)
    assert output is not None
    exprs = _parse_named_expressions(output)

    ll, lp, s = Symbol("ll"), Symbol("lp"), Symbol("s")

    # Tr[γ^μ /l γ_μ (/l+/p)] = -2 Tr[/l (/l+/p)] = -8(l² + l·p)
    assert exprs["gcontract"] == -8 * ll - 8 * lp


def test_self_energy_form_trace_massive():
    """Self-energy trace for massive fermions includes mass terms."""
    program = """#-
Off Statistics;
Symbols s,ll,lp,me,me2;
Vectors l,p;
Indices mu;

Local gcontract = g_(1,mu)*(g_(1,l)+me*gi_(1))*g_(1,mu)*(g_(1,l)+g_(1,p)+me*gi_(1));
trace4,1;
id l.l = ll;
id l.p = lp;
id p.p = s;
id me^2 = me2;
print gcontract;
.end
"""
    output = _run_form(program)
    assert output is not None
    exprs = _parse_named_expressions(output)

    ll, lp, me2 = Symbol("ll"), Symbol("lp"), Symbol("me2")
    # Massive: -8(l² + l·p) + 16m² from mass insertions
    assert exprs["gcontract"] == -8 * ll - 8 * lp + 16 * me2


# ── PV mapping ───────────────────────────────────────────────────────────────

def test_self_energy_pv_has_correct_integrals():
    """Self-energy PV expansion should contain B-type and A0 integrals."""
    from feynman_engine.core.models import Diagram, Edge, Vertex

    diag = Diagram(
        id=1,
        process="e+ e- -> e+ e-",
        theory="EW",
        loop_order=1,
        topology="self-energy",
        vertices=[Vertex(id=0, particles=["e-", "gamma"]),
                  Vertex(id=1, particles=["e-", "gamma"])],
        edges=[
            Edge(id=0, start_vertex=0, end_vertex=1, particle="e-", is_external=False),
            Edge(id=1, start_vertex=1, end_vertex=0, particle="e-", is_external=False),
        ],
    )

    m_e_sq = Symbol("m_e", positive=True) ** 2
    result = _form_self_energy_reduce("me", "m_e", [m_e_sq, m_e_sq], diag, "EW")
    assert result is not None
    assert isinstance(result, PVExpansion)
    assert result.topology == LoopTopology.SELF_ENERGY
    assert len(result.terms) > 0

    integral_types = {type(k).__name__ for k in result.terms.keys()}
    # Should have at least B0 and A0
    assert "B0Integral" in integral_types or "A0Integral" in integral_types


def test_auto_pv_reduce_returns_none_for_qed():
    """auto_pv_reduce should return None for QED (Denner forms preferred)."""
    from feynman_engine.amplitudes.loop_tensor_reduction import auto_pv_reduce
    from feynman_engine.core.models import Diagram, Edge, Vertex

    diag = Diagram(
        id=1, process="e+ e- -> e+ e-", theory="QED",
        loop_order=1, topology="self-energy",
        vertices=[Vertex(id=0, particles=["e-", "gamma"]),
                  Vertex(id=1, particles=["e-", "gamma"])],
        edges=[
            Edge(id=0, start_vertex=0, end_vertex=1, particle="e-", is_external=False),
            Edge(id=1, start_vertex=1, end_vertex=0, particle="e-", is_external=False),
        ],
    )
    result = auto_pv_reduce(diag, "QED")
    assert result is None


def test_auto_pv_reduce_returns_none_for_tree():
    """auto_pv_reduce should return None for tree-level diagrams."""
    from feynman_engine.amplitudes.loop_tensor_reduction import auto_pv_reduce
    from feynman_engine.core.models import Diagram, Vertex

    diag = Diagram(
        id=1, process="e+ e- -> mu+ mu-", theory="EW",
        loop_order=0, topology="s-channel",
        vertices=[Vertex(id=0, particles=["e-", "e+", "gamma"])],
        edges=[],
    )
    result = auto_pv_reduce(diag, "EW")
    assert result is None


# ── Integration test: EW loop diagram ────────────────────────────────────────

def test_ew_loop_uses_form_tensor_reduction():
    """For EW theory, the FORM tensor reduction should be attempted."""
    from feynman_engine.core.generator import generate_diagrams
    from feynman_engine.physics.translator import parse_process
    from feynman_engine.amplitudes.loop import pv_reduce

    spec = parse_process("e+ e- -> mu+ mu-", theory="EW", loops=1)
    diagrams = generate_diagrams(spec)
    loop_diags = [d for d in diagrams if d.loop_order == 1]

    if not loop_diags:
        pytest.skip("No 1-loop EW diagrams generated")

    # At least some should get PV-reduced.
    reduced = 0
    for d in loop_diags:
        exp = pv_reduce(d, "EW")
        if exp is not None:
            reduced += 1
            assert len(exp.terms) > 0

    assert reduced > 0, "No EW loop diagrams were PV-reduced"
