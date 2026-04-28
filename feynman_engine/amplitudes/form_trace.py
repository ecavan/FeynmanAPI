"""FORM-based symbolic trace computation for tree-level 2→2 amplitudes.

Uses the FORM symbolic algebra program to compute Dirac gamma-matrix traces
and Lorentz contractions, then applies coupling constants, propagator
denominators, and spin/color averaging in Python.

This module is an alternative to the SymPy-based ``symbolic.py`` backend.
It handles QCD color algebra (via the ``color`` module) which the SymPy
backend cannot.

The pipeline:
    1. QGRAF diagrams → FORM program (gamma traces + kinematics)
    2. Execute FORM → polynomial in Mandelstam variables
    3. Parse FORM output → SymPy expression
    4. Apply couplings, denominators, color factors → AmplitudeResult
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from sympy import Integer, Rational, Symbol, cancel, latex, symbols, sympify

from feynman_engine.amplitudes.types import AmplitudeResult
from feynman_engine.core.generator import generate_diagrams
from feynman_engine.core.models import Diagram, Edge, ParticleType
from feynman_engine.form import find_form_binary
from feynman_engine.physics.registry import TheoryRegistry
from feynman_engine.physics.translator import parse_process


# ── Mandelstam symbols ────────────────────────────────────────────────────────
s_sym, t_sym, u_sym = symbols("s t u", real=True)

_SUPPORTED_TOPOLOGIES = {"s-channel", "t-channel", "u-channel"}


def _to_form_name(name: str) -> str:
    """Sanitize a symbol name for FORM (no underscores allowed)."""
    return name.replace("_", "")


def _from_form_name(form_name: str, originals: set[str]) -> str:
    """Reverse-map a FORM name back to the original SymPy symbol name."""
    for orig in originals:
        if _to_form_name(orig) == form_name:
            return orig
    return form_name


def form_available() -> bool:
    """Return True if a FORM binary is accessible."""
    return find_form_binary() is not None


# ── Public API ────────────────────────────────────────────────────────────────

def get_form_amplitude(process: str, theory: str = "QED") -> Optional[AmplitudeResult]:
    """Compute spin-summed |M|² for a tree-level process using FORM.

    Handles:
    - All 2→2 QCD processes including those with 3-gluon vertices
      (qq̄→gg, qg→qg) using physical polarization sums and SU(3) color.
    - 2→3 QED processes: ff̄→f'f̄'γ (bremsstrahlung).

    Returns None if FORM is unavailable or the process is unsupported.
    """
    if not form_available():
        return None

    theory = theory.upper()
    spec = parse_process(process.strip(), theory=theory, loops=0)

    if len(spec.incoming) != 2:
        return None

    # 2→3 processes.
    if len(spec.outgoing) == 3:
        return _get_2to3_qed_amplitude(spec, theory)

    if len(spec.outgoing) != 2:
        return None

    diagrams = generate_diagrams(spec)
    if not diagrams:
        return None

    tree = [d for d in diagrams if d.loop_order == 0 and d.topology in _SUPPORTED_TOPOLOGIES]
    if not tree:
        return None

    # Try two-fermion-line analysis (boson exchange between two fermion pairs).
    analyzed = []
    for d in tree:
        info = _analyze_diagram(d)
        if info is not None:
            analyzed.append((d, info))

    if analyzed:
        return _run_form_pipeline(analyzed, spec, theory, "two-fermion-line")

    # Try single-fermion-line analysis (Compton-type: 2 fermions + 2 bosons).
    # Only use FORM if ALL tree diagrams are successfully analyzed — a partial
    # set gives wrong results (missing diagram contributions to |M|²).
    compton = []
    all_compton_ok = True
    for d in tree:
        info = _analyze_compton_diagram(d)
        if info is not None:
            # For QCD, set light quark masses (u,d,s) to zero — they are
            # negligible at any collider energy and zeroing them preserves
            # exact flavour universality (matching curated/PYTHIA results).
            if theory == "QCD" and info.fermion_mass_label in {"m_u", "m_d", "m_s"}:
                info.fermion_mass_label = None
                info.in_mass_label = None
                info.out_mass_label = None
                info.prop_mass_label = None
            compton.append((d, info))
        else:
            all_compton_ok = False

    if compton and all_compton_ok:
        return _run_form_pipeline(compton, spec, theory, "single-fermion-line")

    # Try QCD gluon-vertex analysis (qq̄→gg, qg→qg with 3-gluon vertices).
    if theory == "QCD":
        gluon_result = _try_qcd_gluon_vertex(spec, tree, theory)
        if gluon_result is not None:
            return gluon_result

    return None


def get_form_decay(process: str, theory: str = "EW") -> Optional[AmplitudeResult]:
    """Compute spin-summed |M|² and decay width Γ for a 1→2 tree-level decay.

    Handles: V→ff (Z→ll, W→lν), S→ff (H→bb, H→ττ), V→VV (future).
    Returns None if the process is unsupported.
    """
    theory = theory.upper()
    spec = parse_process(process.strip(), theory=theory, loops=0)

    if len(spec.incoming) != 1 or len(spec.outgoing) != 2:
        return None

    diagrams = generate_diagrams(spec)
    if not diagrams:
        return None

    tree = [d for d in diagrams if d.loop_order == 0]
    if len(tree) != 1:
        return None  # tree-level 1→2 should be a single diagram

    diag = tree[0]

    # Classify the parent particle.
    parent_name = spec.incoming[0]
    parent = TheoryRegistry.get_particle(theory, parent_name)

    # Classify the daughters.
    d1_name, d2_name = spec.outgoing[0], spec.outgoing[1]
    d1 = TheoryRegistry.get_particle(theory, d1_name)
    d2 = TheoryRegistry.get_particle(theory, d2_name)

    if not parent.mass or parent.mass == "0":
        return None  # massless particles don't decay

    M = Symbol(parent.mass)  # parent mass symbol

    # Determine decay type.
    d1_is_fermion = d1.particle_type in {ParticleType.FERMION, ParticleType.ANTIFERMION}
    d2_is_fermion = d2.particle_type in {ParticleType.FERMION, ParticleType.ANTIFERMION}

    if d1_is_fermion and d2_is_fermion:
        is_vector = parent.propagator_style.value in {"photon", "boson", "gluon", "charged boson"}
        is_scalar = parent.propagator_style.value in {"scalar", "dashed"}

        if is_vector:
            return _decay_vector_to_ff(parent, d1, d2, M, theory, spec)
        elif is_scalar:
            return _decay_scalar_to_ff(parent, d1, d2, M, theory, spec)

    return None


def _decay_vector_to_ff(parent, d1, d2, M, theory, spec) -> AmplitudeResult:
    """Compute V → f f̄ decay (Z→ll, W→lν, etc.).

    Spin-summed |M|²:
        Σ |M|² = g² × Tr[(/q1 + m1) γ^μ (/q2 - m2) γ^ν] × (-g_μν + p_μp_ν/M²)

    For massless daughters:
        Σ |M|² = g² × 8(q1·q2) × (1 + ...) = g² × 4M²

    Spin-averaged: divide by (2J+1) = 3 for vector parent.

    Decay width: Γ = |p*|/(8πM²) × |M̄|²
    where |p*| = (1/2M)√(M² - (m1+m2)²)√(M² - (m1-m2)²).
    """
    from sympy import sqrt, pi as sym_pi

    m1_label = _particle_mass_label(theory, d1.name)
    m2_label = _particle_mass_label(theory, d2.name)
    m1 = Symbol(m1_label) if m1_label else Integer(0)
    m2 = Symbol(m2_label) if m2_label else Integer(0)

    # Coupling: use the standard vertex coupling.
    g = _coupling_for_vertex(theory, parent.name, d1.name)

    # Trace for V→ff: Tr[(/q1+m1) γ^μ (/q2∓m2) γ^ν] × (-g_μν + p_μ p_ν/M²)
    #
    # 1→2 kinematics: p = q1 + q2, so  q1·q2 = (M² - m1² - m2²)/2
    # The trace Tr[(/q1+m1) γ^μ (/q2-m2) γ^ν] = 4(q1^μ q2^ν + q1^ν q2^μ - g^μν(q1·q2 - m1 m2))
    #
    # Contract with (-g_μν + p_μ p_ν/M²):
    # -g_μν part: -4(2(q1·q2) - 4(q1·q2 - m1 m2)) = -4(-2(q1·q2) + 4 m1 m2)
    #   = 8(q1·q2) - 16 m1 m2
    # Wait, let me be more careful.
    #
    # T^μν = 4[q1^μ q2^ν + q1^ν q2^μ - g^μν(q1·q2 - m1·m2)]
    # -g_μν × T^μν = -4[2(q1·q2) - 4(q1·q2 - m1 m2)] = -4[2q1·q2 - 4q1·q2 + 4m1m2]
    #              = -4[-2q1·q2 + 4m1m2] = 8(q1·q2) - 16 m1 m2
    #
    # p_μ p_ν/M² × T^μν = 4/M² [2(q1·p)(q2·p) - (p·p)(q1·q2 - m1m2)]
    #                    = 4/M² [2(q1·p)(q2·p) - M²(q1·q2 - m1m2)]
    #
    # In rest frame: q1·p = E1 M, q2·p = E2 M, E1+E2 = M
    # q1·p = E1 M = (M² + m1² - m2²)/2
    # q2·p = E2 M = (M² + m2² - m1²)/2
    #
    # The full contraction:
    # Σ|M|² = g² × {8(q1·q2) - 16 m1 m2 + 4/M² × [2 (q1·p)(q2·p) - M²(q1·q2-m1m2)]}

    q1q2 = (M**2 - m1**2 - m2**2) / 2
    q1p = (M**2 + m1**2 - m2**2) / 2
    q2p = (M**2 + m2**2 - m1**2) / 2

    # -g_μν contraction
    gterm = 8 * q1q2 - 16 * m1 * m2

    # p_μ p_ν / M² contraction
    pterm = Rational(4, 1) / M**2 * (2 * q1p * q2p - M**2 * (q1q2 - m1 * m2))

    trace_contracted = g**2 * (gterm + pterm)

    # Spin average: divide by (2J+1) = 3 for vector parent.
    # Color factor: Nc = 3 for quarks in final state, 1 for leptons.
    color_mult = Integer(1)
    if d1.color and d1.color not in {"1", "singlet"}:
        color_mult = Integer(3)  # quark final state

    msq_avg = trace_contracted * color_mult / Integer(3)
    msq_avg = cancel(msq_avg)

    # Decay width: Γ = |p*| / (8π M²) × |M̄|²
    # |p*|² = [M² - (m1+m2)²][M² - (m1-m2)²] / (4M²)
    # |p*| = √(...) / (2M)
    pstar_sq = ((M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)) / (4 * M**2)
    pstar = sqrt(pstar_sq)
    width = pstar / (8 * sym_pi * M**2) * msq_avg
    width = cancel(width)

    return AmplitudeResult(
        process=spec.raw,
        theory=theory,
        msq=msq_avg,
        msq_latex=latex(msq_avg),
        integral_latex=latex(width),
        description=f"Tree-level {parent.name}→{d1.name} {d2.name} decay: spin-averaged |M̄|² and partial width Γ",
        notes=f"Γ = {latex(width)}\nSpin-averaged over parent (2J+1=3). Color factor: {color_mult}.",
        backend="form-decay",
    )


def _decay_scalar_to_ff(parent, d1, d2, M, theory, spec) -> AmplitudeResult:
    """Compute S → f f̄ decay (H→bb, H→ττ, etc.).

    Spin-summed |M|²:
        Σ |M|² = y² × Tr[(/q1 + m1)(/q2 - m2)] = y² × 4(q1·q2 - m1 m2)

    For Yukawa coupling y_f = m_f √2 / v, the standard result:
        Σ |M|² = 2 m_f² M_H² / v² × (1 - 4m_f²/M_H²)  ... wait, let me use generic coupling.

    Spin-averaged: divide by (2J+1) = 1 for scalar parent.
    """
    from sympy import sqrt, pi as sym_pi

    m1_label = _particle_mass_label(theory, d1.name)
    m2_label = _particle_mass_label(theory, d2.name)
    m1 = Symbol(m1_label) if m1_label else Integer(0)
    m2 = Symbol(m2_label) if m2_label else Integer(0)

    # Coupling: Yukawa.
    # For the Higgs, the coupling is y_f = m_f/v, but we use a generic symbol.
    y = Symbol(f"y_{d1.name.rstrip('+-').replace('~','')}")

    # Trace: Tr[(/q1+m1)(/q2-m2)] = 4(q1·q2 - m1 m2)
    q1q2 = (M**2 - m1**2 - m2**2) / 2

    trace = 4 * (q1q2 - m1 * m2)
    # = 4 × [(M²-m1²-m2²)/2 - m1 m2]
    # = 2M² - 2m1² - 2m2² - 4m1m2
    # For m1=m2=m_f: = 2M² - 4m_f² - 4m_f² = 2(M² - 4m_f²)... wait
    # = 2M² - 2m² - 2m² - 4m² = 2M² - 8m² ... that's wrong.
    # Actually: 4[(M²-2m²)/2 - m²] = 4[M²/2 - m² - m²] = 2M² - 8m²
    # Hmm, that doesn't match standard. Let me recalculate.
    # For S→f f̄: the amplitude is M = y ū(q1) v(q2).
    # |M|² spin-summed = y² Tr[(/q1+m)(/q2-m)]  (note: v-spinor gives (/q2-m))
    #                  = y² × 4(q1·q2 - m²)
    #                  = y² × 4[(M²-2m²)/2 - m²]
    #                  = y² × 4[M²/2 - m² - m²]
    #                  = y² × (2M² - 8m²)
    # Standard result: Σ|M|² = y² × 2(M² - 4m²). Hmm close but off.
    # Wait: q1·q2 = (p² - q1² - q2²)/2 = (M² - m1² - m2²)/2
    # For m1=m2=m: q1·q2 = (M² - 2m²)/2
    # Tr[(/q1+m)(/q2-m)] = Tr[/q1 /q2 - m /q1 + m /q2 - m²]
    #                     = 4q1·q2 - 4m² (using Tr[/a /b] = 4a·b, Tr[/a] = 0, Tr[1] = 4)
    # Wait, no: Tr[/q1 /q2] = 4q1·q2, Tr[m /q2] = 0, Tr[-m /q1] = 0, Tr[-m²I] = -4m²
    # Total = 4(q1·q2 - m²) = 4[(M²-2m²)/2 - m²] = 4[M²/2 - 2m²] = 2M² - 8m²
    # But standard: Γ(H→ff) ∝ m_f²(1-4m_f²/M_H²)^{3/2}
    # Hmm, the |M|² = y_f² × 2(M² - 4m²) and then phase space gives the β^3.
    # Actually 2M² - 8m² = 2(M² - 4m²). ✓ Good, that matches.

    msq = y**2 * trace

    # Color factor for quarks.
    color_mult = Integer(1)
    if d1.color and d1.color not in {"1", "singlet"}:
        color_mult = Integer(3)

    # Spin average: 1 for scalar parent (2J+1 = 1).
    msq_avg = msq * color_mult
    msq_avg = cancel(msq_avg)

    # Decay width.
    pstar_sq = ((M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)) / (4 * M**2)
    pstar = sqrt(pstar_sq)
    width = pstar / (8 * sym_pi * M**2) * msq_avg
    width = cancel(width)

    return AmplitudeResult(
        process=spec.raw,
        theory=theory,
        msq=msq_avg,
        msq_latex=latex(msq_avg),
        integral_latex=latex(width),
        description=f"Tree-level {parent.name}→{d1.name} {d2.name} decay: |M̄|² and partial width Γ",
        notes=f"Γ = {latex(width)}\nYukawa coupling y = m_f/v (Standard Model). Color factor: {color_mult}.",
        backend="form-decay",
    )


def _run_form_pipeline(
    analyzed: list[tuple[Diagram, object]],
    spec: object,
    theory: str,
    mode: str,
) -> Optional[AmplitudeResult]:
    """Common FORM pipeline: build program, run, parse, assemble."""
    all_mass_labels: set[str] = set()
    for _, info in analyzed:
        if info.in_mass_label:
            all_mass_labels.add(info.in_mass_label)
        if info.out_mass_label:
            all_mass_labels.add(info.out_mass_label)
        if hasattr(info, "fermion_mass_label") and info.fermion_mass_label:
            all_mass_labels.add(info.fermion_mass_label)

    if mode == "single-fermion-line":
        program = _build_compton_form_program(analyzed, theory)
    else:
        program = _build_form_program(analyzed, theory)

    raw_output = _run_form(program)
    if raw_output is None:
        return None

    parsed = _parse_form_output(raw_output, all_mass_labels)
    if not parsed:
        return None

    if mode == "single-fermion-line":
        msq = _assemble_compton_msq(analyzed, parsed, theory)
    else:
        msq = _assemble_msq(analyzed, parsed, theory)

    if msq is None or msq == 0:
        return None

    # For Compton-type (single-fermion-line) processes, the cross-term trace
    # produces t(s+t+u) factors that only vanish under the Mandelstam
    # constraint.  Apply t = Σm² - s - u to simplify.
    # Two-fermion-line traces simplify naturally without this.
    if mode == "single-fermion-line":
        mass_sum_sq = Integer(0)
        for pname in spec.incoming + spec.outgoing:
            p_info = TheoryRegistry.get_particle(theory, pname)
            if p_info.mass and p_info.mass != "0":
                mass_sum_sq += Symbol(p_info.mass) ** 2
        t_val = mass_sum_sq - s_sym - u_sym
        msq = msq.subs(Symbol("t", real=True), t_val)

    msq = cancel(msq)

    return AmplitudeResult(
        process=spec.raw,
        theory=theory,
        msq=msq,
        msq_latex=latex(msq),
        integral_latex=None,
        description="Exact tree-level |M̄|² computed via FORM trace + QGRAF diagrams",
        notes="FORM-computed Dirac traces with analytic SU(N) color factors",
        backend="form-symbolic",
    )


# ── Diagram analysis ─────────────────────────────────────────────────────────

class _DiagramInfo:
    """Extracted information needed to build FORM traces."""

    __slots__ = (
        "topology", "mediator_name", "mediator_mass_label",
        "coupling_in", "coupling_out",
        "in_fermion_mom", "in_antifermion_mom",
        "out_fermion_mom", "out_antifermion_mom",
        "in_mass_label", "out_mass_label",
        "symmetry_factor",
    )

    def __init__(self):
        pass


def _analyze_diagram(diagram: Diagram) -> Optional[_DiagramInfo]:
    """Extract trace-relevant data from a QGRAF diagram.

    Only handles tree-level 2→2 with a single internal vector/scalar boson.
    """
    internals = diagram.internal_edges
    if len(internals) != 1:
        return None

    internal = internals[0]
    mediator = TheoryRegistry.get_particle(diagram.theory, internal.particle)

    # Only vector boson exchange for now.
    if mediator.propagator_style.value not in {
        "photon", "boson", "gluon", "charged boson",
    }:
        return None

    info = _DiagramInfo()
    info.mediator_name = mediator.name
    info.mediator_mass_label = mediator.mass if mediator.mass and mediator.mass != "0" else None
    info.topology = diagram.topology
    info.symmetry_factor = diagram.symmetry_factor or 1.0

    if diagram.topology == "s-channel":
        return _analyze_s_channel(diagram, info)
    elif diagram.topology in {"t-channel", "u-channel"}:
        return _analyze_tu_channel(diagram, internal, info)

    return None


def _analyze_s_channel(diagram: Diagram, info: _DiagramInfo) -> Optional[_DiagramInfo]:
    """Extract fermion momenta for s-channel topology."""
    incoming = [e for e in diagram.external_edges if e.start_vertex < 0]
    outgoing = [e for e in diagram.external_edges if e.end_vertex < 0]

    if len(incoming) != 2 or len(outgoing) != 2:
        return None

    # Classify by fermion/antifermion.
    in_ferm, in_anti = _split_fermion_pair(diagram.theory, incoming)
    out_ferm, out_anti = _split_fermion_pair(diagram.theory, outgoing)
    if any(x is None for x in (in_ferm, in_anti, out_ferm, out_anti)):
        return None

    info.in_fermion_mom = in_ferm.momentum
    info.in_antifermion_mom = in_anti.momentum
    info.out_fermion_mom = out_ferm.momentum
    info.out_antifermion_mom = out_anti.momentum
    info.in_mass_label = _particle_mass_label(diagram.theory, in_ferm.particle)
    info.out_mass_label = _particle_mass_label(diagram.theory, out_ferm.particle)
    info.coupling_in = _coupling_for_vertex(diagram.theory, info.mediator_name, in_ferm.particle)
    info.coupling_out = _coupling_for_vertex(diagram.theory, info.mediator_name, out_ferm.particle)
    return info


def _analyze_tu_channel(diagram: Diagram, internal: Edge, info: _DiagramInfo) -> Optional[_DiagramInfo]:
    """Extract fermion momenta for t/u-channel topology."""
    v_start = internal.start_vertex
    v_end = internal.end_vertex

    def _vertex_ext(vid: int) -> tuple[list[Edge], list[Edge]]:
        in_e = [e for e in diagram.external_edges if e.end_vertex == vid and e.start_vertex < 0]
        out_e = [e for e in diagram.external_edges if e.start_vertex == vid and e.end_vertex < 0]
        return in_e, out_e

    in_A, out_A = _vertex_ext(v_start)
    in_B, out_B = _vertex_ext(v_end)

    if not (len(in_A) == 1 and len(out_A) == 1 and len(in_B) == 1 and len(out_B) == 1):
        return None

    # Reject non-fermion external legs.
    for e in in_A + out_A + in_B + out_B:
        p = TheoryRegistry.get_particle(diagram.theory, e.particle)
        if p.particle_type not in {ParticleType.FERMION, ParticleType.ANTIFERMION}:
            return None

    # Line A: incoming fermion at start vertex → outgoing at same vertex.
    # Line B: incoming fermion at end vertex → outgoing at same vertex.
    info.in_fermion_mom = in_A[0].momentum
    info.in_antifermion_mom = out_A[0].momentum  # "antifermion" slot = outgoing at vertex A
    info.out_fermion_mom = in_B[0].momentum
    info.out_antifermion_mom = out_B[0].momentum
    info.in_mass_label = _particle_mass_label(diagram.theory, in_A[0].particle)
    info.out_mass_label = _particle_mass_label(diagram.theory, in_B[0].particle)
    info.coupling_in = _coupling_for_vertex(diagram.theory, info.mediator_name, in_A[0].particle)
    info.coupling_out = _coupling_for_vertex(diagram.theory, info.mediator_name, in_B[0].particle)
    return info


# ── Compton-type diagram analysis ────────────────────────────────────────────

class _ComptonInfo:
    """Info for single-fermion-line diagrams (Compton, pair annihilation, etc.).

    These have 2 external fermions + 2 external bosons, with 1 internal fermion
    propagator.  The squared amplitude is a single trace on one fermion line.
    """
    __slots__ = (
        "topology", "coupling_in", "coupling_out",
        "fermion_mass_label", "in_mass_label", "out_mass_label",
        "symmetry_factor",
        # Fermion line traversal: ordered list of FORM trace elements.
        # Each element is (type, data) where type is "slash", "gamma", "prop".
        "trace_elements",
        # Propagator denominator info.
        "prop_mandelstam",  # "s", "t", or "u"
        "prop_mass_label",  # mass label for internal fermion
        # How many incoming bosons (for spin averaging).
        "n_incoming_bosons",
    )

    def __init__(self):
        self.in_mass_label = None
        self.out_mass_label = None


def _analyze_compton_diagram(diagram: Diagram) -> Optional[_ComptonInfo]:
    """Analyze a diagram with 1 internal fermion + 2 external fermions + 2 external bosons.

    Handles: e-γ→e-γ, e+e-→γγ, γγ→e+e- and QCD equivalents.
    """
    internals = diagram.internal_edges
    if len(internals) != 1:
        return None

    internal = internals[0]
    int_particle = TheoryRegistry.get_particle(diagram.theory, internal.particle)

    # Internal edge must be a fermion propagator.
    if int_particle.particle_type not in {ParticleType.FERMION, ParticleType.ANTIFERMION}:
        return None

    # External edges: exactly 2 fermions/antifermions and 2 bosons.
    ext_fermions = []
    ext_bosons = []
    for e in diagram.external_edges:
        p = TheoryRegistry.get_particle(diagram.theory, e.particle)
        if p.particle_type in {ParticleType.FERMION, ParticleType.ANTIFERMION}:
            ext_fermions.append(e)
        elif p.particle_type == ParticleType.BOSON:
            ext_bosons.append(e)

    if len(ext_fermions) != 2 or len(ext_bosons) != 2:
        return None

    info = _ComptonInfo()
    info.topology = diagram.topology
    info.symmetry_factor = diagram.symmetry_factor or 1.0

    # Count incoming bosons for spin averaging.
    incoming_bosons = [e for e in ext_bosons if e.start_vertex < 0]
    info.n_incoming_bosons = len(incoming_bosons)

    # Fermion mass from the internal propagator (same species as external fermions).
    ferm_mass = _particle_mass_label(diagram.theory, internal.particle)
    info.fermion_mass_label = ferm_mass
    info.in_mass_label = ferm_mass
    info.out_mass_label = ferm_mass
    info.prop_mass_label = ferm_mass

    # Determine the propagator's Mandelstam variable from topology.
    info.prop_mandelstam = {
        "s-channel": "s", "t-channel": "t", "u-channel": "u",
    }.get(diagram.topology, "t")

    # Find the coupling at each vertex (fermion-boson vertex).
    boson_name = ext_bosons[0].particle  # both bosons are same type for QED
    fermion_name = ext_fermions[0].particle
    info.coupling_in = _coupling_for_vertex(diagram.theory, boson_name, fermion_name)
    info.coupling_out = info.coupling_in  # same vertex type

    # Build the fermion line trace structure.
    # We need to traverse the single fermion line through the two vertices.
    info.trace_elements = _build_compton_trace_elements(diagram, internal, ext_fermions, ext_bosons)

    return info if info.trace_elements else None


def _build_compton_trace_elements(
    diagram: Diagram, internal: Edge,
    ext_fermions: list[Edge], ext_bosons: list[Edge],
) -> Optional[list[tuple[str, str]]]:
    """Build the trace element sequence for a single-fermion-line diagram.

    Returns a list of (type, momentum_or_index) tuples representing the
    fermion-line trace in order.  Types:
        "ferm"  — external fermion completeness: (/p + m) or (/p - m)
        "gamma" — vertex gamma matrix: γ^μ
        "prop"  — internal fermion propagator: (/P_int + m)

    The internal propagator momentum is computed from conservation at one vertex.
    """
    v_start = internal.start_vertex
    v_end = internal.end_vertex

    # Find which external edges connect to each vertex.
    def _ext_at_vertex(vid):
        return [e for e in diagram.external_edges
                if e.end_vertex == vid or e.start_vertex == vid]

    ext_start = _ext_at_vertex(v_start)
    ext_end = _ext_at_vertex(v_end)

    if len(ext_start) != 2 or len(ext_end) != 2:
        return None

    # At each vertex: one fermion/antifermion + one boson.
    def _classify(edges):
        ferm = boson = None
        for e in edges:
            p = TheoryRegistry.get_particle(diagram.theory, e.particle)
            if p.particle_type in {ParticleType.FERMION, ParticleType.ANTIFERMION}:
                ferm = e
            elif p.particle_type == ParticleType.BOSON:
                boson = e
        return ferm, boson

    ferm_start, boson_start = _classify(ext_start)
    ferm_end, boson_end = _classify(ext_end)

    if any(x is None for x in (ferm_start, boson_start, ferm_end, boson_end)):
        return None

    # Determine the internal propagator momentum from conservation.
    # At vertex v_start: momentum in = momentum out.
    # Internal fermion goes from v_start to v_end (or reversed by arrow).
    # Use QGRAF convention: internal edge goes start→end.

    # Figure out if each external edge is incoming or outgoing at this vertex.
    def _mom_sign_at_vertex(edge, vid):
        """Return +1 if momentum flows INTO the vertex, -1 if out."""
        if edge.start_vertex < 0 and edge.end_vertex == vid:
            return +1  # incoming external → flows into vertex
        if edge.start_vertex == vid and edge.end_vertex < 0:
            return -1  # outgoing external → flows out of vertex
        return 0

    # Compute internal propagator momentum from conservation at v_start.
    # The internal edge goes start→end (1→0 typically), but the physical
    # fermion flows end→start (0→1).  The propagator numerator (/P + m) uses
    # the momentum along the fermion arrow, which is the NEGATIVE of the
    # edge-direction momentum.
    #
    # Edge-direction momentum at v_start: P_edge = Σ(in) - Σ(out) at v_start.
    # Fermion-flow momentum: P_ferm = -P_edge.
    int_mom_parts = []
    for e in ext_start:
        sign = _mom_sign_at_vertex(e, v_start)
        # Negate: fermion flow opposes edge direction.
        if sign > 0:
            int_mom_parts.append(f"-{e.momentum}")
        elif sign < 0:
            int_mom_parts.append(f"+{e.momentum}")

    if not int_mom_parts:
        return None

    int_mom_expr = "".join(int_mom_parts).lstrip("+")

    # Determine trace ordering.
    # The trace visits: ext_ferm_1, γ^μ_1, propagator, γ^μ_2, ext_ferm_2
    # For spin sum, the ordering depends on whether the external fermion
    # is particle (+m) or antiparticle (-m).
    ferm_mass_label = _particle_mass_label(diagram.theory, ferm_start.particle)

    # Classify each external fermion as fermion or antifermion.
    p_start = TheoryRegistry.get_particle(diagram.theory, ferm_start.particle)
    p_end = TheoryRegistry.get_particle(diagram.theory, ferm_end.particle)
    is_start_antifermion = p_start.particle_type == ParticleType.ANTIFERMION
    is_end_antifermion = p_end.particle_type == ParticleType.ANTIFERMION

    # Build trace: start from one external fermion, go through both vertices.
    # Trace = Tr[(/p_start ± m) γ^μ_start (/P_int ± m) γ^μ_end (/p_end ± m) γ^ν_end (/P_int ± m) γ^ν_start]
    #
    # But for the SQUARED amplitude, we need M × M*, which for a single
    # fermion line gives:
    # Tr[(/p_ext1 ± m) Γ_vertex1 (/P_int + m) Γ_vertex2 (/p_ext2 ± m) Γ†_vertex2 (/P_int + m) Γ†_vertex1]
    #
    # For QED vertices, Γ = γ^μ which is self-adjoint in the trace, so Γ† = γ^μ.
    # The polarization sum for external photons contracts the paired μ indices → automatic.

    elements = [
        ("ferm", ferm_start.momentum, -1 if is_start_antifermion else +1, ferm_mass_label),
        ("gamma", "mu_v1_a"),
        ("prop", int_mom_expr, +1, ferm_mass_label),
        ("gamma", "mu_v2_a"),
        ("ferm", ferm_end.momentum, -1 if is_end_antifermion else +1, ferm_mass_label),
        ("gamma", "mu_v2_b"),
        ("prop", int_mom_expr, +1, ferm_mass_label),
        ("gamma", "mu_v1_b"),
    ]

    return elements


def _split_fermion_pair(theory: str, edges: list[Edge]) -> tuple[Optional[Edge], Optional[Edge]]:
    """Split a pair of external edges into (fermion, antifermion)."""
    ferm = anti = None
    for e in edges:
        p = TheoryRegistry.get_particle(theory, e.particle)
        if p.particle_type == ParticleType.FERMION:
            ferm = e
        elif p.particle_type == ParticleType.ANTIFERMION:
            anti = e
    return ferm, anti


def _particle_mass_label(theory: str, particle_name: str) -> Optional[str]:
    p = TheoryRegistry.get_particle(theory, particle_name)
    if p.mass and p.mass != "0":
        return p.mass
    return None


def _coupling_for_vertex(theory: str, mediator_name: str, fermion_name: str):
    """Return the symbolic coupling factor for a vertex.

    For γ-fermion vertices the fermion's electric charge magnitude |Q_f| (in
    units of |e|) is included so that pure-quark γ amplitudes carry the
    correct Q_f^n weight.  Without this, all quark flavours would yield the
    same partonic σ̂(qq̄→γγ) and a hadronic convolution over flavours would
    badly overestimate the cross-section.
    """
    if mediator_name == "gamma":
        from feynman_engine.amplitudes.symbolic import _fermion_charge_magnitude
        q = _fermion_charge_magnitude(fermion_name)
        if q == 1:
            return Symbol("e")
        return Rational(q.numerator, q.denominator) * Symbol("e")
    if mediator_name == "g":
        return Symbol("g_s")
    if mediator_name == "Z":
        species = fermion_name.rstrip("+-").replace("~", "")
        return Symbol(f"g_Z_{species}")
    if mediator_name in {"W+", "W-"}:
        return Symbol(f"g_W")
    if mediator_name == "Zp":
        species = fermion_name.rstrip("+-").replace("~", "")
        return Symbol(f"g_Zp_{species}")
    return Symbol(f"g_{mediator_name}")


# ── FORM program generation ──────────────────────────────────────────────────

def _build_form_program(
    analyzed: list[tuple[Diagram, _DiagramInfo]],
    theory: str,
) -> str:
    """Generate a complete FORM program for all diagram self- and cross-traces."""
    # Collect all mass symbols needed.
    mass_labels: set[str] = set()
    for _, info in analyzed:
        if info.in_mass_label:
            mass_labels.add(info.in_mass_label)
        if info.out_mass_label:
            mass_labels.add(info.out_mass_label)
        if info.mediator_mass_label:
            mass_labels.add(info.mediator_mass_label)

    # FORM doesn't allow underscores in names — sanitize.
    form_mass_syms = ",".join(sorted(_to_form_name(m) for m in mass_labels)) if mass_labels else ""
    extra_syms = f",{form_mass_syms}" if form_mass_syms else ""

    lines = [
        "#-",
        "Off Statistics;",
        f"Symbols s,t,u{extra_syms};",
        "Vectors p1,p2,q1,q2;",
    ]

    # Determine how many Lorentz indices we need.
    n_pairs = len(analyzed) * (len(analyzed) + 1) // 2
    idx_names = [f"mu{i}" for i in range(n_pairs * 2)]
    lines.append(f"Indices {','.join(idx_names)};")
    lines.append("")

    # Generate Local expressions for each diagram pair (i, j) with i <= j.
    pair_idx = 0
    pair_names: list[tuple[int, int, str]] = []
    for i, (d_i, info_i) in enumerate(analyzed):
        for j in range(i, len(analyzed)):
            d_j, info_j = analyzed[j]
            mu = idx_names[pair_idx * 2]
            nu = idx_names[pair_idx * 2 + 1]
            name = f"T{i}x{j}"
            pair_names.append((i, j, name))

            if info_i.topology == "s-channel" and info_j.topology == "s-channel":
                # Factorizable: Tr[line1] × Tr[line2]
                expr = _form_s_channel_pair(info_i, info_j, mu, nu)
            elif info_i.topology in {"t-channel", "u-channel"} and info_j.topology == info_i.topology:
                # Same channel, factorizable
                expr = _form_tu_diagonal(info_i, info_j, mu, nu)
            else:
                # Cross-interference: single combined trace (8-gamma)
                expr = _form_cross_trace(info_i, info_j, mu, nu)

            lines.append(f"Local {name} = {expr};")
            lines.append("")
            pair_idx += 1

    # Add trace and contraction instructions.
    # Count distinct spinor lines used.
    max_line = pair_idx * 2  # conservative upper bound
    for line_id in range(1, max_line + 1):
        lines.append(f"trace4,{line_id};")
    lines.append("contract;")
    lines.append("")

    # Kinematics substitutions.
    lines.extend(_kinematics_substitutions(analyzed))
    lines.append("")

    # Print results.
    for _, _, name in pair_names:
        lines.append(f"print +s {name};")
    lines.append(".end")

    return "\n".join(lines)


def _form_s_channel_pair(info_i: _DiagramInfo, info_j: _DiagramInfo, mu: str, nu: str) -> str:
    """Factorizable s-channel × s-channel: Tr[line1] × Tr[line2].

    Line 1 (incoming): Tr[(/p_anti ∓ m) γ^μ (/p_ferm ± m) γ^ν]
    Line 2 (outgoing): Tr[(/p_ferm ± m) γ_μ (/p_anti ∓ m) γ_ν]

    For spin sum: antifermion → (/p - m), fermion → (/p + m).
    """
    line1 = _spinor_line_form(
        1, info_i.in_antifermion_mom, info_i.in_fermion_mom,
        info_i.in_mass_label, mu, nu, antifermion_first=True,
    )
    line2 = _spinor_line_form(
        2, info_j.out_fermion_mom, info_j.out_antifermion_mom,
        info_j.out_mass_label, mu, nu, antifermion_first=False,
    )
    return f"({line1})*({line2})"


def _form_tu_diagonal(info_i: _DiagramInfo, info_j: _DiagramInfo, mu: str, nu: str) -> str:
    """t/u-channel diagonal: Tr[line_A] × Tr[line_B].

    Each line: Tr[(/p_in ± m) γ^μ (/p_out ± m) γ^ν] at one vertex.
    """
    line1 = _spinor_line_form(
        1, info_i.in_fermion_mom, info_i.in_antifermion_mom,
        info_i.in_mass_label, mu, nu, antifermion_first=False,
    )
    line2 = _spinor_line_form(
        2, info_j.out_fermion_mom, info_j.out_antifermion_mom,
        info_j.out_mass_label, mu, nu, antifermion_first=False,
    )
    return f"({line1})*({line2})"


def _form_cross_trace(info_i: _DiagramInfo, info_j: _DiagramInfo, mu: str, nu: str) -> str:
    """Cross-interference: single 8-gamma trace from Fierz rearrangement.

    When two diagrams have different fermion-flow topologies, the spin-summed
    interference M_i × M_j* becomes a single trace over 8 gamma matrices
    (instead of a product of two 4-gamma traces).

    The momentum ordering in the trace follows the Fierz loop — connecting
    the fermion lines of one amplitude to the conjugate of the other.

    For s×t (Bhabha):
        Tr[(/p1 - m)γ^μ(/p2 + m)γ^ν(/q2 + m)γ_μ(/q1 - m)γ_ν]
        Order: s_in_anti, s_in_ferm, s_out_ferm, s_out_anti

    For t×u (Møller):
        Tr[(/q1 + m)γ^μ(/p1 + m)γ^ν(/q2 + m)γ_μ(/p2 + m)γ_ν]
        Order: t_in_anti, t_in_ferm, t_out_anti, t_out_ferm
        (All identical fermions so all mass signs are +m.)
    """
    topos = {info_i.topology, info_j.topology}

    if topos == {"s-channel", "t-channel"} or topos == {"s-channel", "u-channel"}:
        # s×t or s×u: Bhabha-type.
        # The Fierz loop follows the s-channel's two fermion lines
        # connected into one trace by the t/u-channel's conjugate flow.
        s_info = info_i if info_i.topology == "s-channel" else info_j

        parts = [
            _slash_or_mass(1, s_info.in_antifermion_mom, s_info.in_mass_label, sign=-1),
            f"g_(1,{mu})",
            _slash_or_mass(1, s_info.in_fermion_mom, s_info.in_mass_label, sign=+1),
            f"g_(1,{nu})",
            _slash_or_mass(1, s_info.out_fermion_mom, s_info.out_mass_label, sign=+1),
            f"g_(1,{mu})",
            _slash_or_mass(1, s_info.out_antifermion_mom, s_info.out_mass_label, sign=-1),
            f"g_(1,{nu})",
        ]
        return "*".join(parts)

    elif topos == {"t-channel", "u-channel"}:
        # t×u: Møller-type (identical-particle scattering).
        # The Fierz loop order: t.in_anti, t.in_ferm, t.out_anti, t.out_ferm.
        # All legs are the same particle type, so all mass signs are +m.
        t_info = info_i if info_i.topology == "t-channel" else info_j
        mass = t_info.in_mass_label

        parts = [
            _slash_or_mass(1, t_info.in_antifermion_mom, mass, sign=+1),
            f"g_(1,{mu})",
            _slash_or_mass(1, t_info.in_fermion_mom, mass, sign=+1),
            f"g_(1,{nu})",
            _slash_or_mass(1, t_info.out_antifermion_mom, t_info.out_mass_label, sign=+1),
            f"g_(1,{mu})",
            _slash_or_mass(1, t_info.out_fermion_mom, t_info.out_mass_label, sign=+1),
            f"g_(1,{nu})",
        ]
        return "*".join(parts)

    return "0"


def _spinor_line_form(
    line_id: int,
    mom_a: str, mom_b: str,
    mass_label: Optional[str],
    mu: str, nu: str,
    antifermion_first: bool,
) -> str:
    """Build a single FORM spinor-line trace: Tr[(/a ± m) γ^μ (/b ± m) γ^ν].

    antifermion_first=True:  Tr[(/a - m) γ^μ (/b + m) γ^ν]  (incoming s-channel)
    antifermion_first=False: Tr[(/a + m) γ^μ (/b - m) γ^ν]  (outgoing s-channel, or t/u vertex)
    """
    if antifermion_first:
        sign_a, sign_b = -1, +1
    else:
        sign_a, sign_b = +1, -1

    parts = [
        _slash_or_mass(line_id, mom_a, mass_label, sign_a),
        f"g_({line_id},{mu})",
        _slash_or_mass(line_id, mom_b, mass_label, sign_b),
        f"g_({line_id},{nu})",
    ]
    return "*".join(parts)


def _slash_or_mass(line_id: int, momentum: str, mass_label: Optional[str], sign: int) -> str:
    """Build FORM expression for (/p ± m) on a spinor line.

    /p + m → (g_(line, p) + m*gi_(line))
    /p - m → (g_(line, p) - m*gi_(line))
    """
    slash = f"g_({line_id},{momentum})"
    if not mass_label:
        return slash
    fm = _to_form_name(mass_label)
    if sign >= 0:
        return f"({slash}+{fm}*gi_({line_id}))"
    else:
        return f"({slash}-{fm}*gi_({line_id}))"


def _kinematics_substitutions(analyzed: list[tuple[Diagram, _DiagramInfo]]) -> list[str]:
    """Generate FORM kinematics substitution rules.

    Builds the full set of dot products for 2→2 scattering with
    possibly massive external particles.
    """
    # Collect all mass labels for external particles.
    mass_labels: dict[str, Optional[str]] = {}  # momentum → mass_label
    for _, info in analyzed:
        mass_labels[info.in_fermion_mom] = info.in_mass_label
        mass_labels[info.in_antifermion_mom] = info.in_mass_label
        mass_labels[info.out_fermion_mom] = info.out_mass_label
        mass_labels[info.out_antifermion_mom] = info.out_mass_label

    # Standard 2→2 momentum labeling: p1, p2 incoming; q1, q2 outgoing.
    # On-shell: p_i.p_i = m_i^2
    lines: list[str] = []

    for mom, mlabel in mass_labels.items():
        if mlabel:
            lines.append(f"id {mom}.{mom} = {_to_form_name(mlabel)}^2;")
        else:
            lines.append(f"id {mom}.{mom} = 0;")

    # Cross-products from Mandelstam variables.
    # s = (p1+p2)^2 = m1^2 + m2^2 + 2*p1.p2
    # t = (p1-q1)^2 = m1^2 + m3^2 - 2*p1.q1
    # u = (p1-q2)^2 = m1^2 + m4^2 - 2*p1.q2
    # Also: p2.q1 = (m2^2 + m3^2 - u)/2, p2.q2 = (m2^2 + m4^2 - t)/2
    # q1.q2 = (s - m3^2 - m4^2)/2

    def msq(mom: str) -> str:
        ml = mass_labels.get(mom)
        return f"{_to_form_name(ml)}^2" if ml else "0"

    # Map momentum names to roles. The first incoming is p1, second is p2, etc.
    # But QGRAF may give various momentum names. We need to figure out which is which.
    # Use the actual momentum labels from the first analyzed diagram.
    _, info0 = analyzed[0]

    # For s-channel: in_antifermion=p1/p2, in_fermion=p1/p2 depending on convention
    # Let's identify p1, p2, q1, q2 from the external edges of the first diagram.
    d0 = analyzed[0][0]
    incoming_edges = sorted(
        [e for e in d0.external_edges if e.start_vertex < 0],
        key=lambda e: e.id,
    )
    outgoing_edges = sorted(
        [e for e in d0.external_edges if e.end_vertex < 0],
        key=lambda e: e.id,
    )

    if len(incoming_edges) < 2 or len(outgoing_edges) < 2:
        return lines

    p1 = incoming_edges[0].momentum
    p2 = incoming_edges[1].momentum
    q1 = outgoing_edges[0].momentum
    q2 = outgoing_edges[1].momentum

    m1sq = msq(p1)
    m2sq = msq(p2)
    m3sq = msq(q1)
    m4sq = msq(q2)

    lines.append(f"id {p1}.{p2} = (s - {m1sq} - {m2sq})/2;")
    lines.append(f"id {p1}.{q1} = ({m1sq} + {m3sq} - t)/2;")
    lines.append(f"id {p1}.{q2} = ({m1sq} + {m4sq} - u)/2;")
    lines.append(f"id {p2}.{q1} = ({m2sq} + {m3sq} - u)/2;")
    lines.append(f"id {p2}.{q2} = ({m2sq} + {m4sq} - t)/2;")
    lines.append(f"id {q1}.{q2} = (s - {m3sq} - {m4sq})/2;")

    return lines


# ── FORM program generation (Compton-type) ──────────────────────────────────

def _build_compton_form_program(
    analyzed: list[tuple[Diagram, _ComptonInfo]],
    theory: str,
) -> str:
    """Generate a FORM program for Compton-type (single-fermion-line) diagrams."""
    mass_labels: set[str] = set()
    for _, info in analyzed:
        if info.fermion_mass_label:
            mass_labels.add(info.fermion_mass_label)

    form_mass_syms = ",".join(sorted(_to_form_name(m) for m in mass_labels)) if mass_labels else ""
    extra_syms = f",{form_mass_syms}" if form_mass_syms else ""

    lines = [
        "#-",
        "Off Statistics;",
        f"Symbols s,t,u{extra_syms};",
        "Vectors p1,p2,q1,q2;",
    ]

    # We need indices for each diagram pair. Each diagram's trace uses 2 indices
    # (one for each vertex, paired between amplitude and conjugate).
    n_pairs = len(analyzed) * (len(analyzed) + 1) // 2
    idx_names = [f"mu{i}" for i in range(n_pairs * 4)]
    lines.append(f"Indices {','.join(idx_names)};")
    lines.append("")

    pair_idx = 0
    pair_names: list[tuple[int, int, str]] = []

    for i, (d_i, info_i) in enumerate(analyzed):
        for j in range(i, len(analyzed)):
            d_j, info_j = analyzed[j]
            name = f"T{i}x{j}"
            pair_names.append((i, j, name))

            expr = _build_compton_trace_expr(
                info_i, info_j, i, j, idx_names, pair_idx,
            )
            lines.append(f"Local {name} = {expr};")
            lines.append("")
            pair_idx += 1

    # Trace and contraction.
    max_line = pair_idx * 2
    for line_id in range(1, max_line + 1):
        lines.append(f"trace4,{line_id};")
    lines.append("contract;")
    lines.append("")

    # Kinematics.
    lines.extend(_compton_kinematics(analyzed))
    lines.append("")

    for _, _, name in pair_names:
        lines.append(f"print +s {name};")
    lines.append(".end")

    return "\n".join(lines)


def _build_compton_trace_expr(
    info_i: _ComptonInfo, info_j: _ComptonInfo,
    i: int, j: int,
    idx_names: list[str], pair_idx: int,
) -> str:
    """Build FORM trace expression for Compton-type diagram pair.

    For diagonal (i==j): single trace on one FORM spinor line.
    For cross (i!=j): single trace combining both diagram's vertex structures.

    The trace for |M_i|² is:
        Tr[(/p_ext1 ± m) γ^μ (/P_int + m) γ^ν (/p_ext2 ± m) γ^ν (/P_int + m) γ^μ]

    where μ,ν are the photon Lorentz indices (contracted by photon polarization sum).
    """
    line_id = 1  # Single spinor line

    if i == j:
        # Diagonal: use trace elements from info_i.
        return _compton_single_trace(info_i, line_id, idx_names, pair_idx * 4)
    else:
        # Cross-interference: M_i × M_j*.
        # This creates a trace that combines vertex i's amplitude structure
        # with vertex j's conjugate structure.
        return _compton_cross_trace(info_i, info_j, line_id, idx_names, pair_idx * 4)


def _compton_single_trace(
    info: _ComptonInfo, line_id: int,
    idx_names: list[str], idx_offset: int,
) -> str:
    """Build FORM trace for a single Compton-type diagram squared.

    Trace = Tr[(/p1 ± m) γ^μ (/P + m) γ^ν (/p2 ± m) γ^ν (/P + m) γ^μ]
    """
    elements = info.trace_elements
    if not elements:
        return "0"

    # Use paired indices: mu_a and mu_a for same vertex (contraction = pol sum).
    mu1 = idx_names[idx_offset]      # vertex 1 index
    mu2 = idx_names[idx_offset + 1]  # vertex 2 index

    parts = []
    for elem in elements:
        etype = elem[0]
        if etype == "ferm":
            _, mom, sign, mass = elem
            parts.append(_slash_or_mass(line_id, mom, mass, sign))
        elif etype == "gamma":
            # Map generic index names to actual FORM indices.
            idx_name = elem[1]
            if "v1" in idx_name:
                parts.append(f"g_({line_id},{mu1})")
            else:
                parts.append(f"g_({line_id},{mu2})")
        elif etype == "prop":
            _, mom_expr, sign, mass = elem
            parts.append(_slash_composite(line_id, mom_expr, mass, sign))

    return "*".join(parts)


def _compton_cross_trace(
    info_i: _ComptonInfo, info_j: _ComptonInfo,
    line_id: int, idx_names: list[str], idx_offset: int,
) -> str:
    """Build FORM trace for Compton cross-interference.

    For two diagrams with different topologies (e.g., s×u Compton), the photon
    vertex assignments are SWAPPED: the incoming photon is at vertex 1 in one
    diagram and vertex 2 in the other.  The polarization sum contracts indices
    for the SAME physical photon, so the conjugate half must swap mu1↔mu2
    relative to the amplitude half.

    Correct cross trace:
        Tr[ext1 γ^ν P_s γ^μ ext2 γ_ν P_u γ_μ]
    where ν = outgoing photon index (v1 in s-channel, v2 in u-channel)
          μ = incoming photon index (v2 in s-channel, v1 in u-channel)
    """
    if not info_i.trace_elements or not info_j.trace_elements:
        return "0"

    mu1 = idx_names[idx_offset]      # vertex 1 index for diagram i
    mu2 = idx_names[idx_offset + 1]  # vertex 2 index for diagram i

    elems_i = info_i.trace_elements
    elems_j = info_j.trace_elements

    parts = []
    # First half: diagram i's amplitude (elements 0-4: ext1, gamma, prop, gamma, ext2)
    for elem in elems_i[:5]:
        etype = elem[0]
        if etype == "ferm":
            _, mom, sign, mass = elem
            parts.append(_slash_or_mass(line_id, mom, mass, sign))
        elif etype == "gamma":
            idx_name = elem[1]
            if "v1" in idx_name:
                parts.append(f"g_({line_id},{mu1})")
            else:
                parts.append(f"g_({line_id},{mu2})")
        elif etype == "prop":
            _, mom_expr, sign, mass = elem
            parts.append(_slash_composite(line_id, mom_expr, mass, sign))

    # Second half: diagram j's conjugate (elements 5-7: gamma, prop, gamma).
    # SWAP mu1↔mu2 to match the physical photon pairing:
    # diagram j has the opposite photon-vertex assignment.
    for elem in elems_j[5:]:
        etype = elem[0]
        if etype == "gamma":
            idx_name = elem[1]
            # SWAPPED: v1→mu2, v2→mu1 (opposite to amplitude half)
            if "v1" in idx_name:
                parts.append(f"g_({line_id},{mu2})")
            else:
                parts.append(f"g_({line_id},{mu1})")
        elif etype == "prop":
            _, mom_expr, sign, mass = elem
            parts.append(_slash_composite(line_id, mom_expr, mass, sign))

    return "*".join(parts)


def _slash_composite(line_id: int, mom_expr: str, mass_label: Optional[str], sign: int) -> str:
    """Build FORM expression for an internal propagator (/p1+/p2 + m).

    Handles composite momenta like "p1+p2" by expanding to g_(line,p1)+g_(line,p2).
    """
    # Parse the momentum expression: "p1+p2", "p1-q2", etc.
    import re as _re
    parts = _re.findall(r'[+-]?[a-zA-Z]\w*', mom_expr)

    slash_terms = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part.startswith("-"):
            mom = part[1:]
            slash_terms.append(f"-g_({line_id},{mom})")
        elif part.startswith("+"):
            mom = part[1:]
            slash_terms.append(f"+g_({line_id},{mom})")
        else:
            # First term without sign: no prefix. Later terms: add +.
            if i > 0:
                slash_terms.append(f"+g_({line_id},{part})")
            else:
                slash_terms.append(f"g_({line_id},{part})")

    slash = "".join(slash_terms)

    if mass_label:
        fm = _to_form_name(mass_label)
        if sign >= 0:
            return f"({slash}+{fm}*gi_({line_id}))"
        else:
            return f"({slash}-{fm}*gi_({line_id}))"
    return f"({slash})"


def _compton_kinematics(analyzed: list[tuple[Diagram, _ComptonInfo]]) -> list[str]:
    """Generate kinematics substitutions for Compton-type processes."""
    # Get the first diagram's external edges for standard momentum labeling.
    d0 = analyzed[0][0]
    incoming = sorted(
        [e for e in d0.external_edges if e.start_vertex < 0],
        key=lambda e: e.id,
    )
    outgoing = sorted(
        [e for e in d0.external_edges if e.end_vertex < 0],
        key=lambda e: e.id,
    )

    if len(incoming) < 2 or len(outgoing) < 2:
        return []

    p1, p2 = incoming[0].momentum, incoming[1].momentum
    q1, q2 = outgoing[0].momentum, outgoing[1].momentum

    # Determine masses from particle types.
    mass_labels: dict[str, Optional[str]] = {}
    for e in incoming + outgoing:
        mass_labels[e.momentum] = _particle_mass_label(d0.theory, e.particle)

    lines: list[str] = []

    def msq(mom):
        ml = mass_labels.get(mom)
        return f"{_to_form_name(ml)}^2" if ml else "0"

    # On-shell conditions.
    for mom, ml in mass_labels.items():
        if ml:
            lines.append(f"id {mom}.{mom} = {_to_form_name(ml)}^2;")
        else:
            lines.append(f"id {mom}.{mom} = 0;")

    # Mandelstam cross-products.
    m1sq, m2sq = msq(p1), msq(p2)
    m3sq, m4sq = msq(q1), msq(q2)

    lines.append(f"id {p1}.{p2} = (s - {m1sq} - {m2sq})/2;")
    lines.append(f"id {p1}.{q1} = ({m1sq} + {m3sq} - t)/2;")
    lines.append(f"id {p1}.{q2} = ({m1sq} + {m4sq} - u)/2;")
    lines.append(f"id {p2}.{q1} = ({m2sq} + {m3sq} - u)/2;")
    lines.append(f"id {p2}.{q2} = ({m2sq} + {m4sq} - t)/2;")
    lines.append(f"id {q1}.{q2} = (s - {m3sq} - {m4sq})/2;")

    return lines


def _assemble_compton_msq(
    analyzed: list[tuple[Diagram, _ComptonInfo]],
    parsed: dict[str, object],
    theory: str,
) -> object:
    """Assemble |M̄|² for Compton-type processes."""
    from feynman_engine.amplitudes.color import color_factor

    msq = Integer(0)

    for i, (d_i, info_i) in enumerate(analyzed):
        for j in range(i, len(analyzed)):
            d_j, info_j = analyzed[j]

            name = f"T{i}x{j}"
            trace = parsed.get(name)
            if trace is None or trace == 0:
                continue

            c_i = info_i.coupling_in * info_i.coupling_out
            c_j = info_j.coupling_in * info_j.coupling_out
            coupling = c_i * c_j

            # Propagator denominators: for Compton, it's the internal fermion
            # propagator: (P² - m²).
            denom_i = _compton_prop_denom(info_i)
            denom_j = _compton_prop_denom(info_j)
            denom = denom_i * denom_j

            c_color = color_factor(d_i, d_j, theory)
            multiplicity = Integer(1) if i == j else Integer(2)

            # No relative Fermi sign for Compton-type: both diagrams share the
            # same fermion line, so there is no fermion-leg exchange.
            term = multiplicity * c_color * coupling * trace / denom
            msq += term

    # Spin average.
    # For each incoming particle: fermion gives factor 2, boson (photon/gluon) gives factor 2.
    # So spin average = 1/(2 × 2) = 1/4 always for 2→2.
    msq = Rational(1, 4) * msq

    # Color average for QCD.
    if theory == "QCD":
        # Depends on incoming particles. For qg→qg: 1/(3×8) = 1/24.
        # For qq̄→gg: 1/(3×3) = 1/9.
        info0 = analyzed[0][1]
        if info0.n_incoming_bosons == 1:
            msq = Rational(1, 24) * msq  # qg initial state
        elif info0.n_incoming_bosons == 2:
            msq = Rational(1, 64) * msq  # gg initial state
        else:
            msq = Rational(1, 9) * msq  # qq initial state

    return msq


def _compton_prop_denom(info: _ComptonInfo) -> object:
    """Return the propagator denominator for a Compton-type diagram.

    The internal propagator is a fermion with denominator (P² - m²).
    P² equals the Mandelstam variable for the relevant channel.
    """
    inv = {"s": s_sym, "t": t_sym, "u": u_sym}[info.prop_mandelstam]
    if info.prop_mass_label:
        m_sq = Symbol(info.prop_mass_label) ** 2
        return inv - m_sq
    return inv


# ── FORM execution ────────────────────────────────────────────────────────────

def _run_form(program: str) -> Optional[str]:
    """Execute a FORM program and return stdout, or None on failure."""
    form_bin = find_form_binary()
    if form_bin is None:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".frm", delete=False, prefix="feynman_form_",
    ) as f:
        f.write(program)
        f.flush()
        frm_path = Path(f.name)

    try:
        result = subprocess.run(
            [str(form_bin), str(frm_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        frm_path.unlink(missing_ok=True)


# ── Output parsing ────────────────────────────────────────────────────────────

def _parse_form_output(stdout: str, mass_labels: set[str] | None = None) -> dict[str, object]:
    """Parse FORM stdout into {variable_name: sympy_expr} dict."""
    results: dict[str, object] = {}

    # FORM output format:
    #    T0x0 =
    #       8*u^2 + 8*t^2;
    # Or multi-line:
    #    T0x1 =
    #       - 8*u^2;

    # Split into blocks by finding variable assignments.
    pattern = re.compile(r"^\s+(\w+)\s*=$", re.MULTILINE)
    matches = list(pattern.finditer(stdout))

    for k, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[k + 1].start() if k + 1 < len(matches) else len(stdout)

        # Extract the expression text (everything between = and ;).
        expr_text = stdout[start:end]

        # FORM terminates expressions with ';' but the raw text may also
        # contain footer lines (timing info, etc.) after the semicolon.
        # Find the *last* semicolon and truncate there.
        semi_pos = expr_text.rfind(";")
        if semi_pos >= 0:
            expr_text = expr_text[:semi_pos].strip()
        else:
            expr_text = expr_text.strip()

        # Clean up FORM formatting: join continuation lines.
        expr_text = " ".join(expr_text.split())

        # Convert to SymPy.
        try:
            expr = _form_expr_to_sympy(expr_text, mass_labels)
            results[name] = expr
        except Exception:
            continue

    return results


def _form_expr_to_sympy(text: str, mass_labels: set[str] | None = None) -> object:
    """Convert a FORM polynomial expression to a SymPy expression.

    FORM uses ``^`` for powers and ``*`` for multiplication, which is
    compatible with SymPy's ``sympify`` after minor cleanup.
    """
    text = text.strip()
    if not text or text == "0":
        return Integer(0)

    # FORM uses d_(mu,nu) for metric tensor — should be contracted away.
    # If any remain, something went wrong.
    if "d_(" in text:
        return None

    # Replace FORM power notation with Python: ^ → **.
    text = text.replace("^", "**")

    # Build local dict with proper SymPy symbols.
    local_dict = {
        "s": Symbol("s", real=True),
        "t": Symbol("t", real=True),
        "u": Symbol("u", real=True),
        # 2→3 invariants
        "s12": Symbol("s12", real=True),
        "s13": Symbol("s13", real=True),
        "s23": Symbol("s23", real=True),
        "t1": Symbol("t1", real=True),
        "t2": Symbol("t2", real=True),
    }
    # Map FORM-sanitized mass names back to SymPy symbols with original names.
    if mass_labels:
        for orig in mass_labels:
            form_name = _to_form_name(orig)
            local_dict[form_name] = Symbol(orig)

    return sympify(text, locals=local_dict)


# ── |M|² assembly ────────────────────────────────────────────────────────────

def _assemble_msq(
    analyzed: list[tuple[Diagram, _DiagramInfo]],
    parsed: dict[str, object],
    theory: str,
) -> object:
    """Combine FORM trace results with couplings, denominators, and color factors.

    |M̄|² = (1/4) × Σ_{i,j} color_ij × coupling_i × coupling_j × trace_ij / (denom_i × denom_j)

    The factor of 1/4 is the spin average for two spin-1/2 incoming particles.
    The factor of 2 for off-diagonal terms (i≠j) is included.
    """
    from feynman_engine.amplitudes.color import color_factor

    msq = Integer(0)

    for i, (d_i, info_i) in enumerate(analyzed):
        for j in range(i, len(analyzed)):
            d_j, info_j = analyzed[j]

            name = f"T{i}x{j}"
            trace = parsed.get(name)
            if trace is None or trace == 0:
                continue

            # Coupling: product of couplings from both diagrams.
            c_i = info_i.coupling_in * info_i.coupling_out
            c_j = info_j.coupling_in * info_j.coupling_out
            coupling = c_i * c_j

            # Propagator denominators.
            denom_i = _propagator_denom(info_i)
            denom_j = _propagator_denom(info_j)
            denom = denom_i * denom_j

            # Color factor.
            c_color = color_factor(d_i, d_j, theory)

            # Symmetry factor for off-diagonal.
            multiplicity = Integer(1) if i == j else Integer(2)

            # Relative Fermi sign for cross-interference between different
            # topologies.  Going from one topology to another always involves
            # rearranging one fermion line → a factor of −1.
            fermi_sign = Integer(-1) if i != j else Integer(1)

            term = multiplicity * fermi_sign * c_color * coupling * trace / denom
            msq += term

    # Spin average: 1/4 for two spin-1/2 particles.
    msq = Rational(1, 4) * msq

    # Color average: 1/Nc^2 for QCD (both quarks), 1 for QED.
    if theory == "QCD":
        msq = Rational(1, 9) * msq  # 1/(3×3) for two triplet quarks

    return msq


def _propagator_denom(info: _DiagramInfo) -> object:
    """Return the propagator denominator for a single diagram (unsquared).

    For interference M_i × M_j*, the full denominator is denom_i × denom_j.
    This gives |prop|² for diagonal (i=j) and prop_i × prop_j for cross terms.
    """
    inv = {"s-channel": s_sym, "t-channel": t_sym, "u-channel": u_sym}[info.topology]
    if info.mediator_mass_label:
        m_sq = Symbol(info.mediator_mass_label) ** 2
        return inv - m_sq
    return inv


# ══════════════════════════════════════════════════════════════════════════════
# QCD processes with 3-gluon vertices (qq̄→gg, qg→qg)
# ══════════════════════════════════════════════════════════════════════════════
#
# These processes require:
#   1. Physical (axial gauge) polarization sums for external gluons to avoid
#      gauge artifacts from unphysical longitudinal modes.
#   2. 3-gluon vertex Feynman rules in the FORM program.
#   3. SU(3) color factor matrices from color.py.
#
# Gauge choice: Kleiss-Stirling — for each external gluon with momentum q_i,
# the reference vector n_i is the momentum of the OTHER gluon. This gives:
#   Π_{μμ'}(q1) = -g_{μμ'} + (q1_μ q2_{μ'} + q1_{μ'} q2_μ)/(q1·q2)
#   Π_{νν'}(q2) = -g_{νν'} + (q2_ν q1_{ν'} + q2_{ν'} q1_ν)/(q1·q2)
#
# Since q1·q2 = s/2 (massless), we multiply through by (s/2)^2 to clear
# denominators, and divide back in Python.
# ══════════════════════════════════════════════════════════════════════════════


def _try_qcd_gluon_vertex(spec, tree_diagrams, theory):
    """Handle QCD processes with 3-gluon vertices using physical pol sums."""
    from feynman_engine.amplitudes.color import (
        color_average,
        qqbar_to_gg_color,
        qg_to_qg_color,
    )

    # Classify incoming/outgoing particles from the spec (already parsed).
    quarks = {"u", "d", "s", "c", "b", "t", "u~", "d~", "s~", "c~", "b~", "t~"}
    in_quarks = sum(1 for p in spec.incoming if p in quarks)
    in_gluons = sum(1 for p in spec.incoming if p == "g")
    out_quarks = sum(1 for p in spec.outgoing if p in quarks)
    out_gluons = sum(1 for p in spec.outgoing if p == "g")

    if in_quarks == 2 and out_gluons == 2:
        return _qqbar_to_gg_form(spec, tree_diagrams, theory)
    if in_quarks == 1 and in_gluons == 1 and out_quarks == 1 and out_gluons == 1:
        return _qg_to_qg_form(spec, tree_diagrams, theory)
    if in_gluons == 2 and out_gluons == 2:
        return _gg_to_gg_form(spec, tree_diagrams, theory)
    return None


def _qqbar_to_gg_form(spec, tree_diagrams, theory):
    """Compute qq̄→gg using FORM with physical polarization sums.

    Uses the Kleiss-Stirling gauge (n1=q2, n2=q1) and computes all 6
    independent Lorentz traces (tt, uu, tu, ss, ts, us), then assembles
    with SU(3) color factors.
    """
    from feynman_engine.amplitudes.color import qqbar_to_gg_color

    program = _QQBAR_GG_PHYSPOL_PROGRAM
    raw_output = _run_form(program)
    if raw_output is None:
        return None

    parsed = _parse_form_output(raw_output)
    if not parsed:
        return None

    # The FORM traces are multiplied by (s/2)^2 = s^2/4.
    # Divide each by s^2/4.
    norm = s_sym**2 / 4

    trace_map = {}
    for key in ("Ltt", "Luu", "Ltu", "Lss", "Lts", "Lus"):
        val = parsed.get(key)
        if val is not None:
            trace_map[key] = val / norm
        else:
            trace_map[key] = Integer(0)

    # Color factors (color-summed, not averaged).
    C_tt = qqbar_to_gg_color("t", "t")  # 16/3
    C_uu = qqbar_to_gg_color("u", "u")  # 16/3
    C_tu = qqbar_to_gg_color("t", "u")  # -2/3
    C_ss = qqbar_to_gg_color("s", "s")  # 12
    C_ts = qqbar_to_gg_color("t", "s")  # -6
    C_us = qqbar_to_gg_color("u", "s")  # 6

    # Assemble: Σ C_ab × L_ab / (denom_a × denom_b)
    # t-channel: 1/t^2, u-channel: 1/u^2, s-channel: 1/s^2
    # Cross terms get factor of 2 for ab + ba.
    g_s = Symbol("g_s")
    msq = g_s**4 * (
        C_tt * trace_map["Ltt"] / t_sym**2
        + C_uu * trace_map["Luu"] / u_sym**2
        + 2 * C_tu * trace_map["Ltu"] / (t_sym * u_sym)
        + C_ss * trace_map["Lss"] / s_sym**2
        + 2 * C_ts * trace_map["Lts"] / (t_sym * s_sym)
        + 2 * C_us * trace_map["Lus"] / (u_sym * s_sym)
    )

    # Spin average: 1/4 (two spin-1/2 incoming).
    # Color average: 1/9 (Nc × Nc = 3 × 3).
    msq = Rational(1, 36) * msq
    msq = cancel(msq)

    return AmplitudeResult(
        process=spec.raw,
        theory=theory,
        msq=msq,
        msq_latex=latex(msq),
        integral_latex=None,
        description="Exact tree-level qq̄→gg |M̄|² via FORM with physical polarization sums and SU(3) color",
        notes="Kleiss-Stirling gauge (n1=q2, n2=q1). All 6 Lorentz traces computed.",
        backend="form-symbolic",
    )


def _qg_to_qg_form(spec, tree_diagrams, theory):
    """Compute qg→qg using FORM with physical polarization sums.

    Related to qq̄→gg by crossing: s↔t.
    Diagram topologies for qg→qg:
      s-channel fermion: color = T^b T^a
      u-channel fermion: color = T^a T^b
      t-channel 3-gluon: color = [T^a, T^b]
    """
    from feynman_engine.amplitudes.color import qg_to_qg_color

    program = _QG_QG_PHYSPOL_PROGRAM
    raw_output = _run_form(program)
    if raw_output is None:
        return None

    parsed = _parse_form_output(raw_output)
    if not parsed:
        return None

    # Physical pol sum normalization: multiply by (t/2)^2 = t^2/4 in FORM.
    norm = t_sym**2 / 4

    trace_map = {}
    for key in ("Lss", "Luu", "Lsu", "Ltt", "Lst", "Lut"):
        val = parsed.get(key)
        if val is not None:
            trace_map[key] = val / norm
        else:
            trace_map[key] = Integer(0)

    # Color factors for qg→qg.
    C_ss = qg_to_qg_color("s", "s")  # 16/3
    C_uu = qg_to_qg_color("u", "u")  # 16/3
    C_su = qg_to_qg_color("s", "u")  # -2/3
    C_tt = qg_to_qg_color("t", "t")  # 12
    C_st = qg_to_qg_color("s", "t")  # -6
    C_ut = qg_to_qg_color("u", "t")  # 6

    g_s = Symbol("g_s")
    msq = g_s**4 * (
        C_ss * trace_map["Lss"] / s_sym**2
        + C_uu * trace_map["Luu"] / u_sym**2
        + 2 * C_su * trace_map["Lsu"] / (s_sym * u_sym)
        + C_tt * trace_map["Ltt"] / t_sym**2
        + 2 * C_st * trace_map["Lst"] / (s_sym * t_sym)
        + 2 * C_ut * trace_map["Lut"] / (u_sym * t_sym)
    )

    # Spin average: 1/4. Color average: 1/(3×8) = 1/24.
    msq = Rational(1, 96) * msq
    msq = cancel(msq)

    return AmplitudeResult(
        process=spec.raw,
        theory=theory,
        msq=msq,
        msq_latex=latex(msq),
        integral_latex=None,
        description="Exact tree-level qg→qg |M̄|² via FORM with physical polarization sums and SU(3) color",
        notes="Kleiss-Stirling gauge. Crossed from qq̄→gg.",
        backend="form-symbolic",
    )


# ── Pre-built FORM programs for QCD gluon-vertex processes ───────────────────
#
# These are complete FORM programs verified against Combridge et al. (1977)
# and PYTHIA8. They use physical (axial gauge) polarization sums to avoid
# gauge artifacts that arise with the Feynman gauge (-g^{μν}) when
# 3-gluon vertices couple to external gluons.
#
# Convention: p1, p2 = incoming quark, antiquark; q1, q2 = outgoing gluons.
# Physical polarization sum with Kleiss-Stirling gauge (n1=q2, n2=q1):
#   P1_{mu,mup} = -(s/2)*d_(mu,mup) + q1(mu)*q2(mup) + q1(mup)*q2(mu)
#   P2_{nu,nup} = -(s/2)*d_(nu,nup) + q2(nu)*q1(nup) + q2(nup)*q1(nu)
# Results are multiplied by (s/2)^2 = s^2/4; divide in Python.

_QQBAR_GG_PHYSPOL_PROGRAM = r"""#-
Off Statistics;
Symbols s,t,u;
Vectors p1,p2,q1,q2;
Indices mu,nu,mup,nup,sigma,sigmap,alpha,beta,alp,bep;

* qq̄→gg with PHYSICAL polarization sums (Kleiss-Stirling gauge).
* Ref vectors: n1=q2 for gluon q1, n2=q1 for gluon q2.
* P1_{mu,mup} = -(s/2)*d_(mu,mup) + q1(mu)*q2(mup) + q1(mup)*q2(mu)
* P2_{nu,nup} = -(s/2)*d_(nu,nup) + q2(nu)*q1(nup) + q2(nup)*q1(nu)
* Results multiplied by (s/2)^2. Divide by s^2/4 in Python.

* ── Ltt: t-channel squared ──────────────────────────────
Local Ltt = g_(1,p1)*g_(1,mu)*(g_(1,p1)-g_(1,q1))*g_(1,nu)
           *g_(1,p2)*g_(1,nup)*(g_(1,p1)-g_(1,q1))*g_(1,mup)
           *(-(s/2)*d_(mu,mup) + q1(mu)*q2(mup) + q1(mup)*q2(mu))
           *(-(s/2)*d_(nu,nup) + q2(nu)*q1(nup) + q2(nup)*q1(nu));
trace4,1;
contract;
id p1.p1 = 0;
id p2.p2 = 0;
id q1.q1 = 0;
id q2.q2 = 0;
id p1.p2 = s/2;
id p1.q1 = -t/2;
id p1.q2 = -u/2;
id p2.q1 = -u/2;
id p2.q2 = -t/2;
id q1.q2 = s/2;
print +s Ltt;
.sort

* ── Luu: u-channel squared ──────────────────────────────
Local Luu = g_(1,p1)*g_(1,nu)*(g_(1,p1)-g_(1,q2))*g_(1,mu)
           *g_(1,p2)*g_(1,mup)*(g_(1,p1)-g_(1,q2))*g_(1,nup)
           *(-(s/2)*d_(mu,mup) + q1(mu)*q2(mup) + q1(mup)*q2(mu))
           *(-(s/2)*d_(nu,nup) + q2(nu)*q1(nup) + q2(nup)*q1(nu));
trace4,1;
contract;
id p1.p1 = 0;
id p2.p2 = 0;
id q1.q1 = 0;
id q2.q2 = 0;
id p1.p2 = s/2;
id p1.q1 = -t/2;
id p1.q2 = -u/2;
id p2.q1 = -u/2;
id p2.q2 = -t/2;
id q1.q2 = s/2;
print +s Luu;
.sort

* ── Ltu: t×u cross ──────────────────────────────────────
Local Ltu = g_(1,p1)*g_(1,mu)*(g_(1,p1)-g_(1,q1))*g_(1,nu)
           *g_(1,p2)*g_(1,alp)*(g_(1,p1)-g_(1,q2))*g_(1,bep)
           *(-(s/2)*d_(mu,alp) + q1(mu)*q2(alp) + q1(alp)*q2(mu))
           *(-(s/2)*d_(nu,bep) + q2(nu)*q1(bep) + q2(bep)*q1(nu));
trace4,1;
contract;
id p1.p1 = 0;
id p2.p2 = 0;
id q1.q1 = 0;
id q2.q2 = 0;
id p1.p2 = s/2;
id p1.q1 = -t/2;
id p1.q2 = -u/2;
id p2.q1 = -u/2;
id p2.q2 = -t/2;
id q1.q2 = s/2;
print +s Ltu;
.sort

* ── Lss: s-channel squared ──────────────────────────────
Local Lss = g_(1,p1)*g_(1,sigma)*g_(1,p2)*g_(1,sigmap)
  *(d_(sigma,mu)*(p1(nu)+p2(nu)+q1(nu)) + d_(mu,nu)*(q2(sigma)-q1(sigma)) + d_(nu,sigma)*(-q2(mu)-p1(mu)-p2(mu)))
  *(d_(sigmap,mup)*(p1(nup)+p2(nup)+q1(nup)) + d_(mup,nup)*(q2(sigmap)-q1(sigmap)) + d_(nup,sigmap)*(-q2(mup)-p1(mup)-p2(mup)))
  *(-(s/2)*d_(mu,mup) + q1(mu)*q2(mup) + q1(mup)*q2(mu))
  *(-(s/2)*d_(nu,nup) + q2(nu)*q1(nup) + q2(nup)*q1(nu));
trace4,1;
contract;
id p1.p1 = 0;
id p2.p2 = 0;
id q1.q1 = 0;
id q2.q2 = 0;
id p1.p2 = s/2;
id p1.q1 = -t/2;
id p1.q2 = -u/2;
id p2.q1 = -u/2;
id p2.q2 = -t/2;
id q1.q2 = s/2;
print +s Lss;
.sort

* ── Lts: t×s cross ──────────────────────────────────────
Local Lts = g_(1,p1)*g_(1,mu)*(g_(1,p1)-g_(1,q1))*g_(1,nu)*g_(1,p2)*g_(1,sigmap)
  *(d_(sigmap,mup)*(p1(nup)+p2(nup)+q1(nup)) + d_(mup,nup)*(q2(sigmap)-q1(sigmap)) + d_(nup,sigmap)*(-q2(mup)-p1(mup)-p2(mup)))
  *(-(s/2)*d_(mu,mup) + q1(mu)*q2(mup) + q1(mup)*q2(mu))
  *(-(s/2)*d_(nu,nup) + q2(nu)*q1(nup) + q2(nup)*q1(nu));
trace4,1;
contract;
id p1.p1 = 0;
id p2.p2 = 0;
id q1.q1 = 0;
id q2.q2 = 0;
id p1.p2 = s/2;
id p1.q1 = -t/2;
id p1.q2 = -u/2;
id p2.q1 = -u/2;
id p2.q2 = -t/2;
id q1.q2 = s/2;
print +s Lts;
.sort

* ── Lus: u×s cross ──────────────────────────────────────
Local Lus = g_(1,p1)*g_(1,beta)*(g_(1,p1)-g_(1,q2))*g_(1,alpha)*g_(1,p2)*g_(1,sigmap)
  *(d_(sigmap,mup)*(p1(nup)+p2(nup)+q1(nup)) + d_(mup,nup)*(q2(sigmap)-q1(sigmap)) + d_(nup,sigmap)*(-q2(mup)-p1(mup)-p2(mup)))
  *(-(s/2)*d_(alpha,mup) + q1(alpha)*q2(mup) + q1(mup)*q2(alpha))
  *(-(s/2)*d_(beta,nup) + q2(beta)*q1(nup) + q2(nup)*q1(beta));
trace4,1;
contract;
id p1.p1 = 0;
id p2.p2 = 0;
id q1.q1 = 0;
id q2.q2 = 0;
id p1.p2 = s/2;
id p1.q1 = -t/2;
id p1.q2 = -u/2;
id p2.q1 = -u/2;
id p2.q2 = -t/2;
id q1.q2 = s/2;
print +s Lus;
.end
"""

# qg→qg is related to qq̄→gg by crossing s↔t.
# p1=quark (in), p2=gluon (in), q1=quark (out), q2=gluon (out).
# Topologies: s-channel fermion, u-channel fermion, t-channel 3-gluon vertex.
# Only TWO external gluons: p2 (in) and q2 (out).
# Physical pol sums for p2 and q2 with Kleiss-Stirling: n_{p2}=q2, n_{q2}=p2.
#   Π_{μμ'}(p2) = -g_{μμ'} + (p2_μ q2_{μ'} + p2_{μ'} q2_μ)/(p2·q2)
#   Π_{νν'}(q2) = -g_{νν'} + (q2_ν p2_{ν'} + q2_{ν'} p2_ν)/(p2·q2)
# p2·q2 = -t/2, so multiply by (t/2)^2 and divide in Python.

_QG_QG_PHYSPOL_PROGRAM = r"""#-
Off Statistics;
Symbols s,t,u;
Vectors p1,p2,q1,q2;
Indices mu,nu,mup,nup,sigma,sigmap,alpha,beta,alp,bep;

* qg→qg with PHYSICAL polarization sums.
* p1=quark(in), p2=gluon(in,μ), q1=quark(out), q2=gluon(out,ν).
* Kleiss-Stirling gauge: n_{p2}=q2, n_{q2}=p2.
* p2·q2 = -t/2 > 0 for physical scattering (t<0).
* Scaled pol sums (multiply by -t/2):
*   P_{μμ'} = (t/2)*d_(μ,μ') + p2(μ)*q2(μ') + p2(μ')*q2(μ)
*   P_{νν'} = (t/2)*d_(ν,ν') + q2(ν)*p2(ν') + q2(ν')*p2(ν)
* Results multiplied by (-t/2)^2 = t^2/4. Divide by t^2/4 in Python.
*
* Amplitudes:
*   s-chan fermion: ū(q1) γ^ν (/p1+/p2) γ^μ u(p1) × ε_μ(p2) ε*_ν(q2) / s
*   u-chan fermion: ū(q1) γ^μ (/p1-/q2) γ^ν u(p1) × ε_μ(p2) ε*_ν(q2) / u
*   t-chan 3-gluon: ū(q1) γ^σ u(p1) × V_{σ}^{μν} × ε_μ(p2) ε*_ν(q2) / t
*
* V_{σμν}(k1=p1-q1, k2=p2, k3=-q2):
*   = g_{σμ}(p1-q1-p2)_ν + g_{μν}(p2+q2)_σ + g_{νσ}(q1-p1-q2)_μ

* ── Lss: s-channel fermion squared ──────────────────────
Local Lss = g_(1,q1)*g_(1,nu)*(g_(1,p1)+g_(1,p2))*g_(1,mu)
           *g_(1,p1)*g_(1,mup)*(g_(1,p1)+g_(1,p2))*g_(1,nup)
           *(t/2*d_(mu,mup) + p2(mu)*q2(mup) + p2(mup)*q2(mu))
           *(t/2*d_(nu,nup) + q2(nu)*p2(nup) + q2(nup)*p2(nu));
trace4,1;
contract;
id p1.p1 = 0; id p2.p2 = 0; id q1.q1 = 0; id q2.q2 = 0;
id p1.p2 = s/2; id p1.q1 = -t/2; id p1.q2 = -u/2;
id p2.q1 = -u/2; id p2.q2 = -t/2; id q1.q2 = s/2;
print +s Lss;
.sort

* ── Luu: u-channel fermion squared ──────────────────────
Local Luu = g_(1,q1)*g_(1,mu)*(g_(1,p1)-g_(1,q2))*g_(1,nu)
           *g_(1,p1)*g_(1,nup)*(g_(1,p1)-g_(1,q2))*g_(1,mup)
           *(t/2*d_(mu,mup) + p2(mu)*q2(mup) + p2(mup)*q2(mu))
           *(t/2*d_(nu,nup) + q2(nu)*p2(nup) + q2(nup)*p2(nu));
trace4,1;
contract;
id p1.p1 = 0; id p2.p2 = 0; id q1.q1 = 0; id q2.q2 = 0;
id p1.p2 = s/2; id p1.q1 = -t/2; id p1.q2 = -u/2;
id p2.q1 = -u/2; id p2.q2 = -t/2; id q1.q2 = s/2;
print +s Luu;
.sort

* ── Lsu: s×u cross ──────────────────────────────────────
Local Lsu = g_(1,q1)*g_(1,nu)*(g_(1,p1)+g_(1,p2))*g_(1,mu)
           *g_(1,p1)*g_(1,nup)*(g_(1,p1)-g_(1,q2))*g_(1,mup)
           *(t/2*d_(mu,mup) + p2(mu)*q2(mup) + p2(mup)*q2(mu))
           *(t/2*d_(nu,nup) + q2(nu)*p2(nup) + q2(nup)*p2(nu));
trace4,1;
contract;
id p1.p1 = 0; id p2.p2 = 0; id q1.q1 = 0; id q2.q2 = 0;
id p1.p2 = s/2; id p1.q1 = -t/2; id p1.q2 = -u/2;
id p2.q1 = -u/2; id p2.q2 = -t/2; id q1.q2 = s/2;
print +s Lsu;
.sort

* ── Ltt: t-channel 3-gluon squared ─────────────────────
Local Ltt = g_(1,q1)*g_(1,sigma)*g_(1,p1)*g_(1,sigmap)
  *(d_(sigma,mu)*(p1(nu)-q1(nu)-p2(nu)) + d_(mu,nu)*(p2(sigma)+q2(sigma)) + d_(nu,sigma)*(q1(mu)-p1(mu)-q2(mu)))
  *(d_(sigmap,mup)*(p1(nup)-q1(nup)-p2(nup)) + d_(mup,nup)*(p2(sigmap)+q2(sigmap)) + d_(nup,sigmap)*(q1(mup)-p1(mup)-q2(mup)))
  *(t/2*d_(mu,mup) + p2(mu)*q2(mup) + p2(mup)*q2(mu))
  *(t/2*d_(nu,nup) + q2(nu)*p2(nup) + q2(nup)*p2(nu));
trace4,1;
contract;
id p1.p1 = 0; id p2.p2 = 0; id q1.q1 = 0; id q2.q2 = 0;
id p1.p2 = s/2; id p1.q1 = -t/2; id p1.q2 = -u/2;
id p2.q1 = -u/2; id p2.q2 = -t/2; id q1.q2 = s/2;
print +s Ltt;
.sort

* ── Lst: s×t cross ──────────────────────────────────────
Local Lst = g_(1,q1)*g_(1,nu)*(g_(1,p1)+g_(1,p2))*g_(1,mu)*g_(1,p1)*g_(1,sigmap)
  *(d_(sigmap,mup)*(p1(nup)-q1(nup)-p2(nup)) + d_(mup,nup)*(p2(sigmap)+q2(sigmap)) + d_(nup,sigmap)*(q1(mup)-p1(mup)-q2(mup)))
  *(t/2*d_(mu,mup) + p2(mu)*q2(mup) + p2(mup)*q2(mu))
  *(t/2*d_(nu,nup) + q2(nu)*p2(nup) + q2(nup)*p2(nu));
trace4,1;
contract;
id p1.p1 = 0; id p2.p2 = 0; id q1.q1 = 0; id q2.q2 = 0;
id p1.p2 = s/2; id p1.q1 = -t/2; id p1.q2 = -u/2;
id p2.q1 = -u/2; id p2.q2 = -t/2; id q1.q2 = s/2;
print +s Lst;
.sort

* ── Lut: u×t cross ──────────────────────────────────────
Local Lut = g_(1,q1)*g_(1,mu)*(g_(1,p1)-g_(1,q2))*g_(1,nu)*g_(1,p1)*g_(1,sigmap)
  *(d_(sigmap,mup)*(p1(nup)-q1(nup)-p2(nup)) + d_(mup,nup)*(p2(sigmap)+q2(sigmap)) + d_(nup,sigmap)*(q1(mup)-p1(mup)-q2(mup)))
  *(t/2*d_(mu,mup) + p2(mu)*q2(mup) + p2(mup)*q2(mu))
  *(t/2*d_(nu,nup) + q2(nu)*p2(nup) + q2(nup)*p2(nu));
trace4,1;
contract;
id p1.p1 = 0; id p2.p2 = 0; id q1.q1 = 0; id q2.q2 = 0;
id p1.p2 = s/2; id p1.q1 = -t/2; id p1.q2 = -u/2;
id p2.q1 = -u/2; id p2.q2 = -t/2; id q1.q2 = s/2;
print +s Lut;
.end
"""


# ══════════════════════════════════════════════════════════════════════════════
# gg → gg: pure gluon scattering via 3-gluon + 4-gluon vertices
# ══════════════════════════════════════════════════════════════════════════════
#
# p1, p2 incoming gluons; q1, q2 outgoing gluons.  All massless.
# Momenta: p1 + p2 = q1 + q2,  s = 2(p1·p2), t = -2(p1·q1), u = -2(p1·q2).
#
# Three 3-gluon-vertex exchange diagrams:
#   As: s-channel — V_L^{μνσ}(p1,p2,-p1-p2) × (g_{σσ'}/s) × V_R^{σ'ρτ}(p1+p2,-q1,-q2)
#   At: t-channel — V_L^{μρσ}(p1,-q1,q1-p1) × (g_{σσ'}/t) × V_R^{σ'ντ}(p1-q1,p2,-q2)
#   Au: u-channel — V_L^{μτσ}(p1,-q2,q2-p1) × (g_{σσ'}/u) × V_R^{σ'νρ}(p1-q2,p2,-q1)
#
# Note: the gluon propagator is -ig_{σσ'}/k².  The -i is an overall phase
# that drops out in |M|²; the kinematic contraction uses +g_{σσ'}.
#
# The 4-gluon contact vertex decomposes into s,t,u color structures:
#   C_s^{μνρτ} = g^{μρ}g^{ντ} - g^{μτ}g^{νρ}   (color f^{abe}f^{cde})
#   C_t^{μνρτ} = g^{μν}g^{ρτ} - g^{μτ}g^{νρ}   (color f^{ace}f^{bde})
#   C_u^{μνρτ} = g^{μν}g^{ρτ} - g^{μρ}g^{ντ}   (color f^{ade}f^{bce})
#
# The effective kinematic amplitude for each color structure is:
#   K_s = V_L·V_R/s + C_s,  K_t = V_L·V_R/t + C_t,  K_u = V_L·V_R/u + C_u
# (both terms add with the SAME sign — verified against Combridge et al. 1977)
#
# Physical polarization sums for 4 external gluons (Kleiss-Stirling):
#   Π_μμ'(p1,n1=p2) = -g_{μμ'} + (p1_μ p2_{μ'} + p1_{μ'} p2_μ)/(p1·p2)
#   Π_νν'(p2,n2=p1) = -g_{νν'} + (p2_ν p1_{ν'} + p2_{ν'} p1_ν)/(p1·p2)
#   Π_ρρ'(q1,n3=q2) = -g_{ρρ'} + (q1_ρ q2_{ρ'} + q1_{ρ'} q2_ρ)/(q1·q2)
#   Π_ττ'(q2,n4=q1) = -g_{ττ'} + (q2_τ q1_{τ'} + q2_{τ'} q1_τ)/(q1·q2)
#
# Since p1·p2 = s/2 and q1·q2 = s/2, we multiply by (s/2)^4 to clear all
# denominators and divide by s^4/16 in Python.
#
# The FORM program computes Lorentz traces for all 6 independent pairs (ss, tt,
# uu, st, su, tu) where each amplitude includes 3-gluon + contact contributions.

def _generate_gg_gg_form_program() -> str:
    """Generate FORM program for gg→gg with all traces fully inlined.

    FORM cannot reference previously defined Local expressions as tensor
    functions, so each trace must be a single expression with the full
    numerator × numerator × 4 polarization sums written out.
    """
    header = (
        "#-\n"
        "Off Statistics;\n"
        "Symbols s,t,u;\n"
        "Vectors p1,p2,q1,q2;\n"
        "Indices mu,nu,rho,tau,mup,nup,rhop,taup,sig,sigp;\n\n"
    )

    # 3-gluon vertex V(a,b,c; k1,k2,k3) = d_(a,b)*(k1-k2)(c) + cyclic
    # All momenta incoming at each vertex.

    # s-channel numerator (× s to clear propagator):
    # N_s = V_L(mu,nu,sig) × V_R(sig,rho,tau) + s × contact_s
    # The propagator is -ig_{σσ'}/s; the -i is an overall phase that drops
    # out in |M|², so the kinematic contraction uses +g_{σσ'} (= FORM's d_).
    N_s = (
        "((d_(mu,nu)*(p1(sig)-p2(sig))"
        "+d_(nu,sig)*(p1(mu)+2*p2(mu))"
        "+d_(sig,mu)*(-2*p1(nu)-p2(nu)))"
        "*(d_(sig,rho)*(p1(tau)+p2(tau)+q1(tau))"
        "+d_(rho,tau)*(q2(sig)-q1(sig))"
        "+d_(tau,sig)*(-q2(rho)-p1(rho)-p2(rho)))"
        "+s*(d_(mu,rho)*d_(nu,tau)-d_(mu,tau)*d_(nu,rho)))"
    )

    # t-channel numerator (× t):
    N_t = (
        "((d_(mu,rho)*(p1(sig)+q1(sig))"
        "+d_(rho,sig)*(-2*q1(mu)+p1(mu))"
        "+d_(sig,mu)*(q1(rho)-2*p1(rho)))"
        "*(d_(sig,nu)*(p1(tau)-q1(tau)-p2(tau))"
        "+d_(nu,tau)*(p2(sig)+q2(sig))"
        "+d_(tau,sig)*(-q2(nu)-p1(nu)+q1(nu)))"
        "+t*(d_(mu,nu)*d_(rho,tau)-d_(mu,tau)*d_(nu,rho)))"
    )

    # u-channel numerator (× u):
    N_u = (
        "((d_(mu,tau)*(p1(sig)+q2(sig))"
        "+d_(tau,sig)*(-2*q2(mu)+p1(mu))"
        "+d_(sig,mu)*(q2(tau)-2*p1(tau)))"
        "*(d_(sig,nu)*(p1(rho)-q2(rho)-p2(rho))"
        "+d_(nu,rho)*(p2(sig)+q1(sig))"
        "+d_(rho,sig)*(-q1(nu)-p1(nu)+q2(nu)))"
        "+u*(d_(mu,nu)*d_(rho,tau)-d_(mu,rho)*d_(nu,tau)))"
    )

    # Primed copy (mu→mup, nu→nup, rho→rhop, tau→taup, sig→sigp)
    def _prime(expr: str) -> str:
        # Order matters: replace longer tokens first to avoid partial matches.
        out = expr
        for tok, rep in [("sigp", "_SIGP_HOLD_"),
                         ("sig", "sigp"),
                         ("_SIGP_HOLD_", "sigp")]:
            out = out.replace(tok, rep)
        # Now replace the 4 external indices.
        for tok, rep in [("rhop", "_RHOP_HOLD_"),
                         ("taup", "_TAUP_HOLD_"),
                         ("mu,", "mup,"), ("mu)", "mup)"),
                         ("nu,", "nup,"), ("nu)", "nup)"),
                         ("rho,", "rhop,"), ("rho)", "rhop)"),
                         ("tau,", "taup,"), ("tau)", "taup)"),
                         ("_RHOP_HOLD_", "rhop"),
                         ("_TAUP_HOLD_", "taup")]:
            out = out.replace(tok, rep)
        return out

    N_s_p = _prime(N_s)
    N_t_p = _prime(N_t)
    N_u_p = _prime(N_u)

    pol = (
        "*(-(s/2)*d_(mu,mup)+p1(mu)*p2(mup)+p1(mup)*p2(mu))\n"
        "  *(-(s/2)*d_(nu,nup)+p2(nu)*p1(nup)+p2(nup)*p1(nu))\n"
        "  *(-(s/2)*d_(rho,rhop)+q1(rho)*q2(rhop)+q1(rhop)*q2(rho))\n"
        "  *(-(s/2)*d_(tau,taup)+q2(tau)*q1(taup)+q2(taup)*q1(tau))"
    )

    kin = (
        "contract;\n"
        "id p1.p1 = 0; id p2.p2 = 0; id q1.q1 = 0; id q2.q2 = 0;\n"
        "id p1.p2 = s/2; id p1.q1 = -t/2; id p1.q2 = -u/2;\n"
        "id p2.q1 = -u/2; id p2.q2 = -t/2; id q1.q2 = s/2;\n"
    )

    pairs = [
        ("Lss", N_s, N_s_p),
        ("Ltt", N_t, N_t_p),
        ("Luu", N_u, N_u_p),
        ("Lst", N_s, N_t_p),
        ("Lsu", N_s, N_u_p),
        ("Ltu", N_t, N_u_p),
    ]

    body = ""
    for i, (name, Na, Nb) in enumerate(pairs):
        body += f"Local {name} = {Na}\n  *{Nb}\n  {pol};\n{kin}"
        body += f"print +s {name};\n"
        if i < len(pairs) - 1:
            body += ".sort\n\n"
        else:
            body += ".end\n"

    return header + body


_GG_GG_PHYSPOL_PROGRAM = _generate_gg_gg_form_program()


def _gg_to_gg_form(spec, tree_diagrams, theory):
    """Compute gg→gg using FORM with physical polarization sums.

    Pure gluon scattering via s, t, u-channel 3-gluon vertex exchange
    plus 4-gluon contact vertex. No Dirac traces — only tensor contractions.

    The contact vertex decomposes into s, t, u color structures, so each
    effective amplitude is (3-gluon diagram + contact piece) for one color
    structure.
    """
    from feynman_engine.amplitudes.color import gg_to_gg_color

    program = _GG_GG_PHYSPOL_PROGRAM
    raw_output = _run_form(program)
    if raw_output is None:
        return None

    parsed = _parse_form_output(raw_output)
    if not parsed:
        return None

    # Physical pol sum normalization: (s/2)^4 = s^4/16.
    norm = s_sym**4 / 16

    trace_map = {}
    for key in ("Lss", "Ltt", "Luu", "Lst", "Lsu", "Ltu"):
        val = parsed.get(key)
        if val is not None:
            trace_map[key] = val / norm
        else:
            trace_map[key] = Integer(0)

    # Color factors for gg→gg (color-summed).
    C_ss = gg_to_gg_color("s", "s")  # 72
    C_tt = gg_to_gg_color("t", "t")  # 72
    C_uu = gg_to_gg_color("u", "u")  # 72
    C_st = gg_to_gg_color("s", "t")  # 36
    C_su = gg_to_gg_color("s", "u")  # -36
    C_tu = gg_to_gg_color("t", "u")  # 36

    # Assemble: Σ C_ab × L_ab / (denom_a × denom_b)
    # Each effective numerator already has one factor of the channel variable
    # (Ns contains factor s, etc.), so the propagator is 1/s × 1/s = 1/s²,
    # and the Lorentz trace L_ss = Ns×Ns/(pol norm), which ~s^2 × (...)/s^4.
    # The denominators are s², t², u², st, su, tu.
    g_s = Symbol("g_s")
    msq = g_s**4 * (
        C_ss * trace_map["Lss"] / s_sym**2
        + C_tt * trace_map["Ltt"] / t_sym**2
        + C_uu * trace_map["Luu"] / u_sym**2
        + 2 * C_st * trace_map["Lst"] / (s_sym * t_sym)
        + 2 * C_su * trace_map["Lsu"] / (s_sym * u_sym)
        + 2 * C_tu * trace_map["Ltu"] / (t_sym * u_sym)
    )

    # Spin average: 1/4 (two spin-1 gluons × 2 helicities each).
    # Color average: 1/(Nc²-1)² = 1/64 (two adjoint gluons).
    msq = Rational(1, 256) * msq
    msq = cancel(msq)

    return AmplitudeResult(
        process=spec.raw,
        theory=theory,
        msq=msq,
        msq_latex=latex(msq),
        integral_latex=None,
        description="Exact tree-level gg→gg |M̄|² via FORM with physical polarization sums and SU(3) color",
        notes="Physical (Kleiss-Stirling) gauge. 3-gluon + 4-gluon contact vertices. All 6 Lorentz traces computed.",
        backend="form-symbolic",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2→3 QED: ff̄ → f'f̄'γ  (bremsstrahlung from any external fermion leg)
# ══════════════════════════════════════════════════════════════════════════════
#
# 4 tree-level diagrams (e+e- → μ+μ-γ as prototype):
#   D1: FSR from f'+ (μ+) — virtual γ(s-chan), internal f'(q1+q3)
#   D2: FSR from f'- (μ-) — virtual γ(s-chan), internal f'(q2+q3)
#   D3: ISR from f+ (e+)  — virtual γ(s12-chan), internal f(p1-q3)
#   D4: ISR from f- (e-)  — virtual γ(s12-chan), internal f(p2-q3)
#
# Momenta: p1(e+), p2(e-) incoming; q1(μ+), q2(μ-), q3(γ) outgoing.
# All massless.
#
# The FORM program computes 10 independent spin-summed traces T_{ij}
# (4 diagonal + 6 cross-interference) as polynomials in dot products.
# Result is in terms of p1.q1, p1.q2, p1.q3, p2.q1, p2.q2, p2.q3,
# q1.q2, q1.q3, q2.q3 (plus p1.p2 = s/2).
#
# The spin-averaged |M̄|² is assembled in Python:
#   |M̄|² = (e⁶/4) × Σ_{ij} mult_ij × T_ij / (denom_i × denom_j)
#
# Photon polarization sum (-g_{ρρ'}) is included in the FORM expressions.
# ══════════════════════════════════════════════════════════════════════════════

def _build_2to3_form_program(m_in_form: str = "", m_out_form: str = "") -> str:
    """Build the FORM program for ff̄→f'f̄'γ bremsstrahlung.

    Parameters
    ----------
    m_in_form : str
        FORM-safe mass symbol for initial-state fermion (e.g. "me").
        Empty string for massless.
    m_out_form : str
        FORM-safe mass symbol for final-state fermion (e.g. "mmu").
        Empty string for massless.

    Returns the complete FORM program as a string.
    """
    mi = m_in_form   # initial-state mass label (or "")
    mo = m_out_form   # final-state mass label (or "")

    # --- helpers for FORM expressions ---
    def ferm(L, mom, mass):
        """External fermion spin sum: /p + m."""
        s = f"g_({L},{mom})"
        return f"({s}+{mass}*gi_({L}))" if mass else s

    def anti(L, mom, mass):
        """External antifermion spin sum: /p - m."""
        s = f"g_({L},{mom})"
        return f"({s}-{mass}*gi_({L}))" if mass else s

    def prop_sum(L, moms, mass):
        """Internal propagator numerator: Σ /p_i + m."""
        parts = "+".join(f"g_({L},{m})" for m in moms)
        core = parts if len(moms) == 1 else parts
        return f"({core}+{mass}*gi_({L}))" if mass else f"({core})"

    def prop_diff(L, pos, neg, mass):
        """Internal propagator numerator: /p_pos - /p_neg + m."""
        core = f"g_({L},{pos})-g_({L},{neg})"
        return f"({core}+{mass}*gi_({L}))" if mass else f"({core})"

    g = lambda L, idx: f"g_({L},{idx})"  # bare gamma matrix

    # --- electron line pieces ---
    # s-channel electron trace (4 gammas): used in D1, D2 amplitudes
    # Tr[(/p1+m_in) γ^α (/p2-m_in) γ^{α'}]
    el_s = f"{ferm(1,'p1',mi)}*{g(1,'al')}*{anti(1,'p2',mi)}*{g(1,'alp')}"

    # ISR e- propagator trace pieces (6 gammas): used in D3
    # From amplitude: (/p1+m_in) γ^ρ (/p1-/q3+m_in) γ^β
    # From conjugate: (/p2-m_in) γ^{β'} (/p1-/q3+m_in) γ^{ρ'}
    el_d3_amp = f"{ferm(1,'p1',mi)}*{g(1,'rho')}*{prop_diff(1,'p1','q3',mi)}*{g(1,'be')}"
    el_d3_conj = f"{anti(1,'p2',mi)}*{g(1,'bep')}*{prop_diff(1,'p1','q3',mi)}*{g(1,'rhop')}"

    # ISR e+ propagator trace pieces (6 gammas): used in D4
    # From amplitude: (/p1+m_in) γ^β (/p2-/q3+m_in) γ^ρ
    # From conjugate: (/p2-m_in) γ^{ρ'} (/p2-/q3+m_in) γ^{β'}
    el_d4_amp = f"{ferm(1,'p1',mi)}*{g(1,'be')}*{prop_diff(1,'p2','q3',mi)}*{g(1,'rho')}"
    el_d4_conj = f"{anti(1,'p2',mi)}*{g(1,'rhop')}*{prop_diff(1,'p2','q3',mi)}*{g(1,'bep')}"

    # --- muon line pieces ---
    # s-channel muon trace (4 gammas): used in D3, D4 amplitudes
    # Tr[(/q1+m_out) γ^β (/q2-m_out) γ^{β'}]
    mu_s = f"{ferm(2,'q1',mo)}*{g(2,'be')}*{anti(2,'q2',mo)}*{g(2,'bep')}"

    # FSR μ- propagator trace pieces (8 gammas): used in D1
    # From amplitude: (/q1+m_out) γ^ρ (/q1+/q3+m_out) γ^α
    # From conjugate: (/q2-m_out) γ^{α'} (/q1+/q3+m_out) γ^{ρ'}
    mu_d1_amp = f"{ferm(2,'q1',mo)}*{g(2,'rho')}*{prop_sum(2,['q1','q3'],mo)}*{g(2,'al')}"
    mu_d1_conj = f"{anti(2,'q2',mo)}*{g(2,'alp')}*{prop_sum(2,['q1','q3'],mo)}*{g(2,'rhop')}"

    # FSR μ+ propagator trace pieces (8 gammas): used in D2
    # From amplitude: (/q1+m_out) γ^α (/q2+/q3+m_out) γ^ρ
    # From conjugate: (/q2-m_out) γ^{ρ'} (/q2+/q3+m_out) γ^{α'}
    mu_d2_amp = f"{ferm(2,'q1',mo)}*{g(2,'al')}*{prop_sum(2,['q2','q3'],mo)}*{g(2,'rho')}"
    mu_d2_conj = f"{anti(2,'q2',mo)}*{g(2,'rhop')}*{prop_sum(2,['q2','q3'],mo)}*{g(2,'alp')}"

    # --- FSR×ISR cross muon lines (6 gammas) ---
    # D1 amplitude muon part + D3/D4 conjugate muon part:
    # D1 amp:  (/q1+m) γ^ρ (/q1+/q3+m) γ^α
    # D3 conj muon: (/q2-m) γ^{β'}   (just the simple vertex)
    mu_d1xd3 = f"{ferm(2,'q1',mo)}*{g(2,'rho')}*{prop_sum(2,['q1','q3'],mo)}*{g(2,'al')}*{anti(2,'q2',mo)}*{g(2,'bep')}"

    # D2 amp:  (/q1+m) γ^α (/q2+/q3+m) γ^ρ
    # D3 conj muon: (/q2-m) γ^{β'}
    mu_d2xd3 = f"{ferm(2,'q1',mo)}*{g(2,'al')}*{prop_sum(2,['q2','q3'],mo)}*{g(2,'rho')}*{anti(2,'q2',mo)}*{g(2,'bep')}"

    # FSR×ISR electron cross lines (6 gammas):
    # D1 amp electron: (/p1+m_in) γ^α  ×  D3 conj electron: (/p2-m_in) γ^{β'} (/p1-/q3+m_in) γ^{ρ'}
    el_d1xd3 = f"{ferm(1,'p1',mi)}*{g(1,'al')}*{anti(1,'p2',mi)}*{g(1,'bep')}*{prop_diff(1,'p1','q3',mi)}*{g(1,'rhop')}"

    # D1 amp electron × D4 conj electron: (/p2-m_in) γ^{ρ'} (/p2-/q3+m_in) γ^{β'}
    el_d1xd4 = f"{ferm(1,'p1',mi)}*{g(1,'al')}*{anti(1,'p2',mi)}*{g(1,'rhop')}*{prop_diff(1,'p2','q3',mi)}*{g(1,'bep')}"

    # --- build on-shell conditions ---
    onshell = []
    onshell.append(f"id p1.p1 = {mi+'^2' if mi else '0'};")
    onshell.append(f"id p2.p2 = {mi+'^2' if mi else '0'};")
    onshell.append(f"id q1.q1 = {mo+'^2' if mo else '0'};")
    onshell.append(f"id q2.q2 = {mo+'^2' if mo else '0'};")
    onshell.append("id q3.q3 = 0;")

    # --- symbol declarations ---
    sym_decl = ""
    mass_syms = {s for s in (mi, mo) if s}
    if mass_syms:
        sym_decl = f"Symbol {','.join(sorted(mass_syms))};\n"

    # --- assemble the FORM program ---
    program = f"""#-
Off Statistics;
Vectors p1,p2,q1,q2,q3;
Indices al,alp,be,bep,rho,rhop;
{sym_decl}
* FSR x FSR group
Local D11 = (-d_(rho,rhop))
  *{el_s}
  *{mu_d1_amp}
  *{mu_d1_conj};

Local D22 = (-d_(rho,rhop))
  *{el_s}
  *{mu_d2_amp}
  *{mu_d2_conj};

Local D12 = (-d_(rho,rhop))
  *{el_s}
  *{mu_d1_amp}
  *{mu_d2_conj};

* ISR x ISR group
Local D33 = (-d_(rho,rhop))
  *{el_d3_amp}
  *{el_d3_conj}
  *{mu_s};

Local D44 = (-d_(rho,rhop))
  *{el_d4_amp}
  *{el_d4_conj}
  *{mu_s};

Local D34 = (-d_(rho,rhop))
  *{el_d3_amp}
  *{el_d4_conj}
  *{mu_s};

* FSR x ISR cross group
Local D13 = (-d_(rho,rhop))
  *{el_d1xd3}
  *{mu_d1xd3};

Local D14 = (-d_(rho,rhop))
  *{el_d1xd4}
  *{mu_d1xd3};

Local D23 = (-d_(rho,rhop))
  *{el_d1xd3}
  *{mu_d2xd3};

Local D24 = (-d_(rho,rhop))
  *{el_d1xd4}
  *{mu_d2xd3};

trace4,1;
trace4,2;
contract;

{chr(10).join(onshell)}

print +s D11;
print +s D22;
print +s D12;
print +s D33;
print +s D44;
print +s D34;
print +s D13;
print +s D14;
print +s D23;
print +s D24;
.end
"""
    return program


# Keep backward compat: the massless program is just the default
_EEMM_GAMMA_FORM_PROGRAM = _build_2to3_form_program("", "")


def _get_2to3_qed_amplitude(spec, theory) -> Optional[AmplitudeResult]:
    """Compute |M̄|² for a QED 2→3 process: ff̄ → f'f̄'γ.

    Uses FORM to compute all 10 spin-summed interference traces, then
    assembles with propagator denominators and coupling constants.

    The result is expressed in terms of dot products (p1·q1, etc.)
    rather than Mandelstam invariants, suitable for numerical evaluation
    over RAMBO phase space points.
    """
    if not form_available():
        return None

    # Classify particles.
    quarks = {"u", "d", "s", "c", "b", "t", "u~", "d~", "s~", "c~", "b~", "t~"}
    fermions = {"e+", "e-", "mu+", "mu-", "tau+", "tau-"} | quarks
    bosons = {"gamma", "g"}

    n_out_ferm = sum(1 for p in spec.outgoing if p in fermions)
    n_out_boson = sum(1 for p in spec.outgoing if p in bosons)
    n_in_ferm = sum(1 for p in spec.incoming if p in fermions)

    # Must be ff̄ → f'f̄'γ topology.
    if n_in_ferm != 2 or n_out_ferm != 2 or n_out_boson != 1:
        return None
    if len(spec.outgoing) != 3:
        return None

    # Check that initial fermions are different flavor from final fermions
    # (same-flavor has extra diagrams we don't handle yet).
    in_flavors = set()
    for p in spec.incoming:
        in_flavors.add(p.rstrip("+-").replace("~", ""))
    out_flavors = set()
    for p in spec.outgoing:
        if p in fermions:
            out_flavors.add(p.rstrip("+-").replace("~", ""))

    if in_flavors & out_flavors:
        # Same-flavor case (e.g., e+e-→e+e-γ) — not yet supported.
        return None

    # Detect masses from the particle registry.
    in_ferm = [p for p in spec.incoming if p in fermions]
    out_ferm = [p for p in spec.outgoing if p in fermions]
    m_in_label = _particle_mass_label(theory, in_ferm[0]) if in_ferm else None
    m_out_label = _particle_mass_label(theory, out_ferm[0]) if out_ferm else None
    m_in_form = _to_form_name(m_in_label) if m_in_label else ""
    m_out_form = _to_form_name(m_out_label) if m_out_label else ""

    program = _build_2to3_form_program(m_in_form, m_out_form)
    raw_output = _run_form(program)
    if raw_output is None:
        return None

    traces = _parse_form_2to3_output(raw_output)
    if not traces:
        return None

    from sympy import Rational, cancel

    # Must match the assumptions used in _parse_form_2to3_output (real=True)
    # so that SymPy treats them as the same symbol objects.
    p1p2 = Symbol("p1p2", real=True)
    p1q1 = Symbol("p1q1", real=True)
    p1q2 = Symbol("p1q2", real=True)
    p1q3 = Symbol("p1q3", real=True)
    p2q1 = Symbol("p2q1", real=True)
    p2q2 = Symbol("p2q2", real=True)
    p2q3 = Symbol("p2q3", real=True)
    q1q2 = Symbol("q1q2", real=True)
    q1q3 = Symbol("q1q3", real=True)
    q2q3 = Symbol("q2q3", real=True)

    # Mass-squared symbols for propagator denominators.
    # Use FORM-safe names (no underscores) to match the symbols in the parsed traces.
    m_in_sq = Symbol(m_in_form, real=True, positive=True) ** 2 if m_in_form else Integer(0)
    m_out_sq = Symbol(m_out_form, real=True, positive=True) ** 2 if m_out_form else Integer(0)

    # Propagator denominators with masses:
    # (q1+q3)² = q1² + 2q1·q3 + q3² = m_out² + 2*q1q3
    # (q2+q3)² = m_out² + 2*q2q3
    # (q1+q2)² = 2*m_out² + 2*q1q2
    # (p1-q3)² = m_in² - 2*p1q3
    # (p2-q3)² = m_in² - 2*p2q3
    # s = (p1+p2)² = 2*m_in² + 2*p1p2
    s_val = 2 * m_in_sq + 2 * p1p2
    denom = {
        1: s_val * (m_out_sq + 2 * q1q3),         # s × (q1+q3)²
        2: s_val * (m_out_sq + 2 * q2q3),         # s × (q2+q3)²
        3: (2 * m_out_sq + 2 * q1q2) * (m_in_sq - 2 * p1q3),  # (q1+q2)² × (p1-q3)²
        4: (2 * m_out_sq + 2 * q1q2) * (m_in_sq - 2 * p2q3),  # (q1+q2)² × (p2-q3)²
    }

    # Assemble: |M̄|² = (e⁶/4) Σ mult × T_ij / (denom_i × denom_j)
    e_sym = Symbol("e")
    msq = Integer(0)

    pairs = [
        ("D11", 1, 1, 1), ("D22", 2, 2, 1), ("D12", 1, 2, 2),
        ("D33", 3, 3, 1), ("D44", 4, 4, 1), ("D34", 3, 4, 2),
        ("D13", 1, 3, 2), ("D14", 1, 4, 2), ("D23", 2, 3, 2), ("D24", 2, 4, 2),
    ]

    for name, di, dj, mult in pairs:
        trace = traces.get(name)
        if trace is None or trace == 0:
            continue
        msq += Integer(mult) * trace / (denom[di] * denom[dj])

    msq = e_sym**6 * Rational(1, 4) * msq

    has_masses = bool(m_in_label or m_out_label)
    return AmplitudeResult(
        process=spec.raw,
        theory=theory,
        msq=msq,
        msq_latex=latex(msq),
        integral_latex=None,
        description=(
            f"Tree-level 2→3 |M̄|² via FORM traces — "
            f"4 diagrams (2 FSR + 2 ISR), 10 interference terms"
            f"{', massive fermions' if has_masses else ''}"
        ),
        notes=(
            "Spin-averaged (1/4). Expressed in dot products for MC evaluation. "
            "Use RAMBO phase space integration for cross section."
        ),
        backend="form-symbolic-2to3",
    )


def _parse_form_2to3_output(stdout: str) -> dict[str, object]:
    """Parse FORM output for 2→3 processes where results are in dot products.

    FORM outputs dot products as "p1.q1", "p2.q3", etc. We convert these
    to SymPy symbols named "p1q1", "p2q3", etc.
    """
    results = {}

    # FORM output format:
    #    D11 =
    #       32*p1.q1*p2.q2 - 16*p1.q2*p2.q1;
    pattern = re.compile(r"^\s+(\w+)\s*=$", re.MULTILINE)
    matches = list(pattern.finditer(stdout))

    for k, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[k + 1].start() if k + 1 < len(matches) else len(stdout)

        expr_text = stdout[start:end]
        semi_pos = expr_text.rfind(";")
        if semi_pos >= 0:
            expr_text = expr_text[:semi_pos].strip()
        else:
            expr_text = expr_text.strip()

        expr_text = " ".join(expr_text.split())

        if not expr_text or expr_text == "0":
            results[name] = Integer(0)
            continue

        # Replace "p1.q1" → "p1q1" for SymPy parsing.
        expr_text = expr_text.replace("^", "**")
        expr_text = re.sub(r'([pq]\d)\.([pq]\d)', r'\1\2', expr_text)

        local_dict = {}
        for a in ("p1", "p2", "q1", "q2", "q3"):
            for b in ("p1", "p2", "q1", "q2", "q3"):
                sym_name = a + b
                local_dict[sym_name] = Symbol(sym_name, real=True)

        # Detect mass symbols (any bare identifiers not already covered).
        # FORM mass symbols appear as e.g. "mmu" or "me" in the expression.
        for token in re.findall(r'\b([a-zA-Z_]\w*)\b', expr_text):
            if token not in local_dict and token not in ("E", "I", "pi"):
                local_dict[token] = Symbol(token, real=True, positive=True)

        try:
            expr = sympify(expr_text, locals=local_dict)
            results[name] = expr
        except Exception:
            continue

    return results
