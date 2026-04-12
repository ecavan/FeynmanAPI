"""Generic symbolic tree-level amplitudes built from QGRAF diagram structure."""
from __future__ import annotations

from dataclasses import dataclass

from sympy import Integer, Rational, Symbol, cancel, latex, simplify
from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex, gamma_trace
from sympy.tensor.tensor import TensAdd, TensMul, tensor_heads, tensor_indices

from feynman_engine.amplitudes.types import AmplitudeResult
from feynman_engine.core.models import Diagram, Edge, Particle, ParticleType
from feynman_engine.core.generator import generate_diagrams
from feynman_engine.physics.registry import TheoryRegistry
from feynman_engine.physics.translator import parse_process


_SUPPORTED_TOPOLOGIES = {"s-channel", "t-channel", "u-channel"}
_VECTOR_STYLES = {"fermion", "anti fermion", "photon", "boson", "gluon", "charged boson"}
_FERMION_STYLES = {"fermion", "anti fermion"}


@dataclass
class FermionPair:
    antifermion: Edge
    fermion: Edge
    mass: object


@dataclass
class ScalarPair:
    particle: Edge
    antiparticle: Edge
    mass: object


@dataclass
class DiagramTerm:
    diagram_id: int
    theory: str
    mediator_name: str
    mediator_kind: str
    incoming_pair: FermionPair
    outgoing_pair: FermionPair | ScalarPair
    outgoing_kind: str
    coupling_in: object
    coupling_out: object
    denominator: object
    symmetry_factor: object
    notes: list[str]
    topology: str = "s-channel"


def get_symbolic_amplitude(process: str, theory: str = "QED") -> AmplitudeResult | None:
    """Generate a symbolic tree-level |M|² from supported QGRAF diagrams."""
    theory = theory.upper()
    spec = parse_process(process.strip(), theory=theory, loops=0)

    if len(spec.incoming) != 2 or len(spec.outgoing) != 2:
        return None

    diagrams = generate_diagrams(spec)
    if not diagrams:
        return None

    # Only include tree-level diagrams with supported topologies.
    supported = [d for d in diagrams
                 if d.loop_order == 0 and d.topology in _SUPPORTED_TOPOLOGIES]
    if not supported:
        return None

    terms: list[DiagramTerm] = []
    notes: list[str] = []
    for diagram in supported:
        term = _diagram_to_term(diagram)
        if term is None:
            return None
        terms.append(term)
        notes.extend(term.notes)

    if not terms:
        return None

    # Build kinematic dot-product map from the physical incoming/outgoing edges.
    kin = _universal_kinematics_context(supported[0])

    # Compute spin-summed |M|².
    # Same-topology pairs: factorizable into two 4-gamma traces.
    # t×u pairs: non-factorizable 8-gamma trace, computed via _tu_cross_interference.
    # s×t pairs: require a different spinor-flow analysis; skipped with a note.
    topologies_set = {t.topology for t in terms}
    has_st_cross = bool(topologies_set & {"s-channel"}) and bool(
        topologies_set & {"t-channel", "u-channel"}
    )

    # Compton diagrams need their own trace structure (fermion mediator + photon pol sum).
    compton_terms = [t for t in terms if t.mediator_kind == "compton"]
    normal_terms = [t for t in terms if t.mediator_kind != "compton"]

    # Compton: diagonal |M|² per diagram (s×u Compton interference is a future TODO).
    compton_msq_total = Integer(0)
    for ct in compton_terms:
        compton_msq_total += _compton_msq(ct, kin)

    # Normal (vector/scalar mediator) diagrams: double sum with 1/4 spin average.
    normal_msq = Integer(0)
    for left in normal_terms:
        for right in normal_terms:
            if left.topology == right.topology:
                contrib = _interference(left, right)
                if contrib == 0:
                    continue
                normal_msq += (
                    left.symmetry_factor
                    * right.symmetry_factor
                    * left.coupling_in
                    * left.coupling_out
                    * right.coupling_in
                    * right.coupling_out
                    * contrib
                    / (left.denominator * right.denominator)
                )
            elif {left.topology, right.topology} == {"t-channel", "u-channel"}:
                # 8-gamma cross trace for t×u (and u×t) interference.
                t_term = left if left.topology == "t-channel" else right
                u_term = right if left.topology == "t-channel" else left
                cross = _tu_cross_interference(t_term, u_term)
                if cross == 0:
                    continue
                normal_msq += (
                    left.symmetry_factor
                    * right.symmetry_factor
                    * left.coupling_in
                    * left.coupling_out
                    * right.coupling_in
                    * right.coupling_out
                    * cross
                    / (left.denominator * right.denominator)
                )
            # s×t cross-topology: requires separate spinor-flow analysis; skipped.

    color = _qcd_color_factor(normal_terms or terms)
    msq = Rational(1, 4) * color * normal_msq + compton_msq_total
    msq = simplify(cancel(_tensor_expr_to_scalar(msq, kin)))

    unique_notes = []
    for note in notes:
        if note not in unique_notes:
            unique_notes.append(note)
    if has_st_cross:
        unique_notes.append(
            "s-t cross-topology interference is omitted; "
            "exact s×t interference requires a dedicated spinor-flow analysis."
        )
    if len(compton_terms) > 1:
        unique_notes.append(
            "Compton s×u cross-diagram interference is omitted (future TODO)."
        )

    topologies_present = sorted({t.topology for t in terms})
    compton_note = " (Compton-type)" if compton_terms else ""
    description = (
        f"Generated from {len(terms)} QGRAF tree-level "
        f"{', '.join(topologies_present)} diagram(s){compton_note}"
    )
    return AmplitudeResult(
        process=spec.raw,
        theory=theory,
        msq=msq,
        msq_latex=latex(msq),
        description=description,
        notes=" ".join(unique_notes),
        backend="qgraf-symbolic",
    )


def _diagram_to_term(diagram: Diagram) -> DiagramTerm | None:
    if diagram.topology not in _SUPPORTED_TOPOLOGIES or len(diagram.internal_edges) != 1:
        return None

    internal = diagram.internal_edges[0]
    mediator = TheoryRegistry.get_particle(diagram.theory, internal.particle)

    # ── Compton-type: fermion mediator with mixed fermion+photon external legs ─
    if mediator.propagator_style.value in _FERMION_STYLES:
        return _compton_term(diagram, internal, mediator)

    mediator_kind = _mediator_kind(mediator)
    if mediator_kind is None:
        return None

    # ── t-channel / u-channel path ────────────────────────────────────────────
    if diagram.topology in {"t-channel", "u-channel"}:
        return _diagram_to_term_tu(diagram, internal, mediator, mediator_kind)

    # ── s-channel path ────────────────────────────────────────────────────────
    incoming_edges_by_vertex: dict[int, list[Edge]] = {}
    outgoing_edges_by_vertex: dict[int, list[Edge]] = {}
    for edge in diagram.external_edges:
        if edge.start_vertex < 0 and edge.end_vertex >= 0:
            incoming_edges_by_vertex.setdefault(edge.end_vertex, []).append(edge)
        elif edge.end_vertex < 0 and edge.start_vertex >= 0:
            outgoing_edges_by_vertex.setdefault(edge.start_vertex, []).append(edge)

    incoming_vertex = next((vid for vid, edges in incoming_edges_by_vertex.items() if len(edges) == 2), None)
    outgoing_vertex = next((vid for vid, edges in outgoing_edges_by_vertex.items() if len(edges) == 2), None)
    if incoming_vertex is None or outgoing_vertex is None:
        return None

    incoming_pair = _fermion_pair(diagram.theory, incoming_edges_by_vertex[incoming_vertex])
    if incoming_pair is None:
        return None

    outgoing_edges = outgoing_edges_by_vertex[outgoing_vertex]
    outgoing_pair, outgoing_kind = _outgoing_current(diagram.theory, outgoing_edges, mediator_kind)
    if outgoing_pair is None or outgoing_kind is None:
        return None

    notes: list[str] = []
    coupling_in = _coupling_symbol(diagram.theory, mediator.name, incoming_edges_by_vertex[incoming_vertex])
    coupling_out = _coupling_symbol(diagram.theory, mediator.name, outgoing_edges)

    if diagram.theory == "EW" and mediator.name in {"Z", "W+", "W-"}:
        notes.append(
            "Weak-boson couplings are currently modeled as vector-like symbols; "
            "full chiral gamma5/projector structure is not implemented yet."
        )
    if diagram.theory == "EW" and mediator.name == "H":
        notes.append("Scalar Higgs exchange is modeled with symbolic Yukawa couplings.")
    if diagram.theory == "QCD":
        notes.append("QCD results currently include the s-channel qqbar color factor for supported processes.")

    return DiagramTerm(
        diagram_id=diagram.id,
        theory=diagram.theory,
        mediator_name=mediator.name,
        mediator_kind=mediator_kind,
        incoming_pair=incoming_pair,
        outgoing_pair=outgoing_pair,
        outgoing_kind=outgoing_kind,
        coupling_in=coupling_in,
        coupling_out=coupling_out,
        denominator=_channel_invariant(diagram.topology) - _mass_squared(mediator.mass),
        symmetry_factor=Integer(1) if diagram.symmetry_factor is None else Rational(str(diagram.symmetry_factor)),
        notes=notes,
        topology="s-channel",
    )


def _diagram_to_term_tu(diagram: Diagram, internal: Edge, mediator, mediator_kind: str) -> DiagramTerm | None:
    """Build a DiagramTerm for a t-channel or u-channel diagram.

    For 2→2 scattering via a single exchanged vector boson, each vertex has
    exactly one incoming and one outgoing external fermion line.  We reuse the
    FermionPair fields with the convention:
        FermionPair.fermion      = incoming external edge at that vertex
        FermionPair.antifermion  = outgoing external edge at that vertex
    This gives the correct trace Tr[(p̸_in)γ^μ(p̸_out)γ^ν] after spin summation.
    """
    if mediator_kind != "vector":
        return None  # only vector-boson exchange supported in t/u-channel

    v_start = internal.start_vertex
    v_end = internal.end_vertex

    def vertex_ext(vid: int) -> tuple[list[Edge], list[Edge]]:
        in_e = [e for e in diagram.external_edges if e.end_vertex == vid and e.start_vertex < 0]
        out_e = [e for e in diagram.external_edges if e.start_vertex == vid and e.end_vertex < 0]
        return in_e, out_e

    in_A, out_A = vertex_ext(v_start)
    in_B, out_B = vertex_ext(v_end)

    if not (len(in_A) == 1 and len(out_A) == 1 and len(in_B) == 1 and len(out_B) == 1):
        return None

    # Reject external vector-boson legs (e.g. Compton: external photon + internal fermion)
    for e in in_A + out_A + in_B + out_B:
        p = TheoryRegistry.get_particle(diagram.theory, e.particle)
        if p.particle_type not in {ParticleType.FERMION, ParticleType.ANTIFERMION}:
            return None

    mass_A = _mass_symbol(diagram.theory, in_A[0].particle)
    pair_A = FermionPair(fermion=in_A[0], antifermion=out_A[0], mass=mass_A)

    mass_B = _mass_symbol(diagram.theory, in_B[0].particle)
    pair_B = FermionPair(fermion=in_B[0], antifermion=out_B[0], mass=mass_B)

    notes: list[str] = []
    if diagram.theory == "EW" and mediator.name in {"Z", "W+", "W-"}:
        notes.append(
            "Weak-boson couplings are currently modeled as vector-like symbols; "
            "full chiral gamma5/projector structure is not implemented yet."
        )
    if diagram.theory == "QCD":
        notes.append(
            "QCD t/u-channel results use a placeholder color factor; "
            "exact SU(3) averaging not yet implemented for this topology."
        )

    sym = (
        Integer(1)
        if diagram.symmetry_factor is None
        else Rational(str(diagram.symmetry_factor))
    )

    return DiagramTerm(
        diagram_id=diagram.id,
        theory=diagram.theory,
        mediator_name=mediator.name,
        mediator_kind="vector",
        incoming_pair=pair_A,
        outgoing_pair=pair_B,
        outgoing_kind="fermion_vector",
        coupling_in=_coupling_symbol(diagram.theory, mediator.name, [in_A[0], out_A[0]]),
        coupling_out=_coupling_symbol(diagram.theory, mediator.name, [in_B[0], out_B[0]]),
        denominator=_channel_invariant(diagram.topology) - _mass_squared(mediator.mass),
        symmetry_factor=sym,
        notes=notes,
        topology=diagram.topology,
    )


def _fermion_pair(theory: str, edges: list[Edge]) -> FermionPair | None:
    anti = None
    ferm = None
    for edge in edges:
        particle = TheoryRegistry.get_particle(theory, edge.particle)
        if particle.particle_type == ParticleType.ANTIFERMION:
            anti = edge
        elif particle.particle_type == ParticleType.FERMION:
            ferm = edge
    if anti is None or ferm is None:
        return None
    return FermionPair(antifermion=anti, fermion=ferm, mass=_mass_symbol(theory, ferm.particle))


def _outgoing_current(theory: str, edges: list[Edge], mediator_kind: str) -> tuple[FermionPair | ScalarPair | None, str | None]:
    fermion_pair = _fermion_pair(theory, edges)
    if fermion_pair is not None:
        if mediator_kind == "vector":
            return fermion_pair, "fermion_vector"
        if mediator_kind == "scalar":
            return fermion_pair, "fermion_scalar"
        return None, None

    if mediator_kind != "vector":
        return None, None

    particles = [TheoryRegistry.get_particle(theory, edge.particle) for edge in edges]
    if not all(p.particle_type == ParticleType.SCALAR for p in particles):
        return None, None

    pair = _scalar_pair(theory, edges)
    if pair is None:
        return None, None
    return pair, "scalar_vector"


def _scalar_pair(theory: str, edges: list[Edge]) -> ScalarPair | None:
    first, second = edges
    first_particle = TheoryRegistry.get_particle(theory, first.particle)
    second_particle = TheoryRegistry.get_particle(theory, second.particle)

    if first_particle.antiparticle == second_particle.name:
        return ScalarPair(particle=first, antiparticle=second, mass=_mass_symbol(theory, first.particle))
    if second_particle.antiparticle == first_particle.name:
        return ScalarPair(particle=second, antiparticle=first, mass=_mass_symbol(theory, second.particle))
    if first_particle.name.endswith("~"):
        return ScalarPair(particle=second, antiparticle=first, mass=_mass_symbol(theory, second.particle))
    return ScalarPair(particle=first, antiparticle=second, mass=_mass_symbol(theory, first.particle))


def _mediator_kind(particle: Particle) -> str | None:
    if particle.particle_type == ParticleType.SCALAR:
        return "scalar"
    if particle.propagator_style.value in _VECTOR_STYLES:
        return "vector"
    return None


def _coupling_symbol(theory: str, mediator_name: str, edges: list[Edge]):
    species = _species_key(edges[0].particle)
    if mediator_name == "gamma":
        return Symbol("e")
    if mediator_name == "g":
        return Symbol("g_s")
    if mediator_name == "Z":
        return Symbol(f"g_Z_{species}")
    if mediator_name in {"W+", "W-"}:
        return Symbol(f"g_{_sanitize(mediator_name)}_{_species_key(edges[0].particle)}_{_species_key(edges[1].particle)}")
    if mediator_name == "H":
        return Symbol(f"y_{species}")
    if mediator_name == "Zp":
        return Symbol(f"g_Zp_{species}")
    return Symbol(f"g_{_sanitize(mediator_name)}_{species}")


def _species_key(name: str) -> str:
    return _sanitize(name.rstrip("+-").replace("~", "bar"))


def _sanitize(name: str) -> str:
    return (
        name.replace("+", "p")
        .replace("-", "m")
        .replace("~", "bar")
        .replace("'", "p")
    )


def _channel_invariant(topology: str):
    mapping = {
        "s-channel": Symbol("s"),
        "t-channel": Symbol("t"),
        "u-channel": Symbol("u"),
    }
    return mapping[topology]


def _mass_symbol(theory: str, particle_name: str):
    particle = TheoryRegistry.get_particle(theory, particle_name)
    return _mass_expression(particle.mass)


def _mass_expression(label: str | None):
    if not label or label == "0":
        return Integer(0)
    return Symbol(label)


def _mass_squared(label: str | None):
    mass = _mass_expression(label)
    return Integer(0) if mass == 0 else mass**2


def _universal_kinematics_context(diagram: Diagram) -> dict:
    """Build the dot-product substitution map from the diagram's external edges.

    Keys are either (momentum_label, momentum_label) for p·p self-products or
    frozenset({label_a, label_b}) for cross products.  Values are Mandelstam
    expressions following the 2→2 convention:

        s = (p1+p2)²,  t = (p1-q1)²,  u = (p1-q2)²

    where p1, p2 are the first and second incoming momenta and q1, q2 are the
    first and second outgoing momenta (in QGRAF encounter order).
    This is topology-independent and correct for s-, t-, and u-channel diagrams.
    """
    in_edges = [e for e in diagram.external_edges
                if e.start_vertex < 0 and e.end_vertex >= 0]
    out_edges = [e for e in diagram.external_edges
                 if e.end_vertex < 0 and e.start_vertex >= 0]

    if len(in_edges) != 2 or len(out_edges) != 2:
        return {}

    p1_lbl = in_edges[0].momentum
    p2_lbl = in_edges[1].momentum
    q1_lbl = out_edges[0].momentum
    q2_lbl = out_edges[1].momentum

    m1_sq = _mass_squared(TheoryRegistry.get_particle(diagram.theory, in_edges[0].particle).mass)
    m2_sq = _mass_squared(TheoryRegistry.get_particle(diagram.theory, in_edges[1].particle).mass)
    m3_sq = _mass_squared(TheoryRegistry.get_particle(diagram.theory, out_edges[0].particle).mass)
    m4_sq = _mass_squared(TheoryRegistry.get_particle(diagram.theory, out_edges[1].particle).mass)

    s, t, u = Symbol("s"), Symbol("t"), Symbol("u")

    return {
        (p1_lbl, p1_lbl): m1_sq,
        (p2_lbl, p2_lbl): m2_sq,
        (q1_lbl, q1_lbl): m3_sq,
        (q2_lbl, q2_lbl): m4_sq,
        frozenset({p1_lbl, p2_lbl}): (s - m1_sq - m2_sq) / 2,
        frozenset({q1_lbl, q2_lbl}): (s - m3_sq - m4_sq) / 2,
        frozenset({p1_lbl, q1_lbl}): (m1_sq + m3_sq - t) / 2,
        frozenset({p2_lbl, q2_lbl}): (m2_sq + m4_sq - t) / 2,
        frozenset({p1_lbl, q2_lbl}): (m1_sq + m4_sq - u) / 2,
        frozenset({p2_lbl, q1_lbl}): (m2_sq + m3_sq - u) / 2,
    }


def _tu_cross_interference(t_term: DiagramTerm, u_term: DiagramTerm):
    """8-gamma trace for t×u cross-topology interference (massless fermion limit).

    Computes Tr[/a₁ γ^μ /b₁ γ^ν /a₂ γ_μ /b₂ γ_ν] where:
        a₁ = outgoing fermion at vertex A of t-diagram  (QGRAF label q1 for Møller)
        b₁ = incoming fermion at vertex A of t-diagram  (p1)
        a₂ = outgoing fermion at vertex B of t-diagram  (q2)
        b₂ = incoming fermion at vertex B of t-diagram  (p2)

    The result is a scalar in terms of momentum dot products which feeds into
    _tensor_expr_to_scalar for Mandelstam substitution.
    """
    i0, i1, i2, i3 = tensor_indices("i0 i1 i2 i3", LorentzIndex)
    # Use distinct index names to avoid collision with other concurrent traces.
    xmu, xnu = tensor_indices("xmu xnu", LorentzIndex)

    a1 = _momentum_head(t_term.incoming_pair.antifermion.momentum)
    b1 = _momentum_head(t_term.incoming_pair.fermion.momentum)
    a2 = _momentum_head(t_term.outgoing_pair.antifermion.momentum)
    b2 = _momentum_head(t_term.outgoing_pair.fermion.momentum)

    slash = lambda head, idx: head(idx) * G(-idx)

    return gamma_trace(
        slash(a1, i0) * G(xmu) * slash(b1, i1) * G(xnu)
        * slash(a2, i2) * G(-xmu) * slash(b2, i3) * G(-xnu)
    )


def _compton_term(diagram: Diagram, internal: Edge, mediator) -> DiagramTerm | None:
    """Build a DiagramTerm for Compton-type diagrams (external photon + fermion propagator).

    Handles processes like e⁻γ → e⁻γ where the internal mediator is a fermion and each
    vertex has one fermion leg and one photon leg.

    The amplitude at each vertex is ū γ^μ u with the photon providing the index μ; the
    fermion propagator numerator (p̸ + m) is absorbed into the trace using the propagator
    momentum derived from 4-momentum conservation.

    For the massless fermion limit the polarisation sum −g_μν on both photon legs gives a
    trace of the form Tr[/q₁ γ^ν /p_int γ^μ /p₁ γ_μ /p_int γ_ν] / denom² which reduces
    to a standard 4-gamma trace after γ-matrix contraction identities.
    """
    # Identify source vertex (1 incoming fermion + 1 incoming photon)
    # and sink vertex (1 outgoing fermion + 1 outgoing photon).
    # QGRAF's propagator direction (start→end) may not align with fermion flow,
    # so try both orderings.

    def vertex_legs(vid):
        in_f = [e for e in diagram.external_edges if e.end_vertex == vid and e.start_vertex < 0
                and TheoryRegistry.get_particle(diagram.theory, e.particle).particle_type
                in {ParticleType.FERMION, ParticleType.ANTIFERMION}]
        in_b = [e for e in diagram.external_edges if e.end_vertex == vid and e.start_vertex < 0
                and TheoryRegistry.get_particle(diagram.theory, e.particle).particle_type
                == ParticleType.BOSON]
        out_f = [e for e in diagram.external_edges if e.start_vertex == vid and e.end_vertex < 0
                 and TheoryRegistry.get_particle(diagram.theory, e.particle).particle_type
                 in {ParticleType.FERMION, ParticleType.ANTIFERMION}]
        out_b = [e for e in diagram.external_edges if e.start_vertex == vid and e.end_vertex < 0
                 and TheoryRegistry.get_particle(diagram.theory, e.particle).particle_type
                 == ParticleType.BOSON]
        return in_f, in_b, out_f, out_b

    v_a, v_b = internal.start_vertex, internal.end_vertex
    legs_a = vertex_legs(v_a)
    legs_b = vertex_legs(v_b)

    # Determine which vertex is the source (has 1 in-fermion + 1 in-photon)
    # and which is the sink (has 1 out-fermion + 1 out-photon).
    in_f_src, in_b_src, out_f_snk, out_b_snk = None, None, None, None
    if len(legs_a[0]) == 1 and len(legs_a[1]) == 1:
        in_f_src, in_b_src = legs_a[0], legs_a[1]
        out_f_snk, out_b_snk = legs_b[2], legs_b[3]
    elif len(legs_b[0]) == 1 and len(legs_b[1]) == 1:
        in_f_src, in_b_src = legs_b[0], legs_b[1]
        out_f_snk, out_b_snk = legs_a[2], legs_a[3]

    # Also detect "u-type" Compton: one vertex has (1 in-fermion + 1 out-photon)
    # and the other has (1 in-photon + 1 out-fermion).
    compton_type = None
    if in_f_src is None or out_f_snk is None:
        # Try u-type: vertex with in_f + out_b as "source"
        for va_legs, vb_legs in [(legs_a, legs_b), (legs_b, legs_a)]:
            if (len(va_legs[0]) == 1 and len(va_legs[3]) == 1 and
                    len(vb_legs[1]) == 1 and len(vb_legs[2]) == 1):
                in_f_src = va_legs[0]
                exchange_photon_src = va_legs[3]   # outgoing photon at source vertex
                in_b_src_u = vb_legs[1]            # incoming photon at sink vertex
                out_f_snk = vb_legs[2]
                compton_type = "u-type"
                break
        if compton_type is None:
            return None
    else:
        if not (len(out_f_snk) == 1 and len(out_b_snk) == 1):
            return None
        compton_type = "s-type"

    in_fermion = in_f_src[0]    # e-(p1) incoming

    if compton_type == "s-type":
        in_photon = in_b_src[0]      # γ(p2) absorbed at source vertex
        out_photon = out_b_snk[0]    # γ(q2) emitted at sink vertex
        trace_photon = in_photon     # photon appearing in propagator: p_int = p1+p2
        denom = _channel_invariant("s-channel") - _mass_squared(mediator.mass)
    else:
        # u-type: e-(p1) and γ(q2-out) at source vertex; γ(p2-in) and e-(q1) at sink
        trace_photon_edge = exchange_photon_src[0]  # γ(q2) outgoing at source
        in_photon = in_b_src_u[0]                   # γ(p2) incoming at sink
        out_photon = trace_photon_edge
        trace_photon = trace_photon_edge   # photon in propagator: p_int = p1-q2
        denom = _channel_invariant("u-channel") - _mass_squared(mediator.mass)

    out_fermion = out_f_snk[0]  # e-(q1) outgoing

    m = _mass_symbol(diagram.theory, in_fermion.particle)
    coupling = _coupling_symbol(diagram.theory, "gamma", [in_fermion, out_fermion])

    sym = (Integer(1) if diagram.symmetry_factor is None
           else Rational(str(diagram.symmetry_factor)))

    # Store momenta for _compton_msq:
    #   incoming_pair.fermion  = in_fermion  (e-(p1))
    #   incoming_pair.antifermion = out_fermion (e-(q1))
    #   outgoing_pair.particle = trace_photon   (the photon in the propagator numerator)
    in_pair = FermionPair(
        fermion=in_fermion,
        antifermion=out_fermion,
        mass=m,
    )
    out_pair = ScalarPair(particle=trace_photon, antiparticle=out_photon, mass=Integer(0))

    notes: list[str] = []
    notes.append(
        "Compton-type diagram: polarisation sum −g_μν applied; "
        "massless fermion limit used for internal propagator numerator."
    )

    return DiagramTerm(
        diagram_id=diagram.id,
        theory=diagram.theory,
        mediator_name=mediator.name,
        mediator_kind="compton",
        incoming_pair=in_pair,
        outgoing_pair=out_pair,
        outgoing_kind="compton",
        coupling_in=coupling,
        coupling_out=coupling,
        denominator=denom,
        symmetry_factor=sym,
        notes=notes,
        topology=diagram.topology,
    )


def _compton_msq(t: DiagramTerm, kin: dict):
    """Compute |M|² for a single Compton diagram (massless fermion limit).

    Tr[/q₁ γ^ν (/p₁+/k₁) γ^μ /p₁ γ_μ (/p₁+/k₁) γ_ν] / denom²

    Reduces via γ^μ /A γ_μ = −2/A to:
      → 4 Tr[/q₁ /k₁ /p₁ /k₁]  (massless, p₁²=k₁²=0)
    which is a standard 4-gamma trace.
    """
    # Fermion momenta labels from the diagram term.
    p1 = _momentum_head(t.incoming_pair.fermion.momentum)   # incoming e-
    q1 = _momentum_head(t.incoming_pair.antifermion.momentum)  # outgoing e-
    k1 = _momentum_head(t.outgoing_pair.particle.momentum)     # incoming photon

    i0, i1, i2 = tensor_indices("i0 i1 i2", LorentzIndex)

    slash = lambda head, idx: head(idx) * G(-idx)

    # Tr[/q1 /k1 /p1 /k1] (massless after using γ^μ /A γ_μ = -2/A twice and p1²=k1²=0)
    raw = gamma_trace(slash(q1, i0) * slash(k1, i1) * slash(p1, i2) * slash(k1, i2))
    # Note: the last k1 reuses i2 - that will double-contract; use a fresh index.
    i3 = tensor_indices("i3", LorentzIndex)
    raw = gamma_trace(
        slash(q1, i0) * slash(k1, i1) * slash(p1, i2) * slash(k1, i3)
    )
    scalar = _tensor_expr_to_scalar(raw.contract_metric(LorentzIndex.metric).canon_bp(), kin)
    coupling_sq = t.coupling_in ** 2
    return Rational(1, 4) * coupling_sq**2 * Integer(4) * scalar / t.denominator**2


def _interference(left: DiagramTerm, right: DiagramTerm):
    if left.mediator_kind == "vector" and right.mediator_kind == "vector":
        mu, nu = tensor_indices("mu nu", LorentzIndex)
        incoming = _fermion_interference(left.incoming_pair, right.incoming_pair, "vector", "vector", mu, nu)
        outgoing = _outgoing_interference(
            left.outgoing_pair,
            right.outgoing_pair,
            left.outgoing_kind,
            right.outgoing_kind,
            -mu,
            -nu,
        )
        if incoming is None or outgoing is None:
            return Integer(0)
        return (incoming * outgoing).contract_metric(LorentzIndex.metric).canon_bp()

    if left.mediator_kind == "scalar" and right.mediator_kind == "scalar":
        incoming = _fermion_interference(left.incoming_pair, right.incoming_pair, "scalar", "scalar")
        outgoing = _outgoing_interference(left.outgoing_pair, right.outgoing_pair, left.outgoing_kind, right.outgoing_kind)
        if incoming is None or outgoing is None:
            return Integer(0)
        return incoming * outgoing

    # scalar-vector interference is only supported when both external sides are fermion pairs
    if {left.mediator_kind, right.mediator_kind} == {"scalar", "vector"} and {
        left.outgoing_kind,
        right.outgoing_kind,
    } == {"fermion_scalar", "fermion_vector"}:
        left_in = "vector" if left.mediator_kind == "vector" else "scalar"
        right_in = "vector" if right.mediator_kind == "vector" else "scalar"
        mu = tensor_indices("mu", LorentzIndex)
        incoming = _fermion_interference(left.incoming_pair, right.incoming_pair, left_in, right_in, mu)
        outgoing = _fermion_interference(
            left.outgoing_pair,
            right.outgoing_pair,
            "vector" if left.outgoing_kind == "fermion_vector" else "scalar",
            "vector" if right.outgoing_kind == "fermion_vector" else "scalar",
            -mu,
        )
        if incoming is None or outgoing is None:
            return Integer(0)
        return (incoming * outgoing).contract_metric(LorentzIndex.metric).canon_bp()

    return Integer(0)


def _fermion_interference(left: FermionPair, right: FermionPair, left_kind: str, right_kind: str, *indices):
    if left_kind == "vector" and right_kind == "vector":
        if len(indices) != 2:
            mu, nu = tensor_indices("mu nu", LorentzIndex)
        else:
            mu, nu = indices
        return _fermion_vector_vector(left, right, mu, nu)
    if left_kind == "scalar" and right_kind == "scalar":
        return _fermion_scalar_scalar(left, right)
    if left_kind == "vector" and right_kind == "scalar":
        mu = indices[0] if indices else tensor_indices("mu", LorentzIndex)
        return _fermion_vector_scalar(left, right, mu)
    if left_kind == "scalar" and right_kind == "vector":
        mu = indices[0] if indices else tensor_indices("mu", LorentzIndex)
        return _fermion_scalar_vector(left, right, mu)
    return None


def _outgoing_interference(left, right, left_kind: str, right_kind: str, *indices):
    if left_kind.startswith("fermion") and right_kind.startswith("fermion"):
        return _fermion_interference(
            left,
            right,
            "vector" if left_kind == "fermion_vector" else "scalar",
            "vector" if right_kind == "fermion_vector" else "scalar",
            *indices,
        )

    if left_kind == "scalar_vector" and right_kind == "scalar_vector":
        if len(indices) != 2:
            mu, nu = tensor_indices("mu nu", LorentzIndex)
        else:
            mu, nu = indices
        left_current = _scalar_vector_current(left, mu)
        right_current = _scalar_vector_current(right, nu)
        return left_current * right_current

    return None


def _fermion_vector_vector(left: FermionPair, right: FermionPair, mu, nu):
    i0, i1 = tensor_indices("i0:2", LorentzIndex)
    lf = _momentum_head(left.fermion.momentum)
    la = _momentum_head(left.antifermion.momentum)
    metric = LorentzIndex.metric
    slash = lambda head, idx: head(idx) * G(-idx)

    massless_trace = gamma_trace(
        slash(lf, i0) * G(mu) * slash(la, i1) * G(nu)
    )
    mass_term = -4 * left.mass * right.mass * metric(mu, nu)
    return massless_trace + mass_term


def _fermion_scalar_scalar(left: FermionPair, right: FermionPair):
    lf = _momentum_head(left.fermion.momentum)
    la = _momentum_head(left.antifermion.momentum)
    return 4 * (_dot(lf, la) - left.mass * right.mass)


def _fermion_vector_scalar(left: FermionPair, right: FermionPair, mu):
    lf = _momentum_head(left.fermion.momentum)
    la = _momentum_head(left.antifermion.momentum)
    return 4 * (left.mass * la(mu) - right.mass * lf(mu))


def _fermion_scalar_vector(left: FermionPair, right: FermionPair, mu):
    return _fermion_vector_scalar(right, left, mu)


def _scalar_vector_current(pair: ScalarPair, idx):
    particle = _momentum_head(pair.particle.momentum)
    antiparticle = _momentum_head(pair.antiparticle.momentum)
    return particle(idx) - antiparticle(idx)


def _momentum_head(label: str):
    head = tensor_heads(label, [LorentzIndex])
    return head[0] if isinstance(head, tuple) else head


def _dot(head_a, head_b):
    idx = tensor_indices("d", LorentzIndex)
    return head_a(idx) * head_b(-idx)


def _kinematics_context(incoming_pair: FermionPair, outgoing_pair: FermionPair | ScalarPair):
    return _dot_map_from_pairs(incoming_pair, outgoing_pair)


def _dot_map_from_pairs(incoming_pair: FermionPair, outgoing_pair: FermionPair | ScalarPair):
    s = Symbol("s")
    t = Symbol("t")
    u = Symbol("u")

    p1 = incoming_pair.antifermion.momentum
    p2 = incoming_pair.fermion.momentum
    m1_sq = incoming_pair.mass**2
    m2_sq = incoming_pair.mass**2

    if isinstance(outgoing_pair, FermionPair):
        q1 = outgoing_pair.antifermion.momentum
        q2 = outgoing_pair.fermion.momentum
        m3_sq = outgoing_pair.mass**2
        m4_sq = outgoing_pair.mass**2
    else:
        q1 = outgoing_pair.antiparticle.momentum
        q2 = outgoing_pair.particle.momentum
        m3_sq = outgoing_pair.mass**2
        m4_sq = outgoing_pair.mass**2

    return {
        (p1, p1): m1_sq,
        (p2, p2): m2_sq,
        (q1, q1): m3_sq,
        (q2, q2): m4_sq,
        frozenset((p1, p2)): (s - m1_sq - m2_sq) / 2,
        frozenset((q1, q2)): (s - m3_sq - m4_sq) / 2,
        frozenset((p1, q1)): (m1_sq + m3_sq - t) / 2,
        frozenset((p2, q2)): (m2_sq + m4_sq - t) / 2,
        frozenset((p1, q2)): (m1_sq + m4_sq - u) / 2,
        frozenset((p2, q1)): (m2_sq + m3_sq - u) / 2,
    }
def _tensor_expr_to_scalar(expr, dot_map):
    if hasattr(expr, "contract_metric"):
        expr = expr.contract_metric(LorentzIndex.metric).canon_bp()

    if isinstance(expr, TensAdd):
        return sum(_tensor_expr_to_scalar(arg, dot_map) for arg in expr.args)

    if isinstance(expr, TensMul):
        scalar = expr.coeff
        for left_idx, right_idx in expr.dum:
            left = expr.components[left_idx].name
            right = expr.components[right_idx].name
            key = (left, left) if left == right else frozenset((left, right))
            scalar *= dot_map[key]
        return scalar

    return expr


def _qcd_color_factor(terms: list[DiagramTerm]):
    if not terms or any(term.theory != "QCD" for term in terms):
        return Integer(1)
    topologies = {t.topology for t in terms}
    if topologies == {"s-channel"} and all(
        t.mediator_name == "g" and t.outgoing_kind == "fermion_vector" for t in terms
    ):
        # qqbar → ff̄ via s-channel gluon: SU(3) color average = 4/(Nc²-1)² × Nc = 2/9
        return Rational(2, 9)
    if topologies <= {"t-channel", "u-channel"} and all(t.mediator_name == "g" for t in terms):
        # qq → qq via t/u-channel gluon: SU(3) color average = 4/9
        return Rational(4, 9)
    return Symbol("C_color")
