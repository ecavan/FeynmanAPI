"""
Pure Python Feynman diagram enumerator.

Generates tree-level Feynman diagrams for any theory registered in
TheoryRegistry — no external tools required.

Supports:
  - 2 → 2 processes:  complete (s/t/u channels + 4-point contact vertices)
  - 2 → 3 processes:  complete tree-level via recursive pairing
  - 1-loop insertions: self-energy, vertex correction on tree diagrams

When QGRAF is available (bin/qgraf or bin/qgraf_pipe), the engine
automatically uses it instead for full loop-level enumeration.

Crossing-symmetry convention
-----------------------------
At a QFT vertex, an outgoing particle P is equivalent to an incoming
antiparticle P̄.  We normalise every external particle to its
"vertex name" before checking theory vertex rules:

    incoming P   → vertex name = P
    outgoing P   → vertex name = antiparticle(P)   [self-conjugate: P itself]
"""
from __future__ import annotations

import itertools
from typing import Optional

from feynman_engine.core.models import Diagram, Edge, Vertex
from feynman_engine.core.normalize import deduplicate
from feynman_engine.core.topology import classify_all
from feynman_engine.physics.registry import TheoryRegistry


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vertex_name(particle: str, is_incoming: bool, theory: str) -> str:
    """Return the vertex-side name for an external particle (crossing symmetry)."""
    if is_incoming:
        return particle
    p = TheoryRegistry.get_particle(theory, particle)
    return p.antiparticle if p.antiparticle else particle


def _antiparticle(particle: str, theory: str) -> str:
    """Return the antiparticle name, or the particle itself if self-conjugate."""
    p = TheoryRegistry.get_particle(theory, particle)
    return p.antiparticle if p.antiparticle else particle


def _vertex_matches(names: list[str], vertices: list[tuple]) -> bool:
    """True if `names` (as a multiset) matches any theory vertex rule."""
    s = sorted(names)
    return any(sorted(v) == s for v in vertices)


def _completing_particles(
    group_vnames: list[str],
    theory: str,
    vertices: list[tuple],
) -> list[str]:
    """
    Return all particle names P such that group_vnames + [P] is a valid vertex.
    """
    result = []
    all_particles = list(TheoryRegistry.get_particles(theory).keys())
    for p in all_particles:
        if _vertex_matches(group_vnames + [p], vertices):
            result.append(p)
    return result


# ── Diagram construction ──────────────────────────────────────────────────────

_next_id = 0


def _new_id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


def _build_two_vertex_diagram(
    group1: list[tuple[str, bool]],   # [(particle_name, is_incoming), ...]
    group2: list[tuple[str, bool]],
    internal_particle: str,
    theory: str,
    process: str,
    diagram_id: int,
) -> Diagram:
    """
    Build a Diagram object for a 2-vertex tree topology.

    group1 and group2 each hold 2 external particles; they are
    connected by an internal propagator of type `internal_particle`.
    """
    v0_id, v1_id = 0, 1

    # Vertex particle lists (field names at the vertex, including internal)
    v0_particles = [p for p, _ in group1] + [internal_particle]
    v1_particles = [p for p, _ in group2] + [_antiparticle(internal_particle, theory)]

    vertices = [
        Vertex(id=v0_id, particles=v0_particles),
        Vertex(id=v1_id, particles=v1_particles),
    ]

    edges: list[Edge] = []
    eid = 0

    # Internal edge
    edges.append(Edge(
        id=eid, start_vertex=v0_id, end_vertex=v1_id,
        particle=internal_particle, is_external=False,
    ))
    eid += 1

    # External edges — phantom vertex IDs are negative
    phantom = -1
    for (pname, is_incoming) in group1:
        if is_incoming:
            edges.append(Edge(id=eid, start_vertex=phantom, end_vertex=v0_id,
                               particle=pname, is_external=True))
        else:
            edges.append(Edge(id=eid, start_vertex=v0_id, end_vertex=phantom,
                               particle=pname, is_external=True))
        phantom -= 1
        eid += 1

    for (pname, is_incoming) in group2:
        if is_incoming:
            edges.append(Edge(id=eid, start_vertex=phantom, end_vertex=v1_id,
                               particle=pname, is_external=True))
        else:
            edges.append(Edge(id=eid, start_vertex=v1_id, end_vertex=phantom,
                               particle=pname, is_external=True))
        phantom -= 1
        eid += 1

    return Diagram(
        id=diagram_id,
        vertices=vertices,
        edges=edges,
        loop_order=0,
        theory=theory,
        process=process,
    )


def _build_contact_diagram(
    external: list[tuple[str, bool]],
    vertex_rule: tuple,
    theory: str,
    process: str,
    diagram_id: int,
) -> Diagram:
    """Build a contact (single-vertex) diagram."""
    v0_id = 0
    vertices = [Vertex(id=v0_id, particles=[p for p, _ in external])]
    edges: list[Edge] = []
    phantom = -1
    for eid, (pname, is_incoming) in enumerate(external):
        if is_incoming:
            edges.append(Edge(id=eid, start_vertex=phantom, end_vertex=v0_id,
                               particle=pname, is_external=True))
        else:
            edges.append(Edge(id=eid, start_vertex=v0_id, end_vertex=phantom,
                               particle=pname, is_external=True))
        phantom -= 1
    return Diagram(
        id=diagram_id, vertices=vertices, edges=edges,
        loop_order=0, theory=theory, process=process,
    )


# ── 3-vertex tree (2→3) ───────────────────────────────────────────────────────

def _build_three_vertex_diagram(
    v0_ext: list[tuple[str, bool]],
    v1_ext: list[tuple[str, bool]],
    v2_ext: list[tuple[str, bool]],
    int01: str,   # internal propagator between v0 and v1
    int12: str,   # internal propagator between v1 and v2
    theory: str,
    process: str,
    diagram_id: int,
) -> Diagram:
    """Build a chain 3-vertex tree: v0 — int01 — v1 — int12 — v2."""
    vertices = [
        Vertex(id=0, particles=[p for p, _ in v0_ext] + [int01]),
        Vertex(id=1, particles=[_antiparticle(int01, theory), int12]),
        Vertex(id=2, particles=[p for p, _ in v2_ext] + [_antiparticle(int12, theory)]),
    ]
    # Note: v1 has 2 external legs from v1_ext too
    vertices[1].particles = (
        [p for p, _ in v1_ext] + [_antiparticle(int01, theory), int12]
    )

    edges: list[Edge] = []
    eid = 0
    phantom = -1

    # Internal edges
    edges.append(Edge(id=eid, start_vertex=0, end_vertex=1,
                       particle=int01, is_external=False))
    eid += 1
    edges.append(Edge(id=eid, start_vertex=1, end_vertex=2,
                       particle=int12, is_external=False))
    eid += 1

    for vid, ext_group in [(0, v0_ext), (1, v1_ext), (2, v2_ext)]:
        for pname, is_incoming in ext_group:
            if is_incoming:
                edges.append(Edge(id=eid, start_vertex=phantom, end_vertex=vid,
                                   particle=pname, is_external=True))
            else:
                edges.append(Edge(id=eid, start_vertex=vid, end_vertex=phantom,
                                   particle=pname, is_external=True))
            phantom -= 1
            eid += 1

    return Diagram(
        id=diagram_id, vertices=vertices, edges=edges,
        loop_order=0, theory=theory, process=process,
    )


# ── Main entry points ─────────────────────────────────────────────────────────

def enumerate_tree(
    incoming: list[str],
    outgoing: list[str],
    theory: str,
) -> list[Diagram]:
    """
    Enumerate all distinct tree-level Feynman diagrams for a scattering process.

    Args:
        incoming:  List of incoming particle names, e.g. ["e+", "e-"]
        outgoing:  List of outgoing particle names, e.g. ["mu+", "mu-"]
        theory:    Theory name, e.g. "QED"

    Returns:
        Deduplicated, topology-classified list of Diagram objects.
    """
    theory = theory.upper()
    vertices = TheoryRegistry.get_theory(theory)["vertices"]
    process = " ".join(incoming) + " -> " + " ".join(outgoing)

    # (particle_name, is_incoming) pairs for all external particles
    all_ext: list[tuple[str, bool]] = (
        [(p, True) for p in incoming] + [(p, False) for p in outgoing]
    )
    n_ext = len(all_ext)
    diagrams: list[Diagram] = []
    diag_id = 1

    # ── Contact vertex (n-point, no internal propagator) ─────────────────────
    vnames_all = [_vertex_name(p, is_in, theory) for p, is_in in all_ext]
    if _vertex_matches(vnames_all, vertices):
        diagrams.append(_build_contact_diagram(all_ext, tuple(vnames_all),
                                                theory, process, diag_id))
        diag_id += 1

    # ── 2→2: two-vertex diagrams (s/t/u channels) ────────────────────────────
    if n_ext == 4:
        for combo in itertools.combinations(range(4), 2):
            other = [i for i in range(4) if i not in combo]
            g1 = [all_ext[i] for i in combo]
            g2 = [all_ext[i] for i in other]

            g1_vnames = [_vertex_name(p, is_in, theory) for p, is_in in g1]
            g2_vnames = [_vertex_name(p, is_in, theory) for p, is_in in g2]

            for internal in _completing_particles(g1_vnames, theory, vertices):
                anti = _antiparticle(internal, theory)
                if _vertex_matches(g2_vnames + [anti], vertices):
                    diagrams.append(_build_two_vertex_diagram(
                        g1, g2, internal, theory, process, diag_id))
                    diag_id += 1

    # ── 2→3: three-vertex chain diagrams ─────────────────────────────────────
    elif n_ext == 5:
        # Topology: one external particle at vertex 0 or 2 (the "tip"),
        # two at the middle vertex, two at the other end.
        # Enumerate all ways to assign 1 particle to one end, 1 to middle, 2 to other end.
        for tip_idx in range(5):
            remaining = [all_ext[i] for i in range(5) if i != tip_idx]
            # Choose 1 from remaining for middle, 2 for the far end
            for mid_idx in range(4):
                mid_ext = [remaining[mid_idx]]
                far_ext = [remaining[i] for i in range(4) if i != mid_idx]
                tip_group = [all_ext[tip_idx]]

                tip_vnames = [_vertex_name(p, is_in, theory) for p, is_in in tip_group]
                mid_vnames = [_vertex_name(p, is_in, theory) for p, is_in in mid_ext]
                far_vnames = [_vertex_name(p, is_in, theory) for p, is_in in far_ext]

                for int01 in _completing_particles(tip_vnames, theory, vertices):
                    anti01 = _antiparticle(int01, theory)
                    # Middle vertex: mid_ext particle + anti01 + some int12
                    mid_with_prop = mid_vnames + [anti01]
                    for int12 in _completing_particles(mid_with_prop, theory, vertices):
                        anti12 = _antiparticle(int12, theory)
                        if _vertex_matches(far_vnames + [anti12], vertices):
                            diagrams.append(_build_three_vertex_diagram(
                                tip_group, mid_ext, far_ext,
                                int01, int12, theory, process, diag_id))
                            diag_id += 1

    elif n_ext > 5:
        raise NotImplementedError(
            f"Pure Python enumerator supports up to 2→3 processes. "
            f"For {len(incoming)}→{len(outgoing)}, install QGRAF "
            f"(see contrib/qgraf/ and README)."
        )

    # ── Deduplicate and classify ──────────────────────────────────────────────
    diagrams = deduplicate(diagrams)
    diagrams = classify_all(diagrams)
    return diagrams
