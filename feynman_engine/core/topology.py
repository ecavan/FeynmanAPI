"""
Classify Feynman diagram topology.

For tree-level diagrams:
  - s-channel: single internal propagator connecting the incoming and outgoing blobs
  - t-channel: momentum transfer between particles on the same side
  - u-channel: like t but with crossed legs
  - self-energy: single external leg on each side, one loop

For loop diagrams:
  - vertex correction, self-energy, box, triangle, etc.
"""
from __future__ import annotations

import networkx as nx

from feynman_engine.core.models import Diagram


def _build_internal_graph(diagram: Diagram) -> nx.Graph:
    """Build a graph of only the internal propagators."""
    G = nx.Graph()
    for v in diagram.vertices:
        G.add_node(v.id)
    for e in diagram.edges:
        if not e.is_external:
            G.add_edge(e.start_vertex, e.end_vertex, particle=e.particle)
    return G


def _external_vertex_groups(diagram: Diagram) -> tuple[set[int], set[int]]:
    """
    Return (incoming_vertices, outgoing_vertices) based on external edges.

    For a 2→2 process, "incoming" are vertices attached to the first two external legs,
    "outgoing" to the last two. This is a heuristic based on momentum labeling.
    """
    ext_edges = [e for e in diagram.edges if e.is_external]
    # Vertices at the ends of external edges
    ext_vertices = set()
    for e in ext_edges:
        ext_vertices.add(e.start_vertex)
        ext_vertices.add(e.end_vertex)
    return ext_vertices, ext_vertices  # simplified: treat all as the same group


def classify_topology(diagram: Diagram) -> str:
    """
    Assign a topology label to a diagram.

    Returns one of: "s-channel", "t-channel", "u-channel",
    "self-energy", "vertex correction", "triangle", "box", "bubble",
    "tadpole", or "unknown".
    """
    n_vertices = len(diagram.vertices)
    n_internal = len(diagram.internal_edges)
    n_external = len(diagram.external_edges)
    loops = diagram.loop_order

    # ── Tree-level (loops == 0) ────────────────────────────────────────────
    if loops == 0:
        if n_internal == 1:
            # Single internal propagator: s, t, or u channel
            internal_edge = diagram.internal_edges[0]
            v_start = internal_edge.start_vertex
            v_end = internal_edge.end_vertex

            # For each real vertex, count incoming vs outgoing external legs.
            # External edge conventions (from enumerator):
            #   incoming particle: start_vertex < 0 (phantom), end_vertex = real vertex
            #   outgoing particle: start_vertex = real vertex, end_vertex < 0 (phantom)
            def ext_in_out(vid: int) -> tuple[int, int]:
                n_in = sum(1 for e in diagram.external_edges if e.end_vertex == vid)
                n_out = sum(1 for e in diagram.external_edges if e.start_vertex == vid)
                return n_in, n_out

            in0, out0 = ext_in_out(v_start)
            in1, out1 = ext_in_out(v_end)

            # s-channel: one vertex is a pure annihilation (all incoming),
            # the other is pure creation (all outgoing)
            if (in0 > 0 and out0 == 0) or (in1 > 0 and out1 == 0):
                return "s-channel"

            # t-channel: each vertex has mixed in+out external legs (scattering)
            return "t-channel"

        if n_internal == 2:
            return "t-channel"

        return "tree-level"

    # ── One loop ──────────────────────────────────────────────────────────
    if loops == 1:
        if n_internal == 2:
            return "self-energy"
        if n_internal == 3:
            return "vertex correction"
        if n_internal == 4:
            return "box"
        if n_internal == 1:
            return "tadpole"
        return f"1-loop ({n_internal} props)"

    # ── Higher loops ──────────────────────────────────────────────────────
    return f"{loops}-loop"


def classify_all(diagrams: list[Diagram]) -> list[Diagram]:
    """Assign topology labels to all diagrams in-place. Returns the list."""
    for d in diagrams:
        d.topology = classify_topology(d)
    return diagrams
