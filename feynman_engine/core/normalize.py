"""
Deduplicate Feynman diagrams via graph isomorphism.

Two diagrams are equivalent if their underlying labeled graphs are isomorphic,
where edges are labeled by particle type and vertices by their degree/particle set.

Uses NetworkX's VF2 isomorphism algorithm with node/edge attribute matching.
"""
from __future__ import annotations

import hashlib
import json

import networkx as nx

from feynman_engine.core.models import Diagram


def _diagram_to_graph(diagram: Diagram) -> nx.Graph:
    """Convert a Diagram to a NetworkX graph with edge and node attributes."""
    G = nx.MultiGraph()

    for v in diagram.vertices:
        G.add_node(v.id, particles=sorted(v.particles))

    for e in diagram.edges:
        G.add_edge(
            e.start_vertex,
            e.end_vertex,
            key=e.id,
            particle=e.particle,
            is_external=e.is_external,
        )

    return G


def _node_match(n1: dict, n2: dict) -> bool:
    return sorted(n1.get("particles", [])) == sorted(n2.get("particles", []))


def _edge_match(e1: dict, e2: dict) -> bool:
    # e1/e2 are dicts of {edge_key: edge_attrs} for MultiGraph
    attrs1 = sorted((v["particle"], v["is_external"]) for v in e1.values())
    attrs2 = sorted((v["particle"], v["is_external"]) for v in e2.values())
    return attrs1 == attrs2


def _canonical_hash(diagram: Diagram) -> str:
    """
    Compute a canonical hash for a diagram based on its sorted edge/vertex structure.

    This is a fast pre-filter before the full isomorphism check.
    Two non-isomorphic diagrams will never have the same hash (with high probability).
    Two isomorphic diagrams will always have the same hash.
    """
    edge_signature = sorted(
        (e.particle, e.is_external) for e in diagram.edges
    )
    vertex_signature = sorted(
        tuple(sorted(v.particles)) for v in diagram.vertices
    )
    payload = json.dumps({
        "theory": diagram.theory,
        "process": diagram.process,
        "loops": diagram.loop_order,
        "edges": edge_signature,
        "vertices": vertex_signature,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def deduplicate(diagrams: list[Diagram]) -> list[Diagram]:
    """
    Remove topologically duplicate diagrams.

    Strategy:
    1. Group diagrams by their canonical hash (fast bucket filter).
    2. Within each bucket, run full graph isomorphism to confirm duplicates.

    Returns a deduplicated list, preserving the first occurrence of each topology.
    """
    # Assign canonical hashes
    for d in diagrams:
        d.canonical_hash = _canonical_hash(d)

    # Group by hash
    buckets: dict[str, list[Diagram]] = {}
    for d in diagrams:
        buckets.setdefault(d.canonical_hash, []).append(d)

    unique: list[Diagram] = []

    for bucket in buckets.values():
        if len(bucket) == 1:
            unique.append(bucket[0])
            continue

        # Full isomorphism check within this hash bucket
        accepted: list[tuple[Diagram, nx.Graph]] = []
        for candidate in bucket:
            G_candidate = _diagram_to_graph(candidate)
            is_dup = False
            for _, G_accepted in accepted:
                gm = nx.algorithms.isomorphism.GraphMatcher(
                    G_candidate, G_accepted,
                    node_match=_node_match,
                    edge_match=_edge_match,
                )
                if gm.is_isomorphic():
                    is_dup = True
                    break
            if not is_dup:
                accepted.append((candidate, G_candidate))

        unique.extend(d for d, _ in accepted)

    # Restore original ordering (stable)
    id_order = {d.id: i for i, d in enumerate(diagrams)}
    unique.sort(key=lambda d: id_order[d.id])

    return unique
