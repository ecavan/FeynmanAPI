"""
Parse QGRAF output (feynman.sty format) into Diagram objects.

Output format produced by contrib/qgraf/styles/feynman.sty:

    D <diagram_index> <sign> <symmetry_factor>
    I <field> <field_index> <momentum>    (one per incoming external particle)
    O <field> <field_index> <momentum>    (one per outgoing external particle)
    P <field> <field_index> <dual-field> <dual-field_index>  (one per internal propagator)
    V <field>(<field_index>) <field>(<field_index>) ...      (one per vertex)
    END

Graph reconstruction algorithm
--------------------------------
QGRAF assigns a unique integer "field index" to every field instance in a
diagram.  External particles get negative indices; internal propagator ends
get positive indices.

1.  Build index→vertex_id from V lines (each V line is one vertex, numbered
    in order 0, 1, 2, …; all field indices appearing in that line map to it).
2.  Incoming external edge:  phantom_src → vertex_of(field_index)
3.  Outgoing external edge:  vertex_of(field_index) → phantom_dst
4.  Internal edge:            vertex_of(field_idx_A) → vertex_of(dual_idx_B)
    (particle name taken from P line, not the crossing-symmetry dual shown in V)
"""
from __future__ import annotations

import re
from typing import Optional

from feynman_engine.core.models import Diagram, Edge, Vertex
from feynman_engine.physics.registry import TheoryRegistry

# ── Regex patterns ─────────────────────────────────────────────────────────────

_RE_D    = re.compile(r"^D\s+(\d+)\s+([+-])\s+(\S+)", re.MULTILINE)
_RE_I    = re.compile(r"^I\s+(\S+)\s+(-?\d+)\s+(\S+)", re.MULTILINE)
_RE_O    = re.compile(r"^O\s+(\S+)\s+(-?\d+)\s+(\S+)", re.MULTILINE)
_RE_P    = re.compile(r"^P\s+(\S+)\s+(-?\d+)\s+(\S+)\s+(-?\d+)", re.MULTILINE)
_RE_V    = re.compile(r"^V((?:\s+\S+\(-?\d+\))+)", re.MULTILINE)
_RE_VLEG = re.compile(r"(\S+)\((-?\d+)\)")


# ── Block splitting ────────────────────────────────────────────────────────────

def _split_blocks(raw: str) -> list[str]:
    """Split QGRAF feynman.sty output into one block per diagram (D … END)."""
    blocks: list[str] = []
    current: list[str] = []
    in_block = False

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("D ") and not in_block:
            in_block = True
            current = [line]
        elif stripped == "END" and in_block:
            current.append(line)
            blocks.append("\n".join(current))
            current = []
            in_block = False
        elif in_block:
            current.append(line)

    return blocks


# ── Single-block parser ────────────────────────────────────────────────────────

def _parse_block(block: str, theory: str, process: str) -> Diagram:
    """Parse one D…END block into a Diagram object."""

    # ── Header ────────────────────────────────────────────────────────────────
    m = _RE_D.search(block)
    if not m:
        raise ValueError("Missing D header in block")
    diag_id   = int(m.group(1))
    sign      = 1 if m.group(2) == "+" else -1
    try:
        symm_raw = float(m.group(3))
    except ValueError:
        symm_raw = 1.0

    # ── External particles ────────────────────────────────────────────────────
    incoming: list[tuple[str, int, str]] = []   # (field, idx, momentum)
    for m in _RE_I.finditer(block):
        fname = TheoryRegistry.from_qgraf_name(theory, m.group(1))
        incoming.append((fname, int(m.group(2)), m.group(3)))

    outgoing: list[tuple[str, int, str]] = []
    for m in _RE_O.finditer(block):
        fname = TheoryRegistry.from_qgraf_name(theory, m.group(1))
        outgoing.append((fname, int(m.group(2)), m.group(3)))

    # ── Internal propagators ─────────────────────────────────────────────────
    props: list[tuple[str, int, int]] = []   # (field, idx_a, idx_b)
    for m in _RE_P.finditer(block):
        fname = TheoryRegistry.from_qgraf_name(theory, m.group(1))
        props.append((fname, int(m.group(2)), int(m.group(4))))

    # ── Vertices ──────────────────────────────────────────────────────────────
    # Each V line is one vertex (in encounter order → vertex id 0, 1, 2, …)
    vert_field_lists: list[list[tuple[str, int]]] = []
    for m in _RE_V.finditer(block):
        legs = [(TheoryRegistry.from_qgraf_name(theory, f), int(i))
                for f, i in _RE_VLEG.findall(m.group(1))]
        vert_field_lists.append(legs)

    # ── Index → vertex_id mapping ─────────────────────────────────────────────
    idx_to_vid: dict[int, int] = {}
    for vid, legs in enumerate(vert_field_lists):
        for _, idx in legs:
            idx_to_vid[idx] = vid

    # ── Vertex objects ────────────────────────────────────────────────────────
    # Use the field names from the I/O/P context to correctly name vertex particles.
    # QGRAF shows dual (crossed) fields in V lines for outgoing particles; we
    # rebuild the particle list from the edge context instead.
    vertex_particles: dict[int, list[str]] = {vid: [] for vid in range(len(vert_field_lists))}

    ext_by_idx: dict[int, tuple[str, bool]] = {}  # idx → (particle, is_incoming)
    for fname, idx, _ in incoming:
        ext_by_idx[idx] = (fname, True)
    for fname, idx, _ in outgoing:
        ext_by_idx[idx] = (fname, False)

    for fname, idx_a, idx_b in props:
        vid_a = idx_to_vid.get(idx_a)
        vid_b = idx_to_vid.get(idx_b)
        if vid_a is not None:
            vertex_particles[vid_a].append(fname)
        if vid_b is not None:
            vertex_particles[vid_b].append(fname)

    for idx, (fname, _) in ext_by_idx.items():
        vid = idx_to_vid.get(idx)
        if vid is not None:
            vertex_particles[vid].append(fname)

    vertices = [
        Vertex(id=vid, particles=vertex_particles.get(vid, []))
        for vid in range(len(vert_field_lists))
    ]

    # ── Edge objects ──────────────────────────────────────────────────────────
    edges: list[Edge] = []
    eid      = 0
    phantom  = -1

    for fname, idx, momentum in incoming:
        vid = idx_to_vid.get(idx)
        if vid is None:
            continue
        edges.append(Edge(
            id=eid, start_vertex=phantom, end_vertex=vid,
            particle=fname, is_external=True, momentum=momentum,
        ))
        phantom -= 1
        eid += 1

    for fname, idx, momentum in outgoing:
        vid = idx_to_vid.get(idx)
        if vid is None:
            continue
        edges.append(Edge(
            id=eid, start_vertex=vid, end_vertex=phantom,
            particle=fname, is_external=True, momentum=momentum,
        ))
        phantom -= 1
        eid += 1

    for fname, idx_a, idx_b in props:
        v_start = idx_to_vid.get(idx_a)
        v_end   = idx_to_vid.get(idx_b)
        if v_start is None or v_end is None:
            continue
        edges.append(Edge(
            id=eid, start_vertex=v_start, end_vertex=v_end,
            particle=fname, is_external=False,
        ))
        eid += 1

    # ── Loop order: L = I - V + 1 (connected diagram) ────────────────────────
    n_internal  = len(props)
    n_vertices  = len(vertices)
    loop_order  = max(0, n_internal - n_vertices + 1)

    return Diagram(
        id=diag_id,
        vertices=vertices,
        edges=edges,
        loop_order=loop_order,
        symmetry_factor=float(sign) * symm_raw,
        theory=theory,
        process=process,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_qgraf_output(raw: str, theory: str, process: str) -> list[Diagram]:
    """
    Parse full QGRAF feynman.sty output text into a list of Diagram objects.

    Args:
        raw:     Raw text content of QGRAF output file.
        theory:  Theory name (e.g. "QED") for particle name translation.
        process: Human-readable process string for metadata.

    Returns:
        List of Diagram objects, one per diagram in the QGRAF output.
    """
    blocks = _split_blocks(raw)
    if not blocks:
        return []

    diagrams: list[Diagram] = []
    for idx, block in enumerate(blocks):
        try:
            diagrams.append(_parse_block(block, theory, process))
        except Exception as exc:
            raise ValueError(
                f"Failed to parse diagram block {idx + 1}: {exc}\n\nBlock:\n{block}"
            ) from exc

    return diagrams
