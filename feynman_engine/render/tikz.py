"""
Generate TikZ-Feynman LaTeX code from Diagram objects.

Uses Jinja2 templating to produce compilable .tex files.
"""
from __future__ import annotations

from jinja2 import Environment, BaseLoader

from feynman_engine.core.models import Diagram, Edge, PropagatorStyle
from feynman_engine.physics.registry import TheoryRegistry


# ── TikZ-Feynman style mapping ────────────────────────────────────────────────

_STYLE_COLOR: dict[str, str] = {
    "fermion":      "draw=blue!70!black",
    "anti fermion": "draw=blue!70!black",
    "photon":       "draw=orange!80!black",
    "boson":        "draw=red!70!black",
    "gluon":        "draw=green!55!black",
    "scalar":       "draw=violet!80!black",
    "ghost":        "draw=gray!70",
    "plain":        "",
}


def _propagator_style(particle_name: str, theory: str) -> str:
    """Return 'style[, color]' string for a TikZ-Feynman edge."""
    try:
        p = TheoryRegistry.get_particle(theory, particle_name)
        base_style = p.propagator_style.value
    except ValueError:
        base_style = "plain"
    color = _STYLE_COLOR.get(base_style, "")
    return f"{base_style}, {color}" if color else base_style


# ── TikZ template ─────────────────────────────────────────────────────────────

_TIKZ_TEMPLATE = r"""
\documentclass[border=8pt]{standalone}
\usepackage{tikz-feynman}
\tikzfeynmanset{compat=1.1.0}
\begin{document}
\feynmandiagram [
  large, layered layout,
  every edge={line width=0.75pt},
] {
{%- for edge in edges %}
  {{ edge.start }} -- [{{ edge.style }}{% if edge.label %}, edge label=${{ edge.label }}${% endif %}] {{ edge.end }}{% if not loop.last %},{% endif %}

{%- endfor %}
};
\end{document}
"""

_env      = Environment(loader=BaseLoader())
_template = _env.from_string(_TIKZ_TEMPLATE)


def _vertex_name(vid: int) -> str:
    """Convert vertex integer ID to a valid TikZ node name (no hyphens)."""
    return f"vn{-vid}" if vid < 0 else f"v{vid}"


def diagram_to_tikz(diagram: Diagram) -> str:
    """
    Convert a Diagram to a compilable TikZ-Feynman LaTeX string.

    Returns the full .tex document content.
    """
    edge_specs = []
    for edge in diagram.edges:
        style = _propagator_style(edge.particle, diagram.theory)
        if edge.is_external:
            # One endpoint is a phantom vertex (negative ID).
            # Always put the real (non-phantom) vertex as start and the
            # external label node as end so the diagram graph stays connected.
            if edge.start_vertex < 0:  # incoming: phantom→real
                start_name = f"ext{edge.id}"
                end_name   = _vertex_name(edge.end_vertex)
            else:                       # outgoing: real→phantom
                start_name = _vertex_name(edge.start_vertex)
                end_name   = f"ext{edge.id}"
            edge_specs.append({
                "start": start_name,
                "end":   end_name,
                "style": style,
                "label": edge.momentum or edge.particle,
            })
        else:
            edge_specs.append({
                "start": _vertex_name(edge.start_vertex),
                "end":   _vertex_name(edge.end_vertex),
                "style": style,
                "label": edge.momentum or "",
            })

    return _template.render(edges=edge_specs, diagram=diagram)


def diagrams_to_tikz(diagrams: list[Diagram]) -> dict[int, str]:
    """Return {diagram_id: tikz_code} for all diagrams."""
    return {d.id: diagram_to_tikz(d) for d in diagrams}
