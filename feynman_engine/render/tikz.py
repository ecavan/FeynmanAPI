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
    "fermion":       "draw=blue!70!black",
    "anti fermion":  "draw=blue!70!black",
    "photon":        "draw=orange!80!black",
    "boson":         "draw=red!70!black",
    "gluon":         "draw=green!55!black",
    "scalar":        "draw=violet!80!black",
    "ghost":         "draw=gray!70",
    "charged boson": "draw=red!70!black",
    "plain":         "",
}


# ── Particle name → LaTeX math string ─────────────────────────────────────────

_LATEX_LABEL: dict[str, str] = {
    # Leptons
    "e-":       r"e^-",
    "e+":       r"e^+",
    "mu-":      r"\mu^-",
    "mu+":      r"\mu^+",
    "tau-":     r"\tau^-",
    "tau+":     r"\tau^+",
    # Neutrinos
    "nu_e":     r"\nu_e",
    "nu_e~":    r"\bar{\nu}_e",
    "nu_mu":    r"\nu_\mu",
    "nu_mu~":   r"\bar{\nu}_\mu",
    "nu_tau":   r"\nu_\tau",
    "nu_tau~":  r"\bar{\nu}_\tau",
    # Gauge bosons
    "gamma":    r"\gamma",
    "g":        r"g",
    "Z":        r"Z",
    "W+":       r"W^+",
    "W-":       r"W^-",
    # Higgs
    "H":        r"H",
    # Quarks (Python registry names after from_qgraf_name)
    "u":        r"u",
    "u~":       r"\bar{u}",
    "d":        r"d",
    "d~":       r"\bar{d}",
    "s":        r"s",
    "s~":       r"\bar{s}",
    "c":        r"c",
    "c~":       r"\bar{c}",
    "b":        r"b",
    "b~":       r"\bar{b}",
    "t":        r"t",
    "t~":       r"\bar{t}",
    # Ghosts
    "gh":       r"c",
    "gh~":      r"\bar{c}",
    # BSM
    "chi":      r"\chi",
    "chi~":     r"\bar{\chi}",
    "Zp":       r"Z'",
    "phi":      r"\phi",
    "phi~":     r"\phi^\dagger",
}


def _particle_latex(name: str) -> str:
    """Return a LaTeX math-mode string for the particle name (no delimiters)."""
    if name in _LATEX_LABEL:
        return _LATEX_LABEL[name]
    # Minimal fallback for unknown particles
    return name.replace("~", r"^\dagger").replace("_", r"\_")


def _propagator_style(particle_name: str, theory: str) -> str:
    """Return 'style[, color]' string for a TikZ-Feynman edge."""
    try:
        p = TheoryRegistry.get_particle(theory, particle_name)
        base_style = p.propagator_style.value
    except ValueError:
        base_style = "plain"
    color = _STYLE_COLOR.get(base_style, "")
    return "{}, {}".format(base_style, color) if color else base_style


# ── TikZ template ─────────────────────────────────────────────────────────────

# External nodes use the TikZ-Feynman [particle={...}] decoration, which
# renders them as plain text labels with no filled dot — the correct
# convention for labelled external legs in a Feynman diagram.

# Two templates: tree-level uses layered layout (clean L→R flow for DAGs);
# loop diagrams use spring layout (force-directed, handles cycles correctly —
# layered layout breaks cyclic graphs into disconnected visual components).
_TIKZ_TEMPLATE_LAYERED = r"""
\documentclass[border=8pt]{standalone}
\usepackage{tikz-feynman}
\tikzfeynmanset{compat=1.1.0}
\begin{document}
\feynmandiagram [
  large, layered layout,
  every edge={line width=0.75pt},
] {
{%- for edge in edges %}
  {{ edge.start }} -- [{{ edge.style }}] {{ edge.end }}{% if not loop.last %},{% endif %}

{%- endfor %}
};
\end{document}
"""

_TIKZ_TEMPLATE_SPRING = r"""
\documentclass[border=8pt]{standalone}
\usepackage{tikz-feynman}
\tikzfeynmanset{compat=1.1.0}
\begin{document}
\feynmandiagram [
  large, spring layout,
  every edge={line width=0.75pt},
] {
{%- for edge in edges %}
  {{ edge.start }} -- [{{ edge.style }}] {{ edge.end }}{% if not loop.last %},{% endif %}

{%- endfor %}
};
\end{document}
"""

_env              = Environment(loader=BaseLoader())
_tmpl_layered     = _env.from_string(_TIKZ_TEMPLATE_LAYERED)
_tmpl_spring      = _env.from_string(_TIKZ_TEMPLATE_SPRING)


def _vertex_name(vid: int) -> str:
    """Convert vertex integer ID to a valid TikZ node name (no hyphens)."""
    return "vn{}".format(-vid) if vid < 0 else "v{}".format(vid)


def _external_node(edge_id: int, particle_name: str) -> str:
    r"""
    Build the TikZ node spec for an external particle endpoint.

    Produces:  ext0 [particle={\(\gamma\)}]
    The [particle={...}] option tells TikZ-Feynman to render this node as a
    plain text label (no visible dot), which is the standard visual for
    labelled external legs.
    """
    label = _particle_latex(particle_name)
    # Format: ext<id> [particle={\(<label>\)}]
    # Python str.format: {{ → { , }} → } , \\ → single backslash
    return "ext{} [particle={{\\({}\\)}}]".format(edge_id, label)


def diagram_to_tikz(diagram: Diagram) -> str:
    """
    Convert a Diagram to a compilable TikZ-Feynman LaTeX string.

    External leg endpoints carry [particle={...}] labels; internal
    propagators are drawn without momentum labels for a clean output.

    Returns the full .tex document content.
    """
    edge_specs = []
    for edge in diagram.edges:
        style = _propagator_style(edge.particle, diagram.theory)
        if edge.is_external:
            ext_node = _external_node(edge.id, edge.particle)
            if edge.start_vertex < 0:   # incoming: phantom → real vertex
                start_name = ext_node
                end_name   = _vertex_name(edge.end_vertex)
            else:                        # outgoing: real vertex → phantom
                start_name = _vertex_name(edge.start_vertex)
                end_name   = ext_node
        else:
            start_name = _vertex_name(edge.start_vertex)
            end_name   = _vertex_name(edge.end_vertex)

        edge_specs.append({
            "start": start_name,
            "end":   end_name,
            "style": style,
        })

    # Spring layout handles cyclic graphs (loops) correctly; layered layout
    # is a DAG algorithm that disconnects nodes when cycles are present.
    tmpl = _tmpl_spring if diagram.loop_order > 0 else _tmpl_layered
    return tmpl.render(edges=edge_specs, diagram=diagram)


def diagrams_to_tikz(diagrams: list[Diagram]) -> dict[int, str]:
    """Return {diagram_id: tikz_code} for all diagrams."""
    return {d.id: diagram_to_tikz(d) for d in diagrams}
