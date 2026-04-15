"""Best-effort amplitude proxies for processes without an exact backend."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import sqrt
from typing import Optional

from sympy import Float, Integer, Rational, latex, pi, simplify

from particle import Particle as PDGParticle

from feynman_engine.amplitudes.loop import PVExpansion, pv_reduce
from feynman_engine.amplitudes.looptools_bridge import evaluate_pv_expansion, is_available as looptools_available
from feynman_engine.amplitudes.pdg_masses import MASS_GEV as _MASS_DEFAULTS_GEV
from feynman_engine.amplitudes.symbolic import get_loop_integral_latex, get_tree_integral_latex
from feynman_engine.amplitudes.types import AmplitudeResult
from feynman_engine.core.generator import generate_diagrams
from feynman_engine.core.models import Diagram, ParticleType
from feynman_engine.physics.registry import TheoryRegistry
from feynman_engine.physics.translator import parse_process


ALPHA_EM = 1.0 / 137.036
ALPHA_S = 0.118
E_EM = sqrt(4.0 * float(pi) * ALPHA_EM)
G_S = sqrt(4.0 * float(pi) * ALPHA_S)
SIN2_THETA_W = 0.23122
COS_THETA_W = sqrt(1.0 - SIN2_THETA_W)
G_WEAK = 0.653
G_Z = G_WEAK / COS_THETA_W
VEV_GEV = 246.0



@dataclass(frozen=True)
class RepresentativePoint:
    label: str
    scale_gev: float
    sqrt_s_gev: float | None = None
    s_gev2: float | None = None
    t_gev2: float | None = None
    u_gev2: float | None = None
    parent_mass_gev: float | None = None
    channel_q_sq: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        payload = {
            "label": self.label,
            "scale_gev": round(self.scale_gev, 6),
        }
        if self.sqrt_s_gev is not None:
            payload["sqrt_s_gev"] = round(self.sqrt_s_gev, 6)
        if self.s_gev2 is not None:
            payload["s_gev2"] = round(self.s_gev2, 6)
        if self.t_gev2 is not None:
            payload["t_gev2"] = round(self.t_gev2, 6)
        if self.u_gev2 is not None:
            payload["u_gev2"] = round(self.u_gev2, 6)
        if self.parent_mass_gev is not None:
            payload["parent_mass_gev"] = round(self.parent_mass_gev, 6)
        return payload


def _mass_gev(theory: str, particle_name: str) -> float:
    particle = TheoryRegistry.get_particle(theory, particle_name)
    if particle.mass_mev is not None:
        return max(float(particle.mass_mev) / 1000.0, 0.0)
    if particle.mass in (None, "", "0"):
        return 0.0
    return float(_MASS_DEFAULTS_GEV.get(str(particle.mass), 0.0))


def _width_gev(theory: str, particle_name: str) -> float:
    particle = TheoryRegistry.get_particle(theory, particle_name)
    if particle.width_mev is not None:
        return max(float(particle.width_mev) / 1000.0, 0.0)

    mass = _mass_gev(theory, particle_name)
    if mass <= 0.0:
        return 0.0

    if particle_name in {"Zp"}:
        return 0.03 * mass
    if particle.particle_type == ParticleType.BOSON and particle_name in {"Z", "W+", "W-", "H"}:
        return 0.02 * mass
    return 0.0


def _spin_states(theory: str, particle_name: str) -> int:
    particle = TheoryRegistry.get_particle(theory, particle_name)

    if particle.pdg_id is not None:
        try:
            pdg_particle = PDGParticle.from_pdgid(particle.pdg_id)
            j_val = getattr(pdg_particle, "J", None)
            if j_val is not None:
                states = int(round(2.0 * float(j_val) + 1.0))
                if particle_name in {"gamma", "g"} and states == 3:
                    return 2
                return max(states, 1)
        except Exception:
            pass

    if particle.particle_type in {ParticleType.FERMION, ParticleType.ANTIFERMION}:
        return 2
    if particle.particle_type == ParticleType.BOSON:
        return 2 if particle_name in {"gamma", "g"} else 3
    return 1


def _color_dimension(color: str | None) -> int:
    return {
        "1": 1,
        "3": 3,
        "3bar": 3,
        "8": 8,
    }.get(color or "1", 1)


def _incoming_average(spec, theory: str) -> Float:
    avg = Float(1.0)
    for particle_name in spec.incoming:
        particle = TheoryRegistry.get_particle(theory, particle_name)
        avg *= Float(_spin_states(theory, particle_name))
        avg *= Float(_color_dimension(particle.color))
    return avg if avg > 0 else Float(1.0)


def _representative_point(spec, theory: str) -> RepresentativePoint:
    incoming_masses = [_mass_gev(theory, name) for name in spec.incoming]
    outgoing_masses = [_mass_gev(theory, name) for name in spec.outgoing]

    if len(spec.incoming) == 1:
        threshold = max(sum(outgoing_masses), 0.5)
        parent_mass = max(incoming_masses[0] if incoming_masses else 0.0, threshold * 1.05, 1.0)
        scale = parent_mass
        scale_sq = scale * scale
        return RepresentativePoint(
            label="parent rest frame with a symmetric decay configuration",
            scale_gev=scale,
            parent_mass_gev=parent_mass,
            channel_q_sq={
                "s-channel": scale_sq,
                "t-channel": -0.25 * scale_sq,
                "u-channel": -0.25 * scale_sq,
                "default": scale_sq / max(1, len(spec.outgoing)),
            },
        )

    threshold = max(sum(incoming_masses), sum(outgoing_masses), 0.5)
    sqrt_s = max(10.0, 1.2 * threshold)
    s_val = sqrt_s * sqrt_s

    if len(spec.incoming) == 2 and len(spec.outgoing) == 2:
        return RepresentativePoint(
            label="center-of-mass frame at cos(theta)=0",
            scale_gev=sqrt_s,
            sqrt_s_gev=sqrt_s,
            s_gev2=s_val,
            t_gev2=-0.5 * s_val,
            u_gev2=-0.5 * s_val,
            channel_q_sq={
                "s-channel": s_val,
                "t-channel": -0.5 * s_val,
                "u-channel": -0.5 * s_val,
                "default": 0.5 * s_val,
            },
        )

    return RepresentativePoint(
        label="center-of-mass frame at a symmetric multi-leg phase-space point",
        scale_gev=sqrt_s,
        sqrt_s_gev=sqrt_s,
        s_gev2=s_val,
        channel_q_sq={
            "s-channel": s_val,
            "t-channel": -s_val / max(2, len(spec.outgoing)),
            "u-channel": -s_val / max(2, len(spec.outgoing)),
            "default": s_val / max(1, len(spec.outgoing)),
        },
    )


def _fermion_charge(theory: str, particle_name: str) -> float:
    return float(TheoryRegistry.get_particle(theory, particle_name).charge or 0.0)


def _t3_for_particle(particle_name: str) -> float:
    base = particle_name.rstrip("~+-")
    if base in {"e", "mu", "tau", "d", "s", "b"}:
        return -0.5
    if base in {"nu_e", "nu_mu", "nu_tau", "u", "c", "t"}:
        return 0.5
    return 0.0


def _z_fermion_coupling(theory: str, particle_name: str) -> Float:
    q_val = _fermion_charge(theory, particle_name)
    t3_val = _t3_for_particle(particle_name)
    g_v = t3_val - 2.0 * q_val * SIN2_THETA_W
    g_a = t3_val
    strength = G_Z * sqrt(g_v * g_v + g_a * g_a)
    return Float(max(strength, 1e-6))


def _higgs_yukawa(theory: str, particle_name: str) -> Float:
    return Float(max(_mass_gev(theory, particle_name) / VEV_GEV, 1e-6))


def _gauge_quartic_strength(names: list[str]) -> Float:
    names_set = set(names)
    if names_set == {"W+", "W-", "gamma", "gamma"}:
        return Float(E_EM ** 2)
    if names_set == {"W+", "W-", "Z", "gamma"}:
        return Float(E_EM * G_WEAK * COS_THETA_W)
    if names_set == {"W+", "W-", "Z", "Z"} or names_set == {"W+", "W-"}:
        return Float((G_WEAK * COS_THETA_W) ** 2)
    return Float(G_WEAK ** 2)


def _vertex_strength(theory: str, names: list[str], point: RepresentativePoint) -> Float:
    particle_objs = [TheoryRegistry.get_particle(theory, name) for name in names]
    names_set = set(names)

    if theory == "QED":
        charges = [abs(float(p.charge or 0.0)) for p in particle_objs if abs(float(p.charge or 0.0)) > 0]
        return Float(E_EM * (charges[0] if charges else 1.0))

    if theory == "QCD":
        gluon_count = sum(1 for name in names if name == "g")
        if gluon_count >= 4:
            return Float(G_S ** 2)
        return Float(G_S)

    if theory == "EW":
        if names_set == {"W+", "W-", "gamma"}:
            return Float(E_EM)
        if names_set == {"W+", "W-", "Z"}:
            return Float(G_WEAK * COS_THETA_W)
        if len(names) == 4 and {"W+", "W-"} <= names_set:
            return _gauge_quartic_strength(names)
        if names_set == {"H", "W+", "W-"}:
            return Float(max(G_WEAK * _MASS_DEFAULTS_GEV["m_W"] / max(point.scale_gev, 1.0), 0.1))
        if names_set == {"H", "Z"} or names_set == {"H", "Z", "Z"}:
            return Float(max(G_Z * _MASS_DEFAULTS_GEV["m_Z"] / max(point.scale_gev, 1.0), 0.1))
        if "gamma" in names_set:
            charged = next((name for name in names if abs(_fermion_charge(theory, name)) > 0.0), None)
            return Float(E_EM * abs(_fermion_charge(theory, charged))) if charged else Float(E_EM)
        if "Z" in names_set:
            fermions = [name for name in names if TheoryRegistry.get_particle(theory, name).particle_type in {ParticleType.FERMION, ParticleType.ANTIFERMION}]
            if fermions:
                return _z_fermion_coupling(theory, fermions[0])
            return Float(G_Z * COS_THETA_W)
        if "W+" in names_set or "W-" in names_set:
            return Float(G_WEAK / sqrt(2.0))
        if "H" in names_set:
            fermions = [name for name in names if TheoryRegistry.get_particle(theory, name).particle_type in {ParticleType.FERMION, ParticleType.ANTIFERMION}]
            if fermions:
                return _higgs_yukawa(theory, fermions[0])
            if len(names) >= 3:
                return Float(0.13)
        return Float(max(G_WEAK, 0.1))

    if theory == "BSM":
        if "gamma" in names_set:
            charged = next((name for name in names if abs(_fermion_charge(theory, name)) > 0.0), None)
            return Float(E_EM * abs(_fermion_charge(theory, charged))) if charged else Float(E_EM)
        if "Zp" in names_set:
            if "chi" in names_set or "chi~" in names_set:
                return Float(0.6)
            return Float(0.35)
        if names.count("chi") + names.count("chi~") >= 4:
            return Float(0.25)
        return Float(0.35)

    return Float(1.0)


def _propagator_weight(theory: str, edge, topology: str | None, point: RepresentativePoint) -> Float:
    particle = TheoryRegistry.get_particle(theory, edge.particle)
    mass = _mass_gev(theory, edge.particle)
    width = _width_gev(theory, edge.particle)
    q_sq = point.channel_q_sq.get(topology or "", point.channel_q_sq.get("default", point.scale_gev ** 2))
    q_sq_sym = Float(q_sq)
    m_sq_sym = Float(mass * mass)
    width_term = Float((mass * width) ** 2)
    regulator = Float(max((point.scale_gev ** 4) * 1e-10, 1e-12))

    denom = (q_sq_sym - m_sq_sym) ** 2 + width_term + regulator
    numerator = Float(1.0)
    if particle.particle_type in {ParticleType.FERMION, ParticleType.ANTIFERMION}:
        numerator = Float(max(abs(q_sq), mass * mass, 1.0))
    return simplify(numerator / denom)


def _tree_proxy_expr(diagrams: list[Diagram], spec, theory: str, point: RepresentativePoint):
    total = Integer(0)
    for diagram in diagrams:
        diagram_factor = Float(abs(float(diagram.symmetry_factor or 1.0)))
        for vertex in diagram.vertices:
            diagram_factor *= _vertex_strength(theory, vertex.particles, point) ** 2
        for edge in diagram.internal_edges:
            diagram_factor *= _propagator_weight(theory, edge, diagram.topology, point)
        total += diagram_factor
    return simplify(total / _incoming_average(spec, theory))


def get_approximate_tree_amplitude(process: str, theory: str = "QED") -> Optional[AmplitudeResult]:
    theory = theory.upper()
    spec = parse_process(process.strip(), theory=theory, loops=0)
    diagrams = generate_diagrams(spec)
    tree_diagrams = [diagram for diagram in diagrams if diagram.loop_order == 0]
    if not tree_diagrams:
        return None

    point = _representative_point(spec, theory)
    expr = _tree_proxy_expr(tree_diagrams, spec, theory, point)
    value = Float(max(float(expr.evalf()), 0.0))

    return AmplitudeResult(
        process=spec.raw,
        theory=theory,
        msq=value,
        msq_latex=latex(value.evalf(6)),
        integral_latex=get_tree_integral_latex(spec.raw, theory),
        description="Best-effort pointwise |M|^2 proxy from generated tree diagrams",
        notes=(
            "approximate-pointwise: built from QGRAF tree diagrams using default couplings, "
            "spin/color averaging, and Breit-Wigner-like propagator weights. "
            f"Evaluated at a representative point: {point.label}."
        ),
        backend="tree-proxy",
        approximation_level="approximate-pointwise",
        evaluation_point=point.as_dict(),
    )


def _substitution_map(point: RepresentativePoint) -> dict[str, float]:
    subs = {
        "s": point.s_gev2 if point.s_gev2 is not None else point.scale_gev ** 2,
        "t": point.t_gev2 if point.t_gev2 is not None else -0.5 * point.scale_gev ** 2,
        "u": point.u_gev2 if point.u_gev2 is not None else -0.5 * point.scale_gev ** 2,
        "alpha": ALPHA_EM,
        "alpha_s": ALPHA_S,
    }
    for key, value in _MASS_DEFAULTS_GEV.items():
        subs[key] = value * value
    return subs


def _substitute_numeric(value, substitutions: dict[str, float]):
    if hasattr(value, "subs"):
        value = value.subs({sym: substitutions[sym.name] for sym in value.free_symbols if sym.name in substitutions})
    return float(value)


def _numeric_integral(integral, substitutions: dict[str, float]):
    values = {}
    for item in fields(integral):
        values[item.name] = _substitute_numeric(getattr(integral, item.name), substitutions)
    return type(integral)(**values)


def _numeric_expansion(expansion: PVExpansion, substitutions: dict[str, float]) -> PVExpansion:
    terms = {}
    for integral, coeff in expansion.terms.items():
        terms[_numeric_integral(integral, substitutions)] = _substitute_numeric(coeff, substitutions)
    return PVExpansion(
        process=expansion.process,
        diagram_id=expansion.diagram_id,
        topology=expansion.topology,
        terms=terms,
        uv_divergent=expansion.uv_divergent,
        ir_divergent=expansion.ir_divergent,
        notes=list(expansion.notes),
    )


def get_approximate_loop_amplitude(
    process: str,
    theory: str = "QED",
    loops: int = 1,
) -> Optional[AmplitudeResult]:
    theory = theory.upper()
    spec = parse_process(process.strip(), theory=theory, loops=loops)
    diagrams = generate_diagrams(spec)
    loop_diagrams = [diagram for diagram in diagrams if diagram.loop_order == loops]
    if not loop_diagrams:
        return None

    point = _representative_point(spec, theory)
    substitutions = _substitution_map(point)

    expansions: list[PVExpansion] = []
    for diagram in loop_diagrams[:48]:
        expansion = pv_reduce(diagram, theory)
        if expansion is not None:
            expansions.append(expansion)

    numeric_values: list[complex] = []
    if looptools_available():
        for expansion in expansions:
            try:
                numeric = evaluate_pv_expansion(_numeric_expansion(expansion, substitutions))
            except Exception:
                numeric = None
            if numeric is not None:
                numeric_values.append(numeric)

    if numeric_values:
        loop_proxy = abs(sum(numeric_values))
        tree_reference = get_approximate_tree_amplitude(spec.raw, theory)
        tree_value = 0.0
        if tree_reference is not None:
            try:
                tree_value = max(float(tree_reference.msq), 0.0)
            except Exception:
                tree_value = 0.0
        corrected = (sqrt(tree_value) + loop_proxy) ** 2 if tree_value > 0.0 else loop_proxy ** 2
        notes = (
            "approximate-pointwise: PV-reduced loop diagrams were evaluated numerically with LoopTools "
            "at a representative point, then combined into a positive |M|^2 proxy. "
            f"Evaluated at: {point.label}."
        )
        if len(loop_diagrams) > len(expansions):
            notes += " Some loop diagrams could not be reduced and were omitted from the proxy."
        if len(loop_diagrams) > 48:
            notes += " The proxy used the first 48 loop diagrams to keep evaluation bounded."
        return AmplitudeResult(
            process=spec.raw,
            theory=theory,
            msq=Float(corrected),
            msq_latex=latex(Float(corrected).evalf(6)),
            integral_latex=get_loop_integral_latex(spec.raw, theory, loops=loops),
            description="Best-effort pointwise 1-loop |M|^2 proxy from PV reductions",
            notes=notes,
            backend="looptools-proxy",
            approximation_level="approximate-pointwise",
            evaluation_point=point.as_dict(),
        )

    tree_reference = get_approximate_tree_amplitude(spec.raw, theory)
    if tree_reference is None:
        return None

    coupling = E_EM if theory in {"QED", "EW", "BSM"} else G_S
    suppression = (coupling / (16.0 * float(pi) * float(pi))) ** max(loops, 1)
    loop_proxy = max(float(tree_reference.msq), 0.0) * max(suppression, 1e-8)
    return AmplitudeResult(
        process=spec.raw,
        theory=theory,
        msq=Float(loop_proxy),
        msq_latex=latex(Float(loop_proxy).evalf(6)),
        integral_latex=get_loop_integral_latex(spec.raw, theory, loops=loops),
        description="Fallback pointwise loop |M|^2 proxy",
        notes=(
            "approximate-pointwise: LoopTools evaluation was unavailable for this loop configuration, "
            "so the loop correction was estimated by applying a loop-factor suppression to the tree proxy."
        ),
        backend="loop-factor-proxy",
        approximation_level="approximate-pointwise",
        evaluation_point=point.as_dict(),
    )
