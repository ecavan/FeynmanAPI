from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

try:
    from particle import Particle as PDGParticle
except ImportError:  # pragma: no cover - optional at import time
    PDGParticle = None


class ParticleType(str, Enum):
    FERMION = "fermion"
    ANTIFERMION = "antifermion"
    BOSON = "boson"
    SCALAR = "scalar"
    GHOST = "ghost"


class PropagatorStyle(str, Enum):
    """Maps to TikZ-Feynman edge styles."""
    FERMION = "fermion"
    ANTI_FERMION = "anti fermion"
    PHOTON = "photon"
    BOSON = "boson"
    GLUON = "gluon"
    SCALAR = "scalar"
    GHOST = "ghost"
    CHARGED_BOSON = "charged boson"


class Particle(BaseModel):
    name: str                           # e.g. "e-", "gamma", "g"
    pdg_id: Optional[int] = None        # PDG particle ID
    particle_type: ParticleType
    mass: Optional[str] = None          # symbolic mass, e.g. "m_e", "0"
    charge: Optional[float] = None
    color: Optional[str] = None         # color rep: "3", "8", "1"
    propagator_style: PropagatorStyle
    antiparticle: Optional[str] = None  # name of antiparticle (None if self-conjugate)
    pdg_name: Optional[str] = None
    latex_name: Optional[str] = None
    mass_mev: Optional[float] = None
    width_mev: Optional[float] = None

    def model_post_init(self, __context) -> None:
        if self.pdg_id is None or PDGParticle is None:
            return
        try:
            pdg_particle = PDGParticle.from_pdgid(self.pdg_id)
        except Exception:
            return

        if self.charge is None:
            self.charge = pdg_particle.charge
        if self.pdg_name is None:
            self.pdg_name = pdg_particle.pdg_name
        if self.latex_name is None:
            self.latex_name = pdg_particle.latex_name
        if self.mass_mev is None:
            self.mass_mev = pdg_particle.mass
        if self.width_mev is None:
            self.width_mev = pdg_particle.width


class Vertex(BaseModel):
    id: int
    particles: list[str]                # particle names meeting at this vertex
    coupling: Optional[str] = None      # e.g. "e" for QED vertex


class Edge(BaseModel):
    id: int
    start_vertex: int
    end_vertex: int
    particle: str                       # particle name
    is_external: bool = False
    momentum: Optional[str] = None      # symbolic momentum label


class Diagram(BaseModel):
    id: int
    vertices: list[Vertex]
    edges: list[Edge]
    loop_order: int
    symmetry_factor: Optional[float] = None
    theory: str                         # "QED", "QCD", etc.
    process: str                        # "e+ e- -> mu+ mu-"
    topology: Optional[str] = None      # "s-channel", "t-channel", "box", etc.
    canonical_hash: Optional[str] = None  # for deduplication

    @property
    def external_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.is_external]

    @property
    def internal_edges(self) -> list[Edge]:
        return [e for e in self.edges if not e.is_external]


class GenerationResult(BaseModel):
    diagrams: list[Diagram] = Field(default_factory=list)
    images: dict[int, bytes] = Field(default_factory=dict)   # diagram_id → image bytes
    tikz_code: dict[int, str] = Field(default_factory=dict)  # diagram_id → TikZ string
    summary: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
