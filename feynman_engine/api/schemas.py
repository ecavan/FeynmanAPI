"""Request and response Pydantic schemas for the REST API."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DiagramFilters(BaseModel):
    """QGRAF-level filters applied during diagram generation."""
    no_tadpoles: bool = Field(default=True,  description="Exclude tadpole diagrams")
    one_pi:      bool = Field(default=False, description="1-particle-irreducible diagrams only")
    connected:   bool = Field(default=True,  description="Connected diagrams only")


class GenerateRequest(BaseModel):
    process: str = Field(
        ...,
        description="Scattering process string, e.g. 'e+ e- -> mu+ mu-'",
        examples=["e+ e- -> mu+ mu-", "e+ e- -> e+ e-"],
    )
    theory: str = Field(default="QED", description="Theory name: QED, QCD, EW, BSM")
    loops: int = Field(default=0, ge=0, le=2, description="Loop order (0=tree, 1=one-loop)")
    output_format: str = Field(
        default="svg",
        description="Output image format: svg, tikz",
    )
    filters: DiagramFilters = Field(
        default_factory=DiagramFilters,
        description="Diagram topology filters",
    )


class DiagramResponse(BaseModel):
    id: int
    process: str
    theory: str
    topology: Optional[str] = None
    loop_order: int
    symmetry_factor: Optional[float] = None
    n_vertices: int
    n_edges: int
    tikz_code: Optional[str] = None
    image_b64: Optional[str] = None    # base64-encoded SVG bytes
    image_format: Optional[str] = None


class GenerateResponse(BaseModel):
    diagrams: list[DiagramResponse]
    summary: dict
    metadata: dict


class AmplitudeResponse(BaseModel):
    process: str
    theory: str
    description: str
    msq_latex: str          # LaTeX string for spin-averaged |M|²
    msq_sympy: str          # sympy str() form
    notes: str
    supported: bool


class ParticleResponse(BaseModel):
    name: str
    pdg_id: Optional[int] = None
    particle_type: str
    mass: Optional[str] = None
    charge: Optional[float] = None
    propagator_style: str


class DescribeResponse(BaseModel):
    valid: bool
    process: str
    theory: str
    incoming: list[dict]
    outgoing: list[dict]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
