"""Request and response Pydantic schemas for the REST API."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DiagramFilters(BaseModel):
    """Filters applied during diagram generation."""
    no_tadpoles:      bool = Field(default=True,  description="Exclude tadpole diagrams")
    one_pi:           bool = Field(default=False, description="1-particle-irreducible diagrams only")
    connected:        bool = Field(default=True,  description="Connected diagrams only")
    unique_topologies: bool = Field(default=False, description="Return only one representative diagram per topology class")


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
    loops: int = 0
    description: str
    msq_latex: str          # LaTeX string for spin-averaged |M|²
    msq_sympy: str          # sympy str() form
    integral_latex: Optional[str] = None # unintegrated formula or loop integrals
    notes: str
    backend: Optional[str] = None
    supported: bool
    has_msq: bool = False
    has_integral: bool = False
    availability_message: Optional[str] = None
    approximation_level: str = "exact-symbolic"
    evaluation_point: Optional[dict] = None
    # Honesty flags — what this amplitude can actually be used for.
    # See AmplitudeResult.features for the canonical map.
    is_symbolic_function: bool = True
    features: Optional[dict] = None


class LoopIntegralResponse(BaseModel):
    process: str
    theory: str
    loops: int
    integral_latex: Optional[str] = None
    has_integral: bool
    notes: str = ""
    availability_message: Optional[str] = None


class ParticleResponse(BaseModel):
    name: str
    pdg_id: Optional[int] = None
    pdg_name: Optional[str] = None
    latex_name: Optional[str] = None
    particle_type: str
    mass: Optional[str] = None
    mass_mev: Optional[float] = None
    width_mev: Optional[float] = None
    charge: Optional[float] = None
    propagator_style: str


class DescribeResponse(BaseModel):
    valid: bool
    process: str
    theory: str
    incoming: list[dict]
    outgoing: list[dict]


class CrossSectionResponse(BaseModel):
    process: str
    theory: str
    sqrt_s_gev: float
    s_gev2: float
    sigma_pb: float
    sigma_uncertainty_pb: float
    dsigma_at_cos0_pb: Optional[float] = None
    has_tchannel_pole: bool
    cos_theta_range: list[float]
    eps: float
    converged: bool
    formula_latex: str
    supported: bool


class DecayWidthResponse(BaseModel):
    process: str
    theory: str
    parent: str
    daughters: list[str]
    parent_mass_gev: float
    daughter_masses_gev: list[float]
    msq_value: float
    msq_latex: str
    width_gev: float
    width_mev: float
    branching_ratio: Optional[float] = None  # Γ_partial / Γ_total when known
    pdg_width_mev: Optional[float] = None
    pct_off_pdg: Optional[float] = None
    backend: Optional[str] = None
    notes: str = ""
    trust_level: str = "approximate"
    trust_reference: Optional[str] = None
    accuracy_caveat: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
