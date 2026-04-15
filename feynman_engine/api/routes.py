"""FastAPI route definitions."""
from __future__ import annotations

import base64
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import Response

from feynman_engine.render.compiler import tikz_to_pdf, RenderError, MissingDependencyError
from feynman_engine.api.schemas import (
    AmplitudeResponse,
    CrossSectionResponse,
    DescribeResponse,
    DiagramResponse,
    GenerateRequest,
    GenerateResponse,
    LoopIntegralResponse,
    ParticleResponse,
)
from feynman_engine.engine import FeynmanEngine
from feynman_engine.physics.registry import TheoryRegistry
from feynman_engine.physics.translator import parse_process
from feynman_engine.physics.amplitude import (
    get_amplitude,
    get_best_effort_loop_amplitude,
    list_supported_processes,
)

from feynman_engine.amplitudes import (
    get_loop_integral_latex,
    get_tree_integral_latex,
    get_loop_amplitude,
    looptools_available,
    get_loop_curated_results,
    evaluate_photon_selfenergy,
    evaluate_vertex_form_factor,
    evaluate_schwinger_amm,
    evaluate_vacuum_polarisation,
)
from feynman_engine.amplitudes.renorm import (
    alpha_running,
    alpha_s_running,
    qed_renormalised_photon_selfenergy,
    qed_renormalised_vertex_ff,
    renorm_status,
)

router = APIRouter(prefix="/api")
_engine = FeynmanEngine()

# Diagram store for /diagram/{id}/svg and /tikz endpoints
_last_result_store: dict[int, dict] = {}


@router.get("/status", summary="Backend and dependency status")
def status():
    return _engine.status()


@router.get("/theories", response_model=list[str], summary="List available theories")
def list_theories():
    return _engine.list_theories()


@router.get(
    "/theories/{theory}/particles",
    response_model=list[ParticleResponse],
    summary="List particles for a theory",
)
def list_particles(theory: str = Path(...)):
    try:
        particles = TheoryRegistry.get_particles(theory.upper())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return [
        ParticleResponse(
            name=p.name, pdg_id=p.pdg_id,
            pdg_name=p.pdg_name,
            latex_name=p.latex_name,
            particle_type=p.particle_type.value,
            mass=p.mass,
            mass_mev=p.mass_mev,
            width_mev=p.width_mev,
            charge=p.charge,
            propagator_style=p.propagator_style.value,
        )
        for p in particles.values()
    ]


@router.post("/generate", response_model=GenerateResponse, summary="Generate Feynman diagrams")
def generate(request: GenerateRequest):
    try:
        result = _engine.generate(
            process=request.process,
            theory=request.theory,
            loops=request.loops,
            output_format=request.output_format,
            filters=request.filters.model_dump(),
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Post-process: unique topology filter (keep first representative per topology class).
    diagrams = result.diagrams
    if request.filters.unique_topologies:
        seen: set[str] = set()
        filtered: list = []
        for d in diagrams:
            topo = d.topology or "unknown"
            if topo not in seen:
                seen.add(topo)
                filtered.append(d)
        diagrams = filtered

    _last_result_store.clear()
    for d in diagrams:
        _last_result_store[d.id] = {
            "diagram": d,
            "tikz": result.tikz_code.get(d.id),
            "image": result.images.get(d.id),
            "format": request.output_format,
        }

    diagram_responses = []
    for d in diagrams:
        image_bytes = result.images.get(d.id)
        image_b64 = base64.b64encode(image_bytes).decode() if image_bytes else None
        diagram_responses.append(DiagramResponse(
            id=d.id,
            process=d.process,
            theory=d.theory,
            topology=d.topology,
            loop_order=d.loop_order,
            symmetry_factor=d.symmetry_factor,
            n_vertices=len(d.vertices),
            n_edges=len(d.edges),
            tikz_code=result.tikz_code.get(d.id),
            image_b64=image_b64,
            image_format=request.output_format if image_b64 else None,
        ))

    # Rebuild summary to reflect filtered diagram set.
    from collections import Counter
    topology_counts = Counter(d.topology or "unknown" for d in diagrams)
    summary = {
        **result.summary,
        "total_diagrams": len(diagrams),
        "topology_counts": dict(topology_counts),
    }
    if request.filters.unique_topologies:
        summary["unique_topologies_filter"] = True

    return GenerateResponse(
        diagrams=diagram_responses,
        summary=summary,
        metadata=result.metadata,
    )


@router.get("/diagram/{diagram_id}/tikz", response_class=Response,
            summary="Get TikZ source for a diagram")
def get_tikz(diagram_id: int = Path(...)):
    entry = _last_result_store.get(diagram_id)
    if not entry or not entry.get("tikz"):
        raise HTTPException(status_code=404,
                            detail=f"Diagram {diagram_id} not found. Call /api/generate first.")
    return Response(content=entry["tikz"], media_type="text/plain")


@router.get("/diagram/{diagram_id}/svg", response_class=Response,
            summary="Get rendered SVG for a diagram")
def get_svg(diagram_id: int = Path(...)):
    entry = _last_result_store.get(diagram_id)
    if not entry or not entry.get("image"):
        raise HTTPException(status_code=404,
                            detail=f"No SVG for diagram {diagram_id}. Call /api/generate first.")
    return Response(content=entry["image"], media_type="image/svg+xml")


@router.get("/diagram/{diagram_id}/pdf", response_class=Response,
            summary="Get rendered PDF for a diagram")
def get_pdf(diagram_id: int = Path(...)):
    entry = _last_result_store.get(diagram_id)
    if not entry or not entry.get("tikz"):
        raise HTTPException(status_code=404,
                            detail=f"Diagram {diagram_id} not found. Call /api/generate first.")
    try:
        pdf_bytes = tikz_to_pdf(entry["tikz"])
    except (RenderError, MissingDependencyError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="diagram_{diagram_id}.pdf"'},
    )


@router.get("/describe", response_model=DescribeResponse,
            summary="Validate and describe a process")
def describe_process(
    process: str = Query(...),
    theory: str = Query(default="QED"),
):
    try:
        info = _engine.describe_process(process, theory)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return DescribeResponse(**info)


@router.get("/amplitude", response_model=AmplitudeResponse,
            summary="Get the best available spin-averaged |M|² estimate for a process")
def get_amplitude_endpoint(
    process: str = Query(..., description="e.g. 'e+ e- -> mu+ mu-'"),
    theory:  str = Query(default="QED"),
    loops:   int = Query(default=0, ge=0, le=2),
):
    process_clean = process.strip()
    theory_upper = theory.upper()

    try:
        parse_process(process_clean, theory_upper, loops=loops)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if loops == 0:
        result = get_amplitude(process_clean, theory_upper)
    else:
        result = get_best_effort_loop_amplitude(process_clean, theory_upper, loops=loops)

    if result is not None:
        integral_latex = getattr(result, "integral_latex", None)
        if not integral_latex:
            integral_latex = (
                get_tree_integral_latex(process_clean, theory_upper)
                if loops == 0
                else get_loop_integral_latex(process_clean, theory_upper, loops=loops)
            )
        has_msq = result.msq is not None
        return AmplitudeResponse(
            process=result.process,
            theory=result.theory,
            loops=loops,
            description=result.description,
            msq_latex=result.msq_latex if has_msq else "",
            msq_sympy=str(result.msq) if has_msq else "",
            integral_latex=integral_latex,
            notes=result.notes,
            backend=getattr(result, "backend", None),
            supported=has_msq,
            has_msq=has_msq,
            has_integral=bool(integral_latex),
            availability_message=(
                None
                if integral_latex
                else "An integral representation is not available yet for this process."
            ),
            approximation_level=getattr(result, "approximation_level", "exact-symbolic"),
            evaluation_point=getattr(result, "evaluation_point", None),
        )

    integral = (
        get_tree_integral_latex(process_clean, theory_upper)
        if loops == 0
        else get_loop_integral_latex(process_clean, theory_upper, loops=loops)
    )
    if integral is None:
        supported = [p["process"] for p in list_supported_processes()]
        message = (
            f"No {loops}-loop |M|² or integral representation is currently available for "
            f"'{process_clean}' in {theory_upper}. Processes with full |M|²: {supported}"
        )
        return AmplitudeResponse(
            process=process_clean,
            theory=theory_upper,
            loops=loops,
            description=f"{loops}-loop Feynman amplitude" if loops else "Tree-level Feynman amplitude",
            msq_latex="",
            msq_sympy="",
            integral_latex=None,
            notes=(
                "This process is valid, but the current backend has no amplitude or "
                "integral representation for it yet."
            ),
            backend="unavailable",
            supported=False,
            has_msq=False,
            has_integral=False,
            availability_message=message,
            approximation_level="unavailable",
        )
    return AmplitudeResponse(
        process=process_clean,
        theory=theory_upper,
        loops=loops,
        description=f"{loops}-loop Feynman amplitude" if loops else "Tree-level Feynman amplitude",
        msq_latex="",
        msq_sympy="",
        integral_latex=integral,
        notes=(
            "Spin-averaged |M|² is not available yet for this process/order, "
            "but the integral representation is shown below."
        ),
        backend="integral-only",
        supported=False,
        has_msq=False,
        has_integral=True,
        availability_message=(
            "Spin-averaged |M|² is not available yet for this process, "
            "but the integral representation is shown below."
        ),
        approximation_level="integral-only",
    )


@router.get("/amplitude/loop-integral", response_model=LoopIntegralResponse,
            summary="Get LaTeX for the loop-level integral of a process")
def get_loop_integral_endpoint(
    process: str = Query(..., description="e.g. 'e+ e- -> mu+ mu-'"),
    theory:  str = Query(default="QED"),
    loops:   int = Query(default=1, ge=1, le=2),
):
    """Return a LaTeX string showing the unevaluated loop integral."""
    process_clean = process.strip()
    theory_upper = theory.upper()

    try:
        parse_process(process_clean, theory_upper, loops=loops)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    result = get_loop_integral_latex(process_clean, theory_upper, loops=loops)
    if result is None:
        return LoopIntegralResponse(
            process=process_clean,
            theory=theory_upper,
            loops=loops,
            integral_latex=None,
            has_integral=False,
            notes="The loop integral representation is not available yet for this process/order.",
            availability_message=(
                f"Could not generate a {loops}-loop integral representation for "
                f"'{process_clean}' in {theory_upper}."
            ),
        )
    return LoopIntegralResponse(
        process=process_clean,
        theory=theory_upper,
        loops=loops,
        integral_latex=result,
        has_integral=True,
    )


@router.get("/amplitude/loop-pv",
            summary="PV-reduced 1-loop amplitude with symbolic decomposition")
def get_loop_pv_endpoint(
    process: str = Query(..., description="e.g. 'e+ e- -> mu+ mu-'"),
    theory:  str = Query(default="QED"),
    loops:   int = Query(default=1, ge=1, le=1),
):
    """Return the Passarino-Veltman decomposition of the first 1-loop diagram.

    The response includes:
    - ``integral_latex``: the full symbolic PV expansion as LaTeX
    - ``pv_terms``: each PV integral with its symbolic coefficient, separated
      into individual structured entries
    - ``topology``: self-energy, triangle, or box

    Numerical evaluation of the scalar integrals requires LoopTools.
    """
    from feynman_engine.amplitudes.loop import get_loop_pv_decomposition

    expansion = get_loop_pv_decomposition(process.strip(), theory.upper(), loops=loops)
    if expansion is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Could not generate PV-reduced loop amplitude for '{process}' "
                f"in {theory} at {loops}-loop. "
                "Check that QGRAF can generate diagrams for this process."
            ),
        )

    # Also get the full AmplitudeResult for integral_latex and metadata.
    result = get_loop_amplitude(process.strip(), theory.upper(), loops=loops)

    return {
        "process": process.strip(),
        "theory": theory.upper(),
        "loops": loops,
        "topology": expansion.topology.value,
        "integral_latex": expansion.to_latex(),
        "pv_terms": expansion.to_terms_list(),
        "uv_divergent": expansion.uv_divergent,
        "ir_divergent": expansion.ir_divergent,
        "description": result.description if result else "",
        "notes": expansion.notes,
        "backend": result.backend if result else "pv-reduction",
        "looptools_available": looptools_available(),
    }


@router.get("/amplitude/loop-evaluate",
            summary="Numerically evaluate a 1-loop amplitude via LoopTools")
def get_loop_evaluate_endpoint(
    observable: str = Query(
        ...,
        description=(
            "Which 1-loop observable to evaluate.  Choices:\n"
            "  'photon_selfenergy'   — Σ_T(k²) vacuum polarisation\n"
            "  'vertex_ff'           — δF₁(q²) vertex form factor correction\n"
            "  'schwinger_amm'       — a_e = F₂(0) = α/(2π) Schwinger term\n"
            "  'vacuum_polarisation' — Π(q²) leptonic VP\n"
        ),
    ),
    q_sq: float = Query(default=1.0, description="Squared momentum transfer q² (GeV²)"),
    m_sq: float = Query(default=2.611e-7, description="Internal mass squared m² (GeV²); default = m_e² = (0.511 MeV)²"),
    theory: str = Query(default="QED"),
):
    """Numerically evaluate a curated 1-loop amplitude using LoopTools.

    Requires ``feynman install-looptools`` to be run first.  Returns the complex
    numerical value of the requested observable.

    Default kinematic point: q² = 1 GeV², m² = m_e² (≈ 2.61×10⁻⁷ GeV²).
    """
    lt_avail = looptools_available()
    if not lt_avail:
        raise HTTPException(
            status_code=503,
            detail=(
                "LoopTools library not found. "
                "Run 'feynman install-looptools' to build it from the bundled source."
            ),
        )

    obs = observable.strip().lower()
    try:
        if obs == "photon_selfenergy":
            val = evaluate_photon_selfenergy(q_sq, m_sq)
            latex_expr = r"\Sigma_T(k^2) = \frac{\alpha}{\pi}\left[2A_0(m^2) - (4m^2-k^2)B_0(k^2;m^2,m^2)\right]"
        elif obs == "vertex_ff":
            if abs(q_sq) < 1e-12:
                raise HTTPException(status_code=422, detail="q² must be non-zero for vertex form factor.")
            val = evaluate_vertex_form_factor(q_sq, m_sq)
            latex_expr = r"\delta F_1(q^2) = \frac{\alpha}{2\pi}\left[-B_0(m^2;0,m^2) + \frac{4m^2-q^2/2}{q^2}C_0(m^2,m^2,q^2;0,m^2,m^2)\right]"
        elif obs == "schwinger_amm":
            val = evaluate_schwinger_amm()
            latex_expr = r"a_e = \frac{\alpha}{2\pi} \approx 1.1614 \times 10^{-3}"
        elif obs == "vacuum_polarisation":
            val = evaluate_vacuum_polarisation(q_sq, m_sq)
            latex_expr = r"\Pi(q^2) = \frac{\alpha}{3\pi}\left[B_0(0;m^2,m^2) - B_0(q^2;m^2,m^2)\right]"
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown observable '{observable}'. Choose from: photon_selfenergy, vertex_ff, schwinger_amm, vacuum_polarisation.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Numerical evaluation failed: {exc}")

    if val is None:
        raise HTTPException(status_code=500, detail="Numerical evaluation returned None (LoopTools error).")

    # Build individual PV integral values for observables that decompose into them.
    pv_integrals = None
    from feynman_engine.amplitudes.looptools_bridge import A0, B0, C0
    try:
        if obs == "photon_selfenergy":
            a0_val = A0(m_sq)
            b0_val = B0(q_sq, m_sq, m_sq)
            pv_integrals = [
                {
                    "integral": f"A0({m_sq})",
                    "integral_latex": f"A_0({m_sq})",
                    "value_real": float(a0_val.real) if isinstance(a0_val, complex) else float(a0_val),
                    "value_imag": float(a0_val.imag) if isinstance(a0_val, complex) else 0.0,
                    "coefficient_latex": r"\frac{2\alpha}{\pi}",
                },
                {
                    "integral": f"B0({q_sq}, {m_sq}, {m_sq})",
                    "integral_latex": f"B_0({q_sq}, {m_sq}, {m_sq})",
                    "value_real": float(b0_val.real) if isinstance(b0_val, complex) else float(b0_val),
                    "value_imag": float(b0_val.imag) if isinstance(b0_val, complex) else 0.0,
                    "coefficient_latex": r"-\frac{\alpha}{\pi}(4m^2 - k^2)",
                },
            ]
        elif obs == "vertex_ff":
            b0_val = B0(m_sq, 0.0, m_sq)
            c0_val = C0(m_sq, m_sq, q_sq, 0.0, m_sq, m_sq)
            pv_integrals = [
                {
                    "integral": f"B0({m_sq}, 0, {m_sq})",
                    "integral_latex": f"B_0({m_sq}; 0, {m_sq})",
                    "value_real": float(b0_val.real) if isinstance(b0_val, complex) else float(b0_val),
                    "value_imag": float(b0_val.imag) if isinstance(b0_val, complex) else 0.0,
                    "coefficient_latex": r"-\frac{\alpha}{2\pi}",
                },
                {
                    "integral": f"C0({m_sq}, {m_sq}, {q_sq}, 0, {m_sq}, {m_sq})",
                    "integral_latex": f"C_0({m_sq}, {m_sq}, {q_sq}; 0, {m_sq}, {m_sq})",
                    "value_real": float(c0_val.real) if isinstance(c0_val, complex) else float(c0_val),
                    "value_imag": float(c0_val.imag) if isinstance(c0_val, complex) else 0.0,
                    "coefficient_latex": r"\frac{\alpha}{2\pi}\frac{4m^2 - q^2/2}{q^2}",
                },
            ]
        elif obs == "vacuum_polarisation":
            b0_0 = B0(0.0, m_sq, m_sq)
            b0_q = B0(q_sq, m_sq, m_sq)
            pv_integrals = [
                {
                    "integral": f"B0(0, {m_sq}, {m_sq})",
                    "integral_latex": f"B_0(0; {m_sq}, {m_sq})",
                    "value_real": float(b0_0.real) if isinstance(b0_0, complex) else float(b0_0),
                    "value_imag": float(b0_0.imag) if isinstance(b0_0, complex) else 0.0,
                    "coefficient_latex": r"+\frac{\alpha}{3\pi}",
                },
                {
                    "integral": f"B0({q_sq}, {m_sq}, {m_sq})",
                    "integral_latex": f"B_0({q_sq}; {m_sq}, {m_sq})",
                    "value_real": float(b0_q.real) if isinstance(b0_q, complex) else float(b0_q),
                    "value_imag": float(b0_q.imag) if isinstance(b0_q, complex) else 0.0,
                    "coefficient_latex": r"-\frac{\alpha}{3\pi}",
                },
            ]
    except Exception:
        pass  # pv_integrals stays None if we can't compute individual values

    response = {
        "observable": obs,
        "theory": theory.upper(),
        "q_sq_gev2": q_sq,
        "m_sq_gev2": m_sq,
        "value_real": float(val.real) if isinstance(val, complex) else float(val),
        "value_imag": float(val.imag) if isinstance(val, complex) else 0.0,
        "value_abs": abs(val),
        "latex_expr": latex_expr,
        "looptools_available": True,
    }
    if pv_integrals is not None:
        response["pv_integrals"] = pv_integrals
    return response


@router.get("/amplitude/loop-curated", summary="List curated 1-loop amplitude results with symbolic PV decomposition")
def list_loop_curated():
    """Return all curated 1-loop amplitude results with their PV integral expressions.

    Each result includes ``msq_latex`` (the symbolic formula using PV integrals),
    ``integral_latex`` (the Feynman integral representation), and the symbolic
    coefficient of each PV scalar integral.
    """
    results = get_loop_curated_results()
    return [
        {
            "process": r.process,
            "theory": r.theory,
            "description": r.description,
            "msq_latex": r.msq_latex,
            "integral_latex": r.integral_latex,
            "notes": r.notes,
            "backend": r.backend,
        }
        for r in results
    ]


@router.get("/amplitude/renorm-status", summary="UV renormalisation status and available counterterms")
def get_renorm_status():
    """Return the renormalisation status and list of available counterterms."""
    return renorm_status()


@router.get("/amplitude/running-coupling", summary="Evaluate running coupling α or α_s at scale q²")
def get_running_coupling(
    coupling: str = Query(..., description="'alpha' (QED) or 'alpha_s' (QCD)"),
    q_sq:     float = Query(..., description="Renormalisation scale q² (GeV²)"),
    n_f:      int   = Query(default=5, ge=1, le=6, description="Active quark flavors (for α_s only)"),
):
    """Evaluate the 1-loop running coupling constant at scale q² (GeV²).

    For QED: α(q²) uses the electron loop contribution.
    For QCD: α_s(q²) uses the MS-bar 1-loop β function with n_f active flavors.
    Reference point: α(M_Z) ≈ 1/128, α_s(M_Z) = 0.1179 (PDG 2023).
    """
    try:
        if coupling.lower() in ("alpha", "qed"):
            val = alpha_running(q_sq)
            return {"coupling": "alpha", "q_sq_gev2": q_sq, "value": val,
                    "inverse": 1.0 / val, "description": "QED α(q²) at 1-loop"}
        elif coupling.lower() in ("alpha_s", "alphas", "qcd"):
            val = alpha_s_running(q_sq, n_f=n_f)
            return {"coupling": "alpha_s", "q_sq_gev2": q_sq, "n_f": n_f, "value": val,
                    "description": f"QCD α_s(q²) at 1-loop with n_f={n_f}"}
        else:
            raise HTTPException(status_code=422, detail="coupling must be 'alpha' or 'alpha_s'")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/amplitude/renorm-selfenergy", summary="MS-bar renormalised photon self-energy")
def get_renorm_selfenergy(
    k_sq:     float = Query(..., description="Photon virtuality k² (GeV²)"),
    m_sq:     float = Query(default=2.611e-7, description="Internal mass² (GeV²); default=m_e²"),
    mu_sq:    float = Query(default=1.0, description="Renorm. scale μ² (GeV²)"),
    theory:   str   = Query(default="QED"),
):
    """Compute the MS-bar renormalised photon self-energy Σ̂_T(k²) = Σ_T(k²) − Σ_T(0).

    Requires LoopTools to be installed (feynman install-looptools).
    """
    if not looptools_available():
        raise HTTPException(status_code=503, detail="LoopTools not available. Run 'feynman install-looptools'.")
    val = qed_renormalised_photon_selfenergy(k_sq, m_sq, mu_sq_val=mu_sq)
    if val is None:
        raise HTTPException(status_code=500, detail="Renormalised self-energy evaluation failed.")
    return {
        "k_sq_gev2": k_sq, "m_sq_gev2": m_sq, "mu_sq_gev2": mu_sq,
        "sigma_hat_real": float(val.real), "sigma_hat_imag": float(val.imag),
        "description": "MS-bar renormalised photon self-energy Σ̂_T(k²) = Σ_T(k²) − Σ_T(0)",
        "latex": r"\hat{\Sigma}_T(k^2) = \Sigma_T(k^2) - \Sigma_T(0)",
    }


@router.get(
    "/amplitude/cross-section",
    summary="Integrate |M̄|² to get total cross-section in pb",
)
def get_cross_section(
    process:  str   = Query(..., description="e.g. 'e+ e- -> mu+ mu-'"),
    theory:   str   = Query(default="QED"),
    sqrt_s:   float = Query(..., description="Centre-of-mass energy √s in GeV"),
    eps:      float = Query(default=1e-3, description="Endpoint cutoff for t-channel poles"),
    n_events: int   = Query(default=100_000, description="MC samples (for 2→N with N≥3)"),
    min_invariant_mass: float = Query(default=0.0, description="Minimum invariant mass cut in GeV (for IR-safe 2→N cross sections)"),
    method:   str   = Query(default="auto", description="Integration method: 'auto', 'rambo', or 'vegas'"),
    n_iter:   int   = Query(default=10, description="Vegas iterations (only for method=vegas)"),
    n_eval_per_iter: int = Query(default=50_000, description="Evaluations per Vegas iteration"),
    alpha:    Optional[float] = Query(default=None, description="Override α_em"),
    alpha_s:  Optional[float] = Query(default=None, description="Override α_s"),
):
    """Numerically integrate the spin-averaged |M̄|² to give σ(process) in pb.

    For 2→2 processes, uses deterministic cosθ integration (scipy.quad).
    For 2→N (N≥3), uses either flat RAMBO MC or Vegas adaptive MC sampling.

    Set method='vegas' for processes with sharp kinematic features (t-channel
    poles, resonances) where flat sampling converges slowly.
    """
    from feynman_engine.amplitudes.cross_section import (
        total_cross_section,
        total_cross_section_mc,
        total_cross_section_vegas,
    )

    coupling_overrides: dict = {}
    if alpha is not None:
        coupling_overrides["alpha"] = alpha
    if alpha_s is not None:
        coupling_overrides["alpha_s"] = alpha_s

    # Detect multiplicity to route to the right integrator.
    try:
        spec = parse_process(process.strip(), theory.upper())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    n_out = len(spec.outgoing)

    if n_out == 2 and method != "vegas":
        result = total_cross_section(
            process=process.strip(),
            theory=theory.upper(),
            sqrt_s=sqrt_s,
            coupling_vals=coupling_overrides if coupling_overrides else None,
            eps=eps,
        )
    elif method == "vegas" or (method == "auto" and n_out >= 3):
        result = total_cross_section_vegas(
            process=process.strip(),
            theory=theory.upper(),
            sqrt_s=sqrt_s,
            coupling_vals=coupling_overrides if coupling_overrides else None,
            n_iter=n_iter,
            n_eval_per_iter=n_eval_per_iter,
            min_invariant_mass=min_invariant_mass,
        )
    else:
        result = total_cross_section_mc(
            process=process.strip(),
            theory=theory.upper(),
            sqrt_s=sqrt_s,
            coupling_vals=coupling_overrides if coupling_overrides else None,
            n_events=n_events,
            min_invariant_mass=min_invariant_mass,
        )

    if not result.get("supported", False):
        raise HTTPException(
            status_code=404,
            detail=result.get("error", f"Cross-section unavailable for '{process}' in {theory}."),
        )

    return result


@router.get("/amplitude/loop-analytic",
            summary="Evaluate a PV scalar integral analytically (no LoopTools required)")
def get_loop_analytic_endpoint(
    integral_type: str = Query(
        ...,
        description="Integral type: 'A0', 'B0', 'B1', 'B00', 'C0', 'D0'",
    ),
    p_sq: Optional[float] = Query(default=None, description="External momentum squared p² (B-type, C₀, D₀)"),
    m_sq: Optional[float] = Query(default=None, description="Mass squared m² (A₀ shorthand)"),
    m1_sq: Optional[float] = Query(default=None, description="First internal mass squared m₁²"),
    m2_sq: Optional[float] = Query(default=None, description="Second internal mass squared m₂²"),
    m3_sq: Optional[float] = Query(default=None, description="Third internal mass squared m₃² (C₀, D₀)"),
    m4_sq: Optional[float] = Query(default=None, description="Fourth internal mass squared m₄² (D₀)"),
    p1_sq: Optional[float] = Query(default=None, description="First external momentum squared p₁² (C₀, D₀)"),
    p2_sq: Optional[float] = Query(default=None, description="Second external momentum squared p₂² (C₀, D₀)"),
    p12_sq: Optional[float] = Query(default=None, description="(p₁+p₂)² (C₀) or s (D₀)"),
    p3_sq: Optional[float] = Query(default=None, description="Third external momentum squared p₃² (D₀)"),
    p4_sq: Optional[float] = Query(default=None, description="Fourth external momentum squared p₄² (D₀)"),
    s: Optional[float] = Query(default=None, description="Mandelstam s (D₀)"),
    t: Optional[float] = Query(default=None, description="Mandelstam t (D₀)"),
    mu_sq: float = Query(default=1.0, description="Renormalisation scale μ² (GeV²)"),
):
    """Evaluate a Passarino-Veltman scalar integral using analytic closed-form formulas.

    **No LoopTools required** — pure Python/SymPy/mpmath evaluation.

    Returns the numerical value (with Δ_UV = 0, matching LoopTools conventions),
    the LaTeX formula for the integral, and metadata about the kinematic
    configuration (special case detected, UV/IR divergence info).

    Supported integrals:
    - **A₀(m²)**: tadpole — exact for all masses
    - **B₀(p²; m₁², m₂²)**: bubble — all 6 special cases + general
    - **B₁, B₀₀**: tensor bubble — via PV reduction identities
    - **C₀(p₁², p₂², p₁₂²; m₁², m₂², m₃²)**: triangle — Li₂ closed form for one-mass, dblquad for general spacelike
    - **D₀**: box — massless box formula only
    """
    from feynman_engine.amplitudes.analytic_integrals import (
        analytic_A0, analytic_B0, analytic_B1, analytic_B00,
        analytic_C0, analytic_D0,
    )

    itype = integral_type.strip().upper()

    try:
        if itype == "A0":
            m = m_sq if m_sq is not None else (m1_sq if m1_sq is not None else None)
            if m is None:
                raise HTTPException(status_code=422, detail="A₀ requires m_sq (or m1_sq).")
            val = analytic_A0(m, mu_sq=mu_sq)
            latex_expr = rf"A_0({m}) = m^2 \left(\Delta_{{UV}} + 1 - \ln\frac{{m^2}}{{\mu^2}}\right)"
            args_info = {"m_sq": m}

        elif itype == "B0":
            if p_sq is None or m1_sq is None or m2_sq is None:
                raise HTTPException(status_code=422, detail="B₀ requires p_sq, m1_sq, m2_sq.")
            val = analytic_B0(p_sq, m1_sq, m2_sq, mu_sq=mu_sq)
            latex_expr = rf"B_0({p_sq};\, {m1_sq},\, {m2_sq})"
            args_info = {"p_sq": p_sq, "m1_sq": m1_sq, "m2_sq": m2_sq}

        elif itype == "B1":
            if p_sq is None or m1_sq is None or m2_sq is None:
                raise HTTPException(status_code=422, detail="B₁ requires p_sq, m1_sq, m2_sq.")
            val = analytic_B1(p_sq, m1_sq, m2_sq, mu_sq=mu_sq)
            latex_expr = rf"B_1({p_sq};\, {m1_sq},\, {m2_sq})"
            args_info = {"p_sq": p_sq, "m1_sq": m1_sq, "m2_sq": m2_sq}

        elif itype == "B00":
            if p_sq is None or m1_sq is None or m2_sq is None:
                raise HTTPException(status_code=422, detail="B₀₀ requires p_sq, m1_sq, m2_sq.")
            val = analytic_B00(p_sq, m1_sq, m2_sq, mu_sq=mu_sq)
            latex_expr = rf"B_{{00}}({p_sq};\, {m1_sq},\, {m2_sq})"
            args_info = {"p_sq": p_sq, "m1_sq": m1_sq, "m2_sq": m2_sq}

        elif itype == "C0":
            _p1 = p1_sq if p1_sq is not None else 0.0
            _p2 = p2_sq if p2_sq is not None else 0.0
            _p12 = p12_sq if p12_sq is not None else (p_sq if p_sq is not None else None)
            _m1 = m1_sq if m1_sq is not None else 0.0
            _m2 = m2_sq if m2_sq is not None else 0.0
            _m3 = m3_sq if m3_sq is not None else 0.0
            if _p12 is None:
                raise HTTPException(status_code=422, detail="C₀ requires at least p12_sq (or p_sq as shorthand).")
            val = analytic_C0(_p1, _p2, _p12, _m1, _m2, _m3, mu_sq=mu_sq)
            latex_expr = rf"C_0({_p1},\, {_p2},\, {_p12};\, {_m1},\, {_m2},\, {_m3})"
            args_info = {"p1_sq": _p1, "p2_sq": _p2, "p12_sq": _p12,
                         "m1_sq": _m1, "m2_sq": _m2, "m3_sq": _m3}

        elif itype == "D0":
            _p1 = p1_sq if p1_sq is not None else 0.0
            _p2 = p2_sq if p2_sq is not None else 0.0
            _p3 = p3_sq if p3_sq is not None else 0.0
            _p4 = p4_sq if p4_sq is not None else 0.0
            _s = s if s is not None else (p12_sq if p12_sq is not None else None)
            _t = t
            if _s is None or _t is None:
                raise HTTPException(status_code=422, detail="D₀ requires s and t (or p12_sq for s).")
            _m1 = m1_sq if m1_sq is not None else 0.0
            _m2 = m2_sq if m2_sq is not None else 0.0
            _m3 = m3_sq if m3_sq is not None else 0.0
            _m4 = m4_sq if m4_sq is not None else 0.0
            val = analytic_D0(_p1, _p2, _p3, _p4, _s, _t, _m1, _m2, _m3, _m4, mu_sq=mu_sq)
            latex_expr = rf"D_0({_p1},\, {_p2},\, {_p3},\, {_p4},\, {_s},\, {_t};\, {_m1},\, {_m2},\, {_m3},\, {_m4})"
            args_info = {"p1_sq": _p1, "p2_sq": _p2, "p3_sq": _p3, "p4_sq": _p4,
                         "s": _s, "t": _t,
                         "m1_sq": _m1, "m2_sq": _m2, "m3_sq": _m3, "m4_sq": _m4}

        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown integral type '{integral_type}'. Choose from: A0, B0, B1, B00, C0, D0.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analytic evaluation failed: {exc}")

    if val is None:
        return {
            "integral_type": itype,
            "arguments": args_info,
            "mu_sq": mu_sq,
            "value_real": None,
            "value_imag": None,
            "latex": latex_expr,
            "supported": False,
            "note": "This kinematic configuration is not supported by the analytic formulas. Use LoopTools for numerical evaluation.",
            "looptools_available": looptools_available(),
        }

    return {
        "integral_type": itype,
        "arguments": args_info,
        "mu_sq": mu_sq,
        "value_real": float(val.real) if isinstance(val, complex) else float(val),
        "value_imag": float(val.imag) if isinstance(val, complex) else 0.0,
        "value_abs": abs(val),
        "latex": latex_expr,
        "supported": True,
        "note": "Evaluated using analytic closed-form formulas (no LoopTools required). Delta_UV = 0, matching LoopTools conventions.",
    }


@router.get("/amplitude/processes", summary="List processes with pre-computed amplitudes")
def list_amplitude_processes():
    return list_supported_processes()
