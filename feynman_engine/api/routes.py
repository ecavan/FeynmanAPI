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

    # Cross-theory loop-induced processes (e.g. g g → H, H → γγ) live in
    # the curated 1-loop registry under whatever theory was used to register
    # them, but the requested theory's particle list may not contain all
    # the legs (e.g. EW model has H but no g; QCD has g but no H).  Try
    # the curated lookup first to bypass parse_process for these cases.
    if loops == 1:
        from feynman_engine.amplitudes.loop_curated import get_loop_curated_amplitude
        for theory_try in (theory_upper, "EW", "QCD", "QED"):
            curated = get_loop_curated_amplitude(process_clean, theory_try)
            if curated is not None and curated.msq not in (None, 0):
                # Skip parse_process; return curated directly.
                integral_latex = getattr(curated, "integral_latex", None)
                return AmplitudeResponse(
                    process=curated.process,
                    theory=curated.theory,
                    loops=loops,
                    description=curated.description or "1-loop curated amplitude",
                    msq_latex=curated.msq_latex or "",
                    msq_sympy=str(curated.msq) if curated.msq is not None else "",
                    integral_latex=integral_latex,
                    notes=curated.notes or "",
                    backend=curated.backend or "curated-1loop",
                    supported=True,
                    has_msq=True,
                    has_integral=integral_latex is not None,
                    is_symbolic_function=True,
                    features={},
                    approximation_level=getattr(curated, "approximation_level", "exact-symbolic"),
                    evaluation_point=None,
                )

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
            # The integral expression comes from QGRAF diagram generation,
            # which can fail for cross-theory processes (e.g. bb̄→γγ in pure
            # QCD model has no photon vertex).  The curated |M̄|² is still
            # valid; just omit the integral expression in that case.
            try:
                integral_latex = (
                    get_tree_integral_latex(process_clean, theory_upper)
                    if loops == 0
                    else get_loop_integral_latex(process_clean, theory_upper, loops=loops)
                )
            except Exception:
                integral_latex = None
        has_msq = result.msq is not None
        features = getattr(result, "features", None) or {}

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
            is_symbolic_function=True,
            features=features,
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
    order:    str   = Query(default="LO", description="Perturbative order: 'LO' or 'NLO'"),
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

    Set order='NLO' to compute the NLO cross-section.  For QED e+e-→ff̄
    (different flavor) uses the exact analytic K-factor K = 1 + 3α/(4π).
    For all other 2→N processes (QED, QCD, EW), uses the running-coupling
    approximation: α(Q²)/α(μ₀²) evaluated at Q² = s via the 1-loop
    β-function (QCD) or vacuum polarisation (QED).  The coupling power is
    auto-detected from the symbolic amplitude.

    Set method='vegas' for processes with sharp kinematic features (t-channel
    poles, resonances) where flat sampling converges slowly.

    Every result includes a ``trust_level`` field
    (validated/approximate/rough) plus a ``trust_reference`` describing
    where the number was anchored.  Processes known to give wrong answers
    are refused with HTTP 422 rather than returning a misleading number.
    """
    from feynman_engine.physics.trust import classify, TrustLevel, trust_payload

    # For pp processes, classify against the actual hadronic theory (auto-
    # detect from final state) instead of the user-passed theory, which
    # often defaults to QED for hadronic queries.
    process_lower = process.strip().lower()
    is_hadronic = process_lower.startswith("p p ->") or process_lower.startswith("p p->")
    if is_hadronic:
        # Probe the hadronic registry directly; auto-detect theory if needed.
        from feynman_engine.amplitudes.hadronic import _detect_partonic_theory
        final_state = process.strip().split("->", 1)[1].strip() if "->" in process else ""
        trust_theory = (theory.upper() if theory and theory.upper() != "QED"
                        else _detect_partonic_theory(final_state))
    else:
        trust_theory = theory

    # ── Trust gate ─────────────────────────────────────────────────────
    trust_entry = classify(process.strip(), trust_theory, order)
    if trust_entry.trust_level == TrustLevel.BLOCKED:
        raise HTTPException(
            status_code=422,
            detail={
                "process": process.strip(),
                "theory": trust_theory.upper() if trust_theory else theory.upper(),
                "order": order.upper(),
                "trust_level": "blocked",
                "block_reason": trust_entry.block_reason,
                "workaround": trust_entry.workaround,
                "reference": trust_entry.reference,
            },
        )

    # ── Hadronic route (detect "p p ->" processes) ────────────────────
    if is_hadronic:
        from feynman_engine.amplitudes.hadronic import hadronic_cross_section
        result = hadronic_cross_section(
            process=process.strip(),
            sqrt_s=sqrt_s,
            theory=theory.upper(),
            order=order.upper(),
            m_ll_min=min_invariant_mass if min_invariant_mass > 0 else 60.0,
            m_ll_max=120.0,
        )
        if not result.get("supported", False):
            raise HTTPException(
                status_code=404,
                detail=result.get("error", f"Hadronic cross-section unavailable for '{process}'."),
            )
        result.update(trust_payload(trust_entry))
        return result

    # ── NLO route ───────────────────────────────────────────────────────
    if order.upper() == "NLO":
        from feynman_engine.amplitudes.nlo_cross_section import nlo_cross_section

        result = nlo_cross_section(
            process=process.strip(),
            theory=theory.upper(),
            sqrt_s=sqrt_s,
            n_events=n_events,
            min_invariant_mass=min_invariant_mass,
        )
        if not result.get("supported", False):
            raise HTTPException(
                status_code=404,
                detail=result.get("error", f"NLO cross-section unavailable for '{process}' in {theory}."),
            )
        result.update(trust_payload(trust_entry))
        return result

    # ── LO route (existing) ─────────────────────────────────────────────
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
        err = result.get("error", f"Cross-section unavailable for '{process}' in {theory}.")
        # Below-threshold and non-positive √s are kinematic input errors → 422.
        # Missing-|M|² / unregistered process → 404 (resource not found).
        is_kinematic_error = (
            "below the production threshold" in err
            or "must be positive" in err
            or "must be > 0" in err
        )
        raise HTTPException(
            status_code=422 if is_kinematic_error else 404,
            detail=err,
        )

    result.update(trust_payload(trust_entry))
    return result


# ──────────────────────────────────────────────────────────────────────────
# Decay-width endpoint
# ──────────────────────────────────────────────────────────────────────────
#
# 2-body decay rate for X → 1 + 2:
#     Γ = |M̄|² · |p|/(8π M²)
# where |p| = √λ(M², m₁², m₂²) / (2M) is the daughter momentum in the
# parent rest frame and λ is the Källén function.  For decays into
# identical particles (e.g. H → γγ) divide by an extra 1/2.
#
# PDG 2024 reference values are baked in for the major SM channels so the
# response can show "X% off PDG" alongside the engine result.

# (process, theory) → (PDG width in MeV, source label)
_PDG_DECAY_WIDTHS_MEV: dict[tuple[str, str], tuple[float, str]] = {
    # Z decays — PDG 2024 partial widths
    ("Z -> e+ e-",      "EW"): (83.91,  "PDG 2024 Γ(Z→e+e-)"),
    ("Z -> mu+ mu-",    "EW"): (83.99,  "PDG 2024 Γ(Z→μ+μ-)"),
    ("Z -> tau+ tau-",  "EW"): (84.08,  "PDG 2024 Γ(Z→τ+τ-)"),
    ("Z -> u u~",       "EW"): (299.0,  "PDG 2024 Γ(Z→uū) (per quark colour summed)"),
    ("Z -> c c~",       "EW"): (299.0,  "PDG 2024 Γ(Z→cc̄)"),
    ("Z -> d d~",       "EW"): (382.9,  "PDG 2024 Γ(Z→dd̄)"),
    ("Z -> s s~",       "EW"): (382.9,  "PDG 2024 Γ(Z→ss̄)"),
    ("Z -> b b~",       "EW"): (375.4,  "PDG 2024 Γ(Z→bb̄) (slightly suppressed by m_b)"),
    # Higgs decays — PDG 2024 / LHC HWG YR4
    ("H -> b b~",       "EW"): (2.41,   "PDG 2024 Γ(H→bb̄)"),
    ("H -> tau+ tau-",  "EW"): (0.260,  "PDG 2024 Γ(H→τ+τ-)"),
    ("H -> c c~",       "EW"): (0.117,  "PDG 2024 Γ(H→cc̄)"),
    ("H -> mu+ mu-",    "EW"): (8.93e-4,"PDG 2024 Γ(H→μ+μ-)"),
    # Top decay (1→2 width, dominated by t→bW)
    ("t -> b W+",       "EW"): (1420.0, "PDG 2024 Γ(t→bW) ≈ 1.42 GeV"),
    # W decays — PDG 2024 partial widths (BR ≈ 10.86% per leptonic channel)
    ("W+ -> e+ nu_e",       "EW"): (226.5,  "PDG 2024 Γ(W→eν) = 226.5 MeV"),
    ("W- -> e- nu_e~",      "EW"): (226.5,  "PDG 2024 Γ(W→eν) = 226.5 MeV"),
    ("W+ -> mu+ nu_mu",     "EW"): (226.5,  "PDG 2024 Γ(W→μν) = 226.5 MeV"),
    ("W- -> mu- nu_mu~",    "EW"): (226.5,  "PDG 2024 Γ(W→μν) = 226.5 MeV"),
    ("W+ -> tau+ nu_tau",   "EW"): (226.0,  "PDG 2024 Γ(W→τν) = 226.0 MeV (tiny m_τ effect)"),
    ("W- -> tau- nu_tau~",  "EW"): (226.0,  "PDG 2024 Γ(W→τν) = 226.0 MeV (tiny m_τ effect)"),
}

# (parent, theory) → total width MeV.  Used to compute branching ratios.
_PDG_TOTAL_WIDTHS_MEV: dict[tuple[str, str], float] = {
    ("Z", "EW"): 2495.2,    # PDG 2024 Γ_Z = 2.4952 GeV
    ("H", "EW"): 4.10,      # SM Higgs total width at m_H = 125 GeV
    ("W+", "EW"): 2085.0,   # PDG 2024 Γ_W = 2.085 GeV
    ("W-", "EW"): 2085.0,
    ("t", "EW"): 1420.0,    # ≈ Γ(t→bW)
}


@router.get(
    "/amplitude/decay-width",
    summary="Compute Γ(X → AB) from the curated |M̄|² and 2-body kinematics",
)
def get_decay_width(
    process: str = Query(..., description="Decay process, e.g. 'Z -> e+ e-' or 'H -> b b~'"),
    theory:  str = Query(default="EW"),
):
    """Return the 2-body decay width Γ(X → 1 + 2) in MeV using

        Γ = (1/2!^id) · |M̄|² · |p|/(8π M²)

    where |p| is the daughter 3-momentum in the parent rest frame and the
    1/2! factor applies for identical daughters.  |M̄|² comes from the
    curated ``form-decay`` backend (with full V-A structure for Z and
    Yukawa structure for Higgs) using engine-tuned coupling defaults
    (``g_Z_e`` = 0.180, ``y_b`` evaluated at m_H scale, etc.).

    For SM channels in the PDG comparison table, the response also
    includes ``pdg_width_mev`` and the percent deviation, plus a branching
    ratio when the parent's total width is known.
    """
    import math
    from feynman_engine.physics.amplitude import get_amplitude
    from feynman_engine.physics.translator import parse_process
    from feynman_engine.amplitudes.cross_section import _build_coupling_defaults
    from feynman_engine.amplitudes.pdg_masses import MASS_GEV

    proc_clean = process.strip()
    theory_upper = theory.upper()

    try:
        spec = parse_process(proc_clean, theory=theory_upper)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse '{proc_clean}': {exc}")
    if len(spec.incoming) != 1 or len(spec.outgoing) != 2:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Decay-width endpoint requires a 1→2 process (got "
                f"{len(spec.incoming)}→{len(spec.outgoing)}).  "
                "Use /api/amplitude/cross-section for 2→N scattering."
            ),
        )

    parent_name = spec.incoming[0]
    daughter_names = list(spec.outgoing)

    # Parent and daughter masses
    def _mass_for(particle: str) -> float:
        # PDG masses keyed as 'm_<flavor>' or 'm_<boson>'
        candidates = (
            f"m_{particle.replace('+', '').replace('-', '').replace('~', '')}",
            f"m_{particle.lower()}",
        )
        for k in candidates:
            if k in MASS_GEV:
                return MASS_GEV[k]
        return 0.0

    M_parent = _mass_for(parent_name)
    if M_parent <= 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Could not look up parent mass for '{parent_name}'.  "
                "Decay-width computation needs a known parent mass."
            ),
        )
    m1 = _mass_for(daughter_names[0])
    m2 = _mass_for(daughter_names[1])

    # Need the |M̄|² formula
    amp = get_amplitude(proc_clean, theory_upper)
    if amp is None or amp.msq is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No |M̄|² available for '{proc_clean}' in {theory_upper}.  "
                "Decay-width computation requires a curated form-decay entry."
            ),
        )

    # Substitute defaults for couplings + masses
    defaults = _build_coupling_defaults(theory_upper)
    try:
        free_syms = list(getattr(amp.msq, "free_symbols", []))
        sub_map = {sym: defaults[sym.name] for sym in free_syms if sym.name in defaults}
        msq_val = float(amp.msq.subs(sub_map))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to numerically evaluate |M̄|² for '{proc_clean}': "
                f"{type(exc).__name__}: {exc}.  Likely a missing coupling default."
            ),
        )

    # Kinematic threshold
    if M_parent <= m1 + m2:
        return {
            "process": proc_clean, "theory": theory_upper,
            "parent": parent_name, "daughters": daughter_names,
            "parent_mass_gev": M_parent,
            "daughter_masses_gev": [m1, m2],
            "msq_value": msq_val, "msq_latex": amp.msq_latex or "",
            "width_gev": 0.0, "width_mev": 0.0,
            "branching_ratio": 0.0,
            "pdg_width_mev": None, "pct_off_pdg": None,
            "backend": amp.backend,
            "notes": (
                f"Kinematically forbidden: M_parent = {M_parent:.3f} GeV < "
                f"m1 + m2 = {m1+m2:.3f} GeV."
            ),
            "trust_level": "validated",
            "trust_reference": "Kinematic threshold",
        }

    # 2-body kinematics: |p|/M = √λ(1, m1²/M², m2²/M²) / 2
    lam = (M_parent**2 - (m1 + m2) ** 2) * (M_parent**2 - (m1 - m2) ** 2)
    p_daughter = math.sqrt(max(lam, 0.0)) / (2.0 * M_parent)

    # Identical-particle factor (½ for X → AA)
    identical = 1.0 if daughter_names[0] != daughter_names[1] else 0.5
    width_gev = identical * msq_val * p_daughter / (8.0 * math.pi * M_parent**2)
    width_mev = width_gev * 1000.0

    # PDG comparison if available
    pdg_entry = _PDG_DECAY_WIDTHS_MEV.get((proc_clean, theory_upper))
    pdg_width_mev: Optional[float] = None
    trust_reference: Optional[str] = None
    pct_off: Optional[float] = None
    trust_level = "approximate"
    if pdg_entry is not None:
        pdg_width_mev, trust_reference = pdg_entry
        if pdg_width_mev > 0:
            pct_off = abs(width_mev - pdg_width_mev) / pdg_width_mev * 100.0
            if pct_off < 5.0:
                trust_level = "validated"
            elif pct_off < 30.0:
                trust_level = "approximate"
            else:
                trust_level = "rough"

    # Branching ratio if parent total width is known
    total_width_mev = _PDG_TOTAL_WIDTHS_MEV.get((parent_name, theory_upper))
    branching_ratio: Optional[float] = None
    if total_width_mev and total_width_mev > 0:
        branching_ratio = width_mev / total_width_mev

    accuracy_caveat = None
    # Both `form-decay` (auto-generated trace) and the explicit `curated`
    # W±/Z formulas use the V-A approximation with engine-tuned effective
    # couplings — same disclosure applies.
    if amp.backend in ("form-decay", "curated") and parent_name in (
        "Z", "W+", "W-", "H"
    ):
        accuracy_caveat = (
            "Z and W decays use the V-A approximation via engine-tuned "
            "effective couplings (typically ~7-10% PDG accuracy).  Higgs "
            "decays use MS-bar quark masses at the Higgs scale "
            "(m_b(m_H) ≈ 2.95 GeV) — within ~10% of PDG for H→bb̄/cc̄ and "
            "exact for leptonic channels (no QCD running)."
        )

    return {
        "process": proc_clean,
        "theory": theory_upper,
        "parent": parent_name,
        "daughters": daughter_names,
        "parent_mass_gev": M_parent,
        "daughter_masses_gev": [m1, m2],
        "msq_value": msq_val,
        "msq_latex": amp.msq_latex or "",
        "width_gev": width_gev,
        "width_mev": width_mev,
        "branching_ratio": branching_ratio,
        "pdg_width_mev": pdg_width_mev,
        "pct_off_pdg": pct_off,
        "backend": amp.backend,
        "notes": amp.notes or "",
        "trust_level": trust_level,
        "trust_reference": trust_reference,
        "accuracy_caveat": accuracy_caveat,
    }


@router.get(
    "/amplitude/openloops-virtual-k",
    summary="Virtual NLO K-factor via OpenLoops (per phase-space point)",
)
def get_openloops_virtual_k(
    process: str = Query(..., description="Process e.g. 'u u~ -> e+ e-'"),
    sqrt_s:  float = Query(..., description="Centre-of-mass energy in GeV"),
    theory:  str = Query(default="QCD", description="Used to pick α_s vs α_em prefactor"),
):
    """Evaluate the NLO virtual K-factor via OpenLoops 2 at one phase-space point.

    Returns ``K_virt = 1 + (α/(2π)) · loop_finite/tree`` plus the raw
    pieces (tree, loop_finite, loop_ir1, loop_ir2).  The IR poles
    (``loop_ir1``, ``loop_ir2``) are returned for cross-checking against
    the universal Catani-Seymour I-operator — they should cancel
    analytically once full CS subtraction is wired up in Phase 2.

    Requires OpenLoops 2 to be installed AND the relevant process library
    (e.g. ``ppllj`` for Drell-Yan) to be downloaded via
    ``feynman install-process``.
    """
    from feynman_engine.amplitudes.nlo_cross_section_openloops import (
        virtual_k_factor_openloops,
    )
    from feynman_engine.amplitudes.openloops_bridge import (
        is_available as openloops_available,
        installed_processes,
    )

    if not openloops_available():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OpenLoops 2 not installed.",
                "workaround": (
                    "Install via `feynman install-openloops` (requires gfortran). "
                    "Then `feynman install-process ppllj` (or another process "
                    "library) for the relevant amplitude class."
                ),
            },
        )

    try:
        result = virtual_k_factor_openloops(process.strip(), float(sqrt_s), theory=theory)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"OpenLoops evaluation failed: {type(exc).__name__}: {exc}",
                "installed_process_libraries": installed_processes(),
                "hint": (
                    "Make sure the OpenLoops process library covering this "
                    "channel is installed.  e.g. 'ppllj' covers all "
                    "Drell-Yan partonic channels (qq̄ → l+l-)."
                ),
            },
        )
    if not result.get("supported", False):
        raise HTTPException(
            status_code=422,
            detail=result,
        )
    return result


@router.get(
    "/amplitude/nlo-general",
    summary="Generic NLO σ via OpenLoops + Catani-Seymour subtraction (V2 prototype)",
)
def get_nlo_general(
    process: str = Query(..., description="Born process e.g. 'u u~ -> e+ e-'"),
    sqrt_s:  float = Query(..., description="Partonic centre-of-mass energy in GeV"),
    n_events: int = Query(default=5000, description="MC samples for the real-emission integral"),
    alpha_s: float = Query(default=0.118, description="α_s value"),
):
    """Compute σ_NLO for an arbitrary Born process using:

      1. OpenLoops 2 for the virtual amplitude V (already validated)
      2. OpenLoops 2 for the real-emission tree |R|²
      3. In-house Catani-Seymour dipoles for the local subtraction
      4. In-house analytic CS I-operator for the integrated dipoles

    Status (V2 first cut): the framework works end-to-end and gives
    K_partonic ≈ 1.01-1.04 for Drell-Yan-like processes near the Z peak.
    Off-resonance and at high energies the MC variance dominates; this is
    addressed in V2.1 with Vegas adaptive sampling + min-pT cuts.

    Currently supports 2→2 colour-neutral final states with QCD
    radiation (q q̄ → l+l-, q q̄ → V, ...).  Multi-jet + multi-coloured
    final states require colour-decomposed Born amplitudes (Phase 2.2+).
    """
    from feynman_engine.amplitudes.nlo_general import (
        nlo_cross_section_general, make_openloops_born_callback,
    )
    from feynman_engine.amplitudes.openloops_bridge import is_available
    from feynman_engine.amplitudes.phase_space import rambo_massless

    if not is_available():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OpenLoops 2 not installed.",
                "workaround": "Install via `feynman install-openloops`.",
            },
        )

    proc_clean = process.strip()

    # 1. Born |M|² callback via OpenLoops (consistent with R)
    try:
        born_callback = make_openloops_born_callback(proc_clean)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"OpenLoops Born registration failed: {type(exc).__name__}: {exc}",
                "hint": (
                    "Make sure the OpenLoops process library covering this Born "
                    "is installed (`feynman install-process <name>`)."
                ),
            },
        )

    # 2. Born σ via RAMBO 2-body integration
    import math
    import numpy as np
    GEV2_TO_PB = 0.3893793721e9
    n_born = max(2000, n_events // 2)
    fm, w = rambo_massless(n_final=2, sqrt_s=sqrt_s, n_events=n_born)
    E_beam = sqrt_s / 2.0
    p_a = np.broadcast_to(np.array([E_beam, 0, 0,  E_beam]), (n_born, 4)).copy()
    p_b = np.broadcast_to(np.array([E_beam, 0, 0, -E_beam]), (n_born, 4)).copy()
    born_msq = born_callback(p_a, p_b, [fm[:, 0], fm[:, 1]])
    sigma_per_event = (1.0 / (2.0 * sqrt_s ** 2)) * born_msq * w * GEV2_TO_PB
    sigma_born_pb = float(sigma_per_event.mean())
    sigma_born_err = float(sigma_per_event.std() / math.sqrt(n_born))

    # 3. Full NLO via the generic CS subtraction pipeline
    try:
        result = nlo_cross_section_general(
            born_process=proc_clean,
            sqrt_s_gev=sqrt_s,
            born_msq_callback=born_callback,
            sigma_born_pb=sigma_born_pb,
            n_events_real=n_events,
            alpha_s=alpha_s,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"Generic NLO computation failed: {type(exc).__name__}: {exc}",
            },
        )

    return {
        "process": result.process,
        "sqrt_s_gev": result.sqrt_s_gev,
        "sigma_born_pb": result.sigma_born_pb,
        "sigma_born_uncertainty_pb": sigma_born_err,
        "sigma_virtual_plus_idipole_pb": result.sigma_virtual_plus_idipole_pb,
        "sigma_real_minus_dipoles_pb": result.sigma_real_minus_dipoles_pb,
        "sigma_nlo_pb": result.sigma_nlo_pb,
        "k_factor": result.k_factor,
        "method": result.method,
        "trust_level": result.trust_level,
        "accuracy_caveat": result.accuracy_caveat,
        "notes": result.notes,
        "n_events_real": n_events,
        "alpha_s": alpha_s,
    }


@router.get(
    "/amplitude/openloops-loop-induced",
    summary="Loop-induced |M_loop|² via OpenLoops (gg→H, gg→ZZ, gg→HH, …)",
)
def get_openloops_loop_induced(
    process: str = Query(..., description="Process e.g. 'g g -> H' or 'g g -> Z Z'"),
    sqrt_s:  float = Query(..., description="Centre-of-mass energy in GeV"),
):
    """Evaluate a loop-induced amplitude (Born-tree = 0) via OpenLoops 2.

    For processes like ``g g → H`` (heavy-top loop), ``g g → Z Z``,
    ``g g → H H`` the leading order *is* the 1-loop amplitude, so the
    virtual-K-factor concept doesn't apply.  This endpoint returns the
    raw |M_loop|² (and its IR pieces, which should be zero for true
    loop-induced processes — non-zero values flag a numerical issue).

    Requires the relevant OpenLoops process library: ``pph`` for
    gg→H, ``ppvv`` for gg→VV, ``pphh`` for gg→HH.
    """
    from feynman_engine.amplitudes.nlo_cross_section_openloops import (
        virtual_k_factor_openloops,
    )
    from feynman_engine.amplitudes.openloops_bridge import (
        is_available as openloops_available,
        installed_processes,
    )

    if not openloops_available():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OpenLoops 2 not installed.",
                "workaround": "Install via `feynman install-openloops`.",
            },
        )

    try:
        result = virtual_k_factor_openloops(process.strip(), float(sqrt_s), theory="QCD")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"OpenLoops evaluation failed: {type(exc).__name__}: {exc}",
                "installed_process_libraries": installed_processes(),
            },
        )

    if not result.get("supported", False):
        raise HTTPException(status_code=422, detail=result)

    # Flag whether this is genuinely loop-induced (tree ≈ 0)
    if result.get("k_factor") is None:
        # tree was 0 → loop-induced result already structured by the helper
        return result
    # Tree ≠ 0 → the user asked for a non-loop-induced process via this
    # endpoint.  Return the data anyway with a warning.
    result["warning"] = (
        f"Process '{process}' has a non-zero Born tree amplitude "
        f"(tree = {result.get('tree'):.3g}); this is not a loop-induced "
        "process.  Use /amplitude/openloops-virtual-k for the K-factor instead."
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


@router.get(
    "/amplitude/hadronic-cross-section",
    summary="Hadronic (pp) cross-section via PDF convolution",
)
def get_hadronic_cross_section(
    process: str = Query(..., description="e.g. 'p p -> mu+ mu-' or 'p p -> t t~'"),
    sqrt_s: float = Query(..., description="Proton-proton CM energy in GeV (e.g. 13000, 14000)"),
    theory: str = Query(
        default="",
        description="Partonic theory: 'QCD', 'QCDQED', 'EW', 'QED'. Auto-detected from final state if blank.",
    ),
    order: str = Query(default="LO", description="Perturbative order: 'LO' or 'NLO'"),
    pdf_name: str = Query(
        default="auto",
        description=(
            "PDF set: 'auto' (LHAPDF CT18LO if installed, else built-in), "
            "'LO-simple' (built-in), or any LHAPDF set name (e.g. 'NNPDF40_lo_as_01180')."
        ),
    ),
    pdf_member: int = Query(default=0, description="LHAPDF set member (0 = central)"),
    mu_f: Optional[float] = Query(default=None, description="Factorization scale in GeV (auto if blank)"),
    m_ll_min: float = Query(default=60.0, description="Min dilepton mass in GeV (Drell-Yan only)"),
    m_ll_max: float = Query(default=120.0, description="Max dilepton mass in GeV (Drell-Yan only)"),
    n_grid: int = Query(default=25, description="Grid points per channel (generic path)"),
    n_events_mc: int = Query(default=15_000, description="MC events per grid point (2→N≥3 channels)"),
    min_partonic_cm: Optional[float] = Query(
        default=None,
        description=(
            "Minimum partonic √ŝ in GeV (generic path).  Required for fully-massless "
            "final states (γγ, gg, jj, light qq̄) where σ̂ ∝ 1/ŝ is IR-divergent; "
            "defaults to 50 GeV in that case.  Ignored for DY/tt̄ specialized paths."
        ),
    ),
    min_pT: float = Query(
        default=0.0,
        description=(
            "Per-particle pT cut in GeV for 2→2 massless final states "
            "(diphoton, dijet, etc.).  Restricts the partonic angular "
            "integration to |cosθ*| < √(1−4pT²/ŝ).  LHC photon analyses "
            "typically use pT > 20-30 GeV."
        ),
    ),
):
    """Compute a hadronic (proton-proton) cross-section by convolving partonic
    cross-sections with parton distribution functions.

    Three execution paths:

    - **Drell-Yan** (`p p -> mu+ mu-`, `p p -> e+ e-`, `p p -> tau+ tau-`) —
      analytic γ\\*/Z partonic σ̂ integrated over M_ll.
    - **Top pairs** (`p p -> t t~`) — gg + qq̄ partonic σ̂ grid.
    - **Generic** (any other `p p -> F`) — enumerates every parton channel
      (a, b) for which an amplitude exists, builds σ̂(√ŝ) grids, and convolves.
      Channels without an amplitude are silently skipped; the function returns
      ``supported=False`` (404) only when no channel works at all.

    PDFs default to LHAPDF CT18LO if the bindings are installed, otherwise the
    built-in LO-simple parametrization.  Use ``order='NLO'`` to apply running-
    coupling rescaling of σ̂.
    """
    from feynman_engine.amplitudes.hadronic import hadronic_cross_section

    result = hadronic_cross_section(
        process=process.strip(),
        sqrt_s=sqrt_s,
        theory=theory.upper() if theory else None,
        pdf_name=pdf_name,
        pdf_member=pdf_member,
        mu_f=mu_f,
        order=order.upper(),
        m_ll_min=m_ll_min,
        m_ll_max=m_ll_max,
        n_grid=n_grid,
        n_events_mc=n_events_mc,
        min_partonic_cm=min_partonic_cm,
        min_pT=min_pT,
    )
    if not result.get("supported", False):
        raise HTTPException(
            status_code=404,
            detail=result.get("error", f"Hadronic cross-section unavailable for '{process}'."),
        )
    return result


@router.get(
    "/amplitude/differential-distribution",
    summary="Differential cross-section histogram dσ/dX",
)
def get_differential_distribution(
    process: str = Query(..., description="e.g. 'e+ e- -> mu+ mu-' or 'p p -> mu+ mu-'"),
    sqrt_s: float = Query(..., description="Centre-of-mass energy in GeV"),
    observable: str = Query(
        ...,
        description=(
            "One of: cos_theta, pT_lepton, pT_photon, eta_lepton, y_lepton, "
            "M_inv, M_ll, DR_ll."
        ),
    ),
    bin_min: float = Query(..., description="Lower edge of histogram range"),
    bin_max: float = Query(..., description="Upper edge of histogram range"),
    n_bins: int = Query(default=20, description="Number of histogram bins"),
    theory: str = Query(default="", description="Theory (auto-detect if blank for pp)"),
    order: str = Query(
        default="LO",
        description="LO, NLO-running (rescale all bins), or NLO-subtracted (e+e-→μ+μ- only)",
    ),
    n_events: int = Query(default=100_000, description="MC samples for 2→N (ignored for 2→2 cos θ)"),
    min_invariant_mass: float = Query(
        default=1.0, description="IR cut on pairwise invariants (GeV) for 2→N",
    ),
    pdf_name: str = Query(default="auto", description="PDF set name (hadronic only)"),
    mu_f: Optional[float] = Query(default=None, description="Factorization scale (hadronic only)"),
    min_partonic_cm: Optional[float] = Query(
        default=None, description="Minimum partonic √ŝ (GeV, hadronic generic path only)",
    ),
):
    """Compute dσ/dX as a histogram in pb / unit(X).

    For ``e+ e- -> ...`` and similar partonic processes, runs the per-event
    histogram fill against the engine's |M̄|².  For ``p p -> ...`` processes,
    enumerates parton channels and convolves with PDF luminosities at each τ
    grid point.

    Returns bin edges, centers, widths, dσ/dX, MC uncertainty per bin, and
    the integrated σ_total.  Suitable for direct rendering as a bar histogram.
    """
    import numpy as np
    from feynman_engine.amplitudes.differential import (
        differential_distribution, hadronic_differential_distribution,
    )

    bin_edges = np.linspace(bin_min, bin_max, n_bins + 1).tolist()
    proc_lower = process.strip().lower()
    is_hadronic = proc_lower.startswith("p p ->") or proc_lower.startswith("p p->")

    # Trust gate: refuse blocked processes here too
    from feynman_engine.physics.trust import classify, TrustLevel, trust_payload
    inferred_theory = (theory.upper() if theory else
                       ("EW" if is_hadronic else "QED"))
    trust_entry = classify(process.strip(), inferred_theory, order)
    if trust_entry.trust_level == TrustLevel.BLOCKED:
        raise HTTPException(
            status_code=422,
            detail={
                "process": process.strip(),
                "theory": inferred_theory,
                "trust_level": "blocked",
                "block_reason": trust_entry.block_reason,
                "workaround": trust_entry.workaround,
            },
        )

    if is_hadronic:
        result = hadronic_differential_distribution(
            process=process.strip(),
            sqrt_s=sqrt_s,
            observable=observable,
            bin_edges=bin_edges,
            theory=theory.upper() if theory else None,
            pdf_name=pdf_name,
            mu_f=mu_f,
            n_events=n_events,
            min_invariant_mass=min_invariant_mass,
            min_partonic_cm=min_partonic_cm,
        )
    else:
        result = differential_distribution(
            process=process.strip(),
            theory=theory.upper() if theory else "QED",
            sqrt_s=sqrt_s,
            observable=observable,
            bin_edges=bin_edges,
            order=order,
            n_events=n_events,
            min_invariant_mass=min_invariant_mass,
        )

    if not result.get("supported", False):
        raise HTTPException(
            status_code=404,
            detail=result.get("error", f"Differential unavailable for '{process}'."),
        )
    result.update(trust_payload(trust_entry))
    return result


@router.get("/amplitude/processes", summary="List processes with pre-computed amplitudes")
def list_amplitude_processes():
    return list_supported_processes()
