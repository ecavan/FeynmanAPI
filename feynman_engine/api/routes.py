"""FastAPI route definitions."""
from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import Response

from feynman_engine.render.compiler import tikz_to_pdf, RenderError, MissingDependencyError
from feynman_engine.api.schemas import (
    AmplitudeResponse,
    DescribeResponse,
    DiagramResponse,
    GenerateRequest,
    GenerateResponse,
    ParticleResponse,
)
from feynman_engine.engine import FeynmanEngine
from feynman_engine.physics.registry import TheoryRegistry
from feynman_engine.physics.amplitude import get_amplitude, list_supported_processes

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
            particle_type=p.particle_type.value,
            mass=p.mass, charge=p.charge,
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

    _last_result_store.clear()
    for d in result.diagrams:
        _last_result_store[d.id] = {
            "diagram": d,
            "tikz": result.tikz_code.get(d.id),
            "image": result.images.get(d.id),
            "format": request.output_format,
        }

    diagram_responses = []
    for d in result.diagrams:
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

    return GenerateResponse(
        diagrams=diagram_responses,
        summary=result.summary,
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
            summary="Get tree-level spin-averaged |M|² for a process")
def get_amplitude_endpoint(
    process: str = Query(..., description="e.g. 'e+ e- -> mu+ mu-'"),
    theory:  str = Query(default="QED"),
):
    result = get_amplitude(process.strip(), theory.upper())
    if result is None:
        # Return what processes are supported
        supported = [p["process"] for p in list_supported_processes()]
        raise HTTPException(
            status_code=404,
            detail=f"No pre-computed amplitude for '{process}' in {theory}. "
                   f"Supported: {supported}",
        )
    return AmplitudeResponse(
        process=result.process,
        theory=result.theory,
        description=result.description,
        msq_latex=result.msq_latex,
        msq_sympy=str(result.msq),
        notes=result.notes,
        supported=True,
    )


@router.get("/amplitude/processes", summary="List processes with pre-computed amplitudes")
def list_amplitude_processes():
    return list_supported_processes()
