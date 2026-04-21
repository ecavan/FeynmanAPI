"""FastAPI application factory."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from feynman_engine.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="FeynmanEngine API",
        description="Generate and render Feynman diagrams for high-energy physics processes.",
        version="0.1.1",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all origins in development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # Prefer the packaged frontend so `pip install feynman-engine` serves the UI.
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()
