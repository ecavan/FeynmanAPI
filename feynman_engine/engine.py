"""
FeynmanEngine — top-level public API.

Usage:
    from feynman_engine import FeynmanEngine

    engine = FeynmanEngine()
    result = engine.generate("e+ e- -> mu+ mu-", theory="QED", loops=0)
    print(result.summary)
"""
from __future__ import annotations

import hashlib
import time
from typing import Literal

from feynman_engine.core.generator import generate_diagrams, backend_name, qgraf_available
from feynman_engine.core.models import Diagram, GenerationResult
from feynman_engine.physics.registry import TheoryRegistry
from feynman_engine.physics.translator import parse_process
from feynman_engine.qgraf import qgraf_source_available
from feynman_engine.render.compiler import compile_all
from feynman_engine.render.tikz import diagrams_to_tikz


OutputFormat = Literal["svg", "png", "tikz", "pdf"]

# In-memory cache keyed by (process, theory, loops) SHA-256
_cache: dict[str, GenerationResult] = {}


def _cache_key(process: str, theory: str, loops: int) -> str:
    raw = f"{process.strip()}|{theory.upper()}|{loops}"
    return hashlib.sha256(raw.encode()).hexdigest()


class FeynmanEngine:
    """
    High-level API for generating and rendering Feynman diagrams.

    QGRAF is required for all diagram generation in this project.
    """

    def generate(
        self,
        process: str,
        theory: str = "QED",
        loops: int = 0,
        output_format: OutputFormat = "svg",
        use_cache: bool = True,
        filters: dict | None = None,
    ) -> GenerationResult:
        """
        Generate all Feynman diagrams for a scattering process.

        Args:
            process:       e.g. "e+ e- -> mu+ mu-"
            theory:        "QED", "QCD", or "EW"
            loops:         0 = tree-level
                           1+ = loop-level
            output_format: "svg"   — rendered image (needs lualatex + pdf2svg)
                           "tikz"  — raw LaTeX source (no extra tools needed)
                           "png" / "pdf" — needs lualatex + pdf2svg

        Returns:
            GenerationResult with .diagrams, .images, .tikz_code, .summary, .metadata
        """
        key = _cache_key(process, theory, loops)
        if use_cache and key in _cache and not filters:
            return _cache[key]

        t_start = time.monotonic()

        spec = parse_process(process, theory, loops)
        diagrams = generate_diagrams(spec, filters=filters)

        tikz_codes = diagrams_to_tikz(diagrams)

        images: dict[int, bytes] = {}
        if output_format in ("svg", "png", "pdf"):
            images = compile_all(tikz_codes)

        elapsed = time.monotonic() - t_start

        topology_counts: dict[str, int] = {}
        for d in diagrams:
            top = d.topology or "unknown"
            topology_counts[top] = topology_counts.get(top, 0) + 1

        result = GenerationResult(
            diagrams=diagrams,
            images=images,
            tikz_code=tikz_codes,
            summary={
                "total_diagrams": len(diagrams),
                "topology_counts": topology_counts,
                "loop_order": loops,
            },
            metadata={
                "process": process,
                "theory": theory,
                "loops": loops,
                "output_format": output_format,
                "elapsed_seconds": round(elapsed, 3),
                "backend": backend_name(),
            },
        )

        if use_cache:
            _cache[key] = result
        return result

    # ── Convenience methods ────────────────────────────────────────────────

    def list_theories(self) -> list[str]:
        return TheoryRegistry.list_theories()

    def list_particles(self, theory: str) -> list[dict]:
        return [p.model_dump() for p in TheoryRegistry.get_particles(theory).values()]

    def describe_process(self, process: str, theory: str = "QED") -> dict:
        """Validate a process string and return particle info without generating diagrams."""
        spec = parse_process(process, theory, loops=0)
        registry = TheoryRegistry.get_particles(theory)
        return {
            "valid": True,
            "process": spec.raw,
            "theory": spec.theory,
            "incoming": [registry[p].model_dump() for p in spec.incoming],
            "outgoing": [registry[p].model_dump() for p in spec.outgoing],
        }

    def status(self) -> dict:
        """Return backend and dependency status."""
        import shutil
        from pathlib import Path

        lualatex_paths = [
            "/usr/local/texlive/2026basic/bin/universal-darwin/lualatex",
            "/Library/TeX/texbin/lualatex",
        ]
        lualatex_found = any(Path(p).exists() for p in lualatex_paths) or bool(shutil.which("lualatex"))

        from feynman_engine.amplitudes.looptools_bridge import is_available as _lt_avail
        return {
            "backend": backend_name(),
            "qgraf_available": qgraf_available(),
            "qgraf_source_available": qgraf_source_available(),
            "lualatex_available": lualatex_found,
            "pdf2svg_available": bool(shutil.which("pdf2svg")),
            "looptools_available": _lt_avail(),
            "theories": self.list_theories(),
        }
