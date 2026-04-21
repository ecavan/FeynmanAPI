"""Shared dependency diagnostics for CLI and API status reporting."""
from __future__ import annotations

import shutil
from pathlib import Path

from feynman_engine.amplitudes.looptools_bridge import (
    find_library_path as find_looptools_library,
    is_available as looptools_available,
)
from feynman_engine.core.generator import (
    backend_name,
    qgraf_available,
    qgraf_binary_path,
)
from feynman_engine.form import find_form_binary, form_source_available
from feynman_engine.qgraf import qgraf_source_available


def _find_program(*names: str) -> Path | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _find_lualatex() -> Path | None:
    known_paths = [
        "/usr/local/texlive/2026basic/bin/universal-darwin/lualatex",
        "/Library/TeX/texbin/lualatex",
    ]
    for raw in known_paths:
        path = Path(raw)
        if path.exists():
            return path
    return _find_program("lualatex")


def collect_diagnostics() -> dict:
    qgraf_path = qgraf_binary_path()
    form_path = find_form_binary()
    looptools_path = find_looptools_library()
    lualatex_path = _find_lualatex()
    pdf2svg_path = _find_program("pdf2svg")
    gfortran_path = _find_program("gfortran")
    make_path = _find_program("make", "gmake")
    cc_path = _find_program("cc", "gcc", "clang")

    return {
        "backend": backend_name(),
        "qgraf": {
            "available": qgraf_available(),
            "binary_path": str(qgraf_path) if qgraf_path else None,
            "source_available": qgraf_source_available(),
        },
        "form": {
            "available": form_path is not None,
            "binary_path": str(form_path) if form_path else None,
            "source_available": form_source_available(),
        },
        "looptools": {
            "available": looptools_available(),
            "library_path": str(looptools_path) if looptools_path else None,
        },
        "rendering": {
            "lualatex_available": lualatex_path is not None,
            "lualatex_path": str(lualatex_path) if lualatex_path else None,
            "pdf2svg_available": pdf2svg_path is not None,
            "pdf2svg_path": str(pdf2svg_path) if pdf2svg_path else None,
        },
        "toolchain": {
            "gfortran_path": str(gfortran_path) if gfortran_path else None,
            "make_path": str(make_path) if make_path else None,
            "cc_path": str(cc_path) if cc_path else None,
        },
    }


def summarize_doctor_report(diagnostics: dict) -> str:
    lines = [
        "FeynmanEngine doctor",
        f"  Backend: {diagnostics['backend']}",
        (
            "  QGRAF: "
            f"{'ok' if diagnostics['qgraf']['available'] else 'missing'}"
            f" | binary={diagnostics['qgraf']['binary_path'] or 'not found'}"
            f" | source={'yes' if diagnostics['qgraf']['source_available'] else 'no'}"
        ),
        (
            "  FORM: "
            f"{'ok' if diagnostics['form']['available'] else 'missing'}"
            f" | binary={diagnostics['form']['binary_path'] or 'not found'}"
            f" | source={'yes' if diagnostics['form']['source_available'] else 'no'}"
        ),
        (
            "  LoopTools: "
            f"{'ok' if diagnostics['looptools']['available'] else 'missing'}"
            f" | library={diagnostics['looptools']['library_path'] or 'not found'}"
        ),
        (
            "  Rendering: "
            f"lualatex={'ok' if diagnostics['rendering']['lualatex_available'] else 'missing'}, "
            f"pdf2svg={'ok' if diagnostics['rendering']['pdf2svg_available'] else 'missing'}"
        ),
        (
            "  Toolchain: "
            f"gfortran={diagnostics['toolchain']['gfortran_path'] or 'missing'}, "
            f"make={diagnostics['toolchain']['make_path'] or 'missing'}, "
            f"cc={diagnostics['toolchain']['cc_path'] or 'missing'}"
        ),
    ]

    if not diagnostics["qgraf"]["available"] or not diagnostics["form"]["available"]:
        lines.append("  Recommendation: run `feynman setup` for the recommended native setup.")
    elif not diagnostics["looptools"]["available"]:
        lines.append("  Recommendation: run `feynman install-looptools` for numerical 1-loop evaluation.")
    else:
        lines.append("  Recommendation: native dependencies look ready.")

    return "\n".join(lines)
