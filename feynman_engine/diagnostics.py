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
from feynman_engine.lhapdf import (
    lhapdf_source_available,
    is_lhapdf_installed_at,
    default_lhapdf_install_prefix,
)
from feynman_engine.openloops import (
    is_openloops_installed_at,
    openloops_source_available,
)
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


def _collect_lhapdf_status() -> dict:
    """Inspect LHAPDF: source archive present, library importable, sets installed."""
    # Try to import lhapdf — pdf.py auto-discovers it on import
    try:
        from feynman_engine.amplitudes.pdf import _lhapdf_available, _try_locate_lhapdf_install
        _try_locate_lhapdf_install()
        lhapdf_importable = _lhapdf_available()
    except Exception:
        lhapdf_importable = False

    # Detect install prefix (mirrors the auto-discovery list in pdf.py).
    detected_prefix = None
    detected_sets = []
    for prefix_str in (
        "/opt/lhapdf",
        "/tmp/lhapdf-install",
        "/usr/local",
        "/opt/homebrew",
        str(Path.home() / "lhapdf-install"),
        str(Path.home() / ".local" / "lhapdf"),
    ):
        prefix = Path(prefix_str)
        if is_lhapdf_installed_at(prefix):
            detected_prefix = prefix
            sets_dir = prefix / "share" / "LHAPDF"
            if sets_dir.is_dir():
                detected_sets = sorted(
                    p.name for p in sets_dir.iterdir()
                    if p.is_dir() and (p / f"{p.name}_0000.dat").exists()
                )
            break

    pdf_set_version = None
    if lhapdf_importable:
        try:
            import lhapdf as _lhapdf
            pdf_set_version = _lhapdf.version()
        except Exception:
            pass

    return {
        "available": lhapdf_importable,
        "version": pdf_set_version,
        "install_prefix": str(detected_prefix) if detected_prefix else None,
        "installed_sets": detected_sets,
        "source_available": lhapdf_source_available(),
    }


def _collect_openloops_status() -> dict:
    """Inspect OpenLoops: source archive, library importable, installed processes."""
    try:
        from feynman_engine.amplitudes.openloops_bridge import (
            is_available as openloops_available,
            install_prefix as openloops_install_prefix,
            installed_processes as openloops_installed_processes,
        )
        ol_importable = openloops_available()
        detected_prefix = openloops_install_prefix()
        installed_procs = openloops_installed_processes() if detected_prefix else []
    except Exception:
        ol_importable = False
        detected_prefix = None
        installed_procs = []

    # Even if the bindings can't be loaded, surface the install prefix when
    # we can detect the on-disk layout (helps the user diagnose chdir/import
    # failures from the doctor output).
    if detected_prefix is None:
        for prefix_str in (
            "/opt/openloops",
            "/tmp/openloops-install",
            "/usr/local/openloops",
            str(Path.home() / ".local" / "openloops"),
            str(Path.home() / "openloops-install"),
        ):
            prefix = Path(prefix_str)
            if is_openloops_installed_at(prefix):
                detected_prefix = prefix
                break

    return {
        "available": ol_importable,
        "install_prefix": str(detected_prefix) if detected_prefix else None,
        "installed_processes": installed_procs,
        "source_available": openloops_source_available(),
    }


def collect_diagnostics() -> dict:
    qgraf_path = qgraf_binary_path()
    form_path = find_form_binary()
    looptools_path = find_looptools_library()
    lualatex_path = _find_lualatex()
    pdf2svg_path = _find_program("pdf2svg")
    gfortran_path = _find_program("gfortran")
    make_path = _find_program("make", "gmake")
    cc_path = _find_program("cc", "gcc", "clang")
    cxx_path = _find_program("g++", "c++", "clang++")

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
        "lhapdf": _collect_lhapdf_status(),
        "openloops": _collect_openloops_status(),
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
            "cxx_path": str(cxx_path) if cxx_path else None,
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
        # LHAPDF is optional; gracefully degrade if not in diagnostics dict.
        (
            "  LHAPDF: "
            f"{'ok' if diagnostics.get('lhapdf', {}).get('available') else 'missing'}"
            f" | version={diagnostics.get('lhapdf', {}).get('version') or 'n/a'}"
            f" | prefix={diagnostics.get('lhapdf', {}).get('install_prefix') or 'not found'}"
            f" | sets={diagnostics.get('lhapdf', {}).get('installed_sets') or 'none'}"
        ),
        # OpenLoops is optional; gates the generic-NLO path.
        (
            "  OpenLoops: "
            f"{'ok' if diagnostics.get('openloops', {}).get('available') else 'missing'}"
            f" | prefix={diagnostics.get('openloops', {}).get('install_prefix') or 'not found'}"
            f" | processes={diagnostics.get('openloops', {}).get('installed_processes') or 'none'}"
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
            f"cc={diagnostics['toolchain']['cc_path'] or 'missing'}, "
            f"c++={diagnostics['toolchain'].get('cxx_path') or 'missing'}"
        ),
    ]

    lhapdf = diagnostics.get("lhapdf", {})
    if not diagnostics["qgraf"]["available"] or not diagnostics["form"]["available"]:
        lines.append("  Recommendation: run `feynman setup` for the recommended native setup.")
    elif not diagnostics["looptools"]["available"]:
        lines.append("  Recommendation: run `feynman install-looptools` for numerical 1-loop evaluation.")
    elif not lhapdf.get("available"):
        lines.append(
            "  Recommendation: run `feynman install-lhapdf` for percent-level "
            "PDF accuracy on hadron-collider σ (built-in PDF gives factor-of-2-3)."
        )
    elif not lhapdf.get("installed_sets"):
        lines.append(
            "  Recommendation: run `feynman install-pdf-set CT18LO` to install "
            "a default PDF set."
        )
    elif not diagnostics.get("openloops", {}).get("available"):
        lines.append(
            "  Recommendation: run `feynman install-openloops` for generic NLO "
            "via OpenLoops (enables full virtual + tree for arbitrary processes)."
        )
    elif not diagnostics.get("openloops", {}).get("installed_processes"):
        lines.append(
            "  Recommendation: run `feynman install-process ppllj` (or any other "
            "OpenLoops process library) to enable generic NLO evaluations."
        )
    else:
        lines.append("  Recommendation: native dependencies look ready.")

    return "\n".join(lines)
