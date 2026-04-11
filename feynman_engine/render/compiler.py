"""
Compile TikZ-Feynman LaTeX source to SVG via lualatex + pdf2svg.

Each compilation runs in an isolated temporary directory.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


LUALATEX_TIMEOUT = 90   # seconds per diagram
PDF2SVG_TIMEOUT  = 30

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_BUNDLED_TEXMF = _PROJECT_ROOT / "contrib" / "texmf"

# Known install paths for lualatex (BasicTeX on macOS installs here)
_LUALATEX_SEARCH = [
    "/usr/local/texlive/2026basic/bin/universal-darwin/lualatex",
    "/usr/local/texlive/2025/bin/universal-darwin/lualatex",
    "/usr/local/texlive/2024/bin/universal-darwin/lualatex",
    "/Library/TeX/texbin/lualatex",
    "lualatex",   # system PATH fallback
]


class RenderError(RuntimeError):
    pass


class MissingDependencyError(RenderError):
    pass


def _find_tool(name: str, extra_paths: list[str] | None = None) -> str:
    """Locate an executable, checking extra paths before PATH."""
    search = extra_paths or []
    for candidate in search:
        if Path(candidate).is_file() and Path(candidate).stat().st_mode & 0o111:
            return candidate
    found = shutil.which(name)
    if found:
        return found
    install_hint = {
        "lualatex": (
            "Install BasicTeX:\n"
            "  brew install --cask basictex\n"
            "  eval \"$(/usr/libexec/path_helper)\"\n"
            "  sudo tlmgr install tikz-feynman"
        ),
        "pdf2svg": "  brew install pdf2svg",
    }.get(name, f"  Install {name} and ensure it is on PATH.")
    raise MissingDependencyError(
        f"'{name}' not found.\n{install_hint}"
    )


def _crop_svg(svg_bytes: bytes, padding: float = 8.0) -> bytes:
    """
    Crop a LaTeX/pdf2svg SVG to its actual content bounding box.

    lualatex produces a fixed large page (e.g. 20 cm × 20 cm).  The diagram
    sits at the top-left corner.  We scan <path> and <use> coordinates outside
    the <defs> section to compute the content bounding box, then update the
    SVG width / height / viewBox accordingly.
    """
    try:
        svg_text = svg_bytes.decode("utf-8")

        # Strip <defs> — glyph definitions there have font-unit coordinates
        # (typically 0–5) that would shrink the bounding box incorrectly.
        body = re.sub(r"<defs\b[^>]*>.*?</defs>", "", svg_text, flags=re.DOTALL)

        xs: list[float] = []
        ys: list[float] = []

        # <use x="N" y="N"> — placed glyph positions
        for m in re.finditer(r"<use\b[^>]*>", body):
            tag = m.group(0)
            mx = re.search(r'\bx="([-\d.]+)"', tag)
            my = re.search(r'\by="([-\d.]+)"', tag)
            if mx:
                xs.append(float(mx.group(1)))
            if my:
                ys.append(float(my.group(1)))

        # <path d="..."> — diagram strokes in absolute SVG coordinates
        for m in re.finditer(r'\bd="([^"]+)"', body):
            nums = re.findall(r"-?\d+\.?\d*", m.group(1))
            for i in range(0, len(nums) - 1, 2):
                try:
                    xs.append(float(nums[i]))
                    ys.append(float(nums[i + 1]))
                except ValueError:
                    pass

        if not xs or not ys:
            return svg_bytes

        # Content starts near (0, 0) with margin=0pt geometry
        min_x = max(0.0, min(xs) - padding)
        min_y = max(0.0, min(ys) - padding)
        max_x = max(xs) + padding
        max_y = max(ys) + padding
        w = max_x - min_x
        h = max_y - min_y

        if w <= 0 or h <= 0:
            return svg_bytes

        # Replace the three root SVG attributes in-place
        svg_text = re.sub(
            r'width="[^"]+" height="[^"]+" viewBox="[^"]+"',
            f'width="{w:.3f}pt" height="{h:.3f}pt"'
            f' viewBox="{min_x:.3f} {min_y:.3f} {w:.3f} {h:.3f}"',
            svg_text,
            count=1,
        )
        return svg_text.encode("utf-8")

    except Exception:
        # Never let cropping break a successful render
        return svg_bytes


def _lualatex_env() -> dict[str, str]:
    """
    Build a subprocess environment that points lualatex to our bundled
    contrib/texmf directory first (contains a standalone.cls fallback).
    """
    env = os.environ.copy()
    bundled = str(_BUNDLED_TEXMF)
    # TEXINPUTS format: path// (recursive) + :: (append default paths)
    existing = env.get("TEXINPUTS", "")
    env["TEXINPUTS"] = f"{bundled}/tex/latex//:{existing}:"
    return env


def tikz_to_svg(tikz_code: str) -> bytes:
    """
    Compile a TikZ-Feynman standalone .tex document to SVG bytes.

    Pipeline: .tex → lualatex (×2 for layout) → .pdf → pdf2svg → .svg
    The resulting SVG is cropped to the diagram's actual bounding box.
    """
    lualatex = _find_tool("lualatex", _LUALATEX_SEARCH)
    pdf2svg  = _find_tool("pdf2svg")
    env      = _lualatex_env()

    with tempfile.TemporaryDirectory(prefix="feynman_render_") as tmpdir:
        tmpdir   = Path(tmpdir)
        tex_file = tmpdir / "diagram.tex"
        pdf_file = tmpdir / "diagram.pdf"
        svg_file = tmpdir / "diagram.svg"

        tex_file.write_text(tikz_code)

        # Two lualatex passes — second pass lets TikZ-Feynman finalise layout
        for _pass in range(2):
            result = subprocess.run(
                [lualatex,
                 "--interaction=nonstopmode",
                 "--halt-on-error",
                 str(tex_file)],
                cwd=str(tmpdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=LUALATEX_TIMEOUT,
            )
            if result.returncode != 0:
                raise RenderError(
                    f"lualatex failed (exit {result.returncode}).\n"
                    f"Log (last 2000 chars):\n{result.stdout[-2000:]}"
                )

        if not pdf_file.exists():
            raise RenderError("lualatex succeeded but produced no PDF.")

        result = subprocess.run(
            [pdf2svg, str(pdf_file), str(svg_file)],
            capture_output=True, text=True, timeout=PDF2SVG_TIMEOUT,
        )
        if result.returncode != 0:
            raise RenderError(
                f"pdf2svg failed (exit {result.returncode}).\n"
                f"stderr: {result.stderr}"
            )
        if not svg_file.exists():
            raise RenderError("pdf2svg succeeded but produced no SVG.")

        raw_svg = svg_file.read_bytes()
        return _crop_svg(raw_svg)


def tikz_to_pdf(tikz_code: str) -> bytes:
    """
    Compile a TikZ-Feynman standalone .tex document to PDF bytes.

    Same pipeline as tikz_to_svg but stops after lualatex (no pdf2svg).
    Returns raw PDF bytes suitable for download.
    """
    lualatex = _find_tool("lualatex", _LUALATEX_SEARCH)
    env      = _lualatex_env()

    with tempfile.TemporaryDirectory(prefix="feynman_pdf_") as tmpdir:
        tmpdir   = Path(tmpdir)
        tex_file = tmpdir / "diagram.tex"
        pdf_file = tmpdir / "diagram.pdf"

        tex_file.write_text(tikz_code)

        for _pass in range(2):
            result = subprocess.run(
                [lualatex,
                 "--interaction=nonstopmode",
                 "--halt-on-error",
                 str(tex_file)],
                cwd=str(tmpdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=LUALATEX_TIMEOUT,
            )
            if result.returncode != 0:
                raise RenderError(
                    f"lualatex failed (exit {result.returncode}).\n"
                    f"Log (last 2000 chars):\n{result.stdout[-2000:]}"
                )

        if not pdf_file.exists():
            raise RenderError("lualatex succeeded but produced no PDF.")

        return pdf_file.read_bytes()


def compile_all(tikz_codes: dict[int, str]) -> dict[int, bytes]:
    """
    Compile a batch of TikZ diagrams to SVG, running up to 4 in parallel.

    Diagrams that fail to compile are skipped with a warning.
    Returns {diagram_id: svg_bytes}.
    """
    results: dict[int, bytes] = {}

    def _compile_one(item: tuple[int, str]) -> tuple[int, bytes | None]:
        diagram_id, code = item
        try:
            return diagram_id, tikz_to_svg(code)
        except (RenderError, MissingDependencyError) as exc:
            print(f"Warning: diagram {diagram_id} render failed: {exc}", file=sys.stderr)
            return diagram_id, None

    max_workers = min(4, len(tikz_codes))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_compile_one, item) for item in tikz_codes.items()]
        for future in as_completed(futures):
            diagram_id, svg = future.result()
            if svg is not None:
                results[diagram_id] = svg

    return results
