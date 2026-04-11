"""
Diagram generator — selects the best available backend automatically.

Priority:
  1. bin/qgraf_pipe  (patched stdin/stderr version — from QGraf GitLab)
  2. bin/qgraf       (standard file-based QGRAF binary)
  3. Python enumerator (built-in, no external tools required)

For loops > 0, only QGRAF can enumerate diagrams.  The Python enumerator
raises NotImplementedError for loops > 0, prompting the user to install QGRAF.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from feynman_engine.core.models import Diagram
from feynman_engine.core.enumerator import enumerate_tree
from feynman_engine.physics.translator import ProcessSpec, write_qgraf_dat

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_BIN_DIR = _PROJECT_ROOT / "bin"
_QGRAF_PIPE = _BIN_DIR / "qgraf_pipe"
_QGRAF_STD  = _BIN_DIR / "qgraf"
_QGRAF_CONTRIB_MODELS = _PROJECT_ROOT / "contrib" / "qgraf" / "models"
_QGRAF_CONTRIB_STYLES = _PROJECT_ROOT / "contrib" / "qgraf" / "styles"
_FEYNMAN_STY = _QGRAF_CONTRIB_STYLES / "feynman.sty"

QGRAF_TIMEOUT = 60  # seconds


class QGRAFError(RuntimeError):
    pass


# ── Backend detection ─────────────────────────────────────────────────────────

def _qgraf_pipe_path() -> Path | None:
    if _QGRAF_PIPE.exists() and os.access(_QGRAF_PIPE, os.X_OK):
        return _QGRAF_PIPE
    sys_path = shutil.which("qgraf_pipe")
    return Path(sys_path) if sys_path else None


def _qgraf_std_path() -> Path | None:
    if _QGRAF_STD.exists() and os.access(_QGRAF_STD, os.X_OK):
        return _QGRAF_STD
    sys_path = shutil.which("qgraf")
    return Path(sys_path) if sys_path else None


def qgraf_available() -> bool:
    return _qgraf_pipe_path() is not None or _qgraf_std_path() is not None


def backend_name() -> str:
    if _qgraf_pipe_path():
        return f"qgraf_pipe ({_qgraf_pipe_path()})"
    if _qgraf_std_path():
        return f"qgraf ({_qgraf_std_path()})"
    return "python-enumerator"


# ── QGRAF pipe backend ────────────────────────────────────────────────────────

def _qgraf_options(filters: dict | None) -> str:
    """Build the QGRAF options= string from filter settings."""
    opts = []
    f = filters or {}
    if f.get("no_tadpoles", True):
        opts.append("notadpole")
    if f.get("one_pi", False):
        opts.append("noexternal1pr")
    return ", ".join(opts) if opts else "notadpole"


def _run_qgraf_pipe(spec: ProcessSpec, filters: dict | None = None) -> str:
    """
    Run qgraf_pipe (stdin/stderr version from QGraf GitLab).

    Input format (sent to stdin):
        <qgraf.dat lines, each prefixed with a space>
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        <style file lines, each prefixed with a space>
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        <model file lines, each prefixed with a space>
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Diagram output is on stderr.
    """
    binary = _qgraf_pipe_path()
    theory_lower = spec.theory.lower()

    model_file = _QGRAF_CONTRIB_MODELS / f"{theory_lower}.mod"
    style_file  = _QGRAF_CONTRIB_STYLES / "diagram.sty"

    if not model_file.exists():
        raise QGRAFError(f"QGRAF model file not found: {model_file}")
    if not style_file.exists():
        raise QGRAFError(f"QGRAF style file not found: {style_file}")

    options_str = _qgraf_options(filters)
    in_str  = ", ".join(spec.qgraf_incoming)
    out_str = ", ".join(spec.qgraf_outgoing)
    dat_lines = [
        "output= 'qgraf_output.txt' ;",
        f"style= 'diagram.sty' ;",
        f"model= 'model.mod' ;",
        f"in= {in_str} ;",
        f"out= {out_str} ;",
        f"loops= {spec.loops} ;",
        "loop_momentum= k ;",
        f"options= {options_str} ;",
    ]
    style_lines = style_file.read_text().splitlines()
    model_lines = model_file.read_text().splitlines()

    SEP = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

    def prefix(lines):
        return "\n".join(f" {line}" for line in lines)

    stdin_text = (
        prefix(dat_lines) + f"\n{SEP}\n"
        + prefix(style_lines) + f"\n{SEP}\n"
        + prefix(model_lines) + f"\n{SEP}\n"
    )

    result = subprocess.run(
        [str(binary)],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=QGRAF_TIMEOUT,
    )

    if result.returncode != 0:
        raise QGRAFError(
            f"qgraf_pipe exited with code {result.returncode}.\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )

    # Output is on stderr for the pipe version
    return result.stderr


# ── QGRAF standard backend ────────────────────────────────────────────────────

def _run_qgraf_std(spec: ProcessSpec, filters: dict | None = None) -> str:
    """
    Run standard file-based QGRAF binary.
    Copies model + style into a tempdir, writes qgraf.dat with relative paths,
    runs the binary from that directory, and returns the output text.
    """
    binary = _qgraf_std_path()

    _MODEL_FILES = {
        "QED": "qed.mod",
        "QCD": "qcd.mod",
        "EW":  "electroweak.mod",
        "BSM": "bsm.mod",
    }
    model_filename = _MODEL_FILES.get(spec.theory, f"{spec.theory.lower()}.mod")
    model_src = _QGRAF_CONTRIB_MODELS / model_filename
    style_src = _FEYNMAN_STY

    if not model_src.exists():
        raise QGRAFError(f"QGRAF model file not found: {model_src}")
    if not style_src.exists():
        raise QGRAFError(f"QGRAF style file not found: {style_src}")

    with tempfile.TemporaryDirectory(prefix="feynman_") as tmpdir:
        tmpdir = Path(tmpdir)
        shutil.copy(model_src,  tmpdir / "model.mod")
        shutil.copy(style_src,  tmpdir / "feynman.sty")

        # QGRAF has a line-length limit (~80 chars) on qgraf.dat entries.
        # Use only basenames — QGRAF resolves them relative to its cwd.
        write_qgraf_dat(
            spec=spec,
            model_name="model.mod",
            style_name="feynman.sty",
            output_name="output.txt",
            qgraf_dat_path=str(tmpdir / "qgraf.dat"),
            options=_qgraf_options(filters),
        )

        result = subprocess.run(
            [str(binary)],
            cwd=str(tmpdir),
            capture_output=True,
            text=True,
            timeout=QGRAF_TIMEOUT,
        )
        if result.returncode != 0:
            raise QGRAFError(
                f"QGRAF exited with code {result.returncode}.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        output_file = tmpdir / "output.txt"
        if not output_file.exists():
            raise QGRAFError("QGRAF ran but produced no output file.")
        return output_file.read_text()


# ── Public API ────────────────────────────────────────────────────────────────

def generate_diagrams(spec: ProcessSpec, filters: dict | None = None) -> list[Diagram]:
    """
    Generate Feynman diagrams for the given ProcessSpec.

    Uses QGRAF when available, falls back to the pure Python enumerator for
    tree-level processes.

    Returns a list of Diagram objects (deduplicated, topology classified).
    """
    # QGRAF path: supports any loop order
    if qgraf_available():
        from feynman_engine.core.parser import parse_qgraf_output
        from feynman_engine.core.normalize import deduplicate
        from feynman_engine.core.topology import classify_all

        pipe = _qgraf_pipe_path()
        if pipe:
            raw = _run_qgraf_pipe(spec, filters)
        else:
            raw = _run_qgraf_std(spec, filters)

        diagrams = parse_qgraf_output(raw, theory=spec.theory, process=spec.raw)
        # QGRAF guarantees no duplicate diagrams in its output — skip our
        # graph-isomorphism deduplication (which would incorrectly merge
        # physically distinct diagrams that share the same abstract topology,
        # e.g. the two Compton diagrams).
        diagrams = classify_all(diagrams)
        return diagrams

    # Python enumerator: tree-level only
    if spec.loops > 0:
        raise NotImplementedError(
            f"Loop diagrams (loops={spec.loops}) require QGRAF. "
            "See contrib/qgraf/ for compilation instructions, or email the "
            "author at paulo.nogueira@tecnico.ulisboa.pt to request the source."
        )

    return enumerate_tree(spec.incoming, spec.outgoing, spec.theory)
