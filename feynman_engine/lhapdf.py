"""Helpers for locating and building LHAPDF from a bundled source archive.

LHAPDF (Buckley et al., EPJ C 75 (2015) 132) is a C++ library + Python
bindings for evaluating any published parton distribution function set
(CT18LO, NNPDF40, MSHT20, ...).  It's the standard tool for hadron-collider
predictions in particle physics.

Building from source requires a C++ compiler, make, and the Python headers.
Once built, the bindings are auto-discovered by ``feynman_engine.amplitudes.pdf``
and used as the default PDF for ``hadronic_cross_section`` and friends.

After building, you also need at least one PDF set:

    feynman install-pdf-set CT18LO

(or download from http://lhapdfsets.web.cern.ch/lhapdfsets/current/).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from importlib.resources import as_file, files
from pathlib import Path


class LHAPDFBuildError(RuntimeError):
    pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SOURCE_ARCHIVE = "LHAPDF-6.5.5.tar.gz"

# Default install location: writable system path or user cache.  We prefer
# /tmp/lhapdf-install in CI / one-shot setups (matches what
# `_try_locate_lhapdf_install` in pdf.py probes), and ~/.local/lhapdf for
# persistent user installs.
_DEFAULT_INSTALL_PREFIX = "/tmp/lhapdf-install"
_USER_INSTALL_PREFIX = Path.home() / ".local" / "lhapdf"


def repo_root() -> Path:
    return _PROJECT_ROOT


def lhapdf_source_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("FEYNMAN_LHAPDF_SOURCE")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(repo_root() / _DEFAULT_SOURCE_ARCHIVE)
    candidates.append(
        repo_root() / "feynman_engine" / "resources" / "lhapdf" / _DEFAULT_SOURCE_ARCHIVE
    )
    return candidates


def lhapdf_source_available() -> bool:
    """Return True when the LHAPDF source tarball is present (project root,
    package resources, or via the FEYNMAN_LHAPDF_SOURCE override)."""
    if any(path.is_file() for path in lhapdf_source_candidates()):
        return True
    resource = files("feynman_engine").joinpath(
        f"resources/lhapdf/{_DEFAULT_SOURCE_ARCHIVE}"
    )
    return resource.is_file()


def _writable_directory(path: Path) -> bool:
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return os.access(parent, os.W_OK)


def default_lhapdf_install_prefix() -> Path:
    """Where to install LHAPDF by default.  Honors FEYNMAN_LHAPDF_PREFIX."""
    env_path = os.environ.get("FEYNMAN_LHAPDF_PREFIX")
    if env_path:
        return Path(env_path).expanduser()

    candidate = Path(_DEFAULT_INSTALL_PREFIX)
    if candidate.exists() or _writable_directory(candidate):
        return candidate
    return _USER_INSTALL_PREFIX


def is_lhapdf_installed_at(prefix: Path) -> bool:
    """Quick check: does this prefix contain a usable LHAPDF install?"""
    if not prefix.is_dir():
        return False
    # Look for the Python module dir
    lib_dir = prefix / "lib"
    if not lib_dir.is_dir():
        return False
    py_dirs = list(lib_dir.glob("python*/site-packages/lhapdf"))
    return any(d.is_dir() for d in py_dirs)


def build_lhapdf(target: str | Path | None = None, force: bool = False) -> Path:
    """Build LHAPDF from the bundled source archive.

    Compiles the LHAPDF C++ library and Python bindings, installs them to
    ``target`` (or the default prefix from ``default_lhapdf_install_prefix()``).
    The resulting prefix can be auto-discovered by
    ``feynman_engine.amplitudes.pdf._try_locate_lhapdf_install()``.

    Parameters
    ----------
    target : str | Path | None
        Install prefix (e.g. ``/usr/local`` or ``/tmp/lhapdf-install``).
        Defaults to ``default_lhapdf_install_prefix()``.
    force : bool
        Rebuild even if LHAPDF is already installed at the target.

    Returns
    -------
    Path
        The install prefix.

    Requirements
    ------------
    - C++ compiler (g++ / clang++)
    - make
    - Python headers (python3-dev / python3-devel) matching the active interpreter
    """
    cxx = shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        raise LHAPDFBuildError(
            "A C++ compiler is required to build LHAPDF.\n"
            "  macOS:  xcode-select --install (provides clang++)\n"
            "  Debian: apt-get install g++"
        )
    make = shutil.which("make") or shutil.which("gmake")
    if not make:
        raise LHAPDFBuildError(
            "make is required to build LHAPDF.\n"
            "  macOS:  xcode-select --install\n"
            "  Debian: apt-get install make"
        )

    install_prefix = (
        Path(target).expanduser() if target else default_lhapdf_install_prefix()
    )
    if is_lhapdf_installed_at(install_prefix) and not force:
        return install_prefix

    install_prefix.mkdir(parents=True, exist_ok=True)

    direct_sources = [path for path in lhapdf_source_candidates() if path.is_file()]
    if direct_sources:
        return _build_lhapdf_from_archive(
            direct_sources[0], install_prefix, cxx, make,
        )

    resource = files("feynman_engine").joinpath(
        f"resources/lhapdf/{_DEFAULT_SOURCE_ARCHIVE}"
    )
    if resource.is_file():
        with as_file(resource) as resource_path:
            return _build_lhapdf_from_archive(
                Path(resource_path), install_prefix, cxx, make,
            )

    raise LHAPDFBuildError(
        f"No LHAPDF source archive found. Expected {_DEFAULT_SOURCE_ARCHIVE} "
        "in the project root, package resources, or FEYNMAN_LHAPDF_SOURCE."
    )


def _build_lhapdf_from_archive(
    source_archive: Path,
    install_prefix: Path,
    cxx: str,
    make: str,
) -> Path:
    """Configure / make / make install for a bundled LHAPDF tarball."""
    with tempfile.TemporaryDirectory(prefix="feynman_lhapdf_build_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        build_dir = tmpdir_path / "src"
        build_dir.mkdir()

        # LHAPDF tarballs are gzipped tar archives.
        subprocess.run(
            ["tar", "-xzf", str(source_archive), "-C", str(build_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        subdirs = [d for d in build_dir.iterdir() if d.is_dir()]
        if len(subdirs) != 1 or not (subdirs[0] / "configure").exists():
            raise LHAPDFBuildError(
                f"Could not find LHAPDF 'configure' in extracted tarball {source_archive}."
            )
        src_dir = subdirs[0]

        env = {
            **os.environ,
            "PYTHON": sys.executable,
            "CXX": cxx,
        }

        # Configure
        result = subprocess.run(
            [
                str(src_dir / "configure"),
                f"--prefix={install_prefix}",
            ],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(src_dir),
        )
        if result.returncode != 0:
            raise LHAPDFBuildError(
                "LHAPDF configure failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # Make
        result = subprocess.run(
            [make, "-j4"],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(src_dir),
        )
        if result.returncode != 0:
            raise LHAPDFBuildError(
                "LHAPDF make failed.\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

        # Make install
        result = subprocess.run(
            [make, "install"],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(src_dir),
        )
        if result.returncode != 0:
            raise LHAPDFBuildError(
                "LHAPDF make install failed.\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

        return install_prefix


# ─── PDF set installer ────────────────────────────────────────────────────────

# Recommended starter sets for various physics regimes.  CT18LO is the
# default our `get_pdf("auto")` looks for.
DEFAULT_PDF_SET = "CT18LO"
COMMON_PDF_SETS = [
    "CT18LO",      # CTEQ-TEA LO (recommended default)
    "CT18NLO",     # CTEQ-TEA NLO
    "NNPDF40_lo_as_01180",  # NNPDF 4.0 LO
    "MSHT20lo_as130",  # MSHT 2020 LO
]
_PDF_SETS_URL_BASE = "http://lhapdfsets.web.cern.ch/lhapdfsets/current/"


def install_pdf_set(set_name: str = DEFAULT_PDF_SET, prefix: str | Path | None = None) -> Path:
    """Download and unpack an LHAPDF PDF set.

    Saves the set to ``<prefix>/share/LHAPDF/<set_name>``.  After install the
    set will be visible to LHAPDF if ``LHAPDF_DATA_PATH`` is configured (which
    ``feynman_engine.amplitudes.pdf`` does automatically).

    Parameters
    ----------
    set_name : str
        Any LHAPDF set name (e.g. ``"CT18LO"``, ``"NNPDF40_lo_as_01180"``).
    prefix : str | Path | None
        LHAPDF install prefix.  Defaults to ``default_lhapdf_install_prefix()``.

    Returns
    -------
    Path
        The directory where the set was installed.
    """
    install_prefix = (
        Path(prefix).expanduser() if prefix else default_lhapdf_install_prefix()
    )
    if not is_lhapdf_installed_at(install_prefix):
        raise LHAPDFBuildError(
            f"LHAPDF not installed at {install_prefix}. "
            "Run `feynman install-lhapdf` first."
        )

    sets_dir = install_prefix / "share" / "LHAPDF"
    sets_dir.mkdir(parents=True, exist_ok=True)
    target_dir = sets_dir / set_name

    if target_dir.is_dir() and any(target_dir.iterdir()):
        return target_dir

    url = f"{_PDF_SETS_URL_BASE}{set_name}.tar.gz"
    tarball = sets_dir / f"{set_name}.tar.gz"
    print(f"[install-pdf-set] Downloading {url} ...")
    result = subprocess.run(
        ["curl", "-fsL", url, "-o", str(tarball), "--max-time", "120"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise LHAPDFBuildError(
            f"Failed to download {url}.  Check the set name and your internet "
            f"connection.\nstderr: {result.stderr}"
        )

    print(f"[install-pdf-set] Extracting {tarball} ...")
    result = subprocess.run(
        ["tar", "-xzf", str(tarball), "-C", str(sets_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise LHAPDFBuildError(
            f"Failed to extract {tarball}.  stderr: {result.stderr}"
        )

    tarball.unlink(missing_ok=True)
    return target_dir
