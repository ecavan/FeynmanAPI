"""Helpers for locating and building OpenLoops from a bundled source archive.

OpenLoops 2 (Buccioni, Lang, Lindert, Maierhöfer, Pozzorini, Zhang, Zoller,
EPJ C 79 (2019) 866, arXiv:1907.13071) is a Fortran 90 + Python package for
the automated generation and numerical evaluation of tree and one-loop
amplitudes for arbitrary Standard Model processes.  In FeynmanEngine it
backs the "generic NLO for arbitrary processes" path: any process with an
installed OpenLoops library gets a full NLO virtual + tree from OpenLoops,
combined with our Catani-Seymour dipole subtraction for the real-emission
side.

Building from source requires gfortran (Fortran 2003), Python (with headers
on Linux), and SCons (bundled).  Once built, the install prefix is
auto-discovered by ``feynman_engine.amplitudes.openloops_bridge``.

After building you also need at least one process library, downloaded via:

    feynman install-process ppllj   # pp -> l+l- (+ jet)
    feynman install-process pphtt   # pp -> H tt~

Process libraries live under ``<prefix>/proclib`` and are downloaded from
the OpenLoops repository (https://openloops.hepforge.org/process_library.php).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from importlib.resources import as_file, files
from pathlib import Path


class OpenLoopsBuildError(RuntimeError):
    pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SOURCE_ARCHIVE = "OpenLoops-OpenLoops-2.1.4.tar.gz"

_DEFAULT_INSTALL_PREFIX = "/opt/openloops"
_TMP_INSTALL_PREFIX = "/tmp/openloops-install"
_USER_INSTALL_PREFIX = Path.home() / ".local" / "openloops"


def repo_root() -> Path:
    return _PROJECT_ROOT


def openloops_source_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("FEYNMAN_OPENLOOPS_SOURCE")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(repo_root() / _DEFAULT_SOURCE_ARCHIVE)
    candidates.append(
        repo_root() / "feynman_engine" / "resources" / "openloops" / _DEFAULT_SOURCE_ARCHIVE
    )
    return candidates


def openloops_source_available() -> bool:
    """Return True when the OpenLoops source tarball is present."""
    if any(path.is_file() for path in openloops_source_candidates()):
        return True
    resource = files("feynman_engine").joinpath(
        f"resources/openloops/{_DEFAULT_SOURCE_ARCHIVE}"
    )
    return resource.is_file()


def _writable_directory(path: Path) -> bool:
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return os.access(parent, os.W_OK)


def default_openloops_install_prefix() -> Path:
    """Where to install OpenLoops by default.  Honors FEYNMAN_OPENLOOPS_PREFIX."""
    env_path = os.environ.get("FEYNMAN_OPENLOOPS_PREFIX")
    if env_path:
        return Path(env_path).expanduser()

    for candidate in (Path(_DEFAULT_INSTALL_PREFIX), Path(_TMP_INSTALL_PREFIX)):
        if candidate.exists() or _writable_directory(candidate):
            return candidate
    return _USER_INSTALL_PREFIX


def is_openloops_installed_at(prefix: Path) -> bool:
    """Quick check: does this prefix contain a usable OpenLoops install?

    A usable install has the core ``libopenloops`` shared library and the
    Python bindings under ``pyol/tools/openloops.py``.
    """
    if not prefix.is_dir():
        return False
    lib_dir = prefix / "lib"
    if not lib_dir.is_dir():
        return False
    has_core_lib = any(
        (lib_dir / name).exists()
        for name in (
            "libopenloops.so", "libopenloops.dylib",
            "libopenloops.2.dylib", "libopenloops.2.so",
        )
    )
    has_py = (prefix / "pyol" / "tools" / "openloops.py").is_file()
    return has_core_lib and has_py


def installed_process_libraries(prefix: Path | None = None) -> list[str]:
    """List process libraries installed under ``<prefix>/proclib``."""
    install_prefix = Path(prefix).expanduser() if prefix else default_openloops_install_prefix()
    proclib = install_prefix / "proclib"
    if not proclib.is_dir():
        return []
    procs: set[str] = set()
    for info in proclib.glob("libopenloops_*.info"):
        # Filename: libopenloops_<process>_<spec>.info → strip "libopenloops_" and "_<spec>.info"
        stem = info.stem[len("libopenloops_"):]
        # Strip the trailing _<spec> (e.g. _lt, _ppt, _t)
        if "_" in stem:
            stem = stem.rsplit("_", 1)[0]
        procs.add(stem)
    return sorted(procs)


def build_openloops(target: str | Path | None = None, force: bool = False) -> Path:
    """Build OpenLoops from the bundled source archive.

    Extracts the OpenLoops source, runs the bundled SCons build, and copies
    the resulting tree (binaries, ``lib/``, ``pyol/``, ``include/``, the
    ``openloops`` driver script) into ``target``.

    Parameters
    ----------
    target : str | Path | None
        Install prefix (e.g. ``/opt/openloops``).
        Defaults to ``default_openloops_install_prefix()``.
    force : bool
        Rebuild even if OpenLoops is already installed at the target.

    Returns
    -------
    Path
        The install prefix.

    Requirements
    ------------
    - gfortran (Fortran 2003)
    - Python interpreter (the bundled SCons drives the build)
    """
    gfortran = shutil.which("gfortran")
    if not gfortran:
        raise OpenLoopsBuildError(
            "gfortran is required to build OpenLoops.\n"
            "  macOS:  brew install gcc\n"
            "  Debian: apt-get install gfortran"
        )

    install_prefix = (
        Path(target).expanduser() if target else default_openloops_install_prefix()
    )
    if is_openloops_installed_at(install_prefix) and not force:
        return install_prefix

    install_prefix.mkdir(parents=True, exist_ok=True)

    direct_sources = [path for path in openloops_source_candidates() if path.is_file()]
    if direct_sources:
        return _build_openloops_from_archive(direct_sources[0], install_prefix, gfortran)

    resource = files("feynman_engine").joinpath(
        f"resources/openloops/{_DEFAULT_SOURCE_ARCHIVE}"
    )
    if resource.is_file():
        with as_file(resource) as resource_path:
            return _build_openloops_from_archive(
                Path(resource_path), install_prefix, gfortran,
            )

    raise OpenLoopsBuildError(
        f"No OpenLoops source archive found.  Expected {_DEFAULT_SOURCE_ARCHIVE} "
        "in the project root, package resources, or FEYNMAN_OPENLOOPS_SOURCE."
    )


def _build_openloops_from_archive(
    source_archive: Path,
    install_prefix: Path,
    gfortran: str,
) -> Path:
    """Extract, build, and stage OpenLoops into ``install_prefix``."""
    with tempfile.TemporaryDirectory(prefix="feynman_openloops_build_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        build_dir = tmpdir_path / "src"
        build_dir.mkdir()

        subprocess.run(
            ["tar", "-xzf", str(source_archive), "-C", str(build_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        subdirs = [d for d in build_dir.iterdir() if d.is_dir()]
        if len(subdirs) != 1 or not (subdirs[0] / "scons").exists():
            raise OpenLoopsBuildError(
                f"Could not find OpenLoops 'scons' in extracted tarball {source_archive}."
            )
        src_dir = subdirs[0]

        env = {
            **os.environ,
            "PYTHON": sys.executable,
            "FC": gfortran,
        }

        # SCons is bundled — `./scons` invokes the local copy.
        result = subprocess.run(
            [sys.executable, "./scons"],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(src_dir),
        )
        if result.returncode != 0:
            raise OpenLoopsBuildError(
                "OpenLoops build failed.\n"
                f"stdout:\n{result.stdout[-3000:]}\n"
                f"stderr:\n{result.stderr[-3000:]}"
            )

        # OpenLoops is a self-contained tree — copy the built source tree to
        # the install prefix.  We need: openloops driver, lib/, pyol/,
        # lib_src/ (for rebuilding processes), include/, scons + scons-local
        # (for libinstall), proclib/ (created on first libinstall).
        for entry in ("openloops", "openloops.cfg.tmpl", "scons", "SConstruct",
                      "lib", "pyol", "lib_src", "include", "scons-local",
                      "config", "examples"):
            src = src_dir / entry
            dst = install_prefix / entry
            if not src.exists():
                continue
            if dst.exists():
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            if src.is_dir():
                shutil.copytree(src, dst, symlinks=True)
            else:
                shutil.copy2(src, dst)
                if entry in ("openloops", "scons"):
                    os.chmod(dst, 0o755)

        # Make sure proclib exists so installed_process_libraries() works.
        (install_prefix / "proclib").mkdir(exist_ok=True)

        return install_prefix


# ─── Process library installer ────────────────────────────────────────────────

# Curated starter set of OpenLoops process libraries.  See
# https://openloops.hepforge.org/process_library.php for the full list.
DEFAULT_PROCESS_LIBRARY = "ppllj"

# Curated starter pack — covers the major LHC analyses (Drell-Yan, top, Higgs,
# di-boson, VBF, ttH).  Each library is ~50-100 MB after compilation; the
# Docker image bundles ``ppllj`` only, additional libraries are user-installed
# via ``feynman install-process <name>``.
COMMON_PROCESS_LIBRARIES = [
    "ppllj",   # pp → l+l- (+ jet)         — Drell-Yan + jet (default, bundled)
    "pptt",    # pp → t t~                 — top pair (NLO QCD)
    "ppttj",   # pp → t t~ + jet           — tt + jet (NLO+1jet)
    "pphtt",   # pp → t t~ H               — ttH associated production
    "pph",     # pp → H                    — gluon-fusion Higgs (loop-induced!)
    "ppvv",    # pp → V V                  — di-boson incl. gg→VV loop-induced
    "pphjj",   # pp → H + 2 jets           — VBF Higgs (cross-check our calibrated path)
    "pphh",    # pp → H H                  — di-Higgs (loop-induced ggHH)
    "ppvvj",   # pp → V V + jet            — boosted V+jet topology
    "ppvjj",   # pp → V + 2 jets           — V+jets topology
]

# Loop-induced processes (Born has no tree contribution; the leading order is
# the 1-loop amplitude).  When OpenLoops is asked for these, ``tree`` will be
# 0 and we should use ``loop²`` directly as the |M|² rather than computing a
# K-factor.
LOOP_INDUCED_PROCESSES: set[str] = {
    "pph",     # gg → H (heavy-top loop)
    "pphh",    # gg → HH (box + triangle)
    # Note: ppvv contains BOTH tree (qq̄→VV) and loop-induced (gg→VV) parts
    # depending on the partonic channel selected.
}


def install_process_library(
    process: str = DEFAULT_PROCESS_LIBRARY,
    prefix: str | Path | None = None,
) -> Path:
    """Download and compile an OpenLoops process library.

    Invokes ``<prefix>/openloops libinstall <process>``.  The driver
    downloads the process source from the OpenLoops repository, compiles
    it in place, and produces ``<prefix>/proclib/libopenloops_<process>_*.dylib``
    (or ``.so`` on Linux).

    Parameters
    ----------
    process : str
        OpenLoops process name (e.g. ``"ppllj"``, ``"pptt"``).
    prefix : str | Path | None
        OpenLoops install prefix.  Defaults to ``default_openloops_install_prefix()``.

    Returns
    -------
    Path
        The proclib directory containing the new library.
    """
    install_prefix = (
        Path(prefix).expanduser() if prefix else default_openloops_install_prefix()
    )
    if not is_openloops_installed_at(install_prefix):
        raise OpenLoopsBuildError(
            f"OpenLoops not installed at {install_prefix}. "
            "Run `feynman install-openloops` first."
        )

    driver = install_prefix / "openloops"
    if not driver.is_file():
        raise OpenLoopsBuildError(
            f"OpenLoops driver script not found at {driver}."
        )

    print(f"[install-process] Downloading + compiling {process} ...")
    result = subprocess.run(
        [str(driver), "libinstall", process],
        cwd=str(install_prefix),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise OpenLoopsBuildError(
            f"OpenLoops libinstall {process} failed.\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )
    proclib = install_prefix / "proclib"
    return proclib
