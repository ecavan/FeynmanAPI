"""Helpers for locating and building LoopTools from a bundled source archive.

LoopTools (Hahn & Pérez-Victoria, CPC 118 (1999) 153) is a Fortran library
for numerical evaluation of Passarino-Veltman scalar integrals A₀, B₀, C₀, D₀.

Building from source requires gfortran and make (standard on Linux/macOS with
developer tools installed).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from importlib.resources import as_file, files
from pathlib import Path


class LoopToolsBuildError(RuntimeError):
    pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SOURCE_ARCHIVE = "LoopTools-2.16.tar"
_USER_CACHE_DIR = Path.home() / ".cache" / "feynman-engine" / "lib"

# The compiled shared library will be named differently per platform.
import sys as _sys
_LIB_FILENAME = "liblooptools.dylib" if _sys.platform == "darwin" else "liblooptools.so"


def repo_root() -> Path:
    return _PROJECT_ROOT


def repo_looptools_lib() -> Path:
    return repo_root() / "bin" / _LIB_FILENAME


def cache_looptools_lib() -> Path:
    return _USER_CACHE_DIR / _LIB_FILENAME


def looptools_source_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("FEYNMAN_LOOPTOOLS_SOURCE")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(repo_root() / _DEFAULT_SOURCE_ARCHIVE)
    candidates.append(repo_root() / "feynman_engine" / "resources" / "looptools" / _DEFAULT_SOURCE_ARCHIVE)
    return candidates


def looptools_source_available() -> bool:
    if any(path.is_file() for path in looptools_source_candidates()):
        return True
    resource = files("feynman_engine").joinpath(f"resources/looptools/{_DEFAULT_SOURCE_ARCHIVE}")
    return resource.is_file()


def _writable_directory(path: Path) -> bool:
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return os.access(parent, os.W_OK)


def default_looptools_lib_target() -> Path:
    env_path = os.environ.get("FEYNMAN_LOOPTOOLS_LIB")
    if env_path:
        return Path(env_path).expanduser()

    repo_target = repo_looptools_lib()
    if repo_target.exists() or _writable_directory(repo_target):
        return repo_target
    return cache_looptools_lib()


def build_looptools(target: str | Path | None = None, force: bool = False) -> Path:
    """
    Build LoopTools from the bundled or configured source archive.

    Produces a shared library (liblooptools.dylib on macOS, liblooptools.so on
    Linux) suitable for loading via ctypes.

    The default target is ``./bin/liblooptools.{dylib,so}`` when the project
    directory is writable, otherwise ``~/.cache/feynman-engine/lib/``.

    Requirements
    ------------
    - gfortran (any GCC version ≥ 6)
    - make
    """
    gfortran = shutil.which("gfortran")
    if not gfortran:
        raise LoopToolsBuildError(
            "gfortran is required to build LoopTools. Install a Fortran compiler first.\n"
            "  macOS:  brew install gcc\n"
            "  Debian: apt-get install gfortran"
        )
    make = shutil.which("make") or shutil.which("gmake")
    if not make:
        raise LoopToolsBuildError(
            "make is required to build LoopTools. Install build tools first.\n"
            "  macOS:  xcode-select --install\n"
            "  Debian: apt-get install make"
        )

    output_path = Path(target).expanduser() if target else default_looptools_lib_target()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    direct_sources = [path for path in looptools_source_candidates() if path.is_file()]
    if direct_sources:
        return _build_looptools_from_archive(direct_sources[0], output_path, gfortran, make)

    resource = files("feynman_engine").joinpath(f"resources/looptools/{_DEFAULT_SOURCE_ARCHIVE}")
    if resource.is_file():
        with as_file(resource) as resource_path:
            return _build_looptools_from_archive(Path(resource_path), output_path, gfortran, make)

    raise LoopToolsBuildError(
        "No LoopTools source archive was found. Expected LoopTools-2.16.tar in the project "
        "root, package resources, or FEYNMAN_LOOPTOOLS_SOURCE."
    )


def _build_looptools_from_archive(
    source_archive: Path,
    output_path: Path,
    gfortran: str,
    make: str,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="feynman_looptools_build_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        build_dir = tmpdir_path / "src"
        build_dir.mkdir()

        # LoopTools-2.16.tar is an uncompressed POSIX tar (no gzip).
        subprocess.run(
            ["tar", "-xf", str(source_archive), "-C", str(build_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        # The archive may extract into a subdirectory (LoopTools-2.16/) or directly.
        configure_dir = build_dir
        subdirs = [d for d in build_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and (subdirs[0] / "configure").exists():
            configure_dir = subdirs[0]
        elif not (configure_dir / "configure").exists():
            raise LoopToolsBuildError(
                f"Could not find LoopTools 'configure' script in {source_archive}."
            )

        install_dir = tmpdir_path / "install"
        install_dir.mkdir()

        # Configure with -fPIC so the static archive can be linked into a shared lib.
        result = subprocess.run(
            [
                str(configure_dir / "configure"),
                f"--prefix={install_dir}",
            ],
            env={**os.environ, "FFLAGS": "-fPIC -O2", "CFLAGS": "-fPIC -O2"},
            capture_output=True,
            text=True,
            cwd=str(configure_dir),
        )
        if result.returncode != 0:
            raise LoopToolsBuildError(
                "LoopTools configure failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        result = subprocess.run(
            [make, "-j4"],
            capture_output=True,
            text=True,
            cwd=str(configure_dir),
        )
        if result.returncode != 0:
            raise LoopToolsBuildError(
                "LoopTools make failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # Locate the static archive produced by make.
        static_libs = list(configure_dir.glob("build/libooptools.a"))
        if not static_libs:
            static_libs = list(configure_dir.rglob("libooptools.a"))
        if not static_libs:
            raise LoopToolsBuildError(
                "make succeeded but libooptools.a was not found in the build tree."
            )
        static_lib = static_libs[0]

        # Create a shared library from the static archive.
        tmp_dylib = tmpdir_path / _LIB_FILENAME
        if _sys.platform == "darwin":
            link_cmd = [
                gfortran, "-dynamiclib",
                "-o", str(tmp_dylib),
                "-Wl,-all_load", str(static_lib),
                "-lgfortran", "-lm",
            ]
        else:
            link_cmd = [
                gfortran, "-shared",
                "-o", str(tmp_dylib),
                f"-Wl,--whole-archive,{static_lib},--no-whole-archive",
                "-lgfortran", "-lm",
            ]

        result = subprocess.run(
            link_cmd,
            capture_output=True,
            text=True,
            cwd=str(tmpdir_path),
        )
        if result.returncode != 0:
            raise LoopToolsBuildError(
                f"Failed to link {_LIB_FILENAME} from libooptools.a.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        shutil.copy2(tmp_dylib, output_path)
        output_path.chmod(0o755)
        return output_path
