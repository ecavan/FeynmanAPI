"""Helpers for locating and building QGRAF from a bundled source archive."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from importlib.resources import as_file, files
from pathlib import Path


class QGrafBuildError(RuntimeError):
    pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SOURCE_ARCHIVE = "qgraf-3.6.10.tgz"
_USER_CACHE_DIR = Path.home() / ".cache" / "feynman-engine" / "bin"


def repo_root() -> Path:
    return _PROJECT_ROOT


def repo_qgraf_bin() -> Path:
    return repo_root() / "bin" / "qgraf"


def cache_qgraf_bin() -> Path:
    return _USER_CACHE_DIR / "qgraf"


def qgraf_source_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("FEYNMAN_QGRAF_SOURCE")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(repo_root() / _DEFAULT_SOURCE_ARCHIVE)
    candidates.append(repo_root() / "feynman_engine" / "resources" / "qgraf" / _DEFAULT_SOURCE_ARCHIVE)
    return candidates


def qgraf_source_available() -> bool:
    if any(path.is_file() for path in qgraf_source_candidates()):
        return True
    resource = files("feynman_engine").joinpath(f"resources/qgraf/{_DEFAULT_SOURCE_ARCHIVE}")
    return resource.is_file()


def _writable_directory(path: Path) -> bool:
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return os.access(parent, os.W_OK)


def default_qgraf_bin_target() -> Path:
    env_path = os.environ.get("FEYNMAN_QGRAF_BIN")
    if env_path:
        return Path(env_path).expanduser()

    repo_target = repo_qgraf_bin()
    if repo_target.exists() or _writable_directory(repo_target):
        return repo_target
    return cache_qgraf_bin()


def build_qgraf(target: str | Path | None = None, force: bool = False) -> Path:
    """
    Build QGRAF from the bundled or configured source archive.

    The default target is `./bin/qgraf` when the project directory is writable,
    otherwise `~/.cache/feynman-engine/bin/qgraf`.
    """
    gfortran = shutil.which("gfortran")
    if not gfortran:
        raise QGrafBuildError(
            "gfortran is required to build QGRAF. Install a Fortran compiler first."
        )

    output_path = Path(target).expanduser() if target else default_qgraf_bin_target()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    direct_sources = [path for path in qgraf_source_candidates() if path.is_file()]
    if direct_sources:
        return _build_qgraf_from_archive(direct_sources[0], output_path, gfortran)

    resource = files("feynman_engine").joinpath(f"resources/qgraf/{_DEFAULT_SOURCE_ARCHIVE}")
    if resource.is_file():
        with as_file(resource) as resource_path:
            return _build_qgraf_from_archive(Path(resource_path), output_path, gfortran)

    raise QGrafBuildError(
        "No QGRAF source archive was found. Expected qgraf-3.6.10.tgz in "
        "feynman_engine/resources/qgraf/ (the bundled location) or set "
        "FEYNMAN_QGRAF_SOURCE to point at an alternate archive."
    )


def _build_qgraf_from_archive(source_archive: Path, output_path: Path, gfortran: str) -> Path:
    with tempfile.TemporaryDirectory(prefix="feynman_qgraf_build_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        subprocess.run(
            ["tar", "-xzf", str(source_archive), "-C", str(tmpdir_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        source_files = sorted(tmpdir_path.glob("qgraf-*.f*"))
        if not source_files:
            raise QGrafBuildError(
                f"No QGRAF Fortran source file was found in archive {source_archive}."
            )

        tmp_output = tmpdir_path / "qgraf"
        result = subprocess.run(
            [gfortran, "-O2", "-o", str(tmp_output), str(source_files[0])],
            capture_output=True,
            text=True,
            cwd=str(tmpdir_path),
        )
        if result.returncode != 0:
            raise QGrafBuildError(
                "Failed to compile QGRAF.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        tmp_output.chmod(0o755)
        shutil.copy2(tmp_output, output_path)
        output_path.chmod(0o755)
        return output_path
