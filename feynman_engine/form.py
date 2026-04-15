"""Helpers for locating and building FORM from a bundled source archive."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from importlib.resources import as_file, files
from pathlib import Path


class FormBuildError(RuntimeError):
    pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SOURCE_ARCHIVE = "form-5.0.0.tar.gz"
_USER_CACHE_DIR = Path.home() / ".cache" / "feynman-engine" / "bin"


def repo_root() -> Path:
    return _PROJECT_ROOT


def repo_form_bin() -> Path:
    return repo_root() / "bin" / "form"


def cache_form_bin() -> Path:
    return _USER_CACHE_DIR / "form"


def form_source_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("FEYNMAN_FORM_SOURCE")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(repo_root() / _DEFAULT_SOURCE_ARCHIVE)
    candidates.append(repo_root() / "feynman_engine" / "resources" / "form" / _DEFAULT_SOURCE_ARCHIVE)
    return candidates


def form_source_available() -> bool:
    if any(path.is_file() for path in form_source_candidates()):
        return True
    resource = files("feynman_engine").joinpath(f"resources/form/{_DEFAULT_SOURCE_ARCHIVE}")
    return resource.is_file()


def _writable_directory(path: Path) -> bool:
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return os.access(parent, os.W_OK)


def default_form_bin_target() -> Path:
    env_path = os.environ.get("FEYNMAN_FORM_BIN")
    if env_path:
        return Path(env_path).expanduser()

    repo_target = repo_form_bin()
    if repo_target.exists() or _writable_directory(repo_target):
        return repo_target
    return cache_form_bin()


def find_form_binary() -> Path | None:
    """Locate a usable FORM binary.

    Discovery order: FEYNMAN_FORM_BIN env → ./bin/form → user cache → system PATH.
    """
    env_path = os.environ.get("FEYNMAN_FORM_BIN")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return p

    repo_bin = repo_form_bin()
    if repo_bin.is_file():
        return repo_bin

    cache_bin = cache_form_bin()
    if cache_bin.is_file():
        return cache_bin

    system = shutil.which("form")
    if system:
        return Path(system)

    return None


def build_form(target: str | Path | None = None, force: bool = False) -> Path:
    """
    Build FORM from the bundled or configured source archive.

    FORM is a C program that uses autotools (./configure && make).
    Requires a C compiler (cc/gcc) and make.

    The default target is ``./bin/form`` when the project directory is writable,
    otherwise ``~/.cache/feynman-engine/bin/form``.
    """
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise FormBuildError(
            "A C compiler (cc, gcc, or clang) is required to build FORM. "
            "Install a C compiler first."
        )

    make = shutil.which("make") or shutil.which("gmake")
    if not make:
        raise FormBuildError(
            "make (or gmake) is required to build FORM. Install it first."
        )

    output_path = Path(target).expanduser() if target else default_form_bin_target()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    direct_sources = [path for path in form_source_candidates() if path.is_file()]
    if direct_sources:
        return _build_form_from_archive(direct_sources[0], output_path, cc, make)

    resource = files("feynman_engine").joinpath(f"resources/form/{_DEFAULT_SOURCE_ARCHIVE}")
    if resource.is_file():
        with as_file(resource) as resource_path:
            return _build_form_from_archive(Path(resource_path), output_path, cc, make)

    raise FormBuildError(
        "No FORM source archive was found. Expected form-5.0.0.tar.gz in the project root, "
        "package resources, or FEYNMAN_FORM_SOURCE."
    )


def _build_form_from_archive(
    source_archive: Path, output_path: Path, cc: str, make: str,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="feynman_form_build_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        subprocess.run(
            ["tar", "-xzf", str(source_archive), "-C", str(tmpdir_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Find the extracted directory (e.g. form-5.0.0/)
        subdirs = [d for d in tmpdir_path.iterdir() if d.is_dir() and d.name.startswith("form")]
        if not subdirs:
            raise FormBuildError(
                f"No FORM source directory found in archive {source_archive}."
            )
        build_dir = subdirs[0]

        # Configure
        configure = build_dir / "configure"
        if not configure.is_file():
            raise FormBuildError(
                f"No configure script found in {build_dir}."
            )

        env = os.environ.copy()
        env["CC"] = cc

        result = subprocess.run(
            [str(configure), "--disable-float", "--disable-parform"],
            capture_output=True,
            text=True,
            cwd=str(build_dir),
            env=env,
        )
        if result.returncode != 0:
            raise FormBuildError(
                "FORM configure failed.\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

        # Make
        result = subprocess.run(
            [make, "-j4"],
            capture_output=True,
            text=True,
            cwd=str(build_dir),
            env=env,
        )
        if result.returncode != 0:
            raise FormBuildError(
                "FORM make failed.\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

        # Find the built binary
        form_bin = build_dir / "sources" / "form"
        if not form_bin.is_file():
            # Some versions put it directly in the build dir
            form_bin = build_dir / "form"
        if not form_bin.is_file():
            raise FormBuildError(
                f"FORM binary not found after build. Searched:\n"
                f"  {build_dir / 'sources' / 'form'}\n"
                f"  {build_dir / 'form'}"
            )

        form_bin.chmod(0o755)
        shutil.copy2(form_bin, output_path)
        output_path.chmod(0o755)
        return output_path
