"""Bridge to OpenLoops 2 for tree + one-loop amplitude evaluation.

OpenLoops 2 (Buccioni, Lang, Lindert, Maierhöfer, Pozzorini, Zhang, Zoller,
EPJ C 79 (2019) 866, arXiv:1907.13071) generates and numerically evaluates
tree and one-loop amplitudes for arbitrary Standard Model processes.  In
FeynmanEngine it provides the path to "generic NLO" — any process whose
OpenLoops process library is installed gets a full virtual + tree from
OpenLoops, combined with our Catani-Seymour dipole subtraction for the
real-emission piece (see ``nlo_cross_section_openloops``).

This module:

1. **Auto-discovers** an OpenLoops install at import time (mirrors the
   LHAPDF auto-discovery in ``pdf.py``).
2. Lazily imports the OpenLoops Python bindings on first use.
3. Provides ``register_process`` / ``evaluate_loop_squared`` thin wrappers
   that translate FeynmanEngine process strings (``"u u~ -> e+ e-"``) to
   OpenLoops PDG strings (``"2 -2 -> 11 -11"``).

If OpenLoops is not installed, ``is_available()`` returns False and the
trust system falls back to BLOCKED for processes that would have used it.
"""
from __future__ import annotations

import glob
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

from feynman_engine.openloops import (
    default_openloops_install_prefix,
    is_openloops_installed_at,
)


# ─── Auto-discovery ───────────────────────────────────────────────────────────

# Probe the same location set we use for LHAPDF.  /opt/openloops is the Docker
# default; /tmp/openloops-install is the `feynman install-openloops` default in
# CI/dev; /usr/local matches a manual ``./scons && cp -r ...``.
_PROBE_LOCATIONS: tuple[str, ...] = (
    "/opt/openloops",
    "/tmp/openloops-install",
    "/usr/local/openloops",
    str(Path.home() / ".local" / "openloops"),
    str(Path.home() / "openloops-install"),
)


def _try_locate_openloops_install() -> Optional[str]:
    """Probe common install prefixes for an OpenLoops install.

    Returns the install prefix as a string, or None if not found.

    Side effect: when found, prepends ``<prefix>/pyol/tools`` to ``sys.path``
    and exports ``OPENLOOPS_PREFIX``.  The actual ``import openloops`` is
    deferred — it's done lazily inside an ``_open_in_prefix()`` chdir so
    OpenLoops's CWD-relative dylib lookup resolves to ``<prefix>/lib``.
    """
    # Already importable?
    try:
        import openloops  # noqa: F401
        return None
    except ImportError:
        pass

    # Honor an explicit FEYNMAN_OPENLOOPS_PREFIX or OPENLOOPS_PREFIX override.
    explicit = os.environ.get("FEYNMAN_OPENLOOPS_PREFIX") or os.environ.get("OPENLOOPS_PREFIX")
    candidates = [explicit] if explicit else []
    candidates += list(_PROBE_LOCATIONS)
    # Last resort: configured default (honors FEYNMAN_OPENLOOPS_PREFIX too).
    candidates.append(str(default_openloops_install_prefix()))

    for prefix in candidates:
        if not prefix:
            continue
        prefix_path = Path(prefix).expanduser()
        if not is_openloops_installed_at(prefix_path):
            continue

        py_dir = str(prefix_path / "pyol" / "tools")
        if py_dir not in sys.path:
            sys.path.insert(0, py_dir)

        for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
            existing = os.environ.get(var, "")
            paths = [p for p in existing.split(":") if p]
            for sub in ("lib", "proclib"):
                lib_dir = str(prefix_path / sub)
                if Path(lib_dir).is_dir() and lib_dir not in paths:
                    paths.append(lib_dir)
            os.environ[var] = ":".join(paths)

        os.environ["OPENLOOPS_PREFIX"] = str(prefix_path)
        return str(prefix_path)

    return None


# Run discovery on module import so `import openloops` works downstream.
_try_locate_openloops_install()


# ─── Availability ────────────────────────────────────────────────────────────


def install_prefix() -> Optional[Path]:
    """Return the discovered install prefix, or None."""
    p = os.environ.get("OPENLOOPS_PREFIX")
    return Path(p) if p else None


class _CwdInPrefix:
    """Temporarily chdir into the OpenLoops install prefix.

    OpenLoops's ``openloops.py`` resolves the core dylib via a CWD-relative
    path (``pathlib.Path().absolute() / "lib/libopenloops.dylib"``), and
    looks up process libraries under ``./proclib``.  We localize the chdir
    to the openloops import / process registration so the rest of the
    application keeps its own working directory.
    """

    def __enter__(self):
        prefix = install_prefix()
        if prefix is None:
            raise RuntimeError("OpenLoops install prefix not discovered")
        self._old = os.getcwd()
        os.chdir(prefix)
        return prefix

    def __exit__(self, exc_type, exc, tb):
        os.chdir(self._old)


_openloops_module = None


def _load_openloops():
    """Import the openloops module with CWD set to the install prefix."""
    global _openloops_module
    if _openloops_module is not None:
        return _openloops_module
    if install_prefix() is None:
        return None
    try:
        with _CwdInPrefix():
            import openloops  # noqa: WPS433 — local import is intentional
        _openloops_module = openloops
        return openloops
    except Exception:
        return None


def is_available() -> bool:
    """Return True if the OpenLoops Python bindings can be imported."""
    return _load_openloops() is not None


def installed_processes() -> list[str]:
    """List process libraries installed at the discovered prefix."""
    prefix = install_prefix()
    if not prefix:
        return []
    proclib = prefix / "proclib"
    if not proclib.is_dir():
        return []
    procs: set[str] = set()
    for info in proclib.glob("libopenloops_*.info"):
        stem = info.stem[len("libopenloops_"):]
        if "_" in stem:
            stem = stem.rsplit("_", 1)[0]
        procs.add(stem)
    return sorted(procs)


# ─── Process string translation ──────────────────────────────────────────────

# OpenLoops uses PDG numeric IDs for particles in its register_process call.
# Map our amplitude module's particle names to PDG IDs.
_PDG: dict[str, int] = {
    # Quarks
    "u": 2, "u~": -2, "ubar": -2,
    "d": 1, "d~": -1, "dbar": -1,
    "c": 4, "c~": -4, "cbar": -4,
    "s": 3, "s~": -3, "sbar": -3,
    "t": 6, "t~": -6, "tbar": -6,
    "b": 5, "b~": -5, "bbar": -5,
    # Leptons
    "e-": 11, "e+": -11,
    "mu-": 13, "mu+": -13,
    "tau-": 15, "tau+": -15,
    "nue": 12, "nuebar": -12, "nu_e": 12,
    "numu": 14, "numubar": -14, "nu_mu": 14,
    "nutau": 16, "nutaubar": -16, "nu_tau": 16,
    # Bosons
    "g": 21, "gluon": 21,
    "ph": 22, "photon": 22, "a": 22, "gamma": 22,
    "Z": 23, "z": 23,
    "W+": 24, "W-": -24, "w+": 24, "w-": -24,
    "H": 25, "h": 25, "higgs": 25,
    # Hadrons treated as parton initiators (for OpenLoops we expand at the
    # PDF level — caller should replace 'p'/'pbar' before passing here).
}


def to_pdg_string(process: str) -> str:
    """Translate a FeynmanEngine process string to OpenLoops PDG string.

    Examples
    --------
    >>> to_pdg_string("u u~ -> e+ e-")
    '2 -2 -> -11 11'
    """
    if "->" not in process:
        raise ValueError(f"Process must contain '->': {process!r}")
    lhs, rhs = process.split("->")
    in_parts = [p for p in lhs.split() if p]
    out_parts = [p for p in rhs.split() if p]

    def _lookup(name: str) -> int:
        if name in _PDG:
            return _PDG[name]
        # Try lowercase / common aliases
        lower = name.lower()
        if lower in _PDG:
            return _PDG[lower]
        raise ValueError(f"No PDG id for particle {name!r} in process {process!r}")

    in_ids = " ".join(str(_lookup(p)) for p in in_parts)
    out_ids = " ".join(str(_lookup(p)) for p in out_parts)
    return f"{in_ids} -> {out_ids}"


# ─── Process registration cache ──────────────────────────────────────────────

_register_lock = threading.Lock()


@lru_cache(maxsize=128)
def _cached_register(pdg_string: str, amptype: str):
    """Register an OpenLoops Process, cached on (PDG-string, amptype)."""
    ol = _load_openloops()
    if ol is None:
        raise RuntimeError("OpenLoops bindings unavailable")
    with _register_lock, _CwdInPrefix():
        return ol.Process(pdg_string, amptype)


def register_process(process: str, amptype: str = "loop"):
    """Register an OpenLoops process.  ``amptype`` is ``"tree"`` or ``"loop"``."""
    if not is_available():
        raise RuntimeError(
            "OpenLoops Python bindings are not importable.  "
            "Run `feynman install-openloops` and `feynman install-process <name>`."
        )
    pdg = to_pdg_string(process)
    return _cached_register(pdg, amptype)


def evaluate_loop_squared(process: str, sqrt_s_gev: float) -> dict:
    """Evaluate |M|² and the loop interference at √s.

    Returns
    -------
    dict
        With keys:
          - ``tree``  — |M_tree|²
          - ``loop_finite`` — finite part of 2·Re(M*_tree · M_loop)
          - ``loop_ir1``    — 1/ε pole coefficient (for IR cross-check)
          - ``loop_ir2``    — 1/ε² pole coefficient (should be zero for IR-safe NLO)

    The result is for a *random* phase-space point at √s.
    """
    proc = register_process(process, "loop")
    with _register_lock, _CwdInPrefix():
        me = proc.evaluate(float(sqrt_s_gev))
    return {
        "tree": float(me.tree),
        "loop_finite": float(me.loop.finite),
        "loop_ir1": float(me.loop.ir1),
        "loop_ir2": float(me.loop.ir2),
    }
