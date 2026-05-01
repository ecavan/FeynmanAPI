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
    prefix = install_prefix()
    if prefix is None:
        return None
    try:
        with _CwdInPrefix():
            import openloops  # noqa: WPS433 — local import is intentional
            # Tell the Fortran layer where the install lives so its
            # process-library lookup uses an absolute path rather than
            # CWD-relative ``proclib/``.  Without this, register_process
            # calls Fortran ``exit()`` on a missing proclib even when CWD
            # is correct (the chdir is itself fragile across threads).
            try:
                openloops.set_parameter("install_path", str(prefix) + "/")
            except Exception:
                pass
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


def get_openloops_parameter(name: str, kind: str = "double") -> float:
    """Query a runtime OpenLoops parameter (α_s, α_qed, μ_R, …).

    OpenLoops doesn't expose its α_s as an Python attribute — you have to
    register a process first to trigger initialisation, then query the
    underlying Fortran parameter table.  This helper does both.

    Parameters
    ----------
    name : str
        Parameter name as known to OpenLoops (e.g. ``"alpha_s"``,
        ``"alpha_qed"``, ``"mu"``, ``"mass(6)"``).
    kind : str
        ``"double"`` or ``"int"``.

    Returns
    -------
    float
        Current value of the parameter in OpenLoops's internal scheme.
    """
    ol = _load_openloops()
    if ol is None:
        raise RuntimeError("OpenLoops not available; cannot query parameter")
    # Trigger init by registering a small process if not already done
    with _register_lock, _CwdInPrefix():
        try:
            ol.Process("1 -1 -> 11 -11", "loop")
        except Exception:
            pass  # already registered or library missing — we still query
        if kind == "double":
            return float(ol.get_parameter_double(name))
        if kind == "int":
            return int(ol.get_parameter_int(name))
        raise ValueError(f"Unknown parameter kind: {kind!r}")


def evaluate_color_correlated_amplitude(
    process: str, momenta_5xn: "np.ndarray",
) -> tuple[float, "np.ndarray"]:
    """Evaluate the colour-correlated Born amplitude via OpenLoops.

    Returns (|M_tree|², cc_matrix_flat) where cc_matrix_flat is a length
    n*(n-1)/2 array containing ⟨B|T_i·T_k|B⟩ for each unordered pair (i,k)
    with i<k.

    For 2-coloured-leg processes (q q̄ → singlets) this reproduces -C_F
    on the (q, q̄) pair entry.  For multi-coloured-leg processes (e.g.
    q q̄ → t t̄, g g → t t̄) it gives the full colour-flow-decomposed
    correlator matrix that's needed for CS dipole assembly beyond the
    2-leg approximation.

    Parameters
    ----------
    process : str
        Engine process string.  Must be registerable via OpenLoops.
    momenta_5xn : (5*n,) array
        Flattened momentum array in OpenLoops layout (E, px, py, pz, m
        for each particle).

    Returns
    -------
    tree : float
    cc : ndarray of length n*(n-1)/2  (i<k order, flattened)
    """
    import ctypes
    import numpy as np
    ol = _load_openloops()
    if ol is None:
        raise RuntimeError("OpenLoops bindings unavailable")
    with _register_lock, _CwdInPrefix():
        # Bypass Process('cc') because OpenLoops's Python wrapper has a
        # Python 3 division bug in the cc-buffer allocation.  Use the
        # 'tree' Process to register, then call evaluate_cc_c directly.
        proc = ol.Process(_pdg_string(process), "tree")
        n = proc.n
        n_pairs = (n * (n - 1)) // 2
        pp = (ctypes.c_double * (5 * n))(*momenta_5xn.astype(np.float64).tolist())
        tree_buf = (ctypes.c_double * 1)()
        cc_buf = (ctypes.c_double * n_pairs)()
        ewcc_buf = (ctypes.c_double * 1)()
        # Make sure OpenLoops is started before the first eval call
        if not ol.start.started:
            ol.start()
            ol.start.started = True
        ol.evaluate_cc_c(proc.id, pp, tree_buf, cc_buf, ewcc_buf)
        return float(tree_buf[0]), np.array(cc_buf[:], dtype=np.float64)


def _pdg_string(process: str) -> str:
    """Translate engine process string to OpenLoops PDG string."""
    return to_pdg_string(process)


def get_openloops_alpha_s() -> float:
    """Return α_s as currently configured inside OpenLoops.

    OpenLoops's default is α_s(μ_R=100 GeV) = 0.1258086856…  This differs
    from the PDG value α_s(M_Z) = 0.1179 because OpenLoops's μ_R default
    is 100 GeV, not M_Z.  Use this value for IR-pole cancellation
    consistency with the loop_finite/loop_ir1/loop_ir2 returned by
    ``evaluate_loop_squared``.
    """
    try:
        return get_openloops_parameter("alpha_s")
    except Exception:
        return 0.118  # fallback to PDG α_s(M_Z)


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
