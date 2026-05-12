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
    parts = process.split("->", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Malformed process string: {process!r}")
    lhs, rhs = parts
    in_parts = [p for p in lhs.split() if p and p != "->"]
    out_parts = [p for p in rhs.split() if p and p != "->"]

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
    """Register an OpenLoops Process, cached on (PDG-string, amptype).

    Resets all coupling-order parameters to defaults before registration
    to avoid sticky global state from prior calls (especially EW NLO
    registrations that set loop_order_ew to non-default values).
    """
    ol = _load_openloops()
    if ol is None:
        raise RuntimeError("OpenLoops bindings unavailable")
    with _register_lock, _CwdInPrefix():
        # Reset coupling orders to defaults so this "default" register
        # doesn't inherit settings from a prior register_process_with_orders
        # call that touched the same global parameters.
        ol.set_parameter("order_qcd", -1)
        ol.set_parameter("order_ew", -1)
        ol.set_parameter("loop_order_qcd", -1)
        ol.set_parameter("loop_order_ew", -1)
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


class OpenLoopsRegistrationError(RuntimeError):
    """Raised when OpenLoops can't register a process with the requested orders.

    Most common cause: the process library installed with ``feynman install-process``
    was compiled with different coupling orders (e.g. QCD NLO only, no EW NLO).
    Install a library with EW NLO support — for ``e+ e- -> ll`` use ``eell_ew``;
    for ``pp -> ll + jet`` use ``ppllj_ew``; etc.  See
    https://openloops.hepforge.org/process_library.php for the full list.
    """


# ─── Coupling-order-aware registration (EW NLO support) ──────────────────────
#
# OpenLoops 2 supports independent QCD and EW perturbative orders.  For any
# physical process there are TWO order specifications that need to match the
# compiled library:
#
#   order_qcd / order_ew    — coupling order at *amplitude* level (= number
#                             of QCD/EW vertices in the tree diagram).
#   loop_order_qcd / loop_order_ew — coupling order of the LOOP amplitude
#                             (= tree order + extra QCD/EW vertices added
#                             by the 1-loop correction).
#
# The amplitude-squared NLO interference 2 Re(M*_tree · M_loop) therefore has
# total coupling order = order_X + loop_order_X = 2 × tree_X + 1 (for one
# extra X coupling at the loop level).
#
# Examples for e+ e- → μ+ μ-:
#   QED tree:                      order_ew=2, loop_order_ew=2 (no loop)
#   QED-only NLO (γ-only loop):    order_ew=2, loop_order_ew=3, library=*_ew
#   EW NLO (γ + Z + W loops):      order_ew=2, loop_order_ew=3, library=eell_ew
#                                   — same call, but the library has EW=2,1
#                                   compiled in (vs QED-only EW=2,1 with
#                                   QED=1 sub-flag in the .info)
#
# Examples for u u~ → e+ e- (DY):
#   QCD NLO:    order_qcd=0, order_ew=2, loop_order_qcd=1, loop_order_ew=2
#   EW NLO:     order_qcd=0, order_ew=2, loop_order_qcd=0, loop_order_ew=3


@lru_cache(maxsize=256)
def _cached_register_with_orders(
    pdg_string: str,
    amptype: str,
    order_qcd: int,
    order_ew: int,
    loop_order_qcd: int,
    loop_order_ew: int,
):
    """Register an OpenLoops Process with explicit coupling orders.

    Cached on (pdg, amptype, order_qcd, order_ew, loop_order_qcd, loop_order_ew).
    Use sentinel value -1 for any order parameter to leave it at the OL default
    (which means "any order accepted by the library").

    IMPORTANT: OL's coupling-order parameters are GLOBAL state (sticky across
    Process registrations).  We always set ALL four orders before each register
    call to avoid leaking state from a previous registration into the next.
    """
    ol = _load_openloops()
    if ol is None:
        raise RuntimeError("OpenLoops bindings unavailable")
    with _register_lock, _CwdInPrefix():
        # Always set ALL four orders to either the requested value or -1 (default)
        # to avoid sticky global state from prior Process registrations.
        ol.set_parameter("order_qcd", int(order_qcd))
        ol.set_parameter("order_ew", int(order_ew))
        ol.set_parameter("loop_order_qcd", int(loop_order_qcd))
        ol.set_parameter("loop_order_ew", int(loop_order_ew))
        return ol.Process(pdg_string, amptype)


def register_process_with_orders(
    process: str,
    amptype: str = "loop",
    order_qcd: int = -1,
    order_ew: int = -1,
    loop_order_qcd: int = -1,
    loop_order_ew: int = -1,
):
    """Register an OpenLoops process with explicit QCD/EW coupling orders.

    Parameters
    ----------
    process : str
        FeynmanEngine process string, e.g. ``"e+ e- -> mu+ mu-"``.
    amptype : {"tree", "loop", "loop2"}
        OpenLoops amplitude type.
    order_qcd, order_ew : int
        Coupling order at the *amplitude* level (number of QCD/EW vertices
        in the tree diagram).  -1 = library default.
    loop_order_qcd, loop_order_ew : int
        Coupling order of the loop amplitude.  For an EW NLO correction to a
        process whose tree is purely EW, set ``loop_order_ew = order_ew + 1``
        and ``loop_order_qcd = 0``.

    Raises
    ------
    RuntimeError if OpenLoops is unavailable.
    OpenLoopsRegistrationError if the requested orders aren't compiled in
        the installed process library.
    """
    if not is_available():
        raise RuntimeError(
            "OpenLoops Python bindings are not importable.  "
            "Run `feynman install-openloops` and `feynman install-process <name>`."
        )
    pdg = to_pdg_string(process)
    try:
        return _cached_register_with_orders(
            pdg, amptype,
            int(order_qcd), int(order_ew),
            int(loop_order_qcd), int(loop_order_ew),
        )
    except Exception as exc:
        raise OpenLoopsRegistrationError(
            f"Failed to register {process!r} (pdg={pdg!r}) with amptype={amptype}, "
            f"orders qcd={order_qcd},ew={order_ew} loop_qcd={loop_order_qcd},"
            f"loop_ew={loop_order_ew}.  Likely the relevant process library "
            f"is not installed or wasn't compiled with these orders.  "
            f"Underlying error: {exc}"
        ) from exc


def get_openloops_parameter(name: str, kind: str = "double") -> float:
    """Query a runtime OpenLoops parameter (α_s, α_qed, μ_R, …).

    OpenLoops doesn't expose its α_s as an Python attribute — you have
    to register a process first to trigger initialisation, then query
    the underlying Fortran parameter table.  This helper does the latter
    only — relies on the caller to have registered ANY process first
    (which is true for almost all code paths that need the parameter).

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
    # The OL Fortran layer needs ONE registered process before parameters
    # are queryable.  We assume the caller has already registered something
    # (which is true in all our code paths — virtual K-factor calls register
    # the EW NLO process before calling this).  If nothing has been registered
    # yet, query may return a default; that's acceptable for our use cases.
    with _register_lock, _CwdInPrefix():
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
    """Evaluate |M|² and the loop interference at a random phase-space point.

    For QCD NLO of standard processes (uses default OL orders).

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


def evaluate_loop_with_orders(
    process: str,
    sqrt_s_gev: float,
    order_qcd: int = -1,
    order_ew: int = -1,
    loop_order_qcd: int = -1,
    loop_order_ew: int = -1,
    n_psp_samples: int = 1,
    seed: int = 42,
) -> dict:
    """Evaluate the loop amplitude with explicit coupling orders.

    For EW NLO of e.g. ``e+ e- -> mu+ mu-``: pass ``order_ew=2,
    loop_order_ew=3, order_qcd=0, loop_order_qcd=0`` against a library
    compiled with EW=2,1 (e.g. ``eell_ew``).

    Parameters
    ----------
    process : str
        Engine process string.
    sqrt_s_gev : float
        Centre-of-mass energy in GeV.
    order_qcd, order_ew : int
        Tree-level coupling orders (-1 = library default).
    loop_order_qcd, loop_order_ew : int
        Loop-level coupling orders (-1 = library default).
    n_psp_samples : int
        Number of random RAMBO phase-space points to average over.  For an
        IR-safe quantity like K_virt = loop_finite/tree at fixed √s, the
        ratio is constant in s, t, u for a 2→2 s-channel process, so a
        single sample suffices.  Use n_psp_samples > 1 for processes
        where the K-factor varies non-trivially across phase space (e.g.
        2→3 with multiple invariants).
    seed : int
        RNG seed for Python-side RAMBO PSP generation, ensuring
        reproducible averages.  Without an explicit seed OpenLoops's
        internal RAMBO state is non-reproducible across calls.

    Returns
    -------
    dict
        ``tree``, ``loop_finite``, ``loop_ir1``, ``loop_ir2`` —
        averaged over the requested PSP samples.  Plus
        ``tree_psp_std``, ``loop_finite_psp_std`` — std-dev across samples
        as a phase-space-variation diagnostic.
    """
    import numpy as np
    from feynman_engine.amplitudes.phase_space import rambo_massless

    proc = register_process_with_orders(
        process, "loop",
        order_qcd=order_qcd,
        order_ew=order_ew,
        loop_order_qcd=loop_order_qcd,
        loop_order_ew=loop_order_ew,
    )

    # Determine number of final-state particles for Python RAMBO.
    # OL's proc.n is the total external particle count.
    n_final = int(proc.n) - 2  # 2 incoming
    if n_final < 1:
        # Fall back to OL's internal RAMBO if we can't reason about kinematics
        n_final = None

    trees, loops, ir1s, ir2s = [], [], [], []
    with _register_lock, _CwdInPrefix():
        if n_final is None or n_psp_samples == 1:
            # Use OL's internal RAMBO (non-reproducible but simpler)
            for _ in range(max(1, n_psp_samples)):
                me = proc.evaluate(float(sqrt_s_gev))
                trees.append(float(me.tree))
                loops.append(float(me.loop.finite))
                ir1s.append(float(me.loop.ir1))
                ir2s.append(float(me.loop.ir2))
        else:
            # Reproducible: Python RAMBO + pp-array evaluate
            rng = np.random.default_rng(seed)
            momenta, _ = rambo_massless(n_final, sqrt_s_gev, n_psp_samples, rng)
            E = sqrt_s_gev / 2.0
            p1 = np.array([E, 0.0, 0.0,  E])
            p2 = np.array([E, 0.0, 0.0, -E])
            for ev in range(n_psp_samples):
                pp = np.zeros(5 * (2 + n_final), dtype=np.float64)
                pp[0:4] = p1; pp[4] = 0.0
                pp[5:9] = p2; pp[9] = 0.0
                for k in range(n_final):
                    pp[5 * (2 + k): 5 * (2 + k) + 4] = momenta[ev, k, :]
                    pp[5 * (2 + k) + 4] = 0.0
                try:
                    me = proc.evaluate(pp)
                except Exception:
                    # Fall back to OL's internal RAMBO if pp eval fails
                    me = proc.evaluate(float(sqrt_s_gev))
                trees.append(float(me.tree))
                loops.append(float(me.loop.finite))
                ir1s.append(float(me.loop.ir1))
                ir2s.append(float(me.loop.ir2))

    trees = np.asarray(trees)
    loops = np.asarray(loops)
    ir1s = np.asarray(ir1s)
    ir2s = np.asarray(ir2s)
    return {
        "tree": float(trees.mean()),
        "tree_psp_std": float(trees.std()) if len(trees) > 1 else 0.0,
        "loop_finite": float(loops.mean()),
        "loop_finite_psp_std": float(loops.std()) if len(loops) > 1 else 0.0,
        "loop_ir1": float(ir1s.mean()),
        "loop_ir2": float(ir2s.mean()),
        "n_psp_samples": int(n_psp_samples),
    }


# ─── EW NLO library availability check ──────────────────────────────────────

# Map each FeynmanEngine process pattern to the OpenLoops library that
# provides its EW NLO loop.  Used by ``ensure_ew_nlo_library_for`` to give
# users a clear error message + the install command.
_EW_NLO_LIBRARY_MAP: dict[str, str] = {
    # e+ e- → l+ l- (massless leptons)
    "ee_to_ll": "eell_ew",
    # e+ e- → t t̄
    "ee_to_tt": "eett_ew",
    # e+ e- → V V (W+W-, ZZ)
    "ee_to_vv": "eevv_ew",
    # pp → l+ l- (Drell-Yan)
    "pp_to_ll": "pp2l2nj_ew",  # broader: includes neutrals + jets
    "pp_to_llj": "ppllj_ew",
    # pp → V (single-boson)
    "pp_to_v": "ppvj_ew",
    "pp_to_vj": "ppvj_ew",
    "pp_to_vv": "ppvv_ew",
    "pp_to_vvj": "ppvvj_ew",
    # pp → t t̄
    "pp_to_tt": "pptt_ew",
    "pp_to_ttj": "ppttj_ew",
    # pp → t W
    "pp_to_tw": "pptw_ew",
    # pp → H + V
    "pp_to_hv": "pphv_ew",
    "pp_to_hll": "pphll_ew",
    # pp → H + tt
    "pp_to_htt": "pphtt_ew",
    "pp_to_httj": "pphttj_ew",
    # pp → 4l
    "pp_to_4l": "pp4lj_ew",
}


def ew_nlo_library_for(process: str) -> Optional[str]:
    """Return the OpenLoops EW NLO library name for a given process, if known.

    Performs a coarse pattern match on the in/out particle content.  Returns
    None if no known library matches — the caller can either fall back to
    the analytic Sudakov path or surface a "library not available" error.
    """
    if "->" not in process:
        return None
    parts = process.split("->", maxsplit=1)
    if len(parts) != 2:
        return None
    lhs, rhs = parts
    in_parts = [p for p in lhs.split() if p and p != "->"]
    out_parts = [p for p in rhs.split() if p and p != "->"]

    in_set = set(p.replace("~", "").rstrip("+-") for p in in_parts)
    out_set = set(p.replace("~", "").rstrip("+-") for p in out_parts)

    leptons = {"e", "mu", "tau"}
    quarks = {"u", "d", "s", "c", "b"}
    bosons = {"W", "Z", "gamma", "ph", "a"}

    def is_lepton_pair(seq: list[str]) -> bool:
        if len(seq) != 2:
            return False
        a, b = seq
        return (a == "e+" and b == "e-") or (a == "e-" and b == "e+") or \
               (a == "mu+" and b == "mu-") or (a == "mu-" and b == "mu+") or \
               (a == "tau+" and b == "tau-") or (a == "tau-" and b == "tau+") or \
               (a in {"e+", "mu+", "tau+"} and b in {"e-", "mu-", "tau-"}) or \
               (a in {"e-", "mu-", "tau-"} and b in {"e+", "mu+", "tau+"})

    is_pp = "p" in in_set or in_set <= quarks | {"g"}
    is_ee = is_lepton_pair(in_parts)
    out_top = "t" in out_set
    out_lepton_pair = is_lepton_pair(out_parts)
    out_vv = sum(1 for p in out_parts if p in {"W+", "W-", "Z"}) == 2
    out_h = "H" in out_parts or "h" in out_parts
    out_jets = sum(1 for p in out_parts if p == "g") + \
               sum(1 for p in out_parts if p.replace("~", "") in quarks)

    out_v_count = sum(1 for p in out_parts if p in {"W+", "W-", "Z"})

    # e+ e- collisions
    if is_ee:
        if out_top and out_h:
            return "eehtt_ew" if "eehtt_ew" in installed_processes() else "eehtt"
        if out_top:
            return "eett_ew"
        if out_h and out_v_count >= 1:
            # ee → V H (e.g. ee → Z H, ee → W+ W- H eventually) — wired
            # 2026-05-11 to fix missing ZH NLO EW routing.
            return "eehv_ew"
        if out_h and out_lepton_pair:
            # ee → ℓ⁺ℓ⁻ H (resolved H decay via Z → ll̄)
            return "eehll_ew"
        if out_vv:
            return "eevv_ew"
        if out_lepton_pair:
            return "eell_ew"

    # pp collisions
    if is_pp:
        if out_top and out_h:
            return "pphtt_ew"
        if out_top:
            if out_jets >= 1:
                return "ppttj_ew"
            return "pptt_ew"
        if out_h and out_v_count >= 2:
            # pp → H V V (e.g. H W W) — NLO EW library
            return "pphvv_ew"
        if out_vv:
            return "ppvv_ew"
        if out_h and out_lepton_pair:
            return "pphll_ew"
        if out_lepton_pair:
            if out_jets >= 1:
                return "ppllj_ew"
            return "pp2l2nj_ew"
        if out_h and out_v_count >= 1:
            # pp → V H (Higgsstrahlung NLO EW)
            return "pphv_ew"

    return None


def has_ew_nlo_library(process: str) -> bool:
    """Quick check: is an EW NLO library installed for this process?

    Returns True if the relevant library appears in ``installed_processes()``.
    Use ``ew_nlo_library_for(process)`` to discover the expected name.
    """
    libname = ew_nlo_library_for(process)
    if libname is None:
        return False
    return libname in installed_processes()
