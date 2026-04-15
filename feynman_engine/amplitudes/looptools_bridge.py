"""LoopTools ctypes bridge for numerical evaluation of PV scalar integrals.

LoopTools (Hahn & Pérez-Victoria, CPC 118 (1999) 153) evaluates the
Passarino-Veltman scalar integrals A₀, B₀, C₀, D₀ in dimensional
regularisation to arbitrary numerical precision.

Installation
------------
Run the bundled installer::

    feynman install-looptools

This compiles LoopTools-2.16 from the bundled source archive and places
the shared library at ``<project_root>/bin/liblooptools.{so,dylib}``.

Alternatively, set the environment variable::

    export LOOPTOOLS_LIB=/path/to/liblooptools.{so,dylib}

Status
------
- Library loading and ``is_available()`` check: **implemented**
- B₀, C₀, D₀ stubs with correct Fortran symbol names: **implemented**
- Full ctypes signatures (all variants A0…D0): **implemented**
- Tested against LoopTools reference values: **pending** (requires compiled library)

Notes on Fortran symbol names
------------------------------
gfortran on macOS/Linux exports Fortran names as lower-case with a trailing
underscore: ``b0_``, ``c0_``, ``d0_``, ``a0_``.  The LoopTools library adds a
``ltini_`` / ``ltexi_`` pair for initialisation (set μ² = renormalisation scale).

All arguments are passed by reference (Fortran calling convention).
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

# ── Library loading ───────────────────────────────────────────────────────────

_ENV_VAR = "LOOPTOOLS_LIB"
_lib: Optional[ctypes.CDLL] = None


def _candidate_paths() -> list[Path]:
    """Ordered list of paths to try when LOOPTOOLS_LIB is not set."""
    root = Path(__file__).parent.parent.parent  # project root
    return [
        # Primary install location (feynman install-looptools)
        root / "bin" / "liblooptools.dylib",
        root / "bin" / "liblooptools.so",
        # Legacy lib/ location
        root / "lib" / "liblooptools.dylib",
        root / "lib" / "liblooptools.so",
        # System-wide locations
        Path("/usr/local/lib/liblooptools.so"),
        Path("/opt/homebrew/lib/liblooptools.dylib"),
        Path("/opt/homebrew/lib/liblooptools.so"),
    ]


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    path = os.environ.get(_ENV_VAR)
    if not path:
        for candidate in _candidate_paths():
            if candidate.exists():
                path = str(candidate)
                break

    if not path:
        raise RuntimeError(
            "LoopTools library not found.\n"
            f"  Set {_ENV_VAR}=/path/to/liblooptools.{{so,dylib}}, or\n"
            "  Compile LoopTools-2.16 and copy the shared library to lib/liblooptools.{{so,dylib}}.\n"
            "  See feynman_engine/amplitudes/looptools_bridge.py for instructions."
        )

    _lib = ctypes.CDLL(path)

    # Initialise LoopTools (set renormalisation scale μ² = 1 GeV² by default).
    try:
        _lib.ltini_()
    except AttributeError:
        pass   # some builds initialise automatically

    return _lib


def is_available() -> bool:
    """Return True if the LoopTools shared library is loadable."""
    try:
        _load()
        return True
    except RuntimeError:
        return False


def set_mu_sq(mu_sq: float) -> None:
    """Set the dimensional-regularisation scale μ² (GeV²)."""
    lib = _load()
    lib.setmudim_(ctypes.byref(ctypes.c_double(mu_sq)))


# ── Scalar integrals ──────────────────────────────────────────────────────────
# All functions follow the LoopTools Fortran interface:
#   DOUBLE COMPLEX FUNCTION B0(p2, m1sq, m2sq)
# Arguments are DOUBLE PRECISION, passed BY REFERENCE.
#
# Calling convention: gfortran returns DOUBLE COMPLEX as a C struct {double re, double im}
# (the compiler places it in two FP registers on ARM64 / in a returned struct on x86-64).
# argtypes: each argument is POINTER(c_double) (Fortran by-reference).

class _Complex16(ctypes.Structure):
    """Maps to DOUBLE COMPLEX returned by value from gfortran."""
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


def _call(lib_func, *vals: float) -> complex:
    """Call a LoopTools Fortran function with n scalar arguments (by reference).

    The function must have signature  DOUBLE COMPLEX FUNCTION f(a, b, ...)
    where every argument is DOUBLE PRECISION (passed by reference).
    """
    n = len(vals)
    lib_func.restype = _Complex16
    lib_func.argtypes = [ctypes.POINTER(ctypes.c_double)] * n
    result = lib_func(*[ctypes.byref(ctypes.c_double(v)) for v in vals])
    return complex(result.re, result.im)


def A0(m_sq: float) -> complex:
    """Scalar tadpole A₀(m²).

    Diverges as 1/ε in dim-reg; the finite part depends on the renormalisation
    scheme.  A₀(m²) = m²(log(μ²/m²) + 1) in MS-bar (LoopTools returns the
    finite part only, UV pole subtracted via dim-reg).
    """
    lib = _load()
    return _call(lib.a0_, m_sq)


def B0(p_sq: float, m1_sq: float, m2_sq: float) -> complex:
    """Scalar bubble B₀(p², m₁², m₂²).

    UV-divergent (1/ε pole), IR-finite for m > 0.
    """
    lib = _load()
    return _call(lib.b0_, p_sq, m1_sq, m2_sq)


def B1(p_sq: float, m1_sq: float, m2_sq: float) -> complex:
    """Coefficient of p^μ in the vector integral B^μ = B₁ p^μ."""
    lib = _load()
    return _call(lib.b1_, p_sq, m1_sq, m2_sq)


def B00(p_sq: float, m1_sq: float, m2_sq: float) -> complex:
    """Coefficient of g^{μν} in the tensor integral B^{μν} = B₀₀ g^{μν} + B₁₁ p^μp^ν."""
    lib = _load()
    return _call(lib.b00_, p_sq, m1_sq, m2_sq)


def B11(p_sq: float, m1_sq: float, m2_sq: float) -> complex:
    """Coefficient of p^μp^ν in the tensor integral B^{μν} = B₀₀ g^{μν} + B₁₁ p^μp^ν."""
    lib = _load()
    return _call(lib.b11_, p_sq, m1_sq, m2_sq)


# ── Indexed accessor helpers ─────────────────────────────────────────────────
# LoopTools provides c0i_ and d0i_ as generic coefficient accessors.
# The named wrapper functions (c1_, c00_, d1_, d00_ etc.) are NOT exported
# in all builds.  The indexed API is the only reliable interface for tensor
# coefficients.
#
# Index conventions (from clooptools.h):
#   C: cc0=0, cc1=1, cc2=2, cc00=3, cc11=4, cc12=5, cc22=6
#   D: dd0=0, dd1=1, dd2=2, dd3=3, dd00=4, dd11=5, dd12=6, dd13=7,
#      dd22=8, dd23=9, dd33=10

def _c0i(index: int, p1_sq: float, p2_sq: float, p12_sq: float,
         m1_sq: float, m2_sq: float, m3_sq: float) -> complex:
    """Call c0i_(index, ...) — generic C-point coefficient accessor."""
    lib = _load()
    lib.c0i_.restype = _Complex16
    lib.c0i_.argtypes = [ctypes.POINTER(ctypes.c_int)] + [ctypes.POINTER(ctypes.c_double)] * 6
    idx = ctypes.c_int(index)
    result = lib.c0i_(
        ctypes.byref(idx),
        *[ctypes.byref(ctypes.c_double(v)) for v in [p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq]],
    )
    return complex(result.re, result.im)


def _d0i(index: int, p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
         p12_sq: float, p23_sq: float,
         m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Call d0i_(index, ...) — generic D-point coefficient accessor."""
    lib = _load()
    lib.d0i_.restype = _Complex16
    lib.d0i_.argtypes = [ctypes.POINTER(ctypes.c_int)] + [ctypes.POINTER(ctypes.c_double)] * 10
    idx = ctypes.c_int(index)
    result = lib.d0i_(
        ctypes.byref(idx),
        *[ctypes.byref(ctypes.c_double(v)) for v in [p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq]],
    )
    return complex(result.re, result.im)


# ── C-point tensor coefficients (via c0i_) ──────────────────────────────────

def C1(p1_sq: float, p2_sq: float, p12_sq: float,
       m1_sq: float, m2_sq: float, m3_sq: float) -> complex:
    """Coefficient of p₁^μ in the triangle vector integral."""
    return _c0i(1, p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq)

def C2(p1_sq: float, p2_sq: float, p12_sq: float,
       m1_sq: float, m2_sq: float, m3_sq: float) -> complex:
    """Coefficient of p₂^μ in the triangle vector integral."""
    return _c0i(2, p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq)

def C00(p1_sq: float, p2_sq: float, p12_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float) -> complex:
    """Coefficient of g^{μν} in the rank-2 triangle integral."""
    return _c0i(3, p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq)

def C11(p1_sq: float, p2_sq: float, p12_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float) -> complex:
    """Coefficient of p₁^μp₁^ν in the rank-2 triangle integral."""
    return _c0i(4, p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq)

def C12(p1_sq: float, p2_sq: float, p12_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float) -> complex:
    """Coefficient of (p₁^μp₂^ν + p₂^μp₁^ν)/2 in the rank-2 triangle integral."""
    return _c0i(5, p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq)

def C22(p1_sq: float, p2_sq: float, p12_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float) -> complex:
    """Coefficient of p₂^μp₂^ν in the rank-2 triangle integral."""
    return _c0i(6, p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq)


# ── D-point tensor coefficients (via d0i_) ──────────────────────────────────

def D1(p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
       p12_sq: float, p23_sq: float,
       m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Coefficient of p₁^μ in the box vector integral."""
    return _d0i(1, p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq)

def D2(p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
       p12_sq: float, p23_sq: float,
       m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Coefficient of p₂^μ in the box vector integral."""
    return _d0i(2, p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq)

def D3(p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
       p12_sq: float, p23_sq: float,
       m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Coefficient of p₃^μ in the box vector integral."""
    return _d0i(3, p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq)

def D00(p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
        p12_sq: float, p23_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Coefficient of g^{μν} in the rank-2 box integral."""
    return _d0i(4, p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq)

def D11(p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
        p12_sq: float, p23_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Coefficient of p₁^μp₁^ν in the rank-2 box integral."""
    return _d0i(5, p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq)

def D12(p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
        p12_sq: float, p23_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Coefficient of (p₁^μp₂^ν + p₂^μp₁^ν)/2 in the rank-2 box integral."""
    return _d0i(6, p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq)

def D13(p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
        p12_sq: float, p23_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Coefficient of (p₁^μp₃^ν + p₃^μp₁^ν)/2 in the rank-2 box integral."""
    return _d0i(7, p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq)

def D22(p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
        p12_sq: float, p23_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Coefficient of p₂^μp₂^ν in the rank-2 box integral."""
    return _d0i(8, p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq)

def D23(p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
        p12_sq: float, p23_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Coefficient of (p₂^μp₃^ν + p₃^μp₂^ν)/2 in the rank-2 box integral."""
    return _d0i(9, p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq)

def D33(p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
        p12_sq: float, p23_sq: float,
        m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float) -> complex:
    """Coefficient of p₃^μp₃^ν in the rank-2 box integral."""
    return _d0i(10, p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq, m1_sq, m2_sq, m3_sq, m4_sq)


def C0(
    p1_sq: float, p2_sq: float, p12_sq: float,
    m1_sq: float, m2_sq: float, m3_sq: float,
) -> complex:
    """Scalar triangle C₀(p₁², p₂², (p₁+p₂)², m₁², m₂², m₃²).

    UV-finite; IR-divergent when internal masses vanish.
    """
    lib = _load()
    return _call(lib.c0_, p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq)


def D0(
    p1_sq: float, p2_sq: float, p3_sq: float, p4_sq: float,
    p12_sq: float, p23_sq: float,
    m1_sq: float, m2_sq: float, m3_sq: float, m4_sq: float,
) -> complex:
    """Scalar box D₀(p₁²,…,p₄², p₁₂², p₂₃², m₁²,…,m₄²).

    UV-finite; IR-divergent for massless internal lines.
    LoopTools uses the Passarino-Veltman convention with 10 arguments.
    """
    lib = _load()
    return _call(
        lib.d0_,
        p1_sq, p2_sq, p3_sq, p4_sq, p12_sq, p23_sq,
        m1_sq, m2_sq, m3_sq, m4_sq,
    )


# ── Convenience evaluator ─────────────────────────────────────────────────────

def evaluate_pv_expansion(expansion) -> Optional[complex]:
    """Evaluate a PVExpansion numerically using LoopTools.

    Returns None if LoopTools is not available or if any integral fails.
    Each term's coefficient must be a pure Python float or a SymPy expression
    that evaluates to a float after substituting kinematics.

    Parameters
    ----------
    expansion : PVExpansion
        Output of ``loop.pv_reduce()``.

    Returns
    -------
    complex or None
        The numerical value of the loop amplitude, or None on failure.
    """
    if not is_available():
        return None

    from feynman_engine.amplitudes.loop import (
        A0Integral,
        B0Integral, B1Integral, B00Integral, B11Integral,
        C0Integral, C1Integral, C2Integral, C00Integral, C11Integral, C12Integral, C22Integral,
        D0Integral, D00Integral, D1Integral, D2Integral, D3Integral,
        D11Integral, D12Integral, D13Integral, D22Integral, D23Integral, D33Integral,
    )

    result = complex(0.0)
    try:
        for integral, coeff in expansion.terms.items():
            coeff_val = float(coeff)
            # 1-point
            if isinstance(integral, A0Integral):
                val = A0(float(integral.m_sq))
            # 2-point
            elif isinstance(integral, B0Integral):
                val = B0(float(integral.p_sq), float(integral.m1_sq), float(integral.m2_sq))
            elif isinstance(integral, B1Integral):
                val = B1(float(integral.p_sq), float(integral.m1_sq), float(integral.m2_sq))
            elif isinstance(integral, B00Integral):
                val = B00(float(integral.p_sq), float(integral.m1_sq), float(integral.m2_sq))
            elif isinstance(integral, B11Integral):
                val = B11(float(integral.p_sq), float(integral.m1_sq), float(integral.m2_sq))
            # 3-point
            elif isinstance(integral, C0Integral):
                val = C0(float(integral.p1_sq), float(integral.p2_sq), float(integral.p12_sq),
                         float(integral.m1_sq), float(integral.m2_sq), float(integral.m3_sq))
            elif isinstance(integral, C1Integral):
                val = C1(float(integral.p1_sq), float(integral.p2_sq), float(integral.p12_sq),
                         float(integral.m1_sq), float(integral.m2_sq), float(integral.m3_sq))
            elif isinstance(integral, C2Integral):
                val = C2(float(integral.p1_sq), float(integral.p2_sq), float(integral.p12_sq),
                         float(integral.m1_sq), float(integral.m2_sq), float(integral.m3_sq))
            elif isinstance(integral, C00Integral):
                val = C00(float(integral.p1_sq), float(integral.p2_sq), float(integral.p12_sq),
                          float(integral.m1_sq), float(integral.m2_sq), float(integral.m3_sq))
            elif isinstance(integral, C11Integral):
                val = C11(float(integral.p1_sq), float(integral.p2_sq), float(integral.p12_sq),
                          float(integral.m1_sq), float(integral.m2_sq), float(integral.m3_sq))
            elif isinstance(integral, C12Integral):
                val = C12(float(integral.p1_sq), float(integral.p2_sq), float(integral.p12_sq),
                          float(integral.m1_sq), float(integral.m2_sq), float(integral.m3_sq))
            elif isinstance(integral, C22Integral):
                val = C22(float(integral.p1_sq), float(integral.p2_sq), float(integral.p12_sq),
                          float(integral.m1_sq), float(integral.m2_sq), float(integral.m3_sq))
            # 4-point
            elif isinstance(integral, D0Integral):
                val = D0(float(integral.p1_sq), float(integral.p2_sq),
                         float(integral.p3_sq), float(integral.p4_sq),
                         float(integral.p12_sq), float(integral.p23_sq),
                         float(integral.m1_sq), float(integral.m2_sq),
                         float(integral.m3_sq), float(integral.m4_sq))
            elif isinstance(integral, D00Integral):
                val = D00(float(integral.p1_sq), float(integral.p2_sq),
                          float(integral.p3_sq), float(integral.p4_sq),
                          float(integral.p12_sq), float(integral.p23_sq),
                          float(integral.m1_sq), float(integral.m2_sq),
                          float(integral.m3_sq), float(integral.m4_sq))
            elif isinstance(integral, D1Integral):
                val = D1(float(integral.p1_sq), float(integral.p2_sq),
                         float(integral.p3_sq), float(integral.p4_sq),
                         float(integral.p12_sq), float(integral.p23_sq),
                         float(integral.m1_sq), float(integral.m2_sq),
                         float(integral.m3_sq), float(integral.m4_sq))
            elif isinstance(integral, D2Integral):
                val = D2(float(integral.p1_sq), float(integral.p2_sq),
                         float(integral.p3_sq), float(integral.p4_sq),
                         float(integral.p12_sq), float(integral.p23_sq),
                         float(integral.m1_sq), float(integral.m2_sq),
                         float(integral.m3_sq), float(integral.m4_sq))
            elif isinstance(integral, D3Integral):
                val = D3(float(integral.p1_sq), float(integral.p2_sq),
                         float(integral.p3_sq), float(integral.p4_sq),
                         float(integral.p12_sq), float(integral.p23_sq),
                         float(integral.m1_sq), float(integral.m2_sq),
                         float(integral.m3_sq), float(integral.m4_sq))
            elif isinstance(integral, D11Integral):
                val = D11(float(integral.p1_sq), float(integral.p2_sq),
                          float(integral.p3_sq), float(integral.p4_sq),
                          float(integral.p12_sq), float(integral.p23_sq),
                          float(integral.m1_sq), float(integral.m2_sq),
                          float(integral.m3_sq), float(integral.m4_sq))
            elif isinstance(integral, D12Integral):
                val = D12(float(integral.p1_sq), float(integral.p2_sq),
                          float(integral.p3_sq), float(integral.p4_sq),
                          float(integral.p12_sq), float(integral.p23_sq),
                          float(integral.m1_sq), float(integral.m2_sq),
                          float(integral.m3_sq), float(integral.m4_sq))
            elif isinstance(integral, D13Integral):
                val = D13(float(integral.p1_sq), float(integral.p2_sq),
                          float(integral.p3_sq), float(integral.p4_sq),
                          float(integral.p12_sq), float(integral.p23_sq),
                          float(integral.m1_sq), float(integral.m2_sq),
                          float(integral.m3_sq), float(integral.m4_sq))
            elif isinstance(integral, D22Integral):
                val = D22(float(integral.p1_sq), float(integral.p2_sq),
                          float(integral.p3_sq), float(integral.p4_sq),
                          float(integral.p12_sq), float(integral.p23_sq),
                          float(integral.m1_sq), float(integral.m2_sq),
                          float(integral.m3_sq), float(integral.m4_sq))
            elif isinstance(integral, D23Integral):
                val = D23(float(integral.p1_sq), float(integral.p2_sq),
                          float(integral.p3_sq), float(integral.p4_sq),
                          float(integral.p12_sq), float(integral.p23_sq),
                          float(integral.m1_sq), float(integral.m2_sq),
                          float(integral.m3_sq), float(integral.m4_sq))
            elif isinstance(integral, D33Integral):
                val = D33(float(integral.p1_sq), float(integral.p2_sq),
                          float(integral.p3_sq), float(integral.p4_sq),
                          float(integral.p12_sq), float(integral.p23_sq),
                          float(integral.m1_sq), float(integral.m2_sq),
                          float(integral.m3_sq), float(integral.m4_sq))
            else:
                return None
            result += coeff_val * val
    except Exception:
        return None

    return result
