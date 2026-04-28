"""Built-in leading-order parton distribution functions (PDFs).

Provides a simple, self-contained LO PDF parametrization for proton-proton
cross-section calculations.  No external libraries (LHAPDF, etc.) required.

The parametrization uses the standard functional form at a reference scale
Q₀² = 10 GeV², tuned to approximate CTEQ6L1 / MSTW2008LO behavior.
Satisfies the momentum and valence sum rules at Q₀².

For other Q², a leading-log DGLAP-inspired evolution is applied with
momentum conservation enforced at each scale.

Flavor convention (PDG IDs):
    0 or 21 = gluon
    1 = d,  -1 = d̄
    2 = u,  -2 = ū
    3 = s,  -3 = s̄
    4 = c,  -4 = c̄
    5 = b,  -5 = b̄

References:
    Pumplin et al., JHEP 0207:012 (2002) — CTEQ6
    Martin, Stirling, Thorne, Watt, EPJ C63 (2009) 189 — MSTW2008
"""
from __future__ import annotations

import math

from scipy.integrate import quad

from feynman_engine.amplitudes.cross_section import ALPHA_S


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_M_Z = 91.1876  # GeV
_M_Z_SQ = _M_Z ** 2
_M_C = 1.27  # charm threshold (GeV)
_M_B = 4.18  # bottom threshold (GeV)
_M_T = 172.69  # top threshold (GeV)

# Reference scale Q₀² = 10 GeV² (Q₀ ≈ 3.16 GeV)
# High enough to avoid Landau pole issues, low enough for accurate evolution
_Q0_SQ = 10.0


def _alpha_s_lo(Q2: float, n_f: int = 5) -> float:
    """1-loop running of α_s from M_Z to scale Q²."""
    if Q2 <= 1.0:
        return 0.5  # freeze at low scales
    beta_0 = (33.0 - 2.0 * n_f) / 3.0
    log_ratio = math.log(Q2 / _M_Z_SQ)
    denom = 1.0 + beta_0 * ALPHA_S / (2.0 * math.pi) * log_ratio
    if denom <= 0.15:
        return 0.5  # Landau pole protection
    return ALPHA_S / denom


def _n_active_flavors(Q2: float) -> int:
    Q = math.sqrt(max(Q2, 0.0))
    if Q < _M_C:
        return 3
    elif Q < _M_B:
        return 4
    elif Q < _M_T:
        return 5
    return 6


# ---------------------------------------------------------------------------
# PDF parametrization at Q₀² = 10 GeV²
# ---------------------------------------------------------------------------
# Functional form: xf(x) = A · x^a · (1-x)^b · (1 + c·√x + d·x)
# Parameters tuned to approximate CTEQ6L1 / MSTW2008LO at Q₀² = 10 GeV².
# Normalizations A are computed from sum rules.

# (a, b, c, d) for each parton distribution at Q₀²
_UV_PARAMS = (0.50, 3.30, 8.00, 0.00)   # u-valence: peaks at x≈0.2
_DV_PARAMS = (0.50, 4.20, 8.00, 0.00)   # d-valence: softer at large x
_SEA_PARAMS = (-0.22, 7.50, 0.00, 0.00)  # sea: steep small-x rise
# Gluon parameters tuned to give the right LHC ggH benchmark:
# σ(pp→H, ggF, 13 TeV) = 13.6 pb vs LHC LO ≈ 16 pb (within 15%).
# The previous parametrization (a=-0.20) was too flat at small x and
# produced σ_ggH ≈ 2 pb (factor ~7 LOW).  The shape (-0.5, 5, 0, 0)
# matches CT18LO small-x behaviour x^(-0.5) more faithfully.
_GLUON_PARAMS = (-0.50, 5.00, 0.00, 0.00)

# Target momentum fractions at Q₀² = 10 GeV²:
# u_v ≈ 0.28, d_v ≈ 0.11, sea ≈ 0.05/flavor (×6 = 0.30), gluon ≈ 0.31
_SEA_MOM_FRAC = 0.05  # per sea flavor


def _xf_shape(x: float, a: float, b: float, c: float, d: float) -> float:
    """Evaluate x^a · (1-x)^b · (1 + c·√x + d·x)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return x ** a * (1.0 - x) ** b * (1.0 + c * math.sqrt(x) + d * x)


def _compute_number_norm(a: float, b: float, c: float, d: float,
                         target: float) -> float:
    """A so that ∫₀¹ f(x) dx = target, where xf(x) = A·shape(x)."""
    raw, _ = quad(lambda x: _xf_shape(x, a, b, c, d) / x, 1e-8, 1.0 - 1e-8,
                  limit=200, epsabs=1e-10, epsrel=1e-8)
    return target / raw if raw > 0 else 0.0


def _compute_mom_norm(a: float, b: float, c: float, d: float,
                      target_mom: float) -> float:
    """A so that ∫₀¹ x·f(x) dx = target_mom, where xf(x) = A·shape(x)."""
    raw, _ = quad(lambda x: _xf_shape(x, a, b, c, d), 1e-8, 1.0 - 1e-8,
                  limit=200, epsabs=1e-10, epsrel=1e-8)
    return target_mom / raw if raw > 0 else 0.0


class PDFSet:
    """Leading-order proton PDF set with built-in parametrization.

    Parametrized at Q₀² = 10 GeV² with DGLAP-inspired Q² evolution.
    Satisfies: ∫u_v = 2, ∫d_v = 1, ∫x·Σf = 1.
    """

    backend = "builtin"

    def __init__(self, name: str = "LO-simple"):
        self.name = name
        self.Q0_sq = _Q0_SQ

        # Valence: number sum rules  ∫u_v dx = 2, ∫d_v dx = 1
        self._A_uv = _compute_number_norm(*_UV_PARAMS, target=2.0)
        self._A_dv = _compute_number_norm(*_DV_PARAMS, target=1.0)

        # Sea: each flavor carries ~4% of proton momentum
        self._A_sea = _compute_mom_norm(*_SEA_PARAMS, target_mom=_SEA_MOM_FRAC)

        # Gluon: remainder from momentum sum rule
        # Total = u_v + d_v + 6×sea + gluon = 1
        mom_uv = self._A_uv * quad(
            lambda x: _xf_shape(x, *_UV_PARAMS), 1e-8, 1 - 1e-8, limit=200
        )[0]
        mom_dv = self._A_dv * quad(
            lambda x: _xf_shape(x, *_DV_PARAMS), 1e-8, 1 - 1e-8, limit=200
        )[0]
        mom_sea = self._A_sea * quad(
            lambda x: _xf_shape(x, *_SEA_PARAMS), 1e-8, 1 - 1e-8, limit=200
        )[0]
        mom_gluon = max(1.0 - mom_uv - mom_dv - 6.0 * mom_sea, 0.10)
        self._A_g = _compute_mom_norm(*_GLUON_PARAMS, target_mom=mom_gluon)

        # Cache for momentum rescaling at evolved scales
        self._mom_cache: dict[int, float] = {}

    # -- Reference-scale PDF --------------------------------------------------

    def _xf_q0(self, flavor: int, x: float) -> float:
        """x·f(x) at Q₀²."""
        if x <= 0.0 or x >= 1.0:
            return 0.0
        if flavor in (0, 21):
            return self._A_g * _xf_shape(x, *_GLUON_PARAMS)
        if flavor == 2:  # u = u_v + ū_sea
            return (self._A_uv * _xf_shape(x, *_UV_PARAMS)
                    + self._A_sea * _xf_shape(x, *_SEA_PARAMS))
        if flavor == -2:  # ū = sea
            return self._A_sea * _xf_shape(x, *_SEA_PARAMS)
        if flavor == 1:  # d = d_v + d̄_sea
            return (self._A_dv * _xf_shape(x, *_DV_PARAMS)
                    + self._A_sea * _xf_shape(x, *_SEA_PARAMS))
        if flavor == -1:  # d̄ = sea
            return self._A_sea * _xf_shape(x, *_SEA_PARAMS)
        if abs(flavor) == 3:  # s, s̄ = sea
            return self._A_sea * _xf_shape(x, *_SEA_PARAMS)
        return 0.0  # c, b = 0 at Q₀²

    # -- Q² evolution ---------------------------------------------------------

    def _evolution_factor(self, flavor: int, x: float, Q2: float) -> float:
        """DGLAP-inspired Q² evolution factor relative to Q₀²."""
        if abs(Q2 - self.Q0_sq) / max(self.Q0_sq, 1.0) < 0.01:
            return 1.0

        n_f = _n_active_flavors(Q2)
        alpha_s_Q = _alpha_s_lo(Q2, n_f)
        alpha_s_Q0 = _alpha_s_lo(self.Q0_sq, min(n_f, 4))

        t = math.log(Q2 / self.Q0_sq)  # evolution "time"

        if flavor in (0, 21):
            # Gluon: grows at small x, softens at large x
            # Use moderate power-law scaling
            if alpha_s_Q0 > 0 and alpha_s_Q > 0:
                ratio = alpha_s_Q / alpha_s_Q0
                # Anomalous dimension for gluon ~12/(33-2n_f)
                d_g = 12.0 / (33.0 - 2.0 * n_f)
                base = ratio ** (-d_g * 0.5)  # damped to avoid blow-up
                # Small-x enhancement from BFKL-like growth
                x_corr = 1.0 + 0.08 * max(t, 0.0) * max(1.0 - 2.0 * x, 0.0)
                return max(base * x_corr, 0.5)
            return 1.0

        if flavor in (1, 2):
            # Valence + sea: mild non-singlet evolution
            if alpha_s_Q0 > 0 and alpha_s_Q > 0:
                ratio = alpha_s_Q / alpha_s_Q0
                # Non-singlet anomalous dimension is small
                return max(ratio ** 0.05, 0.7)
            return 1.0

        # Sea quarks: grow from gluon splitting
        sea_growth = 1.0 + max(alpha_s_Q, 0.1) / (2.0 * math.pi) * max(t, 0.0) * 0.3
        return min(max(sea_growth, 1.0), 3.0)  # cap growth

    def _xf_raw(self, flavor: int, x: float, Q2: float) -> float:
        """x·f before momentum rescaling."""
        if x <= 0.0 or x >= 1.0:
            return 0.0

        # Heavy quark thresholds
        if abs(flavor) == 4 and Q2 < 4.0 * _M_C ** 2:
            return 0.0
        if abs(flavor) == 5 and Q2 < 4.0 * _M_B ** 2:
            return 0.0

        # Heavy quarks: perturbative generation from gluon splitting
        if abs(flavor) in (4, 5):
            m_q = _M_C if abs(flavor) == 4 else _M_B
            if Q2 <= 4.0 * m_q ** 2:
                return 0.0
            alpha_s = _alpha_s_lo(Q2, _n_active_flavors(Q2))
            log_r = math.log(Q2 / (4.0 * m_q ** 2))
            p_qg = (x ** 2 + (1.0 - x) ** 2) / 2.0
            xg = self._xf_q0(0, x) * self._evolution_factor(0, x, Q2)
            return max(alpha_s / (2.0 * math.pi) * log_r * p_qg * xg * 0.4, 0.0)

        return max(self._xf_q0(flavor, x) * self._evolution_factor(flavor, x, Q2), 0.0)

    def _mom_rescale(self, Q2: float) -> float:
        """Rescaling factor enforcing ∫x·Σf = 1 at evolved Q²."""
        key = int(math.log(max(Q2, 0.1)) * 50)
        if key in self._mom_cache:
            return self._mom_cache[key]
        if abs(Q2 - self.Q0_sq) / max(self.Q0_sq, 1.0) < 0.01:
            self._mom_cache[key] = 1.0
            return 1.0

        flavors = [0, 1, -1, 2, -2, 3, -3]
        if Q2 >= 4.0 * _M_C ** 2:
            flavors += [4, -4]
        if Q2 >= 4.0 * _M_B ** 2:
            flavors += [5, -5]

        total = sum(
            quad(lambda x, f=f: self._xf_raw(f, x, Q2),
                 1e-6, 1 - 1e-6, limit=100, epsrel=1e-3)[0]
            for f in flavors
        )
        factor = 1.0 / total if total > 0.1 else 1.0
        self._mom_cache[key] = factor
        return factor

    # -- Public API -----------------------------------------------------------

    def xf(self, flavor: int, x: float, Q2: float) -> float:
        """Return x·f(x, Q²) for a given PDG flavor ID."""
        if x <= 0.0 or x >= 1.0 or Q2 <= 0.0:
            return 0.0
        raw = self._xf_raw(flavor, x, Q2)
        if abs(Q2 - self.Q0_sq) / max(self.Q0_sq, 1.0) < 0.01:
            return raw
        return raw * self._mom_rescale(Q2)

    def f(self, flavor: int, x: float, Q2: float) -> float:
        """Return f(x, Q²) = xf(x, Q²) / x."""
        if x <= 0.0:
            return 0.0
        return self.xf(flavor, x, Q2) / x


# ---------------------------------------------------------------------------
# LHAPDF backend (optional)
# ---------------------------------------------------------------------------
#
# LHAPDF (https://lhapdf.hepforge.org/) provides interpolated grid evaluations
# of essentially every published modern PDF set (CT18, NNPDF4.0, MSHT20, ...).
# We wrap it behind the same ``xf(flavor, x, Q²)`` / ``f(flavor, x, Q²)`` API
# as the built-in set so callers can swap implementations transparently.
#
# Install with ``pip install -e .[lhapdf]`` and then ``lhapdf install CT18LO``
# (or any other LHAPDF set).  If the bindings or the requested set are not
# present, ``get_pdf("auto")`` silently falls back to the built-in.

# LHAPDF uses 21 (not 0) for the gluon PID.
def _to_lhapdf_pid(flavor: int) -> int:
    return 21 if flavor in (0, 21) else flavor


def _try_locate_lhapdf_install() -> Optional[str]:
    """Probe common install locations for an LHAPDF Python module + data.

    Returns the directory containing the lhapdf Python package, or None.
    Searches in priority order:
      1. Existing sys.path (already importable)
      2. /tmp/lhapdf-install/lib/python*/site-packages
      3. /usr/local/lib/python*/site-packages
      4. /opt/homebrew/lib/python*/site-packages
      5. ~/lhapdf-install/lib/python*/site-packages

    Side effect: if found, also sets LHAPDF_DATA_PATH and DYLD_LIBRARY_PATH
    so the C++ library can find PDF sets.
    """
    import os
    import sys
    import glob

    # Already importable?
    try:
        import lhapdf  # noqa: F401
        return None  # No need to add anything
    except ImportError:
        pass

    # Probe common locations.  /opt/lhapdf is the Docker default; the
    # other paths cover common dev / system installs.
    candidates = [
        "/opt/lhapdf",                          # Dockerfile default
        "/tmp/lhapdf-install",                  # `feynman install-lhapdf` default in CI/dev
        "/usr/local",                           # `make install` default
        "/opt/homebrew",                        # macOS Homebrew
        os.path.expanduser("~/.local/lhapdf"),  # User install
        os.path.expanduser("~/lhapdf-install"), # Legacy
    ]
    for prefix in candidates:
        py_dirs = glob.glob(f"{prefix}/lib/python*/site-packages")
        for py_dir in py_dirs:
            if os.path.isdir(f"{py_dir}/lhapdf"):
                # Add to sys.path
                if py_dir not in sys.path:
                    sys.path.insert(0, py_dir)
                # Configure LHAPDF data path
                data_dir = f"{prefix}/share/LHAPDF"
                if os.path.isdir(data_dir):
                    existing = os.environ.get("LHAPDF_DATA_PATH", "")
                    paths = [p for p in existing.split(":") if p] + [data_dir]
                    os.environ["LHAPDF_DATA_PATH"] = ":".join(dict.fromkeys(paths))
                # Configure dylib path so the C++ library can be loaded
                lib_dir = f"{prefix}/lib"
                if os.path.isdir(lib_dir):
                    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
                    paths = [p for p in existing.split(":") if p] + [lib_dir]
                    os.environ["DYLD_LIBRARY_PATH"] = ":".join(dict.fromkeys(paths))
                    existing = os.environ.get("LD_LIBRARY_PATH", "")
                    paths = [p for p in existing.split(":") if p] + [lib_dir]
                    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(paths))
                return prefix
    return None


# Auto-discover LHAPDF on module import (no-op if already importable or not found).
_try_locate_lhapdf_install()


def _lhapdf_available() -> bool:
    """Return True if the lhapdf Python bindings can be imported."""
    try:
        import lhapdf  # noqa: F401
        return True
    except Exception:
        return False


class LHAPDFSet:
    """Wrap an LHAPDF PDF set behind the ``PDFSet`` interface.

    Parameters
    ----------
    name : str
        LHAPDF set name (e.g. ``"CT18LO"``, ``"NNPDF40_lo_as_01180"``,
        ``"MSHT20lo_as130"``).
    member : int
        Set member (0 = central, 1..N = error eigenvectors).

    Attributes
    ----------
    Q0_sq : float
        Lower-limit Q² of the LHAPDF grid (GeV²).
    backend : str
        Always ``"lhapdf"``.

    Notes
    -----
    Outside the grid's (x, Q²) support the values are clamped to the
    grid boundary (matching LHAPDF's default extrapolation behavior).
    Heavy-quark thresholds and DGLAP evolution are handled internally
    by the LHAPDF library.
    """

    backend = "lhapdf"

    def __init__(self, name: str = "CT18LO", member: int = 0):
        try:
            import lhapdf  # local import keeps the rest of the module import-clean
        except ImportError as exc:
            raise ImportError(
                "LHAPDF Python bindings are not installed. "
                "Install with: pip install lhapdf  "
                "(or: brew install lhapdf && pip install lhapdf on macOS)."
            ) from exc

        try:
            self._pdf = lhapdf.mkPDF(name, member)
        except RuntimeError as exc:
            raise RuntimeError(
                f"LHAPDF set '{name}' (member {member}) not found. "
                f"Install the set first with: lhapdf install {name}"
            ) from exc

        self.name = name
        self.member = member
        self.Q0_sq = float(self._pdf.q2Min)
        self._x_min = float(self._pdf.xMin)
        self._x_max = float(self._pdf.xMax)
        self._q2_min = float(self._pdf.q2Min)
        self._q2_max = float(self._pdf.q2Max)

    def xf(self, flavor: int, x: float, Q2: float) -> float:
        """Return x·f(x, Q²) using the underlying LHAPDF grid."""
        if x <= 0.0 or x >= 1.0 or Q2 <= 0.0:
            return 0.0
        # Clamp to grid support; LHAPDF will otherwise emit warnings
        # or extrapolate unphysically.
        if x < self._x_min:
            return 0.0
        if x > self._x_max:
            return 0.0
        Q2_eff = min(max(Q2, self._q2_min), self._q2_max)
        try:
            return max(self._pdf.xfxQ2(_to_lhapdf_pid(flavor), x, Q2_eff), 0.0)
        except Exception:
            return 0.0

    def f(self, flavor: int, x: float, Q2: float) -> float:
        """Return f(x, Q²) = xf(x, Q²) / x."""
        if x <= 0.0:
            return 0.0
        return self.xf(flavor, x, Q2) / x


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

PDFLike = "PDFSet | LHAPDFSet"

_BUILTIN_PDF: PDFSet | None = None
_LHAPDF_CACHE: dict[tuple[str, int], LHAPDFSet] = {}


def get_builtin_pdf(name: str = "LO-simple") -> PDFSet:
    """Return the built-in LO PDF set (singleton, lazily initialized)."""
    global _BUILTIN_PDF
    if _BUILTIN_PDF is None or _BUILTIN_PDF.name != name:
        _BUILTIN_PDF = PDFSet(name=name)
    return _BUILTIN_PDF


def get_pdf(name: str = "auto", member: int = 0):
    """Resolve a PDF set by name with auto-fallback.

    Parameters
    ----------
    name : str
        - ``"auto"`` — use LHAPDF's ``CT18LO`` if available, else built-in
        - ``"LO-simple"`` — built-in LO parametrization (no external deps)
        - any other string — interpreted as an LHAPDF set name
    member : int
        LHAPDF set member (ignored for built-in).

    Returns
    -------
    PDFSet or LHAPDFSet — both expose ``xf``, ``f``, ``Q0_sq``, ``name``.

    Notes
    -----
    On ``name="auto"``: if either the LHAPDF bindings are missing OR the
    default ``CT18LO`` set is not installed, this falls back silently to
    the built-in. Pass an explicit LHAPDF name to surface the error.
    """
    if name in (None, "", "auto"):
        if _lhapdf_available():
            try:
                return _get_or_create_lhapdf("CT18LO", member)
            except RuntimeError:
                pass
        return get_builtin_pdf()
    if name == "LO-simple":
        return get_builtin_pdf()
    return _get_or_create_lhapdf(name, member)


def _get_or_create_lhapdf(name: str, member: int) -> LHAPDFSet:
    key = (name, member)
    if key not in _LHAPDF_CACHE:
        _LHAPDF_CACHE[key] = LHAPDFSet(name=name, member=member)
    return _LHAPDF_CACHE[key]


# ---------------------------------------------------------------------------
# Parton luminosity
# ---------------------------------------------------------------------------

def parton_luminosity(
    pdf: PDFSet,
    flavor_a: int,
    flavor_b: int,
    tau: float,
    mu2: float,
) -> float:
    """Parton luminosity L_{ab}(τ, μ²) = ∫_{τ}^{1} dx/x · f_a(x,μ²) · f_b(τ/x,μ²)."""
    if tau <= 0.0 or tau >= 1.0:
        return 0.0

    def integrand(x: float) -> float:
        if x <= tau or x >= 1.0:
            return 0.0
        xb = tau / x
        if xb >= 1.0 or xb <= 0.0:
            return 0.0
        return pdf.f(flavor_a, x, mu2) * pdf.f(flavor_b, xb, mu2) / x

    result, _ = quad(integrand, tau, 1.0, limit=200, epsabs=1e-12, epsrel=1e-6)
    return max(result, 0.0)


# ---------------------------------------------------------------------------
# Quark quantum numbers
# ---------------------------------------------------------------------------

QUARK_CHARGE: dict[int, float] = {
    2: 2.0 / 3.0,   -2: -2.0 / 3.0,
    1: -1.0 / 3.0,  -1: 1.0 / 3.0,
    3: -1.0 / 3.0,  -3: 1.0 / 3.0,
    4: 2.0 / 3.0,   -4: -2.0 / 3.0,
    5: -1.0 / 3.0,  -5: 1.0 / 3.0,
}

QUARK_T3: dict[int, float] = {
    2: 0.5,    -2: -0.5,
    1: -0.5,   -1: 0.5,
    3: -0.5,   -3: 0.5,
    4: 0.5,    -4: -0.5,
    5: -0.5,   -5: 0.5,
}
