"""Tests for the generic symbolic amplitude backend."""

from sympy import Symbol, cancel, expand, simplify

from feynman_engine.amplitudes import get_symbolic_amplitude
from feynman_engine.physics.amplitude import get_amplitude, get_curated_amplitude



def test_qed_symbolic_matches_curated_in_massless_limit():
    symbolic = get_symbolic_amplitude("e+ e- -> mu+ mu-", "QED")
    curated = get_curated_amplitude("e+ e- -> mu+ mu-", "QED")

    assert symbolic is not None
    assert curated is not None

    reduced = expand(cancel(simplify(symbolic.msq.subs({Symbol("m_e"): 0, Symbol("m_mu"): 0}) - curated.msq)))
    values_by_name = {"e": 2, "s": 11, "t": -3, "u": -8}
    substitutions = {symbol: values_by_name[symbol.name] for symbol in reduced.free_symbols}
    assert simplify(reduced.subs(substitutions)) == 0


def test_qcd_symbolic_s_channel_support():
    result = get_symbolic_amplitude("u u~ -> d d~", "QCD")

    assert result is not None
    assert result.backend == "qgraf-symbolic"

    reduced = simplify(result.msq.subs({Symbol("m_u"): 0, Symbol("m_d"): 0}))
    expected = 4 * Symbol("g_s")**4 * (Symbol("t")**2 + Symbol("u")**2) / (9 * Symbol("s")**2)
    assert simplify(reduced - expected) == 0


def test_ew_symbolic_backend_notes_limitations():
    result = get_symbolic_amplitude("e+ e- -> mu+ mu-", "EW")

    assert result is not None
    assert result.backend == "qgraf-symbolic"
    assert "gamma5" in result.notes
    assert "g_Z_e" in str(result.msq)
    assert "y_e" in str(result.msq)


def test_bsm_symbolic_backend_support():
    result = get_symbolic_amplitude("e+ e- -> chi chi~", "BSM")

    assert result is not None
    assert result.backend == "qgraf-symbolic"
    assert "g_Zp_e" in str(result.msq)
    assert "g_Zp_chi" in str(result.msq)


def test_bhabha_symbolic_partial():
    """Bhabha (s+t channel) is now computed symbolically; interference is partial."""
    result = get_symbolic_amplitude("e+ e- -> e+ e-", "QED")

    assert result is not None
    assert result.backend == "qgraf-symbolic"
    # Should contain s-channel and t-channel contributions
    assert "s-channel" in result.description
    assert "t-channel" in result.description
    # Missing s-t interference is noted
    assert "interference" in result.notes.lower()
    # Result should be a positive real number in the massless limit
    e, s, t = Symbol("e"), Symbol("s"), Symbol("t")
    massless = result.msq.subs({Symbol("m_e"): 0})
    msq_num = float(massless.subs({e: 2, s: 10, t: -3, Symbol("u"): -7}))
    assert msq_num > 0


def test_curated_amplitude_directly_accessible():
    """Curated amplitudes remain directly accessible via get_curated_amplitude."""
    curated = get_curated_amplitude("e+ e- -> e+ e-", "QED")
    assert curated is not None
    assert curated.backend == "curated"


def test_t_channel_qed_scattering():
    """Single t-channel diagram (e-μ- → e-μ-) gives exact result."""
    result = get_symbolic_amplitude("e- mu- -> e- mu-", "QED")

    assert result is not None
    assert result.backend == "qgraf-symbolic"
    # Should be the single t-channel diagram
    assert "t-channel" in result.description

    # In massless limit: |M|² = 2e^4(s^2+u^2)/t^2
    e, s, t, u = Symbol("e"), Symbol("s"), Symbol("t"), Symbol("u")
    reduced = simplify(
        result.msq.subs({Symbol("m_e"): 0, Symbol("m_mu"): 0})
        - 2 * e**4 * (s**2 + u**2) / t**2
    )
    vals = {e: 2, s: 11, t: -3, u: -8}
    assert simplify(reduced.subs(vals)) == 0


def test_moller_scattering_symbolic():
    """Møller (e-e- → e-e-) gives the exact result including t×u interference."""
    result = get_symbolic_amplitude("e- e- -> e- e-", "QED")

    assert result is not None
    assert result.backend == "qgraf-symbolic"
    assert "t-channel" in result.description
    assert "u-channel" in result.description

    # Massless Møller: |M̄|² = 2e⁴(s⁴+t⁴+u⁴)/(t²u²)  [on s+t+u=0 surface]
    e, s, t, u = Symbol("e"), Symbol("s"), Symbol("t"), Symbol("u")
    massless = result.msq.subs({Symbol("m_e"): 0})
    expected = 2 * e**4 * (s**4 + t**4 + u**4) / (t**2 * u**2)
    # Verify on the physical constraint surface s+t+u=0
    vals = {e: 2, s: 10, t: -3, u: -7}  # s+t+u=0
    assert simplify((massless - expected).subs(vals)) == 0


def test_compton_scattering_exact():
    """Compton (e⁻γ → e⁻γ) matches Klein-Nishina formula in massless limit."""
    result = get_symbolic_amplitude("e- gamma -> e- gamma", "QED")

    assert result is not None
    assert result.backend == "qgraf-symbolic"
    assert "Compton" in result.description

    # Massless Klein-Nishina: |M̄|² = −2e⁴(u/s + s/u)
    e, s, u = Symbol("e"), Symbol("s"), Symbol("u")
    massless = result.msq.subs({Symbol("m_e"): 0})
    expected = -2 * e**4 * (u / s + s / u)
    # Verify on physical point s=10, t=-6, u=-4 (s+t+u=0)
    vals = {e: 2, s: 10, u: -4}
    assert simplify((massless - expected).subs(vals)) == 0
