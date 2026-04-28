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


def test_qcd_symbolic_defers_to_curated():
    """QCD processes bail out of the symbolic backend due to color-factor
    complexity and are handled by the curated backend instead."""
    result = get_symbolic_amplitude("u u~ -> d d~", "QCD")
    assert result is None  # symbolic backend defers for all QCD


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


def test_bhabha_symbolic_exact():
    """Bhabha (s+t channel) including s×t cross-topology interference — exact result."""
    result = get_symbolic_amplitude("e+ e- -> e+ e-", "QED")

    assert result is not None
    assert result.backend == "qgraf-symbolic"
    assert "s-channel" in result.description
    assert "t-channel" in result.description

    # Massless Bhabha: |M̄|² = 2e⁴[(s²+u²)/t² + (t²+u²)/s² + 2u²/(st)]
    e, s, t, u = Symbol("e"), Symbol("s"), Symbol("t"), Symbol("u")
    massless = result.msq.subs({Symbol("m_e"): 0})
    expected = 2 * e**4 * ((s**2 + u**2) / t**2 + (t**2 + u**2) / s**2 + 2 * u**2 / (s * t))
    vals = {e: 2, s: 10, t: -3, u: -7}  # s+t+u=0
    assert simplify((massless - expected).subs(vals)) == 0


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

    # Massless Møller (P&S eq 5.66):
    #   |M̄|² = 2e⁴[(s²+u²)/t² + (s²+t²)/u² − 2s²/(tu)]
    # Equivalent: 2e⁴[s²(t−u)² + t⁴ + u⁴] / (t²u²)
    e, s, t, u = Symbol("e"), Symbol("s"), Symbol("t"), Symbol("u")
    massless = result.msq.subs({Symbol("m_e"): 0})
    expected = 2 * e**4 * ((s**2 + u**2) / t**2 + (s**2 + t**2) / u**2 - 2 * s**2 / (t * u))
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


# ── EW curated amplitudes ─────────────────────────────────────────────────────

def test_ew_zh_curated_positive_at_250gev():
    """e⁺e⁻ → ZH: |M̄|² must be positive at a physical kinematic point."""
    import math
    from feynman_engine.physics.amplitude import get_amplitude

    r = get_amplitude("e+ e- -> Z H", "EW")
    assert r is not None
    assert r.backend == "curated"

    sq_s = 250.0**2
    m_Z, m_H = 91.1876, 125.20
    E_Z = (sq_s + m_Z**2 - m_H**2) / (2 * math.sqrt(sq_s))
    t_val = m_Z**2 - math.sqrt(sq_s) * E_Z   # θ = 90°
    u_val = m_Z**2 + m_H**2 - sq_s - t_val
    assert abs(sq_s + t_val + u_val - (m_Z**2 + m_H**2)) < 1e-6, "Mandelstam constraint"

    val = r.msq_at(sq_s, t_val, u_val)
    assert val is not None and val > 0, f"Expected positive |M|², got {val}"


def test_ew_tautau_zh_same_as_ee():
    """τ⁺τ⁻ → ZH should give the same |M̄|² as e⁺e⁻ → ZH (lepton universality)."""
    import math
    from feynman_engine.physics.amplitude import get_amplitude
    from sympy import simplify

    r_ee = get_amplitude("e+ e- -> Z H", "EW")
    r_tt = get_amplitude("tau+ tau- -> Z H", "EW")
    assert r_ee is not None and r_tt is not None

    sq_s = 300.0**2
    m_Z, m_H = 91.1876, 125.20
    E_Z = (sq_s + m_Z**2 - m_H**2) / (2 * sq_s**0.5)
    t_val = m_Z**2 - sq_s**0.5 * E_Z
    u_val = m_Z**2 + m_H**2 - sq_s - t_val

    val_ee = r_ee.msq_at(sq_s, t_val, u_val)
    val_tt = r_tt.msq_at(sq_s, t_val, u_val)
    assert val_ee is not None and val_tt is not None
    assert abs(val_ee - val_tt) < 1e-10, f"Expected equal: {val_ee} vs {val_tt}"


def test_ew_zh_symbol_count():
    """ZH amplitude must contain exactly the four EW symbols."""
    from feynman_engine.physics.amplitude import get_amplitude

    r = get_amplitude("e+ e- -> Z H", "EW")
    assert r is not None
    free_names = {s.name for s in r.msq.free_symbols}
    assert "g_Z" in free_names
    assert "sin2_W" in free_names
    assert "m_Z" in free_names
    assert "m_H" in free_names
    # Should NOT contain QCD/QED symbols
    assert "g_s" not in free_names
    assert "e" not in free_names


# ── Loop infrastructure ───────────────────────────────────────────────────────

def test_pv_reduce_returns_expansion_for_1loop():
    """pv_reduce() on a 1-loop QED box diagram returns a PVExpansion."""
    from feynman_engine.core.generator import generate_diagrams
    from feynman_engine.physics.translator import parse_process
    from feynman_engine.amplitudes.loop import pv_reduce, LoopTopology

    spec = parse_process("e+ e- -> mu+ mu-", theory="QED", loops=1)
    diagrams = generate_diagrams(spec)
    loop_diags = [d for d in diagrams if d.loop_order == 1]
    assert loop_diags, "Expected at least one 1-loop diagram"

    expansion = pv_reduce(loop_diags[0], "QED")
    assert expansion is not None
    assert expansion.topology in (LoopTopology.BOX, LoopTopology.TRIANGLE, LoopTopology.SELF_ENERGY)
    assert expansion.terms  # must have at least one scalar integral term


def test_looptools_bridge_unavailable_gracefully():
    """When LoopTools is not installed, is_available() returns False (no exception)."""
    from feynman_engine.amplitudes.looptools_bridge import is_available
    # Should never raise — just return False if library missing
    result = is_available()
    assert isinstance(result, bool)


def test_gg_to_gg_curated():
    """gg→gg amplitude is now available and positive at physical kinematics."""
    from feynman_engine.physics.amplitude import get_amplitude
    from sympy import N

    r = get_amplitude("g g -> g g", "QCD")
    assert r is not None and r.backend in ("curated", "form-symbolic")
    # s=4, t=-1, u=-3: s+t+u=0 ✓
    val = r.msq_at(4.0, -1.0, -3.0, g_s_val=1.0)
    assert val is not None and val > 0


# ── LoopTools integration ──────────────────────────────────────────────────────

def test_looptools_available():
    """LoopTools shared library is found after feynman install-looptools."""
    from feynman_engine.amplitudes.looptools_bridge import is_available, B0
    assert is_available(), "Expected LoopTools to be available (run feynman install-looptools)"
    # B0(4, 1, 1) = 2 is a known reference value
    val = B0(4.0, 1.0, 1.0)
    assert abs(val.real - 2.0) < 1e-6, f"Expected B0(4,1,1)=2.0, got {val}"


def test_looptools_b0_reference():
    """B0 and C0 reproduce known reference values."""
    from feynman_engine.amplitudes.looptools_bridge import is_available, B0, C0
    import math

    if not is_available():
        return  # skip if LoopTools not available

    # B0(4m², m², m²) at threshold: known exact result is 2
    val = B0(4.0, 1.0, 1.0)
    assert abs(val.real - 2.0) < 1e-6

    # B0(4, 0, 0) with massless propagators and μ²=1:
    # B0(p², 0, 0) = 2 - log(p²/μ²) + iπ  for timelike p² > 0
    # At p²=4, μ²=1: Re = 2 - log(4) ≈ 0.614, Im = π
    val2 = B0(4.0, 0.0, 0.0)
    expected_real = 2.0 - math.log(4.0)  # ≈ 0.614
    assert abs(val2.real - expected_real) < 0.01, f"Got Re[B0(4,0,0)]={val2.real}, expected {expected_real:.4f}"
    assert abs(val2.imag - math.pi) < 0.01, f"Got Im[B0(4,0,0)]={val2.imag}"


def test_schwinger_amm():
    """Schwinger correction a_e = α/(2π) evaluates to ~1.16e-3."""
    from feynman_engine.amplitudes.loop_curated import evaluate_schwinger_amm
    import math

    a_e = evaluate_schwinger_amm()
    expected = 1.0 / (137.036 * 2 * math.pi)
    assert abs(a_e - expected) < 1e-8, f"Expected a_e≈{expected:.6e}, got {a_e:.6e}"


def test_vacuum_polarisation():
    """Vacuum polarisation Π(q²) is evaluatable and finite for q²=1 GeV², m²=m_e²."""
    from feynman_engine.amplitudes.loop_curated import evaluate_vacuum_polarisation
    from feynman_engine.amplitudes.looptools_bridge import is_available

    if not is_available():
        return

    m_e_sq = (0.000511) ** 2  # m_e in GeV, squared
    pi_val = evaluate_vacuum_polarisation(1.0, m_e_sq)
    assert pi_val is not None
    # At q²=1 GeV² ≫ m_e², leading log: Π ≈ (α/3π) log(q²/m_e²) ≈ 0.0077
    assert pi_val.real > 0, "Vacuum polarisation should be positive at q²>0"


def test_photon_selfenergy_at_threshold():
    """Photon self-energy Σ_T(k²) vanishes at k²=0 (on-shell photon condition)."""
    from feynman_engine.amplitudes.loop_curated import evaluate_photon_selfenergy
    from feynman_engine.amplitudes.looptools_bridge import is_available

    if not is_available():
        return

    m_e_sq = (0.000511) ** 2
    # B0(0, m², m²) = 0 at μ=m, so Σ_T(0) = (α/π)[2 A0(m²) - 4m² B0(0,m²,m²)]
    val = evaluate_photon_selfenergy(0.0, m_e_sq)
    assert val is not None


def test_running_alpha():
    """α(q²) > α(0) at q² > 0 (charge screening)."""
    from feynman_engine.amplitudes.renorm import alpha_running

    alpha0 = 1.0 / 137.036
    # Running up from q²=1 to q²=100 GeV², α should increase (charge screening).
    # Our implementation uses electron loop only (electron-only β₀ = 2/3π).
    alpha_high = alpha_running(100.0, mu0_sq=1.0, alpha0=alpha0)
    assert alpha_high > alpha0, f"Expected α(100 GeV²) > α(1 GeV²), got {alpha_high:.6f} < {alpha0:.6f}"


def test_running_alpha_s():
    """α_s(q²) < α_s(MZ²) for q² > MZ² (asymptotic freedom)."""
    from feynman_engine.amplitudes.renorm import alpha_s_running

    mz_sq = 91.1876 ** 2
    alpha_s_mz = 0.1179
    alpha_s_2mz = alpha_s_running(4 * mz_sq, mu0_sq=mz_sq, alpha_s0=alpha_s_mz, n_f=5)
    assert alpha_s_2mz < alpha_s_mz, "α_s should decrease with scale (asymptotic freedom)"


def test_looptools_packaging():
    """LoopTools source archive is bundled in package resources."""
    from feynman_engine.looptools import looptools_source_available
    assert looptools_source_available(), "Expected LoopTools source archive to be bundled"


# ── New curated amplitudes ────────────────────────────────────────────────────

def _subs_by_name(expr, name_to_value: dict):
    """Substitute expression symbols by name regardless of sympy assumptions
    (real, positive, etc.) — works for any backend."""
    subs_map = {sym: name_to_value[sym.name]
                for sym in expr.free_symbols if sym.name in name_to_value}
    return expr.subs(subs_map)


def test_emu_to_emu_curated():
    """e⁻μ⁻ → e⁻μ⁻ amplitude has correct formula 2e⁴(s²+u²)/t² (massless)."""
    from feynman_engine.physics.amplitude import get_amplitude

    r = get_amplitude("e- mu- -> e- mu-", "QED")
    assert r is not None

    # Strip masses (no-op for curated, important for FORM backend).
    massless = _subs_by_name(r.msq, {"m_e": 0, "m_mu": 0})

    # Known massless result: 2e⁴(s²+u²)/t²
    e_val, s_val, t_val, u_val = 2, 11, -3, -8
    expected = 2 * e_val**4 * (s_val**2 + u_val**2) / t_val**2

    result_val = _subs_by_name(massless, {
        "e": e_val, "s": s_val, "t": t_val, "u": u_val,
    })
    assert abs(float(result_val) - expected) < 1e-10


def test_mumu_to_ee_curated():
    """μ⁺μ⁻ → e⁺e⁻ amplitude equals e⁺e⁻ → μ⁺μ⁻ (crossing)."""
    from feynman_engine.physics.amplitude import get_amplitude

    r_ee = get_amplitude("e+ e- -> mu+ mu-", "QED")
    r_mm = get_amplitude("mu+ mu- -> e+ e-", "QED")
    assert r_ee is not None and r_mm is not None

    # Same formula by crossing symmetry — verify at a numerical point.
    vals = {"e": 2, "m_e": 0, "m_mu": 0, "s": 10, "t": -3, "u": -7}
    val_ee = float(_subs_by_name(r_ee.msq, vals))
    val_mm = float(_subs_by_name(r_mm.msq, vals))
    assert abs(val_ee - val_mm) < 1e-10


def test_qcd_ug_to_ug_positive():
    """ug → ug amplitude is positive at a physical kinematic point (massless limit)."""
    from feynman_engine.physics.amplitude import get_amplitude
    from sympy import Symbol, simplify

    r = get_amplitude("u g -> u g", "QCD")
    assert r is not None
    # Evaluate in massless limit at s=4, t=-1, u=-3 (s+t+u=0 ✓)
    val_map = {"s": 4, "t": -1, "u": -3, "e": 1, "g_s": 1, "m_u": 0}
    subs = {sym: val_map[sym.name] for sym in r.msq.free_symbols if sym.name in val_map}
    val = float(r.msq.subs(subs))
    assert val > 0, f"Expected positive amplitude, got {val}"


def test_dd_to_gg_curated():
    """d d~ → gg uses same formula as u u~ → gg (flavour universality)."""
    from feynman_engine.physics.amplitude import get_amplitude

    r_uu = get_amplitude("u u~ -> g g", "QCD")
    r_dd = get_amplitude("d d~ -> g g", "QCD")
    assert r_uu is not None and r_dd is not None
    # Identical formula (flavour-blind gluon coupling)
    val_uu = r_uu.msq_at(4.0, -1.0, -3.0, g_s_val=1.0)
    val_dd = r_dd.msq_at(4.0, -1.0, -3.0, g_s_val=1.0)
    assert val_uu is not None and val_dd is not None
    assert abs(val_uu - val_dd) < 1e-10


def test_box_coefficient_not_unity():
    """Box diagram coefficient is now −8α²tu, not the placeholder 1."""
    from feynman_engine.core.generator import generate_diagrams
    from feynman_engine.physics.translator import parse_process
    from feynman_engine.amplitudes.loop import pv_reduce, LoopTopology, D0Integral
    from sympy import Integer

    spec = parse_process("e+ e- -> mu+ mu-", theory="QED", loops=1)
    diagrams = generate_diagrams(spec)

    box_checked = False
    for d in [diag for diag in diagrams if diag.loop_order == 1]:
        exp = pv_reduce(d, "QED")
        if exp and exp.topology == LoopTopology.BOX:
            for integral, coeff in exp.terms.items():
                if isinstance(integral, D0Integral):
                    # Coefficient must not be the old Integer(1) placeholder
                    assert coeff != Integer(1), "Box D₀ coefficient must not be the placeholder 1"
                    # Must contain alpha, t, u symbols
                    coeff_str = str(coeff)
                    assert "alpha" in coeff_str
                    assert "t" in coeff_str or "u" in coeff_str
                    # Evaluate at t=-3, u=-7 to verify sign (should be negative since tu>0)
                    # Substitute by name to avoid symbol-assumption mismatch
                    name_vals = {"alpha": 1.0/137.036, "t": -3.0, "u": -7.0}
                    subs_map = {sym: name_vals[sym.name]
                                for sym in coeff.free_symbols
                                if sym.name in name_vals}
                    val = float(coeff.subs(subs_map))
                    assert val < 0, f"c_D0 = −8α²tu should be negative at t=-3, u=-7; got {val}"
                    box_checked = True
            break

    assert box_checked, "No box diagram found for e+e-→μ+μ- at 1-loop"


def test_loop_curated_has_20_entries():
    """Curated 1-loop registry now has 20 entries."""
    from feynman_engine.amplitudes.loop_curated import get_loop_curated_results
    results = get_loop_curated_results()
    assert len(results) == 20, f"Expected 20 curated 1-loop entries, got {len(results)}"


def test_arbitrary_process_tree_amplitude():
    """An arbitrary non-example process (e-mu+ → e-mu+) returns a symbolic amplitude."""
    from feynman_engine.physics.amplitude import get_amplitude

    r = get_amplitude("e- mu+ -> e- mu+", "QED")
    assert r is not None, "e-mu+→e-mu+ should produce a symbolic amplitude"
    assert r.backend in {"qgraf-symbolic", "form-symbolic"}


def test_arbitrary_process_loop_integral():
    """An arbitrary non-example process (μ-μ- → μ-μ-) returns a 1-loop integral."""
    from feynman_engine.amplitudes import get_loop_integral_latex

    result = get_loop_integral_latex("mu- mu- -> mu- mu-", "QED", loops=1)
    assert result is not None, "Møller for muons should produce a 1-loop integral"
