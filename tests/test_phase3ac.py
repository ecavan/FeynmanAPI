"""Tests for Phase 3a (tensor PV reduction) and Phase 3c (cross-section integration)."""
from __future__ import annotations

import math
import pytest
from sympy import Symbol, symbols


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3a — tensor integral dataclasses
# ─────────────────────────────────────────────────────────────────────────────

def test_tensor_integral_dataclasses_importable():
    """All new tensor integral types are importable from feynman_engine.amplitudes."""
    from feynman_engine.amplitudes import (
        B1Integral, B00Integral, B11Integral,
        C1Integral, C2Integral, C00Integral, C11Integral, C12Integral, C22Integral,
        D00Integral, D1Integral, D2Integral, D3Integral,
    )
    # Instantiate one to verify frozen dataclass works
    b1 = B1Integral(p_sq=1.0, m1_sq=0.5, m2_sq=0.5)
    assert str(b1) == "B1(1.0, 0.5, 0.5)"
    b00 = B00Integral(p_sq=4.0, m1_sq=1.0, m2_sq=1.0)
    assert "B00" in str(b00)
    c00 = C00Integral(p1_sq=1.0, p2_sq=1.0, p12_sq=10.0,
                      m1_sq=0.0, m2_sq=1.0, m3_sq=1.0)
    assert "C00" in str(c00)


def test_tensor_integral_latex():
    """Tensor integral dataclasses produce valid LaTeX strings."""
    from feynman_engine.amplitudes import B1Integral, C00Integral
    b1 = B1Integral(p_sq=Symbol("s"), m1_sq=Symbol("m", positive=True)**2,
                    m2_sq=Symbol("m", positive=True)**2)
    latex_str = b1.latex()
    assert "B_1" in latex_str

    c00 = C00Integral(p1_sq=Symbol("m")**2, p2_sq=Symbol("m")**2,
                      p12_sq=Symbol("s"),
                      m1_sq=0, m2_sq=Symbol("m")**2, m3_sq=Symbol("m")**2)
    assert "C_{00}" in c00.latex()


def test_looptools_bridge_tensor_functions_defined():
    """All new LoopTools tensor bridge functions exist (even if library unavailable)."""
    import feynman_engine.amplitudes.looptools_bridge as lt
    for name in ("B11", "C1", "C2", "C00", "C11", "C12", "C22",
                 "D00", "D1", "D2", "D3"):
        assert hasattr(lt, name), f"looptools_bridge missing: {name}"


def test_pv_reduce_photon_se_uses_scalar_denner_form():
    """Photon SE expansion uses the correct Denner scalar form: A₀ and B₀.

    The d-dimensional PV reduction of Tr[γ^μ(l/+m)γ^ν(l/−k/+m)] gives
    Σ_T = (α/π)[2A₀(m²) − (4m²−k²)B₀(k²;m²,m²)].  A naive 4D tensor
    formula (8B₀₀ − 4A₀ ± 4k²B₁) is INCORRECT because 4×B₀₀ + k²×B₁₁ ≠
    A₀ + m²B₀ in dim-reg (off by finite ε×UV-pole terms).
    """
    from feynman_engine.core.generator import generate_diagrams
    from feynman_engine.physics.translator import parse_process
    from feynman_engine.amplitudes.loop import pv_reduce, LoopTopology
    from feynman_engine.amplitudes import A0Integral, B0Integral
    from sympy import Integer

    spec = parse_process("e+ e- -> e+ e-", theory="QED", loops=1)
    diagrams = generate_diagrams(spec)

    for d in [diag for diag in diagrams if diag.loop_order == 1]:
        exp = pv_reduce(d, "QED")
        if exp and exp.topology == LoopTopology.SELF_ENERGY:
            integral_types = {type(k).__name__ for k in exp.terms}
            # Correct scalar Denner form uses A0 and B0
            assert "A0Integral" in integral_types or "B0Integral" in integral_types, (
                f"Expected A0Integral or B0Integral in photon SE, got {integral_types}"
            )
            # No coefficient should be the trivial placeholder
            for integral, coeff in exp.terms.items():
                assert coeff != Integer(1), (
                    f"Placeholder coefficient 1 found for {type(integral).__name__}"
                )
            return

    pytest.skip("No self-energy diagram found; QGRAF may not be installed.")


def test_vertex_correction_uses_b0_and_c0():
    """Triangle vertex correction uses B₀ and C₀ scalar integrals (correct Denner form)."""
    from feynman_engine.core.generator import generate_diagrams
    from feynman_engine.physics.translator import parse_process
    from feynman_engine.amplitudes.loop import pv_reduce, LoopTopology
    from feynman_engine.amplitudes import B0Integral, C0Integral

    spec = parse_process("e+ e- -> mu+ mu-", theory="QED", loops=1)
    diagrams = generate_diagrams(spec)

    for d in [diag for diag in diagrams if diag.loop_order == 1]:
        exp = pv_reduce(d, "QED")
        if exp and exp.topology == LoopTopology.TRIANGLE:
            integral_types = {type(k).__name__ for k in exp.terms}
            assert "B0Integral" in integral_types or "C0Integral" in integral_types, (
                f"Vertex correction should use B0/C0, got {integral_types}"
            )
            return

    pytest.skip("No triangle diagram found.")


def test_evaluate_pv_expansion_handles_tensor_types():
    """evaluate_pv_expansion does not return None when given B1/B00/C00 types."""
    from feynman_engine.amplitudes import (
        B1Integral, B00Integral, C00Integral,
        evaluate_pv_expansion, looptools_available,
    )
    from feynman_engine.amplitudes.loop import PVExpansion, LoopTopology
    from sympy import Integer, Float

    if not looptools_available():
        pytest.skip("LoopTools not installed")

    # Minimal PVExpansion with B1 and B00 — evaluate at s=4, m²=1
    expansion = PVExpansion(
        process="test",
        diagram_id=0,
        topology=LoopTopology.SELF_ENERGY,
        terms={
            B00Integral(p_sq=4.0, m1_sq=1.0, m2_sq=1.0): Float(1.0),
            B1Integral(p_sq=4.0, m1_sq=1.0, m2_sq=1.0): Float(1.0),
        },
    )
    result = evaluate_pv_expansion(expansion)
    assert result is not None, "evaluate_pv_expansion returned None for B1/B00 integrals"
    assert isinstance(result, complex)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3c — cross-section integration
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_section_importable():
    """cross_section module is importable and exports expected functions."""
    from feynman_engine.amplitudes.cross_section import (
        total_cross_section, differential_cross_section, GEV2_TO_PB,
    )
    assert GEV2_TO_PB > 1e7


def test_differential_cross_section_ee_mumu():
    """dσ/d(cosθ) for e+e-→μ+μ- is positive and finite at cosθ=0."""
    from feynman_engine.amplitudes.cross_section import differential_cross_section

    s_val = 100.0   # (10 GeV)² — well above threshold
    val = differential_cross_section("e+ e- -> mu+ mu-", "QED", s_val=s_val, cos_theta=0.0)
    assert val is not None, "differential_cross_section returned None"
    assert math.isfinite(val), f"dσ/d(cosθ) is not finite: {val}"
    assert val > 0, f"dσ/d(cosθ) should be positive, got {val}"


def test_total_cross_section_ee_mumu_order_of_magnitude():
    """σ(e+e-→μ+μ-) at √s=10 GeV is ~ 0.85 nb = 850 pb (standard QED formula 4πα²/3s)."""
    from feynman_engine.amplitudes.cross_section import total_cross_section

    result = total_cross_section("e+ e- -> mu+ mu-", "QED", sqrt_s=10.0)
    assert result.get("supported"), f"Cross-section not supported: {result}"
    assert result["converged"], "Integration did not converge"

    sigma = result["sigma_pb"]
    assert math.isfinite(sigma) and sigma > 0, f"σ must be positive finite, got {sigma}"

    # Standard analytic result: σ = 4πα²/(3s) ≈ 865 pb at √s=10 GeV
    alpha = 1.0 / 137.036
    s_val = 100.0  # 10² GeV²
    sigma_analytic_pb = 4 * math.pi * alpha**2 / (3 * s_val) * 3.8938e8
    # Allow 1% tolerance (massless approximation is exact here)
    rel_err = abs(sigma - sigma_analytic_pb) / sigma_analytic_pb
    assert rel_err < 0.01, (
        f"σ={sigma:.2f} pb, analytic={sigma_analytic_pb:.2f} pb, rel_err={rel_err:.4f}"
    )


def test_total_cross_section_result_structure():
    """total_cross_section returns expected keys."""
    from feynman_engine.amplitudes.cross_section import total_cross_section

    result = total_cross_section("e+ e- -> mu+ mu-", "QED", sqrt_s=10.0)
    for key in ("sigma_pb", "sigma_uncertainty_pb", "converged",
                "has_tchannel_pole", "cos_theta_range", "formula_latex"):
        assert key in result, f"Missing key '{key}' in cross-section result"


def test_differential_cross_section_1_plus_cos2():
    """dσ/d(cosθ) for e+e-→μ+μ- follows the (1+cos²θ) angular distribution."""
    from feynman_engine.amplitudes.cross_section import differential_cross_section
    import math

    s_val = 100.0
    # dσ/dΩ ∝ (1+cos²θ); ratio at cosθ=0.9 vs cosθ=0 should be (1+0.81)/(1+0) = 1.81
    val_0  = differential_cross_section("e+ e- -> mu+ mu-", "QED", s_val, 0.0)
    val_09 = differential_cross_section("e+ e- -> mu+ mu-", "QED", s_val, 0.9)
    assert val_0 is not None and val_09 is not None
    ratio = val_09 / val_0
    expected = (1 + 0.9**2) / (1 + 0.0**2)  # = 1.81
    assert abs(ratio - expected) < 0.01, f"Angular distribution ratio {ratio:.4f} ≠ {expected:.4f}"


def test_has_tchannel_pole_bhabha():
    """Bhabha scattering (e+e-→e+e-) is correctly identified as having a t-channel pole."""
    from feynman_engine.amplitudes.cross_section import _has_tchannel_pole
    from feynman_engine.physics.amplitude import get_amplitude

    r = get_amplitude("e+ e- -> e+ e-", "QED")
    if r is None:
        pytest.skip("Bhabha amplitude unavailable")
    has_pole = _has_tchannel_pole(r.msq)
    assert has_pole, "Bhabha scattering should have a t-channel pole"


def test_total_cross_section_unsupported_returns_error():
    """total_cross_section returns an error dict for an unknown process."""
    from feynman_engine.amplitudes.cross_section import total_cross_section

    result = total_cross_section("H -> Z Z", "EW", sqrt_s=200.0)
    # Either error (no amplitude) or supported=False
    assert not result.get("supported", True) or "error" in result


def test_cross_section_coupling_override():
    """Passing alpha override changes the cross-section proportionally."""
    from feynman_engine.amplitudes.cross_section import total_cross_section

    r1 = total_cross_section("e+ e- -> mu+ mu-", "QED", sqrt_s=10.0)
    r2 = total_cross_section("e+ e- -> mu+ mu-", "QED", sqrt_s=10.0,
                             coupling_vals={"e": (4 * math.pi / 137.036) ** 0.5 * 2})
    # σ ∝ e⁴, so doubling e should increase σ by factor 16
    if r1.get("supported") and r2.get("supported"):
        ratio = r2["sigma_pb"] / r1["sigma_pb"]
        assert abs(ratio - 16.0) < 0.5, f"Expected ratio ≈ 16 (e doubled → e⁴×16), got {ratio:.2f}"
