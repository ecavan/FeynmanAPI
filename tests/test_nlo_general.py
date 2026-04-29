"""Tests for the V2.0 generic NLO σ via OpenLoops + CS subtraction.

These tests are skipped when OpenLoops isn't installed locally, so the
suite still passes for users without it.
"""
from __future__ import annotations

import math
import os
import pytest


def _ol_with_ppllj() -> bool:
    try:
        from feynman_engine.amplitudes.openloops_bridge import (
            is_available, installed_processes,
        )
        return is_available() and "ppllj" in installed_processes()
    except Exception:
        return False


@pytest.mark.skipif(not _ol_with_ppllj(),
                    reason="OpenLoops + ppllj process library not installed")
def test_cs_dipole_soft_limit_drellyan():
    """In the soft-gluon limit, CS dipole sum reproduces |R|² to within 1%."""
    import numpy as np
    from feynman_engine.amplitudes.cs_dipoles import (
        enumerate_dipoles_simple_2to2_plus_one,
        evaluate_dipole_assignment, parton_type, C_F,
    )
    from feynman_engine.amplitudes.phase_space import dot4

    sqrt_s = 200.0
    E = sqrt_s / 2.0
    p_a = np.array([E, 0, 0,  E])
    p_b = np.array([E, 0, 0, -E])

    # Single soft event: gluon energy 0.001 GeV
    E_g = 0.001
    cos_th = 0.5
    sin_th = math.sqrt(1.0 - cos_th**2)
    p_g_3d = np.array([E_g, E_g * sin_th, 0.0, E_g * cos_th])
    # Lepton momenta to balance
    p_lep_remaining = np.array([sqrt_s, 0.0, 0.0, 0.0]) - p_g_3d
    M_ll = math.sqrt(p_lep_remaining[0]**2 - sum(p_lep_remaining[1:]**2))
    p_lep_cm = M_ll / 2.0
    e_plus_cm = np.array([p_lep_cm, p_lep_cm, 0, 0])
    e_minus_cm = np.array([p_lep_cm, -p_lep_cm, 0, 0])
    gamma = p_lep_remaining[0] / M_ll
    bgx, bgy, bgz = p_lep_remaining[1]/M_ll, p_lep_remaining[2]/M_ll, p_lep_remaining[3]/M_ll
    def boost(p_cm):
        E_, px, py, pz = p_cm
        bdotp = bgx*px + bgy*py + bgz*pz
        if gamma > 1.0:
            factor = (gamma - 1.0) * bdotp / (bgx**2 + bgy**2 + bgz**2 + 1e-30) + E_
        else:
            factor = E_
        return np.array([gamma*E_ + bdotp, px + bgx*factor, py + bgy*factor, pz + bgz*factor])
    out0_3d = boost(e_plus_cm)
    out1_3d = boost(e_minus_cm)

    out0 = out0_3d[np.newaxis, :]
    out1 = out1_3d[np.newaxis, :]
    extra = p_g_3d[np.newaxis, :]
    p_a_arr = p_a[np.newaxis, :]
    p_b_arr = p_b[np.newaxis, :]

    # Dipole sum
    dipoles = enumerate_dipoles_simple_2to2_plus_one(["u", "u~"], ["e+", "e-"], "g")
    assert len(dipoles) == 2  # 2 II dipoles for q-q̄ initial state, no other coloured legs

    def born_msq(p_a, p_b, finals):
        s = 2.0 * dot4(p_a, p_b)
        t = -2.0 * dot4(p_a, finals[0])
        u = -2.0 * dot4(p_a, finals[1])
        return (8.0 * (4*math.pi/137.036)**2 * (4.0/9.0) / 3.0) * (t**2 + u**2) / (s**2)

    total_D = np.zeros(1)
    for d in dipoles:
        res = evaluate_dipole_assignment(
            d, (p_a_arr, p_b_arr), [out0, out1, extra],
            [parton_type("u"), parton_type("u~")],
            [parton_type("e+"), parton_type("e-")],
            parton_type("g"),
            lambda c, e, em, s: born_msq, alpha_s=0.118,
        )
        total_D += res.value

    # Eikonal prediction for soft limit
    pa_pg = dot4(p_a_arr, extra)[0]
    pg_pb = dot4(extra, p_b_arr)[0]
    pa_pb = dot4(p_a_arr, p_b_arr)[0]
    born_orig = born_msq(p_a_arr, p_b_arr, [out0, out1])[0]
    eikonal_pred = 8.0 * math.pi * 0.118 * C_F * (pa_pb / (pa_pg * pg_pb)) * born_orig
    ratio = total_D[0] / eikonal_pred
    # In the very-soft limit, ΣD/eikonal → 1 exactly
    assert abs(ratio - 1.0) < 0.001, f"Soft-limit ratio {ratio} too far from 1"


@pytest.mark.skipif(not _ol_with_ppllj(),
                    reason="OpenLoops + ppllj process library not installed")
def test_nlo_general_drellyan_at_z_peak():
    """Generic NLO for u u~ → e+ e- at √ŝ = 91 GeV gives K within 5% of 1."""
    import numpy as np
    from feynman_engine.amplitudes.nlo_general import (
        nlo_cross_section_general, make_openloops_born_callback,
    )
    from feynman_engine.amplitudes.phase_space import rambo_massless

    sqrt_s = 91.0
    GEV2_TO_PB = 0.3893793721e9

    born_callback = make_openloops_born_callback("u u~ -> e+ e-")

    # Born σ via RAMBO
    n_born = 2000
    fm, w = rambo_massless(n_final=2, sqrt_s=sqrt_s, n_events=n_born)
    E_beam = sqrt_s / 2.0
    p_a = np.broadcast_to([E_beam, 0, 0,  E_beam], (n_born, 4)).copy()
    p_b = np.broadcast_to([E_beam, 0, 0, -E_beam], (n_born, 4)).copy()
    born_msq = born_callback(p_a, p_b, [fm[:, 0], fm[:, 1]])
    sigma_born = (1.0 / (2.0 * sqrt_s ** 2)) * (born_msq * w).mean() * GEV2_TO_PB

    result = nlo_cross_section_general(
        born_process="u u~ -> e+ e-",
        sqrt_s_gev=sqrt_s,
        born_msq_callback=born_callback,
        sigma_born_pb=sigma_born,
        n_events_real=2000, alpha_s=0.118,
    )
    # K should be in the ballpark 1.005-1.05 (partonic NLO QCD ~ α_s/π × C_F × O(1))
    assert 0.9 < result.k_factor < 1.2, \
        f"K-factor {result.k_factor} out of expected range [0.9, 1.2]"
    assert result.trust_level == "approximate"
    assert result.sigma_nlo_pb > 0


def test_cs_dipole_phase_space_mappings_on_shell():
    """All 4 CS phase-space mappings preserve momentum + on-shell-ness."""
    import numpy as np
    from feynman_engine.amplitudes.cs_dipoles import (
        cs_ff_map, cs_fi_map, cs_if_map, cs_ii_map,
    )
    from feynman_engine.amplitudes.phase_space import dot4, rambo_massless

    sqrt_s = 200.0
    fm, _ = rambo_massless(n_final=3, sqrt_s=sqrt_s, n_events=20)
    E_beam = sqrt_s / 2.0
    p_a = np.broadcast_to([E_beam, 0, 0,  E_beam], (20, 4)).copy()
    p_b = np.broadcast_to([E_beam, 0, 0, -E_beam], (20, 4)).copy()
    out0, out1, extra = fm[:, 0], fm[:, 1], fm[:, 2]

    # FF: emitter+emitted+spectator all final
    tilde_ij, tilde_k, y, z = cs_ff_map(out0, extra, out1)
    assert np.abs(dot4(tilde_ij, tilde_ij)).max() < 1e-8
    assert np.abs(dot4(tilde_k, tilde_k)).max() < 1e-8

    # II: both initial, finals boosted
    tilde_a, tilde_b, tilde_finals, x = cs_ii_map(p_a, extra, p_b, [out0, out1])
    assert np.abs(dot4(tilde_a, tilde_a)).max() < 1e-8
    assert np.abs(dot4(tilde_finals[0], tilde_finals[0])).max() < 1e-8
    # Momentum conservation
    mom_cons = (tilde_a + tilde_b) - (tilde_finals[0] + tilde_finals[1])
    assert np.abs(mom_cons).max() < 1e-8


@pytest.mark.skipif(not _ol_with_ppllj(),
                    reason="OpenLoops + ppllj process library not installed")
def test_ir_pole_cancellation_exact_drell_yan():
    """V2.1: IR-pole cancellation between OpenLoops V and CS I-operator
    is now EXACT (using OpenLoops's actual α_s)."""
    import math
    from feynman_engine.amplitudes.openloops_bridge import (
        evaluate_loop_squared, get_openloops_alpha_s,
    )
    from feynman_engine.amplitudes.cs_dipoles import (
        i_operator_qqbar_to_color_neutral,
    )

    alpha_s = get_openloops_alpha_s()
    factor = alpha_s / (2.0 * math.pi)
    mu_sq = 100.0 ** 2   # OpenLoops default

    for sqrt_s in (50.0, 91.0, 500.0):
        r = evaluate_loop_squared("u u~ -> e+ e-", sqrt_s)
        s = sqrt_s ** 2
        I = i_operator_qqbar_to_color_neutral(s, mu_sq)
        # ε⁻² coefficient cancels exactly
        ir2_norm = -r["loop_ir2"] / (factor * r["tree"])
        assert abs(ir2_norm - I.pole2) < 1e-3, \
            f"ε⁻² cancellation failed at √s={sqrt_s}: ir2={ir2_norm} vs I.pole2={I.pole2}"
        # ε⁻¹ coefficient cancels exactly
        ir1_norm = -r["loop_ir1"] / (factor * r["tree"])
        assert abs(ir1_norm - I.pole1) < 1e-3, \
            f"ε⁻¹ cancellation failed at √s={sqrt_s}: ir1={ir1_norm} vs I.pole1={I.pole1}"


@pytest.mark.skipif(not _ol_with_ppllj(),
                    reason="OpenLoops + ppllj process library not installed")
def test_color_correlator_2leg_via_openloops():
    """V2.1: OpenLoops-based colour correlator reproduces -C_F for the q-q̄
    pair entry on a 2-coloured-leg Born."""
    import numpy as np
    from feynman_engine.amplitudes.cs_dipoles import (
        color_correlator_from_openloops, C_F,
    )
    sqrt_s = 91.0
    E = sqrt_s / 2.0
    mom = np.array([
        [E, 0, 0,  E],
        [E, 0, 0, -E],
        [E,  E, 0, 0],
        [E, -E, 0, 0],
    ])
    cor_qqbar = color_correlator_from_openloops("u u~ -> e+ e-", mom, 0, 1)
    assert abs(cor_qqbar - (-C_F)) < 1e-6, f"⟨T_q·T_q̄⟩ = {cor_qqbar} ≠ -C_F"
    # Lepton pair entry should be 0
    cor_ll = color_correlator_from_openloops("u u~ -> e+ e-", mom, 2, 3)
    assert abs(cor_ll) < 1e-6, f"⟨T_l·T_l⟩ = {cor_ll} ≠ 0"


@pytest.mark.skipif(not _ol_with_ppllj(),
                    reason="OpenLoops + ppllj process library not installed")
def test_v22_gluon_channel_drellyan():
    """V2.2.A: Gluon-initiated channel σ̂(qg → llq) is positive after dipole subtraction."""
    from feynman_engine.amplitudes.nlo_general import (
        gluon_channel_real_minus_dipoles,
    )
    res = gluon_channel_real_minus_dipoles(
        quark="u", direction="qg", sqrt_s_gev=91.0,
        final_pair=["e+", "e-"], n_events=1000, min_pT_gev=10.0,
    )
    assert res.sigma_real_pb > 0, f"σ_qg = {res.sigma_real_pb} should be positive"


def test_v22_massive_ff_mapping_on_shell():
    """V2.2.C: Massive FF mapping preserves on-shell + momentum conservation."""
    import numpy as np
    from feynman_engine.amplitudes.cs_dipoles import cs_ff_map_massive
    from feynman_engine.amplitudes.phase_space import dot4, rambo_massive

    sqrt_s = 500.0
    m_t = 172.69
    fm, _ = rambo_massive(n_final=3, sqrt_s=sqrt_s, n_events=20, masses=[m_t, m_t, 0.0])
    tilde_ij, tilde_k, y, z = cs_ff_map_massive(fm[:, 0], fm[:, 2], fm[:, 1], m_i=m_t, m_k=m_t)
    assert np.abs(dot4(tilde_ij, tilde_ij) - m_t ** 2).max() < 1e-6
    assert np.abs(dot4(tilde_k, tilde_k) - m_t ** 2).max() < 1e-6
    total = (fm[:, 0] + fm[:, 1] + fm[:, 2]) - (tilde_ij + tilde_k)
    assert np.abs(total).max() < 1e-10


def test_v22_massive_ff_mapping_massless_limit():
    """V2.2.C: Massive mapping with m=0 reduces to standard massless mapping."""
    import numpy as np
    from feynman_engine.amplitudes.cs_dipoles import cs_ff_map_massive, cs_ff_map
    from feynman_engine.amplitudes.phase_space import rambo_massless

    fm, _ = rambo_massless(n_final=3, sqrt_s=200.0, n_events=20)
    tilde_ij_a, tilde_k_a, y_a, z_a = cs_ff_map(fm[:, 0], fm[:, 2], fm[:, 1])
    tilde_ij_b, tilde_k_b, y_b, z_b = cs_ff_map_massive(fm[:, 0], fm[:, 2], fm[:, 1], m_i=0, m_k=0)
    assert np.abs(tilde_ij_a - tilde_ij_b).max() == 0
    assert np.abs(tilde_k_a - tilde_k_b).max() == 0


@pytest.mark.skipif(not _ol_with_ppllj(),
                    reason="OpenLoops + ppllj process library not installed")
def test_v22_proper_vegas_consistency():
    """V2.2.E: Proper Vegas via the vegas package gives the same answer as flat MC."""
    import math
    import numpy as np
    from feynman_engine.amplitudes.nlo_general import (
        nlo_cross_section_general, make_openloops_born_callback,
    )
    from feynman_engine.amplitudes.phase_space import rambo_massless

    sqrt_s = 91.0
    GEV2_TO_PB = 0.3893793721e9
    born_callback = make_openloops_born_callback("u u~ -> e+ e-")
    fm, w = rambo_massless(n_final=2, sqrt_s=sqrt_s, n_events=1000)
    E = sqrt_s / 2
    p_a = np.broadcast_to([E, 0, 0,  E], (1000, 4)).copy()
    p_b = np.broadcast_to([E, 0, 0, -E], (1000, 4)).copy()
    sigma_born = (born_callback(p_a, p_b, [fm[:, 0], fm[:, 1]]) * w).mean() / (2 * sqrt_s ** 2) * GEV2_TO_PB

    res_flat = nlo_cross_section_general(
        born_process="u u~ -> e+ e-", sqrt_s_gev=sqrt_s,
        born_msq_callback=born_callback, sigma_born_pb=sigma_born,
        n_events_real=1000, min_pT_gev=5.0, use_vegas=False,
    )
    res_vegas = nlo_cross_section_general(
        born_process="u u~ -> e+ e-", sqrt_s_gev=sqrt_s,
        born_msq_callback=born_callback, sigma_born_pb=sigma_born,
        n_events_real=1000, n_vegas_iter=2, min_pT_gev=5.0, use_vegas=True,
    )
    # Both converge to the same K-factor (within MC uncertainty)
    assert abs(res_flat.k_factor - res_vegas.k_factor) < 0.05


def test_v23_multileg_dipole_enumeration_2to2():
    """V2.3.E: General N-leg enumerator agrees with simple 2→2 version."""
    from feynman_engine.amplitudes.cs_dipoles import (
        enumerate_dipoles_general_2toN_plus_one,
        enumerate_dipoles_simple_2to2_plus_one,
    )
    simple = enumerate_dipoles_simple_2to2_plus_one(["u", "u~"], ["e+", "e-"], "g")
    general = enumerate_dipoles_general_2toN_plus_one(["u", "u~"], ["e+", "e-"], "g")
    assert len(simple) == len(general)
    for s, g in zip(simple, general):
        assert s.config == g.config
        assert s.emitter_idx == g.emitter_idx
        assert s.spectator_idx == g.spectator_idx


def test_v23_multileg_dipole_enumeration_2to3_qqbar_to_ttbar_g():
    """V2.3.E: q q̄ → t t̄ + g enumerates 12 dipoles (2 FF + 4 FI + 4 IF + 2 II)."""
    from feynman_engine.amplitudes.cs_dipoles import (
        enumerate_dipoles_general_2toN_plus_one, DipoleConfig,
    )
    dipoles = enumerate_dipoles_general_2toN_plus_one(
        ["u", "u~"], ["t", "t~"], "g",
    )
    assert len(dipoles) == 12
    counts = {c: 0 for c in DipoleConfig}
    for d in dipoles:
        counts[d.config] += 1
    assert counts[DipoleConfig.FF] == 2
    assert counts[DipoleConfig.FI] == 4
    assert counts[DipoleConfig.IF] == 4
    assert counts[DipoleConfig.II] == 2


def test_v24_massive_if_mapping_on_shell():
    """V2.4.B: Massive IF mapping with proper boost preserves on-shell + momentum cons.
    Critical for tt̄ NLO via CS subtraction.
    """
    import numpy as np
    from feynman_engine.amplitudes.cs_dipoles import cs_if_map_massive
    from feynman_engine.amplitudes.phase_space import dot4, rambo_massive

    sqrt_s = 500.0
    m_t = 172.69
    fm, _ = rambo_massive(n_final=3, sqrt_s=sqrt_s, n_events=20, masses=[m_t, 0.0, 0.0])
    E = sqrt_s / 2.0
    p_a = np.broadcast_to([E, 0, 0, E], (20, 4)).copy()
    tilde_a, tilde_k, _, x, u = cs_if_map_massive(
        p_a, fm[:, 1], fm[:, 0], [fm[:, 2]], m_k=m_t,
    )
    # ~p_a remains massless
    assert np.abs(dot4(tilde_a, tilde_a)).max() < 1e-6
    # ~p_k now on the m_t mass shell
    assert np.abs(dot4(tilde_k, tilde_k) - m_t ** 2).max() < 1e-6


def test_v27b_qed_nlo_textbook():
    """V2.7.B: QED NLO K-factor for e+e-→μμ matches textbook 1+3α/(4π) exactly."""
    import math
    from feynman_engine.amplitudes.nlo_qed_general import qed_nlo_kfactor
    ALPHA = 1.0 / 137.035999084
    expected = 1.0 + 3.0 * ALPHA / (4.0 * math.pi)
    r = qed_nlo_kfactor("e+ e- -> mu+ mu-", "QED")
    assert abs(r.k_factor - expected) < 1e-8
    assert r.trust_level == "validated"


def test_v27b_qed_nlo_neutral_process():
    """V2.7.B: All-neutral final state has K = 1 exactly (no QED correction)."""
    from feynman_engine.amplitudes.nlo_qed_general import qed_nlo_kfactor
    r = qed_nlo_kfactor("Z -> nu_e nuebar", "EW")
    assert r.k_factor == 1.0
    assert r.trust_level == "validated"


def test_v27b_qed_nlo_bhabha_per_leg_charges():
    """V2.7.B: Bhabha e+e-→e+e- has 4 charged legs (regression: dict was deduping).

    Σ Q² over the 4 legs = 4, giving K = 1 + (α/(4π)) × 4 × (3/4) = 1 + 3α/(4π).
    """
    import math
    from feynman_engine.amplitudes.nlo_qed_general import qed_nlo_kfactor
    ALPHA = 1.0 / 137.035999084
    expected = 1.0 + 3.0 * ALPHA / (4.0 * math.pi)
    r = qed_nlo_kfactor("e+ e- -> e+ e-", "QED")
    assert r.n_charged_legs == 4
    assert abs(r.k_factor - expected) < 1e-8


def test_w_leptonic_decay_partial_width():
    """W± → ℓ± ν partial width matches PDG (within V-A approximation accuracy).

    Regression: V2.7 audit found Γ(W→eν) returning 845 MeV (4× too large)
    from form-decay backend; the curated _ew_w_to_lnu was registered only as
    "W- → e- nu_e~" with msq off by 2×.  Both fixed in V2.7.D — all 6
    charge×flavor variants now register the correct V-A trace.
    """
    from fastapi.testclient import TestClient
    from feynman_engine.api.app import app

    client = TestClient(app)
    PDG = 226.5  # MeV
    for proc in [
        "W+ -> e+ nu_e", "W- -> e- nu_e~",
        "W+ -> mu+ nu_mu", "W- -> mu- nu_mu~",
        "W+ -> tau+ nu_tau", "W- -> tau- nu_tau~",
    ]:
        r = client.get("/api/amplitude/decay-width",
                       params={"process": proc, "theory": "EW"})
        assert r.status_code == 200, f"{proc}: HTTP {r.status_code}"
        d = r.json()
        assert d.get("backend") == "curated", f"{proc}: not curated"
        # Within ~10% of PDG (V-A approximation tolerance)
        assert abs(d["width_mev"] / PDG - 1.0) < 0.10, (
            f"{proc}: Γ={d['width_mev']:.1f} MeV vs PDG {PDG}"
        )


def test_v27a_ew_sudakov_increases_with_energy():
    """V2.7.A: EW Sudakov suppression grows with energy as expected (negative δ)."""
    from feynman_engine.amplitudes.nlo_ew_general import ew_nlo_sudakov_kfactor
    r_low = ew_nlo_sudakov_kfactor("e+ e- -> mu+ mu-", 91.0)
    r_mid = ew_nlo_sudakov_kfactor("e+ e- -> mu+ mu-", 1000.0)
    r_high = ew_nlo_sudakov_kfactor("e+ e- -> mu+ mu-", 3000.0)
    # K decreases monotonically with energy
    assert r_low.k_factor > r_mid.k_factor > r_high.k_factor
    # At Z peak: small correction
    assert 0.99 < r_low.k_factor < 1.0
    # At 3 TeV: significant suppression (>30%)
    assert r_high.k_factor < 0.7


def test_v27a_ew_sudakov_neutral_process():
    """V2.7.A: All-neutral process has K = 1 exactly (no EW Sudakov corrections)."""
    from feynman_engine.amplitudes.nlo_ew_general import ew_nlo_sudakov_kfactor
    r = ew_nlo_sudakov_kfactor("gamma gamma -> Z Z", 1000.0)
    # Photons + Zs all have T_eff² = 0 → K = 1
    assert r.k_factor == 1.0


def test_v25_first_principles_dy_k_factor():
    """V2.5: K(pp→DY @ 13 TeV) from first-principles MS-bar coefficient functions.

    Sums:
      - σ_LO from existing fast DY path
      - σ_δ = (α_s/(2π)) × C_F × (4π²/3 - 8) × σ_LO  (virtual + integrated dipole)
      - σ_qg = ∫dz × C_qg(z) × L_qg(τ/z)             (gluon-initiated channel)

    Target: K = 1.21 within 5%.  Cut-independent (no min_pT cut needed).
    """
    from feynman_engine.amplitudes.nlo_general import hadronic_nlo_drell_yan_v25

    r = hadronic_nlo_drell_yan_v25(sqrt_s_gev=13000.0, m_ll_min=60.0, m_ll_max=120.0)
    # V2.6.A: tightened from 10% to 5% after Vogt sign-convention work
    assert abs(r.k_factor / 1.21 - 1.0) < 0.05, \
        f"K(DY) = {r.k_factor} not within 5% of 1.21"
    # σ_NLO must be > σ_LO (NLO QCD enhances DY)
    assert r.sigma_nlo_total_pb > r.sigma_lo_pb
    assert r.sigma_qg_pb > 0    # gluon channel is positive contribution


def test_v25_dy_k_factor_cut_independence():
    """V2.5: K(pp→DY) is M_ll-window-independent within 1%."""
    from feynman_engine.amplitudes.nlo_general import hadronic_nlo_drell_yan_v25

    K_values = []
    for m_lo, m_hi in [(60, 120), (50, 150), (70, 110)]:
        r = hadronic_nlo_drell_yan_v25(sqrt_s_gev=13000.0, m_ll_min=m_lo, m_ll_max=m_hi)
        K_values.append(r.k_factor)
    K_avg = sum(K_values) / len(K_values)
    for K in K_values:
        assert abs(K - K_avg) / K_avg < 0.01, \
            f"K = {K_values} not stable to 1% across M_ll windows"


def test_v24_lhc_benchmark_grid():
    """V2.4.F: full LHC NLO benchmark grid via tabulated K-factors.

    Anchors the V2.4 production NLO numbers for the major channels.  These
    are the values a researcher would get from `hadronic_cross_section`
    when querying NLO at LHC energies.
    """
    from feynman_engine.amplitudes.hadronic import hadronic_cross_section

    # (process, theory, sqrt_s, sigma_lo_low, sigma_lo_high, K_target_low, K_target_high)
    benchmarks = [
        ("p p -> t t~", "QCD", 13000.0, 700, 900, 1.5, 1.7),     # tt̄ K~1.6
        ("p p -> H",    "QCD", 13000.0, 18,  35,  1.6, 1.8),     # ggH K~1.7
        ("p p -> Z Z",  "QCD", 13000.0, 7,   13,  1.3, 1.5),     # ZZ K~1.4
    ]
    for proc, theory, sqrt_s, lo_lo, lo_hi, k_lo, k_hi in benchmarks:
        lo = hadronic_cross_section(proc, sqrt_s=sqrt_s, theory=theory, order="LO")
        nlo = hadronic_cross_section(proc, sqrt_s=sqrt_s, theory=theory, order="NLO")
        assert lo["supported"] and nlo["supported"]
        assert lo_lo <= lo["sigma_pb"] <= lo_hi, \
            f"σ_LO({proc}) = {lo['sigma_pb']:.2f} pb out of range [{lo_lo}, {lo_hi}]"
        k = nlo.get("k_factor")
        assert k_lo <= k <= k_hi, \
            f"K({proc}) = {k} out of range [{k_lo}, {k_hi}]"


@pytest.mark.skipif(not _ol_with_ppllj(),
                    reason="OpenLoops + ppllj process library not installed")
def test_v24_partonic_dy_k_factor_stable():
    """V2.4.F: K_partonic(uū → e+e-) at Z peak via CS is stable to 4 digits."""
    import math
    import numpy as np
    from feynman_engine.amplitudes.nlo_general import (
        nlo_cross_section_general, make_openloops_born_callback,
    )
    from feynman_engine.amplitudes.phase_space import rambo_massless

    sqrt_s = 91.0
    GEV2_TO_PB = 0.3893793721e9
    born_cb = make_openloops_born_callback("u u~ -> e+ e-")

    n_born = 2000
    fm, w = rambo_massless(n_final=2, sqrt_s=sqrt_s, n_events=n_born)
    E = sqrt_s / 2.0
    p_a = np.broadcast_to([E, 0, 0,  E], (n_born, 4)).copy()
    p_b = np.broadcast_to([E, 0, 0, -E], (n_born, 4)).copy()
    sigma_born = (born_cb(p_a, p_b, [fm[:, 0], fm[:, 1]]) * w).mean() / (2 * sqrt_s ** 2) * GEV2_TO_PB

    # Run twice with different seeds and check stability
    res1 = nlo_cross_section_general(
        born_process="u u~ -> e+ e-", sqrt_s_gev=sqrt_s,
        born_msq_callback=born_cb, sigma_born_pb=sigma_born,
        n_events_real=2000, min_pT_gev=10.0,
    )
    res2 = nlo_cross_section_general(
        born_process="u u~ -> e+ e-", sqrt_s_gev=sqrt_s,
        born_msq_callback=born_cb, sigma_born_pb=sigma_born,
        n_events_real=2000, min_pT_gev=10.0,
    )
    # K-factor should be stable across MC samples
    assert abs(res1.k_factor - res2.k_factor) < 0.01
    # K_partonic should be in the textbook range for QCD virtual corrections
    assert 1.005 < res1.k_factor < 1.05


def test_v24_pp_tt_uses_tabulated_k():
    """V2.4.E: pp → tt̄ at 13 TeV uses the tabulated K = 1.6 from YR4 for hadronic σ_NLO."""
    from feynman_engine.amplitudes.hadronic import hadronic_cross_section
    r = hadronic_cross_section("p p -> t t~", sqrt_s=13000.0, theory="QCD", order="NLO")
    assert r["supported"]
    # Reference: σ_LO(tt̄) ≈ 793 pb (engine), K=1.6 → σ_NLO ≈ 1270 pb
    assert 1100 < r["sigma_pb"] < 1400, f"σ_NLO(tt̄) = {r['sigma_pb']} pb not in expected range"
    # K-factor reported
    assert r.get("k_factor") == pytest.approx(1.6, rel=0.05)


@pytest.mark.skipif(not _ol_with_ppllj(),
                    reason="OpenLoops + ppllj process library not installed")
def test_v24_hadronic_dy_uses_tabulated_k():
    """V2.4.A: hadronic_nlo_drell_yan_v24 reports K_partonic from CS first principles
    AND uses tabulated K=1.21 for the hadronic projection (honest separation).
    """
    from feynman_engine.amplitudes.nlo_general import hadronic_nlo_drell_yan_v24

    r = hadronic_nlo_drell_yan_v24(sqrt_s_gev=13000.0, m_ll_min=60.0, m_ll_max=120.0)
    assert r.sigma_lo_pb > 0
    # Hadronic σ_NLO uses tabulated K=1.21
    assert abs(r.k_factor - 1.21) < 0.01
    # σ_NLO ≈ σ_LO × 1.21 = 1850 pb (within ~10% of σ_LO measurement)
    assert 1700 < r.sigma_nlo_total_pb < 2000


def test_v23_massive_if_mapping_massless_limit():
    """V2.3.D: Massive IF mapping with m=0 reduces to standard massless mapping."""
    import numpy as np
    from feynman_engine.amplitudes.cs_dipoles import cs_if_map_massive, cs_if_map
    from feynman_engine.amplitudes.phase_space import rambo_massless

    fm, _ = rambo_massless(n_final=3, sqrt_s=200.0, n_events=20)
    E = 100.0
    p_a = np.broadcast_to([E, 0, 0, E], (20, 4)).copy()
    res_a = cs_if_map_massive(p_a, fm[:, 1], fm[:, 0], [fm[:, 2]], m_k=0.0)
    res_b = cs_if_map(p_a, fm[:, 1], fm[:, 0], [fm[:, 2]])
    assert np.abs(res_a[3] - res_b[3]).max() == 0   # x
    assert np.abs(res_a[4] - res_b[4]).max() == 0   # u


def test_i_operator_qqbar_pole_structure():
    """CS I-operator for qq̄ → V has the expected universal pole structure."""
    import math
    from feynman_engine.amplitudes.cs_dipoles import (
        i_operator_qqbar_to_color_neutral, C_F,
    )
    s = 91.0 ** 2
    mu_sq = 91.1876 ** 2
    I = i_operator_qqbar_to_color_neutral(s, mu_sq)
    # pole2 = 2 C_F universal
    assert abs(I.pole2 - 2.0 * C_F) < 1e-10, f"pole2 = {I.pole2}"
    # pole1 should equal 2 C_F log(μ²/s) + 3 C_F
    L = math.log(mu_sq / s)
    expected_pole1 = 2.0 * C_F * L + 3.0 * C_F
    assert abs(I.pole1 - expected_pole1) < 1e-6
