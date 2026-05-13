"""Closed-form 3-body Fermi decay widths.

For pure 4-fermion (V-A × V-A) decays in the Fermi-theory limit
(momentum transfer ≪ m_W), the 3-body Γ has a closed Sargent-Cabibbo
form that avoids any phase-space integration:

    Γ(f₁ → f₂ + ν̄ + ν) = G_F² · M⁵ / (192π³) · f(r) · |V_CKM|²

where M = m(f₁), r = m(f₂)/M, and the kinematic phase-space factor is

    f(r) = 1 - 8 r² + 8 r⁶ - r⁸ - 24 r⁴ ln(r)

(Sargent 1932 in the limit r→0; full formula Sirlin 1978, Marciano 1988.)

Supported channels — all leptonic and CKM-diagonal semileptonic where
the final-state quark mass is small enough that the Sargent form holds:

  μ⁻ → e⁻ ν̄_e ν_μ
  τ⁻ → e⁻ ν̄_e ν_τ
  τ⁻ → μ⁻ ν̄_μ ν_τ
  τ⁻ → d ū ν_τ          (hadronic, ×N_c=3, ×|V_ud|²)
  τ⁻ → s ū ν_τ          (hadronic, ×N_c=3, ×|V_us|²)
  b → c ℓ⁻ ν̄_ℓ          (b-quark semileptonic, ×|V_cb|²)
  b → u ℓ⁻ ν̄_ℓ          (b-quark semileptonic, ×|V_ub|²)
  c → s ℓ⁺ ν_ℓ          (c-quark semileptonic, ×|V_cs|²)

The formula breaks down for processes with intermediate W-resonance
(t → b ℓ ν is on-shell W → use 1→2 chain) or non-V-A structure.
"""
from __future__ import annotations

import math
from typing import Optional

from feynman_engine.amplitudes.pdg_masses import MASS_GEV

# ────────────────────────────────────────────────────────────────────────────
# Physical constants
# ────────────────────────────────────────────────────────────────────────────

_G_F = 1.1663787e-5     # Fermi constant in GeV⁻² (PDG 2024)
_N_C = 3                # quark colours
_HBAR_GEV_S = 6.582119569e-25  # ℏ in GeV·s, used if we want τ; Γ alone fine

# CKM matrix elements (PDG 2024 averages).  Only diagonal+light needed.
_CKM = {
    ("u", "d"): 0.97435,
    ("u", "s"): 0.22500,
    ("u", "b"): 0.00382,
    ("c", "d"): 0.22486,
    ("c", "s"): 0.97349,
    ("c", "b"): 0.04182,
    ("t", "d"): 0.00857,
    ("t", "s"): 0.04110,
    ("t", "b"): 0.999118,
}

def _ckm(q_up: str, q_dn: str) -> float:
    """|V_CKM|: look up V_{up,down} with up-type and down-type keys."""
    return _CKM.get((q_up, q_dn), _CKM.get((q_dn, q_up), 0.0))


# ────────────────────────────────────────────────────────────────────────────
# Sargent kinematic factor
# ────────────────────────────────────────────────────────────────────────────

def _sargent_f(r: float) -> float:
    """f(r) = 1 - 8r² + 8r⁶ - r⁸ - 24r⁴ ln(r).

    r = m_final_fermion / m_parent.  For r → 0 (massless final fermion)
    f(r) → 1; the leading correction is -8r².
    """
    if r <= 0.0:
        return 1.0
    if r >= 1.0:
        return 0.0
    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r4 * r4
    return 1.0 - 8.0 * r2 + 8.0 * r6 - r8 - 24.0 * r4 * math.log(r)


# ────────────────────────────────────────────────────────────────────────────
# Process recognition
# ────────────────────────────────────────────────────────────────────────────

_LEPTON_KEYS = {
    "e-":   "e",   "e+":  "e",
    "mu-":  "mu",  "mu+": "mu",
    "tau-": "tau", "tau+": "tau",
}

_LEPTON_NU_KEY = {
    "e":   ("nu_e",   "nu_e~"),
    "mu":  ("nu_mu",  "nu_mu~"),
    "tau": ("nu_tau", "nu_tau~"),
}


def _parse_3body(parent: str, daughters: list[str]) -> Optional[dict]:
    """Classify a 1→3 process as 'leptonic-fermi' or 'semileptonic'.

    Returns dict with keys:
      kind:       'leptonic-fermi' | 'semileptonic-quark' | 'tau-hadronic'
      m_parent:   parent mass (GeV)
      m_daughter: 'heavy' charged-fermion daughter mass (GeV)
      colour_factor:  1 or N_c=3
      ckm_sq:     |V_CKM|² for the W vertex (1.0 for pure leptonic)
      label:      human-readable
    or None if the process doesn't fit the Sargent template.
    """
    if len(daughters) != 3:
        return None

    p_key = parent.rstrip("+-").replace("~", "")
    # Parent must be a lepton or a charged quark
    if p_key in {"mu", "tau"}:
        parent_kind = "lepton"
        parent_mass = MASS_GEV.get(f"m_{p_key}", 0.0)
    elif p_key in {"b", "c"}:
        parent_kind = "quark"
        parent_mass = MASS_GEV.get(f"m_{p_key}", 0.0)
    else:
        return None
    if parent_mass <= 0:
        return None

    # Categorise daughters
    charged_lep = None
    neutrinos = []
    light_quark = None
    quark_pair_up = None
    quark_pair_dn = None
    for d in daughters:
        d_clean = d.rstrip("+-").replace("~", "")
        if d in _LEPTON_KEYS:
            if charged_lep is None:
                charged_lep = d
            else:
                # Two charged leptons → not a Fermi decay
                return None
        elif d_clean.startswith("nu"):
            neutrinos.append(d)
        elif d_clean in {"u", "d", "s", "c", "b"}:
            if d_clean in {"u", "c", "t"}:
                quark_pair_up = d_clean
            elif d_clean in {"d", "s", "b"}:
                quark_pair_dn = d_clean
            light_quark = d_clean
        else:
            return None

    # ─── Case 1: leptonic Fermi (μ → eν̄ν, τ → eν̄ν, τ → μν̄ν) ───
    if parent_kind == "lepton" and charged_lep and len(neutrinos) == 2:
        lep_key = _LEPTON_KEYS[charged_lep]
        # The neutrinos must be one of each generation: the daughter-lepton ν̄
        # + the parent-lepton ν (or charge-conjugate for τ⁺ etc.).
        # We don't strictly check generations here — universal Fermi formula
        # applies as long as the topology is leptonic.
        m_daughter = MASS_GEV.get(f"m_{lep_key}", 0.0)
        return {
            "kind": "leptonic-fermi",
            "m_parent": parent_mass,
            "m_daughter": m_daughter,
            "colour_factor": 1,
            "ckm_sq": 1.0,
            "label": f"{parent} → {charged_lep} ν̄ ν (universal Fermi)",
        }

    # ─── Case 2: tau → quark q̄' ν_τ (semileptonic on tau, but NO charged lepton) ───
    if parent_kind == "lepton" and p_key == "tau" and quark_pair_up and quark_pair_dn:
        # Mass to subtract is the heavier of the two outgoing quarks.
        m_q_up = MASS_GEV.get(f"m_{quark_pair_up}", 0.0)
        m_q_dn = MASS_GEV.get(f"m_{quark_pair_dn}", 0.0)
        m_daughter = max(m_q_up, m_q_dn)
        ckm = _ckm(quark_pair_up, quark_pair_dn)
        return {
            "kind": "tau-hadronic",
            "m_parent": parent_mass,
            "m_daughter": m_daughter,
            "colour_factor": _N_C,
            "ckm_sq": ckm * ckm,
            "label": f"{parent} → {quark_pair_up} {quark_pair_dn}~ ν_τ (N_c=3, |V_{quark_pair_up}{quark_pair_dn}|²={ckm**2:.4f})",
        }

    # ─── Case 3: heavy-quark semileptonic (b → c ℓν̄, b → u ℓν̄, c → s ℓν) ───
    if parent_kind == "quark" and charged_lep and len(neutrinos) == 1 and light_quark:
        # m_daughter is the light quark mass (largest correction)
        m_daughter = MASS_GEV.get(f"m_{light_quark}", 0.0)
        # CKM for the heavy → light transition (e.g. b→c uses V_cb)
        ckm = _ckm(p_key, light_quark) if light_quark in {"u","c"} or p_key in {"u","c","t"} \
                                       else _ckm(light_quark, p_key)
        return {
            "kind": "semileptonic-quark",
            "m_parent": parent_mass,
            "m_daughter": m_daughter,
            "colour_factor": 1,
            "ckm_sq": ckm * ckm,
            "label": f"{parent} → {light_quark} {charged_lep} ν̄ (|V_{p_key}{light_quark}|²={ckm**2:.4f})",
        }

    return None


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────

def three_body_fermi_width(process: str) -> Optional[dict]:
    """Closed-form 3-body decay width via the Sargent-Cabibbo formula.

    Parameters
    ----------
    process : str
        Process string like ``"tau- -> e- nu_e~ nu_tau"`` or
        ``"b -> c e- nu_e~"``.

    Returns
    -------
    dict with keys ``width_gev``, ``width_mev``, ``m_parent_gev``,
    ``m_daughter_gev``, ``sargent_factor_f``, ``kind``, ``ckm_sq``,
    ``colour_factor``, ``formula`` or None if the process doesn't
    match a Sargent template.
    """
    if "->" not in process:
        return None
    parent, rest = process.split("->", 1)
    parent = parent.strip()
    daughters = [d for d in rest.strip().split() if d]
    if not parent or not daughters:
        return None

    cls = _parse_3body(parent, daughters)
    if cls is None:
        return None

    M = cls["m_parent"]
    m = cls["m_daughter"]
    r = m / M if M > 0 else 0.0
    f_r = _sargent_f(r)

    width_gev = (
        _G_F * _G_F * M ** 5 / (192.0 * math.pi ** 3)
        * f_r
        * cls["colour_factor"]
        * cls["ckm_sq"]
    )

    return {
        "process":           process,
        "width_gev":         width_gev,
        "width_mev":         width_gev * 1000.0,
        "m_parent_gev":      M,
        "m_daughter_gev":    m,
        "r":                 r,
        "sargent_factor_f":  f_r,
        "kind":              cls["kind"],
        "ckm_sq":            cls["ckm_sq"],
        "colour_factor":     cls["colour_factor"],
        "formula":           "Γ = G_F²·M⁵/(192π³) · f(r) · N_c · |V_CKM|²",
        "method":            "sargent-cabibbo-closed-form",
        "trust_level":       "validated",
        "reference":         "Sargent 1932; Sirlin PRD 22 (1980); Marciano-Sirlin PRL 61 (1988); PDG 2024 §10.",
        "label":             cls["label"],
        "supported":         True,
    }


# ────────────────────────────────────────────────────────────────────────────
# Tau hadronic resonance modes (#3b)
# ────────────────────────────────────────────────────────────────────────────
#
# Sargent gives Σ Γ(τ → q q̄' ν) = (V_ud² + V_us²) × Γ(τ→eν̄ν) × N_c.  This
# captures the inclusive "high-M_qq̄" hadronic decay but misses the low-mass
# resonance dominated channels (τ → ρ ν, τ → π ν, τ → K ν, ...) which
# contribute ~11% of the total τ width.
#
# PDG 2024 (Particle Listings, τ):
#   BR(τ → π⁻ ν_τ)          = (10.82 ± 0.05)%
#   BR(τ → K⁻ ν_τ)          = ( 0.696 ± 0.010)%
#   BR(τ → π⁻ π⁰ ν_τ)       = (25.49 ± 0.09)%   ← dominated by ρ(770)
#   BR(τ → K⁻ π⁰ ν_τ)       = ( 0.433 ± 0.015)%  ← dominated by K*(892)
#   BR(τ → π⁻ K⁰ ν_τ)       = ( 0.840 ± 0.014)%
#   BR(τ → 3π ν_τ)           = ( 9.31 ± 0.05)%
#   BR(τ → eν̄ν)              = (17.82 ± 0.04)%
#   BR(τ → μν̄ν)              = (17.39 ± 0.04)%
#   BR(τ → q q̄' ν, inclusive) ≈ 64.79% (sum of all hadronic)
#
# τ total width Γ_τ = ℏ / τ_τ = 6.582e-25 / 2.903e-13 s = 2.267e-12 GeV.
# Sargent inclusive prediction Σ Γ(τ→qq̄'ν) = (0.97435² + 0.225²) × Γ_e × 3
#   ≈ 1.0007 × 4.04e-13 × 3 = 1.213e-12 GeV
#   = BR ~53.5% of Γ_τ.  The PDG inclusive BR is 64.8%, so Sargent
#   under-predicts by ~11 percentage points (the gap is the resonance
#   contribution).

# Tabulated PDG 2024 partial widths (GeV) — these are *experimental*, used to
# provide accurate Γ for resonance modes that the perturbative quark-level
# Sargent calculation can't compute from first principles (low-Q² QCD).
_TAU_RESONANCE_BR_PDG24 = {
    "tau- -> pi- nu_tau":            0.1082,
    "tau- -> K- nu_tau":             0.00696,
    "tau- -> pi- pi0 nu_tau":        0.2549,   # ρ(770) dominant
    "tau- -> K- pi0 nu_tau":         0.00433,  # K*(892)
    "tau- -> pi- K0 nu_tau":         0.00840,
    "tau- -> pi- pi+ pi- nu_tau":    0.0931,
    "tau- -> pi- pi0 pi0 nu_tau":    0.0918,
    "tau- -> K- K+ pi- nu_tau":      0.00144,
}
_TAU_TOTAL_WIDTH_GEV = 6.582119569e-25 / 2.903e-13   # ≈ 2.268e-12 GeV
_TAU_TOTAL_BR_HADRONIC = 0.6479                       # PDG 2024
_TAU_BR_LEPTONIC_EACH = 0.1782                        # PDG 2024 (eν̄ν or μν̄ν)


def tau_resonance_width(process: str) -> Optional[dict]:
    """PDG-tabulated partial width for a τ resonance/hadronic decay channel.

    Sargent can't predict ρ, K*, multi-pion resonance modes from first
    principles (low-Q² QCD non-perturbative).  This function looks up the
    PDG-measured branching ratios and returns Γ_partial = BR × Γ_τ.

    Parameters
    ----------
    process : str
        e.g. ``"tau- -> pi- nu_tau"``, ``"tau- -> pi- pi0 nu_tau"``.

    Returns
    -------
    dict with ``width_gev``, ``BR``, ``method='pdg-tabulated'``,
    ``trust_level='validated'``, or None if the channel isn't tabulated.
    """
    proc = process.strip()
    if proc not in _TAU_RESONANCE_BR_PDG24:
        return None
    br = _TAU_RESONANCE_BR_PDG24[proc]
    return {
        "process":     proc,
        "width_gev":   br * _TAU_TOTAL_WIDTH_GEV,
        "width_mev":   br * _TAU_TOTAL_WIDTH_GEV * 1000.0,
        "BR":          br,
        "method":      "pdg-tabulated",
        "kind":        "tau-resonance",
        "trust_level": "validated",
        "reference":   "PDG 2024 Particle Listings, τ section.",
        "label":       proc,
        "supported":   True,
    }


def tau_branching_summary() -> dict:
    """Inventory of all known τ decay modes and their BR contributions.

    Compares the Sargent-perturbative inclusive estimate against the PDG
    sum of resonance + multi-meson channels to expose the ~11% gap.
    """
    # Sargent inclusive prediction Γ(τ→qq̄'ν) — sum of u-d~ and u-s~ channels
    r_ud = three_body_fermi_width("tau- -> d u~ nu_tau")
    r_us = three_body_fermi_width("tau- -> s u~ nu_tau")
    sargent_inclusive_gev = 0.0
    if r_ud and r_ud.get("supported"):
        sargent_inclusive_gev += r_ud["width_gev"]
    if r_us and r_us.get("supported"):
        sargent_inclusive_gev += r_us["width_gev"]
    sargent_br = sargent_inclusive_gev / _TAU_TOTAL_WIDTH_GEV

    pdg_resonance_br = sum(_TAU_RESONANCE_BR_PDG24.values())

    return {
        "sargent_inclusive_BR":  sargent_br,
        "sargent_inclusive_width_gev": sargent_inclusive_gev,
        "pdg_tabulated_resonance_BR": pdg_resonance_br,
        "pdg_total_hadronic_BR": _TAU_TOTAL_BR_HADRONIC,
        "leptonic_BR_each":      _TAU_BR_LEPTONIC_EACH,
        "tau_total_width_gev":   _TAU_TOTAL_WIDTH_GEV,
        "note": (
            "Sargent gives the inclusive perturbative-quark estimate.  "
            "PDG resonance BRs sum to the dominant exclusive channels.  "
            "The gap between Sargent inclusive (~53.5%) and PDG total "
            "hadronic (64.79%) is the ~11% resonance enhancement at "
            "low Q² that perturbative QCD can't reach."
        ),
    }
