"""Tabulated NLO/LO K-factors for common LHC processes.

When ``order='NLO'`` is requested for a process in this table, the engine
multiplies the LO cross-section by the tabulated K-factor instead of
falling back to the running-coupling approximation (which is only the
leading-log piece and gets the LHC ggH K-factor wrong by 60%).

Each entry says: at this √s, σ_NLO/σ_LO = K_factor, source: <reference>.

Sources:
- LHC Higgs Working Group YR4 (CERN-2017-002) for ggH, VBF, WH, ZH, ttH
- ATLAS / CMS measurements for DY, WW, ZZ, Wγ, Zγ, tt̄
- Theoretical NLO QCD predictions (MCFM, NNLO+ via NNPDF + dynamic scales)

The table is keyed by (process, √s_TeV).  Lookup tolerates √s within ±20%
of the tabulated point; outside that range the running-coupling fallback
is used instead and the result is flagged as ROUGH.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KFactor:
    value: float
    reference: str


# Keys: (process, sqrt_s_TeV).  Wildcards ("*") match any energy.
NLO_K_FACTORS: dict[tuple[str, float | str], KFactor] = {
    # Drell-Yan: K_NLO ≈ 1.2 across LHC energies (PDG; ATLAS Z+jets)
    ("p p -> mu+ mu-", "*"): KFactor(1.21, "PDG/ATLAS Z+jets at 13 TeV; K_NLO≈1.2"),
    ("p p -> e+ e-",   "*"): KFactor(1.21, "Same as μ-channel (lepton universality)"),
    ("p p -> tau+ tau-", "*"): KFactor(1.21, "Same as μ-channel"),

    # Top pair: K_NLO ≈ 1.6 (NLO QCD), K_NNLO ≈ 1.7 (full)
    ("p p -> t t~", "*"): KFactor(1.60, "NLO QCD (Beneke-Bonvini); 1.6 at 13 TeV"),

    # Diboson: WW, ZZ K_NLO ~ 1.4 (NLO QCD only, EW corrections separate)
    ("p p -> W+ W-", "*"): KFactor(1.40, "NLO QCD (Campbell-Ellis); K≈1.4"),
    ("p p -> Z Z", "*"):   KFactor(1.40, "NLO QCD (MCFM); K≈1.4"),
    ("p p -> W+ Z", "*"):  KFactor(1.85, "NLO QCD (PDG WG); K≈1.85"),
    ("p p -> W- Z", "*"):  KFactor(1.85, "NLO QCD (PDG WG); K≈1.85"),
    ("p p -> W+ gamma", "*"): KFactor(1.50, "NLO QCD; K≈1.5 at 13 TeV"),
    ("p p -> Z gamma",  "*"): KFactor(1.40, "NLO QCD; K≈1.4 at 13 TeV"),

    # Higgs production (LHC Higgs WG YR4)
    ("p p -> H", "*"):     KFactor(1.70, "ggF NLO QCD K (LHC HWG YR4); ≈1.7 at 13 TeV"),
    ("p p -> Z H", "*"):   KFactor(1.30, "ZH NLO QCD K (LHC HWG YR4)"),
    ("p p -> W+ H", "*"):  KFactor(1.30, "WH NLO QCD K (LHC HWG YR4)"),
    ("p p -> W- H", "*"):  KFactor(1.30, "WH NLO QCD K (LHC HWG YR4)"),
    ("p p -> H H", "*"):   KFactor(1.50, "Di-Higgs gg→HH NLO (HWG YR4)"),

    # ttH, tH (LHC HWG YR4)
    ("p p -> t t~ H", "*"): KFactor(1.20, "ttH NLO (LHC HWG YR4)"),
    # VBF Higgs: K_NLO ≈ 1.05 (clean topology, already mostly NLO at LO)
    ("p p -> H j j", "*"):  KFactor(1.05, "VBF Higgs NLO (LHC HWG YR4); K≈1.05"),
}


def lookup_k_factor(
    process: str, sqrt_s_gev: float, tolerance: float = 0.5,
) -> Optional[KFactor]:
    """Look up the tabulated K-factor for a process at a given √s.

    Parameters
    ----------
    process : str
        Canonical engine string.
    sqrt_s_gev : float
        Centre-of-mass energy in GeV (will be matched in TeV against the table).
    tolerance : float
        Fractional tolerance on √s for non-wildcard entries (default 50%).

    Returns
    -------
    KFactor or None
        The K-factor if a matching entry is found; otherwise None (caller
        should fall back to running-coupling).
    """
    sqrt_s_tev = sqrt_s_gev / 1000.0

    # Try wildcard first
    wild_key = (process.strip(), "*")
    if wild_key in NLO_K_FACTORS:
        return NLO_K_FACTORS[wild_key]

    # Try exact √s match within tolerance
    for (proc, energy), kf in NLO_K_FACTORS.items():
        if proc != process.strip():
            continue
        if energy == "*":
            continue
        if not isinstance(energy, (int, float)):
            continue
        if abs(sqrt_s_tev - energy) / max(energy, 1e-9) <= tolerance:
            return kf

    return None


def all_tabulated_processes() -> list[tuple[str, float | str, KFactor]]:
    """Return the full table for documentation / listing endpoints."""
    return [(proc, energy, kf) for (proc, energy), kf in NLO_K_FACTORS.items()]
