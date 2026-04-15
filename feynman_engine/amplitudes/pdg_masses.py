"""Shared PDG particle mass lookup.

Provides a single authoritative mass table queried from the ``particle``
package at import time, with fallback entries for BSM particles that have
no PDG ID.
"""
from __future__ import annotations

from particle import Particle as PDGParticle


# BSM masses that have no PDG entry — these are model parameters.
_BSM_MASS_GEV: dict[str, float] = {
    "m_Zp":  1000.0,
    "m_chi": 100.0,
}


def _build_pdg_mass_table() -> dict[str, float]:
    """Build a symbol-name → mass (GeV) mapping from PDG data."""
    # Map of (symbol name, PDG ID) for particles we care about.
    _PDG_ENTRIES: list[tuple[str, int]] = [
        ("m_e",   11),
        ("m_mu",  13),
        ("m_tau", 15),
        ("m_u",   2),
        ("m_d",   1),
        ("m_s",   3),
        ("m_c",   4),
        ("m_b",   5),
        ("m_t",   6),
        ("m_W",   24),
        ("m_Z",   23),
        ("m_H",   25),
    ]
    table: dict[str, float] = {}
    for symbol_name, pdg_id in _PDG_ENTRIES:
        try:
            p = PDGParticle.from_pdgid(pdg_id)
            mass_mev = p.mass  # in MeV
            if mass_mev is not None:
                table[symbol_name] = float(mass_mev) / 1000.0  # MeV → GeV
            else:
                table[symbol_name] = 0.0
        except Exception:
            table[symbol_name] = 0.0
    # Add BSM entries
    table.update(_BSM_MASS_GEV)
    return table


MASS_GEV: dict[str, float] = _build_pdg_mass_table()
"""Particle masses in GeV, keyed by symbolic mass name (e.g. ``"m_e"``).

Standard Model values come from the ``particle`` package (PDG data).
BSM values (Z', chi) are model parameters.
"""
