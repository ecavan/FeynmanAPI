"""Tests for cs_dipoles_ol.py — generic photon dipole enumerator.

Validates the QED dipole enumeration logic against expected counts and
charge-correlator structure for several known processes.

The actual Monte-Carlo subtraction (``evaluate_qed_dipole_sum`` integrated
over RAMBO PSP) is INFRASTRUCTURE-ONLY at this point — see the warning in
``ew_real_kfactor_openloops`` docstring for the validation gap.
"""
from __future__ import annotations

import pytest

from feynman_engine.amplitudes.cs_dipoles_ol import (
    enumerate_qed_dipoles,
    DipoleAssignment,
)


# ─── Enumerator: e+ e- → μ+ μ- γ ───────────────────────────────────────────

def test_enumerate_eemumu_gamma_count():
    """e+ e- → μ+ μ- γ has 4 charged legs × 3 spectators each = 12 dipoles."""
    dipoles = enumerate_qed_dipoles(
        incoming=["e+", "e-"],
        outgoing_with_photon=["mu+", "mu-", "gamma"],
    )
    assert len(dipoles) == 12, (
        f"Expected 12 dipoles for e+e-→μμγ, got {len(dipoles)}"
    )


def test_enumerate_eemumu_gamma_configs():
    """All 4 dipole configs (FF, FI, IF, II) appear with correct counts."""
    dipoles = enumerate_qed_dipoles(
        incoming=["e+", "e-"],
        outgoing_with_photon=["mu+", "mu-", "gamma"],
    )
    counts = {"FF": 0, "FI": 0, "IF": 0, "II": 0}
    for d in dipoles:
        counts[d.config] += 1
    # 2 FF (μ+ ↔ μ- both directions), 2 II (e+ ↔ e- both),
    # 4 IF (each initial × 2 finals), 4 FI (each final × 2 initials)
    assert counts["FF"] == 2, f"Expected 2 FF, got {counts['FF']}"
    assert counts["II"] == 2, f"Expected 2 II, got {counts['II']}"
    assert counts["IF"] == 4, f"Expected 4 IF, got {counts['IF']}"
    assert counts["FI"] == 4, f"Expected 4 FI, got {counts['FI']}"


def test_enumerate_charge_correlator_structure():
    """Σ_{i≠k} Q_i Q_k = (Σ Q)² - Σ Q² for 4 charged legs.

    For 4 unit-charge legs (e+, e-, μ+, μ-): Σ Q = 0, Σ Q² = 4,
    so the eikonal correlator sum is 0 - 4 = -4.
    """
    dipoles = enumerate_qed_dipoles(
        incoming=["e+", "e-"],
        outgoing_with_photon=["mu+", "mu-", "gamma"],
    )
    total_correlator = sum(d.charge_correlator for d in dipoles)
    expected = -4.0  # (Σ Q)² - Σ Q² = 0 - 4 = -4
    assert abs(total_correlator - expected) < 1e-10, (
        f"Σ Q_i Q_k = {total_correlator}, expected {expected}"
    )


# ─── Enumerator: photon-only outgoing ──────────────────────────────────────

def test_enumerate_no_photon_returns_empty():
    """A non-radiative process returns no dipoles."""
    dipoles = enumerate_qed_dipoles(
        incoming=["e+", "e-"],
        outgoing_with_photon=["mu+", "mu-"],  # no photon
    )
    assert dipoles == []


def test_enumerate_neutral_only_returns_empty():
    """All-neutral particles → no charged dipoles."""
    dipoles = enumerate_qed_dipoles(
        incoming=["nu_e", "nu_e"],
        outgoing_with_photon=["Z", "gamma"],
    )
    assert dipoles == []


# ─── Enumerator: pp → ll γ (charged-quark → lepton-pair) ───────────────────

def test_enumerate_uubar_emumu_gamma_count():
    """u u~ → e+ e- γ: 4 charged legs (u, u~, e+, e-) → 12 dipoles."""
    dipoles = enumerate_qed_dipoles(
        incoming=["u", "u~"],
        outgoing_with_photon=["e+", "e-", "gamma"],
    )
    assert len(dipoles) == 12


def test_enumerate_uubar_emumu_gamma_quark_charges():
    """Up-quark charge correlator: Q_u = +2/3, |Q|² = 4/9 ≠ 1."""
    dipoles = enumerate_qed_dipoles(
        incoming=["u", "u~"],
        outgoing_with_photon=["e+", "e-", "gamma"],
    )
    # Find a (u, u~) dipole — should have Q_u × Q_u~ = (+2/3)(-2/3) = -4/9
    uubar_dipoles = [d for d in dipoles if d.emitter_idx == 0 and d.spectator_idx == 1]
    assert len(uubar_dipoles) == 1
    assert abs(uubar_dipoles[0].charge_correlator - (-4.0/9)) < 1e-10


# ─── Auto-detection of photon ──────────────────────────────────────────────

def test_enumerate_explicit_photon_idx():
    """Explicit photon index works."""
    dipoles_auto = enumerate_qed_dipoles(
        incoming=["e+", "e-"],
        outgoing_with_photon=["gamma", "mu+", "mu-"],  # photon FIRST
    )
    dipoles_explicit = enumerate_qed_dipoles(
        incoming=["e+", "e-"],
        outgoing_with_photon=["gamma", "mu+", "mu-"],
        photon_idx_in_outgoing=0,
    )
    assert len(dipoles_auto) == len(dipoles_explicit) == 12
    # auto detection should pick the first 'gamma'
    for da, de in zip(dipoles_auto, dipoles_explicit):
        assert da.photon_idx == de.photon_idx


# ─── Symmetry: same enumeration regardless of charge sign ──────────────────

def test_enumerate_charge_pairs_symmetric():
    """e+ e- → μ+ μ- γ and e- e+ → μ- μ+ γ should give the same dipole count."""
    d1 = enumerate_qed_dipoles(["e+", "e-"], ["mu+", "mu-", "gamma"])
    d2 = enumerate_qed_dipoles(["e-", "e+"], ["mu-", "mu+", "gamma"])
    assert len(d1) == len(d2) == 12
    # Charge correlator sum is invariant under charge swap (Q_i Q_k unchanged)
    assert sum(d.charge_correlator for d in d1) == sum(d.charge_correlator for d in d2)


# ─── DipoleAssignment dataclass ────────────────────────────────────────────

def test_dipole_assignment_charge_correlator_property():
    """DipoleAssignment.charge_correlator returns Q_emitter × Q_spectator."""
    d = DipoleAssignment(
        emitter_idx=0, photon_idx=4, spectator_idx=1,
        config="II",
        Q_emitter=1.0, Q_spectator=-1.0,
        is_emitter_initial=True, is_spectator_initial=True,
    )
    assert d.charge_correlator == -1.0

    d2 = DipoleAssignment(
        emitter_idx=0, photon_idx=4, spectator_idx=2,
        config="IF",
        Q_emitter=2.0/3, Q_spectator=2.0/3,
        is_emitter_initial=True, is_spectator_initial=False,
    )
    assert abs(d2.charge_correlator - 4.0/9) < 1e-10
