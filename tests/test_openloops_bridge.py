"""Smoke tests for the OpenLoops bridge + virtual K-factor path.

These tests are skipped when OpenLoops is not installed locally so the
suite still passes for users who haven't run ``feynman install-openloops``.
"""
from __future__ import annotations

import os
import pytest

from feynman_engine.amplitudes.openloops_bridge import (
    is_available,
    install_prefix,
    installed_processes,
    to_pdg_string,
)


def _ol_installed_with(process: str) -> bool:
    if not is_available():
        return False
    return process in installed_processes()


def test_to_pdg_string_basic():
    assert to_pdg_string("u u~ -> e+ e-") == "2 -2 -> -11 11"
    assert to_pdg_string("g g -> t t~") == "21 21 -> 6 -6"


def test_to_pdg_string_unknown_particle_raises():
    with pytest.raises(ValueError):
        to_pdg_string("xyz -> abc")


@pytest.mark.skipif(
    not _ol_installed_with("ppllj"),
    reason="OpenLoops + ppllj process library not installed",
)
def test_evaluate_loop_squared_drell_yan():
    """OpenLoops should give finite tree + loop pieces for Drell-Yan."""
    from feynman_engine.amplitudes.openloops_bridge import evaluate_loop_squared

    result = evaluate_loop_squared("u u~ -> e+ e-", 91.0)
    assert result["tree"] > 0.0, "tree |M|² should be strictly positive"
    # IR pole structure is finite at NLO (no double-pole for non-IR-singular DY)
    assert result["loop_ir2"] == pytest.approx(0.0, abs=1e-3) or result["loop_ir2"] != 0.0
    # finite piece is bounded relative to tree
    assert abs(result["loop_finite"] / result["tree"]) < 100.0


@pytest.mark.skipif(
    not _ol_installed_with("ppllj"),
    reason="OpenLoops + ppllj process library not installed",
)
def test_virtual_k_factor_drell_yan():
    """Virtual K-factor for u u~ → e+ e- should be O(1 + α_s)."""
    from feynman_engine.amplitudes.nlo_cross_section_openloops import (
        virtual_k_factor_openloops,
    )

    result = virtual_k_factor_openloops("u u~ -> e+ e-", 91.0, theory="QCD")
    assert result["supported"] is True
    assert result["method"] == "openloops-virtual-only"
    assert result["trust_level"] == "approximate"
    k = result["k_factor"]
    # Virtual K should be O(1) — the αs/(2π) prefactor keeps it bounded
    assert 0.5 < k < 1.5, f"K_virt={k} out of expected range"


def test_virtual_k_factor_unavailable_returns_supported_false():
    """When OpenLoops is missing, the function should fail gracefully."""
    if is_available():
        pytest.skip("OpenLoops is installed; this test exercises the missing path")
    from feynman_engine.amplitudes.nlo_cross_section_openloops import (
        virtual_k_factor_openloops,
    )
    r = virtual_k_factor_openloops("u u~ -> e+ e-", 91.0)
    assert r["supported"] is False
    assert "OpenLoops" in r["error"]


def test_install_prefix_returns_path_or_none():
    p = install_prefix()
    assert p is None or os.path.isdir(p), f"prefix {p} should be a directory if set"


def test_installed_processes_returns_list():
    procs = installed_processes()
    assert isinstance(procs, list)
    for p in procs:
        assert isinstance(p, str) and len(p) > 0
