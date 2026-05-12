"""Validation for the OpenLoops process catalog and pack definitions."""
from __future__ import annotations

import pytest

from feynman_engine.resources.openloops import (
    PACKS, estimate_install, libraries_for_process, library_meta, load_catalog,
    pack_summary, resolve_pack,
)


def test_catalog_loads_and_has_libraries():
    cat = load_catalog()
    assert cat["libraries"], "catalog has no libraries — generator never ran?"
    assert cat["process_index"], "catalog has no process index"
    assert len(cat["libraries"]) >= 100, (
        f"expected ≥100 OL libraries, got {len(cat['libraries'])}"
    )


@pytest.mark.parametrize("process,expected_lib", [
    # Each tuple: process query + at least one library that MUST cover it.
    ("g g -> t t~",        "pptt"),
    ("e+ e- -> mu+ mu-",   "eell_ew"),
    ("e+ e- -> t t~",      "eett_ew"),
    ("e+ e- -> Z Z",       "eevv_ew"),
    ("u u~ -> Z Z",        "ppvv"),
])
def test_catalog_lookup_known_processes(process, expected_lib):
    libs = libraries_for_process(process)
    assert expected_lib in libs, (
        f"expected catalog to map {process!r} to {expected_lib!r}; "
        f"got {libs}"
    )


def test_hadronic_lookup_expands_proton():
    """`p p -> X` should resolve to libraries via partonic initiators."""
    libs = libraries_for_process("p p -> t t~")
    assert "pptt" in libs


@pytest.mark.parametrize("pack_name", [
    name for name, pack in PACKS.items() if pack.get("libraries") is not None
])
def test_explicit_packs_only_reference_real_libraries(pack_name):
    """Every library in a hardcoded pack list must exist in the catalog."""
    libs = PACKS[pack_name]["libraries"]
    for lib in libs:
        assert library_meta(lib) is not None, (
            f"pack {pack_name!r} references non-existent library {lib!r}.  "
            "Either rename the library or remove it from the pack."
        )


@pytest.mark.parametrize("pack_name", list(PACKS))
def test_pack_summaries_resolve(pack_name):
    """Every pack should resolve to a non-empty library list (or be 'minimal')."""
    summary = pack_summary(pack_name)
    libs = summary["resolved_libraries"]
    if pack_name == "minimal":
        return
    assert len(libs) > 0, f"pack {pack_name!r} resolves to empty library list"
    est = summary["estimate"]
    assert est["disk_mb"] > 0
    assert est["build_min"] > 0


def test_estimate_install_scales_with_count():
    """Install estimate should grow monotonically with library count."""
    one = estimate_install(["pptt"])
    five = estimate_install(["pptt", "ppllj", "eell_ew", "eett_ew", "eevv_ew"])
    assert five["disk_mb"] > one["disk_mb"]
    assert five["build_min"] > one["build_min"]
