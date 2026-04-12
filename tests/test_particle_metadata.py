"""Tests for particle-package-backed metadata on theory particles."""

from feynman_engine.physics.registry import TheoryRegistry


def test_qed_particle_has_pdg_metadata():
    electron = TheoryRegistry.get_particles("QED")["e-"]

    assert electron.pdg_name == "e"
    assert electron.latex_name == r"e^{-}"
    assert electron.mass_mev is not None
    assert electron.mass_mev > 0


def test_ew_boson_has_numeric_mass_metadata():
    z_boson = TheoryRegistry.get_particles("EW")["Z"]

    assert z_boson.pdg_name == "Z"
    assert z_boson.latex_name == r"Z^{0}"
    assert z_boson.mass_mev is not None
    assert z_boson.mass_mev > 90000
