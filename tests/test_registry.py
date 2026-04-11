"""Tests for the theory registry, including runtime BSM registration."""
import pytest

from feynman_engine.physics.registry import TheoryRegistry
from feynman_engine.core.models import Particle, ParticleType, PropagatorStyle


class TestBuiltinTheories:
    def test_all_builtin_theories_present(self):
        theories = TheoryRegistry.list_theories()
        assert "QED" in theories
        assert "QCD" in theories
        assert "EW" in theories

    def test_unknown_theory_raises(self):
        with pytest.raises(ValueError, match="Unknown theory"):
            TheoryRegistry.get_theory("NONEXISTENT")

    def test_theory_name_case_insensitive(self):
        qed1 = TheoryRegistry.get_theory("QED")
        qed2 = TheoryRegistry.get_theory("qed")
        assert qed1 is qed2

    def test_builtin_cannot_be_unregistered(self):
        with pytest.raises(ValueError, match="Cannot unregister"):
            TheoryRegistry.unregister("QED")


class TestCustomTheoryRegistration:
    """Test runtime registration of custom (BSM) theories."""

    _MOCK_THEORY = {
        "particles": {
            "X": Particle(
                name="X", pdg_id=9999,
                particle_type=ParticleType.SCALAR,
                mass="m_X", charge=0.0, color="1",
                propagator_style=PropagatorStyle.SCALAR,
            ),
            "X~": Particle(
                name="X~", pdg_id=-9999,
                particle_type=ParticleType.SCALAR,
                mass="m_X", charge=0.0, color="1",
                propagator_style=PropagatorStyle.SCALAR,
                antiparticle="X",
            ),
        },
        "vertices": [("X", "X~", "gamma")],
        "model_file": "bsm_scalar.mod",
        "qgraf_name_map": {"X": "Xp", "X~": "Xa"},
        "qgraf_name_reverse": {"Xp": "X", "Xa": "X~"},
    }

    def test_register_custom_theory(self):
        TheoryRegistry.register("BSM_SCALAR", self._MOCK_THEORY)
        assert "BSM_SCALAR" in TheoryRegistry.list_theories()

    def test_registered_theory_particles_accessible(self):
        TheoryRegistry.register("BSM_SCALAR2", self._MOCK_THEORY)
        particles = TheoryRegistry.get_particles("BSM_SCALAR2")
        assert "X" in particles

    def test_register_missing_key_raises(self):
        bad_theory = {"particles": {}, "vertices": []}
        with pytest.raises(ValueError, match="missing keys"):
            TheoryRegistry.register("BAD", bad_theory)

    def test_unregister_custom_theory(self):
        TheoryRegistry.register("TEMP_THEORY", self._MOCK_THEORY)
        assert "TEMP_THEORY" in TheoryRegistry.list_theories()
        TheoryRegistry.unregister("TEMP_THEORY")
        assert "TEMP_THEORY" not in TheoryRegistry.list_theories()

    def test_unregister_nonexistent_is_noop(self):
        """Unregistering a non-existent custom theory should not raise."""
        TheoryRegistry.unregister("DOES_NOT_EXIST")  # should not raise
