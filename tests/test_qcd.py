"""Tests for QCD theory definition and registry integration."""
import pytest

from feynman_engine.physics.registry import TheoryRegistry
from feynman_engine.physics.translator import parse_process
from feynman_engine.core.models import ParticleType, PropagatorStyle


class TestQCDParticles:
    def test_all_quarks_present(self):
        particles = TheoryRegistry.get_particles("QCD")
        for flavor in ["u", "d", "s", "c", "b", "t"]:
            assert flavor in particles, f"Missing quark: {flavor}"
            assert f"{flavor}~" in particles, f"Missing antiquark: {flavor}~"

    def test_gluon_present(self):
        particles = TheoryRegistry.get_particles("QCD")
        assert "g" in particles
        g = particles["g"]
        assert g.propagator_style == PropagatorStyle.GLUON
        assert g.color == "8"
        assert g.mass == "0"
        assert g.antiparticle is None  # self-conjugate

    def test_ghosts_present(self):
        particles = TheoryRegistry.get_particles("QCD")
        assert "gh" in particles
        assert "gh~" in particles
        assert particles["gh"].particle_type == ParticleType.GHOST

    def test_quark_types(self):
        particles = TheoryRegistry.get_particles("QCD")
        assert particles["u"].particle_type == ParticleType.FERMION
        assert particles["u~"].particle_type == ParticleType.ANTIFERMION

    def test_quark_color(self):
        particles = TheoryRegistry.get_particles("QCD")
        assert particles["t"].color == "3"
        assert particles["t~"].color == "3bar"

    def test_top_mass_symbolic(self):
        particles = TheoryRegistry.get_particles("QCD")
        assert particles["t"].mass == "m_t"

    def test_up_quark_charge(self):
        particles = TheoryRegistry.get_particles("QCD")
        assert abs(particles["u"].charge - 2/3) < 1e-10

    def test_down_quark_charge(self):
        particles = TheoryRegistry.get_particles("QCD")
        assert abs(particles["d"].charge - (-1/3)) < 1e-10


class TestQCDVertices:
    def test_qqg_vertices_all_flavors(self):
        """Each quark flavor must have a qqg vertex."""
        theory = TheoryRegistry.get_theory("QCD")
        vertices = theory["vertices"]
        for flavor in ["u", "d", "s", "c", "b", "t"]:
            found = any(
                set(v) == {flavor, f"{flavor}~", "g"}
                for v in vertices
                if len(v) == 3
            )
            assert found, f"Missing qqg vertex for {flavor}"

    def test_3gluon_vertex(self):
        theory = TheoryRegistry.get_theory("QCD")
        vertices = theory["vertices"]
        found = any(set(v) == {"g"} and len(v) == 3 for v in vertices)
        assert found, "Missing 3-gluon vertex"

    def test_4gluon_vertex(self):
        theory = TheoryRegistry.get_theory("QCD")
        vertices = theory["vertices"]
        found = any(set(v) == {"g"} and len(v) == 4 for v in vertices)
        assert found, "Missing 4-gluon vertex"

    def test_ghost_gluon_vertex(self):
        theory = TheoryRegistry.get_theory("QCD")
        vertices = theory["vertices"]
        found = any(set(v) == {"gh", "gh~", "g"} for v in vertices)
        assert found, "Missing ghost-gluon vertex"


class TestQCDQGRAFNames:
    def test_gluon_qgraf_name(self):
        assert TheoryRegistry.to_qgraf_name("QCD", "g") == "G"

    def test_quark_qgraf_name(self):
        assert TheoryRegistry.to_qgraf_name("QCD", "u") == "uq"
        assert TheoryRegistry.to_qgraf_name("QCD", "u~") == "ua"

    def test_reverse_mapping(self):
        assert TheoryRegistry.from_qgraf_name("QCD", "G") == "g"
        assert TheoryRegistry.from_qgraf_name("QCD", "tq") == "t"
        assert TheoryRegistry.from_qgraf_name("QCD", "ta") == "t~"


class TestQCDTranslator:
    def test_gg_to_qq(self):
        spec = parse_process("g g -> u u~", theory="QCD")
        assert spec.incoming == ["g", "g"]
        assert spec.outgoing == ["u", "u~"]

    def test_qgraf_names_for_gg(self):
        spec = parse_process("g g -> u u~", theory="QCD")
        assert spec.qgraf_incoming == ["G", "G"]

    def test_unknown_particle_in_qcd(self):
        with pytest.raises(ValueError, match="Unknown particle"):
            parse_process("e+ e- -> mu+ mu-", theory="QCD")
