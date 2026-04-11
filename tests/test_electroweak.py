"""Tests for Electroweak (SM) theory definition and registry integration."""
import pytest

from feynman_engine.physics.registry import TheoryRegistry
from feynman_engine.physics.translator import parse_process
from feynman_engine.core.models import ParticleType, PropagatorStyle


class TestEWParticles:
    def test_all_generations_leptons(self):
        particles = TheoryRegistry.get_particles("EW")
        for name in ["e-", "e+", "mu-", "mu+", "tau-", "tau+"]:
            assert name in particles

    def test_all_generations_neutrinos(self):
        particles = TheoryRegistry.get_particles("EW")
        for name in ["nu_e", "nu_mu", "nu_tau"]:
            assert name in particles

    def test_all_quark_flavors(self):
        particles = TheoryRegistry.get_particles("EW")
        for name in ["u", "d", "s", "c", "b", "t", "u~", "d~", "s~", "c~", "b~", "t~"]:
            assert name in particles

    def test_gauge_bosons_present(self):
        particles = TheoryRegistry.get_particles("EW")
        for name in ["gamma", "Z", "W+", "W-"]:
            assert name in particles

    def test_higgs_present(self):
        particles = TheoryRegistry.get_particles("EW")
        assert "H" in particles
        h = particles["H"]
        assert h.particle_type == ParticleType.SCALAR
        assert h.propagator_style == PropagatorStyle.SCALAR
        assert h.mass == "m_H"

    def test_photon_massless(self):
        assert TheoryRegistry.get_particles("EW")["gamma"].mass == "0"

    def test_w_charged(self):
        particles = TheoryRegistry.get_particles("EW")
        assert particles["W+"].charge == +1.0
        assert particles["W-"].charge == -1.0
        assert particles["W+"].antiparticle == "W-"

    def test_z_neutral(self):
        assert TheoryRegistry.get_particles("EW")["Z"].charge == 0.0

    def test_w_propagator_style(self):
        particles = TheoryRegistry.get_particles("EW")
        assert particles["W+"].propagator_style == PropagatorStyle.CHARGED_BOSON


class TestEWVertices:
    def test_photon_couples_to_electrons(self):
        vertices = TheoryRegistry.get_theory("EW")["vertices"]
        found = any(set(v) == {"e-", "e+", "gamma"} for v in vertices)
        assert found

    def test_z_couples_to_neutrinos(self):
        vertices = TheoryRegistry.get_theory("EW")["vertices"]
        found = any(set(v) == {"nu_e", "nu_e~", "Z"} for v in vertices)
        assert found

    def test_higgs_ww_coupling(self):
        vertices = TheoryRegistry.get_theory("EW")["vertices"]
        found = any(set(v) == {"H", "W+", "W-"} for v in vertices)
        assert found

    def test_higgs_zzh_coupling(self):
        vertices = TheoryRegistry.get_theory("EW")["vertices"]
        found = any(set(v) == {"H", "Z", "Z"} for v in vertices)
        assert found

    def test_wwz_triple_gauge(self):
        vertices = TheoryRegistry.get_theory("EW")["vertices"]
        found = any(set(v) == {"W+", "W-", "Z"} for v in vertices)
        assert found

    def test_wwgamma_triple_gauge(self):
        vertices = TheoryRegistry.get_theory("EW")["vertices"]
        found = any(set(v) == {"W+", "W-", "gamma"} for v in vertices)
        assert found

    def test_higgs_self_coupling(self):
        vertices = TheoryRegistry.get_theory("EW")["vertices"]
        found_3 = any(v.count("H") == 3 for v in vertices if len(v) == 3)
        found_4 = any(v.count("H") == 4 for v in vertices if len(v) == 4)
        assert found_3, "Missing HHH vertex"
        assert found_4, "Missing HHHH vertex"

    def test_top_yukawa(self):
        vertices = TheoryRegistry.get_theory("EW")["vertices"]
        found = any(set(v) == {"t", "t~", "H"} for v in vertices)
        assert found


class TestEWQGRAFNames:
    def test_photon_qgraf_name(self):
        assert TheoryRegistry.to_qgraf_name("EW", "gamma") == "A"

    def test_w_plus_qgraf_name(self):
        assert TheoryRegistry.to_qgraf_name("EW", "W+") == "Wp"

    def test_higgs_qgraf_name(self):
        assert TheoryRegistry.to_qgraf_name("EW", "H") == "H"

    def test_reverse_tau(self):
        assert TheoryRegistry.from_qgraf_name("EW", "taum") == "tau-"


class TestEWTranslator:
    def test_ee_to_ww(self):
        spec = parse_process("e+ e- -> W+ W-", theory="EW")
        assert spec.incoming == ["e+", "e-"]
        assert spec.outgoing == ["W+", "W-"]

    def test_ee_to_zh(self):
        spec = parse_process("e+ e- -> Z H", theory="EW")
        assert spec.outgoing == ["Z", "H"]

    def test_alias_higgs(self):
        spec = parse_process("e+ e- -> Z higgs", theory="EW")
        assert "H" in spec.outgoing

    def test_alias_w_lowercase(self):
        spec = parse_process("e+ e- -> w+ w-", theory="EW")
        assert spec.outgoing == ["W+", "W-"]
