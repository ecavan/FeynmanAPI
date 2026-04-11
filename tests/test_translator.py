"""Tests for the process string translator."""
import pytest

from feynman_engine.physics.translator import parse_process


class TestParseProcess:
    def test_standard_arrow(self):
        spec = parse_process("e+ e- -> mu+ mu-", theory="QED")
        assert spec.incoming == ["e+", "e-"]
        assert spec.outgoing == ["mu+", "mu-"]

    def test_unicode_arrow(self):
        spec = parse_process("e+ e- → mu+ mu-", theory="QED")
        assert spec.incoming == ["e+", "e-"]
        assert spec.outgoing == ["mu+", "mu-"]

    def test_compton_scattering(self):
        spec = parse_process("e- gamma -> e- gamma", theory="QED")
        assert spec.incoming == ["e-", "gamma"]
        assert spec.outgoing == ["e-", "gamma"]

    def test_bhabha(self):
        spec = parse_process("e+ e- -> e+ e-", theory="QED")
        assert spec.incoming == ["e+", "e-"]
        assert spec.outgoing == ["e+", "e-"]

    def test_loop_order_stored(self):
        spec = parse_process("e+ e- -> mu+ mu-", theory="QED", loops=1)
        assert spec.loops == 1

    def test_theory_normalized_to_uppercase(self):
        spec = parse_process("e+ e- -> mu+ mu-", theory="qed")
        assert spec.theory == "QED"

    def test_unknown_particle_raises(self):
        with pytest.raises(ValueError, match="Unknown particle"):
            parse_process("e+ e- -> quark gluon", theory="QED")

    def test_missing_arrow_raises(self):
        with pytest.raises(ValueError, match="Cannot find arrow"):
            parse_process("e+ e- mu+ mu-", theory="QED")

    def test_qgraf_name_mapping(self):
        spec = parse_process("e+ e- -> mu+ mu-", theory="QED")
        assert spec.qgraf_incoming == ["ep", "em"]
        assert spec.qgraf_outgoing == ["mup", "mum"]
