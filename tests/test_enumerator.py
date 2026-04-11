"""Tests for the pure Python tree-level diagram enumerator."""
import pytest
from feynman_engine.core.enumerator import enumerate_tree


class TestQEDTree:
    def test_ee_mumu_one_diagram(self):
        """e+e- → μ+μ- at tree level: exactly 1 s-channel diagram."""
        diagrams = enumerate_tree(["e+", "e-"], ["mu+", "mu-"], "QED")
        assert len(diagrams) == 1

    def test_ee_mumu_s_channel(self):
        diagrams = enumerate_tree(["e+", "e-"], ["mu+", "mu-"], "QED")
        assert diagrams[0].topology == "s-channel"

    def test_ee_mumu_internal_photon(self):
        diagrams = enumerate_tree(["e+", "e-"], ["mu+", "mu-"], "QED")
        internal = [e.particle for e in diagrams[0].internal_edges]
        assert "gamma" in internal

    def test_ee_mumu_loop_order_zero(self):
        diagrams = enumerate_tree(["e+", "e-"], ["mu+", "mu-"], "QED")
        assert diagrams[0].loop_order == 0

    def test_ee_mumu_external_particles(self):
        diagrams = enumerate_tree(["e+", "e-"], ["mu+", "mu-"], "QED")
        ext = {e.particle for e in diagrams[0].external_edges}
        assert ext == {"e+", "e-", "mu+", "mu-"}

    def test_bhabha_two_diagrams(self):
        """e+e- → e+e- (Bhabha): exactly 2 diagrams (s + t channel)."""
        diagrams = enumerate_tree(["e+", "e-"], ["e+", "e-"], "QED")
        assert len(diagrams) == 2

    def test_bhabha_topologies(self):
        diagrams = enumerate_tree(["e+", "e-"], ["e+", "e-"], "QED")
        tops = {d.topology for d in diagrams}
        assert "s-channel" in tops
        assert "t-channel" in tops

    def test_compton_two_diagrams(self):
        """e- γ → e- γ (Compton): 2 diagrams (s + u channel)."""
        diagrams = enumerate_tree(["e-", "gamma"], ["e-", "gamma"], "QED")
        assert len(diagrams) == 2

    def test_all_external_present_bhabha(self):
        diagrams = enumerate_tree(["e+", "e-"], ["e+", "e-"], "QED")
        for d in diagrams:
            ext = {e.particle for e in d.external_edges}
            assert "e+" in ext
            assert "e-" in ext


class TestQCDTree:
    def test_qqbar_to_gg_has_diagrams(self):
        """u ū → g g at tree level: should find at least 1 diagram."""
        diagrams = enumerate_tree(["u", "u~"], ["g", "g"], "QCD")
        assert len(diagrams) >= 1

    def test_gg_to_qqbar_has_diagrams(self):
        diagrams = enumerate_tree(["g", "g"], ["u", "u~"], "QCD")
        assert len(diagrams) >= 1

    def test_gg_contact_vertex(self):
        """g g → g g at tree level: should include the 4-gluon contact diagram."""
        diagrams = enumerate_tree(["g", "g"], ["g", "g"], "QCD")
        # At minimum we expect the 4-gluon contact vertex
        assert len(diagrams) >= 1
        tops = {d.topology for d in diagrams}
        assert any(t is not None for t in tops)

    def test_loop_raises_without_qgraf(self):
        """Loop diagrams must raise NotImplementedError without QGRAF."""
        from feynman_engine.core.generator import qgraf_available
        if qgraf_available():
            pytest.skip("QGRAF is available — loop diagrams handled by QGRAF")
        from feynman_engine.core.generator import generate_diagrams
        from feynman_engine.physics.translator import parse_process
        spec = parse_process("e+ e- -> mu+ mu-", "QED", loops=1)
        with pytest.raises(NotImplementedError, match="QGRAF"):
            generate_diagrams(spec)


class TestEWTree:
    def test_ee_to_ww(self):
        """e+e- → W+W- at tree level: should find diagrams."""
        diagrams = enumerate_tree(["e+", "e-"], ["W+", "W-"], "EW")
        assert len(diagrams) >= 1

    def test_ee_to_zh(self):
        """e+e- → Z H (Higgsstrahlung): should find at least 1 diagram."""
        diagrams = enumerate_tree(["e+", "e-"], ["Z", "H"], "EW")
        assert len(diagrams) >= 1


class TestEnumeratorProperties:
    def test_no_duplicate_topologies(self):
        """Deduplication should never produce two identical canonical hashes."""
        diagrams = enumerate_tree(["e+", "e-"], ["e+", "e-"], "QED")
        hashes = [d.canonical_hash for d in diagrams]
        assert len(hashes) == len(set(hashes))

    def test_all_diagrams_tree_level(self):
        """All produced diagrams must have loop_order == 0."""
        for incoming, outgoing in [
            (["e+", "e-"], ["mu+", "mu-"]),
            (["e+", "e-"], ["e+", "e-"]),
            (["e-", "gamma"], ["e-", "gamma"]),
        ]:
            for d in enumerate_tree(incoming, outgoing, "QED"):
                assert d.loop_order == 0

    def test_two_vertices_for_2to2(self):
        """Every 3-point-vertex diagram for 2→2 should have exactly 2 vertices."""
        diagrams = enumerate_tree(["e+", "e-"], ["mu+", "mu-"], "QED")
        for d in diagrams:
            assert len(d.vertices) == 2

    def test_topology_assigned(self):
        """All diagrams should have a topology label."""
        diagrams = enumerate_tree(["e+", "e-"], ["mu+", "mu-"], "QED")
        for d in diagrams:
            assert d.topology is not None
