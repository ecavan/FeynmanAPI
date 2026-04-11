"""Tests for diagram deduplication via graph isomorphism."""
import pytest

from feynman_engine.core.models import Diagram, Edge, Vertex
from feynman_engine.core.normalize import deduplicate


def _make_schannel(diagram_id: int, theory: str = "QED") -> Diagram:
    """Create a minimal s-channel e+e- → mu+mu- diagram."""
    return Diagram(
        id=diagram_id,
        theory=theory,
        process="e+ e- -> mu+ mu-",
        loop_order=0,
        vertices=[
            Vertex(id=0, particles=["e-", "e+", "gamma"]),
            Vertex(id=1, particles=["mu-", "mu+", "gamma"]),
        ],
        edges=[
            Edge(id=0, start_vertex=0, end_vertex=1, particle="gamma", is_external=False),
            Edge(id=1, start_vertex=0, end_vertex=-1, particle="e+", is_external=True),
            Edge(id=2, start_vertex=0, end_vertex=-2, particle="e-", is_external=True),
            Edge(id=3, start_vertex=1, end_vertex=-3, particle="mu+", is_external=True),
            Edge(id=4, start_vertex=1, end_vertex=-4, particle="mu-", is_external=True),
        ],
    )


def _make_tchannel(diagram_id: int) -> Diagram:
    """Create a minimal t-channel Bhabha diagram (topologically distinct from s-channel)."""
    return Diagram(
        id=diagram_id,
        theory="QED",
        process="e+ e- -> e+ e-",
        loop_order=0,
        vertices=[
            Vertex(id=0, particles=["e-", "e-", "gamma"]),
            Vertex(id=1, particles=["e+", "e+", "gamma"]),
        ],
        edges=[
            Edge(id=0, start_vertex=0, end_vertex=1, particle="gamma", is_external=False),
            Edge(id=1, start_vertex=0, end_vertex=-1, particle="e-", is_external=True),
            Edge(id=2, start_vertex=0, end_vertex=-2, particle="e-", is_external=True),
            Edge(id=3, start_vertex=1, end_vertex=-3, particle="e+", is_external=True),
            Edge(id=4, start_vertex=1, end_vertex=-4, particle="e+", is_external=True),
        ],
    )


class TestDeduplicate:
    def test_no_duplicates_unchanged(self):
        """A list with no duplicates should be returned as-is."""
        diagrams = [_make_schannel(1), _make_tchannel(2)]
        result = deduplicate(diagrams)
        assert len(result) == 2

    def test_exact_duplicate_removed(self):
        """Two identical diagrams should be reduced to one."""
        d1 = _make_schannel(1)
        d2 = _make_schannel(2)
        result = deduplicate([d1, d2])
        assert len(result) == 1

    def test_triple_duplicate_removed(self):
        """Three identical diagrams should reduce to one."""
        diagrams = [_make_schannel(i) for i in range(1, 4)]
        result = deduplicate(diagrams)
        assert len(result) == 1

    def test_first_occurrence_preserved(self):
        """The first-occurring diagram should be kept, not a later copy."""
        d1 = _make_schannel(1)
        d2 = _make_schannel(2)
        result = deduplicate([d1, d2])
        assert result[0].id == 1

    def test_distinct_topologies_both_kept(self):
        """s-channel and t-channel are distinct — both should survive."""
        diagrams = [_make_schannel(1), _make_tchannel(2)]
        result = deduplicate(diagrams)
        assert len(result) == 2

    def test_canonical_hash_assigned(self):
        """All diagrams should have a canonical_hash after deduplication."""
        diagrams = [_make_schannel(1), _make_tchannel(2)]
        result = deduplicate(diagrams)
        for d in result:
            assert d.canonical_hash is not None
            assert len(d.canonical_hash) == 64  # SHA-256 hex
