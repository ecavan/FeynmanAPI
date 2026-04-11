"""Tests for topology classification."""
import pytest

from feynman_engine.core.models import Diagram, Edge, Vertex
from feynman_engine.core.topology import classify_topology


def _s_channel_diagram() -> Diagram:
    # Incoming e+/e- → phantom points INTO vertex 0 (start_vertex < 0)
    # Outgoing mu+/mu- → vertex 1 points OUT to phantom (end_vertex < 0)
    return Diagram(
        id=1,
        theory="QED",
        process="e+ e- -> mu+ mu-",
        loop_order=0,
        vertices=[
            Vertex(id=0, particles=["e-", "e+", "gamma"]),
            Vertex(id=1, particles=["mu-", "mu+", "gamma"]),
        ],
        edges=[
            Edge(id=0, start_vertex=0, end_vertex=1, particle="gamma", is_external=False),
            Edge(id=1, start_vertex=-1, end_vertex=0, particle="e+", is_external=True),
            Edge(id=2, start_vertex=-2, end_vertex=0, particle="e-", is_external=True),
            Edge(id=3, start_vertex=1, end_vertex=-3, particle="mu+", is_external=True),
            Edge(id=4, start_vertex=1, end_vertex=-4, particle="mu-", is_external=True),
        ],
    )


class TestTopologyClassification:
    def test_s_channel_tree(self):
        d = _s_channel_diagram()
        assert classify_topology(d) == "s-channel"

    def test_zero_loop_one_internal_prop(self):
        """Any tree diagram with 1 internal propagator should get a channel label."""
        d = _s_channel_diagram()
        result = classify_topology(d)
        assert result in ("s-channel", "t-channel", "u-channel")

    def test_one_loop_self_energy(self):
        """1-loop diagram with 2 internal propagators → self-energy."""
        d = Diagram(
            id=2,
            theory="QED",
            process="e- -> e-",
            loop_order=1,
            vertices=[
                Vertex(id=0, particles=["e-", "gamma"]),
                Vertex(id=1, particles=["e-", "gamma"]),
            ],
            edges=[
                Edge(id=0, start_vertex=0, end_vertex=1, particle="e-", is_external=False),
                Edge(id=1, start_vertex=0, end_vertex=1, particle="gamma", is_external=False),
                Edge(id=2, start_vertex=0, end_vertex=-1, particle="e-", is_external=True),
                Edge(id=3, start_vertex=1, end_vertex=-2, particle="e-", is_external=True),
            ],
        )
        assert classify_topology(d) == "self-energy"
