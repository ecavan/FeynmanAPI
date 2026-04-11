"""Tests for the QGRAF-only diagram generator path."""

import pytest

from feynman_engine.core.generator import QGRAFError, generate_diagrams
from feynman_engine.physics.translator import parse_process


def test_generate_diagrams_requires_qgraf(monkeypatch):
    """Generation should fail clearly when no QGRAF executable is available."""
    monkeypatch.setattr("feynman_engine.core.generator._qgraf_pipe_path", lambda: None)
    monkeypatch.setattr("feynman_engine.core.generator._qgraf_std_path", lambda: None)

    spec = parse_process("e+ e- -> mu+ mu-", theory="QED", loops=0)

    with pytest.raises(QGRAFError, match="QGRAF is required"):
        generate_diagrams(spec)
