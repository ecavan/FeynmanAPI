"""Tests for the pure-Python anti-kT jet clustering algorithm."""
from __future__ import annotations

import math

import numpy as np
import pytest

from feynman_engine.amplitudes.jet_clustering import (
    anti_kT, jets_passing_cuts, pT, pseudo_rapidity, rapidity, phi,
)


def _make_particle(E, px, py, pz):
    return np.array([E, px, py, pz], dtype=float)


class TestKinematics:
    def test_pT(self):
        p = _make_particle(10.0, 3.0, 4.0, 5.0)
        assert pT(p) == pytest.approx(5.0)

    def test_phi(self):
        # px = 1, py = 0 → phi = 0
        assert phi(_make_particle(1.0, 1.0, 0.0, 0.0)) == pytest.approx(0.0)
        # px = 0, py = 1 → phi = π/2
        assert phi(_make_particle(1.0, 0.0, 1.0, 0.0)) == pytest.approx(math.pi / 2)

    def test_rapidity(self):
        # massless, pz = 0 → y = 0
        p = _make_particle(1.0, 1.0, 0.0, 0.0)
        assert rapidity(p) == pytest.approx(0.0)
        # massive at rest → y = 0
        p = _make_particle(5.0, 0.0, 0.0, 0.0)
        assert rapidity(p) == pytest.approx(0.0)


class TestSingleParticle:
    """One particle should always cluster into itself."""

    def test_single_isolated_particle(self):
        p = _make_particle(50.0, 30.0, 40.0, 10.0)
        jets = anti_kT([p], R=0.4)
        assert len(jets) == 1
        np.testing.assert_allclose(jets[0], p)


class TestTwoParticles:
    """Two well-separated particles → two jets.  Two close particles → one jet."""

    def test_far_separated(self):
        # Two back-to-back particles, ΔR >> R
        p1 = _make_particle(50.0, 50.0, 0.0, 0.0)         # along +x, η=0, φ=0
        p2 = _make_particle(50.0, -50.0, 0.0, 0.0)        # along -x, η=0, φ=π
        jets = anti_kT([p1, p2], R=0.4)
        assert len(jets) == 2

    def test_close_particles_merge(self):
        # Two collinear particles, ΔR << R should merge
        # Both at η ≈ 0, with a small Δφ
        p1 = _make_particle(50.0, 50.0, 0.0, 0.0)
        # Small angle from p1
        small_phi = 0.05  # rad << R=0.4
        p2 = _make_particle(
            30.0, 30.0 * math.cos(small_phi), 30.0 * math.sin(small_phi), 0.0,
        )
        jets = anti_kT([p1, p2], R=0.4)
        assert len(jets) == 1
        # Merged jet should have E = E1 + E2 = 80
        assert jets[0][0] == pytest.approx(80.0)

    def test_boundary_separation(self):
        """Particles separated by exactly ΔR = R are right at the merge cutoff.
        Anti-kT will typically include them in the same jet when ΔR ≤ R, and
        excluded for ΔR > R.  We test that ΔR > R gives 2 jets.
        """
        p1 = _make_particle(50.0, 50.0, 0.0, 0.0)
        # Place p2 at Δφ = 0.5 > R = 0.4
        dphi = 0.5
        p2 = _make_particle(
            30.0, 30.0 * math.cos(dphi), 30.0 * math.sin(dphi), 0.0,
        )
        jets = anti_kT([p1, p2], R=0.4)
        assert len(jets) == 2


class TestThreeParticles:
    """Anti-kT behaviour with 3 particles — analog of NLO real emission."""

    def test_3body_cluster_to_2(self):
        """One hard particle + a soft particle nearby + one well separated:
        the soft one should cluster with the hard one (forming a 2-jet event).
        """
        p_hard = _make_particle(100.0, 100.0, 0.0, 0.0)
        # Soft, close to p_hard
        p_soft = _make_particle(
            5.0, 5.0 * math.cos(0.1), 5.0 * math.sin(0.1), 0.0,
        )
        # Hard, on the opposite side
        p_opposite = _make_particle(80.0, -80.0, 0.0, 0.0)
        jets = anti_kT([p_hard, p_soft, p_opposite], R=0.4)
        assert len(jets) == 2
        # First jet should be the merged (hard + soft) ≈ 105 GeV
        # Second jet should be p_opposite
        pTs = [pT(j) for j in jets]
        assert pTs[0] > pTs[1]      # sorted descending
        assert jets[0][0] == pytest.approx(105.0)


class TestpTOrdering:
    def test_output_sorted_descending(self):
        # 3 well-separated jets at varying pT
        ps = [
            _make_particle(20.0, 20.0, 0.0, 0.0),                    # pT=20
            _make_particle(50.0, 0.0, 50.0, 0.0),                    # pT=50
            _make_particle(30.0, -30.0 / math.sqrt(2), 0.0,
                           30.0 / math.sqrt(2)),                     # pT=30/√2 ≈ 21.2
        ]
        jets = anti_kT(ps, R=0.4)
        pTs = [pT(j) for j in jets]
        for i in range(len(pTs) - 1):
            assert pTs[i] >= pTs[i + 1]


class TestCuts:
    def test_pT_min_filter(self):
        # Two well-separated particles, one above and one below pT cut
        p_hi = _make_particle(50.0, 50.0, 0.0, 0.0)
        p_lo = _make_particle(10.0, 0.0, 10.0, 0.0)
        jets = jets_passing_cuts([p_hi, p_lo], R=0.4, pT_min=25.0, eta_max=4.5)
        assert len(jets) == 1
        assert pT(jets[0]) == pytest.approx(50.0)

    def test_eta_max_filter(self):
        # Central jet vs forward jet
        p_central = _make_particle(50.0, 50.0, 0.0, 0.0)         # η = 0
        # Forward: large pz, small pT
        p_fwd = _make_particle(100.0, 30.0, 0.0, 95.0)            # η ≈ 1.9
        jets = jets_passing_cuts([p_central, p_fwd], R=0.4, pT_min=25.0, eta_max=1.0)
        assert len(jets) == 1     # only central survives
        # Verify it's the central one
        assert jets[0][3] == pytest.approx(0.0)


class TestBackwardsCompatibility:
    """Anti-kT is a generalisation of kT (with α=−1 instead of α=1 in inverse
    pT weighting).  Verify the algorithm gives sensible structure on a
    common test input.
    """

    def test_two_balanced_back_to_back(self):
        """e+ e- → q q̄ at the partonic level: two balanced jets, back-to-back."""
        p1 = _make_particle(50.0, 50.0, 0.0, 0.0)
        p2 = _make_particle(50.0, -50.0, 0.0, 0.0)
        jets = anti_kT([p1, p2], R=0.4)
        assert len(jets) == 2
        # Both jets should have pT ≈ 50
        for j in jets:
            assert pT(j) == pytest.approx(50.0)
