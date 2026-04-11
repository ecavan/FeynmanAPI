"""
BSM simplified dark matter theory.

Model: Z' (dark photon) mediator + scalar dark matter (chi / chi~)
       built on top of the QED SM sector.

Example processes:
    e+ e- -> chi chi~     (DM pair production via Z')
    chi chi~ -> e+ e-     (DM annihilation)
    e+ e- -> e+ e-        (Bhabha, same as QED)
    mu+ mu- -> chi chi~   (muon annihilation to DM)
"""
from feynman_engine.core.models import Particle, ParticleType, PropagatorStyle

PARTICLES: dict[str, Particle] = {
    # SM QED sector
    "e-": Particle(
        name="e-", pdg_id=11, particle_type=ParticleType.FERMION,
        mass="m_e", charge=-1.0, color="1",
        propagator_style=PropagatorStyle.FERMION, antiparticle="e+",
    ),
    "e+": Particle(
        name="e+", pdg_id=-11, particle_type=ParticleType.ANTIFERMION,
        mass="m_e", charge=+1.0, color="1",
        propagator_style=PropagatorStyle.ANTI_FERMION, antiparticle="e-",
    ),
    "mu-": Particle(
        name="mu-", pdg_id=13, particle_type=ParticleType.FERMION,
        mass="m_mu", charge=-1.0, color="1",
        propagator_style=PropagatorStyle.FERMION, antiparticle="mu+",
    ),
    "mu+": Particle(
        name="mu+", pdg_id=-13, particle_type=ParticleType.ANTIFERMION,
        mass="m_mu", charge=+1.0, color="1",
        propagator_style=PropagatorStyle.ANTI_FERMION, antiparticle="mu-",
    ),
    "gamma": Particle(
        name="gamma", pdg_id=22, particle_type=ParticleType.BOSON,
        mass="0", charge=0.0, color="1",
        propagator_style=PropagatorStyle.PHOTON, antiparticle=None,
    ),
    # Mediator: Z' (dark photon, couples like photon but to DM sector too)
    "Zp": Particle(
        name="Zp", pdg_id=9900022, particle_type=ParticleType.BOSON,
        mass="m_Zp", charge=0.0, color="1",
        propagator_style=PropagatorStyle.BOSON, antiparticle=None,
    ),
    # Scalar dark matter
    "chi": Particle(
        name="chi", pdg_id=9000001, particle_type=ParticleType.SCALAR,
        mass="m_chi", charge=0.0, color="1",
        propagator_style=PropagatorStyle.SCALAR, antiparticle="chi~",
    ),
    "chi~": Particle(
        name="chi~", pdg_id=-9000001, particle_type=ParticleType.SCALAR,
        mass="m_chi", charge=0.0, color="1",
        propagator_style=PropagatorStyle.SCALAR, antiparticle="chi",
    ),
}

VERTICES: list[tuple] = [
    # SM photon couplings
    ("e-",  "e+",  "gamma"),
    ("mu-", "mu+", "gamma"),
    # Z' couplings to SM fermions
    ("e-",  "e+",  "Zp"),
    ("mu-", "mu+", "Zp"),
    # Z' coupling to dark matter
    ("chi", "chi~", "Zp"),
    # Quartic DM contact vertex
    ("chi", "chi~", "chi", "chi~"),
]

QGRAF_NAME_MAP: dict[str, str] = {
    "e-":    "em",
    "e+":    "ep",
    "mu-":   "mum",
    "mu+":   "mup",
    "gamma": "A",
    "Zp":    "Zp",
    "chi":   "chi",
    "chi~":  "chia",
}

QGRAF_NAME_REVERSE: dict[str, str] = {v: k for k, v in QGRAF_NAME_MAP.items()}

THEORY_NAME = "BSM"
MODEL_FILE  = "bsm.mod"
