"""QED theory definition: particles, vertices, and QGRAF model configuration."""
from feynman_engine.core.models import Particle, ParticleType, PropagatorStyle

# All particles in QED (name → Particle)
PARTICLES: dict[str, Particle] = {
    "e-": Particle(
        name="e-",
        pdg_id=11,
        particle_type=ParticleType.FERMION,
        mass="m_e",
        charge=-1.0,
        color="1",
        propagator_style=PropagatorStyle.FERMION,
        antiparticle="e+",
    ),
    "e+": Particle(
        name="e+",
        pdg_id=-11,
        particle_type=ParticleType.ANTIFERMION,
        mass="m_e",
        charge=+1.0,
        color="1",
        propagator_style=PropagatorStyle.ANTI_FERMION,
        antiparticle="e-",
    ),
    "mu-": Particle(
        name="mu-",
        pdg_id=13,
        particle_type=ParticleType.FERMION,
        mass="m_mu",
        charge=-1.0,
        color="1",
        propagator_style=PropagatorStyle.FERMION,
        antiparticle="mu+",
    ),
    "mu+": Particle(
        name="mu+",
        pdg_id=-13,
        particle_type=ParticleType.ANTIFERMION,
        mass="m_mu",
        charge=+1.0,
        color="1",
        propagator_style=PropagatorStyle.ANTI_FERMION,
        antiparticle="mu-",
    ),
    "tau-": Particle(
        name="tau-",
        pdg_id=15,
        particle_type=ParticleType.FERMION,
        mass="m_tau",
        charge=-1.0,
        color="1",
        propagator_style=PropagatorStyle.FERMION,
        antiparticle="tau+",
    ),
    "tau+": Particle(
        name="tau+",
        pdg_id=-15,
        particle_type=ParticleType.ANTIFERMION,
        mass="m_tau",
        charge=+1.0,
        color="1",
        propagator_style=PropagatorStyle.ANTI_FERMION,
        antiparticle="tau-",
    ),
    "gamma": Particle(
        name="gamma",
        pdg_id=22,
        particle_type=ParticleType.BOSON,
        mass="0",
        charge=0.0,
        color="1",
        propagator_style=PropagatorStyle.PHOTON,
        antiparticle=None,  # self-conjugate
    ),
}

# Allowed QED vertices: sets of 3 particle names (fermion + antifermion + photon)
# Each tuple is (fermion, antifermion, photon) — all orderings are valid
VERTICES: list[tuple[str, str, str]] = [
    ("e-", "e+", "gamma"),
    ("mu-", "mu+", "gamma"),
    ("tau-", "tau+", "gamma"),
]

# Mapping from user-friendly names to canonical QGRAF internal names
# QGRAF model files use simple identifiers without special chars
QGRAF_NAME_MAP: dict[str, str] = {
    "e-": "em",
    "e+": "ep",
    "mu-": "mum",
    "mu+": "mup",
    "tau-": "taum",
    "tau+": "taup",
    "gamma": "A",
}

QGRAF_NAME_REVERSE: dict[str, str] = {v: k for k, v in QGRAF_NAME_MAP.items()}

THEORY_NAME = "QED"
MODEL_FILE = "qed.mod"
