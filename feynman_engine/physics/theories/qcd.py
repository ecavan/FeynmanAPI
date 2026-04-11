"""
QCD theory definition: quarks, gluons, and the full QCD vertex set.

Includes:
  - 6 quark flavors (u, d, s, c, b, t) with antiparticles
  - Gluon (octet, self-conjugate)
  - Ghost / antighost (for loop diagrams, Faddeev-Popov)
  - Vertices: qqg, ggg (3-gluon), gggg (4-gluon), ghost-gluon
"""
from feynman_engine.core.models import Particle, ParticleType, PropagatorStyle

# ── Quarks ────────────────────────────────────────────────────────────────────
_QUARK_DATA = [
    # (name, antiname, pdg_id, mass_symbol, charge)
    ("u",  "u~",  2,  "m_u",  2/3),
    ("d",  "d~",  1,  "m_d", -1/3),
    ("s",  "s~",  3,  "m_s", -1/3),
    ("c",  "c~",  4,  "m_c",  2/3),
    ("b",  "b~",  5,  "m_b", -1/3),
    ("t",  "t~",  6,  "m_t",  2/3),
]

PARTICLES: dict[str, Particle] = {}

for _name, _aname, _pdg, _mass, _charge in _QUARK_DATA:
    PARTICLES[_name] = Particle(
        name=_name,
        pdg_id=_pdg,
        particle_type=ParticleType.FERMION,
        mass=_mass,
        charge=_charge,
        color="3",
        propagator_style=PropagatorStyle.FERMION,
        antiparticle=_aname,
    )
    PARTICLES[_aname] = Particle(
        name=_aname,
        pdg_id=-_pdg,
        particle_type=ParticleType.ANTIFERMION,
        mass=_mass,
        charge=-_charge,
        color="3bar",
        propagator_style=PropagatorStyle.ANTI_FERMION,
        antiparticle=_name,
    )

# ── Gluon ─────────────────────────────────────────────────────────────────────
PARTICLES["g"] = Particle(
    name="g",
    pdg_id=21,
    particle_type=ParticleType.BOSON,
    mass="0",
    charge=0.0,
    color="8",
    propagator_style=PropagatorStyle.GLUON,
    antiparticle=None,  # self-conjugate
)

# ── Ghosts (Faddeev-Popov, needed for loop diagrams) ─────────────────────────
PARTICLES["gh"] = Particle(
    name="gh",
    pdg_id=None,
    particle_type=ParticleType.GHOST,
    mass="0",
    charge=0.0,
    color="8",
    propagator_style=PropagatorStyle.GHOST,
    antiparticle="gh~",
)
PARTICLES["gh~"] = Particle(
    name="gh~",
    pdg_id=None,
    particle_type=ParticleType.GHOST,
    mass="0",
    charge=0.0,
    color="8",
    propagator_style=PropagatorStyle.GHOST,
    antiparticle="gh",
)

# ── Vertices ──────────────────────────────────────────────────────────────────
# Each tuple lists particle names at a vertex (all orderings valid for QGRAF)
VERTICES: list[tuple] = []

# quark-antiquark-gluon for each flavor
for _name, _aname, *_ in _QUARK_DATA:
    VERTICES.append((_name, _aname, "g"))

# 3-gluon vertex
VERTICES.append(("g", "g", "g"))

# 4-gluon vertex
VERTICES.append(("g", "g", "g", "g"))

# Ghost-gluon vertex
VERTICES.append(("gh", "gh~", "g"))

# ── QGRAF name mapping ────────────────────────────────────────────────────────
# QGRAF identifiers must be simple alphanumeric
QGRAF_NAME_MAP: dict[str, str] = {
    "u":  "uq", "u~": "ua",
    "d":  "dq", "d~": "da",
    "s":  "sq", "s~": "sa",
    "c":  "cq", "c~": "ca",
    "b":  "bq", "b~": "ba",
    "t":  "tq", "t~": "ta",
    "g":  "G",
    "gh": "gh", "gh~": "gha",
}

QGRAF_NAME_REVERSE: dict[str, str] = {v: k for k, v in QGRAF_NAME_MAP.items()}

THEORY_NAME = "QCD"
MODEL_FILE = "qcd.mod"
