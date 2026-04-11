"""
Standard Model Electroweak theory definition.

Includes:
  - All SM fermions: 3 generations of leptons + quarks
  - Gauge bosons: γ, W+, W-, Z
  - Higgs boson: H
  - Full SM vertex set (tree-level)

This is the full Standard Model gauge sector. For BSM extensions use the
UFO model loader in feynman_engine/physics/ufo_loader.py.
"""
from feynman_engine.core.models import Particle, ParticleType, PropagatorStyle

PARTICLES: dict[str, Particle] = {}

# ── Leptons ───────────────────────────────────────────────────────────────────
_LEPTON_DATA = [
    # (name, antiname, pdg_id, mass_symbol, charge)
    ("e-",   "e+",   11,  "m_e",   -1.0),
    ("mu-",  "mu+",  13,  "m_mu",  -1.0),
    ("tau-", "tau+", 15,  "m_tau", -1.0),
    ("nu_e",   "nu_e~",   12, "0", 0.0),
    ("nu_mu",  "nu_mu~",  14, "0", 0.0),
    ("nu_tau", "nu_tau~", 16, "0", 0.0),
]

for _name, _aname, _pdg, _mass, _charge in _LEPTON_DATA:
    PARTICLES[_name] = Particle(
        name=_name, pdg_id=_pdg,
        particle_type=ParticleType.FERMION,
        mass=_mass, charge=_charge, color="1",
        propagator_style=PropagatorStyle.FERMION,
        antiparticle=_aname,
    )
    PARTICLES[_aname] = Particle(
        name=_aname, pdg_id=-_pdg,
        particle_type=ParticleType.ANTIFERMION,
        mass=_mass, charge=-_charge, color="1",
        propagator_style=PropagatorStyle.ANTI_FERMION,
        antiparticle=_name,
    )

# ── Quarks ────────────────────────────────────────────────────────────────────
_QUARK_DATA = [
    ("u",  "u~",  2,  "m_u",  2/3),
    ("d",  "d~",  1,  "m_d", -1/3),
    ("s",  "s~",  3,  "m_s", -1/3),
    ("c",  "c~",  4,  "m_c",  2/3),
    ("b",  "b~",  5,  "m_b", -1/3),
    ("t",  "t~",  6,  "m_t",  2/3),
]

for _name, _aname, _pdg, _mass, _charge in _QUARK_DATA:
    PARTICLES[_name] = Particle(
        name=_name, pdg_id=_pdg,
        particle_type=ParticleType.FERMION,
        mass=_mass, charge=_charge, color="3",
        propagator_style=PropagatorStyle.FERMION,
        antiparticle=_aname,
    )
    PARTICLES[_aname] = Particle(
        name=_aname, pdg_id=-_pdg,
        particle_type=ParticleType.ANTIFERMION,
        mass=_mass, charge=-_charge, color="3bar",
        propagator_style=PropagatorStyle.ANTI_FERMION,
        antiparticle=_name,
    )

# ── Gauge bosons ──────────────────────────────────────────────────────────────
PARTICLES["gamma"] = Particle(
    name="gamma", pdg_id=22,
    particle_type=ParticleType.BOSON,
    mass="0", charge=0.0, color="1",
    propagator_style=PropagatorStyle.PHOTON,
    antiparticle=None,
)
PARTICLES["Z"] = Particle(
    name="Z", pdg_id=23,
    particle_type=ParticleType.BOSON,
    mass="m_Z", charge=0.0, color="1",
    propagator_style=PropagatorStyle.BOSON,
    antiparticle=None,
)
PARTICLES["W+"] = Particle(
    name="W+", pdg_id=24,
    particle_type=ParticleType.BOSON,
    mass="m_W", charge=+1.0, color="1",
    propagator_style=PropagatorStyle.CHARGED_BOSON,
    antiparticle="W-",
)
PARTICLES["W-"] = Particle(
    name="W-", pdg_id=-24,
    particle_type=ParticleType.BOSON,
    mass="m_W", charge=-1.0, color="1",
    propagator_style=PropagatorStyle.CHARGED_BOSON,
    antiparticle="W+",
)

# ── Higgs ─────────────────────────────────────────────────────────────────────
PARTICLES["H"] = Particle(
    name="H", pdg_id=25,
    particle_type=ParticleType.SCALAR,
    mass="m_H", charge=0.0, color="1",
    propagator_style=PropagatorStyle.SCALAR,
    antiparticle=None,
)

# ── Vertices (Standard Model tree-level) ─────────────────────────────────────
VERTICES: list[tuple] = []

# QED-like photon couplings to all charged fermions
_CHARGED_FERMION_PAIRS = [
    ("e-", "e+"), ("mu-", "mu+"), ("tau-", "tau+"),
    ("u", "u~"), ("d", "d~"), ("s", "s~"), ("c", "c~"), ("b", "b~"), ("t", "t~"),
]
for _f, _af in _CHARGED_FERMION_PAIRS:
    VERTICES.append((_f, _af, "gamma"))
    VERTICES.append((_f, _af, "Z"))

# Z couplings to neutrinos (neutral current)
for _nu in ["nu_e", "nu_mu", "nu_tau"]:
    VERTICES.append((_nu, f"{_nu}~", "Z"))

# W couplings: charged leptons ↔ neutrinos
_LEPTON_W_PAIRS = [
    ("e-", "nu_e"), ("mu-", "nu_mu"), ("tau-", "nu_tau"),
]
for _l, _nu in _LEPTON_W_PAIRS:
    VERTICES.append((_l,  f"{_l[:-1]}+", "W-"))  # e- ebar W-... actually need:
    VERTICES.append((_nu, f"{_l[:-1]}+", "W-"))
    VERTICES.append((_l,  f"nu_{_l[:-1].replace('u-','u')}~", "W+"))

# Simpler W-fermion vertices (CKM-approximate, diagonal)
for _l, _nu in [("e-", "nu_e"), ("mu-", "nu_mu"), ("tau-", "tau-")]:
    # lepton W vertices
    VERTICES.append((_l, f"{_nu}~", "W+"))
    VERTICES.append((f"{_l[:-1]}+", _nu, "W-"))

# Quark W vertices (diagonal CKM)
for _up, _down in [("u","d"), ("c","s"), ("t","b")]:
    VERTICES.append((_up, f"{_down}~", "W+"))
    VERTICES.append((f"{_up}~", _down, "W-"))
    VERTICES.append((_up, f"{_up}~", "Z"))
    VERTICES.append((_down, f"{_down}~", "Z"))

# Gauge self-couplings
VERTICES.append(("W+", "W-", "gamma"))
VERTICES.append(("W+", "W-", "Z"))
VERTICES.append(("W+", "W-", "gamma", "gamma"))
VERTICES.append(("W+", "W-", "Z", "Z"))
VERTICES.append(("W+", "W-", "Z", "gamma"))
VERTICES.append(("W+", "W+", "W-", "W-"))

# Higgs couplings
VERTICES.append(("H", "H", "Z", "Z"))
VERTICES.append(("H", "Z", "Z"))
VERTICES.append(("H", "W+", "W-"))
VERTICES.append(("H", "H", "W+", "W-"))
VERTICES.append(("H", "H", "H"))
VERTICES.append(("H", "H", "H", "H"))

# Yukawa (Higgs to massive fermions)
for _f, _af in [("e-","e+"), ("mu-","mu+"), ("tau-","tau+"),
                ("t","t~"), ("b","b~"), ("c","c~")]:
    VERTICES.append((_f, _af, "H"))

# ── QGRAF name mapping ────────────────────────────────────────────────────────
QGRAF_NAME_MAP: dict[str, str] = {
    "e-": "em",    "e+": "ep",
    "mu-": "mum",  "mu+": "mup",
    "tau-": "taum","tau+": "taup",
    "nu_e": "nue",     "nu_e~": "nuea",
    "nu_mu": "numu",   "nu_mu~": "numua",
    "nu_tau": "nutau", "nu_tau~": "nutaua",
    "u": "uq",  "u~": "ua",
    "d": "dq",  "d~": "da",
    "s": "sq",  "s~": "sa",
    "c": "cq",  "c~": "ca",
    "b": "bq",  "b~": "ba",
    "t": "tq",  "t~": "ta",
    "gamma": "A",
    "Z": "Z",
    "W+": "Wp", "W-": "Wm",
    "H": "H",
}
QGRAF_NAME_REVERSE: dict[str, str] = {v: k for k, v in QGRAF_NAME_MAP.items()}

THEORY_NAME = "EW"
MODEL_FILE = "electroweak.mod"
