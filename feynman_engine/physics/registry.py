"""Particle and theory registry — central lookup for all known theories."""
from feynman_engine.core.models import Particle
from feynman_engine.physics.theories import qed as _qed
from feynman_engine.physics.theories import qcd as _qcd
from feynman_engine.physics.theories import electroweak as _ew
from feynman_engine.physics.theories import bsm as _bsm


class TheoryRegistry:
    """
    Maps theory names to their particle content and configuration.

    Built-in theories: QED, QCD, EW (Standard Model Electroweak).
    Custom theories (BSM via UFO) can be registered at runtime with
    TheoryRegistry.register().
    """

    _theories: dict[str, dict] = {
        "QED": {
            "particles": _qed.PARTICLES,
            "vertices": _qed.VERTICES,
            "model_file": _qed.MODEL_FILE,
            "qgraf_name_map": _qed.QGRAF_NAME_MAP,
            "qgraf_name_reverse": _qed.QGRAF_NAME_REVERSE,
        },
        "QCD": {
            "particles": _qcd.PARTICLES,
            "vertices": _qcd.VERTICES,
            "model_file": _qcd.MODEL_FILE,
            "qgraf_name_map": _qcd.QGRAF_NAME_MAP,
            "qgraf_name_reverse": _qcd.QGRAF_NAME_REVERSE,
        },
        "EW": {
            "particles": _ew.PARTICLES,
            "vertices": _ew.VERTICES,
            "model_file": _ew.MODEL_FILE,
            "qgraf_name_map": _ew.QGRAF_NAME_MAP,
            "qgraf_name_reverse": _ew.QGRAF_NAME_REVERSE,
        },
        "QCDQED": {
            "particles": _qcd.PARTICLES,
            "vertices": _qcd.VERTICES,
            "model_file": "qcdqed.mod",
            "qgraf_name_map": _qcd.QGRAF_NAME_MAP,
            "qgraf_name_reverse": _qcd.QGRAF_NAME_REVERSE,
        },
        "BSM": {
            "particles": _bsm.PARTICLES,
            "vertices": _bsm.VERTICES,
            "model_file": _bsm.MODEL_FILE,
            "qgraf_name_map": _bsm.QGRAF_NAME_MAP,
            "qgraf_name_reverse": _bsm.QGRAF_NAME_REVERSE,
        },
    }

    @classmethod
    def register(cls, name: str, theory_dict: dict) -> None:
        """
        Register a custom theory (e.g. loaded from a UFO model).

        Args:
            name:         Theory name (will be uppercased).
            theory_dict:  Dict with keys: particles, vertices, model_file,
                          qgraf_name_map, qgraf_name_reverse.
        """
        required = {"particles", "vertices", "model_file", "qgraf_name_map", "qgraf_name_reverse"}
        missing = required - set(theory_dict.keys())
        if missing:
            raise ValueError(f"Theory dict missing keys: {missing}")
        cls._theories[name.upper()] = theory_dict

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a registered theory. Built-in theories cannot be removed."""
        name = name.upper()
        if name in ("QED", "QCD", "EW"):
            raise ValueError(f"Cannot unregister built-in theory '{name}'.")
        cls._theories.pop(name, None)

    @classmethod
    def list_theories(cls) -> list[str]:
        return list(cls._theories.keys())

    @classmethod
    def get_theory(cls, name: str) -> dict:
        name = name.upper()
        if name not in cls._theories:
            raise ValueError(f"Unknown theory '{name}'. Available: {cls.list_theories()}")
        return cls._theories[name]

    @classmethod
    def get_particles(cls, theory: str) -> dict[str, Particle]:
        return cls.get_theory(theory)["particles"]

    @classmethod
    def get_particle(cls, theory: str, name: str) -> Particle:
        particles = cls.get_particles(theory)
        if name not in particles:
            raise ValueError(f"Unknown particle '{name}' in {theory}. Known: {list(particles.keys())}")
        return particles[name]

    @classmethod
    def to_qgraf_name(cls, theory: str, name: str) -> str:
        mapping = cls.get_theory(theory)["qgraf_name_map"]
        return mapping.get(name, name)

    @classmethod
    def from_qgraf_name(cls, theory: str, qgraf_name: str) -> str:
        reverse = cls.get_theory(theory)["qgraf_name_reverse"]
        return reverse.get(qgraf_name, qgraf_name)
