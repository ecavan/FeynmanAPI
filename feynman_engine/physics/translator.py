"""Translate a user process string into QGRAF configuration files."""
from __future__ import annotations

import re
from dataclasses import dataclass

from feynman_engine.physics.registry import TheoryRegistry


@dataclass
class ProcessSpec:
    """Parsed representation of a scattering process."""
    raw: str
    incoming: list[str]    # particle names
    outgoing: list[str]    # particle names
    theory: str
    loops: int

    @property
    def qgraf_incoming(self) -> list[str]:
        return [TheoryRegistry.to_qgraf_name(self.theory, p) for p in self.incoming]

    @property
    def qgraf_outgoing(self) -> list[str]:
        return [TheoryRegistry.to_qgraf_name(self.theory, p) for p in self.outgoing]


# Canonical aliases for process string parsing
_ALIASES: dict[str, str] = {
    # QED
    "electron": "e-",
    "positron": "e+",
    "muon": "mu-",
    "antimuon": "mu+",
    "photon": "gamma",
    "γ": "gamma",
    "e": "e-",          # bare "e" defaults to electron
    # QCD
    "gluon": "g",
    "up": "u",
    "down": "d",
    "strange": "s",
    "charm": "c",
    "bottom": "b",
    "top": "t",
    "anti-up": "u~",
    "anti-down": "d~",
    "anti-strange": "s~",
    "anti-charm": "c~",
    "anti-bottom": "b~",
    "anti-top": "t~",
    # EW
    "higgs": "H",
    "w+": "W+",
    "w-": "W-",
    "z": "Z",
    "z0": "Z",
    "tau": "tau-",
    "antitau": "tau+",
}


def _normalize_particle_name(raw: str) -> str:
    raw = raw.strip()
    return _ALIASES.get(raw, raw)


def parse_process(process_str: str, theory: str, loops: int = 0) -> ProcessSpec:
    """
    Parse a process string like "e+ e- -> mu+ mu-" into a ProcessSpec.

    Supports:
      - Arrow formats: "->", "→", ">", "to"
      - Space-separated particle lists on each side
    """
    # Normalize arrow
    for arrow in ["->", "→", " to ", " > "]:
        if arrow in process_str:
            parts = process_str.split(arrow, 1)
            break
    else:
        raise ValueError(
            f"Cannot find arrow in process string '{process_str}'. "
            "Use '->' to separate incoming and outgoing particles."
        )

    incoming_raw = parts[0].strip().split()
    outgoing_raw = parts[1].strip().split()

    incoming = [_normalize_particle_name(p) for p in incoming_raw]
    outgoing = [_normalize_particle_name(p) for p in outgoing_raw]

    # Validate all particles are known in this theory
    theory_upper = theory.upper()
    known = TheoryRegistry.get_particles(theory_upper)
    for p in incoming + outgoing:
        if p not in known:
            raise ValueError(
                f"Unknown particle '{p}' for theory {theory_upper}. "
                f"Known particles: {list(known.keys())}"
            )

    return ProcessSpec(
        raw=process_str.strip(),
        incoming=incoming,
        outgoing=outgoing,
        theory=theory_upper,
        loops=loops,
    )


def write_qgraf_dat(
    spec: ProcessSpec,
    model_name: str,
    style_name: str,
    output_name: str,
    qgraf_dat_path: str,
    options: str = "notadpole",
) -> None:
    """
    Write the qgraf.dat master configuration file.

    Args:
        model_name:   Basename of the model file (e.g. 'model.mod').
                      Must be a short name — QGRAF has an ~80-char line limit.
        style_name:   Basename of the style file (e.g. 'feynman.sty').
        output_name:  Basename of the output file (e.g. 'output.txt').
        qgraf_dat_path: Full path where qgraf.dat will be written.
    """
    in_str  = ", ".join(spec.qgraf_incoming)
    out_str = ", ".join(spec.qgraf_outgoing)

    content = (
        f"output= '{output_name}' ;\n"
        f"style= '{style_name}' ;\n"
        f"model= '{model_name}' ;\n"
        f"in= {in_str} ;\n"
        f"out= {out_str} ;\n"
        f"loops= {spec.loops} ;\n"
        f"loop_momentum= k ;\n"
        f"options= {options} ;\n"
    )
    with open(qgraf_dat_path, "w") as f:
        f.write(content)
