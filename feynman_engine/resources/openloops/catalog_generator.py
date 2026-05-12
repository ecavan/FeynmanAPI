"""Generate openloops_catalog.json from OL's channels_public.rinfo.

The catalog maps every (process-multiset → list of OL libraries that cover it),
plus per-library metadata (theory class, expected disk/time, dominant order).
This is consumed at runtime when an un-curated process query is missing the
matching OL library — the API uses the catalog to surface a precise
`feynman install-process <name>` suggestion in the 422 response.

Run: python -m feynman_engine.resources.openloops.catalog_generator [rinfo_path]
Output: feynman_engine/resources/openloops/catalog.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ─── OL channel-name decoding ──────────────────────────────────────────────────
# OL packs every external particle as a 1-3 character code; channel strings
# concatenate codes in canonical sort order (particle-before-antiparticle,
# alphabetical).  Greedy longest-prefix match decodes them.
_OL_PART_CODES: list[tuple[str, str]] = [
    # 3-char (anti-neutrinos)
    ("nex", "nu_e~"), ("nmx", "nu_mu~"), ("nlx", "nu_tau~"),
    # 2-char (neutrinos + antiparticles + W+)
    ("ne",  "nu_e"),  ("nm",  "nu_mu"),  ("nl",  "nu_tau"),
    ("ex", "e+"), ("mx", "mu+"), ("lx", "tau+"),
    ("ux", "u~"), ("dx", "d~"), ("cx", "c~"),
    ("sx", "s~"), ("tx", "t~"), ("bx", "b~"),
    ("wx", "W+"),
    # 1-char (particles + self-conjugate bosons)
    ("e", "e-"), ("m", "mu-"), ("l", "tau-"),
    ("u", "u"), ("d", "d"), ("c", "c"),
    ("s", "s"), ("t", "t"), ("b", "b"),
    ("w", "W-"), ("g", "g"), ("a", "gamma"), ("z", "Z"), ("h", "H"),
]


def parse_channel(channel: str) -> list[str]:
    """Greedy longest-match decode of an OL channel name."""
    parts: list[str] = []
    i = 0
    while i < len(channel):
        for code, name in _OL_PART_CODES:
            if channel.startswith(code, i):
                parts.append(name)
                i += len(code)
                break
        else:
            raise ValueError(f"Unknown OL particle code at {i}: {channel[i:]!r}")
    return parts


# ─── Theory classification ─────────────────────────────────────────────────────

# Heuristic for binning libraries into pedagogical "theory" buckets.  We use
# the LEADING coupling order (lowest qcd, ew indices in the .info line) since
# that determines the dominant LO physics; libraries with α_s > 0 are QCD,
# α_s = 0 + α > 0 are pure-EW.  Mixed (α_s, α both > 0) is QCDxEW.
def classify_theory(qcd_order: tuple[int, int], ew_order: tuple[int, int]) -> str:
    qcd_lo = qcd_order[0]
    ew_lo = ew_order[0]
    if qcd_lo > 0 and ew_lo == 0:
        return "QCD"
    if qcd_lo == 0 and ew_lo > 0:
        return "EW"
    if qcd_lo > 0 and ew_lo > 0:
        return "QCDxEW"
    return "OTHER"


# ─── Channel parsing ───────────────────────────────────────────────────────────

# Lines look like:
#   eell_ew eeexex 1 EW=2,1 QCD=0,0 MODEL=sm OLMode=3 Type=lt ...
# We extract: library, channel, EW=(LO,NLO), QCD=(LO,NLO), Type, MODEL.
_LINE_RE = re.compile(
    r"^(?P<lib>[A-Za-z0-9_]+)\s+"
    r"(?P<chan>[a-z]+)\s+"
    r"(?P<sub>\d+)\s+"
    r"EW=(?P<ew_lo>\d+),(?P<ew_nlo>\d+)\s+"
    r"QCD=(?P<qcd_lo>\d+),(?P<qcd_nlo>\d+)\s+"
    r".*?MODEL=(?P<model>\S+).*?Type=(?P<typ>\S+)"
)


def parse_rinfo(path: str) -> list[dict]:
    """Parse channels_public.rinfo into per-channel records."""
    out: list[dict] = []
    with open(path) as fp:
        for raw in fp:
            line = raw.strip()
            if not line or line.startswith("#") or "options=" in line:
                continue
            m = _LINE_RE.match(line)
            if m is None:
                continue
            try:
                particles = parse_channel(m.group("chan"))
            except ValueError:
                continue
            out.append({
                "library":   m.group("lib"),
                "channel":   m.group("chan"),
                "subindex":  int(m.group("sub")),
                "ew_order":  (int(m.group("ew_lo")), int(m.group("ew_nlo"))),
                "qcd_order": (int(m.group("qcd_lo")), int(m.group("qcd_nlo"))),
                "type":      m.group("typ"),
                "model":     m.group("model"),
                "particles": particles,
            })
    return out


# ─── Library aggregation ───────────────────────────────────────────────────────

def aggregate_libraries(records: list[dict]) -> dict[str, dict]:
    """One row per OL library, merged across all sub-channels.

    Drops per-channel detail (>700 KB) — that data lives in OL's own
    `channels_public.rinfo` and isn't needed at runtime.  Keeps only
    the fields that drive theory bucketing, install advice, and UI.
    """
    libs: dict[str, dict] = defaultdict(lambda: {
        "n_channels": 0,
        "qcd_orders": set(),
        "ew_orders": set(),
        "types": set(),
        "models": set(),
    })
    for r in records:
        lib = libs[r["library"]]
        lib["n_channels"] += 1
        lib["qcd_orders"].add(r["qcd_order"])
        lib["ew_orders"].add(r["ew_order"])
        lib["types"].add(r["type"])
        lib["models"].add(r["model"])

    out: dict[str, dict] = {}
    for name, info in libs.items():
        qcd_los = [o[0] for o in info["qcd_orders"]]
        ew_los = [o[0] for o in info["ew_orders"]]
        canonical_qcd = (min(qcd_los), min(o[1] for o in info["qcd_orders"] if o[0] == min(qcd_los)))
        canonical_ew = (min(ew_los), min(o[1] for o in info["ew_orders"] if o[0] == min(ew_los)))
        out[name] = {
            "name":       name,
            "theory":     classify_theory(canonical_qcd, canonical_ew),
            "qcd_orders": sorted([list(o) for o in info["qcd_orders"]]),
            "ew_orders":  sorted([list(o) for o in info["ew_orders"]]),
            "types":      sorted(info["types"]),
            "models":     sorted(info["models"]),
            "n_channels": info["n_channels"],
        }
    return out


# ─── Process → library reverse index ───────────────────────────────────────────

def canonical_multiset_key(particles: list[str]) -> str:
    """Stable string for a particle multiset (sorted alphabetically)."""
    return " ".join(sorted(particles))


def build_process_index(records: list[dict]) -> dict[str, list[str]]:
    """For every distinct particle multiset, list libraries that cover it."""
    idx: dict[str, set[str]] = defaultdict(set)
    for r in records:
        key = canonical_multiset_key(r["particles"])
        idx[key].add(r["library"])
    return {k: sorted(libs) for k, libs in idx.items()}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    rinfo_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "/tmp/openloops-install/proclib/channels_public.rinfo"
    )
    if not os.path.exists(rinfo_path):
        print(f"channels_public.rinfo not found at {rinfo_path}", file=sys.stderr)
        print("Pass an explicit path, or install OpenLoops first.", file=sys.stderr)
        return 1

    records = parse_rinfo(rinfo_path)
    if not records:
        print("No channels parsed — check the file format.", file=sys.stderr)
        return 1

    libraries = aggregate_libraries(records)
    process_index = build_process_index(records)

    out_path = Path(__file__).resolve().parent / "catalog.json"
    payload = {
        "generated_from": rinfo_path,
        "n_records":      len(records),
        "n_libraries":    len(libraries),
        "n_distinct_particle_multisets": len(process_index),
        "libraries":      libraries,
        "process_index":  process_index,
    }
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    print(f"Wrote {out_path} — {len(libraries)} libraries, "
          f"{len(records)} channels, {len(process_index)} distinct processes")

    # Tiny stats summary
    by_theory: dict[str, int] = defaultdict(int)
    for lib in libraries.values():
        by_theory[lib["theory"]] += 1
    print("Libraries by theory bucket:")
    for th, n in sorted(by_theory.items(), key=lambda kv: -kv[1]):
        print(f"  {th:8} {n:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
