"""OpenLoops process catalog and pack definitions.

The catalog (built from `channels_public.rinfo`) maps every supported
particle multiset to the OL libraries that cover it, plus per-library
metadata (theory class, channel count).  When a user queries a process
that has no curated formula AND no installed OL library, the engine
uses this catalog to surface a precise install suggestion.

Packs group libraries by user role / theory focus rather than by OL
naming.  Each pack carries an honest disk + time estimate so the
setup wizard can show users what they're about to commit to.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    """Load the OL catalog JSON (cached)."""
    if not _CATALOG_PATH.exists():
        return {"libraries": {}, "process_index": {}}
    with open(_CATALOG_PATH) as fp:
        return json.load(fp)


_PROTON_INITIATORS = ("u", "u~", "d", "d~", "s", "s~", "c", "c~", "b", "b~", "g")


def libraries_for_process(process: str) -> list[str]:
    """Return list of OL libraries that cover the given process string.

    Looks up by canonical particle multiset (sorted alphabetically).  For
    hadronic queries (`p p -> X`), tries every (initiator1, initiator2)
    combination over partonic initiators (u, d, s, c, b, g + antiquarks)
    and returns the union of libraries that match any combination.

    Examples:
      'g g -> t t~'      → ['pptt', 'pptt_ew']
      'e+ e- -> mu+ mu-' → ['eell_ew']
      'p p -> H'         → ['pph', 'pphj', 'pphjj', ...]
    """
    if "->" not in process:
        return []
    parts = process.split("->", 1)
    incoming = parts[0].split()
    outgoing = parts[1].split()
    cat = load_catalog()
    proc_idx = cat.get("process_index", {})

    # Hadronic case: expand 'p p' over all partonic initiator pairs.
    has_proton = any(p == "p" for p in incoming)
    if has_proton:
        slots = [
            (_PROTON_INITIATORS if p == "p" else (p,))
            for p in incoming
        ]
        # Enumerate the cartesian product of initiator slots.
        from itertools import product
        found: set[str] = set()
        for combo in product(*slots):
            multiset = sorted(list(combo) + outgoing)
            key = " ".join(multiset)
            for lib in proc_idx.get(key, []):
                found.add(lib)
        return sorted(found)

    multiset = sorted(incoming + outgoing)
    key = " ".join(multiset)
    return list(proc_idx.get(key, []))


def library_meta(library: str) -> dict[str, Any] | None:
    """Per-library metadata: theory bucket, channel count, etc."""
    return load_catalog().get("libraries", {}).get(library)


def estimate_install(libraries: list[str]) -> dict[str, Any]:
    """Honest size + time estimate for a set of libraries.

    Calibrated from real measurements on a 7-library install (Apple Silicon,
    gfortran-13, default precision).  Median library compiles to ~20 MB
    in ~2 minutes; multi-leg libraries (eella_ew, eevvjj-class) can hit
    100 MB and 10 minutes.
    """
    n = len(libraries)
    HEAVY = {"eella_ew", "eevvjj", "ppvvjj", "ppwjjj", "ppzjjj", "ppvvv",
             "ppvvvj", "ppvvvv", "ppwajj", "ppwwjj", "pp4lj", "pp4lj_ew"}
    disk_mb = 0
    build_min = 0
    for lib in libraries:
        if lib in HEAVY:
            disk_mb  += 80
            build_min += 8
        else:
            disk_mb  += 20
            build_min += 2
    return {
        "n_libraries": n,
        "disk_mb":     disk_mb,
        "build_min":   build_min,
        "human_disk":  f"~{disk_mb/1024:.1f} GB" if disk_mb >= 1024 else f"~{disk_mb} MB",
        "human_time":  (
            f"~{build_min // 60}h {build_min % 60}m" if build_min >= 60
            else f"~{build_min} min"
        ),
    }


# ─── Semantic packs ────────────────────────────────────────────────────────────
# Each pack is a curated set of libraries grouped by user intent.  Names use
# physics terminology (textbook, lhc-higgs, …) rather than OL library codes.

PACKS: dict[str, dict[str, Any]] = {
    "textbook": {
        "label":       "Textbook physics (recommended starter)",
        "description": (
            "Drell-Yan, top pair, Bhabha, e+e-→μμ.  Covers ~80% of "
            "introductory HEP textbook examples."
        ),
        "audience":    "Students, instructors, anyone running canonical 2→2 LO+NLO checks",
        "libraries":   ["ppllj", "pptt", "eell_ew", "eett_ew"],
    },
    "lhc-higgs": {
        "label":       "LHC Higgs measurements",
        "description": "ggF (loop²), H+jet, VBF, ttH, di-Higgs, Hbb̄.",
        "audience":    "ATLAS/CMS Higgs analysts; theorists studying SM Higgs",
        # Real public OL libraries (gg→H is loop-induced, so loop² libs use the
        # `_2` suffix; VBF uses `_vbf`).
        "libraries":   ["pphj2", "pphjj2", "pphjj_vbf", "pphtt", "pphh2", "pphbb"],
    },
    "lhc-top": {
        "label":       "LHC top-quark physics",
        "description": "tt̄, tt̄+jets, tt̄W, tt̄Z, single-top.",
        "audience":    "Top physics groups; anyone working on m_t, σ_tt̄, asymmetries",
        "libraries":   ["pptt", "ppttj", "ppwtt", "ppztt", "tbln", "tbqq"],
    },
    "lhc-vplus": {
        "label":       "LHC W/Z + jets (Drell-Yan + V+jets)",
        "description": "pp→W+jets, pp→Z+jets, pp→ll+jets, pp→Wγ.",
        "audience":    "Standard candles, V+jets backgrounds, EW precision at LHC",
        "libraries":   ["ppllj", "ppvj", "ppwjj", "ppzjj", "ppwjjj", "ppzjjj"],
    },
    "lhc-diboson": {
        "label":       "LHC di-boson and tri-boson",
        "description": "WW, WZ, ZZ, Wγ, Zγ at NLO, tri-boson WWZ/ZZZ where available.",
        "audience":    "Anomalous-coupling searches, EWPT-at-LHC, di-boson cross-checks",
        "libraries":   ["ppvv", "ppvvj", "ppvvv", "ppwwjj"],
    },
    "ee-future": {
        "label":       "Future lepton collider (FCC-ee / ILC / muon collider)",
        "description": "e+e-→ll, ZZ, WW, ZH, tt̄, tt̄H at EW NLO.",
        "audience":    "FCC-ee, ILC, CEPC, muon-collider physics studies",
        "libraries":   ["eell_ew", "eella_ew", "eett_ew", "eevv_ew", "eevvjj",
                        "eehtt", "eehv_ew"],
    },
    "qcd-only": {
        "label":       "Pure QCD processes",
        "description": "Every OL library where the LO is pure QCD (α_s only).",
        "audience":    "Lattice/perturbative QCD researchers, jet physics",
        "libraries":   None,
        "auto_filter": {"theory": "QCD"},
    },
    "ew-only": {
        "label":       "Pure-EW processes (lepton-collider + EW NLO at LHC)",
        "description": "Every OL library where the LO is pure EW (α only).",
        "audience":    "Lepton-collider, EW NLO corrections at LHC, Sudakov regime",
        "libraries":   None,
        "auto_filter": {"theory": "EW"},
    },
    "all-lhc-nlo": {
        "label":       "Everything LHC NLO QCD (heavy install)",
        "description": "Combined lhc-higgs + lhc-top + lhc-vplus + lhc-diboson packs.",
        "audience":    "LHC analysts who want every common measurement covered",
        "libraries":   None,
        "auto_combine": ["lhc-higgs", "lhc-top", "lhc-vplus", "lhc-diboson"],
    },
    "all": {
        "label":       "Everything (~5-6 GB, ~7 hours)",
        "description": (
            "All 222 OpenLoops public libraries.  Use only if you have a "
            "specific reason; most users want a focused pack."
        ),
        "audience":    "Library distributors, OL benchmark validators, SM completists",
        "libraries":   None,
        "auto_filter": {"theory": "*"},
    },
}


def resolve_pack(name: str) -> list[str]:
    """Expand a pack name into a concrete library list."""
    if name not in PACKS:
        raise KeyError(f"Unknown pack {name!r}.  Available: {list(PACKS)}")
    pack = PACKS[name]
    libs = pack.get("libraries")
    if libs is not None:
        return list(libs)
    if "auto_combine" in pack:
        combined: list[str] = []
        seen: set[str] = set()
        for child in pack["auto_combine"]:
            for lib in resolve_pack(child):
                if lib not in seen:
                    combined.append(lib)
                    seen.add(lib)
        return combined
    if "auto_filter" in pack:
        catalog = load_catalog()
        theory_filter = pack["auto_filter"]["theory"]
        out = []
        for lib_name, meta in catalog.get("libraries", {}).items():
            if theory_filter == "*" or meta.get("theory") == theory_filter:
                out.append(lib_name)
        return sorted(out)
    return []


def pack_summary(name: str) -> dict[str, Any]:
    """Pack metadata + concrete library list + size/time estimate."""
    pack = PACKS[name].copy()
    libs = resolve_pack(name)
    pack["resolved_libraries"] = libs
    pack["estimate"] = estimate_install(libs)
    return pack
