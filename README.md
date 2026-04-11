# Feynman Engine

A self-contained Feynman diagram generator for particle physics. Give it a scattering process, get back diagrams — as SVG images, TikZ LaTeX, or structured JSON.

Powered by [QGRAF](http://cfif.ist.utl.pt/~paulo/qgraf.html) for diagram enumeration and [TikZ-Feynman](https://ctan.org/pkg/tikz-feynman) for rendering. No Mathematica, no cloud dependencies.

## What it does

- Enumerates all Feynman diagrams for a given process at any loop order
- Renders diagrams to SVG/PNG/PDF via lualatex
- Supports QED, QCD, Electroweak (full SM), and a BSM dark matter model
- REST API + web UI for interactive use
- Python library for scripting

## Supported theories

| Theory | Particles | Example process |
|--------|-----------|-----------------|
| QED | e±, μ±, γ | `e+ e- -> mu+ mu-` |
| QCD | 6 quarks, gluon, ghosts | `u u~ -> g g` |
| EW | Full SM gauge sector + Higgs | `e+ e- -> W+ W-` |
| BSM | QED + Z' mediator + scalar DM χ | `e+ e- -> chi chi~` |

Custom theories can be registered at runtime via `TheoryRegistry.register()` — supply a QGRAF model file and particle definitions.

## Quickstart

```bash
git clone <repo>
cd FeynmanAPI

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Compile QGRAF

QGRAF is free academic software. Download the Fortran source from the [author's page](http://cfif.ist.utl.pt/~paulo/qgraf.html) (requires a brief email registration), then:

```bash
mkdir -p bin
gfortran -O2 -o bin/qgraf qgraf-X.X.fXX
```

Place the binary at `bin/qgraf` — the engine auto-detects it.

> **Without QGRAF:** tree-level diagrams still work via the built-in pure Python enumerator. Loop diagrams require QGRAF.

### Install rendering dependencies (for SVG output)

```bash
brew install --cask basictex   # lualatex
sudo tlmgr install tikz-feynman
brew install pdf2svg
```

### Run the web app

```bash
python main.py serve
# open http://localhost:8000
```

### Python API

```python
from feynman_engine import FeynmanEngine

engine = FeynmanEngine()

# Tree-level e+e- → μ+μ-
result = engine.generate("e+ e- -> mu+ mu-", theory="QED", loops=0)
print(result.summary)
# {'total_diagrams': 1, 'topology_counts': {'s-channel': 1}, 'loop_order': 0}

# Get TikZ source
for diagram_id, tikz in result.tikz_code.items():
    print(tikz)

# 1-loop QED (requires QGRAF)
result = engine.generate("e+ e- -> mu+ mu-", theory="QED", loops=1)
print(result.summary)
# {'total_diagrams': 10, ...}

# BSM dark matter
result = engine.generate("e+ e- -> chi chi~", theory="BSM", loops=0)
# 1 diagram: e+e- annihilation to χχ̄ via Z' mediator
```

### REST API

```bash
# Check backend status
curl http://localhost:8000/api/status

# Generate diagrams
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"process": "e+ e- -> mu+ mu-", "theory": "QED", "loops": 0, "output_format": "tikz"}'

# Get SVG for a diagram (after /generate)
curl http://localhost:8000/api/diagram/1/svg > diagram.svg

# Get TikZ source
curl http://localhost:8000/api/diagram/1/tikz

# Validate a process string
curl "http://localhost:8000/api/describe?process=e%2B+e-+-%3E+mu%2B+mu-&theory=QED"

# List particles in a theory
curl http://localhost:8000/api/theories/QED/particles
```

The full OpenAPI docs are at `http://localhost:8000/docs`.

## CLI

```bash
# Start the server
python main.py serve [--host 0.0.0.0] [--port 8000]

# Generate from the command line
python main.py generate "e+ e- -> mu+ mu-" --theory QED --loops 0
```

## Output formats

| Format | Requires | Description |
|--------|----------|-------------|
| `tikz` | nothing | Raw LaTeX source you can drop into any document |
| `svg`  | lualatex + pdf2svg | Vector image, embeds in web/HTML |
| `pdf`  | lualatex | Print-quality PDF |
| `png`  | lualatex + pdf2svg | Raster image |

## System requirements

| Tool | Required for | Install |
|------|-------------|---------|
| Python ≥ 3.11 | everything | — |
| QGRAF | loop diagrams | see above |
| gfortran | compiling QGRAF | `brew install gcc` |
| lualatex | SVG/PDF/PNG output | `brew install --cask basictex` |
| pdf2svg | SVG output | `brew install pdf2svg` |

## Project structure

```
feynman_engine/
├── core/
│   ├── models.py       — Pydantic data models (Diagram, Particle, Edge, …)
│   ├── generator.py    — Runs QGRAF; falls back to Python enumerator
│   ├── parser.py       — Parses QGRAF output into Diagram objects
│   ├── enumerator.py   — Pure Python tree-level fallback
│   ├── normalize.py    — Graph-isomorphism deduplication (NetworkX)
│   └── topology.py     — Classifies s/t/u-channel, box, self-energy, …
├── physics/
│   ├── registry.py     — Central theory registry
│   ├── translator.py   — Parses process strings; writes qgraf.dat
│   └── theories/       — QED, QCD, EW, BSM definitions
├── render/
│   ├── tikz.py         — Diagram → TikZ-Feynman LaTeX (Jinja2)
│   └── compiler.py     — lualatex + pdf2svg pipeline
└── api/
    ├── app.py          — FastAPI application
    ├── routes.py       — REST endpoints
    └── schemas.py      — Request/response models

contrib/qgraf/
├── models/             — QGRAF model files (qed.mod, qcd.mod, electroweak.mod, bsm.mod)
└── styles/             — feynman.sty (QGRAF output format)
```
