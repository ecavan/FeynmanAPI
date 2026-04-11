# Feynman Engine

QGRAF-first Feynman diagram generation for particle physics.

Give the app a process like `e+ e- -> mu+ mu-` and it returns enumerated diagrams as TikZ, SVG/PDF, and structured JSON through a FastAPI backend plus a browser UI.

## Current Direction

This repository now treats QGRAF as the single source of truth for diagram enumeration.

- No pure-Python fallback path
- QGRAF is required for tree-level and loop-level generation
- LaTeX rendering is handled with `lualatex`
- SVG export is handled with `pdf2svg`
- The long-term roadmap is to add a QGRAF-compatible amplitude stack that feels as easy to use as FeynCalc, but without requiring Mathematica

## What It Already Does

- Enumerates Feynman diagrams with QGRAF
- Parses QGRAF output into structured diagram models
- Classifies topologies such as `s-channel`, `t-channel`, and `box`
- Renders diagrams to TikZ, SVG, and PDF
- Serves a web UI and REST API from one FastAPI app
- Includes built-in QED, QCD, EW, and BSM theory definitions
- Includes a small symbolic amplitude module for a few hand-derived benchmark processes

## Supported Theories

| Theory | Example |
|---|---|
| `QED` | `e+ e- -> mu+ mu-` |
| `QCD` | `u u~ -> g g` |
| `EW` | `e+ e- -> W+ W-` |
| `BSM` | `e+ e- -> chi chi~` |

## Requirements

Core requirements:

- Python 3.11+
- QGRAF executable, or the bundled QGRAF source archive plus `gfortran`

Optional rendering requirements:

- `lualatex` for PDF/SVG rendering
- `pdf2svg` for SVG rendering

Important packaging note:

- This repo now includes the `qgraf-3.6.10.tgz` source archive and a build helper.
- Docker/Render builds a Linux `qgraf` binary from that source automatically.
- Local installs can compile QGRAF from the bundled source with `feynman install-qgraf`.
- If no `qgraf` binary is present, the app will try to auto-build one on first use from the bundled source archive.
- You still need a Fortran compiler available locally for native builds.

## Local Setup

```bash
git clone <repo>
cd FeynmanAPI

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Install Modes

### Core Install

Use this if you want:

- QGRAF-based diagram enumeration
- TikZ output
- the API and browser UI
- symbolic work that does not require SVG/PDF/PNG rendering

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
feynman install-qgraf
```

### Core + Rendering Install

Use this if you also want:

- SVG output
- PDF output
- PNG output

After the core install, add the OS-level rendering tools below.

### Install QGRAF

Download the QGRAF source from the official page:

- [QGRAF official page](http://cfif.ist.utl.pt/~paulo/qgraf.html)

Compile it and place the executable at `bin/qgraf`:

```bash
mkdir -p bin
gfortran -O2 -o bin/qgraf qgraf-X.X.f
chmod +x bin/qgraf
```

Or use the bundled source archive and helper:

```bash
source .venv/bin/activate
feynman install-qgraf
```

If `gfortran` is installed, starting the app or generating a process will also auto-build `qgraf` when needed.

### One-Command Rendering Installs

macOS:

```bash
brew install --cask basictex && brew install pdf2svg
```

Then install the TikZ-Feynman TeX package:

```bash
sudo tlmgr install tikz-feynman
```

Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y texlive-luatex texlive-pictures texlive-latex-extra texlive-science pdf2svg
```

### Rendering Notes

- `tikz` output does not require `lualatex` or `pdf2svg`.
- `svg`, `pdf`, and `png` output do require the rendering tools above.
- If you want the lightest local setup, start with TikZ-only and add rendering later.

### Verify Setup

```bash
./setup.sh
```

The setup script now fails fast if `bin/qgraf` is missing.
If the bundled source archive is present and `gfortran` is installed, it will build `qgraf` for you automatically.

## Running The App

Browser UI:

```bash
source .venv/bin/activate
uvicorn feynman_engine.api.app:app --reload
```

Then open:

- `http://localhost:8000/`
- `http://localhost:8000/docs`
- `http://localhost:8000/api/status`

CLI:

```bash
source .venv/bin/activate
python main.py generate "e+ e- -> mu+ mu-" --theory QED --loops 0
```

## REST API

```bash
curl http://localhost:8000/api/status

curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"process":"e+ e- -> mu+ mu-","theory":"QED","loops":0,"output_format":"tikz"}'
```

## Render Deployment

This project should be deployed on Render as a `Web Service` using Docker.

Recommended Render settings:

- Runtime: `Docker`
- Dockerfile path: `./Dockerfile`
- Health check path: `/api/status`
- Auto deploy: enabled

The included [Dockerfile](Dockerfile) now:

- builds a Linux `qgraf` binary from `qgraf-3.6.10.tgz`
- installs Python, LaTeX, and `pdf2svg` automatically inside the container
- starts the FastAPI app with `uvicorn`

Start command inside the container:

```bash
uvicorn feynman_engine.api.app:app --host 0.0.0.0 --port $PORT
```

Important:

- Render no longer depends on the checked-in macOS binary.
- The container build compiles a fresh Linux `qgraf` from the bundled source archive.
- Render/Docker also installs the LaTeX rendering stack automatically, so deployed SVG/PDF rendering works without extra user setup.
- If you remove the bundled source archive, Docker builds will stop producing `qgraf`.

## Architecture

```text
frontend/
  Browser UI

feynman_engine/api/
  FastAPI routes and schemas

feynman_engine/core/
  QGRAF execution, parsing, topology classification, data models

feynman_engine/physics/
  Theory registry, process translation, amplitude prototypes

feynman_engine/render/
  TikZ generation and TeX/SVG compilation

contrib/qgraf/
  QGRAF model files and style files
```

## Researcher-Ready Priorities

The most important remaining steps before broad research use are:

1. Ship or build platform-specific QGRAF binaries for macOS and Linux.
2. Decide whether to keep source-based builds only, or also publish prebuilt platform-specific binaries.
3. Replace the current handwritten amplitude lookup module with a general tree-level amplitude backend.
4. Add more validation around kinematics, couplings, and theory/model metadata.
5. Add provenance in the UI so users can see the backend, model, and assumptions used for each result.
