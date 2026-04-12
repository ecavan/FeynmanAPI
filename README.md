# FeynmanEngine

A Feynman diagram generator and tree-level amplitude calculator for particle physics. Give it a process like `e+ e- -> mu+ mu-` and get back enumerated diagrams as SVG/TikZ plus the symbolic spin-averaged |M|² expression.

**Built on proven HEP tooling:**
- [QGRAF](http://cfif.ist.utl.pt/~paulo/qgraf.html) — the industry-standard Feynman diagram enumerator used in professional NLO/NNLO calculations worldwide
- [SymPy](https://www.sympy.org/) — symbolic algebra including `GammaMatrix` for exact Dirac trace computation
- [TikZ-Feynman](https://ctan.org/pkg/tikz-feynman) — the standard LaTeX package for publication-quality Feynman diagrams
- [FastAPI](https://fastapi.tiangolo.com/) — modern async Python web framework
- [NetworkX](https://networkx.org/) — graph library used for topology classification

---

## What it does

- Enumerates Feynman diagrams for any process in QED, QCD, electroweak, or BSM theories using QGRAF
- Classifies diagram topologies: s-channel, t-channel, u-channel, triangle, box, self-energy, etc.
- Renders publication-quality SVG diagrams via TikZ-Feynman and LuaLaTeX
- Computes symbolic spin-averaged |M|² at tree level using exact Dirac traces via SymPy
- Serves a REST API and browser UI from a single FastAPI app
- Supports loop diagram generation (topology classification only; loop integrals are not computed)

### Amplitude coverage

| Process type | Example | Status |
|---|---|---|
| s-channel annihilation | e⁺e⁻ → μ⁺μ⁻ | exact |
| s-channel multi-mediator | e⁺e⁻ → μ⁺μ⁻ (EW: γ+Z+H) | exact |
| t-channel single diagram | e⁻μ⁻ → e⁻μ⁻ | exact |
| t+u channel (Møller, QCD qq→qq) | e⁻e⁻ → e⁻e⁻ | diagonal terms only* |
| s+t channel (Bhabha) | e⁺e⁻ → e⁺e⁻ | diagonal terms only* |
| BSM dark matter | e⁺e⁻ → χχ̄ via Z' | exact |
| Loop amplitudes | — | not computed |

\* Cross-topology interference requires non-factorizable 8-gamma traces; individual diagram contributions are exact.

---

## Local setup

**Requirements:** Python 3.11+, gfortran (to compile QGRAF), LuaLaTeX + pdf2svg (for SVG rendering)

```bash
git clone https://github.com/ecavan/FeynmanAPI.git
cd FeynmanAPI

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Build the QGRAF binary from the bundled source archive
feynman install-qgraf
```

### Install rendering tools (for SVG output)

macOS:
```bash
brew install --cask basictex && brew install pdf2svg
sudo tlmgr install tikz-feynman standalone
```

Ubuntu/Debian:
```bash
sudo apt-get install -y texlive-luatex texlive-pictures texlive-science pdf2svg
```

---

## Run the app

```bash
source .venv/bin/activate
uvicorn feynman_engine.api.app:app --reload
```

Open **http://localhost:8000** for the browser UI, or **http://localhost:8000/docs** for the API explorer.

### Quick API example

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"process": "e+ e- -> mu+ mu-", "theory": "QED", "loops": 0}'
```

---

## Supported theories

| Theory | Example processes |
|---|---|
| `QED` | e⁺e⁻→μ⁺μ⁻, e⁻μ⁻→e⁻μ⁻, e⁻e⁻→e⁻e⁻, Compton |
| `QCD` | uū→gg, ud→ud, uu→uu, 1-loop corrections |
| `EW` | e⁺e⁻→μ⁺μ⁻ (γ+Z+H), e⁺e⁻→τ⁺τ⁻, W/Z decay |
| `BSM` | e⁺e⁻→χχ̄ via Z' (add custom UFO models) |

---

## Deploy on Render.com

This project ships a `Dockerfile` that:
1. Compiles a Linux QGRAF binary from the bundled source archive
2. Installs Python, LuaLaTeX, TikZ-Feynman, and pdf2svg

Create a **Web Service** on [render.com](https://render.com) with:
- Runtime: `Docker`
- Health check: `/api/status`
- Start command auto-detected from the Dockerfile

---

## Run tests

```bash
source .venv/bin/activate
pytest
```

92 tests, no external services required (QGRAF binary must be built first).
