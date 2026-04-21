# FeynmanEngine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19673075.svg)](https://doi.org/10.5281/zenodo.19673075)
[![PyPI](https://img.shields.io/pypi/v/feynman-engine)](https://pypi.org/project/feynman-engine/)

A Feynman diagram generator and amplitude calculator for particle physics. Give it a process like `e+ e- -> mu+ mu-` and get back enumerated diagrams as SVG/TikZ, the symbolic spin-averaged |M|^2, integrated cross-sections, and --- for standard processes --- numerically evaluated 1-loop results via LoopTools.

**Built on proven HEP tooling:**
- [QGRAF](http://cfif.ist.utl.pt/~paulo/qgraf.html) --- the industry-standard Feynman diagram enumerator used in professional NLO/NNLO calculations worldwide
- [FORM](https://www.nikhef.nl/~form/) --- the standard symbolic algebra engine for high-energy physics trace calculations
- [LoopTools](https://www.feynarts.de/looptools/) --- the standard one-loop scalar/tensor integral library by Hahn & Perez-Victoria
- [SymPy](https://www.sympy.org/) --- symbolic algebra including `GammaMatrix` for exact Dirac trace computation
- [TikZ-Feynman](https://ctan.org/pkg/tikz-feynman) --- the standard LaTeX package for publication-quality Feynman diagrams
- [FastAPI](https://fastapi.tiangolo.com/) --- modern async Python web framework
- [SciPy](https://scipy.org/) --- numerical integration for cross-section calculations

## Citation guidance

If you use FeynmanEngine in research, cite the software itself and also cite the wrapped tools that are materially involved in your workflow. In particular:

- Foundational Feynman-diagram formalism: R. P. Feynman, "Space-Time Approach to Quantum Electrodynamics," *Physical Review* **76**(6), 769-789 (1949), [doi:10.1103/PhysRev.76.769](https://doi.org/10.1103/PhysRev.76.769)
- Passarino-Veltman reduction for 1-loop amplitudes: G. Passarino and M. J. G. Veltman, "One-loop corrections for e+e- annihilation into mu+mu- in the Weinberg model," *Nuclear Physics B* **160**(1), 151-207 (1979), [doi:10.1016/0550-3213(79)90234-7](https://doi.org/10.1016/0550-3213(79)90234-7)
- Practical one-loop techniques and electroweak radiative corrections: A. Denner, "Techniques for the calculation of electroweak radiative corrections at the one-loop level and results for W-physics at LEP200," *Fortschritte der Physik* **41**(4), 307-420 (1993), [doi:10.1002/prop.2190410402](https://doi.org/10.1002/prop.2190410402)
- QGRAF for diagram generation: P. Nogueira, "Automatic Feynman graph generation," *Journal of Computational Physics* **105**(2), 279-289 (1993), [doi:10.1006/jcph.1993.1074](https://doi.org/10.1006/jcph.1993.1074)
- FORM for symbolic trace/algebra workflows: J. A. M. Vermaseren, "New features of FORM," arXiv:[math-ph/0010025](https://arxiv.org/abs/math-ph/0010025) (2000)
- LoopTools for numerical 1-loop evaluation: T. Hahn and M. Perez-Victoria, "Automatized one-loop calculations in four and D dimensions," *Computer Physics Communications* **118**(2-5), 153-165 (1999), [doi:10.1016/S0010-4655(98)00173-8](https://doi.org/10.1016/S0010-4655(98)00173-8)
- RAMBO for flat multiparticle phase-space generation: R. Kleiss, W. J. Stirling, and S. D. Ellis, "A new Monte Carlo treatment of multiparticle phase space at high energies," *Computer Physics Communications* **40**(2-3), 359-373 (1986), [doi:10.1016/0010-4655(86)90119-0](https://doi.org/10.1016/0010-4655(86)90119-0)

Classic background textbooks that fit the scope of this project include M. E. Peskin and D. V. Schroeder, *An Introduction to Quantum Field Theory* (1995), and R. K. Ellis, W. J. Stirling, and B. R. Webber, *QCD and Collider Physics* (1996).

---

## What it does

### Diagram generation and rendering
- Enumerates Feynman diagrams for any process in QED, QCD, QCD+QED (mixed), electroweak, or BSM theories using QGRAF
- Classifies diagram topologies: s-channel, t-channel, u-channel, triangle, box, self-energy, etc.
- Renders publication-quality SVG diagrams via TikZ-Feynman and LuaLaTeX

### Tree-level amplitudes
- Computes symbolic spin-averaged |M|^2 at tree level via three backends:
  1. **FORM symbolic traces** --- handles QCD color algebra (SU(3) structure constants, physical polarization sums for external gluons, 3-gluon + 4-gluon contact vertices)
  2. **SymPy Dirac traces** --- exact gamma-matrix traces for QED/EW/BSM
  3. **Curated formulas** --- 54 verified results from Combridge, Peskin/Schroeder, Ellis/Stirling/Webber, Schwartz, Grozin, Gunion/Haber/Kane/Dawson
- All QCD 2->2 cross-sections verified against PYTHIA8/Combridge et al. (1977) --- ratio 1.0000 at all kinematic points

### QCD with full color algebra
- SU(3) color factor matrices for all 2->2 QCD channels (qq->gg, qg->qg, gg->gg, qq->qq)
- Kleiss-Stirling physical polarization sums for external gluons --- avoids gauge artifacts from unphysical longitudinal modes
- gg->gg computed via 3-gluon vertex exchange (s, t, u channels) + 4-gluon contact vertex decomposed into color-ordered amplitudes
- Color factors verified numerically via explicit Gell-Mann matrix traces

### Full Feynman integral expressions
- Every tree-level and 1-loop result includes the textbook Feynman integral: external spinors/polarizations, vertex factors, propagators, and momentum-conservation delta functions
- QCD vertices show proper Feynman rules: 3-gluon `g_s f^{abc}[g^{mu nu}(k1-k2)^rho + cyclic]`, 4-gluon contact, and quark-gluon `-ig_s T^a gamma^mu`
- Gluon propagators show color delta: `-i delta^{ab} g^{mu nu} / k^2`
- Each external boson gets a distinct Lorentz index (mu, nu, rho, sigma, ...)

### Cross-sections
- 2->2 differential and total cross-sections with full massive kinematics (Kallen function, threshold detection) via SciPy adaptive quadrature
- 2->N flat Monte Carlo integration via RAMBO phase-space algorithm (Kleiss, Stirling, Ellis 1986)
- **Vegas adaptive MC** for 2->N processes with sharp kinematic features (t-channel poles, resonances) --- importance sampling learns where |M|^2 is largest and concentrates samples there, converging much faster than flat RAMBO for peaked integrands
- Validated: sigma(e+e- -> mu+mu-) at sqrt(s)=10 GeV matches the analytic 4*pi*alpha^2/(3s) ~ 865 pb to <1% across all three integration methods (scipy.quad, RAMBO, Vegas)

### 1-loop infrastructure
- Full Passarino-Veltman tensor integral basis through 4-point rank-2 (A0 through D33)
- **Symbolic PV decomposition**: every 1-loop diagram is expressed as a linear combination of PV scalar integrals with fully symbolic coefficients in (s, t, u, masses, couplings) --- not just numbers
- **Analytic closed-form PV integrals** (no LoopTools required): A₀, B₀ (all 6 special cases + general), B₁, B₀₀, C₀ (Li₂ closed form for one-mass triangle, Feynman parameter dblquad for general spacelike), D₀ (massless box). Pure Python/SymPy/mpmath --- works without any external Fortran library
- Automatic PV tensor reduction via FORM
- LoopTools bridge (ctypes) for numerical evaluation of scalar and tensor integrals
- Individual PV integral values displayed alongside the symbolic formula when LoopTools is available
- 20 curated 1-loop results: self-energies, vertex corrections, VP corrections, box diagrams, ghost sector, running couplings, and Higgs loop-induced decays
- MS-bar running couplings: alpha(mu^2) and alpha_s(mu^2)

### API and frontend
- REST API and browser UI from a single FastAPI app
- Swagger documentation at `/docs`

---

## Amplitude coverage

### Tree level

| Process type | Examples | Backend | Status |
|---|---|---|---|
| QED 2->2 | e+e- -> mu+mu-, Bhabha, Moller, Compton | SymPy / curated | exact |
| QED 2->3 bremsstrahlung | e+e- -> mu+mu- gamma (5 diagrams) | FORM | exact |
| QCD qq <-> gg | u u~ -> g g, g g -> u u~ | FORM + SU(3) color | exact |
| QCD qg -> qg | u g -> u g (all flavors) | FORM + SU(3) color | exact |
| QCD gg -> gg | g g -> g g (3-gluon + 4-gluon contact) | FORM + SU(3) color | exact |
| QCD qq -> qq | u d -> u d, u u -> u u | curated | exact |
| QCD+QED mixed | qq -> gamma gamma, qq -> gamma g, qg -> q gamma | curated | exact |
| EW multi-mediator | e+e- -> mu+mu- (gamma + Z + H) | SymPy | exact |
| EW Higgsstrahlung | e+e- -> ZH, tau+tau- -> ZH | curated | exact |
| EW pair production | e+e- -> W+W-, e+e- -> ZZ | curated | exact |
| EW Drell-Yan | qq -> l+l-, ud -> e nu | curated | exact |
| EW decays | Z -> ff, W -> lnu, W -> qq, H -> ff, H -> WW/ZZ, t -> bW | curated | exact |
| EW other | muon decay (mu -> e nu nu), e nu -> mu nu | curated | exact |
| BSM dark matter | e+e- -> chi chi~ via Z' | SymPy | exact |

54 curated tree-level amplitudes covering QED, QCD, electroweak, and mixed QCD+QED processes.

### 1-loop (curated, via LoopTools)

20 curated 1-loop results expressed in terms of Passarino-Veltman scalar integrals:

| Category | Observable | Expression | Reference |
|---|---|---|---|
| **QED self-energies** | Photon self-energy | Sigma_T = (alpha/pi)[2A0 - (4m^2 - k^2)B0] | Denner 1993 eq. C.2 |
| | Electron self-energy | Sigma = (alpha/4pi)[2A0 + (2m^2 - p^2)B0] | P&S eq. 7.27 |
| | Running alpha(q^2) | alpha/(1 - Pi_hat(q^2)) | |
| **QED vertex** | Vertex form factor | delta_F1 = (alpha/2pi)[-B0 + (4m^2 - q^2/2)/q^2 * C0] | Denner 1993 |
| | Schwinger AMM | a_e = alpha/(2pi) ~ 1.1614e-3 | analytic |
| **QED 2->2 VP** | e+e- -> mu+mu- VP | delta\|M\|^2 = \|M\|^2_tree * (-2Pi/s) | P&S Ch.7 |
| | Compton VP | s- and u-channel propagator corrections | |
| | Bhabha VP | s- and t-channel propagator corrections | Actis et al. EPJC 66 |
| | Moller VP | t- and u-channel propagator corrections | |
| **QED box** | e+e- -> mu+mu- box | c_D0 = -8*alpha^2*t*u | correct Dirac trace |
| **QCD self-energies** | Quark self-energy | Sigma_q = (alpha_s C_F/4pi)[A0 + (p^2+m^2)B0] | Muta 1998 |
| | Gluon self-energy | beta_0 contribution via B0(k^2;0,0) | |
| | Ghost self-energy | Sigma_ghost = (alpha_s C_A/16pi) p^2 B0 | Pascual & Tarrach |
| | Running alpha_s(q^2) | alpha_s/(1 + alpha_s*beta_0/(4pi)*B0) | Gross & Wilczek 1973 |
| **QCD vertex** | Quark-gluon vertex | delta_V1 = (alpha_s C_F/2pi)[-B0 + ...C0] | |
| | Ghost-gluon vertex | Z_tilde_1 = 1 in Landau gauge (Taylor) | Taylor NPB 33 (1971) |
| | qq -> gg VP | gluon propagator correction at scale s | ESW Ch.7 |
| **Higgs decays** | H -> gg (top loop) | \|M\|^2 = alpha_s^2 m_H^4/(8pi^2 v^2) | Spira et al. NPB 453 |
| | H -> gamma gamma (W+top) | \|M\|^2 propto \|-7 + 16/9\|^2 | Djouadi Phys.Rep. 457 |
| | H -> Z gamma (W+top) | phase-space * form factor | Bergstrom & Hulth 1985 |

Scalar forms verified numerically: at k^2=4 GeV^2, m^2=1 GeV^2, LoopTools gives Sigma_T = 2.000 matching Denner exactly.

### UV renormalization (MS-bar)

- delta_Z3 (photon field strength counterterm)
- delta_m_e, delta_m_q (mass counterterms)
- delta_Z3^g (gluon field strength counterterm)
- Running couplings: alpha(mu^2) and alpha_s(mu^2)
- Renormalized observables: photon self-energy, vertex form factor

### Tensor integral infrastructure

The full Passarino-Veltman tensor integral basis is implemented and available via the LoopTools bridge:

| Rank | Integrals |
|---|---|
| 1-point | A0 |
| 2-point scalar | B0 |
| 2-point tensor | B1, B00, B11 |
| 3-point scalar | C0 |
| 3-point tensor | C1, C2, C00, C11, C12, C22 |
| 4-point scalar | D0 |
| 4-point tensor (rank-1) | D1, D2, D3 |
| 4-point tensor (rank-2) | D00, D11, D12, D13, D22, D23, D33 |

Each integral has a frozen dataclass with `latex()` and `evaluate()` methods, importable from `feynman_engine.amplitudes`. The `evaluate()` method uses the analytic closed-form formulas (no LoopTools required). Automatic PV tensor reduction via FORM is also available for reducing arbitrary 1-loop integrals to the scalar basis.

### Analytic closed-form scalar integrals

All standard 1-loop PV scalar integrals are available as pure-Python analytic functions via `feynman_engine.amplitudes.analytic_integrals`. No LoopTools or Fortran compiler required.

| Integral | Arguments | Method | Coverage |
|---|---|---|---|
| A₀(m²) | tadpole | Exact formula | All masses |
| B₀(p²; m₁², m₂²) | bubble | 6 special-case formulas + Feynman parameter quad | All kinematics |
| B₁(p²; m₁², m₂²) | tensor bubble | PV reduction → A₀ + B₀ | All kinematics (p²≠0) |
| B₀₀(p²; m₁², m₂²) | tensor bubble | PV reduction → A₀ + B₀ + B₁ | All kinematics (p²≠0) |
| C₀(p₁²,p₂²,p₁₂²; m₁²,m₂²,m₃²) | triangle | Li₂ closed form (one-mass), dblquad (general spacelike) | Spacelike + one-mass timelike |
| D₀(pᵢ²,s,t; mᵢ²) | box | Massless box formula | Fully massless only |

All numeric results validated against LoopTools to relative tolerance < 10⁻⁸. Both symbolic mode (SymPy expressions with `Delta_UV`, `log`, `polylog`) and numeric mode (complex numbers with `Delta_UV=0`) are supported.

References: 't Hooft & Veltman (1979), Denner (1993), Ellis & Zanderighi (2008), Passarino & Veltman (1979).

---

## Supported theories

| Theory | Particles | Example processes |
|---|---|---|
| `QED` | e, mu, tau, gamma | e+e- -> mu+mu-, Bhabha, Moller, Compton, bremsstrahlung |
| `QCD` | u, d, s, c, b, t, g (+ ghosts) | u u~ -> g g, g g -> g g, u g -> u g, multi-jet |
| `QCDQED` | u, d, s, c, b, t, g, gamma | u u~ -> gamma g, u g -> u gamma, gamma g -> u u~ |
| `EW` | all SM fermions, gamma, Z, W+/-, H | e+e- -> W+W-, Z/W/H decays, t -> b W+, Drell-Yan |
| `BSM` | chi (dark matter), Z' | e+e- -> chi chi~ via Z' |

---

## Installation

### Option 1: Docker (easiest, everything bundled)

If you want the smoothest install experience, use Docker. The Docker image includes the FastAPI app, packaged frontend, QGRAF, FORM, LoopTools, and the LaTeX/SVG rendering stack.

Build the image from this repository:

```bash
git clone https://github.com/ecavan/FeynmanAPI.git
cd FeynmanAPI

docker build -t feynman-engine .
docker run --rm -p 8000:10000 feynman-engine
```

Then open **http://localhost:8000** for the browser UI, or **http://localhost:8000/docs** for the API explorer.

### Option 2: PyPI / pip install (lightest Python install)

Use this if you want the Python package and browser UI first, and are okay installing heavier native tools only if you need them.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install feynman-engine
```

Start the app:

```bash
feynman serve --host 127.0.0.1 --port 8000
```

Then open **http://localhost:8000** for the browser UI, or **http://localhost:8000/docs** for the API explorer.

Optional post-install native tools:

```bash
# Recommended one-command native setup
feynman setup

# If you only want the recommended QGRAF + FORM setup
feynman setup --skip-looptools

# Inspect what is installed and where it was found
feynman doctor
```

Under the hood, `feynman setup` runs the individual installers for:

- `feynman install-qgraf` for diagram enumeration from the bundled QGRAF source archive
- `feynman install-form` for FORM-based symbolic traces, including QCD color algebra and some higher-complexity amplitudes
- `feynman install-looptools` for numerical 1-loop evaluation

This split keeps `pip install` lightweight and reliable, while still giving users a path to the heavier compiled dependencies when they need them.

### Option 3: Local source setup (development)

**Requirements:** Python 3.11+, C compiler (for FORM), gfortran (for QGRAF and LoopTools), LuaLaTeX + pdf2svg (optional, for SVG rendering)

```bash
git clone https://github.com/ecavan/FeynmanAPI.git
cd FeynmanAPI

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Build the bundled native tools in one go
feynman setup
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

If you installed from PyPI or from source, the browser UI is bundled with the package and served by the same FastAPI process as the API. You do not need to start a separate frontend dev server.

```bash
feynman serve --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000** for the browser UI, or **http://localhost:8000/docs** for the API explorer.

For local development from a clone, `uvicorn feynman_engine.api.app:app --reload` 

### Quick API examples

```bash
# Tree-level amplitude
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"process": "e+ e- -> mu+ mu-", "theory": "QED", "loops": 0}'

# Cross-section at sqrt(s) = 10 GeV
curl "http://localhost:8000/api/amplitude/cross-section?process=e%2B+e-+-%3E+mu%2B+mu-&theory=QED&sqrt_s=10.0"

# QCD cross-section
curl "http://localhost:8000/api/amplitude/cross-section?process=g+g+-%3E+g+g&theory=QCD&sqrt_s=100.0"

# Vegas adaptive MC for 2->3 process (converges faster for peaked integrands)
curl "http://localhost:8000/api/amplitude/cross-section?process=e%2B+e-+-%3E+mu%2B+mu-+gamma&theory=QED&sqrt_s=10.0&method=vegas&min_invariant_mass=1.0"

# 1-loop PV decomposition (symbolic coefficients + individual integrals)
curl "http://localhost:8000/api/amplitude/loop-pv?process=e%2B+e-+-%3E+mu%2B+mu-&theory=QED"

# 1-loop observable with numerical PV integral values (requires LoopTools)
curl "http://localhost:8000/api/amplitude/loop-evaluate?observable=photon_selfenergy&q_sq=4.0&m_sq=1.0"

# Analytic PV integral evaluation (NO LoopTools required)
curl "http://localhost:8000/api/amplitude/loop-analytic?integral_type=B0&p_sq=4.0&m1_sq=1.0&m2_sq=1.0"

# Running coupling
curl "http://localhost:8000/api/amplitude/running-coupling?coupling=alpha&q_sq=100.0"
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| POST | `/api/generate` | Enumerate diagrams + amplitude for any process |
| GET | `/api/amplitude/cross-section` | sigma for 2->2 (quad) and 2->N (RAMBO or Vegas) |
| GET | `/api/amplitude/loop-pv` | Symbolic PV decomposition with individual term coefficients |
| GET | `/api/amplitude/loop-evaluate` | Numerically evaluate 1-loop observable via LoopTools |
| GET | `/api/amplitude/loop-analytic` | Evaluate PV scalar integral analytically (no LoopTools required) |
| GET | `/api/amplitude/loop-curated` | List all 20 curated 1-loop results |
| GET | `/api/amplitude/running-coupling` | alpha(mu^2) or alpha_s(mu^2) at given scale |
| GET | `/api/amplitude/renorm-status` | UV counterterms and their values |
| GET | `/api/amplitude/renorm-selfenergy` | Renormalized photon self-energy |
| GET | `/api/theories` | List available theories |
| GET | `/api/theories/{theory}/particles` | Particle registry for a theory |
| GET | `/api/status` | Backend and dependency status |


---


## Architecture

```
feynman_engine/
  amplitudes/
    form_trace.py             # FORM-based traces: QCD gluon vertices, 2->3 bremsstrahlung
    symbolic.py               # SymPy Dirac-trace backend (QED/EW/BSM)
    approximate.py            # Pointwise evaluation via QGRAF diagrams
    loop.py                   # PV topology classification + scalar integral types
    analytic_integrals.py     # Closed-form A₀, B₀, C₀, D₀ (no LoopTools)
    loop_curated.py           # 20 curated 1-loop results
    looptools_bridge.py       # ctypes wrapper to Fortran LoopTools
    loop_tensor_reduction.py  # Automatic FORM-based PV reduction
    color.py                  # SU(3) color algebra matrices
    cross_section.py          # 2->2 cross-section via SciPy quadrature
    phase_space.py            # RAMBO 2->N phase-space generation
    renorm.py                 # MS-bar renormalization + running couplings
  physics/
    amplitude.py              # Amplitude router: FORM -> SymPy -> curated -> approximate
    theories/                 # QED, QCD, QCDQED, EW, BSM particle/vertex defs
  core/
    generator.py              # QGRAF wrapper
    parser.py                 # QGRAF output parser
    models.py                 # Diagram, Edge, Vertex dataclasses
    topology.py               # Topology classification (s/t/u/triangle/box)
  api/
    routes.py                 # FastAPI endpoints
    schemas.py                # Pydantic request/response models
  render/
    tikz.py                   # TikZ-Feynman LaTeX generation
    compiler.py               # LuaLaTeX -> PDF -> SVG pipeline
  form.py                     # FORM build helper
  looptools.py                # LoopTools build helper
  qgraf.py                    # QGRAF build helper
contrib/qgraf/models/
  qed.mod                     # Pure QED (leptons + photon)
  qcd.mod                     # Pure QCD (quarks + gluons + ghosts)
  qcdqed.mod                  # QCD + photon for mixed processes
  electroweak.mod              # Full SM (all fermions + gamma/Z/W/H)
  bsm.mod                     # Z' + dark matter
```

---

## Roadmap

### What you can do today

- Full tree-level |M|^2 for any QED, QCD, QCD+QED, electroweak, or BSM process, with cross-sections
- 1-loop PV decomposition with symbolic coefficients for any process QGRAF can generate
- Analytic closed-form evaluation of all PV scalar integrals (A₀, B₀, C₀, D₀) --- pure Python, no Fortran required
- Numerical evaluation via LoopTools for the full tensor integral basis
- IR-safe observables: Schwinger anomalous magnetic moment alpha/(2pi), running couplings alpha(Q^2) and alpha_s(Q^2), vacuum polarization, form factors
- 2->3 bremsstrahlung matrix elements at tree level (e.g. e+e- -> mu+mu- gamma)
- MS-bar counterterms and running couplings for curated processes

### Parton distribution functions

Needed for hadron-collider cross-sections (pp -> X). The engine computes partonic cross-sections (e.g. gg -> gg) but cannot yet convolve them with proton structure for physical pp predictions at LHC energies. Integrating [LHAPDF](https://lhapdf.hepforge.org/) would enable Drell-Yan, dijet, and Higgs production via gluon fusion predictions.

### Real-emission IR subtraction

1-loop virtual corrections contain infrared divergences that cancel against real-emission diagrams (KLN theorem). The real-emission matrix elements are already computed at tree level --- the missing piece is an IR subtraction scheme (Catani-Seymour dipoles or FKS slicing) to combine virtual + real corrections into a finite, physical NLO cross-section.

### Automatic renormalization for arbitrary processes

The curated 1-loop results already apply correct MS-bar renormalization. Extending this to arbitrary processes requires generating counterterm diagrams and applying scheme-dependent subtractions automatically. The PV integral structure produced by the engine is scheme-independent and ready for users to apply their own renormalization.
