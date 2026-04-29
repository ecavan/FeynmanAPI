# FeynmanEngine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19673075.svg)](https://doi.org/10.5281/zenodo.19673075)
[![PyPI](https://img.shields.io/pypi/v/feynman-engine)](https://pypi.org/project/feynman-engine/)

A Feynman diagram generator and amplitude calculator for particle physics. Give it a process like `e+ e- -> mu+ mu-` and get back enumerated diagrams as SVG/TikZ, the symbolic spin-averaged |M|^2, integrated cross-sections (LO and NLO), and --- for standard processes --- numerically evaluated 1-loop results via LoopTools.

**Built on proven HEP tooling:**
- [QGRAF](http://cfif.ist.utl.pt/~paulo/qgraf.html) --- the industry-standard Feynman diagram enumerator used in professional NLO/NNLO calculations worldwide
- [FORM](https://www.nikhef.nl/~form/) --- the standard symbolic algebra engine for high-energy physics trace calculations
- [LoopTools](https://www.feynarts.de/looptools/) --- the standard one-loop scalar/tensor integral library by Hahn & Perez-Victoria
- [OpenLoops 2](https://openloops.hepforge.org/) --- automated tree + one-loop amplitudes for arbitrary SM processes (Buccioni, Lang, Lindert, Maierhöfer, Pozzorini, Zhang, Zoller)
- [LHAPDF](https://lhapdf.hepforge.org/) --- the standard parton distribution function library for hadron-collider predictions (CT18, NNPDF, MSHT, ...)
- [SymPy](https://www.sympy.org/) --- symbolic algebra including `GammaMatrix` for exact Dirac trace computation
- [TikZ-Feynman](https://ctan.org/pkg/tikz-feynman) --- the standard LaTeX package for publication-quality Feynman diagrams
- [FastAPI](https://fastapi.tiangolo.com/) --- modern async Python web framework
- [SciPy](https://scipy.org/) --- numerical integration for cross-section calculations

## Citations

If you use FeynmanEngine in research, cite the software itself and also cite the wrapped tools that are materially involved in your workflow. In particular:

- Foundational Feynman-diagram formalism: R. P. Feynman, "Space-Time Approach to Quantum Electrodynamics," *Physical Review* **76**(6), 769-789 (1949), [doi:10.1103/PhysRev.76.769](https://doi.org/10.1103/PhysRev.76.769)
- Passarino-Veltman reduction for 1-loop amplitudes: G. Passarino and M. J. G. Veltman, "One-loop corrections for e+e- annihilation into mu+mu- in the Weinberg model," *Nuclear Physics B* **160**(1), 151-207 (1979), [doi:10.1016/0550-3213(79)90234-7](https://doi.org/10.1016/0550-3213(79)90234-7)
- Practical one-loop techniques and electroweak radiative corrections: A. Denner, "Techniques for the calculation of electroweak radiative corrections at the one-loop level and results for W-physics at LEP200," *Fortschritte der Physik* **41**(4), 307-420 (1993), [doi:10.1002/prop.2190410402](https://doi.org/10.1002/prop.2190410402)
- QGRAF for diagram generation: P. Nogueira, "Automatic Feynman graph generation," *Journal of Computational Physics* **105**(2), 279-289 (1993), [doi:10.1006/jcph.1993.1074](https://doi.org/10.1006/jcph.1993.1074)
- FORM for symbolic trace/algebra workflows: J. A. M. Vermaseren, "New features of FORM," arXiv:[math-ph/0010025](https://arxiv.org/abs/math-ph/0010025) (2000)
- LoopTools for numerical 1-loop evaluation: T. Hahn and M. Perez-Victoria, "Automatized one-loop calculations in four and D dimensions," *Computer Physics Communications* **118**(2-5), 153-165 (1999), [doi:10.1016/S0010-4655(98)00173-8](https://doi.org/10.1016/S0010-4655(98)00173-8)
- OpenLoops 2 for generic NLO virtual amplitudes (when used as the backend): F. Buccioni, J.-N. Lang, J. M. Lindert, P. Maierhöfer, S. Pozzorini, H. Zhang, and M. F. Zoller, "OpenLoops 2," *European Physical Journal C* **79**, 866 (2019), arXiv:[1907.13071](https://arxiv.org/abs/1907.13071), [doi:10.1140/epjc/s10052-019-7306-2](https://doi.org/10.1140/epjc/s10052-019-7306-2)
- Collier (used internally by OpenLoops for tensor reduction): A. Denner, S. Dittmaier, and L. Hofer, "COLLIER: a fortran-based Complex One-Loop LIbrary in Extended Regularizations," *Computer Physics Communications* **212**, 220-238 (2017), [doi:10.1016/j.cpc.2016.10.013](https://doi.org/10.1016/j.cpc.2016.10.013)
- OneLOop (used internally by OpenLoops for scalar integrals): A. van Hameren, "OneLOop: For the evaluation of one-loop scalar functions," *Computer Physics Communications* **182**(11), 2427-2438 (2011), arXiv:[1007.4716](https://arxiv.org/abs/1007.4716)
- CutTools (used internally by OpenLoops for OPP reduction): G. Ossola, C. G. Papadopoulos, and R. Pittau, "CutTools: a program implementing the OPP reduction method to compute one-loop amplitudes," *JHEP* **0803**, 042 (2008), arXiv:[0711.3596](https://arxiv.org/abs/0711.3596)
- RAMBO for flat multiparticle phase-space generation: R. Kleiss, W. J. Stirling, and S. D. Ellis, "A new Monte Carlo treatment of multiparticle phase space at high energies," *Computer Physics Communications* **40**(2-3), 359-373 (1986), [doi:10.1016/0010-4655(86)90119-0](https://doi.org/10.1016/0010-4655(86)90119-0)
- Catani-Seymour dipole subtraction for NLO calculations: S. Catani and M. H. Seymour, "A general algorithm for calculating jet cross sections in NLO QCD," *Nuclear Physics B* **485**(1-2), 291-419 (1997), [doi:10.1016/S0550-3213(96)00589-5](https://doi.org/10.1016/S0550-3213(96)00589-5)
- LHAPDF for parton distribution functions (when used as the optional backend): A. Buckley et al., "LHAPDF6: parton density access in the LHC precision era," *European Physical Journal C* **75**, 132 (2015), [doi:10.1140/epjc/s10052-015-3318-8](https://doi.org/10.1140/epjc/s10052-015-3318-8)
- ggH cross-section formula in heavy-top limit: M. Spira, A. Djouadi, D. Graudenz, P. M. Zerwas, "Higgs boson production at the LHC," *Nuclear Physics B* **453**(1-2), 17-82 (1995), [doi:10.1016/0550-3213(95)00379-7](https://doi.org/10.1016/0550-3213(95)00379-7)
- pp→DY NLO K-factor benchmark: C. Anastasiou, L. J. Dixon, K. Melnikov, F. Petriello, "High precision QCD at hadron colliders: electroweak gauge boson rapidity distributions at NNLO," *Physical Review D* **69**, 094008 (2004), arXiv:[hep-ph/0312266](https://arxiv.org/abs/hep-ph/0312266)
- EW Sudakov LL+NLL framework (V2.7 generic-EW NLO): S. Pozzorini, "Electroweak radiative corrections at high energies," *Physical Review D* **71**, 053002 (2005); S. Catani and L. Comelli, *Phys. Lett. B* **446**, 278 (1999)
- Vegas adaptive Monte Carlo (when `vegas` extra is installed): G. P. Lepage, "A new algorithm for adaptive multidimensional integration," *Journal of Computational Physics* **27**, 192 (1978)

Classic background textbooks that fit the scope of this project include M. E. Peskin and D. V. Schroeder, *An Introduction to Quantum Field Theory* (1995), and R. K. Ellis, W. J. Stirling, and B. R. Webber, *QCD and Collider Physics* (1996).

---

## Capabilities

FeynmanEngine takes a process string and produces, end-to-end:

- **Feynman diagrams** — enumerated by QGRAF, classified by topology (s/t/u-channel, triangle, box, self-energy), rendered to SVG/TikZ via TikZ-Feynman.
- **Spin-averaged tree amplitudes** — symbolic |M|² via three backends, in priority order: 54 textbook-verified curated tree formulas + 35 curated 1-loop entries → FORM-traced amplitudes with full SU(3) color algebra → SymPy γ-matrix traces. Approximate single-point fallback is honestly flagged when no symbolic |M|² is available.
- **Cross-sections** — 2→2 deterministic (scipy.quad), 2→N Monte Carlo (RAMBO, Vegas), with full massive Källén kinematics and t-channel pole handling.
- **Decay widths** — 2-body Γ(X → 1+2) for every Z, W, H, top channel via `/api/amplitude/decay-width`, with PDG comparison and `pct_off_pdg` self-disclosure baked into the response.
- **NLO** — five layers: exact analytic K-factor for QED `e+e-→ff'̄` (validated against KLN, K = 1+3α/(4π) to machine precision); universal QED NLO via charge-correlator formula for arbitrary QED processes (Σ Q² × 3/4); EW Sudakov LL+NLL for high-energy LHC tails (`δ_EW = -(α/(4π sin²θ_W)) Σ T_eff² × {L²+3L}`, L = log(s/M_W²)); tabulated NLO/LO ratios from LHC HWG YR4 + ATLAS/CMS for the major LHC channels (ggH, tt̄, WW, ZZ, ZH, DY, VBF, …); first-principles Catani-Seymour subtraction for Drell-Yan (`K(pp→DY @ 13 TeV) = 1.190` vs YR4 1.21, 1.6% off).
- **1-loop infrastructure** — full Passarino-Veltman tensor basis through D₃₃, symbolic PV decomposition, analytic closed-form scalar integrals (A₀, B₀, C₀, D₀ — pure Python, no Fortran required), MS-bar UV renormalization, MS-bar running α_em(μ²) and α_s(μ²).
- **Hadronic σ via PDF convolution** — built-in LO PDF (factor-of-2-3 accuracy, no deps) + LHAPDF auto-discovery for percent-level precision (CT18LO default). Specialized fast paths for Drell-Yan, tt̄, ggH, VBF.
- **Differential observables** — bin-based dσ/dX histograms for cosθ, pT, η, y, M_inv, M_ll, ΔR with NLO running-K rescaling.
- **Trust system** — every result carries a `validated` / `approximate` / `rough` / `blocked` badge; the API refuses (HTTP 422) processes known to give wrong answers rather than returning misleading numbers.  Audited end-to-end: every numerical response is either within stated tolerance, accompanied by an `accuracy_caveat`, or refused with a structured `block_reason` + `workaround`.
- **Browser UI + REST API** — single FastAPI app with tabbed UI (Diagrams + Distributions), Swagger at `/docs`, and a sidebar of 120+ curated example processes organised by theory (QED, QCD, QCD+QED, EW, BSM) with educational descriptions for each.

### Theory coverage

| Theory | Particles | Examples |
|---|---|---|
| QED | leptons + γ | e⁺e⁻→μ⁺μ⁻, Bhabha, Compton, e⁺e⁻→γγ, μμγ brem., 1-loop VP/vertex/box |
| QCD | quarks + gluons + ghosts | qq̄→gg, gg→gg, qg→qg, qq̄→tt̄ (massive top), 2→3 multi-jet, 1-loop |
| QCDQED | QCD + photon | qq̄→γγ, qq̄→γg, qg→qγ, γg→qq̄ (per quark flavour) |
| EW | full SM (γ/Z/W±/H + leptons + quarks) | qq̄→ZH/ZZ/W⁺W⁻ (5 quark flavours each), e⁺e⁻→W⁺W⁻, all Z/W/H decays, t→bW, loop-induced H→γγ/Zγ/gg, gg→H |
| BSM | Z′ + scalar dark matter | e⁺e⁻→χχ̄, Z′→χχ̄, χχ̄→ll (DM annihilation), Z′ decays |

### LHC validation

| Process | Engine σ | LHC LO ref | Status |
|---|---|---|---|
| pp → tt̄ at 13 TeV | 793 pb | 700-830 pb | within 13% |
| pp → DY (60 < M_ll < 120) at 13 TeV | 1530 pb | ~2000 pb | within 25% |
| pp → ZZ at 13 TeV | 7.8 pb | ~10 pb | within 22% |
| pp → H (ggF) at 13 TeV | 22.7 pb | 16-22 pb (PDF dep.) | in range |
| pp → ZH at 13 TeV | 0.27 pb | ~0.5 pb | within 50% |
| pp → H+jj (VBF) at 13 TeV | 3.78 pb | 3.78 pb | exact (calibrated) |
| pp → γγ (pT_γ > 30 GeV) at 13 TeV | 59 pb | 30-50 pb | within 2× |

### What's intentionally out of scope

- **Two-loop and higher.** 1-loop curated entries cover the textbook self-energies, vertex form factors, DGLAP NLO splitting kernels, α_s 2-loop running, and loop-induced Higgs decays (H→γγ/Zγ/gg, gg→H), but full 2-loop diagram generation is not in V1.
- **Full SUSY / SMEFT / 2HDM.** The BSM model file is intentionally minimal (Z′ + scalar dark matter); larger BSM frameworks are *additive* — see the `register_curated_amplitude` extension API below for adding your own |M̄|² for any process. A full MSSM particle content would require new QGRAF model files and is out of scope for V1.
- **NNLO precision physics.** For sub-percent NNLO predictions on a fixed set of observables (V+jet, V+V, single Higgs distributions), use NNLOJET or MCFM — those are HPC-scale Fortran codes with hours-to-days per prediction; this engine is the complementary fast/educational tool.

(Generic 1-loop NLO virtuals for arbitrary QCD processes — previously listed here — are now provided by the bundled OpenLoops 2 backend, which `feynman setup` installs by default.)

---

## Installation

### Docker (recommended — everything bundled)

```bash
docker run -p 8000:8000 ecavan/feynman-api:latest
# Open http://localhost:8000
```

The image includes QGRAF, FORM, LoopTools, LHAPDF (with CT18LO), OpenLoops 2 (with the ppllj process library), and the LaTeX/SVG rendering stack.

### PyPI

```bash
pip install feynman-engine
feynman setup        # builds QGRAF + FORM + LoopTools + LHAPDF + OpenLoops 2 (~10-15 min, one-time)
feynman doctor       # verify everything is in place
feynman serve        # launch the API + UI on http://localhost:8000
```

The full `feynman setup` is the recommended path — it gives you generic NLO QCD virtuals (via OpenLoops 2) and percent-level hadronic-σ accuracy (via LHAPDF + CT18LO) out of the box.

For a faster, lighter install (e.g. teaching, CI without `gfortran`):

```bash
feynman setup --skip-openloops --skip-lhapdf   # ~3 min, drops generic-QCD-NLO + precision PDFs
```

In skip mode: tabulated NLO K-factors still cover the major LHC channels (DY, tt̄, ggH, WW, ZZ, ZH, VBF), and a built-in LO PDF gives factor-of-2-3 accurate hadronic σ. NLO requests for unregistered QCD processes return HTTP 422 with a hint to install OpenLoops.

Add more OpenLoops process libraries on demand: `feynman install-process pptt`, `feynman install-process pphjj`, etc.

For SVG diagram rendering you also need `lualatex` + `pdf2svg` (`brew install basictex pdf2svg` on macOS; `apt-get install texlive-luatex texlive-pictures texlive-science pdf2svg` on Debian/Ubuntu).

---

## Quick API examples

```bash
# Tree-level amplitude
curl "http://localhost:8000/api/amplitude?process=e%2B+e-+-%3E+mu%2B+mu-&theory=QED"

# LO cross-section at √s = 91 GeV
curl "http://localhost:8000/api/amplitude/cross-section?process=e%2B+e-+-%3E+mu%2B+mu-&theory=QED&sqrt_s=91"

# NLO via tabulated K-factor (LHC channels)
curl "http://localhost:8000/api/amplitude/cross-section?process=p+p+-%3E+t+t~&theory=QCD&sqrt_s=13000&order=NLO"

# NLO via universal QED charge-correlator (V2.7)
curl "http://localhost:8000/api/amplitude/cross-section?process=e%2B+e-+-%3E+e%2B+e-&theory=QED&sqrt_s=91&order=NLO"

# NLO via EW Sudakov LL+NLL at high energy (V2.7)
curl "http://localhost:8000/api/amplitude/cross-section?process=e%2B+e-+-%3E+mu%2B+mu-&theory=EW&sqrt_s=1000&order=NLO"

# Hadronic σ (auto-uses LHAPDF + CT18LO if installed)
curl "http://localhost:8000/api/amplitude/hadronic-cross-section?process=p+p+-%3E+H&sqrt_s=13000"

# 2-body decay width Γ(X → 1+2) with PDG comparison
curl "http://localhost:8000/api/amplitude/decay-width?process=Z+-%3E+e%2B+e-&theory=EW"

# Differential dσ/dcosθ histogram (20 bins)
curl "http://localhost:8000/api/amplitude/differential-distribution?process=e%2B+e-+-%3E+mu%2B+mu-&theory=QED&sqrt_s=91&observable=cos_theta&bin_min=-1&bin_max=1&n_bins=20"
```

Full Swagger docs: `http://localhost:8000/docs`. Interactive walkthrough: see [`examples/getting_started.ipynb`](examples/getting_started.ipynb).

### Python API

```python
from feynman_engine.amplitudes.cross_section import total_cross_section
from feynman_engine.amplitudes.hadronic import hadronic_cross_section

r = total_cross_section("e+ e- -> mu+ mu-", "QED", sqrt_s=91.0)
print(r["sigma_pb"], r["trust_level"])    # 10.47 pb, "validated"

r = hadronic_cross_section("p p -> t t~", sqrt_s=13000.0, theory="QCD", order="NLO")
print(r["sigma_pb"], r["k_factor"])       # 1270 pb, K=1.6 (tabulated NLO)
```

To register your own |M|² for an unsupported process:

```python
import sympy as sp
from feynman_engine.physics.amplitude import register_curated_amplitude

s, t, u, e = sp.symbols("s t u e", positive=True)
register_curated_amplitude(
    "my+ my- -> custom_X", "BSM",
    msq=2 * sp.Symbol("g_X")**4 * (t**2 + u**2) / s**2,
    description="Custom BSM 2→2 via single mediator",
)
# Now total_cross_section() works for this process.
```

---

## Architecture

```
feynman_engine/
├── __main__.py            CLI entry point: serve, generate, install-*, doctor
├── core/                  Diagram model, QGRAF interface, generator
├── render/                TikZ → PDF → SVG pipeline
├── amplitudes/
│   ├── symbolic.py        SymPy γ-matrix tree amplitudes
│   ├── form_trace.py      FORM trace + SU(3) color algebra
│   ├── loop.py            PV decomposition, tensor reduction
│   ├── analytic_integrals.py   Closed-form A₀/B₀/C₀/D₀ (no LoopTools needed)
│   ├── looptools_bridge.py     ctypes wrapper for LoopTools numerical eval
│   ├── cross_section.py        scipy.quad / RAMBO / Vegas integrators
│   ├── nlo_cross_section.py    Analytic K, universal QED/EW NLO routes
│   ├── nlo_qed_general.py      universal QED NLO (charge correlator)
│   ├── nlo_ew_general.py       EW Sudakov LL+NLL
│   ├── nlo_general.py          Catani-Seymour generic NLO (DY first-principles)
│   ├── loop_curated.py         35 textbook 1-loop formulas (self-energies, vertices, H→γγ/Zγ/gg, gg→H, DGLAP, …)
│   ├── differential.py         Histogrammed observables (LO + NLO running-K)
│   ├── pdf.py                  Built-in PDF + LHAPDF wrapper + auto-discovery
│   ├── hadronic.py             pp σ via PDF convolution + specialized paths
│   ├── dipole_subtraction.py   Catani-Seymour dipoles (e+e-→μμγ)
│   └── openloops_bridge.py     OpenLoops 2 wrapper (generic NLO virtuals)
├── physics/
│   ├── amplitude.py            Amplitude registry + backend chain
│   ├── trust.py                Trust-level enforcement (BLOCKED → 422)
│   ├── nlo_k_factors.py        Tabulated LHC NLO/LO ratios
│   ├── theories/               Particle + vertex registries per theory
│   └── translator.py           Process-string parser
├── api/                   FastAPI routes + Pydantic schemas
├── frontend/              Browser UI (vanilla JS + SVG rendering)
└── resources/             Bundled HEP source archives (QGRAF, FORM, LoopTools, LHAPDF, OpenLoops)
```

The trust system (`physics/trust.py`) is the safety boundary — every API endpoint that returns a numerical result classifies the request first and refuses (HTTP 422 with a structured `block_reason` + `workaround`) for processes known to produce wrong values.

