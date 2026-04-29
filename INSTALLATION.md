# Installation Guide

Three install paths — pick what fits your use case.

| Path | Time | What you get | When to choose |
|---|---|---|---|
| **Docker** | 1 min | Everything bundled, no build | Easiest. Trying it out, demos. |
| **Full pip** | 10–20 min | All native HEP tools built locally | Most users; you'll get every feature. |
| **Lightweight pip** | 3 min | QGRAF + FORM + LoopTools only | Teaching, CI without `gfortran`, fastest install. |

---

## Docker (easiest)

```bash
docker run -p 8000:8000 ecavan/feynman-api:latest
# Open http://localhost:8000
```

The image bundles QGRAF, FORM, LoopTools, LHAPDF (with CT18LO), OpenLoops 2 (with the `ppllj` process library), and the LaTeX/SVG rendering stack.

No system prerequisites needed beyond Docker itself. **Recommended on Windows** — saves you Windows-specific build issues with Fortran toolchains.

---

## Full pip install (recommended for native users)

### 1. System prerequisites

You need a Fortran + C/C++ toolchain, plus optional LaTeX for SVG diagram rendering.

#### macOS

```bash
# Build toolchain (required)
brew install gcc make           # gcc package gives you gfortran

# LaTeX + SVG rendering (optional but recommended for diagrams)
brew install basictex pdf2svg
sudo tlmgr update --self
sudo tlmgr install tikz-feynman standalone
```

#### Debian / Ubuntu / WSL

```bash
sudo apt-get update
sudo apt-get install -y gfortran g++ make python3-dev

# LaTeX + SVG rendering (optional)
sudo apt-get install -y texlive-luatex texlive-pictures texlive-science pdf2svg
```

#### Windows (native)

Native Windows is **not officially supported** for the local-build path because QGRAF + FORM + LoopTools + LHAPDF + OpenLoops are all Fortran/C/C++ projects with Unix-style build systems. Two options:

- **Use Docker** (recommended): see the Docker section above.
- **Use WSL2** (Windows Subsystem for Linux): install Ubuntu via WSL, then follow the Debian/Ubuntu section above.

### 2. Install + build

```bash
pip install feynman-engine
feynman setup        # ~10–20 min, one-time. Builds QGRAF + FORM + LoopTools + LHAPDF + OpenLoops 2
feynman doctor       # verify all 5 native deps are 'ok'
feynman serve        # http://localhost:8000
```

`feynman setup` will:
1. Compile QGRAF (Fortran, ~30 s)
2. Compile FORM (C, ~1 min)
3. Compile LoopTools (Fortran + C, ~2 min)
4. Compile LHAPDF (C++, ~3-5 min) and download the CT18LO PDF set (~10 MB)
5. Compile OpenLoops 2 via SCons (Fortran, ~5-10 min) and download the `ppllj` process library

If any step fails, the recommended `feynman doctor` output points you at the specific install command to retry.

---

## Lightweight pip install

Skip the slowest dependencies for a 3-minute install:

```bash
pip install feynman-engine
feynman setup --skip-lhapdf --skip-openloops
```

You still get:
- All Feynman diagram generation (QGRAF + FORM)
- All tree-level amplitudes for QED, QCD, EW, BSM
- LO cross-sections via scipy/MC
- Loop amplitudes via the LoopTools backend
- Tabulated NLO K-factors for major LHC channels (DY, tt̄, ggH, WW, ZZ, ZH, VBF)
- Universal QED NLO + EW Sudakov NLO closed-form K-factors
- The browser UI

What you lose:
- **LHAPDF**: hadronic cross-sections fall back to a built-in LO PDF (factor-of-2-3 accuracy instead of percent-level)
- **OpenLoops**: NLO σ for unregistered QCD processes returns `HTTP 422` with a workaround

You can add these later:
```bash
feynman install-lhapdf
feynman install-pdf-set CT18LO
feynman install-openloops
feynman install-process ppllj
```

---

## Verifying the install

```bash
feynman doctor
```

Should report each component as `ok` or `missing`. Example healthy output:

```
FeynmanEngine doctor
  Backend: qgraf (/path/to/bin/qgraf)
  QGRAF: ok | binary=...
  FORM: ok | binary=...
  LoopTools: ok | library=...
  LHAPDF: ok | version=6.5.5 | sets=['CT18LO']
  OpenLoops: ok | prefix=... | processes=['ppllj']
  Rendering: lualatex=ok, pdf2svg=ok
  Toolchain: gfortran=..., make=..., cc=..., c++=...
  Recommendation: native dependencies look ready.
```

If anything is `missing`, the `Recommendation:` line tells you the exact command to fix it.

---

## Troubleshooting

### `gfortran: command not found`
- macOS: `brew install gcc` (the Homebrew `gcc` package includes `gfortran`)
- Debian/Ubuntu: `sudo apt-get install gfortran`

### `feynman setup` hangs on OpenLoops download
The OpenLoops `ppllj` process library is downloaded from openloops.hepforge.org during setup. If it hangs:
- Check internet access
- Re-run with `feynman install-openloops --no-process` to skip the download
- Manually install later: `feynman install-process ppllj`

### `feynman doctor` reports `LoopTools: missing` but `gfortran` is installed
LoopTools requires `gfortran` and a working `make`. On macOS the Apple `gcc` is actually clang (no Fortran). Install Homebrew's gcc package: `brew install gcc`.

### `lualatex: command not found` (SVG rendering missing)
SVG diagram rendering requires `lualatex` + `pdf2svg`. The engine still works without these — you just don't get rendered diagrams (the TikZ source is always returned).

### Anything else
File an issue at https://github.com/ecavan/FeynmanAPI/issues with the output of `feynman doctor` attached.

---

## Upgrading

```bash
pip install --upgrade feynman-engine
# Native binaries built by `feynman setup` are NOT recompiled on package upgrade.
# If a major version bumps a bundled tool (QGRAF, FORM, LoopTools, LHAPDF,
# OpenLoops), re-run:
feynman setup --force
```
