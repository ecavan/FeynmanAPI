# Installation Guide

Two install paths. Pick what fits.

| Path | Time | Get | When |
|---|---|---|---|
| **Docker** | 1 min | Everything bundled, no build | Easiest. Trying it out, demos. |
| **pip** | 10-20 min | All native HEP tools built locally | Most users. |

LHAPDF + CT18LO is a hard prerequisite for hadronic cross-sections and is installed automatically by `feynman setup`. There is no longer a "lightweight" path that skips it — the hadronic σ numbers in the README assume LHAPDF + CT18LO and we don't ship a degraded fallback.

## Docker

```bash
docker run -p 8000:8000 ecavan/feynman-api:latest
# Open http://localhost:8000
```

The image bundles QGRAF, FORM, LoopTools, LHAPDF (with CT18LO), OpenLoops 2 (with the `ppllj` process library), and the LaTeX/SVG rendering stack. No system prerequisites beyond Docker. Recommended on Windows.

## pip install

### 1. System prerequisites

A Fortran and C/C++ toolchain. Optional LaTeX for SVG diagram rendering.

**macOS:**
```bash
brew install gcc make           # the gcc package gives you gfortran
brew install basictex pdf2svg   # optional, for SVG rendering
sudo tlmgr update --self
sudo tlmgr install tikz-feynman standalone
```

**Debian / Ubuntu / WSL:**
```bash
sudo apt-get update
sudo apt-get install -y gfortran g++ make python3-dev
sudo apt-get install -y texlive-luatex texlive-pictures texlive-science pdf2svg
```

**Windows (native)**: not officially supported because the bundled tools (QGRAF, FORM, LoopTools, LHAPDF, OpenLoops 2) all use Unix build systems. Use Docker or [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) instead.

### 2. Install + build

```bash
pip install feynman-engine
feynman setup     # 10-20 min one-time
feynman doctor    # verify all 5 native deps are 'ok'
feynman serve     # http://localhost:8000
```

What `feynman setup` does:

1. Compile QGRAF (Fortran, ~30 s)
2. Compile FORM (C, ~1 min)
3. Compile LoopTools (Fortran + C, ~2 min)
4. Compile LHAPDF (C++, ~3-5 min) and download CT18LO (~10 MB)
5. Compile OpenLoops 2 via SCons (Fortran, ~5-10 min) and download `ppllj`

If a step fails, `feynman doctor` prints the exact retry command.

## Skipping OpenLoops

If you don't need generic NLO QCD virtuals for arbitrary processes (the engine has tabulated K-factors for major LHC channels built-in), you can shave ~5-10 min off the build:

```bash
feynman setup --skip-openloops
```

OpenLoops can be added later:

```bash
feynman install-openloops
feynman install-process ppllj
```

LHAPDF is not optional — `feynman setup` always builds it and installs CT18LO.

## Verifying the install

```bash
feynman doctor
```

A healthy install reports:

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

## Troubleshooting

**`gfortran: command not found`**
On macOS, run `brew install gcc` (the Homebrew gcc package includes gfortran). On Debian/Ubuntu: `sudo apt-get install gfortran`.

**`feynman setup` hangs on the OpenLoops download.**
The `ppllj` process library is downloaded from openloops.hepforge.org during setup. Check your internet connection. To skip the download and install the library later: `feynman install-openloops --no-process`, then `feynman install-process ppllj`.

**`feynman doctor` reports `LoopTools: missing` but `gfortran` is installed.**
LoopTools needs both gfortran and make. On macOS, Apple's `gcc` is actually clang and has no Fortran. Install Homebrew gcc: `brew install gcc`.

**`lualatex: command not found`.**
SVG diagram rendering needs `lualatex` and `pdf2svg`. The engine still works without these (the TikZ source is always returned), you just lose rendered SVGs.

**Anything else.** File an issue at https://github.com/ecavan/FeynmanAPI/issues with the output of `feynman doctor` attached.

## Upgrading

```bash
pip install --upgrade feynman-engine
```

Native binaries built by `feynman setup` are not recompiled on a package upgrade. If a major version bumps a bundled tool (QGRAF, FORM, LoopTools, LHAPDF, OpenLoops), re-run:

```bash
feynman setup --force
```
