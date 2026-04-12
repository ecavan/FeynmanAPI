# Loop amplitude integration via LoopTools / Package-X

This document describes the plan for computing 1-loop amplitudes by bridging to the
Fortran LoopTools library (or the Mathematica Package-X) from Python.

---

## Why this is non-trivial

Tree-level amplitudes reduce to products of 4-gamma traces in terms of external momenta
(Mandelstam s, t, u).  Loop amplitudes require:

1. **Loop-momentum integration** over d⁴k (or d^(4-2ε)k in dimensional regularisation).
2. **Tensor reduction**: rank-N tensor integrals are reduced to scalar integrals via the
   Passarino–Veltman (PV) procedure.
3. **Scalar integrals**: the master integrals B₀, C₀, D₀ (bubble, triangle, box) are
   evaluated numerically or as analytic ε-expansions.
4. **UV renormalisation**: adding counterterm diagrams and choosing a renormalisation
   scheme (MS-bar or on-shell).

The correct industry tool for steps 2-4 is **LoopTools** (Hahn & Pérez-Victoria, 1999),
which wraps the `FF` library and provides a Fortran/C interface.

---

## LoopTools subprocess bridge

### 1. Compile LoopTools

```bash
# Download from https://www.feynarts.de/looptools/
wget https://www.feynarts.de/looptools/LoopTools-2.16.tar.gz
tar -xzf LoopTools-2.16.tar.gz
cd LoopTools-2.16
./configure --prefix=$PWD/install
make && make install
# Produces: install/lib/liblooptools.a  and  install/bin/lt
```

For Docker: add the build step in a multi-stage builder (same pattern as the QGRAF
builder stage in `Dockerfile`).

### 2. Build a thin Python wrapper

LoopTools exposes a Fortran `DOUBLE COMPLEX FUNCTION B0(p², m₁², m₂²)` etc.
Two options:

**Option A — ctypes (no Fortran FFI header needed):**
```python
import ctypes, os

_lt = ctypes.CDLL(os.environ["LOOPTOOLS_LIB"])   # path to liblooptools.so

def B0(p2: float, m1sq: float, m2sq: float) -> complex:
    """Scalar two-point function."""
    re = ctypes.c_double()
    im = ctypes.c_double()
    _lt.b0_(ctypes.byref(ctypes.c_double(p2)),
            ctypes.byref(ctypes.c_double(m1sq)),
            ctypes.byref(ctypes.c_double(m2sq)),
            ctypes.byref(re), ctypes.byref(im))
    return complex(re.value, im.value)
```

Note: Fortran symbol names are lower-case with a trailing underscore (`b0_`, `c0_`,
`d0_`) on Linux/macOS when compiled with gfortran.

**Option B — subprocess via the `lt` binary:**
LoopTools ships an interactive binary `lt`.  Write a small input file, pipe it through
`lt`, parse the output.  Less robust but zero FFI headaches.

### 3. Integrate into FeynmanEngine

`feynman_engine/amplitudes/loop.py` (to be created) should:

1. Accept a `Diagram` object with `loop_order == 1`.
2. Classify the loop topology (triangle, box, self-energy — already done in `topology.py`).
3. Perform Passarino–Veltman tensor reduction symbolically (using SymPy) to express the
   amplitude as a linear combination of scalar integrals B₀, C₀, D₀ with rational
   coefficients in the external kinematics.
4. Call the LoopTools bridge to evaluate each scalar integral numerically.
5. Return an `AmplitudeResult` with `backend="looptools"`.

### 4. Files to create / modify

| File | Change |
|---|---|
| `feynman_engine/amplitudes/loop.py` | New — PV reduction + LoopTools calls |
| `feynman_engine/amplitudes/looptools_bridge.py` | New — ctypes wrapper around liblooptools |
| `feynman_engine/engine.py` | Route loop diagrams to `loop.py` |
| `Dockerfile` | Add LoopTools builder stage (mirrors QGRAF builder) |
| `feynman install-looptools` CLI | New command (mirrors `feynman install-qgraf`) |

### 5. Passarino–Veltman reduction sketch

For a self-energy diagram (1-loop, 2 propagators):
```
B_μν(p²; m₁², m₂²) = A g_μν + B p_μ p_ν
```
The scalar coefficients A, B are solved from a 2×2 system involving B₀ and A₀.

For a triangle (C₀) or box (D₀), use the Denner–Dittmaier reduction tables or
the `FeynCalc` implementation as a reference for the recursion relations.

SymPy can carry the algebra symbolically and produce a Python expression that calls the
LoopTools bridge for the scalar integrals only.

---

## Alternative: Package-X via Mathematica

`feynman_engine/physics/mathematica_bridge.py` already has a stub for this.  If a
Mathematica licence is available:

```python
from feynman_engine.physics.mathematica_bridge import compute_amplitude_feyncalc
result = compute_amplitude_feyncalc(diagram)   # calls wolframclient
```

Package-X (Patel, 2015) provides analytic results for all 1-loop scalar integrals as
hypergeometric functions of the kinematics, with full UV/IR divergence separation in
dimensional regularisation.  This is faster to implement than the LoopTools bridge but
requires Mathematica + Package-X installed.

---

## Timeline estimate

| Milestone | Effort |
|---|---|
| LoopTools compiled + ctypes wrapper | 1–2 days |
| PV tensor reduction for self-energy | 2–3 days |
| PV reduction for triangle + box | 3–5 days |
| UV renormalisation (MS-bar) | 2–3 days |
| End-to-end test: 1-loop QED e+e-→μ+μ- | 1–2 days |
| **Total** | **~2–3 weeks** |
