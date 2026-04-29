#!/bin/bash
set -e

echo "=== Feynman Engine Setup ==="

# 1. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# 2. System dependencies
echo ""
echo "--- System dependencies ---"
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "macOS detected. Install with Homebrew if needed:"
  echo "  brew install gcc basictex pdf2svg"
  echo "  tlmgr install tikz-feynman"
  which gfortran && echo "✓ gfortran" || echo "✗ gfortran (brew install gcc)"
  which lualatex && echo "✓ lualatex"  || echo "✗ lualatex (brew install basictex)"
  which pdf2svg  && echo "✓ pdf2svg"   || echo "✗ pdf2svg (brew install pdf2svg)"
else
  # Ubuntu/Debian
  sudo apt-get update -q
  sudo apt-get install -y gfortran texlive-full pdf2svg
  which gfortran && echo "✓ gfortran"
  which lualatex && echo "✓ lualatex"
  which pdf2svg  && echo "✓ pdf2svg"
fi

# 3. QGRAF
echo ""
echo "--- QGRAF ---"
if [ -f bin/qgraf ]; then
  echo "✓ QGRAF binary found at bin/qgraf"
else
  echo "QGRAF binary not found. Attempting to build from bundled source archive..."
  if python -m feynman_engine install-qgraf; then
    echo "✓ Built QGRAF from bundled source archive"
  else
    echo "✗ Unable to build QGRAF automatically."
    echo "  Ensure gfortran is installed and that qgraf-3.6.10.tgz is present."
    exit 1
  fi
fi

# 4. LoopTools
echo ""
echo "--- LoopTools ---"
LT_LIB=$(ls bin/liblooptools.dylib bin/liblooptools.so 2>/dev/null | head -1)
if [ -n "$LT_LIB" ]; then
  echo "✓ LoopTools library found at $LT_LIB"
else
  echo "LoopTools library not found. Attempting to build from bundled source archive..."
  if python -m feynman_engine install-looptools; then
    echo "✓ Built LoopTools from bundled source archive"
  else
    echo "⚠ Unable to build LoopTools automatically."
    echo "  Loop-level numerical evaluation will be unavailable."
    echo "  Ensure gfortran and make are installed."
  fi
fi

# 5. FORM
echo ""
echo "--- FORM ---"
if [ -f bin/form ]; then
  echo "✓ FORM binary found at bin/form"
else
  echo "FORM binary not found. Attempting to build from bundled source archive..."
  if python -m feynman_engine install-form; then
    echo "✓ Built FORM from bundled source archive"
  else
    echo "⚠ Unable to build FORM automatically."
    echo "  FORM-based trace computation will be unavailable (SymPy fallback will be used)."
    echo "  Ensure a C compiler (cc/gcc/clang) and make are installed."
  fi
fi

# 6. LHAPDF (optional but recommended for hadron-collider physics)
# Skip when SKIP_LHAPDF=1 (e.g. fast iteration without internet).
echo ""
echo "--- LHAPDF (PDFs for pp processes) ---"
if [ -n "$SKIP_LHAPDF" ]; then
  echo "Skipped (SKIP_LHAPDF set)."
elif python -c "import lhapdf" 2>/dev/null; then
  echo "✓ LHAPDF Python bindings already importable"
elif python -c "
import sys
sys.path.insert(0, '/tmp/lhapdf-install/lib/python3.14/site-packages')
sys.path.insert(0, '/tmp/lhapdf-install/lib/python3.13/site-packages')
sys.path.insert(0, '/tmp/lhapdf-install/lib/python3.12/site-packages')
sys.path.insert(0, '/tmp/lhapdf-install/lib/python3.11/site-packages')
import lhapdf
" 2>/dev/null; then
  echo "✓ LHAPDF found at /tmp/lhapdf-install (auto-discovered by feynman_engine)"
else
  echo "LHAPDF not installed. Building from bundled source archive..."
  echo "(This compiles a C++ library — takes a couple of minutes.)"
  if python -m feynman_engine install-lhapdf; then
    echo "✓ Built LHAPDF + installed default CT18LO PDF set"
    echo "  → pp processes now use percent-level-accurate PDFs"
  else
    echo "⚠ Unable to build LHAPDF automatically."
    echo "  pp processes will use the built-in LO-simple PDF (factor-of-2-3 accuracy)."
    echo "  Ensure a C++ compiler (g++/clang++), make, and Python headers are installed."
    echo "  Re-run later with: python -m feynman_engine install-lhapdf"
  fi
fi

# 7. OpenLoops (recommended: enables generic NLO QCD virtuals for any process)
# Skip when SKIP_OPENLOOPS=1.
echo ""
echo "--- OpenLoops 2 (generic NLO QCD virtuals) ---"
if [ -n "$SKIP_OPENLOOPS" ]; then
  echo "Skipped (SKIP_OPENLOOPS set)."
elif [ -d /tmp/ol-build ] || [ -d /opt/openloops ] || [ -d ~/.local/openloops ]; then
  echo "✓ OpenLoops install found (auto-discovered by feynman_engine)"
else
  echo "OpenLoops not installed. Building from bundled source archive..."
  echo "(This compiles Fortran via SCons — takes 5-10 minutes.)"
  if python -m feynman_engine install-openloops; then
    echo "✓ Built OpenLoops + installed default ppllj process library"
    echo "  → generic NLO QCD virtuals now available for any SM process"
  else
    echo "⚠ Unable to build OpenLoops automatically."
    echo "  Without it, NLO σ for unregistered QCD processes will be BLOCKED;"
    echo "  the tabulated K-factors for major LHC channels still work."
    echo "  Ensure gfortran, scons, and a C++ compiler are installed."
    echo "  Re-run later with: python -m feynman_engine install-openloops"
  fi
fi

# 8. Verify Python packages
echo ""
echo "--- Python packages ---"
python -c "import networkx; print('✓ networkx', networkx.__version__)"
python -c "import pydantic; print('✓ pydantic', pydantic.VERSION)"
python -c "import fastapi; print('✓ fastapi', fastapi.__version__)"
python -c "import jinja2; print('✓ jinja2', jinja2.__version__)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Verify everything with:"
echo "  python -m feynman_engine doctor"
echo ""
echo "To start the server:"
echo "  source .venv/bin/activate"
echo "  python main.py serve"
echo "  # then open http://localhost:8000"
