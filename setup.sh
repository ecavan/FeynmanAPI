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

# 6. Verify Python packages
echo ""
echo "--- Python packages ---"
python -c "import networkx; print('✓ networkx', networkx.__version__)"
python -c "import pydantic; print('✓ pydantic', pydantic.VERSION)"
python -c "import fastapi; print('✓ fastapi', fastapi.__version__)"
python -c "import jinja2; print('✓ jinja2', jinja2.__version__)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start the server:"
echo "  source .venv/bin/activate"
echo "  python main.py serve"
echo "  # then open http://localhost:8000"
