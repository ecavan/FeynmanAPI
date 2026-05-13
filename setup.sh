#!/bin/bash
# Convenience wrapper around `feynman setup`.
# Most users should just `pip install feynman-engine && feynman setup`.
# This script exists for source-checkout development: it builds an editable
# venv install and then hands off to the package's setup wizard.
set -e

echo "=== FeynmanEngine: source checkout setup ==="
echo

# 1. Python venv + editable install
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# 2. Platform-specific system deps reminder
echo
echo "--- System deps ---"
if [[ "$OSTYPE" == "darwin"* ]]; then
  which gfortran >/dev/null  && echo "✓ gfortran" || echo "✗ gfortran   → brew install gcc"
  which lualatex >/dev/null  && echo "✓ lualatex" || echo "✗ lualatex   → brew install basictex && tlmgr install tikz-feynman standalone"
  which pdf2svg  >/dev/null  && echo "✓ pdf2svg"  || echo "✗ pdf2svg    → brew install pdf2svg"
else
  which gfortran >/dev/null  && echo "✓ gfortran" || echo "✗ gfortran   → sudo apt-get install gfortran g++ make python3-dev"
  which lualatex >/dev/null  && echo "✓ lualatex" || echo "✗ lualatex   → sudo apt-get install texlive-luatex texlive-pictures texlive-science"
  which pdf2svg  >/dev/null  && echo "✓ pdf2svg"  || echo "✗ pdf2svg    → sudo apt-get install pdf2svg"
fi

# 3. Hand off to the package wizard.
# `feynman setup` is interactive by default: it asks which OpenLoops
# process pack(s) to install (by audience or by theory). LHAPDF + the
# default LO PDF set are always installed — there's no skip flag.
# Non-interactive callers can pass --profile (e.g. `--profile student`).
echo
echo "--- Native deps + OL packs ---"
echo "Launching the FeynmanEngine setup wizard..."
echo "(pass --profile <name> here for non-interactive setup; see "
echo " 'feynman setup --help' for the full list of profiles)"
echo
python -m feynman_engine setup "$@"

echo
echo "=== Done ==="
echo "Verify with:  python -m feynman_engine doctor"
echo "Start server: python main.py serve   # http://localhost:8000"
