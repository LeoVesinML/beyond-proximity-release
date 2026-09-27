#!/usr/bin/env bash
# Build papers/ieee-access/main.pdf and run the IEEE Access compliance audit.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="$REPO/papers/ieee-access"

python3 -B "$REPO/scripts/sync_ieee_access_paper.py"

pushd "$PAPER" >/dev/null
if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -halt-on-error -interaction=nonstopmode main.tex
  bibtex main
  pdflatex -halt-on-error -interaction=nonstopmode main.tex
  pdflatex -halt-on-error -interaction=nonstopmode main.tex
else
  echo "ERROR: pdflatex not found. Install TeX Live or run on a machine with MiKTeX." >&2
  exit 1
fi
popd >/dev/null

python3 -B "$REPO/scripts/check_ieee_access_paper.py"
echo "IEEE Access PDF: $PAPER/main.pdf"
