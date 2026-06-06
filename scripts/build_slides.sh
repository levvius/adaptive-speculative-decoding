#!/usr/bin/env bash
# Build the JointAdaSpec defense deck in the three most common formats:
#   papers/dist/JointAdaSpec_defense.pptx   (master — python-pptx)
#   papers/dist/JointAdaSpec_defense.pdf    (LibreOffice headless)
#   papers/dist/JointAdaSpec_defense.html   (self-contained, poppler raster)
#
# Content comes from papers/pres.md; design from papers/claude_design.md.
# All numbers are the locked experimental values — nothing is invented.
#
# Usage:  bash scripts/build_slides.sh   (or `make slides`)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
DIST="papers/dist"; mkdir -p "$DIST"

echo "[1/4] ensure python-pptx is available"
"$PY" -c "import pptx" 2>/dev/null || "$PY" -m pip install python-pptx

echo "[2/4] author master PPTX"
"$PY" papers/build_slides.py

echo "[3/4] PPTX -> PDF (LibreOffice headless)"
if ! command -v soffice >/dev/null 2>&1; then
  echo "ERROR: soffice (LibreOffice) not found — install it to export PDF/HTML." >&2
  exit 1
fi
rm -f "$DIST/JointAdaSpec_defense.pdf"
soffice --headless -env:UserInstallation=file:///tmp/lo_slides_prof \
  --convert-to pdf --outdir "$DIST" "$DIST/JointAdaSpec_defense.pptx" >/dev/null

echo "[4/4] PDF -> self-contained HTML"
"$PY" papers/build_html_deck.py

echo
echo "Done. Downloadable formats:"
ls -la "$DIST"/JointAdaSpec_defense.pptx "$DIST"/JointAdaSpec_defense.pdf "$DIST"/JointAdaSpec_defense.html
