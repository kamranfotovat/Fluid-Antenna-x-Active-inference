#!/usr/bin/env bash
# Build the paper. MiKTeX lives outside the default PATH of a shell that was
# started before it was installed, so we add it explicitly.
#
# This script fails on THREE things, not just one:
#   - '!' errors            (won't produce a PDF)
#   - Overfull \hbox        (text/equations spilling past the column edge -- in
#                            two-column IEEEtran these silently overprint the
#                            NEXT column, and the PDF still "builds fine")
#   - undefined references  (\ref or \cite that resolved to '?')
set -e
export PATH="/c/Users/kian/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
bibtex   main                              >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true

fail=0

n=$(grep -c '^!' main.log || true)
if [ "$n" -gt 0 ]; then
  echo "ERRORS ($n):"
  grep -A4 '^!' main.log | sed 's/^/  /'
  fail=1
fi

# NOTE: display math reports "Overfull \hbox (Xpt too wide) detected at line N",
# NOT the "in paragraph at lines" wording. Match both.
n=$(grep -c 'Overfull \\hbox' main.log || true)
if [ "$n" -gt 0 ]; then
  echo "OVERFULL BOXES ($n) -- these overprint the adjacent column:"
  grep 'Overfull \\hbox' main.log | sed 's/^/  /'
  fail=1
fi

if grep -q 'undefined' main.log; then
  echo "UNDEFINED REFERENCES:"
  grep -i 'undefined' main.log | sed 's/^/  /'
  fail=1
fi

[ "$fail" -eq 1 ] && { echo "BUILD NOT CLEAN"; exit 1; }

echo "OK  $(grep -oE 'Output written on main.pdf \([0-9]+ pages?' main.log | grep -oE '[0-9]+ pages?')"
grep -E 'Warning--' main.blg 2>/dev/null | sed 's/^/  bib: /' || true
exit 0
