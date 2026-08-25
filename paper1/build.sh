#!/usr/bin/env bash
# Build the paper. MiKTeX lives outside the default PATH of a shell that was
# started before it was installed, so we add it explicitly.
set -e
export PATH="/c/Users/kian/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
bibtex   main                             >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1

n=$(grep -c '^!' main.log || true)
if [ "$n" -gt 0 ]; then
  echo "BUILD FAILED -- $n error(s):"
  grep -A4 '^!' main.log
  exit 1
fi
echo "OK  $(grep -oE 'Output written on main.pdf \([0-9]+ pages?' main.log | grep -oE '[0-9]+ pages?')"
grep -E "Warning--" main.blg 2>/dev/null | sed 's/^/  bib: /' || true
