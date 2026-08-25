#!/usr/bin/env bash
# Preview a single TikZ figure at TRUE IEEEtran column width, without
# rebuilding the whole paper.
#
# Uses pdflatex + render.py (PyMuPDF), NOT latex + dvipng.
#
# dvipng CANNOT render TikZ: in DVI mode TikZ emits PostScript \special's,
# which dvipng does not execute, so every line, arrow and filled shape is
# silently dropped while the text nodes still render. A correct figure comes
# out looking collapsed and broken. Do not "optimize" this back to DVI.
#
# The `standalone` class is likewise avoided -- it is fine with pdflatex, but
# rendering inside a real IEEEtran page shows the figure at the exact width
# and font it will have in the paper, which is what we actually want to check.
#
# Usage: ./build_fig.sh fig_timeline [dpi]
set -e
export PATH="/c/Users/kian/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd "$(dirname "$0")"
F="${1:-fig_timeline}"
DPI="${2:-200}"
HERE="$(pwd -W)"
SP="C:/Users/kian/AppData/Local/Temp/claude/C--Users-kian-ACTIVEINF/c42f12ca-792e-40b1-b61b-75b2aadbf266/scratchpad/figbuild"
mkdir -p "$SP"

cat > "$SP/$F.tex" <<TEX
\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,shapes.geometric}
\pagestyle{empty}
\begin{document}
\begin{figure}[t]
  \centering
  \input{$HERE/$F}
  \caption{PREVIEW -- caption goes here.}
\end{figure}
\mbox{}
\end{document}
TEX

cd "$SP"
pdflatex -interaction=nonstopmode -halt-on-error "$F.tex" >/dev/null 2>&1 || true
n=$(grep -c '^!' "$F.log" || true)
if [ "$n" -gt 0 ]; then echo "FIG ERRORS ($n):"; grep -A5 '^!' "$F.log"; exit 1; fi
grep 'Overfull \\hbox' "$F.log" | sed 's/^/  warn: /' || true
python "$HERE/render.py" "$F.pdf" "$DPI" "$SP" >/dev/null
echo "OK -> $SP/${F}_p1.png"
