# Paper 1 — LaTeX source

Target: **IEEE Wireless Communications Letters** (5 pages including references).
arXiv preprint goes up first.

## Files

| file | what it is |
|---|---|
| `main.tex` | the paper |
| `refs.bib` | bibliography — **every entry needs verifying** (see header of the file) |
| `figs/` | figures, as PDF (vector) not PNG |

## Building

Local (MiKTeX / TeX Live):

```
./build.sh          # or: latexmk -pdf main.tex
```

or the manual four-pass sequence if `latexmk` is missing:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

`bibtex` needs the two extra `pdflatex` passes to resolve `[1]`-style numbers —
if citations render as `[?]`, you skipped one.

Clean up with `latexmk -c` (keeps the PDF) or `latexmk -C` (removes it too).

## Previewing a figure

```
./build_fig.sh fig_timeline      # renders that one figure at true column width
python render.py main.pdf 150    # rasterizes the built PDF to _render/*.png
```

`render.py` needs PyMuPDF (`pip install pymupdf`).

**Do not preview with `latex` + `dvipng`.** In DVI mode TikZ emits its drawing
commands as PostScript specials, which dvipng does not execute -- it renders
the text and silently discards every line, arrow and filled shape, so a
perfectly good figure looks broken. Always rasterize the real PDF.

## Figures

Save every figure from matplotlib as **PDF**, not PNG:

```python
fig.savefig("paper1/figs/name.pdf", bbox_inches="tight")
```

IEEE double-column width is 3.5 in; use `figsize=(3.5, 2.6)` and font size 8 so
the text in the figure matches the caption size after `\includegraphics[width=\columnwidth]`.

## arXiv submission

arXiv wants the **source**, not the PDF. Upload a zip of:

```
main.tex  refs.bib  main.bbl  figs/*.pdf
```

Note `main.bbl` — arXiv does not run `bibtex`, so the compiled bibliography file
must be included. It is produced by the local build.

Do **not** include `.aux`, `.log`, `.out`, `.blg`, or `latexmk` artifacts.

## Before submitting anywhere

- [ ] every `\todo{}` and `\note{}` removed
- [ ] every `% VERIFY` line in `refs.bib` resolved
- [ ] page count within the venue limit
- [ ] author list and affiliations final
