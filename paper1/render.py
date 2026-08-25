r"""Rasterize a PDF to PNG pages.

WHY THIS EXISTS. The obvious shortcut -- compile with `latex` to DVI and run
`dvipng` -- CANNOT be trusted for this paper. In DVI mode TikZ emits its
drawing commands as PostScript \special's, and dvipng does not execute
PostScript: it renders the text nodes and silently discards every line, arrow
and filled shape. The result looks like a broken figure when the figure is
fine. Anything with vector graphics must be rasterized from the real PDF.

Usage:
    python render.py main.pdf [dpi] [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz


def main() -> int:
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "main.pdf")
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else pdf.parent / "_render"
    out.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf)
    for i, page in enumerate(doc, 1):
        p = out / f"{pdf.stem}_p{i}.png"
        page.get_pixmap(dpi=dpi).save(p)
        print(f"{p}  ({page.rect.width:.0f}x{page.rect.height:.0f}pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
