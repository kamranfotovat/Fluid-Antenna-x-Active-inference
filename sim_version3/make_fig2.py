r"""Fig. 2 -- sum rate vs pilot budget. Reads results_tm/fig2_pilot_sweep.json.

Emits a vector PDF at IEEE single-column width (3.5in). PDF, never PNG: the
paper is vector throughout and a raster figure is visibly worse in print.

STYLE. Plain journal figure: full box frame, light dotted grid, framed legend,
filled markers on solid lines, black dashed reference line. No floating tinted
callouts, no curved leader arrows, no shaded regions -- everything a callout
used to say now lives in the caption or the body text, which is where a
reviewer expects to find it.

COLOUR is Okabe-Ito derived (colour-blind safe) and the luminances are spread
far enough apart that the figure survives a greyscale print:
    BLUE   = the proposed scheme
    ORANGE = the operating point the paper argues for
    BLACK  = the full-CSI genie, i.e. the ceiling

Run:  python make_fig2.py [in.json] [out.pdf]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results_tm" / "fig2_pilot_sweep.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper1" / "figs" / "fig2_pilots.pdf"

C_MAIN = "#0072B2"   # proposed
C_OP = "#D55E00"     # the m = 6 operating point
C_GENIE = "black"    # full-CSI ceiling

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "in", "ytick.direction": "in",
    "pdf.fonttype": 42,          # embed as TrueType -- arXiv/IEEE prefer this
})


def main() -> int:
    d = json.loads(IN.read_text(encoding="utf-8"))
    n = d["meta"]["MC_done"]
    N, M = 441, max(int(k) for k in d["rates"])
    ms = np.array(sorted(int(k) for k in d["rates"]))
    mu = np.array([np.mean(d["rates"][str(m)]) for m in ms])
    se = np.array([np.std(d["rates"][str(m)], ddof=1) / np.sqrt(n) for m in ms])
    g = float(np.mean(d["genie"]))

    fig, ax = plt.subplots(figsize=(3.5, 2.02))
    ax.set_axisbelow(True)
    ax.grid(True, ls=":", lw=0.5, color="#B0B0B0")

    # --- genie ceiling
    ax.axhline(g, ls="--", lw=1.0, color=C_GENIE, zorder=3,
               label="full-CSI genie (no pilot or switching cost)")

    # --- 95% CI band, then the curve on top
    ax.fill_between(ms, mu - 1.96 * se, mu + 1.96 * se,
                    color=C_MAIN, alpha=0.18, lw=0, zorder=2)
    ax.plot(ms, mu, "-o", color=C_MAIN, ms=3.8, mfc=C_MAIN, mec=C_MAIN,
            zorder=5, label="proposed, hybrid $n_{\\mathrm{RF}}\\!=\\!2K$")

    # --- the operating point the paper argues for
    i6 = int(np.where(ms == 6)[0][0])
    ax.plot(6, mu[i6], "s", ms=5.6, mfc=C_OP, mec=C_OP, zorder=7,
            ls="none", label="operating point, $m=6$")

    ax.set_xlabel("piloted ports $m$  (of $M=%d$ activated)" % M, labelpad=2)
    ax.set_ylabel("sum rate (bits/s/Hz)", labelpad=2)
    ax.set_xticks(ms)
    ax.set_xlim(ms[0] - 0.45, ms[-1] + 0.45)
    ax.set_ylim(mu.min() - 1.5, g + 1.2)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, edgecolor="black",
              fancybox=False, handlelength=1.9, borderaxespad=0.4,
              labelspacing=0.3, borderpad=0.4).get_frame().set_linewidth(0.5)

    # --- second x-axis: what fraction of the WHOLE array is ever measured
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / N * 100,
                                               lambda p: p * N / 100))
    sec.set_xlabel("candidate ports measured (%)", fontsize=7, labelpad=3)
    sec.set_xticks([m / N * 100 for m in ms])
    sec.set_xticklabels([f"{m/N*100:.2f}" for m in ms], fontsize=6.4)
    sec.tick_params(direction="in", width=0.7, size=3)
    sec.spines["top"].set_linewidth(0.7)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}  ({n} seeds, rule={d['meta'].get('pilot_rule','variance')})")

    for m, a, s in zip(ms, mu, se):
        print(f"  m={m:2d}  {a:7.3f} +/- {s:.3f}   {a/g*100:5.1f}% genie   "
              f"{a/mu[-1]*100:5.1f}% of m=M   ({m/N*100:.2f}% of ports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
