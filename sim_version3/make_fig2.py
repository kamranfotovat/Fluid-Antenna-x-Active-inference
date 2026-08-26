r"""Fig. 2 -- sum rate vs pilot budget. Reads results_tm/fig2_pilot_sweep.json.

Emits a vector PDF at IEEE single-column width (3.5in). PDF, never PNG: the
paper is vector throughout and a raster figure is visibly worse in print.

COLOUR follows Fig. 1 so the two figures read as one story:
    BLUE   = the proposed scheme / activated ports
    ORANGE = the pilot budget, and the operating point the paper argues for
    GREEN  = the full-CSI genie, i.e. the ceiling
Palette is Okabe-Ito derived (colour-blind safe) and the luminances are spread
far enough apart that the figure survives a greyscale print.

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
from matplotlib.patches import FancyBboxPatch  # noqa: F401  (kept for annotation bbox)

ROOT = Path(__file__).resolve().parent.parent
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results_tm" / "fig2_pilot_sweep.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper1" / "figs" / "fig2_pilots.pdf"

C_MAIN = "#1F6FB2"   # proposed
C_BAND = "#1F6FB2"
C_OP = "#D55E00"     # the m = 6 operating point
C_GENIE = "#2E8B57"  # full-CSI ceiling
C_GAP = "#7F8C8D"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
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

    fig, ax = plt.subplots(figsize=(3.5, 2.62))
    ax.set_axisbelow(True)
    ax.grid(True, lw=0.35, color="#DDE1E6")

    # --- the gap the belief has to cover: everything between us and the genie
    ax.fill_between(ms, mu, g, color=C_GAP, alpha=0.07, lw=0, zorder=1)

    # --- genie ceiling
    ax.axhline(g, ls=(0, (5, 2)), lw=1.1, color=C_GENIE, zorder=3)
    ax.text(ms[-1] + 0.05, g + 0.18, "full-CSI genie", fontsize=6.8,
            color=C_GENIE, ha="right", va="bottom", fontweight="bold")

    # --- 95% CI band, then the curve on top
    ax.fill_between(ms, mu - 1.96 * se, mu + 1.96 * se,
                    color=C_BAND, alpha=0.20, lw=0, zorder=4)
    ax.plot(ms, mu, "-", color=C_MAIN, zorder=5,
            label="proposed: AR($p$) belief, hybrid $n_{\\mathrm{RF}}\\!=\\!2K$")
    ax.plot(ms, mu, "o", ms=4.2, mfc="white", mec=C_MAIN, mew=1.3, zorder=6)

    # --- the operating point the paper argues for
    i6 = int(np.where(ms == 6)[0][0])
    ax.plot(6, mu[i6], "o", ms=6.4, mfc=C_OP, mec="white", mew=1.0, zorder=7)
    ax.annotate(f"$m=6$\n{mu[i6]/g*100:.0f}% of genie on\n{6/N*100:.1f}% of the ports",
                xy=(6, mu[i6]), xytext=(6.7, mu[i6] - 5.0),
                fontsize=6.6, color=C_OP, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.28", fc="#FCEFE4",
                          ec=C_OP, lw=0.5, alpha=0.95),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C_OP,
                                connectionstyle="arc3,rad=-0.25"))

    ax.set_xlabel("piloted ports $m$  (of $M=%d$ activated)" % M, labelpad=2)
    ax.set_ylabel("sum rate (bits/s/Hz)", labelpad=2)
    ax.set_xticks(ms)
    ax.set_xlim(ms[0] - 0.45, ms[-1] + 0.45)
    ax.set_ylim(mu.min() - 1.6, g + 1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=1.5,
              borderaxespad=0.3)

    # --- second x-axis: what fraction of the WHOLE array is ever measured
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / N * 100,
                                               lambda p: p * N / 100))
    sec.set_xlabel("candidate ports measured (%)", fontsize=7, labelpad=2.5)
    sec.set_xticks([m / N * 100 for m in ms])
    sec.set_xticklabels([f"{m/N*100:.2f}" for m in ms], fontsize=6.4)
    sec.spines["top"].set_linewidth(0.6)

    # --- second y-axis: the same curve as a fraction of the ceiling
    secy = ax.secondary_yaxis("right", functions=(lambda y: y / g * 100,
                                                  lambda q: q * g / 100))
    secy.set_ylabel("% of genie", fontsize=7, labelpad=3)
    secy.set_yticks([40, 60, 80, 100])
    secy.tick_params(labelsize=6.6)
    secy.spines["right"].set_linewidth(0.6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}  ({n} seeds, rule={d['meta'].get('pilot_rule','variance')})")

    for m, a, s in zip(ms, mu, se):
        print(f"  m={m:2d}  {a:7.3f} +/- {s:.3f}   {a/g*100:5.1f}% genie   "
              f"{a/mu[-1]*100:5.1f}% of m=M   ({m/N*100:.2f}% of ports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
