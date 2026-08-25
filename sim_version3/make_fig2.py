r"""Fig. 2 -- sum rate vs pilot budget. Reads results_tm/fig2_pilot_sweep.json.

Emits a vector PDF at IEEE single-column width (3.5in). Save as PDF, never PNG:
the paper is vector throughout and a raster figure is visibly worse in print.

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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,          # embed as TrueType -- arXiv/IEEE prefer this
})


def main() -> int:
    d = json.loads(IN.read_text(encoding="utf-8"))
    n = d["meta"]["MC_done"]
    N = 441
    M = max(int(k) for k in d["rates"])
    ms = sorted(int(k) for k in d["rates"])
    mu = np.array([np.mean(d["rates"][str(m)]) for m in ms])
    se = np.array([np.std(d["rates"][str(m)], ddof=1) / np.sqrt(n) for m in ms])
    g = float(np.mean(d["genie"]))

    fig, ax = plt.subplots(figsize=(3.5, 2.45))

    ax.axhline(g, ls="--", lw=0.9, color="0.35")
    ax.text(ms[0] + 0.05, g + 0.25, "full-CSI genie (all $N$ ports known)",
            fontsize=6.5, color="0.25", va="bottom")

    ax.errorbar(ms, mu, yerr=1.96 * se, marker="o", ms=3.4, capsize=2,
                color="k", mfc="white", mew=0.9, elinewidth=0.8,
                label="proposed (AR($p$), hybrid $n_{\\mathrm{RF}}\\!=\\!2K$)")

    # the operating point the paper argues for
    i6 = ms.index(6)
    ax.annotate(f"$m=6$: {mu[i6]/g*100:.0f}% of genie\non {6/N*100:.1f}% of the ports",
                xy=(6, mu[i6]), xytext=(6.15, mu[i6] - 4.6), fontsize=6.5,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4",
                                connectionstyle="arc3,rad=0.15"))

    ax.set_xlabel("piloted ports $m$   (of $M=%d$ activated, $N=%d$ candidates)" % (M, N))
    ax.set_ylabel("sum rate (bits/s/Hz)")
    ax.set_xticks(ms)
    ax.set_xlim(ms[0] - 0.5, ms[-1] + 0.5)
    ax.set_ylim(min(mu) - 2.2, g + 1.6)
    ax.grid(True, lw=0.35, color="0.85")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, handlelength=1.6)

    # secondary axis: what fraction of the whole array is measured
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / N * 100, lambda p: p * N / 100))
    sec.set_xlabel("fraction of the candidate ports measured (%)", fontsize=7)
    sec.set_xticks([m / N * 100 for m in ms])
    sec.set_xticklabels([f"{m/N*100:.2f}" for m in ms], fontsize=6.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}  ({n} seeds)")

    for m, a, s in zip(ms, mu, se):
        print(f"  m={m:2d}  {a:7.3f} +/- {s:.3f}   {a/g*100:5.1f}% genie   "
              f"{a/mu[-1]*100:5.1f}% of m=M   ({m/N*100:.2f}% of ports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
