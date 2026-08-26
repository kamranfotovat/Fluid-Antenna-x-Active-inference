r"""Fig. 3 -- the rate/switching frontier, i.e. the objective made visible.

Reads results_tm/fig3_frontier.json. Same palette as Figs. 1-2:
    BLUE   = proposed
    ORANGE = the operating point the paper argues for
    GREEN  = the full-CSI genie

The dashed grey lines are ISO-OBJECTIVE contours: every point on one has the same
value of  sum rate - eta_sw * (ports moved),  the quantity (6) actually maximizes.
They are what let a reader see the result at a glance -- the genie sits above us in
rate but on a WORSE contour, because it buys that rate by reconfiguring the array
almost completely every slot.

Run:  python make_fig3.py [in.json] [out.pdf]
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
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results_tm" / "fig3_frontier.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper1" / "figs" / "fig3_frontier.pdf"

C_MAIN = "#1F6FB2"
C_OP = "#D55E00"
C_GENIE = "#2E8B57"
C_ISO = "#98A2AD"
ETA_REF = 1.0          # the eta the iso-contours are drawn for

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42,
})


def main() -> int:
    d = json.loads(IN.read_text(encoding="utf-8"))
    n = d["meta"]["MC_done"]
    etas = [float(e) for e in d["meta"]["eta_list"]]
    r = np.array([np.mean(d["rate"][str(e)]) for e in etas])
    w = np.array([np.mean(d["switch"][str(e)]) for e in etas])
    gr, gw = float(np.mean(d["genie_rate"])), float(np.mean(d["genie_switch"]))

    fig, ax = plt.subplots(figsize=(3.5, 2.62))
    ax.set_axisbelow(True)
    ax.grid(True, lw=0.35, color="#DDE1E6")

    lo_x, hi_x = -0.6, max(w.max(), gw) * 1.13
    lo_y, hi_y = min(r.min(), gr) - 1.5, max(r.max(), gr) + 1.3

    # ---- iso-objective contours: rate - eta*switches = const
    ours = r - ETA_REF * w
    best_i = int(np.argmax(ours))
    for c in np.arange(np.floor(min(ours.min(), gr - ETA_REF * gw) / 4) * 4,
                       np.ceil(ours.max() / 4) * 4 + 4, 4):
        xs = np.array([lo_x, hi_x])
        ax.plot(xs, c + ETA_REF * xs, ls=(0, (1, 2)), lw=0.65, color=C_ISO, zorder=1)
    ax.text(hi_x * 0.985, lo_y + 0.35, "iso-objective\n(rate $-\\,\\eta_{\\mathrm{sw}}$"
            "$\\times$ switches)", fontsize=6.0, color=C_ISO,
            ha="right", va="bottom", linespacing=1.25)

    # ---- our frontier, traced by sweeping eta_sw
    ax.plot(w, r, "-", color=C_MAIN, zorder=4, label="proposed, sweeping $\\eta_{\\mathrm{sw}}$")
    ax.plot(w, r, "o", ms=4.0, mfc="white", mec=C_MAIN, mew=1.3, zorder=5)
    for e, x, y in zip(etas, w, r):
        if e in (etas[0], etas[-1]):
            ax.annotate(f"$\\eta_{{\\mathrm{{sw}}}}\\!=\\!{e:g}$", xy=(x, y),
                        xytext=(0, -11 if e == etas[0] else 8), textcoords="offset points",
                        fontsize=6.3, color=C_MAIN, ha="center")

    # ---- the operating point
    ax.plot(w[best_i], r[best_i], "o", ms=6.4, mfc=C_OP, mec="white", mew=1.0, zorder=7)
    ax.annotate(f"best objective\n{ours[best_i]:.1f} bits/s/Hz",
                xy=(w[best_i], r[best_i]),
                xytext=(w[best_i] + hi_x * 0.10, r[best_i] - 2.9),
                fontsize=6.4, color=C_OP, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.26", fc="#FCEFE4", ec=C_OP, lw=0.5),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C_OP,
                                connectionstyle="arc3,rad=0.25"))

    # ---- the genie
    ax.plot(gw, gr, "D", ms=5.0, mfc=C_GENIE, mec="white", mew=0.9, zorder=7)
    ax.annotate(f"full-CSI genie\n{gr - ETA_REF*gw:.1f} bits/s/Hz",
                xy=(gw, gr), xytext=(gw - hi_x * 0.04, gr + 0.55),
                fontsize=6.4, color=C_GENIE, ha="right", va="bottom", linespacing=1.2)

    ax.set_xlabel("ports reconfigured per slot", labelpad=2)
    ax.set_ylabel("sum rate (bits/s/Hz)", labelpad=2)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=1.5, borderaxespad=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}  ({n} seeds)")
    print(f"{'eta':>6} | {'rate':>7} | {'switch':>7} | {'objective':>10}")
    for e, x, y, o in zip(etas, w, r, ours):
        print(f"{e:6g} | {y:7.3f} | {x:7.2f} | {o:10.3f}")
    print(f"{'genie':>6} | {gr:7.3f} | {gw:7.2f} | {gr - ETA_REF*gw:10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
