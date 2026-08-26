r"""Fig. 3 -- the rate/switching frontier, i.e. the objective made visible.

Reads results_tm/fig3_frontier.json. Palette matches Figs. 1-2:
    BLUE = proposed, ORANGE = best operating point, GREEN = full-CSI genie.

Two things have to be readable off this plot.

(1) OUR FRONTIER IS ALMOST FLAT IN RATE. Sweeping eta_sw moves the agent from ~19
    reconfigured ports per slot down to 0 while the sum rate barely changes -- it
    actually RISES. That is the result, not a plotting artefact, so the y-axis keeps
    the genie in view rather than zooming in and hiding how small the variation is.

(2) THE GENIE LOSES ON THE OBJECTIVE. It sits well above us in rate but on a much
    worse iso-objective contour, because it reconfigures the array almost completely
    every slot. Rather than draw an unlabelled contour field and make the reader do
    the arithmetic, we highlight exactly two contours -- the one through the genie
    and the one through our best point -- and label them with their values.

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

C_MAIN, C_OP, C_GENIE, C_ISO = "#1F6FB2", "#D55E00", "#2E8B57", "#B4BCC5"
ETA_REF = 1.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.6,
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

    ours = r - ETA_REF * w
    best = int(np.argmax(ours))
    g_obj = gr - ETA_REF * gw
    # switch price above which the genie is beaten by our best low-switching arm
    cross = (gr - r[best]) / max(gw - w[best], 1e-9)

    fig, ax = plt.subplots(figsize=(3.5, 2.72))
    ax.set_axisbelow(True)
    ax.grid(True, lw=0.35, color="#E4E8EC")

    lo_x, hi_x = -1.0, max(w.max(), gw) * 1.10
    lo_y, hi_y = min(r.min(), gr) - 1.1, max(r.max(), gr) + 1.0
    xs = np.array([lo_x, hi_x])

    # ---- faint contour field, then the two that matter
    for c in np.arange(-6, 30, 3):
        ax.plot(xs, c + ETA_REF * xs, ls=(0, (1, 2.5)), lw=0.55, color=C_ISO, zorder=1)
    for c, col in ((g_obj, C_GENIE), (ours[best], C_OP)):
        ax.plot(xs, c + ETA_REF * xs, ls=(0, (3, 2)), lw=1.0, color=col,
                alpha=0.85, zorder=2)

    # ---- frontier
    ax.plot(w, r, "-", color=C_MAIN, zorder=4,
            label="proposed, sweeping $\\eta_{\\mathrm{sw}}$")
    ax.plot(w, r, "o", ms=4.0, mfc="white", mec=C_MAIN, mew=1.3, zorder=5)
    for e, x, y, dx, dy, ha in [(etas[0], w[0], r[0], 0, -12, "center"),
                                (1.0, w[etas.index(1.0)], r[etas.index(1.0)], 0, 7, "center")]:
        ax.annotate(f"$\\eta_{{\\mathrm{{sw}}}}\\!=\\!{e:g}$", xy=(x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.3, color=C_MAIN, ha=ha)

    # ---- best objective point (several eta collapse here)
    ax.plot(w[best], r[best], "o", ms=6.4, mfc=C_OP, mec="white", mew=1.0, zorder=7)
    ax.annotate(f"$\\eta_{{\\mathrm{{sw}}}}\\!\\geq\\!{etas[best]:g}$: array never moves\n"
                f"objective {ours[best]:.1f} bits/s/Hz",
                xy=(w[best], r[best]), xytext=(hi_x * 0.055, lo_y + 0.30),
                fontsize=6.4, color=C_OP, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.26", fc="#FCEFE4", ec=C_OP, lw=0.5),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C_OP,
                                connectionstyle="arc3,rad=-0.3"))

    # ---- genie
    ax.plot(gw, gr, "D", ms=5.0, mfc=C_GENIE, mec="white", mew=0.9, zorder=7)
    ax.annotate(f"full-CSI genie\nobjective {g_obj:.1f}", xy=(gw, gr),
                xytext=(gw - 0.7, gr + 0.30), fontsize=6.4, color=C_GENIE,
                ha="right", va="bottom", linespacing=1.2)

    ax.set_xlabel("ports reconfigured per slot", labelpad=2)
    ax.set_ylabel("sum rate (bits/s/Hz)", labelpad=2)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, handlelength=1.5, borderaxespad=0.35)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)

    print(f"wrote {OUT}  ({n} seeds)")
    print(f"{'eta':>6} | {'rate':>7} | {'switch':>7} | {'obj(eta=1)':>11}")
    for e, x, y, o in zip(etas, w, r, ours):
        print(f"{e:6g} | {y:7.3f} | {x:7.2f} | {o:11.3f}")
    print(f"{'genie':>6} | {gr:7.3f} | {gw:7.2f} | {g_obj:11.3f}")
    print(f"\nrate varies by only {r.max()-r.min():.2f} bits/s/Hz across a "
          f"{w.max()-w.min():.1f} port/slot swing in switching")
    print(f"peak rate at eta={etas[int(np.argmax(r))]:g} ({r.max():.3f}), "
          f"not at eta=0 ({r[0]:.3f}) -- holding ports lets the belief sharpen")
    print(f"genie is beaten on the objective for any eta_sw > {cross:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
