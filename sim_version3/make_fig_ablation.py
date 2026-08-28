r"""Fig. 3 -- the epistemic ablation. Does the information-gain term earn its place?

Reads results_tm/ablation_beta.json.

THE FIGURE HAS TO CARRY THREE CLAIMS.

(1) THE TERM IS LOAD-BEARING. beta_w = 0 is pure rate-greedy selection -- no active
    inference at all -- and it strands the agent near 66% of the genie. Any beta_w > 0
    lifts it to ~84%. This is the answer to "isn't this just rebranding?".

(2) THE COMPARISON IS NOT CONFOUNDED BY MOVEMENT. A sceptic can object that the
    exploring agent simply moves more, and movement is what buys the rate. So we also
    plot the eta_sw = 4 arm, where Table I shows every policy is driven to zero
    switching: there the two agents move IDENTICALLY (not at all) and the epistemic
    one is still ~4 b/s/Hz ahead. That dashed line is the honest version of claim (1).

(3) MORE IS NOT BETTER. Past the knee the agent explores compulsively -- switching
    saturates near 12 ports/slot and the rate falls back. The optimum is interior.

WHY beta_w = 0.25 IS MARKED AS THE OPERATING POINT. It maximises RATE. It does not
maximise the eta_sw = 1 objective -- beta_w = 0.05 does, because it barely moves and
so pays almost no switching. We keep 0.25 deliberately: an agent that reconfigures
0.05 ports per slot is not solving a port-selection problem, it is a static array that
wins the objective by declining to be a fluid antenna. Both arms are plotted so the
reader can see the trade rather than be told about it.

STYLE matches Fig. 2: full box frame, dotted grid, framed legend, filled markers,
black dashed ceiling. The numbers that used to be written onto the axes (the
rate-greedy fraction of genie, the operating point's fraction of genie) are in the
caption and in Section IV-D instead -- text belongs in text.

Run:  python make_fig_ablation.py [in.json] [out.pdf]
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
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results_tm" / "ablation_beta.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper1" / "figs" / "fig3_ablation.pdf"

C_RATE, C_SW, C_GENIE, C_FLAT = "#0072B2", "#D55E00", "black", "#56B4E9"
OP_BETA = 0.25

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.0,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "in", "ytick.direction": "in",
    "pdf.fonttype": 42,
})


def main() -> int:
    d = json.loads(IN.read_text(encoding="utf-8"))
    m = d["meta"]
    n = m["MC_done"]
    betas = [float(b) for b in m["beta_list"]]
    gr = float(np.mean(d["genie_rate"]))

    def series(field, eta):
        return np.array([np.mean(d[field][f"{b:g}|{eta:g}"]) for b in betas])

    r1, w1 = series("rate", 1.0), series("switch", 1.0)
    r4 = series("rate", 4.0)
    # beta_w = 0 is a genuinely unstable arm (it gets stuck in a different place on
    # every seed), so its spread is the honest thing to show.
    se1 = np.array([np.std(d["rate"][f"{b:g}|1"], ddof=1) / np.sqrt(n) for b in betas])

    x = np.arange(len(betas))                 # categorical: beta values cluster near 0
    fig, ax = plt.subplots(figsize=(3.5, 2.02))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", ls=":", lw=0.5, color="#B0B0B0")

    ax.axhline(gr, color=C_GENIE, ls="--", lw=1.0, zorder=3,
               label="full-CSI genie (no pilot/switching cost)")

    ax.fill_between(x, r1 - 1.96 * se1, r1 + 1.96 * se1, color=C_RATE,
                    alpha=0.16, lw=0, zorder=2)
    ax.plot(x, r1, "-o", color=C_RATE, ms=3.8, mfc=C_RATE, mec=C_RATE, zorder=5,
            label="sum rate, $\\eta_{\\mathrm{sw}}\\!=\\!1$")
    ax.plot(x, r4, "--^", color=C_FLAT, ms=3.6, mfc=C_FLAT, mec=C_FLAT, lw=1.0,
            zorder=4, label="switching equalised ($\\eta_{\\mathrm{sw}}\\!=\\!4$)")

    i_op = betas.index(OP_BETA)
    ax.plot(x[i_op], r1[i_op], "s", ms=5.6, mfc=C_SW, mec=C_SW, ls="none",
            zorder=7, label="operating point, $\\beta_w=%g$" % OP_BETA)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_xlabel("epistemic weight $\\beta_w$", labelpad=2)
    ax.set_ylabel("sum rate (bits/s/Hz)", labelpad=2)
    ax.set_xlim(-0.4, len(betas) - 0.6)
    # zoom out so the legend has a clear quadrant at the bottom right: the
    # rate curves are pushed into the upper half and the switching axis is
    # offset so its curve stays there too.
    ax.set_ylim(min(r1.min(), r4.min()) - 5.5, gr + 1.0)

    # ---- switching on the right axis
    ax2 = ax.twinx()
    ax2.plot(x, w1, "-.d", color=C_SW, lw=1.0, ms=3.4, mfc="none", mec=C_SW,
             mew=0.9, zorder=5, label="ports reconfigured / slot")
    ax2.set_ylabel("ports reconfigured / slot", color=C_SW, labelpad=2)
    ax2.tick_params(axis="y", colors=C_SW, direction="in", width=0.7, size=3)
    ax2.set_ylim(-9.5, max(w1.max(), 12.5) * 1.10)
    ax2.set_yticks([0, 5, 10])

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True,
                    framealpha=1.0, edgecolor="black", fancybox=False,
                    handlelength=1.9, borderaxespad=0.4, labelspacing=0.28,
                    borderpad=0.4)
    leg.set_zorder(10)
    leg.get_frame().set_linewidth(0.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)

    print(f"wrote {OUT}  ({n} seeds)")
    gap0 = r4[0]
    gapb = max(r4[1:])
    print(f"  matched switching (eta=4): beta_w=0 -> {gap0:.2f} ({100*gap0/gr:.1f}% genie), "
          f"best beta_w -> {gapb:.2f} ({100*gapb/gr:.1f}%)  =>  +{gapb-gap0:.2f} b/s/Hz")
    print(f"  rate-greedy (beta_w=0, eta=1): {r1[0]:.2f} ({100*r1[0]/gr:.1f}% genie)")
    print(f"  operating point beta_w={OP_BETA:g}: rate {r1[i_op]:.2f} "
          f"({100*r1[i_op]/gr:.1f}% genie), {w1[i_op]:.2f} sw/slot")
    print(f"  rate peaks at beta_w={betas[int(np.argmax(r1))]:g}; "
          f"switching knee between {betas[1]:g} and {OP_BETA:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
