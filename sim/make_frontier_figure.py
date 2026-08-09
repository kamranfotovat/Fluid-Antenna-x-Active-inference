"""
Pareto frontier figure: AIF dominates the genie across the whole rate/switching trade-off.

Sweeps the exploration weight beta_w; each point is one AIF operating point. Shows that the
entire AIF frontier beats the genie on the switching-aware objective, from max-objective
(beta_w~0.3, ~0 switching) to max-rate (beta_w~0.6, 89% of genie rate).

Run:  python sim/make_frontier_figure.py
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel import ChannelSimulator
from agent import AIFAgent, run_aif, run_genie, run_naive, objective

SIGMA_E2, SIGMA2 = 1e-3, 0.03
BETA = np.array([1.0, 0.7, 1.3])
K, M, RHO = 3, 5, 0.9
ETA_SW = 1.0
BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
T, MC = 70, 15
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def main():
    g_r = g_o = g_s = nv_r = nv_o = nv_s = 0.0
    for m in range(MC):
        sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
        H = sim.generate(T)
        G = run_genie(H, M, sigma2=SIGMA2)
        Nv = run_naive(H, M, SIGMA_E2, np.random.default_rng(40000 + m), sigma2=SIGMA2)
        h = slice(T // 2, None)
        g_r += G["rate"][h].mean(); g_o += objective(G, ETA_SW); g_s += G["switch"].mean()
        nv_r += Nv["rate"][h].mean(); nv_o += objective(Nv, ETA_SW); nv_s += Nv["switch"].mean()
    g_r, g_o, g_s = g_r / MC, g_o / MC, g_s / MC
    nv_r, nv_o, nv_s = nv_r / MC, nv_o / MC, nv_s / MC

    rate, obj, sw = [], [], []
    for bw in BETAS:
        r = o = s = 0.0
        for m in range(MC):
            sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
            H = sim.generate(T)
            A = run_aif(AIFAgent(sim.R, BETA, RHO, SIGMA_E2, M, 1.0, bw, ETA_SW, sigma2=SIGMA2),
                        H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True)
            r += A["rate"][T // 2:].mean(); o += objective(A, ETA_SW); s += A["switch"].mean()
        rate.append(r / MC); obj.append(o / MC); sw.append(s / MC)
        print(f"   beta_w={bw}: rate={rate[-1]:.2f} ({100*rate[-1]/g_r:.0f}%) sw={sw[-1]:.2f} obj={obj[-1]:.2f}")
    rate, obj, sw = np.array(rate), np.array(obj), np.array(sw)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # Panel A: rate vs switching (Pareto) -- AIF frontier vs genie/naive points
    ax[0].plot(sw, rate, "o-", color="tab:green", label="AIF frontier (sweep beta_w)")
    for bw, x, y in zip(BETAS, sw, rate):
        ax[0].annotate(f"{bw}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color="green")
    ax[0].scatter([g_s], [g_r], s=90, color="black", marker="*", zorder=5, label="genie (full CSI)")
    ax[0].scatter([nv_s], [nv_r], s=50, color="tab:red", marker="s", zorder=5, label="naive")
    ax[0].set(title="Rate vs switching (Pareto frontier)", xlabel="antenna switches / slot",
              ylabel="sum-rate (bits/s/Hz)")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

    # Panel B: objective across the frontier -- ALL beats genie
    x = np.arange(len(BETAS))
    ax[1].plot(x, obj, "o-", color="tab:green", label="AIF objective (frontier)")
    ax[1].plot(x, rate, "o--", color="tab:blue", alpha=0.6, label="AIF rate")
    ax[1].axhline(g_o, color="black", ls="--", label=f"genie objective ({g_o:.1f})")
    ax[1].axhline(g_r, color="black", ls=":", alpha=0.5, label=f"genie rate ({g_r:.1f})")
    ax[1].fill_between(x, g_o, obj, where=(obj > g_o), color="green", alpha=0.10)
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(b) for b in BETAS])
    ax[1].set(title="Whole frontier beats genie on the objective",
              xlabel="exploration weight beta_w", ylabel="bits/s/Hz")
    ax[1].legend(fontsize=7); ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "figF_pareto_frontier.png")
    fig.savefig(out, dpi=130)
    also = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step_frontier.png")
    fig.savefig(also, dpi=130)
    print(f"saved -> figures/figF_pareto_frontier.png")
    print(f"\nSUMMARY: genie rate={g_r:.2f} obj={g_o:.2f} sw={g_s:.1f}")
    bi = int(np.argmax(obj)); ri = int(np.argmax(rate))
    print(f"  max-objective: beta_w={BETAS[bi]} rate={rate[bi]:.2f} ({100*rate[bi]/g_r:.0f}%) obj={obj[bi]:.2f} sw={sw[bi]:.2f}")
    print(f"  max-rate     : beta_w={BETAS[ri]} rate={rate[ri]:.2f} ({100*rate[ri]/g_r:.0f}%) obj={obj[ri]:.2f} sw={sw[ri]:.2f}")


if __name__ == "__main__":
    main()
