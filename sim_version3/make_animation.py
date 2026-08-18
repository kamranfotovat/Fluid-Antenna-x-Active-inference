"""
Animation of the Active-Inference FAS agent (static regime) -- a TALK / intuition asset.

Shows, slot by slot:
  * the 5x5 FAS port grid coloured by the agent's BELIEF UNCERTAINTY (posterior variance
    per port), with the ACTIVATED ports outlined -- you can watch uncertainty collapse on the
    sensed ports AND on their correlated neighbours (the epistemic term + R at work), then
    re-inflate as CSI ages;
  * a live parameter/variable panel on the left;
  * a running rate/objective plot on the right, with the full-CSI genie as the dashed ceiling.

NOTE: this is a communication tool, not a result. It runs the same locked operating point as
the paper figures (observe-then-precode). Output: animation/aif_port_selection_static.gif

Run from the sim/ directory:  python make_animation.py
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from channel import ChannelSimulator
from agent import AIFAgent, run_genie
from precoding import sinr_and_rates
from config import ACTIVE

# ----------------------------------------------------------------- operating point (from config)
OP = ACTIVE
Nx, Ny = OP.Nx, OP.Ny
N = OP.N
K = OP.K
M = OP.M
RHO = OP.rho
SIGMA2 = OP.sigma2      # 15 dB
SIGMA_E2 = OP.sigma_e2  # ~20 dB pilots
ETA_SW = OP.eta_sw
WX, WY = OP.Wx, OP.Wy
D_MIN = OP.d_min
POS = OP.positions()
SPACING = WX / (Nx - 1) if Nx > 1 else 0.0
T = 45
SEED = 0

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "animation")
os.makedirs(OUTDIR, exist_ok=True)

# Exploration weight -- picks the operating point on the rate/switching frontier:
#   beta_w = 0.25  -> balanced / max-objective : agent LOCKS (~0 switching, ~84-86% rate)
#   beta_w = 0.60  -> max-rate               : agent EXPLORES (~2.5 sw/slot, ~89% rate)
BETA_W = 0.25
OUT = os.path.join(OUTDIR, "aif_port_selection_static.gif")


def switch_count(S, S_prev):
    return 0 if S_prev is None else len(set(S) ^ set(S_prev))


def simulate():
    """Run the agent (observe-then-precode) and record everything per slot."""
    sim = ChannelSimulator(Nx=Nx, Ny=Ny, Wx=WX, Wy=WY, K=K, rho=RHO, beta=1.0, seed=SEED)
    H = sim.generate(T)                              # (T, K, N)

    agent = AIFAgent(R=sim.R, beta=np.ones(K), rho=RHO, sigma_e2=SIGMA_E2, M=M,
                     alpha=1.0, beta_w=BETA_W, eta_sw=ETA_SW, e_sw=1.0, sigma2=SIGMA2, P=1.0,
                     positions=POS, d_min=D_MIN)
    agent.reset()
    rng = np.random.default_rng(1)

    sel = []                                         # selected set per slot
    bvar = np.zeros((T, N))                          # belief uncertainty per port (post-update)
    rate = np.zeros(T); switch = np.zeros(T)
    for t in range(T):
        S = agent.select(first=(t == 0))
        idx = list(S)
        noise = np.sqrt(SIGMA_E2 / 2) * (rng.standard_normal((K, len(idx)))
                                         + 1j * rng.standard_normal((K, len(idx))))
        y = H[t][:, idx] + noise
        agent.bel.update(S, y)                        # observe-then-precode: fresh belief first
        bvar[t] = agent.bel.port_variances().mean(axis=0)
        W = agent.precoder(S)
        Ht = H[t][:, idx].T
        rate[t] = float(sinr_and_rates(Ht, W, agent.sigma2)[1].sum())
        switch[t] = switch_count(S, agent.S_prev)
        agent.S_prev = S
        sel.append(S)

    genie = run_genie(H, M, sigma2=SIGMA2, P=1.0, positions=POS, d_min=D_MIN)["rate"]
    return dict(sel=sel, bvar=bvar, rate=rate, switch=switch, genie=genie)


def build(data):
    sel, bvar, rate, switch, genie = (data[k] for k in ("sel", "bvar", "rate", "switch", "genie"))
    cum_sw = np.cumsum(switch)
    cum_obj = np.cumsum(rate - ETA_SW * switch) / (np.arange(T) + 1)

    fig = plt.figure(figsize=(11.2, 4.4))
    gs = GridSpec(1, 3, width_ratios=[1.05, 1.25, 1.5], wspace=0.28,
                  left=0.02, right=0.97, top=0.9, bottom=0.12)
    ax_txt = fig.add_subplot(gs[0]); ax_txt.axis("off")
    ax_grid = fig.add_subplot(gs[1])
    ax_ts = fig.add_subplot(gs[2])

    fig.suptitle(f"Active-Inference FAS Port Selection  ·  κ = {BETA_W:g},  η_sw = {ETA_SW:g}"
                 f"  ·  observe-then-precode", fontsize=11, weight="bold")

    def update(t):
        S = sel[t]
        # ---- left: parameters + live variables ----
        ax_txt.clear(); ax_txt.axis("off")
        dmin_txt = "off" if D_MIN is None else f"{D_MIN:g}λ"
        fixed = (f"PARAMETERS\n"
                 f"  N = {N}  ({Nx}×{Ny}, {SPACING:g}λ)\n"
                 f"  M = {M}   ({100*M/N:.0f}% budget)\n"
                 f"  d_min = {dmin_txt}\n"
                 f"  K = {K}  users\n"
                 f"  SNR = 15 dB\n"
                 f"  σ_e² = 1e-3\n"
                 f"  ρ = {RHO}\n"
                 f"  β_w (κ) = {BETA_W}\n"
                 f"  η_sw = {ETA_SW}\n")
        pct = 100 * rate[t] / genie[t] if genie[t] > 1e-9 else 0.0
        live = (f"SLOT  t = {t+1} / {T}\n"
                f"  active ports = {sorted(S)}\n"
                f"  sum rate = {rate[t]:5.2f}  b/s/Hz\n"
                f"  genie    = {genie[t]:5.2f}  ({pct:3.0f}%)\n"
                f"  switches this slot = {int(switch[t])}\n"
                f"  cumulative switches = {int(cum_sw[t])}\n"
                f"  avg objective = {cum_obj[t]:5.2f}\n"
                f"  mean belief unc. = {bvar[t].mean():.3f}")
        ax_txt.text(0.0, 1.0, fixed, va="top", ha="left", family="monospace", fontsize=8.6,
                    transform=ax_txt.transAxes, color="#333333")
        ax_txt.text(0.0, 0.44, live, va="top", ha="left", family="monospace", fontsize=8.8,
                    transform=ax_txt.transAxes, color="#0b3d91")

        # ---- middle: belief-uncertainty grid + activated ports ----
        ax_grid.clear()
        grid = bvar[t].reshape(Ny, Nx)
        im = ax_grid.imshow(grid, origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
        lw = 2.6 if Nx <= 8 else 1.6                  # thinner outlines on a dense grid
        for n in S:
            x, y = n % Nx, n // Nx
            ax_grid.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                                        edgecolor="#39FF14", lw=lw))
        step = max(1, Nx // 5)                        # keep ticks readable on a dense grid
        ax_grid.set_xticks(range(0, Nx, step)); ax_grid.set_yticks(range(0, Ny, step))
        ax_grid.set_title("belief uncertainty  (green = activated port)", fontsize=9.2)
        ax_grid.set_xlabel("port x"); ax_grid.set_ylabel("port y")

        # ---- right: rate / objective time series ----
        ax_ts.clear()
        tt = np.arange(1, t + 2)
        ax_ts.plot(tt, genie[:t + 1], "--", color="#888888", lw=1.6, label="genie rate (full CSI)")
        ax_ts.plot(tt, rate[:t + 1], "-", color="#0b3d91", lw=2.0, label="AIF rate (20% CSI)")
        ax_ts.plot(tt, cum_obj[:t + 1], "-", color="#c81e1e", lw=1.6, label="AIF avg objective")
        sw_t = tt[switch[:t + 1] > 0]
        if len(sw_t):
            ax_ts.plot(sw_t, rate[:t + 1][switch[:t + 1] > 0], "v", color="#ff8c00",
                       ms=6, label="switch event")
        ax_ts.set_xlim(1, T); ax_ts.set_ylim(0, max(genie.max(), rate.max()) * 1.15)
        ax_ts.set_xlabel("time slot t"); ax_ts.set_ylabel("bits/s/Hz")
        ax_ts.legend(loc="lower right", fontsize=7.6, framealpha=0.9)
        ax_ts.grid(alpha=0.25)
        return []

    anim = FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(OUT, writer=PillowWriter(fps=4))
    plt.close(fig)
    print("saved", OUT)
    print(f"AIF mean rate {data['rate'].mean():.2f} | genie {data['genie'].mean():.2f} "
          f"({100*data['rate'].mean()/data['genie'].mean():.0f}%) | "
          f"total switches {int(data['switch'].sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta_w", type=float, default=0.25,
                    help="exploration weight: 0.25 balanced/locks, 0.60 max-rate/explores")
    ap.add_argument("--eta_sw", type=float, default=1.0,
                    help="switching-penalty weight (<1 penalizes switching less -> more churn)")
    args = ap.parse_args()
    BETA_W = args.beta_w
    ETA_SW = args.eta_sw
    OUT = os.path.join(OUTDIR, f"aif_port_selection_eta{ETA_SW:.2f}_bw{BETA_W:.2f}.gif")
    build(simulate())
