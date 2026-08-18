"""
Meeting GIFs -- S1 hybrid agent (n_rf=6 -> reproduces the digital results exactly).

Same visual as make_animation.py but at the OP_V3 operating point with:
  N = 441 (21x21, 2x2 lambda),  M in {10, 12} active,  n_rf = 6 RF chains (hybrid transmit),
  beta_w = 0.6 (max-rate / explores),  eta_sw = 1.0,  OP_V3 scenario (K=3, beta=[1,0.7,1.3]).

Output -> ../animation_meeting/  (kept SEPARATE from the earlier animation/ gifs).

Run from sim_version3/:  python make_animation_meeting.py
"""

from __future__ import annotations

import os
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
from config import OP_V3

OP = OP_V3
Nx, Ny, N, K = OP.Nx, OP.Ny, OP.N, OP.K
WX, WY = OP.Wx, OP.Wy
RHO, SIGMA2, SIGMA_E2 = OP.rho, OP.sigma2, OP.sigma_e2
D_MIN, POS = OP.d_min, OP.positions()
BETA = OP.beta
SPACING = WX / (Nx - 1) if Nx > 1 else 0.0
N_RF = 6
BETA_W = 0.6
ETA_SW = 1.0
T = 45
SEED = 0

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "animation_meeting")
os.makedirs(OUTDIR, exist_ok=True)


def switch_count(S, S_prev):
    return 0 if S_prev is None else len(set(S) ^ set(S_prev))


def simulate(M):
    sim = ChannelSimulator(Nx=Nx, Ny=Ny, Wx=WX, Wy=WY, K=K, rho=RHO, beta=BETA, seed=SEED)
    H = sim.generate(T)
    agent = AIFAgent(R=sim.R, beta=BETA, rho=RHO, sigma_e2=SIGMA_E2, M=M,
                     alpha=1.0, beta_w=BETA_W, eta_sw=ETA_SW, e_sw=1.0, sigma2=SIGMA2, P=1.0,
                     positions=POS, d_min=D_MIN, n_rf=N_RF)          # hybrid transmit (n_rf=6)
    agent.reset()
    rng = np.random.default_rng(1)
    sel = []; bvar = np.zeros((T, N)); rate = np.zeros(T); switch = np.zeros(T)
    for t in range(T):
        S = agent.select(first=(t == 0))
        idx = list(S)
        noise = np.sqrt(SIGMA_E2 / 2) * (rng.standard_normal((K, len(idx)))
                                         + 1j * rng.standard_normal((K, len(idx))))
        y = H[t][:, idx] + noise
        agent.bel.update(S, y)                                       # observe-then-precode
        bvar[t] = agent.bel.port_variances().mean(axis=0)
        W = agent.precoder(S)                                        # hybrid F_RF W_BB (n_rf=6)
        Ht = H[t][:, idx].T
        rate[t] = float(sinr_and_rates(Ht, W, agent.sigma2)[1].sum())
        switch[t] = switch_count(S, agent.S_prev)
        agent.S_prev = S
        sel.append(S)
    genie = run_genie(H, M, sigma2=SIGMA2, P=1.0, positions=POS, d_min=D_MIN, n_rf=N_RF)["rate"]
    return dict(sel=sel, bvar=bvar, rate=rate, switch=switch, genie=genie, M=M)


def build(data):
    sel, bvar, rate, switch, genie, M = (data[k] for k in
                                         ("sel", "bvar", "rate", "switch", "genie", "M"))
    cum_sw = np.cumsum(switch)
    cum_obj = np.cumsum(rate - ETA_SW * switch) / (np.arange(T) + 1)
    vmax = float(max(BETA)) if np.ndim(BETA) else float(BETA)
    out = os.path.join(OUTDIR, f"aif_hybrid_M{M}_nrf{N_RF}_bw{BETA_W:.2f}_eta{ETA_SW:.2f}.gif")

    fig = plt.figure(figsize=(11.2, 4.4))
    gs = GridSpec(1, 3, width_ratios=[1.05, 1.25, 1.5], wspace=0.28,
                  left=0.02, right=0.97, top=0.9, bottom=0.12)
    ax_txt = fig.add_subplot(gs[0]); ax_txt.axis("off")
    ax_grid = fig.add_subplot(gs[1])
    ax_ts = fig.add_subplot(gs[2])
    fig.suptitle(f"Active-Inference FAS (hybrid, n_rf={N_RF})  ·  M={M},  β_w={BETA_W:g},  "
                 f"η_sw={ETA_SW:g}  ·  observe-then-precode", fontsize=11, weight="bold")

    def update(t):
        S = sel[t]
        ax_txt.clear(); ax_txt.axis("off")
        dmin_txt = "off" if D_MIN is None else f"{D_MIN:g}λ"
        fixed = (f"PARAMETERS\n"
                 f"  N = {N}  ({Nx}×{Ny}, {SPACING:g}λ)\n"
                 f"  M = {M}   ({100*M/N:.0f}% budget)\n"
                 f"  n_rf = {N_RF}  (hybrid)\n"
                 f"  d_min = {dmin_txt}\n"
                 f"  K = {K}  users\n"
                 f"  SNR = 15 dB\n"
                 f"  σ_e² = 1e-3\n"
                 f"  ρ = {RHO}\n"
                 f"  β_w = {BETA_W}\n"
                 f"  η_sw = {ETA_SW}\n")
        pct = 100 * rate[t] / genie[t] if genie[t] > 1e-9 else 0.0
        live = (f"SLOT  t = {t+1} / {T}\n"
                f"  #active = {len(S)}\n"
                f"  sum rate = {rate[t]:5.2f}  b/s/Hz\n"
                f"  genie    = {genie[t]:5.2f}  ({pct:3.0f}%)\n"
                f"  switches this slot = {int(switch[t])}\n"
                f"  cumulative switches = {int(cum_sw[t])}\n"
                f"  avg objective = {cum_obj[t]:5.2f}\n"
                f"  mean belief unc. = {bvar[t].mean():.3f}")
        ax_txt.text(0.0, 1.0, fixed, va="top", ha="left", family="monospace", fontsize=8.6,
                    transform=ax_txt.transAxes, color="#333333")
        ax_txt.text(0.0, 0.40, live, va="top", ha="left", family="monospace", fontsize=8.8,
                    transform=ax_txt.transAxes, color="#0b3d91")

        ax_grid.clear()
        grid = bvar[t].reshape(Ny, Nx)
        ax_grid.imshow(grid, origin="lower", cmap="inferno", vmin=0.0, vmax=vmax)
        for n in S:
            x, y = n % Nx, n // Nx
            ax_grid.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                                        edgecolor="#39FF14", lw=1.6))
        step = max(1, Nx // 5)
        ax_grid.set_xticks(range(0, Nx, step)); ax_grid.set_yticks(range(0, Ny, step))
        ax_grid.set_title("belief uncertainty  (green = activated port)", fontsize=9.2)
        ax_grid.set_xlabel("port x"); ax_grid.set_ylabel("port y")

        ax_ts.clear()
        tt = np.arange(1, t + 2)
        ax_ts.plot(tt, genie[:t + 1], "--", color="#888888", lw=1.6, label="genie rate (full CSI)")
        ax_ts.plot(tt, rate[:t + 1], "-", color="#0b3d91", lw=2.0, label="AIF rate (partial CSI)")
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
    anim.save(out, writer=PillowWriter(fps=4))
    plt.close(fig)
    print(f"saved {out}")
    print(f"  M={M}: AIF mean rate {rate.mean():.2f} | genie {genie.mean():.2f} "
          f"({100*rate.mean()/genie.mean():.0f}%) | total switches {int(switch.sum())}")


if __name__ == "__main__":
    for M in (10, 12):
        build(simulate(M))
