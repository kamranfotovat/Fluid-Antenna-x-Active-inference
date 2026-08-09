"""
Assemble ALL paper figures into ../figures/.

Generates the two remaining sweeps (Fig A: observation budget; Fig D: exploration weight)
and copies the already-verified figures from the step checks into the folder with clean,
paper-oriented names. Re-run any time to regenerate the folder.

Run:  python sim/make_paper_figures.py
"""

from __future__ import annotations

import os
import shutil
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel import ChannelSimulator                                   # noqa: E402
from agent import (AIFAgent, run_aif, run_genie, run_naive,            # noqa: E402
                   run_random_partial, objective)

# shared operating point
SIGMA_E2, SIGMA2 = 1e-2, 0.03           # 15 dB
BETA = np.array([1.0, 0.7, 1.3])
K, N, RHO = 3, 25, 0.9
ETA_SW, BETA_W = 1.0, 0.25
T, MC = 70, 12

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "figures")


def _aif(R, M, beta_w=BETA_W):
    return AIFAgent(R, BETA, RHO, SIGMA_E2, M, 1.0, beta_w, ETA_SW, sigma2=SIGMA2)


def fig_A_observation_budget():
    """Fig A (headline): performance vs observation budget M/N. AIF observes only M of N ports;
    genie sees all N. Shows AIF tracks a high fraction of genie while observing a fraction."""
    Ms = [3, 4, 5, 6, 7, 8, 10]
    g_r, a_r, n_r = [], [], []
    g_o, a_o, n_o = [], [], []
    for M in Ms:
        gr = ar = nr = go = ao = no = 0.0
        for m in range(MC):
            sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
            H = sim.generate(T)
            G = run_genie(H, M, sigma2=SIGMA2)
            A = run_aif(_aif(sim.R, M), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True)
            Nv = run_naive(H, M, SIGMA_E2, np.random.default_rng(40000 + m), sigma2=SIGMA2)
            h = slice(T // 2, None)
            gr += G["rate"][h].mean(); ar += A["rate"][h].mean(); nr += Nv["rate"][h].mean()
            go += objective(G, ETA_SW); ao += objective(A, ETA_SW); no += objective(Nv, ETA_SW)
        g_r.append(gr / MC); a_r.append(ar / MC); n_r.append(nr / MC)
        g_o.append(go / MC); a_o.append(ao / MC); n_o.append(no / MC)
        print(f"   [Fig A] M={M} ({100*M/N:.0f}%): genie rate={g_r[-1]:.1f} | AIF rate={a_r[-1]:.1f} "
              f"obj={a_o[-1]:.1f} | naive obj={n_o[-1]:.1f}")
    budget = 100 * np.array(Ms) / N

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(budget, g_r, "k--o", label="genie (full CSI)")
    ax[0].plot(budget, a_r, "o-", color="tab:green", label="AIF (ours)")
    ax[0].plot(budget, n_r, "s-", color="tab:red", label="naive (no inference)")
    ax[0].set_title("Rate vs observation budget")
    ax[0].set_xlabel("observation budget  M/N  (% of ports measured)")
    ax[0].set_ylabel("sum-rate (bits/s/Hz)")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

    ax[1].plot(budget, g_o, "k--o", label="genie (full CSI)")
    ax[1].plot(budget, a_o, "o-", color="tab:green", label="AIF (ours)")
    ax[1].plot(budget, n_o, "s-", color="tab:red", label="naive (no inference)")
    ax[1].set_title("Switching-aware objective vs observation budget")
    ax[1].set_xlabel("observation budget  M/N  (% of ports measured)")
    ax[1].set_ylabel("objective: rate - switching (bits/s/Hz)")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "figA_observation_budget.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"   saved {out}")


def fig_D_exploration_weight():
    """Fig D: exploration-weight (beta_w) sweep -- isolates the active-sensing benefit and shows
    the sweet spot; too much exploration wins rate but loses the objective via switching churn."""
    betas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    rate, obj, sw = [], [], []
    for bw in betas:
        r = o = s = 0.0
        for m in range(MC):
            sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
            H = sim.generate(T)
            A = run_aif(_aif(sim.R, 5, beta_w=bw), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True)
            r += A["rate"][T // 2:].mean(); o += objective(A, ETA_SW); s += A["switch"].mean()
        rate.append(r / MC); obj.append(o / MC); sw.append(s / MC)
        print(f"   [Fig D] beta_w={bw}: rate={rate[-1]:.2f} obj={obj[-1]:.2f} switch={sw[-1]:.2f}")
    x = np.arange(len(betas))
    best = int(np.argmax(obj))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, rate, "o-", color="tab:blue", label="rate")
    ax[0].plot(x, obj, "s-", color="tab:orange", label="objective (rate - switch)")
    ax[0].axvline(best, color="grey", ls=":", alpha=0.7)
    ax[0].annotate("sweet spot", (best, obj[best]), textcoords="offset points", xytext=(8, -18), fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([str(b) for b in betas])
    ax[0].set_title("Exploration weight sweep")
    ax[0].set_xlabel("epistemic weight  beta_w  (0 = no exploration)")
    ax[0].set_ylabel("bits/s/Hz"); ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

    ax[1].plot(x, sw, "^-", color="tab:red")
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(b) for b in betas])
    ax[1].set_title("Antenna switching vs exploration weight")
    ax[1].set_xlabel("epistemic weight  beta_w"); ax[1].set_ylabel("ports moved / slot")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "figD_exploration_weight.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"   saved {out}")


def copy_existing():
    """Copy the already-verified step figures into the folder with paper-oriented names."""
    mapping = {
        "step7_baselines.png":        "figR_results_baselines.png",       # main results comparison
        "step7_protocol_doppler.png": "figC_protocol_doppler.png",        # robustness / protocol
        "step8_learning_mismatch.png": "figE_learning_mismatch.png",      # objection-proofing
        "step6_closed_loop_check.png": "figLC_closed_loop_learning.png",  # closed-loop learning curve
        # diagnostics (verification plots, not headline figures)
        "step0_channel_check.png": "diag_step0_channel.png",
        "step3_belief_check.png":  "diag_step3_belief_calibration.png",
        "step4_efe_terms_check.png": "diag_step4_efe_terms.png",
        "step5_greedy_check.png":  "diag_step5_greedy_optimality.png",
    }
    for src, dst in mapping.items():
        s = os.path.join(HERE, src)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(FIGDIR, dst))
            print(f"   copied {src} -> {dst}")
        else:
            print(f"   (missing {src} -- run its verify_*.py first)")


def write_index():
    lines = [
        "# Paper figures", "",
        "Operating point unless noted: N=25 ports (5x5), K=3 users, M=5 activated (20% observation "
        "budget), 15 dB, rho=0.9, beta_w=0.25, eta_sw=1, observe-then-precode.", "",
        "## Headline / results",
        "- **figA_observation_budget.png** - performance vs observation budget M/N (the headline: "
        "AIF tracks a high fraction of the genie while measuring a fraction of ports).",
        "- **figR_results_baselines.png** - AIF vs genie / naive / random: rate is comparable but AIF "
        "wins the switching-aware objective and barely moves the antenna.",
        "- **figC_protocol_doppler.png** - observe-then-precode reaches ~80-89% of genie and is robust "
        "to Doppler; predict-then-act (ablation) collapses.",
        "- **figD_exploration_weight.png** - exploration-weight sweep: sweet spot around beta_w=0.1-0.25.",
        "- **figE_learning_mismatch.png** - learning R from data adapts to a non-Jakes channel "
        "(objection-proofing); learned R_hat matches the oracle.",
        "- **figLC_closed_loop_learning.png** - closed-loop learning curve + rate/objective bars.", "",
        "## Diagnostics (verification plots)",
        "- diag_step0_channel.png - channel generator (Jakes R + AR(1)).",
        "- diag_step3_belief_calibration.png - Kalman belief calibration + CSI aging.",
        "- diag_step4_efe_terms.png - EFE terms (submodular epistemic, conservative pragmatic).",
        "- diag_step5_greedy_optimality.png - greedy vs exhaustive + latency.", "",
        "Regenerate with `python sim/make_paper_figures.py` (after running the verify_*.py that "
        "produce the step plots).",
    ]
    with open(os.path.join(FIGDIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("   wrote figures/README.md")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("Generating Fig A (observation budget) ...")
    fig_A_observation_budget()
    print("Generating Fig D (exploration weight) ...")
    fig_D_exploration_weight()
    print("Copying existing step figures ...")
    copy_existing()
    write_index()
    print("\nDONE -> figures/")


if __name__ == "__main__":
    main()
