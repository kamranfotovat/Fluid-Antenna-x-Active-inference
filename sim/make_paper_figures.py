"""
Generate ALL paper figures into ../figures/, every one at the SAME locked operating point
so all numbers are mutually consistent.

LOCKED OPERATING POINT
  N=25 (5x5), K=3, M=5 (20% observation budget), 15 dB (sigma^2=0.03),
  sigma_e^2=1e-3 (~20 dB pilots), rho=0.9, beta_w=0.25, eta_sw=1, observe-then-precode.

Headline results at this point: AIF rate ~13.8 (~84% of genie), objective ~13.7 (beats genie),
~0 switching.

Run:  python sim/make_paper_figures.py     (a few minutes; regenerates the whole folder)
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel import ChannelSimulator, spatial_correlation, port_positions   # noqa: E402
from agent import (AIFAgent, run_aif, run_genie, run_naive,                  # noqa: E402
                   run_random_partial, objective)
from learning import (gather_correlation, exponential_correlation, set_correlation)  # noqa: E402

# ---- locked operating point -------------------------------------------------
SIGMA_E2, SIGMA2 = 1e-3, 0.03
BETA = np.array([1.0, 0.7, 1.3])
K, N, RHO = 3, 25, 0.9
ETA_SW, BETA_W, M0 = 1.0, 0.25, 5
T, MC = 80, 15

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "figures")
OP = (f"N=25, K=3, M=5 (20% obs), 15 dB, sigma_e^2=1e-3, rho=0.9, "
      f"beta_w=0.25, observe-then-precode")


def _aif(R, M=M0, beta_w=BETA_W, sigma_e2=SIGMA_E2):
    return AIFAgent(R, BETA, RHO, sigma_e2, M, 1.0, beta_w, ETA_SW, sigma2=SIGMA2)


def _sim(seed):
    return ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=seed)


# ============================================================ Fig A: observation budget
def fig_A():
    Ms = [3, 4, 5, 6, 7, 8, 10]
    g_r, a_r, n_r, g_o, a_o, n_o = ([] for _ in range(6))
    for M in Ms:
        gr = ar = nr = go = ao = no = 0.0
        for m in range(MC):
            sim = _sim(m); H = sim.generate(T)
            G = run_genie(H, M, sigma2=SIGMA2)
            A = run_aif(_aif(sim.R, M), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True)
            Nv = run_naive(H, M, SIGMA_E2, np.random.default_rng(40000 + m), sigma2=SIGMA2)
            h = slice(T // 2, None)
            gr += G["rate"][h].mean(); ar += A["rate"][h].mean(); nr += Nv["rate"][h].mean()
            go += objective(G, ETA_SW); ao += objective(A, ETA_SW); no += objective(Nv, ETA_SW)
        g_r.append(gr / MC); a_r.append(ar / MC); n_r.append(nr / MC)
        g_o.append(go / MC); a_o.append(ao / MC); n_o.append(no / MC)
        print(f"   [A] M={M} ({100*M/N:.0f}%): genie {g_r[-1]:.1f} | AIF {a_r[-1]:.1f} ({100*a_r[-1]/g_r[-1]:.0f}%) obj {a_o[-1]:.1f}")
    b = 100 * np.array(Ms) / N
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(b, g_r, "k--o", label="genie (full CSI)")
    ax[0].plot(b, a_r, "o-", color="tab:green", label="AIF (ours)")
    ax[0].plot(b, n_r, "s-", color="tab:red", label="naive (no inference)")
    ax[0].set(title="Rate vs observation budget", xlabel="observation budget M/N (% of ports)",
              ylabel="sum-rate (bits/s/Hz)")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
    ax[1].plot(b, g_o, "k--o", label="genie (full CSI)")
    ax[1].plot(b, a_o, "o-", color="tab:green", label="AIF (ours)")
    ax[1].plot(b, n_o, "s-", color="tab:red", label="naive (no inference)")
    ax[1].set(title="Switching-aware objective vs observation budget",
              xlabel="observation budget M/N (% of ports)", ylabel="objective: rate - switching")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)
    _save(fig, "figA_observation_budget.png")


# ============================================================ Fig R: results bars
def fig_R():
    acc = {k: dict(rate=0.0, obj=0.0, sw=0.0) for k in ["genie", "aif", "naive", "rand"]}
    for m in range(MC):
        sim = _sim(m); H = sim.generate(T)
        runs = {"genie": run_genie(H, M0, sigma2=SIGMA2),
                "aif": run_aif(_aif(sim.R), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True),
                "naive": run_naive(H, M0, SIGMA_E2, np.random.default_rng(40000 + m), sigma2=SIGMA2),
                "rand": run_random_partial(H, M0, SIGMA_E2, np.random.default_rng(50000 + m), sigma2=SIGMA2)}
        for k, r in runs.items():
            acc[k]["rate"] += r["rate"][T // 2:].mean(); acc[k]["obj"] += objective(r, ETA_SW)
            acc[k]["sw"] += r["switch"].mean()
    for k in acc:
        for f in acc[k]:
            acc[k][f] /= MC
    print(f"   [R] genie {acc['genie']['rate']:.1f}/{acc['genie']['obj']:.1f} | "
          f"AIF {acc['aif']['rate']:.1f}/{acc['aif']['obj']:.1f} | naive {acc['naive']['rate']:.1f}/{acc['naive']['obj']:.1f}")
    order = ["genie", "aif", "naive", "rand"]
    labels = ["genie\n(full CSI)", "AIF\n(ours)", "naive\n(no inference)", "random"]
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(x - w / 2, [acc[k]["rate"] for k in order], w, label="rate", color="tab:blue", alpha=0.85)
    ax[0].bar(x + w / 2, [acc[k]["obj"] for k in order], w, label="objective (rate - switch)",
              color="tab:orange", alpha=0.85)
    ax[0].axhline(acc["aif"]["obj"], color="tab:orange", ls=":", alpha=0.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set(ylabel="bits/s/Hz", title="Rate vs switching-aware objective")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3, axis="y")
    ax[1].bar(x, [acc[k]["sw"] for k in order], color="tab:red", alpha=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set(ylabel="ports moved / slot", title="Antenna switching (lower = cheaper)")
    ax[1].grid(True, alpha=0.3, axis="y")
    _save(fig, "figR_results_baselines.png")


# ============================================================ Fig C: protocol + Doppler
def fig_C():
    rhos = [0.95, 0.9, 0.8, 0.7, 0.6]
    g, sf, pa = [], [], []
    for rho in rhos:
        gg = ss = pp = 0.0
        for m in range(MC):
            sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=BETA, seed=m)
            H = sim.generate(T)
            ag = AIFAgent(sim.R, BETA, rho, SIGMA_E2, M0, 1.0, BETA_W, ETA_SW, sigma2=SIGMA2)
            ag2 = AIFAgent(sim.R, BETA, rho, SIGMA_E2, M0, 1.0, BETA_W, ETA_SW, sigma2=SIGMA2)
            gg += run_genie(H, M0, sigma2=SIGMA2)["rate"][T // 2:].mean()
            ss += run_aif(ag, H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True)["rate"][T // 2:].mean()
            pp += run_aif(ag2, H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=False)["rate"][T // 2:].mean()
        g.append(gg / MC); sf.append(ss / MC); pa.append(pp / MC)
        print(f"   [C] rho={rho}: genie {g[-1]:.1f} | observe {sf[-1]:.1f} ({100*sf[-1]/g[-1]:.0f}%) | predict {pa[-1]:.1f}")
    g, sf, pa = np.array(g), np.array(sf), np.array(pa)
    rhos = np.array(rhos)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(rhos, g, "k--o", label="genie (full CSI)")
    ax[0].plot(rhos, sf, "o-", color="tab:green", label="observe-then-precode (ours)")
    ax[0].plot(rhos, pa, "s-", color="tab:red", label="predict-then-act (ablation)")
    ax[0].invert_xaxis()
    ax[0].set(title="Rate vs Doppler correlation", xlabel="temporal correlation rho (higher=slower)",
              ylabel="sum-rate (bits/s/Hz)")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
    ax[1].plot(rhos, 100 * sf / g, "o-", color="tab:green", label="observe-then-precode")
    ax[1].plot(rhos, 100 * pa / g, "s-", color="tab:red", label="predict-then-act")
    ax[1].axhline(100, color="k", ls="--", alpha=0.5, label="genie")
    ax[1].invert_xaxis(); ax[1].set_ylim(0, 105)
    ax[1].set(title="Fraction of genie captured (robustness)", xlabel="temporal correlation rho",
              ylabel="% of genie rate")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)
    _save(fig, "figC_protocol_doppler.png")


# ============================================================ Fig D: exploration weight
def fig_D():
    betas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    rate, obj, sw = [], [], []
    for bw in betas:
        r = o = s = 0.0
        for m in range(MC):
            sim = _sim(m); H = sim.generate(T)
            A = run_aif(_aif(sim.R, beta_w=bw), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True)
            r += A["rate"][T // 2:].mean(); o += objective(A, ETA_SW); s += A["switch"].mean()
        rate.append(r / MC); obj.append(o / MC); sw.append(s / MC)
        print(f"   [D] beta_w={bw}: rate {rate[-1]:.2f} obj {obj[-1]:.2f} sw {sw[-1]:.2f}")
    x = np.arange(len(betas)); best = int(np.argmax(obj))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, rate, "o-", color="tab:blue", label="rate")
    ax[0].plot(x, obj, "s-", color="tab:orange", label="objective (rate - switch)")
    ax[0].axvline(best, color="grey", ls=":", alpha=0.7)
    ax[0].annotate("sweet spot", (best, obj[best]), textcoords="offset points", xytext=(8, -18), fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([str(b) for b in betas])
    ax[0].set(title="Exploration weight sweep", xlabel="epistemic weight beta_w (0 = no exploration)",
              ylabel="bits/s/Hz")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
    ax[1].plot(x, sw, "^-", color="tab:red")
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(b) for b in betas])
    ax[1].set(title="Antenna switching vs exploration weight", xlabel="epistemic weight beta_w",
              ylabel="ports moved / slot")
    ax[1].grid(True, alpha=0.3)
    _save(fig, "figD_exploration_weight.png")


# ============================================================ Fig LC: closed-loop learning curve
def fig_LC():
    curve = np.zeros(T)
    acc = {k: dict(rate=0.0, obj=0.0, sw=0.0) for k in ["genie", "aif", "aif0"]}
    for m in range(MC):
        sim = _sim(m); H = sim.generate(T)
        G = run_genie(H, M0, sigma2=SIGMA2)
        A = run_aif(_aif(sim.R), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True)
        A0 = run_aif(_aif(sim.R, beta_w=0.0), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True)
        curve += A["rate"]
        for k, r in [("genie", G), ("aif", A), ("aif0", A0)]:
            acc[k]["rate"] += r["rate"][T // 2:].mean(); acc[k]["obj"] += objective(r, ETA_SW)
            acc[k]["sw"] += r["switch"].mean()
    curve /= MC
    for k in acc:
        for f in acc[k]:
            acc[k][f] /= MC
    print(f"   [LC] genie {acc['genie']['rate']:.1f} | AIF {acc['aif']['rate']:.1f} | AIF(b=0) {acc['aif0']['rate']:.1f}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(T), curve, lw=1.5, color="tab:green", label="AIF (beta_w=0.25)")
    ax[0].axhline(acc["genie"]["rate"], color="k", ls="--", alpha=0.7, label="genie (full CSI)")
    ax[0].axhline(acc["aif0"]["rate"], color="tab:red", ls=":", alpha=0.8, label="AIF (beta_w=0, no explore)")
    ax[0].set(title="Closed-loop learning curve", xlabel="slot t", ylabel="realized sum-rate (bits/s/Hz)")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
    order = ["genie", "aif", "aif0"]
    labels = ["genie\n(full CSI)", "AIF\n(beta=0.25)", "AIF\n(beta=0)"]
    x = np.arange(3); w = 0.38
    ax[1].bar(x - w / 2, [acc[k]["rate"] for k in order], w, label="rate", color="tab:blue", alpha=0.85)
    ax[1].bar(x + w / 2, [acc[k]["obj"] for k in order], w, label="objective", color="tab:orange", alpha=0.85)
    for i, k in enumerate(order):
        ax[1].text(i, max(acc[k]["rate"], acc[k]["obj"]) + 0.15, f"{acc[k]['sw']:.1f} sw",
                   ha="center", fontsize=7, color="gray")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set(ylabel="bits/s/Hz", title="Rate vs objective (switch/slot annotated)")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3, axis="y")
    _save(fig, "figLC_closed_loop_learning.png")


# ============================================================ Fig E: learning under mismatch
def fig_E():
    pos = port_positions(5, 5, 1.0, 1.0)
    R_exp = exponential_correlation(pos, d0=0.3)     # true (non-Jakes)
    R_jakes = spatial_correlation(pos)               # wrongly assumed
    o = mism = lrn = 0.0
    mce = min(MC, 12)
    for m in range(mce):
        sim = _sim(m); set_correlation(sim, R_exp); H = sim.generate(T)
        simw = _sim(m); set_correlation(simw, R_exp)
        Rhat, _ = gather_correlation(simw, 200, M0, SIGMA_E2, np.random.default_rng(70000 + m))
        o += objective(run_aif(_aif(R_exp), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True), ETA_SW)
        mism += objective(run_aif(_aif(R_jakes), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True), ETA_SW)
        lrn += objective(run_aif(_aif(Rhat), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True), ETA_SW)
    o, mism, lrn = o / mce, mism / mce, lrn / mce
    print(f"   [E] oracle {o:.2f} | assumes-Jakes {mism:.2f} | learned {lrn:.2f}")
    simd = _sim(0); set_correlation(simd, R_exp)
    Rhat_show, _ = gather_correlation(simd, 400, M0, SIGMA_E2, np.random.default_rng(1))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, Mtx, ti in [(ax[0], R_exp, "True R (exponential)"),
                       (ax[1], Rhat_show, "Learned R_hat (from data)"),
                       (ax[2], R_jakes, "Assumed R (Jakes, wrong)")]:
        im = a.imshow(Mtx, cmap="viridis", vmin=-0.3, vmax=1.0)
        a.set(title=ti, xlabel="port j", ylabel="port i"); fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"Model mismatch: learned R_hat tracks true R  "
                 f"(objective  oracle={o:.1f}  learned={lrn:.1f}  assumes-Jakes={mism:.1f})", fontsize=11)
    _save(fig, "figE_learning_mismatch.png")


# ============================================================ helpers
def _save(fig, name):
    fig.tight_layout()
    out = os.path.join(FIGDIR, name)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"   saved {name}")


def copy_diagnostics():
    mp = {"step10_drl_comparison.png": "figB_drl_sample_efficiency.png",   # from verify_step10_drl.py
          "step0_channel_check.png": "diag_step0_channel.png",
          "step3_belief_check.png": "diag_step3_belief_calibration.png",
          "step4_efe_terms_check.png": "diag_step4_efe_terms.png",
          "step5_greedy_check.png": "diag_step5_greedy_optimality.png"}
    for s, d in mp.items():
        p = os.path.join(HERE, s)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(FIGDIR, d)); print(f"   copied diagnostic {d}")


def write_index():
    lines = [
        "# Paper figures", "",
        f"**All figures generated at ONE operating point:** {OP}.", "",
        "Headline: AIF gets ~84% of the genie's rate while measuring only 20% of ports, and BEATS "
        "the genie on the switching-aware objective (it barely moves the antenna).", "",
        "## Results figures",
        "- **figA_observation_budget.png** - performance vs observation budget M/N (headline).",
        "- **figB_drl_sample_efficiency.png** - vs a trained DRL baseline: AIF (zero training, 20% CSI) "
        "beats the fully-trained full-CSI DRL on the objective (from verify_step10_drl.py).",
        "- **figR_results_baselines.png** - AIF vs genie/naive/random (rate + objective + switching).",
        "- **figC_protocol_doppler.png** - observe-then-precode vs predict-then-act vs Doppler.",
        "- **figD_exploration_weight.png** - beta_w sweep (sweet spot ~0.1-0.25).",
        "- **figLC_closed_loop_learning.png** - closed-loop learning curve + bars.",
        "- **figE_learning_mismatch.png** - learning R adapts to a non-Jakes channel.",
        "- **figF_pareto_frontier.png** - the AIF rate/switching Pareto frontier DOMINATES the genie; the "
        "whole frontier (beta_w 0.1-0.7) beats the genie objective, from 84% rate/~0 switching to 89% rate "
        "(from make_frontier_figure.py).", "",
        "## Diagnostics (mechanism-verification plots)",
        "- diag_step0_channel / diag_step3_belief_calibration / diag_step4_efe_terms / "
        "diag_step5_greedy_optimality.", "",
        "Regenerate: `python sim/make_paper_figures.py`.",
    ]
    with open(os.path.join(FIGDIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("   wrote README.md")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    for p in glob.glob(os.path.join(FIGDIR, "*.png")):    # drop stale figures
        os.remove(p)
    print(f"Operating point: {OP}\nMC={MC}, T={T}\n")
    for name, fn in [("A", fig_A), ("R", fig_R), ("C", fig_C), ("D", fig_D),
                     ("LC", fig_LC), ("E", fig_E)]:
        print(f"Fig {name} ...")
        fn()
    copy_diagnostics()
    write_index()
    print("\nDONE -> figures/")


if __name__ == "__main__":
    main()
