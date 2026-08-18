"""
Step 10 -- comparison against a DRL baseline (the literature competitor, Paper-3 style).

A Transformer port-selection policy trained by policy gradient on the SAME Eq.7 objective.
Two honest headline axes where model-based active inference wins:

  L1  Sample efficiency (Fig B).  AIF needs ZERO training; the DRL needs hundreds of episodes
      just to reach its plateau. Plotted as objective vs training iterations.
  L2  No full-CSI advantage.  Even with FULL CSI and full training, the DRL's objective does not
      exceed AIF's PARTIAL-CSI (20% budget) objective -- EFE selection + observe-then-precode is
      at least as good, with no training and 1/5 the CSI.
  L3  Competence gate.  Trained at eta_sw=0 the DRL reaches a high fraction of the genie RATE
      (so it is a competent baseline, not a straw man); at eta_sw=1 it correctly learns to lock.

Run:  python sim/verify_step10_drl.py    (trains on GPU; a few minutes)
"""

from __future__ import annotations

import sys
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator                                    # noqa: E402
from agent import AIFAgent, run_aif, run_genie, run_naive, objective    # noqa: E402
from drl_baseline import train_policy, eval_fullcsi, eval_partial       # noqa: E402

SIGMA_E2, SIGMA2 = 1e-3, 0.03
BETA = np.array([1.0, 0.7, 1.3])
K, M, RHO = 3, 5, 0.9
ETA_SW, BETA_W = 1.0, 0.25
T, MC = 80, 20
SNAP_ITERS = [0, 25, 50, 100, 200, 400]


def _aif(R):
    return AIFAgent(R, BETA, RHO, SIGMA_E2, M, 1.0, BETA_W, ETA_SW, sigma2=SIGMA2)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}, op point: 15 dB, sigma_e^2=1e-3, rho=0.9, beta_w=0.25, eta_sw=1, T={T}, MC={MC}")
    train_sims = [ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=1000 + b)
                  for b in range(48)]

    # ---- train the objective policy (eta_sw=1) with snapshots for the sample-efficiency curve ----
    print("\nTraining DRL policy (eta_sw=1) with snapshots ...")
    policy, hist, snaps = train_policy(train_sims, K, M, sigma2=SIGMA2, eta_sw=ETA_SW,
                                       iters=401, L=16, lr=1e-3, log_every=200, snapshots=set(SNAP_ITERS))

    # ---- competence policy (eta_sw=0) ----
    print("Training competence policy (eta_sw=0) ...")
    pol0, _ = train_policy(train_sims, K, M, sigma2=SIGMA2, eta_sw=0.0, iters=400, L=12, lr=1e-3, log_every=400)

    # ---- evaluate everything on a fresh test set ----
    def eval_all(pol, eta):
        g = dfull = dpart = a = nv = ds_full = 0.0
        for m in range(MC):
            sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=5000 + m)
            H = sim.generate(T)
            h = slice(T // 2, None)
            g += objective(run_genie(H, M, sigma2=SIGMA2), eta)
            rf = eval_fullcsi(pol, H, M, sigma2=SIGMA2, eta_sw=eta)
            dfull += objective(rf, eta); ds_full += rf["switch"].mean()
            dpart += objective(eval_partial(pol, H, M, SIGMA_E2, np.random.default_rng(30000 + m),
                                            sigma2=SIGMA2, eta_sw=eta), eta)
            a += objective(run_aif(_aif(sim.R), H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True), eta)
            nv += objective(run_naive(H, M, SIGMA_E2, np.random.default_rng(40000 + m), sigma2=SIGMA2), eta)
        n = MC
        return dict(genie=g / n, drl_full=dfull / n, drl_part=dpart / n, aif=a / n, naive=nv / n,
                    drl_full_sw=ds_full / n)

    print("\nEvaluating ...")
    R = eval_all(policy, ETA_SW)
    print("\n objective (rate - switching), eta_sw=1:")
    print(f"   genie(full CSI)={R['genie']:.2f} | DRL full-CSI={R['drl_full']:.2f} (sw {R['drl_full_sw']:.2f}) | "
          f"DRL partial={R['drl_part']:.2f} | AIF partial(ours)={R['aif']:.2f} | naive={R['naive']:.2f}")

    # ---- sample-efficiency curve: eval each snapshot (full CSI objective) ----
    print("\nSample-efficiency (DRL objective vs training iters, full CSI):")
    se = {}
    for it in SNAP_ITERS:
        o = 0.0
        for m in range(MC):
            sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=5000 + m)
            H = sim.generate(T)
            o += objective(eval_fullcsi(snaps[it], H, M, sigma2=SIGMA2, eta_sw=ETA_SW), ETA_SW)
        se[it] = o / MC
        print(f"   iter {it:4d}: DRL objective={se[it]:.2f}")

    # ---- competence gate (eta_sw=0, rate) ----
    gr = dr = 0.0
    for m in range(MC):
        sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=5000 + m)
        H = sim.generate(T)
        gr += run_genie(H, M, sigma2=SIGMA2)["rate"][T // 2:].mean()
        dr += eval_fullcsi(pol0, H, M, sigma2=SIGMA2, eta_sw=0.0)["rate"][T // 2:].mean()
    frac = dr / gr

    all_pass = True
    print("\n[L1] Sample efficiency: DRL must TRAIN up; AIF is flat from iter 0")
    ok = se[SNAP_ITERS[0]] < R["aif"] - 1.0 and se[SNAP_ITERS[-1]] > se[SNAP_ITERS[0]] + 1.0
    all_pass &= ok
    print(f"   DRL obj iter0={se[SNAP_ITERS[0]]:.2f} -> iter400={se[SNAP_ITERS[-1]]:.2f}; AIF(no training)={R['aif']:.2f}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n[L2] No full-CSI advantage: AIF partial >= DRL full-CSI on the objective")
    ok = R["aif"] >= R["drl_full"] - 1e-6
    all_pass &= ok
    print(f"   AIF partial={R['aif']:.2f}  vs  DRL full-CSI={R['drl_full']:.2f}  -> {'PASS' if ok else 'FAIL'}")

    print("\n[L3] Competence gate: DRL(eta_sw=0) reaches a high fraction of genie rate")
    ok = frac > 0.80
    all_pass &= ok
    print(f"   DRL(eta=0) full-CSI rate = {dr/MC:.2f} vs genie {gr/MC:.2f} -> frac = {frac*100:.0f}% (>80%)"
          f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + ("=" * 46))
    print(f"STEP 10 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)

    _try_plot(se, R)
    return 0 if all_pass else 1


def _try_plot(se, R):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    its = sorted(se)
    ax[0].plot(its, [se[i] for i in its], "s-", color="tab:red", label="DRL (needs training)")
    ax[0].axhline(R["aif"], color="tab:green", ls="-", lw=2, label="AIF (ours, zero training)")
    ax[0].set(title="Sample efficiency (Fig B)", xlabel="DRL training iterations",
              ylabel="objective (bits/s/Hz)")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

    order = ["genie", "drl_full", "aif", "drl_part", "naive"]
    labels = ["genie\n(full CSI)", "DRL\n(full CSI)", "AIF\n(partial, ours)",
              "DRL\n(partial)", "naive\n(partial)"]
    vals = [R[k] for k in order]
    colors = ["black", "tab:red", "tab:green", "tab:orange", "tab:gray"]
    ax[1].bar(np.arange(len(order)), vals, color=colors, alpha=0.85)
    ax[1].axhline(R["aif"], color="tab:green", ls=":", alpha=0.6)
    ax[1].set_xticks(np.arange(len(order))); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set(title="Objective: AIF (20% CSI, no training) vs DRL", ylabel="objective (bits/s/Hz)")
    ax[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = "step10_drl_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"(saved plot -> sim/{out})")


if __name__ == "__main__":
    raise SystemExit(main())
