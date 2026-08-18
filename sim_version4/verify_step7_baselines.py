"""
Step 7b -- beat the COMPETITORS (not just the genie).

The genie (full CSI) is only a ceiling. The real test is fair partial-CSI competitors that
see the same 20% observation budget and also observe-then-precode. What separates them from
our agent is the generative Kalman belief + the EFE-unified decision (rate + info - switching).

Methods (all partial-CSI except the genie ceiling), at the locked operating point
(15 dB, beta_w=0.25, eta_sw=1.0, observe-then-precode):
  genie          full CSI on all N ports (unrealistic ceiling)
  AIF (ours)     Kalman belief + greedy EFE
  naive          no inference: held point estimate, top-M + round-robin sensing
  random         random selection

Checks
------
  B1  AIF beats the naive competitor on the switching-aware OBJECTIVE by a clear margin.
  B2  AIF is not worse than naive on RATE (fair: both get fresh pilots) and reaches a high
      fraction of the genie's rate.
  B3  AIF's advantage is STABILITY: it switches far less than naive/genie (EFE unifies the
      switching cost into selection).
  B4  AIF >> random on the objective.

Run:  python sim/verify_step7_baselines.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402
from agent import (AIFAgent, run_aif, run_genie, run_naive,          # noqa: E402
                   run_random_partial, objective)

SIGMA_E2, SIGMA2 = 1e-2, 0.03                  # 15 dB
BETA = np.array([1.0, 0.7, 1.3])
K, M, RHO = 3, 5, 0.9
ETA_SW, BETA_W = 1.0, 0.25
T, MC = 80, 18


def main():
    print(f"N=25, K={K}, M={M} (20% obs), 15 dB, rho={RHO}, beta_w={BETA_W}, eta_sw={ETA_SW}, T={T}, MC={MC}")
    acc = {k: dict(obj=0.0, rate=0.0, sw=0.0) for k in ["genie", "aif", "naive", "rand"]}
    for m in range(MC):
        sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
        H = sim.generate(T)
        runs = {
            "genie": run_genie(H, M, sigma2=SIGMA2),
            "aif": run_aif(AIFAgent(sim.R, BETA, RHO, SIGMA_E2, M, 1.0, BETA_W, ETA_SW, sigma2=SIGMA2),
                           H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True),
            "naive": run_naive(H, M, SIGMA_E2, np.random.default_rng(40000 + m), sigma2=SIGMA2),
            "rand": run_random_partial(H, M, SIGMA_E2, np.random.default_rng(50000 + m), sigma2=SIGMA2),
        }
        for k, res in runs.items():
            acc[k]["obj"] += objective(res, ETA_SW)
            acc[k]["rate"] += res["rate"][T // 2:].mean()
            acc[k]["sw"] += res["switch"].mean()
    for k in acc:
        for f in acc[k]:
            acc[k][f] /= MC

    print("\n method  |  objective |  rate  | switch/slot")
    for k in ["genie", "aif", "naive", "rand"]:
        print(f"  {k:6s} |   {acc[k]['obj']:6.2f}  | {acc[k]['rate']:5.2f}  |   {acc[k]['sw']:.2f}")

    all_pass = True

    print("\n[B1] AIF beats the naive competitor on the switching-aware objective")
    margin = (acc["aif"]["obj"] - acc["naive"]["obj"]) / acc["naive"]["obj"]
    ok = margin > 0.15
    all_pass &= ok
    print(f"   AIF obj={acc['aif']['obj']:.2f} vs naive={acc['naive']['obj']:.2f}  (+{margin*100:.0f}%, >15%)"
          f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n[B2] AIF not worse than naive on rate, and high fraction of genie rate")
    frac = acc["aif"]["rate"] / acc["genie"]["rate"]
    ok = acc["aif"]["rate"] >= acc["naive"]["rate"] - 1e-6 and frac > 0.75
    all_pass &= ok
    print(f"   AIF rate={acc['aif']['rate']:.2f} >= naive={acc['naive']['rate']:.2f} | {frac*100:.0f}% of genie"
          f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n[B3] AIF is more stable: fewer switches than naive and genie")
    ok = acc["aif"]["sw"] < acc["naive"]["sw"] and acc["aif"]["sw"] < acc["genie"]["sw"]
    all_pass &= ok
    print(f"   switch/slot: AIF={acc['aif']['sw']:.2f} < naive={acc['naive']['sw']:.2f}, genie={acc['genie']['sw']:.2f}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n[B4] AIF >> random on the objective")
    ok = acc["aif"]["obj"] > 2.0 * acc["rand"]["obj"]
    all_pass &= ok
    print(f"   AIF obj={acc['aif']['obj']:.2f} vs random={acc['rand']['obj']:.2f}  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + ("=" * 46))
    print(f"STEP 7b OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)

    _try_plot(acc)
    return 0 if all_pass else 1


def _try_plot(acc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    order = ["genie", "aif", "naive", "rand"]
    labels = ["genie\n(full CSI)", "AIF\n(ours)", "naive\n(no inference)", "random"]
    rate = [acc[k]["rate"] for k in order]
    obj = [acc[k]["obj"] for k in order]
    sw = [acc[k]["sw"] for k in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(x - w / 2, rate, w, label="rate", color="tab:blue", alpha=0.85)
    ax[0].bar(x + w / 2, obj, w, label="objective (rate - switch)", color="tab:orange", alpha=0.85)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("bits/s/Hz"); ax[0].set_title("Rate vs switching-aware objective")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3, axis="y")
    ax[0].axhline(acc["aif"]["obj"], color="tab:orange", ls=":", alpha=0.5)

    ax[1].bar(x, sw, color="tab:red", alpha=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("ports moved / slot"); ax[1].set_title("Antenna switching (lower = cheaper)")
    ax[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = "step7_baselines.png"
    fig.savefig(out, dpi=120)
    print(f"(saved plot -> sim/{out})")


if __name__ == "__main__":
    raise SystemExit(main())
