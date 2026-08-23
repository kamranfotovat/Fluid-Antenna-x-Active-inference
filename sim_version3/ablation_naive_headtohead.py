r"""
NAIVE HEAD-TO-HEAD -- information-directed vs BLIND exploration.

Closes the one caveat from ablation_epistemic.py: there, beta_w=0 (no exploration) got
stranded at ~63% of genie, so "the epistemic term helps" could be dismissed as "ANY
exploration helps." This script pits AIF's information-seeking selection against a
PASSIVE explorer that also refreshes ports every slot but blindly (round-robin), plus a
random-selection lower bound. If AIF wins at EQUAL-OR-LOWER switching, the novelty is
*information-directed* exploration, not merely "exploration".

Methods (shared channel trajectory, observe-then-precode, digital transmit):
  AIF beta_w=0.25   information-directed selection (the sweet spot)
  AIF beta_w=0.60   information-directed, max-rate corner
  naive refresh=1/2/3  top-M-by-last-power + r round-robin ports (BLIND passive exploration)
  random-partial    random feasible selection (lower bound)
  genie             full-CSI upper bound

Metrics (means over MC seeds, second-half slots):
  rate    realized sum-rate on TRUE channel   switch  ports changed/slot
  obj     rate - eta_sw*switch (honest score)  %genie  rate / genie

Run:  python ablation_naive_headtohead.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator
from agent import (AIFAgent, run_aif, run_naive, run_random_partial, run_genie)
from config import OP_V3

OP = OP_V3
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
T = int(sys.argv[2]) if len(sys.argv) > 2 else 32
HALF = slice(T // 2, None)


def _agent(R, beta_w, sigma_e2, rho):
    return AIFAgent(R=R, beta=OP.beta, rho=rho, sigma_e2=sigma_e2, M=OP.M,
                    alpha=1.0, beta_w=beta_w, eta_sw=OP.eta_sw, e_sw=1.0,
                    sigma2=OP.sigma2, P=OP.P,
                    positions=OP.positions(), d_min=OP.d_min, n_rf=None)


def sweep(methods, sigma_e2, rho, label):
    print(f"\n{'='*70}\n{label}   (sigma_e2={sigma_e2:g}, rho={rho:g})\n{'='*70}")
    print(f"{'method':>16} | {'rate':>7} | {'switch':>7} | {'obj':>8} | {'%genie':>7}")
    print("-" * 58)
    acc = {m: {"rate": [], "sw": [], "obj": [], "pg": []} for m in methods}
    for s in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                               rho=rho, beta=OP.beta, seed=300 + s)
        H = sim.generate(T)
        R = sim.R
        pos = OP.positions()
        gmean = run_genie(H, OP.M, sigma2=OP.sigma2, P=OP.P,
                          positions=pos, d_min=OP.d_min)["rate"][HALF].mean()
        for m in methods:
            rng = np.random.default_rng(700 + s)
            if m.startswith("AIF"):
                bw = float(m.split("=")[1])
                ag = _agent(R, bw, sigma_e2, rho)
                res = run_aif(ag, H, sigma_e2, rng, track_belief=False, sense_first=True)
            elif m.startswith("naive"):
                rf = int(m.split("=")[1])
                res = run_naive(H, OP.M, sigma_e2, rng, sigma2=OP.sigma2, P=OP.P,
                                refresh=rf, positions=pos, d_min=OP.d_min)
            else:  # random-partial
                res = run_random_partial(H, OP.M, sigma_e2, rng, sigma2=OP.sigma2, P=OP.P,
                                         positions=pos, d_min=OP.d_min)
            r = res["rate"][HALF].mean(); sw = res["switch"][HALF].mean()
            acc[m]["rate"].append(r); acc[m]["sw"].append(sw)
            acc[m]["obj"].append(r - OP.eta_sw * sw); acc[m]["pg"].append(100 * r / gmean)
    for m in methods:
        r = np.mean(acc[m]["rate"]); sw = np.mean(acc[m]["sw"])
        obj = np.mean(acc[m]["obj"]); pg = np.mean(acc[m]["pg"])
        print(f"{m:>16} | {r:7.3f} | {sw:7.3f} | {obj:8.3f} | {pg:7.1f}")
    return acc


def main():
    t0 = time.perf_counter()
    print(f"OP_V3: {OP.label()}   |  MC={MC}, T={T}, second-half slots, digital transmit")

    full = ["AIF=0.25", "AIF=0.60", "naive=1", "naive=2", "naive=3", "random"]
    lite = ["AIF=0.25", "naive=1", "naive=2", "naive=3", "random"]

    sweep(full, OP.sigma_e2, OP.rho, "A. NOMINAL")
    print(f"\n{'*'*70}\nB. STRESS -- NOISY PILOTS\n{'*'*70}")
    for se2 in (1e-2, 1e-1):
        sweep(lite, se2, OP.rho, f"noisy pilots sigma_e2={se2:g}")
    print(f"\n{'*'*70}\nC. STRESS -- FAST AGING\n{'*'*70}")
    for r in (0.8, 0.7):
        sweep(lite, OP.sigma_e2, r, f"fast aging rho={r:g}")

    print(f"\n(total {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
