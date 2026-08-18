"""
S2 Step 5 verification -- Light-S2 closed loop (the first real milestone).

At an EQUAL sensing budget of n_rf_sense measurements/slot, does DESIGNING the analog sensing
combiner beat the S1 alternatives? Modes share the SAME channel trajectory, SAME EFE selection,
SAME transmit hybrid -- only the sensing read differs:

    designed  : n_rf_sense EFE-designed analog combinations of all M active ports   [S2]
    subset    : read the n_rf_sense highest-variance individual active ports  [S1 @ same budget]
    random    : n_rf_sense random analog combinations                        [ablation]
    perport   : read all M active ports individually (MORE budget)      [S1 full-read ceiling]

The designed combiner MAXIMISES information gain (log-det mutual information -- the epistemic EFE
term), so the primary, provably-optimised metric is bits of sensing info per slot. (Per-port
squared error is a different, A-optimal criterion and is reported but not the headline.)

Gate (means over MC seeds, second-half slots):
  A. info(designed) > info(subset)   -- smart compression carries MORE info than reading a subset
  B. info(designed) > info(random)   -- the DESIGN matters, not just analog mixing
  C. info(designed) captures a high fraction of the perport (full-budget) ceiling
  D. rate(designed) >= rate(subset) and >= rate(random)  -- the info win carries to throughput

Run:  python sim_version4/verify_s2_step5.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402
from agent import AIFAgent, run_aif_s2         # noqa: E402


def _agent(sim, beta, rho, sigma_e2, M):
    return AIFAgent(sim.R, beta, rho, sigma_e2, M, alpha=1.0, beta_w=0.25, eta_sw=1.0,
                    sigma2=0.03, P=1.0, positions=None, d_min=None, n_rf=None)


def main():
    rho, sigma_e2 = 0.9, 1e-2
    beta = np.array([1.0, 0.7, 1.3]); K = len(beta)
    Nx = Ny = 5; M = 6; n_rf_sense = 3
    MC, T = 5, 30
    HALF = slice(T // 2, None)
    modes = ["perport", "designed", "subset", "random"]
    acc = {m: {"rate": [], "err": [], "info": []} for m in modes}

    for s in range(MC):
        sim = ChannelSimulator(Nx=Nx, Ny=Ny, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=100 + s)
        H = sim.generate(T)
        for m in modes:
            ag = _agent(sim, beta, rho, sigma_e2, M)
            res = run_aif_s2(ag, H, sigma_e2, np.random.default_rng(500 + s), n_rf_sense,
                             sense_mode=m, track_belief=True)
            acc[m]["rate"].append(res["rate"][HALF].mean())
            acc[m]["err"].append(res["real_err"][HALF].mean())
            acc[m]["info"].append(res["info"][HALF].mean())

    R = {m: float(np.mean(acc[m]["rate"])) for m in modes}
    E = {m: float(np.mean(acc[m]["err"])) for m in modes}
    I = {m: float(np.mean(acc[m]["info"])) for m in modes}
    print(f"N={Nx*Ny}, M={M}, n_rf_sense={n_rf_sense} of {M}, K={K}, MC={MC}, T={T}\n")
    print(f"{'mode':10s} | {'sense info (bits)':>17s} | {'rate (bits/slot)':>16s} | {'real_err':>10s}")
    print("-" * 64)
    for m in modes:
        print(f"{m:10s} | {I[m]:17.4f} | {R[m]:16.4f} | {E[m]:10.5f}")

    all_pass = True
    def check(name, cond):
        nonlocal all_pass
        all_pass &= cond
        print(f"   [{ 'PASS' if cond else 'FAIL' }] {name}")

    print("\nGates:")
    check("A  info(designed) > info(subset)", I["designed"] > I["subset"] + 1e-9)
    check("B  info(designed) > info(random)", I["designed"] > I["random"] + 1e-9)
    frac = I["designed"] / I["perport"] if I["perport"] > 0 else 1.0
    check(f"C  info(designed) captures >=60% of perport ceiling (frac={frac:.0%})", frac >= 0.60)
    check("D  rate(designed) >= rate(subset)", R["designed"] >= R["subset"] - 1e-6)
    check("D  rate(designed) >= rate(random)", R["designed"] >= R["random"] - 1e-6)

    print("\n" + "=" * 46)
    print(f"S2 STEP 5 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
