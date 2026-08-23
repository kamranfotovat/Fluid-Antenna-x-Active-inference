r"""
AL-0 -- mismatch premise for ACTIVE LEARNING of the spatial correlation (S1 extension).

Active learning of R only matters if a WRONG R actually costs performance. This confirms the
premise: generate channels from a NON-Jakes correlation (exponential), then run the S1 agent with
  * oracle-R    : agent's belief uses the TRUE (exponential) R          -> upper bound
  * fixed-wrong : agent's belief assumes Jakes (the mismatch)           -> what you get w/o learning
on the SAME trajectories. The oracle-minus-fixed objective gap is the headroom that learning R can
recover. Observe-then-precode, S1 geometry (OP_V3).

Gate: gap > 0 (mismatch measurably hurts) so there is something for active learning to recover.

Run:  python verify_al_step0.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V3
from channel import ChannelSimulator, spatial_correlation
from agent import AIFAgent, run_aif, run_genie, objective
from learning import exponential_correlation, set_correlation

OP = OP_V3
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 4
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
D0 = 0.3                      # exponential correlation length (wavelengths)


def _agent(R):
    return AIFAgent(R=R, beta=OP.beta, rho=OP.rho, sigma_e2=OP.sigma_e2, M=OP.M,
                    alpha=1.0, beta_w=OP.beta_w, eta_sw=OP.eta_sw, e_sw=1.0,
                    sigma2=OP.sigma2, P=OP.P, positions=OP.positions(), d_min=OP.d_min)


def main():
    print(f"OP_V3: {OP.label()}\nMC={MC}, T={T}, TRUE R = exponential(d0={D0}), assumed(wrong) = Jakes\n")
    pos = OP.positions()
    R_jakes = spatial_correlation(pos)
    R_true = exponential_correlation(pos, d0=D0)
    # how different are they?
    off = ~np.eye(OP.N, dtype=bool)
    print(f"||R_true - R_jakes||_F / ||R_jakes||_F = "
          f"{np.linalg.norm(R_true-R_jakes)/np.linalg.norm(R_jakes):.3f}   "
          f"(mean |off-diag diff| = {np.abs(R_true-R_jakes)[off].mean():.3f})\n")

    acc = {"oracle": [], "fixed": [], "genie": []}
    t0 = time.perf_counter()
    for s in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                               rho=OP.rho, beta=OP.beta, seed=100 + s)
        set_correlation(sim, R_true)                     # channel now drawn from exponential R
        H = sim.generate(T)
        g = run_genie(H, OP.M, sigma2=OP.sigma2, P=OP.P, positions=pos, d_min=OP.d_min)
        acc["genie"].append(objective(g, OP.eta_sw))
        for tag, R in [("oracle", R_true), ("fixed", R_jakes)]:
            ag = _agent(R)
            res = run_aif(ag, H, OP.sigma_e2, np.random.default_rng(200 + s), sense_first=True)
            acc[tag].append(objective(res, OP.eta_sw))
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    O = {k: float(np.mean(v)) for k, v in acc.items()}
    gap = O["oracle"] - O["fixed"]
    print(f"\n{'agent':>12} | {'objective':>10}")
    print("-" * 26)
    for k in ("genie", "oracle", "fixed"):
        print(f"{k:>12} | {O[k]:10.3f}")
    print(f"\noracle - fixed (learnable headroom) = {gap:+.3f} objective")
    print(f"[{'PASS' if gap > 0.1 else 'FAIL'}] mismatch measurably hurts (gap > 0.1)")
    print(f"(total {time.perf_counter()-t0:.0f}s)")
    return 0 if gap > 0.1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
