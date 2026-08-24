r"""
Partial-sensing MISMATCH premise -- does a plausible-but-WRONG R cost rate when pilots are limited?

The fair "would-learning-pay" test (identity-R in verify_partial.py is a strawman). True channel =
exponential (non-Jakes); compare in PARTIAL sensing:
  oracle-R : belief uses the true exponential R   -> upper bound
  fixed    : belief assumes Jakes (the mismatch)  -> what you get without learning
across m_sense. Under observe-then-precode this gap was ~0 (AL-0/AL-3: R idle). Here it should be a
REAL gap that shrinks toward 0 at m_sense=M (all fresh -> R idle again) and grows as pilots drop ->
the persistent headroom that active learning of R could recover.

Run:  python verify_partial_mismatch.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V3
from channel import ChannelSimulator, spatial_correlation
from agent import AIFAgent, objective
from learning import exponential_correlation, set_correlation
from partial_sense import run_aif_partial

OP = OP_V3
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 2
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
HALF = slice(T // 2, None)
D0 = 0.3


def _agent(R):
    return AIFAgent(R=R, beta=OP.beta, rho=OP.rho, sigma_e2=OP.sigma_e2, M=OP.M,
                    alpha=1.0, beta_w=OP.beta_w, eta_sw=OP.eta_sw, e_sw=1.0,
                    sigma2=OP.sigma2, P=OP.P, positions=OP.positions(), d_min=OP.d_min)


def main():
    print(f"OP_V3: {OP.label()}\nMC={MC}, T={T}, TRUE=exp(d0={D0}), oracle=exp vs fixed=Jakes\n")
    pos = OP.positions()
    R_jakes = spatial_correlation(pos)
    R_true = exponential_correlation(pos, d0=D0)
    ms_list = [4, 6, 10]
    orc = {m: [] for m in ms_list}
    fix = {m: [] for m in ms_list}
    t0 = time.perf_counter()
    for s in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                               rho=OP.rho, beta=OP.beta, seed=100 + s)
        set_correlation(sim, R_true)
        H = sim.generate(T)
        for m in ms_list:
            ro = run_aif_partial(_agent(R_true), H, OP.sigma_e2, np.random.default_rng(200 + s), m)
            rf = run_aif_partial(_agent(R_jakes), H, OP.sigma_e2, np.random.default_rng(200 + s), m)
            orc[m].append(ro["rate"][HALF].mean())
            fix[m].append(rf["rate"][HALF].mean())
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"\n{'m_sense':>8} | {'oracle-R':>8} | {'fixed(Jakes)':>12} | {'learnable gap':>13}")
    print("-" * 50)
    for m in ms_list:
        o, f = float(np.mean(orc[m])), float(np.mean(fix[m]))
        tag = "  (observe: R idle)" if m >= OP.M else ""
        print(f"{m:>8} | {o:8.3f} | {f:12.3f} | {o-f:+13.3f}{tag}")
    print("\n=> gap ~0 at m_sense=M (all fresh) and grows as pilots drop -> where learning R pays")
    print(f"(total {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    raise SystemExit(main())
