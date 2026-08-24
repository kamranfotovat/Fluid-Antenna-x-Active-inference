r"""
Partial-sensing sweep on the AR(1) S1 model.

Part 1: rate/objective vs m_sense (0=predict-then-precode ... M=observe-then-precode), correct Jakes R.
        Shows how much just a few fresh pilots buy over pure prediction.
Part 2: at a few m_sense, correct Jakes R vs IDENTITY R (no spatial inference). The gap = value of
        R-inference of the un-sensed served ports. Expect ~0 at m_sense=M (R idle, all fresh) and
        growing as m_sense shrinks -> where learning R would finally pay.

Channel is standard Jakes AR(1) (no mismatch). Run:  python verify_partial.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V3
from channel import ChannelSimulator, spatial_correlation
from agent import AIFAgent, run_aif, run_genie, objective
from partial_sense import run_aif_partial

OP = OP_V3
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
HALF = slice(T // 2, None)


def _agent(R):
    return AIFAgent(R=R, beta=OP.beta, rho=OP.rho, sigma_e2=OP.sigma_e2, M=OP.M,
                    alpha=1.0, beta_w=OP.beta_w, eta_sw=OP.eta_sw, e_sw=1.0,
                    sigma2=OP.sigma2, P=OP.P, positions=OP.positions(), d_min=OP.d_min)


def main():
    print(f"OP_V3: {OP.label()}\nMC={MC}, T={T}, Jakes AR(1) channel, second-half slots\n")
    pos = OP.positions()
    R_jakes = spatial_correlation(pos)
    R_eye = np.eye(OP.N)

    ms_list = [0, 2, 4, 6, 8, 10]
    rate = {m: [] for m in ms_list}
    genie = []
    # part 2 accumulators
    ms2 = [0, 4, 8]
    rate_eye = {m: [] for m in ms2}
    t0 = time.perf_counter()
    for s in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                               rho=OP.rho, beta=OP.beta, seed=100 + s)
        H = sim.generate(T)
        genie.append(run_genie(H, OP.M, sigma2=OP.sigma2, P=OP.P,
                     positions=pos, d_min=OP.d_min)["rate"][HALF].mean())
        for m in ms_list:
            r = run_aif_partial(_agent(R_jakes), H, OP.sigma_e2, np.random.default_rng(200 + s), m)
            rate[m].append(r["rate"][HALF].mean())
        for m in ms2:
            r = run_aif_partial(_agent(R_eye), H, OP.sigma_e2, np.random.default_rng(200 + s), m)
            rate_eye[m].append(r["rate"][HALF].mean())
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    g = float(np.mean(genie))
    print(f"\ngenie rate = {g:.3f}\n")
    print("Part 1 -- rate vs pilot budget m_sense (Jakes R):")
    print(f"{'m_sense':>8} | {'rate':>7} | {'%genie':>7} | {'%obs(m=M)':>9}")
    print("-" * 40)
    r_full = float(np.mean(rate[OP.M]))
    for m in ms_list:
        rm = float(np.mean(rate[m]))
        print(f"{m:>8} | {rm:7.3f} | {100*rm/g:7.1f} | {100*rm/r_full:9.1f}")

    print("\nPart 2 -- value of R-inference (Jakes minus Identity R):")
    print(f"{'m_sense':>8} | {'Jakes':>7} | {'Identity':>8} | {'R-gain':>7}")
    print("-" * 38)
    for m in ms2:
        rj = float(np.mean(rate[m])); re = float(np.mean(rate_eye[m]))
        print(f"{m:>8} | {rj:7.3f} | {re:8.3f} | {rj-re:+7.3f}")
    print("\n(expect R-gain ~0 at m_sense=M, growing as m_sense shrinks -> R-inference load-bearing)")
    print(f"(total {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    raise SystemExit(main())
