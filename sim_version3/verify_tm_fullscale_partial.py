r"""
FULL-SCALE partial-sensing sweep -- the pilot-savings curve at OP_V2, with and without the
temporal model.

The one weak spot in the full-scale results: partial sensing reached only 67.3% of genie and gained
just +0.98 from AR(1)->AR(4), measured at a single point (m_sense=4 of M=10). This sweeps m_sense so
the pilot-savings curve is a curve rather than a point, and separates two questions:

  (a) how gracefully does rate fall as pilots are withdrawn?  (the deployable claim)
  (b) does the temporal model help MORE when fewer ports are sensed fresh?

Hypothesis for (b): with fewer fresh pilots, more served ports are carried by the belief, so the
model that propagates the belief forward in time should matter more -- the AR(4)-AR(1) gap should
WIDEN as m_sense falls. (This is the partial-sensing analogue of why predict-then-precode gains
most.) If it does not widen, say so -- the m=4 point already hinted the gain is small here.

m_sense = M is exactly observe-then-precode (all active ports sensed fresh) and anchors the curve.

Run:  python verify_tm_fullscale_partial.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V2
from channel import spatial_correlation
from temporal import generate_spacetime_jakes
from st_belief_lr import STKalmanBeliefLR
from st_belief import run_st
from agent import run_genie

OP = OP_V2
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FD = 0.10
M_LIST = [2, 4, 6, 8, 10]
HALF = slice(T // 2, None)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print(f"FULL-SCALE PARTIAL SENSING -- OP_V2: {OP.label()}")
    print(f"MC={MC}, T={T}, fd={FD}, m_sense sweep {M_LIST} of M={OP.M}\n")
    pos = OP.positions()
    R = spatial_correlation(pos)
    t0 = time.perf_counter()

    acc = {(m, p): [] for m in M_LIST for p in (1, 4)}
    genie = []
    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD, T, OP.K, seed=100 + s)
        genie.append(run_genie(H, OP.M, sigma2=OP.sigma2, P=OP.P, positions=pos,
                               d_min=OP.d_min)["rate"][HALF].mean())
        for m in M_LIST:
            for p in (1, 4):
                bel = STKalmanBeliefLR(R, OP.beta, FD, p, OP.sigma_e2)
                out = run_st(bel, H, OP, np.random.default_rng(200 + s),
                             protocol="partial", m_sense=m)
                acc[(m, p)].append(out["rate"][HALF].mean())
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    A = {k: float(np.mean(v)) for k, v in acc.items()}
    G = float(np.mean(genie))
    full1, full4 = A[(OP.M, 1)], A[(OP.M, 4)]

    print(f"\n{'m_sense':>8} | {'pilots':>7} | {'AR(1)':>8} | {'AR(4)':>8} | {'gain':>7} | "
          f"{'% of full AR(4)':>16} | {'% genie':>8}")
    print("-" * 82)
    for m in M_LIST:
        print(f"{m:8d} | {m/OP.M*100:6.0f}% | {A[(m,1)]:8.3f} | {A[(m,4)]:8.3f} | "
              f"{A[(m,4)]-A[(m,1)]:+7.3f} | {A[(m,4)]/full4*100:15.1f}% | {A[(m,4)]/G*100:7.1f}%")
    print(f"{'genie':>8} | {'':>7} | {'':>8} | {G:8.3f}")

    gains = {m: A[(m, 4)] - A[(m, 1)] for m in M_LIST}
    print("\nGates:")
    check("rate is monotone in pilots (AR(4))",
          all(A[(M_LIST[i], 4)] <= A[(M_LIST[i + 1], 4)] + 0.35 for i in range(len(M_LIST) - 1)),
          " <= ".join(f"{A[(m,4)]:.2f}" for m in M_LIST))
    check("partial sensing degrades GRACEFULLY (half the pilots keep >=70% of the rate)",
          A[(OP.M // 2, 4)] / full4 >= 0.70,
          f"m={OP.M//2}: {A[(OP.M//2,4)]:.3f} = {A[(OP.M//2,4)]/full4*100:.1f}% of full-pilot")
    check("the temporal model helps at every pilot budget", all(g > 0 for g in gains.values()),
          "  ".join(f"m={m}:{gains[m]:+.2f}" for m in M_LIST))
    widens = gains[M_LIST[0]] > gains[M_LIST[-1]]
    check("HYPOTHESIS: the AR(4)-AR(1) gap WIDENS as pilots are withdrawn", widens,
          f"m={M_LIST[0]}: {gains[M_LIST[0]]:+.3f} vs m={M_LIST[-1]}: {gains[M_LIST[-1]]:+.3f}")

    print("\n" + "=" * 72)
    print(f"FULL-SCALE PARTIAL: {'ALL PASS' if ok else 'SEE FAILURES ABOVE'} "
          f"({time.perf_counter()-t0:.0f}s)")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
