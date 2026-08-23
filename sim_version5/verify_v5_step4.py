r"""
V5-4 gate G4 -- closed-loop myopic column agent + baselines.

Runs AIF / genie / naive / random on shared trajectories and checks:
  A. every AIF config is feasible & reachable (Delta_max + min-spacing respected all slots).
  B. AIF beats naive and random on the objective (rate - eta_mv*move).
  C. sensible ceiling: 0 < AIF rate <= genie, and AIF captures a healthy fraction of genie.

Run:  python verify_v5_step4.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_B
from channel import ChannelSimulator
from run_col import run_col_aif, run_col_genie, run_col_naive, run_col_random

OP = OP_B
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 4
T = int(sys.argv[2]) if len(sys.argv) > 2 else 24
HALF = slice(T // 2, None)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print(f"OP_B: {OP.label()}\nMC={MC}, T={T}, second-half slots\n")
    acc = {m: {"rate": [], "move": [], "obj": [], "pg": []} for m in
           ("aif", "naive", "random", "genie")}
    all_feasible = True
    t0 = time.perf_counter()
    for s in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                               rho=OP.rho, beta=OP.beta, seed=400 + s)
        H = sim.generate(T)
        g = run_col_genie(OP, H, np.random.default_rng(900 + s))
        gmean = g["rate"][HALF].mean()
        acc["genie"]["rate"].append(gmean)
        runs = {
            "aif": run_col_aif(OP, H, np.random.default_rng(500 + s)),
            "naive": run_col_naive(OP, H, np.random.default_rng(600 + s)),
            "random": run_col_random(OP, H, np.random.default_rng(700 + s)),
        }
        all_feasible &= runs["aif"]["feasible"]
        for m, r in runs.items():
            rate = r["rate"][HALF].mean(); mv = r["move"][HALF].mean()
            acc[m]["rate"].append(rate); acc[m]["move"].append(mv)
            acc[m]["obj"].append(rate - OP.eta_mv * mv)
            acc[m]["pg"].append(100 * rate / gmean)
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    R = {m: np.mean(acc[m]["rate"]) for m in acc}
    print(f"\n{'method':>8} | {'rate':>7} | {'move':>6} | {'obj':>7} | {'%genie':>7}")
    print("-" * 46)
    for m in ("genie", "aif", "naive", "random"):
        mv = np.mean(acc[m]["move"]) if acc[m]["move"] else 0.0
        obj = np.mean(acc[m]["obj"]) if acc[m]["obj"] else R[m]
        pg = np.mean(acc[m]["pg"]) if acc[m]["pg"] else 100.0
        print(f"{m:>8} | {R[m]:7.3f} | {mv:6.2f} | {obj:7.3f} | {pg:7.1f}")

    obj = {m: np.mean(acc[m]["obj"]) for m in ("aif", "naive", "random")}
    print("\nGates:")
    check("A  every AIF config feasible & reachable", all_feasible)
    check("B1 AIF objective > naive", obj["aif"] > obj["naive"], f"{obj['aif']:.3f} vs {obj['naive']:.3f}")
    check("B2 AIF objective > random", obj["aif"] > obj["random"], f"{obj['aif']:.3f} vs {obj['random']:.3f}")
    check("C1 0 < AIF rate <= genie", 0 < R["aif"] <= R["genie"] + 1e-6)
    frac = 100 * R["aif"] / R["genie"]
    check("C2 AIF captures >= 60% of genie", frac >= 60.0, f"{frac:.1f}%")

    print("\n" + "=" * 44)
    print(f"V5-4 GATE G4: {'ALL PASS' if ok else 'FAILURES ABOVE'}  ({time.perf_counter()-t0:.0f}s)")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
