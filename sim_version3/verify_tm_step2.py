r"""
TM-2 -- does the AR(p) space-time belief lift closed-loop RATE (predict/partial) toward observe?

Reduced N (OP_V1, 5x5=25 ports, M=5) so the exact (pN)-dim filter is cheap. Channel = space-time
Jakes (spatial R x temporal Jakes, fd). Compare AR(1) belief (p=1, current model) vs AR(4) belief,
across protocols:
  observe : sense all M fresh  (p-independent baseline -- fresh sensing dominates)
  predict : precode from predicted belief (no fresh) -- AR(p) should lift this a lot
  partial : sense m_sense of M, infer rest -- AR(p) should lift this too

Gate: AR(4) predict >> AR(1) predict; AR(4) partial > AR(1) partial; observe ~ p-independent.

Run:  python verify_tm_step2.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V1
from channel import spatial_correlation
from agent import run_genie
from temporal import generate_spacetime_jakes
from st_belief import STKalmanBelief, run_st

OP = OP_V1
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FD = 0.10
M_SENSE = 2
HALF = slice(T // 2, None)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print(f"OP_V1 (reduced N): {OP.label()}\nMC={MC}, T={T}, fd={FD}, partial m_sense={M_SENSE}\n")
    pos = OP.positions()
    R = spatial_correlation(pos)
    protocols = ["observe", "predict", "partial"]
    ps = [1, 4]
    acc = {(p, pr): [] for p in ps for pr in protocols}
    genie = []
    t0 = time.perf_counter()
    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD, T, OP.K, seed=100 + s)
        genie.append(run_genie(H, OP.M, sigma2=OP.sigma2, P=OP.P,
                     positions=pos, d_min=OP.d_min)["rate"][HALF].mean())
        for p in ps:
            for pr in protocols:
                bel = STKalmanBelief(R, OP.beta, FD, p, OP.sigma_e2)
                r = run_st(bel, H, OP, np.random.default_rng(200 + s), protocol=pr, m_sense=M_SENSE)
                acc[(p, pr)].append(r["rate"][HALF].mean())
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    g = float(np.mean(genie))
    R_ = {k: float(np.mean(v)) for k, v in acc.items()}
    print(f"\ngenie rate = {g:.3f}\n")
    print(f"{'protocol':>10} | {'AR(1)':>7} | {'AR(4)':>7} | {'AR gain':>7}")
    print("-" * 40)
    for pr in protocols:
        r1, r4 = R_[(1, pr)], R_[(4, pr)]
        print(f"{pr:>10} | {r1:7.3f} | {r4:7.3f} | {r4-r1:+7.3f}")

    print("\nGates:")
    check("AR(4) predict >> AR(1) predict", R_[(4, "predict")] > R_[(1, "predict")] + 0.3,
          f"{R_[(1,'predict')]:.3f} -> {R_[(4,'predict')]:.3f}")
    check("AR(4) partial > AR(1) partial", R_[(4, "partial")] > R_[(1, "partial")] + 0.1,
          f"{R_[(1,'partial')]:.3f} -> {R_[(4,'partial')]:.3f}")
    check("AR(4) >= AR(1) in observe too (temporal aids SELECTION, not just precoding)",
          R_[(4, "observe")] >= R_[(1, "observe")] - 1e-3,
          f"{R_[(1,'observe')]:.3f} -> {R_[(4,'observe')]:.3f} (+{R_[(4,'observe')]-R_[(1,'observe')]:.3f})")
    check("AR(4) predict approaches observe (>=80% of AR(4) observe)",
          R_[(4, "predict")] >= 0.80 * R_[(4, "observe")],
          f"{R_[(4,'predict')]:.3f} vs observe {R_[(4,'observe')]:.3f}")

    print("\n" + "=" * 44)
    print(f"TM-2: {'ALL PASS' if ok else 'FAILURES ABOVE'}  ({time.perf_counter()-t0:.0f}s)")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
