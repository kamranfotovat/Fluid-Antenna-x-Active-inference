r"""
TM-3 premise -- does a WRONG temporal model (wrong Doppler) cost rate? (so learning it pays)

Unlike spatial R (whose accuracy barely mattered), the temporal model is load-bearing. Channel =
space-time Jakes at true fd; belief = AR(4) fit to an ASSUMED fd. Compare oracle-fd vs wrong-fd
(too-slow / too-fast), in predict-then-precode and observe-then-precode. A real, persistent gap =
headroom that ONLINE learning of the Doppler/AR coefficients can recover (TM-3 proper).

Run:  python verify_tm_step3_premise.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V1
from channel import spatial_correlation
from temporal import generate_spacetime_jakes
from st_belief import STKalmanBelief, run_st

OP = OP_V1
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FD_TRUE = 0.10
P = 4
HALF = slice(T // 2, None)


def main():
    print(f"OP_V1: {OP.label()}\nMC={MC}, T={T}, TRUE fd={FD_TRUE}, AR(p={P}) belief\n")
    pos = OP.positions()
    R = spatial_correlation(pos)
    assumed = {"oracle (0.10)": 0.10, "too-slow (0.05)": 0.05, "too-fast (0.20)": 0.20}
    protocols = ["predict", "observe"]
    acc = {(pr, name): [] for pr in protocols for name in assumed}
    t0 = time.perf_counter()
    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD_TRUE, T, OP.K, seed=100 + s)
        for pr in protocols:
            for name, fd in assumed.items():
                bel = STKalmanBelief(R, OP.beta, fd, P, OP.sigma_e2)
                r = run_st(bel, H, OP, np.random.default_rng(200 + s), protocol=pr)
                acc[(pr, name)].append(r["rate"][HALF].mean())
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    R_ = {k: float(np.mean(v)) for k, v in acc.items()}
    print(f"\n{'assumed fd':>16} | " + " | ".join(f"{pr:>8}" for pr in protocols))
    print("-" * 40)
    for name in assumed:
        print(f"{name:>16} | " + " | ".join(f"{R_[(pr,name)]:8.3f}" for pr in protocols))
    print("\nlearnable gap (oracle - wrong):")
    for pr in protocols:
        o = R_[(pr, "oracle (0.10)")]
        gs = o - R_[(pr, "too-slow (0.05)")]
        gf = o - R_[(pr, "too-fast (0.20)")]
        print(f"  {pr:>8}: vs too-slow {gs:+.3f} | vs too-fast {gf:+.3f}")
    print(f"\n(total {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    raise SystemExit(main())
