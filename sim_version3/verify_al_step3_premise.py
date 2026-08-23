r"""
AL-3 premise -- does a DRIFTING R create PERSISTENT headroom (unlike stationary, where it decayed)?

Compare, on a non-stationary channel (g_t(d)=exp(-d/d0(t)), d0 drifting):
  oracle-track : belief adopts the true R_t each slot (knows the drift)   -> upper bound
  fixed        : belief assumes a STATIC R (exponential at the mid d0)    -> best static guess
If tracking the drift beats the best static guess at STEADY STATE (second half), the wrong-R penalty
persists -> there is continuous headroom for active learning to recover. We also print first-half vs
second-half to show the gap does NOT decay (contrast with the stationary case).

Run:  python verify_al_step3_premise.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V3
from agent import AIFAgent, run_aif, objective
from nonstationary import generate_nonstationary, run_aif_track, exp_R, d0_drift

OP = OP_V3
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 2
T = int(sys.argv[2]) if len(sys.argv) > 2 else 100
D0_MID = 0.35


def _agent(R):
    return AIFAgent(R=R, beta=OP.beta, rho=OP.rho, sigma_e2=OP.sigma_e2, M=OP.M,
                    alpha=1.0, beta_w=OP.beta_w, eta_sw=OP.eta_sw, e_sw=1.0,
                    sigma2=OP.sigma2, P=OP.P, positions=OP.positions(), d_min=OP.d_min)


def _obj_halves(res):
    T = len(res["rate"])
    h1 = float(np.mean(res["rate"][:T // 2] - OP.eta_sw * res["switch"][:T // 2]))
    h2 = float(np.mean(res["rate"][T // 2:] - OP.eta_sw * res["switch"][T // 2:]))
    return h1, h2


def main():
    print(f"OP_V3: {OP.label()}\nMC={MC}, T={T}, drift d0(t) in [0.2,0.5], fixed=exp(d0={D0_MID})\n")
    pos = OP.positions()
    R_fixed = exp_R(pos, D0_MID)
    ot2, fx2, ot1, fx1 = [], [], [], []
    t0 = time.perf_counter()
    for s in range(MC):
        H, R_seq, _ = generate_nonstationary(OP.Nx, OP.Ny, OP.Wx, OP.Wy, OP.K, OP.rho, OP.beta,
                                             T, seed=100 + s)
        rt = run_aif_track(_agent(R_fixed), H, R_seq, OP.sigma_e2, np.random.default_rng(200 + s))
        fx = run_aif(_agent(R_fixed), H, OP.sigma_e2, np.random.default_rng(200 + s), sense_first=True)
        a1, a2 = _obj_halves(rt); b1, b2 = _obj_halves(fx)
        ot1.append(a1); ot2.append(a2); fx1.append(b1); fx2.append(b2)
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    O1, O2 = float(np.mean(ot1)), float(np.mean(ot2))
    F1, F2 = float(np.mean(fx1)), float(np.mean(fx2))
    print(f"\n{'':>14}{'1st half':>10}{'2nd half':>10}")
    print(f"{'oracle-track':>14}{O1:10.3f}{O2:10.3f}")
    print(f"{'fixed-static':>14}{F1:10.3f}{F2:10.3f}")
    print(f"{'gap':>14}{O1-F1:+10.3f}{O2-F2:+10.3f}")
    persist = (O2 - F2) > 0.1
    print(f"\n[{'PASS' if persist else 'FAIL'}] drift creates PERSISTENT steady-state headroom "
          f"(2nd-half gap {O2-F2:+.3f} > 0.1)")
    print(f"(total {time.perf_counter()-t0:.0f}s)")
    return 0 if persist else 1


if __name__ == "__main__":
    raise SystemExit(main())
