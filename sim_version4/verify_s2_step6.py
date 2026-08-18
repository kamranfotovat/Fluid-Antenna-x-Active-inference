"""
S2 Step 6 verification -- Medium-S2 at the FULL operating point (N=441, M=10).

Two questions at OP_V4 scale (2x2 lambda, N=441, M=10, K=3), sensing budget n_rf_sense of M:

  (1) Does Light-S2 hold at scale? At equal budget, designed analog sensing should carry more
      information than reading a subset of individual ports, approaching the full-read ceiling.
  (2) Medium-S2: does making SELECTION sensing-aware help? The greedy epistemic term uses the
      compressed-sensing info bound (efe.epistemic_value_compressed) so it prefers active sets
      whose uncertainty fits within n_rf_sense dimensions.

Transmit is DIGITAL here to isolate the sensing contribution (transmit hybrid is orthogonal and
near-lossless -- shown in v3). Modes share the channel trajectory and the transmit precoder.

  perport            : read all M ports individually (S1 select)           [full-read ceiling]
  designed           : n_rf_sense designed analog reads (S1 select)        [Light-S2 at scale]
  subset             : n_rf_sense best individual ports (S1 select)        [S1 @ same budget]

The Medium-S2 (S2-6) comparison isolates SELECTION, holding sensing = designed and using the SAME
aggregate compressed-info objective for both, differing only in the assumed budget (same scale, no
confound):
  budget-unaware     : selection assumes a full M-read budget (sel_nrs = M)
  budget-aware       : selection knows only n_rf_sense reads happen (sel_nrs = n_rf_sense)

Gates (means over MC seeds, second-half slots):
  A. info(designed) > info(subset)                        -- Light-S2 holds at N=441
  B. info(designed) captures >= 50% of the perport ceiling
  C. rate(aware) >= rate(unaware) - tol                   -- sensing-aware selection not worse
     (delta reported; small selection headroom is an expected honest outcome)
  D. rate(designed) >= rate(subset) - tol

Run:  python sim_version4/verify_s2_step6.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V4                        # noqa: E402
from channel import ChannelSimulator            # noqa: E402
from agent import AIFAgent, run_aif_s2          # noqa: E402

OP = OP_V4
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 16
N_RF_SENSE = 4
HALF = slice(T // 2, None)


def _agent():
    return AIFAgent(OP.R(), OP.beta, OP.rho, OP.sigma_e2, OP.M, alpha=1.0, beta_w=OP.beta_w,
                    eta_sw=OP.eta_sw, sigma2=OP.sigma2, P=OP.P,
                    positions=OP.positions(), d_min=OP.d_min, n_rf=None)   # digital transmit


def main():
    print(f"OP_V4: {OP.label()}")
    print(f"n_rf_sense={N_RF_SENSE} of M={OP.M}, transmit=digital, MC={MC}, T={T}\n")
    runs = [("perport", "perport", False),
            ("designed", "designed", False),
            ("subset", "subset", False),
            ("budget-unaware", "designed", "full"),
            ("budget-aware", "designed", True)]
    acc = {name: {"rate": [], "info": []} for name, _, _ in runs}
    t0 = time.perf_counter()
    for s in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                               rho=OP.rho, beta=OP.beta, seed=200 + s)
        H = sim.generate(T)
        for name, mode, aware in runs:
            ag = _agent()
            res = run_aif_s2(ag, H, OP.sigma_e2, np.random.default_rng(700 + s), N_RF_SENSE,
                             sense_mode=mode, sense_aware_select=aware, track_belief=False)
            acc[name]["rate"].append(res["rate"][HALF].mean())
            acc[name]["info"].append(res["info"][HALF].mean())
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    R = {n: float(np.mean(acc[n]["rate"])) for n in acc}
    I = {n: float(np.mean(acc[n]["info"])) for n in acc}
    print(f"\n{'mode':16s} | {'sense info (bits)':>17s} | {'rate (bits/slot)':>16s}")
    print("-" * 58)
    for name, _, _ in runs:
        print(f"{name:16s} | {I[name]:17.4f} | {R[name]:16.4f}")

    all_pass = True
    def check(name, cond):
        nonlocal all_pass
        all_pass &= cond
        print(f"   [{ 'PASS' if cond else 'FAIL' }] {name}")

    print("\nGates:")
    check("A  info(designed) > info(subset)", I["designed"] > I["subset"] + 1e-9)
    frac = I["designed"] / I["perport"] if I["perport"] > 0 else 1.0
    check(f"B  info(designed) >= 50% of perport ceiling (frac={frac:.0%})", frac >= 0.50)
    d_rate = R["budget-aware"] - R["budget-unaware"]
    check(f"C  budget-aware selection not worse (delta rate={d_rate:+.3f} bits/slot)", d_rate >= -0.10)
    check("D  rate(designed) >= rate(subset)", R["designed"] >= R["subset"] - 1e-3)

    print("\n" + "=" * 46)
    print(f"S2 STEP 6 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"(total {time.perf_counter()-t0:.0f}s)")
    print("=" * 46)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
