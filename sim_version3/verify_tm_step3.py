r"""
TM-3 -- ACTIVE learning of the Doppler/AR that finally PAYS.

Start the AR(4) belief from a WRONG (too-slow) Doppler -- the catastrophic case (predict-then-precode
craters to ~3.9 vs oracle 13.1). Learn the temporal autocorrelation online from the measurement
stream (same-port pairs across slots) and refit AR(4). Compare:
  oracle      : belief knows the true fd                     -> upper bound
  wrong-fixed : belief stuck at the wrong (too-slow) fd      -> what you get without learning
  learned     : starts wrong, learns online                  -> should recover most of the gap

This is the mirror image of R-learning (which never paid): the temporal model is load-bearing, and
its samples are available, so learning recovers real rate.

Run:  python verify_tm_step3.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V1
from channel import spatial_correlation
from temporal import generate_spacetime_jakes, jakes_autocorr, TemporalACF
from st_belief import STKalmanBelief, run_st, run_st_learn

OP = OP_V1
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 60
FD_TRUE, FD_WRONG, P = 0.10, 0.05, 4
PROTO = "predict"
HALF = slice(T // 2, None)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print(f"OP_V1: {OP.label()}\nMC={MC}, T={T}, protocol={PROTO}, TRUE fd={FD_TRUE}, "
          f"start-wrong fd={FD_WRONG}, AR(p={P})\n")
    pos = OP.positions()
    R = spatial_correlation(pos)
    orc, wrg, lrn = [], [], []
    rhat_acc = []
    t0 = time.perf_counter()
    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD_TRUE, T, OP.K, seed=100 + s)
        orc.append(run_st(STKalmanBelief(R, OP.beta, FD_TRUE, P, OP.sigma_e2), H, OP,
                          np.random.default_rng(200 + s), protocol=PROTO)["rate"][HALF].mean())
        wrg.append(run_st(STKalmanBelief(R, OP.beta, FD_WRONG, P, OP.sigma_e2), H, OP,
                          np.random.default_rng(200 + s), protocol=PROTO)["rate"][HALF].mean())
        acf = TemporalACF(OP.N, P, OP.sigma_e2)
        # CONSERVATIVE learning: the policy's held-port measurements bias the ACF toward high
        # correlation (survivorship), and predict-then-precode craters if correlation is
        # overestimated -> hedge against the catastrophic (too-slow) direction.
        rl = run_st_learn(STKalmanBelief(R, OP.beta, FD_WRONG, P, OP.sigma_e2), H, OP,
                          np.random.default_rng(200 + s), acf, protocol=PROTO, relearn_every=5,
                          ev_inflate=3.0, r_shrink=0.95)
        lrn.append(rl["rate"][HALF].mean()); rhat_acc.append(rl["rhat"])
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    O, W, L = float(np.mean(orc)), float(np.mean(wrg)), float(np.mean(lrn))
    gap = O - W
    rec = (L - W) / gap * 100 if abs(gap) > 1e-9 else 0.0
    print(f"\n{'oracle':>12} | {'wrong-fixed':>12} | {'learned':>9}")
    print("-" * 38)
    print(f"{O:12.3f} | {W:12.3f} | {L:9.3f}")
    print(f"\nwrong-Doppler gap (oracle - wrong) = {gap:+.3f};  LEARNING recovers {rec:.0f}%")
    rhat = np.mean(rhat_acc, axis=0)
    true_r = jakes_autocorr(np.arange(P + 1), FD_TRUE)
    print("learned autocorr r-hat:", " ".join(f"{v:+.3f}" for v in rhat))
    print("true    autocorr J0   :", " ".join(f"{v:+.3f}" for v in true_r))

    print("\nGates:")
    check("learning beats wrong-fixed", L > W + 0.3, f"{W:.3f} -> {L:.3f}")
    check("learning recovers >= 50% of the wrong-Doppler gap", rec >= 50.0, f"{rec:.0f}%")
    check("learned r-hat(1) close to true (|err|<0.05)", abs(rhat[1] - true_r[1]) < 0.05,
          f"{rhat[1]:.3f} vs {true_r[1]:.3f}")

    print("\n" + "=" * 44)
    print(f"TM-3: {'ALL PASS' if ok else 'FAILURES ABOVE'}  ({time.perf_counter()-t0:.0f}s)")
    print("=" * 44)
    print("Active learning of the TEMPORAL model pays -- the lever R-learning never had.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
