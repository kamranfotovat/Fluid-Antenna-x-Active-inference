r"""
FULL-SCALE temporal-model results -- the whole TM chain at S1's real operating point.

Everything in TM-0..TM-4 was proven at OP_V1 (N=25, M=5) because the exact ST filter costs
O((pN)^3). st_belief_lr.STKalmanBeliefLR removes that limit EXACTLY (R is rank-26 of 441), so we
can now rerun the two results that matter at OP_V2: N=441 ports, M=10 active, K=3, 2-lambda
aperture, lambda/10 pitch -- the regime the paper actually claims.

PART 1 (TM-2 at full scale): does the AR(1) -> AR(4) temporal upgrade still pay across protocols?
PART 2 (TM-3/TM-4 at full scale): does online Doppler learning still recover the wrong-model gap,
        and does the PRINCIPLED estimator (matched normalization + data-driven order selection,
        zero tuned constants) still beat TM-3's hand-tuned hedge?

Run:  python verify_tm_fullscale.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V2
from channel import spatial_correlation
from temporal import generate_spacetime_jakes, jakes_autocorr, TemporalACF
from st_belief_lr import STKalmanBeliefLR
from st_belief import run_st, run_st_learn, run_st_learn_probe
from agent import run_genie

OP = OP_V2
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FD_TRUE, FD_WRONG = 0.10, 0.05
M_SENSE = 4
HALF = slice(T // 2, None)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print(f"FULL SCALE -- OP_V2: {OP.label()}")
    print(f"MC={MC}, T={T}, true fd={FD_TRUE}\n")
    pos = OP.positions()
    R = spatial_correlation(pos)
    t0 = time.perf_counter()

    def bel(fd, p):
        return STKalmanBeliefLR(R, OP.beta, fd, p, OP.sigma_e2)

    print(f"reduced rank r={bel(FD_TRUE, 4).r} of N={OP.N}  "
          f"(exact: R is rank-deficient; see verify_tm_lr.py)\n")

    # =========================================================== PART 1: AR(1) vs AR(4)
    print("=" * 72)
    print("PART 1 -- TM-2 at full scale: AR(1) (current model) vs AR(4), by protocol")
    print("=" * 72)
    protos = ["observe", "predict", "partial"]
    p1 = {(pr, p): [] for pr in protos for p in (1, 4)}
    genie = []
    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD_TRUE, T, OP.K, seed=100 + s)
        genie.append(run_genie(H, OP.M, sigma2=OP.sigma2, P=OP.P, positions=pos,
                               d_min=OP.d_min)["rate"][HALF].mean())
        for pr in protos:
            for p in (1, 4):
                out = run_st(bel(FD_TRUE, p), H, OP, np.random.default_rng(200 + s),
                             protocol=pr, m_sense=M_SENSE)
                p1[(pr, p)].append(out["rate"][HALF].mean())
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    G = float(np.mean(genie))
    A = {k: float(np.mean(v)) for k, v in p1.items()}
    print(f"\n{'protocol':>22} | {'AR(1)':>8} | {'AR(4)':>8} | {'gain':>7} | {'% genie':>8}")
    print("-" * 66)
    for pr in protos:
        lbl = pr if pr != "partial" else f"partial (m={M_SENSE})"
        print(f"{lbl:>22} | {A[(pr,1)]:8.3f} | {A[(pr,4)]:8.3f} | "
              f"{A[(pr,4)]-A[(pr,1)]:+7.3f} | {A[(pr,4)]/G*100:7.1f}%")
    print(f"{'genie (perfect CSI)':>22} | {'':>8} | {G:8.3f} |")

    print("\nGates:")
    check("AR(4) beats AR(1) in predict-then-precode", A[("predict", 4)] > A[("predict", 1)] + 0.5,
          f"{A[('predict',1)]:.3f} -> {A[('predict',4)]:.3f}")
    check("AR(4) >= AR(1) in observe-then-precode (better selection)",
          A[("observe", 4)] >= A[("observe", 1)] - 0.2,
          f"{A[('observe',1)]:.3f} -> {A[('observe',4)]:.3f}")
    check("AR(4) beats AR(1) in partial sensing", A[("partial", 4)] > A[("partial", 1)],
          f"{A[('partial',1)]:.3f} -> {A[('partial',4)]:.3f}")

    # =========================================================== PART 2: learning the Doppler
    print("\n" + "=" * 72)
    print("PART 2 -- TM-3/TM-4 at full scale: online Doppler learning, predict-then-precode")
    print("=" * 72)
    P = 4
    true_r = jakes_autocorr(np.arange(P + 1), FD_TRUE)

    def acf(matched=True):
        return TemporalACF(OP.N, P, OP.sigma_e2, matched=matched)

    arms = {
        "oracle (knows fd)": lambda H, g: run_st(bel(FD_TRUE, P), H, OP, g, protocol="predict"),
        "wrong-fixed (fd=0.05)": lambda H, g: run_st(bel(FD_WRONG, P), H, OP, g, protocol="predict"),
        "TM-3 naive": lambda H, g: run_st_learn(bel(FD_WRONG, P), H, OP, g, acf(False),
                                                protocol="predict", relearn_every=5),
        "TM-3 tuned hedge": lambda H, g: run_st_learn(bel(FD_WRONG, P), H, OP, g, acf(False),
                                                      protocol="predict", relearn_every=5,
                                                      ev_inflate=3.0, r_shrink=0.95),
        "PRINCIPLED (N)+(O)": lambda H, g: run_st_learn_probe(bel(FD_WRONG, P), H, OP, g, acf(True),
                                                              protocol="predict", relearn_every=5,
                                                              probe=False, robust=True),
    }
    p2 = {k: [] for k in arms}
    rh = {k: [] for k in arms}
    orq = {k: [] for k in arms}
    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD_TRUE, T, OP.K, seed=100 + s)
        for name, fn in arms.items():
            out = fn(H, np.random.default_rng(200 + s))
            p2[name].append(out["rate"][HALF].mean())
            if "rhat" in out:
                rh[name].append(out["rhat"])
            if out.get("orders"):
                orq[name].extend(out["orders"])
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    B = {k: float(np.mean(v)) for k, v in p2.items()}
    O, W = B["oracle (knows fd)"], B["wrong-fixed (fd=0.05)"]
    gap = O - W
    print(f"\nwrong-Doppler gap (oracle - wrong-fixed) = {gap:+.3f}\n")
    print(f"{'arm':>24} | {'rate':>8} | {'recovery':>8} | {'r-hat(1)':>9} | {'bias':>7} | {'q':>4}")
    print("-" * 77)
    for name in arms:
        rec = (B[name] - W) / gap * 100 if abs(gap) > 1e-9 else 0.0
        r1 = f"{np.mean([r[1] for r in rh[name]]):.3f}" if rh[name] else "--"
        bi = (f"{np.mean([r[1] for r in rh[name]]) - true_r[1]:+.3f}") if rh[name] else "--"
        q = f"{np.mean(orq[name]):.1f}" if orq[name] else "--"
        print(f"{name:>24} | {B[name]:8.3f} | {rec:7.0f}% | {r1:>9} | {bi:>7} | {q:>4}")
    print(f"\ntrue r(1) = {true_r[1]:.3f}")

    Rp, Rt = B["PRINCIPLED (N)+(O)"], B["TM-3 tuned hedge"]
    rec_p = (Rp - W) / gap * 100
    print("\nGates:")
    check("wrong Doppler is costly at full scale (there IS a gap)", gap > 1.0, f"{gap:+.3f}")
    check("PRINCIPLED beats wrong-fixed", Rp > W + 0.3, f"{W:.3f} -> {Rp:.3f}")
    check("PRINCIPLED >= TM-3's tuned hedge", Rp >= Rt - 0.3, f"{Rp:.3f} vs tuned {Rt:.3f}")
    check("PRINCIPLED recovers >= 50% of the gap", rec_p >= 50.0, f"{rec_p:.0f}%")

    print("\n" + "=" * 72)
    print(f"FULL SCALE: {'ALL PASS' if ok else 'FAILURES ABOVE'}  ({time.perf_counter()-t0:.0f}s)")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
