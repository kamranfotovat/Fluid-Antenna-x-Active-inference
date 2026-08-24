r"""
TM-5 -- what information actually buys, and why AR(2) is the practical ceiling.

FIRST ATTEMPT (recorded because the negative is the useful part). The hypothesis was "information
buys model order": more data -> smaller ACF standard error -> higher affordable AR order -> higher
rate. The rate half held (9.73 -> 12.02, 96% of oracle) but the ORDER half did NOT: the selected
order went 2.67 -> 2.00 as data grew, i.e. DOWN. The early 2.67 was noise in the selector, not a
genuinely affordable higher order.

The reason is arithmetic. TM-4's table says order 4 only wins once se(1) <~ 0.005. Bartlett gives
se ~ sqrt((1+2r^2)/n) with n ~ T*M*K, so se = 0.005 needs n ~ 1e5, i.e. T ~ 7000 slots. No
realistic FAS deployment sees that. So:

    A LEARNER CAN NEVER AFFORD AR(4) IN THIS REGIME -- AR(2) IS THE PRACTICAL CEILING.

That sounds like bad news for the temporal upgrade, but it is not, and this script measures why:
closed-loop RATE saturates long before prediction ERROR does (sigma_e^2 and multiuser interference
dominate the last decade of prediction error), so AR(2) with well-estimated coefficients gets most
of what the AR(4) oracle gets. What information buys is not order -- it is COEFFICIENT ACCURACY at
a low, safe order, plus the selector's refusal to buy an order it cannot afford. That refusal is
the safety mechanism: forcing p=4 without it craters at every horizon.

Arms per horizon: learned (order-selected) | learned FORCED to p=4 | oracle AR(4) | oracle AR(2).

Run:  python verify_tm_step5.py [MC]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V1
from channel import spatial_correlation
from temporal import generate_spacetime_jakes, jakes_autocorr, TemporalACF
from st_belief import STKalmanBelief, run_st, run_st_learn_probe

OP = OP_V1
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FD_TRUE, FD_WRONG, P = 0.10, 0.05, 4
HORIZONS = [20, 40, 80, 160]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print(f"OP_V1: {OP.label()}\nMC={MC}, true fd={FD_TRUE}, start-wrong fd={FD_WRONG}, "
          f"p_max={P}, predict-then-precode\n")
    R = spatial_correlation(OP.positions())
    true_r = jakes_autocorr(np.arange(P + 1), FD_TRUE)
    t0 = time.perf_counter()

    rows = []
    for T in HORIZONS:
        LAST = slice(3 * T // 4, None)               # score after the model has had time to sharpen
        acc = {k: [] for k in ("learn", "forced4", "orc4", "orc2")}
        se1, qs, err1 = [], [], []
        for s in range(MC):
            H = generate_spacetime_jakes(R, OP.beta, FD_TRUE, T, OP.K, seed=100 + s)

            a = TemporalACF(OP.N, P, OP.sigma_e2, matched=True)
            out = run_st_learn_probe(STKalmanBelief(R, OP.beta, FD_WRONG, P, OP.sigma_e2), H, OP,
                                     np.random.default_rng(200 + s), a, protocol="predict",
                                     relearn_every=5, probe=False, robust=True)
            acc["learn"].append(out["rate"][LAST].mean())
            se1.append(out["se"][1]); qs.extend(out["orders"][-3:])
            err1.append(abs(out["rhat"][1] - true_r[1]))

            a2 = TemporalACF(OP.N, P, OP.sigma_e2, matched=True)
            f4 = run_st_learn_probe(STKalmanBelief(R, OP.beta, FD_WRONG, P, OP.sigma_e2), H, OP,
                                    np.random.default_rng(200 + s), a2, protocol="predict",
                                    relearn_every=5, probe=False, robust=False)
            acc["forced4"].append(f4["rate"][LAST].mean())

            for key, p in (("orc4", 4), ("orc2", 2)):
                acc[key].append(run_st(STKalmanBelief(R, OP.beta, FD_TRUE, p, OP.sigma_e2), H, OP,
                                       np.random.default_rng(200 + s),
                                       protocol="predict")["rate"][LAST].mean())
        rows.append(dict(T=T, se=float(np.mean(se1)), q=float(np.mean(qs)),
                         err=float(np.mean(err1)),
                         **{k: float(np.mean(v)) for k, v in acc.items()}))
        print(f"  T={T} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"\n{'slots T':>8} | {'ACF se(1)':>10} | {'|r err|':>8} | {'order q':>8} | "
          f"{'learned':>8} | {'forced p=4':>10} | {'AR(4) orc':>9} | {'AR(2) orc':>9} | {'% orc4':>7}")
    print("-" * 108)
    for r in rows:
        print(f"{r['T']:8d} | {r['se']:10.4f} | {r['err']:8.4f} | {r['q']:8.2f} | "
              f"{r['learn']:8.3f} | {r['forced4']:10.3f} | {r['orc4']:9.3f} | {r['orc2']:9.3f} | "
              f"{r['learn']/r['orc4']*100:6.1f}%")

    lo, hi = rows[0], rows[-1]
    print("\nGates:")
    check("more data -> smaller ACF standard error", hi["se"] < lo["se"],
          f"se(1): {lo['se']:.4f} (T={lo['T']}) -> {hi['se']:.4f} (T={hi['T']})")
    check("more data -> higher rate", hi["learn"] > lo["learn"],
          f"{lo['learn']:.3f} -> {hi['learn']:.3f}")
    check("learned converges toward the AR(4) oracle", hi["learn"] / hi["orc4"] > 0.90,
          f"{hi['learn']/hi['orc4']*100:.1f}% of oracle at T={hi['T']}")
    check("the selector refuses unaffordable orders (q stays low, ~2)", hi["q"] <= 2.5,
          f"q -> {hi['q']:.2f} (not {P})")
    check("that refusal is what saves it: forcing p=4 craters at EVERY horizon",
          all(r["forced4"] < r["learn"] - 1.0 for r in rows),
          "forced " + " / ".join(f"{r['forced4']:.2f}" for r in rows) +
          "  vs learned " + " / ".join(f"{r['learn']:.2f}" for r in rows))
    check("AR(2) oracle already captures most of AR(4) oracle (rate saturates early)",
          hi["orc2"] / hi["orc4"] > 0.85, f"AR(2) {hi['orc2']:.3f} vs AR(4) {hi['orc4']:.3f} "
          f"= {hi['orc2']/hi['orc4']*100:.1f}%")

    print("\n" + "=" * 60)
    print(f"TM-5: {'ALL PASS' if ok else 'FAILURES ABOVE'}  ({time.perf_counter()-t0:.0f}s)")
    print("=" * 60)
    print("Information buys COEFFICIENT ACCURACY at a low safe order -- not order itself.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
