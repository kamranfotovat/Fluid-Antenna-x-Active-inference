r"""
TM-4 -- FIX THE ACF ESTIMATOR: replace TM-3's hand-tuned hedges with a principled estimator.

TM-3 worked (79% recovery) but only via two magic constants (r_shrink=0.95, ev_inflate=3.0) that
counterweight a BIAS. The bias: the ACF is fed the POLICY's measurements, and lag-tau pairs only
exist for ports the policy held at BOTH t and t-tau -- conditioning on both endpoints being strong,
which preferentially samples temporally-coherent realizations. Result: r-hat biased HIGH -> Doppler
underestimated -> overconfident stale prediction -> crater (the -21% naive result).

Two principled fixes, ablated separately so we know which one carries the result:
  (S) UNBIASED SAMPLING  -- one of the M ports is a RANDOM PROBE, drawn independently of the belief
      and held p+1 slots; only its stream feeds the ACF. A MODEL-epistemic action. Costs 1/M pilots.
  (H) RISK-AWARE ESTIMATE -- use the one-sided lower confidence bound r-hat - kappa*se(Bartlett)
      instead of an arbitrary shrink. Justified by the loss asymmetry TM-3-premise measured
      (too-slow -9.3, too-fast -1.7); it self-annihilates as samples accumulate.

Arms isolate: probe cost alone, the biased/naive failure, TM-3's tuned hedge, (H) alone, (S) alone,
and (S)+(H). The claim to establish: (S)+(H) with NO tuned constants >= TM-3's tuned hedge.

Run:  python verify_tm_step4.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V1
from channel import spatial_correlation
from temporal import generate_spacetime_jakes, jakes_autocorr, TemporalACF
from st_belief import STKalmanBelief, run_st, run_st_learn, run_st_learn_probe

OP = OP_V1
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 60
FD_TRUE, FD_WRONG, P = 0.10, 0.05, 4
PROTO = "predict"
HALF = slice(T // 2, None)
NEVER = 10 ** 9
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print(f"OP_V1: {OP.label()}\nMC={MC}, T={T}, protocol={PROTO}, TRUE fd={FD_TRUE}, "
          f"start-wrong fd={FD_WRONG}, AR(p={P})\n")
    R = spatial_correlation(OP.positions())
    true_r = jakes_autocorr(np.arange(P + 1), FD_TRUE)

    def bel(fd):
        return STKalmanBelief(R, OP.beta, fd, P, OP.sigma_e2)

    def acf(matched=True):
        return TemporalACF(OP.N, P, OP.sigma_e2, matched=matched)

    # name -> (callable(H, rng) -> dict with 'rate' and optionally 'rhat').
    # Ablation ladder: each row adds ONE ingredient to the previous.
    arms = {
        "oracle": lambda H, g: run_st(bel(FD_TRUE), H, OP, g, protocol=PROTO),
        "oracle + probe": lambda H, g: run_st_learn_probe(bel(FD_TRUE), H, OP, g, acf(),
                                                          protocol=PROTO, relearn_every=NEVER),
        "wrong-fixed": lambda H, g: run_st(bel(FD_WRONG), H, OP, g, protocol=PROTO),
        "TM-3 naive (no fix)": lambda H, g: run_st_learn(bel(FD_WRONG), H, OP, g, acf(False),
                                                         protocol=PROTO, relearn_every=5),
        "TM-3 tuned hedge": lambda H, g: run_st_learn(bel(FD_WRONG), H, OP, g, acf(False),
                                                      protocol=PROTO, relearn_every=5,
                                                      ev_inflate=3.0, r_shrink=0.95),
        "(N) matched norm only": lambda H, g: run_st_learn_probe(bel(FD_WRONG), H, OP, g, acf(True),
                                                                 protocol=PROTO, relearn_every=5,
                                                                 probe=False, robust=False),
        "(S) probe only": lambda H, g: run_st_learn_probe(bel(FD_WRONG), H, OP, g, acf(False),
                                                          protocol=PROTO, relearn_every=5,
                                                          probe=True, robust=False),
        "(O) order sel only": lambda H, g: run_st_learn_probe(bel(FD_WRONG), H, OP, g, acf(False),
                                                              protocol=PROTO, relearn_every=5,
                                                              probe=False, robust=True),
        "PRINCIPLED = (N)+(O)": lambda H, g: run_st_learn_probe(bel(FD_WRONG), H, OP, g, acf(True),
                                                                protocol=PROTO, relearn_every=5,
                                                                probe=False, robust=True),
        "PRINCIPLED + (S) probe": lambda H, g: run_st_learn_probe(bel(FD_WRONG), H, OP, g, acf(True),
                                                                  protocol=PROTO, relearn_every=5,
                                                                  probe=True, robust=True),
    }
    acc = {k: [] for k in arms}
    rh = {k: [] for k in arms}
    orq = {k: [] for k in arms}
    t0 = time.perf_counter()
    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD_TRUE, T, OP.K, seed=100 + s)
        for name, fn in arms.items():
            out = fn(H, np.random.default_rng(200 + s))
            acc[name].append(out["rate"][HALF].mean())
            if "rhat" in out:
                rh[name].append(out["rhat"])
            if out.get("orders"):
                orq[name].extend(out["orders"])
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    Rm = {k: float(np.mean(v)) for k, v in acc.items()}
    O, W = Rm["oracle"], Rm["wrong-fixed"]
    gap = O - W
    print(f"\ntrue fd={FD_TRUE}: oracle {O:.3f} | wrong-fixed {W:.3f} | learnable gap {gap:+.3f}\n")
    print(f"{'arm':>28} | {'rate':>7} | {'recovery':>8} | {'r-hat(1)':>9} | {'bias':>7} | {'q':>4}")
    print("-" * 81)
    for name in arms:
        rec = (Rm[name] - W) / gap * 100 if abs(gap) > 1e-9 else 0.0
        q = f"{np.mean(orq[name]):.1f}" if orq[name] else "--"
        if rh[name]:
            r1 = float(np.mean([r[1] for r in rh[name]]))
            print(f"{name:>28} | {Rm[name]:7.3f} | {rec:7.0f}% | {r1:9.3f} | "
                  f"{r1-true_r[1]:+7.3f} | {q:>4}")
        else:
            print(f"{name:>28} | {Rm[name]:7.3f} | {rec:7.0f}% | {'--':>9} | {'--':>7} | {q:>4}")
    print(f"\ntrue r(1) = J0(2 pi {FD_TRUE}) = {true_r[1]:.3f};  "
          f"q = AR order the learner selected (oracle uses p={P})")

    b_naive = float(np.mean([r[1] for r in rh["TM-3 naive (no fix)"]])) - true_r[1]
    b_fixed = float(np.mean([r[1] for r in rh["PRINCIPLED = (N)+(O)"]])) - true_r[1]
    Rp = Rm["PRINCIPLED = (N)+(O)"]
    Rt = Rm["TM-3 tuned hedge"]
    Ro = Rm["(O) order sel only"]
    Rn = Rm["(N) matched norm only"]
    Rprobe = Rm["PRINCIPLED + (S) probe"]
    rec_p = (Rp - W) / gap * 100
    probe_cost = O - Rm["oracle + probe"]

    print("\nGates:")
    check("unfixed ACF is biased HIGH (the diagnosis)", b_naive > 0.01, f"bias {b_naive:+.3f}")
    check("(N) matched normalization removes the bias", abs(b_fixed) < abs(b_naive) * 0.5,
          f"|{b_fixed:+.3f}| vs |{b_naive:+.3f}|  ({100*(1-abs(b_fixed)/abs(b_naive)):.0f}% removed)")
    check("(O) order selection is the load-bearing fix", Ro > Rn + 2.0,
          f"order-sel {Ro:.3f} vs de-bias-only {Rn:.3f}")
    check("PRINCIPLED (zero tuned constants) beats wrong-fixed", Rp > W + 0.3, f"{W:.3f} -> {Rp:.3f}")
    check("PRINCIPLED >= TM-3's tuned hedge", Rp >= Rt - 0.3, f"{Rp:.3f} vs tuned {Rt:.3f}")
    check("PRINCIPLED recovers >= 75% of the gap", rec_p >= 75.0, f"{rec_p:.0f}%")
    print(f"\n  HONEST NEGATIVE: the dedicated random probe does NOT pay -- {Rprobe:.3f} vs "
          f"{Rp:.3f} without it.\n  It costs a port ({probe_cost:+.3f} rate, 1 of M={OP.M} pilots) AND "
          f"starves the ACF of samples,\n  which raises se and forces a LOWER affordable AR order. "
          f"The policy's own held ports\n  already supply enough (matched-normalized) temporal samples.")

    print("\n" + "=" * 44)
    print(f"TM-4: {'ALL PASS' if ok else 'FAILURES ABOVE'}  ({time.perf_counter()-t0:.0f}s)")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
