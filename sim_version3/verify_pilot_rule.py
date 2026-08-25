r"""PILOT-RULE A/B -- how should Q_t (which activated ports to pilot) be chosen?

Kian's question: is Q_t picked at random, and if not, can it be made adaptive so that
ports the agent is already confident about go unpiloted?

It is already adaptive -- run_st takes the top m activated ports by posterior variance.
But that is a HEURISTIC, and the paper currently claims something stronger: that both
S_t and Q_t are chosen by minimizing the same expected free energy. The EFE's epistemic
term is a LOG-DET, not a sum of marginal variances:

    Epis(Q) = sum_k log2 det( I + Cov_k(Q) / sigma_e^2 )

The two differ by REDUNDANCY. Marginal variance scores each port alone, so two ports
that are both uncertain AND strongly correlated with each other both look attractive,
even though the second adds almost nothing once the first is measured. The log-det sees
the cross-correlation and spends that pilot elsewhere. They agree only when candidates
are uncorrelated -- and ours sit on a sub-wavelength grid.

Whether that matters here is an empirical question, because the EFE selection already
spreads S_t toward the Jakes decorrelation nulls, which may leave little redundancy for
the log-det to exploit. So: measure it.

    ARM A  variance   -- current code
    ARM B  epistemic  -- greedy log-det, i.e. what the paper says we do

If B >= A we adopt B and the paper's claim becomes true as written. If B < A we keep A
and must weaken the claim to describe the heuristic honestly.

Run:  python verify_pilot_rule.py [MC] [T]
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V2, OP_V3
from channel import spatial_correlation
from temporal import generate_spacetime_jakes
from st_belief_lr import STKalmanBeliefLR
from st_belief import run_st

MC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FD, P_AR = 0.10, 4
M_LIST = [4, 6]            # the budgets where the choice can actually matter
HALF = slice(T // 2, None)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    OP = OP_V3
    print(f"PILOT-RULE A/B   MC={MC}, T={T}, AR({P_AR}), m in {M_LIST}")
    print(f"  {OP.label()}\n", flush=True)
    R = spatial_correlation(OP_V2.positions())

    acc = {(m, r): [] for m in M_LIST for r in ("variance", "epistemic")}
    t0 = time.perf_counter()
    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD, T, OP.K, seed=100 + s)
        for m in M_LIST:
            for rule in ("variance", "epistemic"):
                bel = STKalmanBeliefLR(R, OP.beta, FD, P_AR, OP.sigma_e2)
                out = run_st(bel, H, OP, np.random.default_rng(200 + s),
                             protocol="partial", m_sense=m, pilot_rule=rule)
                acc[(m, rule)].append(float(out["rate"][HALF].mean()))
        print(f"  seed {s} ({time.perf_counter()-t0:.0f}s)  " + "  ".join(
            f"m={m}: var {acc[(m,'variance')][-1]:.2f} / epi {acc[(m,'epistemic')][-1]:.2f}"
            for m in M_LIST), flush=True)

    print(f"\n{'m':>3} | {'variance':>18} | {'epistemic':>18} | {'delta':>8} | winner")
    print("-" * 72)
    for m in M_LIST:
        a = np.array(acc[(m, "variance")]); b = np.array(acc[(m, "epistemic")])
        d = b.mean() - a.mean()
        print(f"{m:3d} | {a.mean():9.3f} +/-{a.std():5.3f} | {b.mean():9.3f} +/-{b.std():5.3f} | "
              f"{d:+8.3f} | {'epistemic' if d > 0 else 'variance'}")

    print("\nPaired per-seed differences (epistemic - variance):")
    for m in M_LIST:
        d = np.array(acc[(m, "epistemic")]) - np.array(acc[(m, "variance")])
        wins = int((d > 0).sum())
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"  m={m}: mean {d.mean():+.3f}, se {se:.3f}, epistemic wins {wins}/{len(d)} seeds")

    print("\nGates:")
    for m in M_LIST:
        d = np.array(acc[(m, "epistemic")]) - np.array(acc[(m, "variance")])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("inf")
        check(f"m={m}: epistemic is not WORSE than variance beyond noise",
              d.mean() > -2 * se, f"delta {d.mean():+.3f} +/- {2*se:.3f} (2se)")

    print("\n" + "=" * 72)
    print(f"PILOT RULE: {'see table' if ok else 'EPISTEMIC LOSES -- weaken the paper claim'}"
          f"  ({time.perf_counter()-t0:.0f}s)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
