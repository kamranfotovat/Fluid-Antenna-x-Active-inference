r"""Is the greedy pilot subset any good? -- Zijun's question, answered by brute force.

Q_t is built by SEQUENTIAL greedy on the epistemic term: at each step add the port that
maximizes Epis(Q u {p}) GIVEN what is already chosen. That is not the same as ranking the
M activated ports once by their individual information gain and taking the top m -- the
value of a port depends on which ports are already in Q, which is the whole point of using
a log-determinant instead of a sum of variances.

Greedy on a monotone submodular set function carries the standard (1-1/e) ~ 0.632
guarantee. But that bound is worst-case and pessimistic. At our operating point the choice
is small enough to settle exactly: C(M,m) = C(10,6) = 210 subsets. So we compare greedy
against the true optimum, per slot, on real belief states from a closed-loop run.

Reports, for each m:
  - the ratio Epis(greedy) / Epis(optimal), worst and mean over slots
  - how often greedy IS the optimal subset
  - how far either is from the max-variance heuristic, for reference

Run:  python verify_pilot_optimality.py [MC] [T]
"""

from __future__ import annotations

import itertools
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import efe
from config import OP_V2, OP_V3
from channel import spatial_correlation
from temporal import generate_spacetime_jakes
from st_belief_lr import STKalmanBeliefLR
from st_belief import choose_pilots
from agent import _obs, _switch_count

MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 20
FD, P_AR = 0.10, 4
M_LIST = [4, 6, 8]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    OP = OP_V3
    print(f"PILOT-SUBSET OPTIMALITY   MC={MC}, T={T}, AR({P_AR}), M={OP.M}")
    print(f"  subsets to enumerate per slot: " + ", ".join(
        f"C({OP.M},{m})={len(list(itertools.combinations(range(OP.M), m)))}" for m in M_LIST))
    print(f"  {OP.label()}\n", flush=True)

    R = spatial_correlation(OP_V2.positions())
    pos = OP.positions()
    stats = {m: {"ratio": [], "hit": 0, "n": 0, "var_ratio": []} for m in M_LIST}
    t0 = time.perf_counter()

    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD, T, OP.K, seed=100 + s)
        bel = STKalmanBeliefLR(R, OP.beta, FD, P_AR, OP.sigma_e2)
        bel.reset()
        rng = np.random.default_rng(200 + s)
        S_prev = None
        for t in range(T):
            if t > 0:
                bel.predict()
            S = efe.greedy_select(bel, OP.M, S_prev=S_prev, alpha=1.0, beta=OP.beta_w,
                                  eta_sw=OP.eta_sw, e_sw=1.0, sigma2=OP.sigma2, P=OP.P,
                                  positions=pos, d_min=OP.d_min)
            idx = list(S)
            for m in M_LIST:
                g = tuple(choose_pilots(bel, idx, m, "epistemic"))
                o = tuple(choose_pilots(bel, idx, m, "exhaustive"))
                v = tuple(choose_pilots(bel, idx, m, "variance"))
                eg, eo, ev = (efe.epistemic_value(bel, x) for x in (g, o, v))
                st = stats[m]
                st["ratio"].append(eg / eo)
                st["var_ratio"].append(ev / eo)
                st["hit"] += int(set(g) == set(o))
                st["n"] += 1
            # advance the loop with the proposed rule at the paper's budget
            S_sense = choose_pilots(bel, idx, 6, "epistemic")
            bel.update(tuple(S_sense), _obs(H[t], S_sense, OP.K, OP.sigma_e2, rng))
            S_prev = S
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"\n{'m':>3} | {'greedy/opt mean':>16} | {'greedy/opt worst':>17} | "
          f"{'greedy == opt':>14} | {'variance/opt':>13}")
    print("-" * 76)
    for m in M_LIST:
        st = stats[m]
        r = np.array(st["ratio"]); vr = np.array(st["var_ratio"])
        print(f"{m:3d} | {r.mean():15.5f}  | {r.min():16.5f}  | "
              f"{st['hit']:6d}/{st['n']:<6d} | {vr.mean():12.5f}")

    print("\nGates:")
    for m in M_LIST:
        r = np.array(stats[m]["ratio"])
        check(f"m={m}: greedy beats the (1-1/e) worst-case bound", r.min() > 1 - 1 / np.e,
              f"worst ratio {r.min():.5f} vs bound {1-1/np.e:.5f}")
    allr = np.concatenate([np.array(stats[m]["ratio"]) for m in M_LIST])
    check("greedy is within 1% of optimal everywhere", allr.min() > 0.99,
          f"worst ratio over all m and slots {allr.min():.5f}")

    print("\n" + "=" * 76)
    print(f"PILOT OPTIMALITY: {'ALL PASS' if ok else 'SEE FAILURES'}  "
          f"({time.perf_counter()-t0:.0f}s)")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
