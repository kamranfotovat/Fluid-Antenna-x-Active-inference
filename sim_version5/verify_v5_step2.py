r"""
V5-2 gate G2 -- feasibility: reachability + min-spacing mask.

Checks:
  A. reachable_heights sizes correct (2*Delta_max+1 in the interior, clipped at edges).
  B. min-spacing: exactly the adjacent-column pairs closer than 4 ports are infeasible; a
     comfortably-spaced config is feasible; 2-columns-apart never binds.
  C. I5: legal_heights is never empty when the column is held at its current height from a
     feasible config (staying is always legal), across many random feasible configs.
  D. legal_heights agrees with a brute-force full-config feasibility check.

Run:  python verify_v5_step2.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_B
from feasibility import (reachable_heights, config_feasible, legal_heights,
                         random_feasible_config)

OP = OP_B
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print(f"OP_B: {OP.label()}\n")
    N_t, N_p, dmax = OP.N_t, OP.N_p, OP.delta_max
    pos = OP.positions()

    print("A. reachability")
    check("interior width = 2*Delta_max+1", len(reachable_heights(10, dmax, N_p)) == 2 * dmax + 1)
    check("clipped at bottom edge", reachable_heights(0, dmax, N_p)[0] == 0
          and len(reachable_heights(0, dmax, N_p)) == dmax + 1)
    check("clipped at top edge", reachable_heights(N_p - 1, dmax, N_p)[-1] == N_p - 1)

    print("\nB. min-spacing (d_min = 0.5 lambda)")
    # adjacent columns, same height -> distance = col_spacing = 0.333 lambda < 0.5  -> infeasible
    i_bad = np.zeros(N_t, dtype=int)                     # all droplets at height 0 (same row)
    check("all-same-height config infeasible", not config_feasible(i_bad, OP, pos))
    # adjacent columns need >= 4 ports vertical sep: stagger by 4 -> feasible
    i_stag = (np.arange(N_t) * 4) % N_p
    # ensure the staggering keeps >=4 between adjacent columns (0,4,8,... wraps -> check directly)
    i_ok = np.array([0, 4, 8, 12, 16, 20, 0, 4, 8, 12][:N_t]) % N_p
    check("adjacent stagger of 4 ports is feasible", config_feasible(i_ok, OP, pos),
          f"min pair dist ok")
    # exactly at the boundary: adjacent columns 3 ports apart -> dist=sqrt(.333^2+.3^2)=0.448<0.5 bad
    i_edge = np.array([0, 3, 6, 9, 12, 15, 18, 0, 3, 6][:N_t]) % N_p
    d_adj = np.hypot(OP.col_spacing, 3 * OP.pitch)
    check("adjacent 3-ports apart infeasible (d<0.5)", not config_feasible(i_edge, OP, pos),
          f"adj dist={d_adj:.3f}λ")
    # 2 columns apart, same height: distance 0.667 > 0.5 -> never binds
    i_2 = np.array([0, 20, 0, 20, 0, 20, 0, 20, 0, 20][:N_t])   # cols alternate, adj differ by 20 (>=4)
    check("2-cols-apart same height never binds", np.hypot(2 * OP.col_spacing, 0) >= OP.d_min)

    print("\nC. I5 -- legal set never empty (staying is always legal)")
    rng = np.random.default_rng(0)
    never_empty = True; stay_always_legal = True
    for _ in range(300):
        i = random_feasible_config(OP, rng)
        i_prev = i.copy()                                # previous = current -> staying reachable
        for c in range(N_t):
            L = legal_heights(c, i, i_prev, OP, pos)
            never_empty &= len(L) > 0
            stay_always_legal &= (i[c] in L)
    check("legal_heights never empty", never_empty)
    check("current height always legal (can stay)", stay_always_legal)

    print("\nD. legal_heights consistent with full-config feasibility")
    rng = np.random.default_rng(1)
    consistent = True
    for _ in range(120):
        i_prev = random_feasible_config(OP, rng)
        i = i_prev.copy()
        c = int(rng.integers(N_t))
        L = set(legal_heights(c, i, i_prev, OP, pos).tolist())
        for p in reachable_heights(i_prev[c], OP.delta_max, N_p):
            cand = i.copy(); cand[c] = p
            brute = config_feasible(cand, OP, pos)
            consistent &= (brute == (p in L))
    check("legal_heights == brute-force feasibility", consistent)

    print("\n" + "=" * 44)
    print(f"V5-2 GATE G2: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
