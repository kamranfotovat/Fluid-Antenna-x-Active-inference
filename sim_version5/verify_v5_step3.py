r"""
V5-3 gate G3 -- movement cost + myopic coordinate-descent selection.

Checks:
  A. I2 -- movement cost with every |Delta i_c| <= 1 equals the number of repositioned droplets,
     and equals |S XOR S_prev| / 2 (each reposition removes one port and adds one).
  B. I4 -- coordinate descent is monotone (G non-increasing across sweeps) and converges.
  C. correctness -- on a small instance (N_t=2, N_p=6) the selector matches an EXHAUSTIVE
     argmin G over all feasible & reachable configs.
  D. myopic selection beats a random feasible config on the objective at full scale.

Run:  python verify_v5_step3.py
"""

from __future__ import annotations

import sys
import itertools
import numpy as np
from dataclasses import replace

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_B
from channel import ChannelSimulator
from agent_col import make_belief, sense_and_update
from columns import pos_to_ports
from feasibility import config_feasible, reachable_heights, random_feasible_config
from efe_col import movement_cost, free_energy, select_myopic

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def exhaustive_min(bel, i_prev, op, pos):
    """Brute-force argmin G over all feasible configs reachable from i_prev."""
    best_i, best_G = None, np.inf
    for combo in itertools.product(range(op.N_p), repeat=op.N_t):
        i = np.array(combo, dtype=int)
        if i_prev is not None:
            if np.any(np.abs(i - i_prev) > op.delta_max):
                continue
        if not config_feasible(i, op, pos):
            continue
        G = free_energy(bel, i, i_prev, op)
        if G < best_G:
            best_G, best_i = G, i
    return best_i, best_G


def main():
    print(f"OP_B: {OP_B.label()}\n")
    OP = OP_B

    # --- A. I2 movement == switching ---
    print("A. I2 -- movement cost reduces to the switch count when |Delta i| <= 1")
    rng = np.random.default_rng(0)
    i2_ok = True
    for _ in range(200):
        i_prev = rng.integers(2, OP.N_p - 2, size=OP.N_t)
        step = rng.integers(-1, 2, size=OP.N_t)          # each column moves -1, 0, or +1
        i = i_prev + step
        n_moved = int(np.sum(step != 0))
        mv = movement_cost(i, i_prev, eta_mv=1.0)
        S, Sp = set(pos_to_ports(i, OP.N_t)), set(pos_to_ports(i_prev, OP.N_t))
        xor = len(S ^ Sp)
        i2_ok &= (abs(mv - n_moved) < 1e-9) and (xor == 2 * n_moved)
    check("movement == #repositioned droplets", i2_ok)
    check("#repositioned == |S XOR S_prev| / 2", i2_ok)

    # --- build a belief with structure for the optimisation checks ---
    sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                           rho=OP.rho, beta=OP.beta, seed=3)
    H = sim.generate(2)
    bel = make_belief(OP)
    i_prev = random_feasible_config(OP, np.random.default_rng(5))
    sense_and_update(bel, H[0], i_prev, OP, np.random.default_rng(6))
    bel.predict()

    # --- B. I4 monotone & convergent ---
    print("\nB. I4 -- coordinate descent monotone & convergent")
    _, _, trace = select_myopic(bel, i_prev, OP, np.random.default_rng(7),
                                n_restart=1, max_sweeps=8, return_trace=True)
    monotone = all(trace[k + 1] <= trace[k] + 1e-9 for k in range(len(trace) - 1))
    converged = len(trace) < 8 or abs(trace[-1] - trace[-2]) < 1e-9
    check("G non-increasing across sweeps", monotone, f"G: {', '.join(f'{g:.3f}' for g in trace)}")
    check("converged (stopped changing)", converged)

    # --- C. correctness vs exhaustive on a small instance ---
    print("\nC. correctness -- matches exhaustive argmin G (N_t=2, N_p=6)")
    small = replace(OP_B, N_t=2, N_p=6, delta_max=6)
    ssim = ChannelSimulator(Nx=small.Nx, Ny=small.Ny, Wx=small.Wx, Wy=small.Wy, K=small.K,
                            rho=small.rho, beta=small.beta, seed=11)
    sH = ssim.generate(2)
    match = True; worst = 0.0
    for trial in range(12):
        sb = make_belief(small)
        ip = random_feasible_config(small, np.random.default_rng(20 + trial))
        sense_and_update(sb, sH[0], ip, small, np.random.default_rng(30 + trial))
        sb.predict()
        pos = small.positions()
        _, Gbrute = exhaustive_min(sb, ip, small, pos)
        _, Gmyo = select_myopic(sb, ip, small, np.random.default_rng(40 + trial),
                                n_restart=4, max_sweeps=8)
        worst = max(worst, Gmyo - Gbrute)
        match &= (Gmyo <= Gbrute + 1e-6)
    check("coordinate descent reaches the global min", match, f"worst gap={worst:.2e}")

    # --- D. myopic beats random feasible ---
    print("\nD. myopic selection beats a random feasible config")
    wins = 0; trials = 12
    for trial in range(trials):
        b = make_belief(OP)
        ip = random_feasible_config(OP, np.random.default_rng(50 + trial))
        sense_and_update(b, H[0], ip, OP, np.random.default_rng(60 + trial))
        b.predict()
        _, Gmyo = select_myopic(b, ip, OP, np.random.default_rng(70 + trial))
        irand = random_feasible_config(OP, np.random.default_rng(80 + trial), i_prev=ip)
        Grand = free_energy(b, irand, ip, OP)
        wins += (Gmyo <= Grand + 1e-9)
    check("myopic <= random feasible on G (all trials)", wins == trials, f"{wins}/{trials}")

    print("\n" + "=" * 44)
    print(f"V5-3 GATE G3: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
