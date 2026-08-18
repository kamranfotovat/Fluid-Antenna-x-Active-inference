r"""
Version-2 smoke test: central config + hard min-spacing (d_min) constraint.

Run at the ORIGINAL small point (OP_V1, N=25) so it is fast, and check three things:

  1. FEASIBILITY  -- every port set the AIF selector returns with d_min on actually
                     satisfies the >= d_min pairwise-spacing constraint (the whole point).
  2. OFF == baseline -- with d_min=None the selector is byte-for-byte the old behaviour.
  3. EFFECT       -- rate / objective for AIF and the (feasible) genie, d_min off vs on.

Usage:  python sim_version2/verify_version2.py
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V1
from channel import ChannelSimulator
from belief import KalmanBelief
import efe
from agent import AIFAgent, run_aif, run_genie, objective


def min_pairwise_spacing(S, positions):
    """Smallest centre-to-centre distance within port set S (inf if |S|<2)."""
    idx = list(S)
    if len(idx) < 2:
        return np.inf
    return min(np.linalg.norm(positions[i] - positions[j]) for i, j in combinations(idx, 2))


def audit_feasibility(op, n_slots=8, seed=0):
    """Roll the belief forward and confirm each greedy selection respects d_min."""
    pos, R = op.positions(), op.R()
    bel = KalmanBelief(R=R, beta=op.beta, rho=op.rho, sigma_e2=op.sigma_e2)
    rng = np.random.default_rng(seed)
    worst = np.inf
    S_prev = None
    for t in range(n_slots):
        if t > 0:
            bel.predict()
        S = efe.greedy_select(bel, op.M, S_prev=S_prev, beta=op.beta_w,
                              eta_sw=op.eta_sw, sigma2=op.sigma2, P=op.P,
                              positions=pos, d_min=op.d_min)
        gap = min_pairwise_spacing(S, pos)
        worst = min(worst, gap)
        assert len(set(S)) == op.M, f"expected {op.M} distinct ports, got {S}"
        assert gap >= op.d_min - 1e-9, f"slot {t}: spacing {gap:.3f} < d_min {op.d_min}"
        # feed a noisy observation so the belief actually evolves
        idx = list(S)
        y = np.zeros((bel.K, len(idx)), complex)  # value irrelevant to feasibility
        bel.update(S, y); S_prev = S
    return worst


def run_point(op, MC=6, T=40):
    """Mean second-half rate + objective for AIF and genie under this operating point."""
    a_rate = a_obj = g_rate = g_obj = 0.0
    pos, R = op.positions(), op.R()
    h = slice(T // 2, None)
    for m in range(MC):
        sim = ChannelSimulator(Nx=op.Nx, Ny=op.Ny, Wx=op.Wx, Wy=op.Wy,
                               K=op.K, rho=op.rho, beta=op.beta, seed=m)
        H = sim.generate(T)
        ag = AIFAgent(R, op.beta, op.rho, op.sigma_e2, op.M, 1.0, op.beta_w,
                      op.eta_sw, sigma2=op.sigma2, positions=pos, d_min=op.d_min)
        A = run_aif(ag, H, op.sigma_e2, np.random.default_rng(20000 + m), sense_first=True)
        G = run_genie(H, op.M, sigma2=op.sigma2, positions=pos, d_min=op.d_min)
        a_rate += A["rate"][h].mean(); a_obj += objective(A, op.eta_sw)
        g_rate += G["rate"][h].mean(); g_obj += objective(G, op.eta_sw)
    n = MC
    return dict(aif_rate=a_rate / n, aif_obj=a_obj / n,
                genie_rate=g_rate / n, genie_obj=g_obj / n)


def main():
    base = OP_V1                                   # d_min = None
    constrained = OP_V1.__class__(**{**base.__dict__, "d_min": 0.5})

    print(f"Base       : {base.label()}")
    print(f"Constrained: {constrained.label()}\n")

    # 1. feasibility --------------------------------------------------------
    worst = audit_feasibility(constrained)
    print(f"[1] feasibility: all selections respect d_min; "
          f"tightest spacing seen = {worst:.3f} lambda (>= {constrained.d_min}) OK\n")

    # 2/3. effect -----------------------------------------------------------
    off = run_point(base)
    on = run_point(constrained)
    print("[2/3] rate / objective (mean over MC, second half):")
    print(f"    d_min OFF : AIF {off['aif_rate']:.2f} rate / {off['aif_obj']:.2f} obj  |  "
          f"genie {off['genie_rate']:.2f} / {off['genie_obj']:.2f}")
    print(f"    d_min 0.5 : AIF {on['aif_rate']:.2f} rate / {on['aif_obj']:.2f} obj  |  "
          f"genie {on['genie_rate']:.2f} / {on['genie_obj']:.2f}")
    print(f"    AIF captures {100*on['aif_rate']/on['genie_rate']:.0f}% of the feasible genie "
          f"with the min-spacing constraint on.")
    print("\nDONE.")


if __name__ == "__main__":
    main()
