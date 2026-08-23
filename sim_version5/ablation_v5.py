r"""
V5-5 gate G5 -- the two ablations that justify the model (paper results).

(a) VALUE OF EXPLOITING CROSS-COLUMN CORRELATION (Option B vs Option A).
    Same dense (lambda/3) channel and everything else fixed; only the agent's BELIEF correlation
    model changes:
        B = full 2-D R      (models cross-column coupling)
        A = block-diag R    (assumes independent columns)
    Swept over Delta_max in {2, 7}. EMPIRICAL FINDING (not the original informed-jump hypothesis):
    the benefit is roughly Delta_max-INDEPENDENT -- it is a PERVASIVE inference gain (B's belief is
    better everywhere, so it selects better AND moves far less), not a niche long-jump effect.
    Gate: B_obj > A_obj at both Delta_max (I6), and B moves less than A (decisive vs blind chasing).

(b) COST OF THE ONE-DROPLET-PER-COLUMN HARDWARE CONSTRAINT (B vs S1-free).
    S1-free = unconstrained greedy selection of M ports anywhere (>= lambda/2 apart), same belief
    and partial CSI, no column/Delta_max constraint.   Gate: B captures most of S1-free's rate
    (the cheap-hardware story), and B <= genie.

Writes a summary to results_v5/ablation_v5.txt.
Run:  python ablation_v5.py [MC] [T]
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
from dataclasses import replace

sys.stdout.reconfigure(encoding="utf-8")

import efe
from config import OP_B
from channel import ChannelSimulator
from columns import sense
from precoding import mmse_precoder, sinr_and_rates
from agent_col import make_belief
from run_col import run_col_aif, run_col_genie, _rate

OP = OP_B
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 16
HALF = slice(T // 2, None)
OUT = os.path.join(os.path.dirname(__file__), "..", "results_v5", "ablation_v5.txt")

_lines = []
def emit(s=""):
    print(s); _lines.append(s)


def run_s1free(op, H, rng):
    """Unconstrained S1-style greedy selection of M ports anywhere (>= d_min apart)."""
    T = H.shape[0]; pos = op.positions()
    bel = make_belief(op)
    rate = np.zeros(T); S_prev = None
    for t in range(T):
        if t > 0:
            bel.predict()
        S = efe.greedy_select(bel, op.M, S_prev=S_prev, alpha=op.alpha, beta=op.beta_w,
                              eta_sw=op.eta_mv, e_sw=1.0, sigma2=op.sigma2, P=op.P,
                              positions=pos, d_min=op.d_min)
        y = sense(H[t], S, op.sigma_e2, rng)
        bel.update(S, y)
        W, _, _ = efe.robust_mmse_from_belief(bel, S, op.sigma2, op.P)
        rate[t] = _rate(H[t], S, W, op.sigma2)
        S_prev = S
    return dict(rate=rate)


def main():
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        emit(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

    t0 = time.perf_counter()
    emit(f"OP_B: {OP.label()}")
    emit(f"MC={MC}, T={T}, second-half slots\n")

    # shared trajectories + genie per seed
    Hs, genie = [], []
    for s in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                               rho=OP.rho, beta=OP.beta, seed=400 + s)
        H = sim.generate(T); Hs.append(H)
        genie.append(run_col_genie(OP, H, np.random.default_rng(900 + s))["rate"][HALF].mean())
    gmean = float(np.mean(genie))

    # ---- (a) B vs A across Delta_max ----
    emit("=" * 60)
    emit("(a) exploiting cross-column correlation:  B (full R) vs A (block R)")
    emit("=" * 60)
    emit(f"{'Delta_max':>9} | {'model':>6} | {'rate':>7} | {'move':>6} | {'obj':>7} | {'%genie':>7}")
    emit("-" * 56)
    gap = {}
    for dm in (2, 7):
        opdm = replace(OP, delta_max=dm)
        res = {}
        for tag, R in [("B", OP.R()), ("A", OP.R_block())]:
            objs, rates, moves = [], [], []
            for s in range(MC):
                r = run_col_aif(opdm, Hs[s], np.random.default_rng(500 + s), R=R)   # observe-then-precode
                rate = r["rate"][HALF].mean(); mv = r["move"][HALF].mean()
                rates.append(rate); moves.append(mv); objs.append(rate - OP.eta_mv * mv)
            res[tag] = (np.mean(rates), np.mean(moves), np.mean(objs))
            emit(f"{dm:>9} | {tag:>6} | {res[tag][0]:7.3f} | {res[tag][1]:6.2f} | "
                 f"{res[tag][2]:7.3f} | {100*res[tag][0]/gmean:7.1f}")
        gap[dm] = res["B"][2] - res["A"][2]
        emit(f"          -> B-minus-A objective at Delta_max={dm}: {gap[dm]:+.3f}")
        emit("")
        move_B, move_A = res["B"][1], res["A"][1]        # movement at this Delta_max (last = 7)

    # ---- (b) B vs S1-free vs genie ----
    emit("=" * 60)
    emit("(b) cost of one-droplet-per-column:  B vs S1-free (unconstrained) vs genie")
    emit("=" * 60)
    b_rate = []; s1_rate = []
    for s in range(MC):
        b_rate.append(run_col_aif(OP, Hs[s], np.random.default_rng(500 + s))["rate"][HALF].mean())
        s1_rate.append(run_s1free(OP, Hs[s], np.random.default_rng(510 + s))["rate"][HALF].mean())
    Rb, Rs1 = float(np.mean(b_rate)), float(np.mean(s1_rate))
    emit(f"{'method':>10} | {'rate':>7} | {'%genie':>7}")
    emit("-" * 32)
    emit(f"{'genie':>10} | {gmean:7.3f} | {100.0:7.1f}")
    emit(f"{'S1-free':>10} | {Rs1:7.3f} | {100*Rs1/gmean:7.1f}")
    emit(f"{'B (col)':>10} | {Rb:7.3f} | {100*Rb/gmean:7.1f}")

    emit("\nGates:")
    check("I6  B_obj > A_obj at Delta_max=2 (cross-column correlation helps)", gap[2] > 0,
          f"gap@2={gap[2]:+.3f}")
    check("I6  B_obj > A_obj at Delta_max=7 (cross-column correlation helps)", gap[7] > 0,
          f"gap@7={gap[7]:+.3f}")
    check("    benefit is pervasive: B moves less than A (decisive vs blind)",
          move_B < move_A - 1e-9, f"move B={move_B:.2f} vs A={move_A:.2f}")
    frac = 100 * Rb / Rs1
    check("    B captures >= 85% of S1-free rate (cheap hardware)", frac >= 85.0, f"{frac:.1f}%")
    check("    B rate <= genie", Rb <= gmean + 1e-6)

    emit("\n" + "=" * 44)
    emit(f"V5-5 GATE G5: {'ALL PASS' if ok else 'FAILURES ABOVE'}  ({time.perf_counter()-t0:.0f}s)")
    emit("=" * 44)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\n(summary written to {os.path.normpath(OUT)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
