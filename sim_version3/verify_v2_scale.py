r"""
OP_V2 scale check: 2 lambda aperture, 21x21 = 441 ports (lambda/10), M=8.

Answers three questions before any full figure sweep:

  A. TIMING     -- how long is ONE greedy selection at N=441 (~N*M evals/slot)? Is a
                   full sweep tolerable?
  B. SPACING    -- with d_min OFF, does the correlation-aware belief spontaneously keep
                   the selected ports >= lambda/2 apart at lambda/10 spacing, or does it
                   cluster? (the key question from the discussion)
  C. FEASIBILITY + EFFECT -- with d_min=0.5 ON, confirm feasibility and compare
                   rate/objective on vs off (short horizon, small MC -> indicative only).

Usage:  python sim_version2/verify_v2_scale.py
"""

from __future__ import annotations

import sys
import time
from itertools import combinations

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V2
from channel import ChannelSimulator
from belief import KalmanBelief
import efe
from agent import AIFAgent, run_aif, run_genie, objective


def min_pairwise_spacing(S, positions):
    idx = list(S)
    if len(idx) < 2:
        return np.inf
    return min(np.linalg.norm(positions[i] - positions[j]) for i, j in combinations(idx, 2))


def time_one_select(op, pos, R):
    """Time a single greedy_select on a fresh (predicted) belief, both arms."""
    bel = KalmanBelief(R=R, beta=op.beta, rho=op.rho, sigma_e2=op.sigma_e2)
    bel.predict()
    out = {}
    for tag, d in [("off", None), ("on", op.d_min)]:
        t0 = time.perf_counter()
        S = efe.greedy_select(bel, op.M, beta=op.beta_w, eta_sw=op.eta_sw,
                              sigma2=op.sigma2, P=op.P, positions=pos, d_min=d)
        dt = time.perf_counter() - t0
        out[tag] = (dt, S, min_pairwise_spacing(S, pos))
    return out


def run_arm(op, d_min, pos, R, MC=2, T=20):
    """Short pipeline; return mean rate/obj and the per-slot min-spacing trace of AIF."""
    a_rate = a_obj = g_rate = 0.0
    spacings = []
    h = slice(T // 2, None)
    for m in range(MC):
        sim = ChannelSimulator(Nx=op.Nx, Ny=op.Ny, Wx=op.Wx, Wy=op.Wy,
                               K=op.K, rho=op.rho, beta=op.beta, seed=m)
        H = sim.generate(T)
        ag = AIFAgent(R, op.beta, op.rho, op.sigma_e2, op.M, 1.0, op.beta_w,
                      op.eta_sw, sigma2=op.sigma2, positions=pos, d_min=d_min)
        # closed loop, but also snapshot the selected set's spacing each slot
        ag.reset()
        for t in range(T):
            S = ag.select(first=(t == 0))
            spacings.append(min_pairwise_spacing(S, pos))
            idx = list(S)
            noise = np.sqrt(op.sigma_e2 / 2) * (np.random.default_rng(1000 * m + t)
                    .standard_normal((op.K, len(idx))) + 1j * np.random.default_rng(2000 * m + t)
                    .standard_normal((op.K, len(idx))))
            y = H[t][:, idx] + noise
            ag.bel.update(S, y)
            ag.S_prev = S
        A = run_aif(AIFAgent(R, op.beta, op.rho, op.sigma_e2, op.M, 1.0, op.beta_w,
                    op.eta_sw, sigma2=op.sigma2, positions=pos, d_min=d_min),
                    H, op.sigma_e2, np.random.default_rng(20000 + m), sense_first=True)
        G = run_genie(H, op.M, sigma2=op.sigma2, positions=pos, d_min=d_min)
        a_rate += A["rate"][h].mean(); a_obj += objective(A, op.eta_sw)
        g_rate += G["rate"][h].mean()
    return dict(aif_rate=a_rate / MC, aif_obj=a_obj / MC, genie_rate=g_rate / MC,
                spacings=np.array(spacings))


def main():
    op = OP_V2
    print(f"OP_V2: {op.label()}\n", flush=True)

    t0 = time.perf_counter()
    pos, R = op.positions(), op.R()
    print(f"[setup] built positions + {R.shape} Jakes R in {time.perf_counter()-t0:.2f}s "
          f"(min eig {np.linalg.eigvalsh(R).min():.2e})\n", flush=True)

    # A. timing -------------------------------------------------------------
    print("[A] timing one greedy selection at N=441 ...", flush=True)
    tsel = time_one_select(op, pos, R)
    for tag in ("off", "on"):
        dt, S, gap = tsel[tag]
        print(f"    d_min {tag:3s}: {dt:.2f}s/slot | tightest spacing {gap:.3f} lambda | S={S}", flush=True)
    print(f"    -> a T=45, MC=15 sweep of ONE figure ~ {tsel['off'][0]*45*15/60:.1f} min of greedy alone\n",
          flush=True)

    # B + C. arms -----------------------------------------------------------
    print("[B/C] short pipeline, d_min OFF ...", flush=True)
    off = run_arm(op, None, pos, R)
    print(f"    OFF: AIF {off['aif_rate']:.2f} rate / {off['aif_obj']:.2f} obj | genie {off['genie_rate']:.2f}",
          flush=True)
    sp = off["spacings"]
    print(f"         spontaneous min-spacing (belief only): median {np.median(sp):.3f}, "
          f"min {sp.min():.3f} lambda; {100*np.mean(sp >= 0.5):.0f}% of slots already >= 0.5\n", flush=True)

    print("[C] short pipeline, d_min = 0.5 ON ...", flush=True)
    on = run_arm(op, op.d_min, pos, R)
    print(f"    ON : AIF {on['aif_rate']:.2f} rate / {on['aif_obj']:.2f} obj | genie {on['genie_rate']:.2f}",
          flush=True)
    print(f"         min-spacing: min {on['spacings'].min():.3f} lambda "
          f"(must be >= 0.5): {'OK' if on['spacings'].min() >= 0.5 - 1e-9 else 'VIOLATION'}\n", flush=True)

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
