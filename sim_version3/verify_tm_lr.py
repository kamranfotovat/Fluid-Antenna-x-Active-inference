r"""
Gate for the reduced-rank ST filter -- it must reproduce the EXACT filter before we trust full scale.

At N=25 both are runnable, so compare them head to head:
  (A) belief agreement : same channel, same sensing schedule -> mu/Sigma match to truncation energy
  (B) closed-loop rate : run_st with each belief gives the same rate (both protocols)
  (C) speed            : the reason we need it at all

Run:  python verify_tm_lr.py
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V1, OP_V2
from channel import spatial_correlation
from temporal import generate_spacetime_jakes
from st_belief import STKalmanBelief, run_st
from st_belief_lr import STKalmanBeliefLR
from agent import _obs

OP = OP_V1
FD, P, T = 0.10, 4, 20
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    R = spatial_correlation(OP.positions())
    print(f"OP_V1: {OP.label()}\nfd={FD}, AR(p={P}), T={T}\n")

    for OPx, nm in [(OP_V1, "OP_V1"), (OP_V2, "OP_V2")]:
        Rx = spatial_correlation(OPx.positions())
        w = np.clip(np.linalg.eigvalsh(Rx)[::-1], 0, None)
        rk = int(np.searchsorted(np.cumsum(w) / w.sum(), 1 - 1e-6) + 1)
        print(f"  {nm}: N={OPx.N:4d}  numerical rank (1-1e-6 energy) = {rk:3d}"
              f"   -> state {P*OPx.N} -> {P*rk}   speedup ~{(OPx.N/rk)**3:.0f}x")

    # ---------------- (A) belief agreement under an identical sensing schedule
    print("\n(A) belief agreement (identical forced sensing schedule)")
    H = generate_spacetime_jakes(R, OP.beta, FD, T, OP.K, seed=7)
    ex = STKalmanBelief(R, OP.beta, FD, P, OP.sigma_e2)
    lr = STKalmanBeliefLR(R, OP.beta, FD, P, OP.sigma_e2)
    print(f"    reduced rank r={lr.r} of N={lr.N}")
    rng = np.random.default_rng(0)
    dmu = dsig = 0.0
    for t in range(T):
        if t > 0:
            ex.predict(); lr.predict()
        S = tuple(sorted(rng.choice(OP.N, OP.M, replace=False).tolist()))
        y = _obs(H[t], list(S), OP.K, OP.sigma_e2, np.random.default_rng(1000 + t))
        ex.update(S, y); lr.update(S, y)
        dmu = max(dmu, np.max(np.abs(ex.mu - lr.mu)))
        dsig = max(dsig, np.max(np.abs(ex.Sigma - lr.Sigma)))
    scale = float(np.max(np.abs(ex.Sigma)))
    check("mean agrees", dmu < 1e-4, f"max|dmu| = {dmu:.2e}")
    check("covariance agrees", dsig / scale < 1e-4, f"max|dSigma|/scale = {dsig/scale:.2e}")

    # ---------------- (A2) at FULL rank the two filters must be bit-identical
    print("\n(A2) full-rank LR == exact filter (proves the reduction, not just the truncation)")
    full = STKalmanBeliefLR(R, OP.beta, FD, P, OP.sigma_e2, rank=OP.N)
    r_ex = run_st(STKalmanBelief(R, OP.beta, FD, P, OP.sigma_e2), H, OP,
                  np.random.default_rng(5), protocol="predict")["rate"]
    r_fr = run_st(full, H, OP, np.random.default_rng(5), protocol="predict")["rate"]
    check("full-rank rate trajectory identical", np.max(np.abs(r_ex - r_fr)) < 1e-6,
          f"max|dr| = {np.max(np.abs(r_ex - r_fr)):.2e} over all {T} slots")

    # ---------------- (B) closed-loop rate agreement (MEAN over seeds)
    # Greedy selection is DISCRETE: a 1e-7 belief difference can flip a near-tie and change the
    # port set, so single trajectories diverge chaotically even though the filters agree. The
    # meaningful gate is that the MEAN rate agrees within Monte-Carlo noise.
    print("\n(B) closed-loop mean rate over seeds (selection is discrete -> compare means)")
    NS = 4
    for proto in ["observe", "predict"]:
        ex_r, lr_r, t_ex, t_lr = [], [], 0.0, 0.0
        for s in range(NS):
            Hs = generate_spacetime_jakes(R, OP.beta, FD, T, OP.K, seed=300 + s)
            t0 = time.perf_counter()
            ex_r.append(run_st(STKalmanBelief(R, OP.beta, FD, P, OP.sigma_e2), Hs, OP,
                               np.random.default_rng(5 + s), protocol=proto)["rate"].mean())
            t_ex += time.perf_counter() - t0
            t0 = time.perf_counter()
            lr_r.append(run_st(STKalmanBeliefLR(R, OP.beta, FD, P, OP.sigma_e2), Hs, OP,
                               np.random.default_rng(5 + s), protocol=proto)["rate"].mean())
            t_lr += time.perf_counter() - t0
        E, L = float(np.mean(ex_r)), float(np.mean(lr_r))
        sem = float(np.std(np.array(ex_r) - np.array(lr_r), ddof=1) / np.sqrt(NS))
        check(f"{proto:>8}: mean rate within MC noise", abs(E - L) < max(3 * sem, 0.05),
              f"exact {E:.4f} vs LR {L:.4f}  (diff {E-L:+.4f}, 3*sem {3*sem:.4f}; "
              f"{t_ex:.1f}s vs {t_lr:.1f}s)")

    # ---------------- (C) full-scale feasibility
    print("\n(C) full-scale (OP_V2, N=441) single-slot timing")
    R2 = spatial_correlation(OP_V2.positions())
    lr2 = STKalmanBeliefLR(R2, OP_V2.beta, FD, P, OP_V2.sigma_e2)
    S2 = tuple(range(0, 10 * 40, 40))
    y2 = _obs(np.zeros((OP_V2.K, OP_V2.N), complex), list(S2), OP_V2.K, OP_V2.sigma_e2,
              np.random.default_rng(0))
    t0 = time.perf_counter()
    for _ in range(5):
        lr2.predict(); lr2.update(S2, y2)
    dt = (time.perf_counter() - t0) / 5
    check("full-scale belief step is cheap", dt < 0.2, f"r={lr2.r}, {dt*1000:.0f} ms/slot")

    print("\n" + "=" * 44)
    print(f"LR gate: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
