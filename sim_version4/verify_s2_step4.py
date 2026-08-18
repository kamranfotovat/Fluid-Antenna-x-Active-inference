"""
S2 Step 4 verification -- the unit-modulus sensing-matrix optimizer design_sensing_matrix.

Gate before S2-5 (Light-S2 closed loop). Checks:

  A. I7 MONOTONE -- the accepted-J trace never decreases (backtracking ascent).
  B. UNIT MODULUS -- |F| = 1 exactly at the returned solution.
  C. I4 BOUNDED + QUALITY -- designed J <= unconstrained water-filling bound (S2-3), and captures
     a high fraction of it (there IS a real gap the design must fight for at small n_rf).
  D. I3 DESIGN > RANDOM -- designed J beats the best of many random unit-modulus F by a margin;
     and the warm start helps (designed >= any single random-restart run).

Run:  python sim_version4/verify_s2_step4.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402
from belief import KalmanBelief                # noqa: E402
import efe                                      # noqa: E402
from sensing import (sense_info, optimal_unconstrained,   # noqa: E402
                     design_sensing_matrix)


def _make_cov_bar(seed=11):
    """A realistic low-rank aggregate active-port covariance after a partial per-port read."""
    rho, sigma_e2 = 0.9, 1e-2
    beta = np.array([1.0, 0.7, 1.3]); K = len(beta)
    sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=seed)
    S = (0, 2, 7, 12, 18, 24); M = len(S)
    bel = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
    bel.update((0, 12, 24), np.zeros((K, 3)))          # sharpen a few ports -> spread the spectrum
    bel.predict()
    cov_bar = np.sum(efe.active_covs(bel, S), axis=0)
    return 0.5 * (cov_bar + cov_bar.conj().T), M, sigma_e2


def _rand_um(M, n_rf, rng):
    return np.exp(1j * rng.uniform(0, 2 * np.pi, size=(M, n_rf)))


def main():
    cov_bar, M, sigma_e2 = _make_cov_bar()
    ev = np.linalg.eigvalsh(cov_bar)
    print(f"Sigma_bar: M={M}, eig range [{ev.min():.2e}, {ev.max():.2e}], sigma_e^2={sigma_e2}")
    all_pass = True

    # ---- A: I7 monotone trace ------------------------------------------------
    print("\n[A] I7: accepted-J trace is monotone non-decreasing")
    _, _, tr = design_sensing_matrix(cov_bar, 3, sigma_e2, rng=np.random.default_rng(0),
                                     return_trace=True)
    drops = [tr[i + 1] - tr[i] for i in range(len(tr) - 1)]
    worst_drop = min(drops) if drops else 0.0
    ok = worst_drop >= -1e-12
    all_pass &= ok
    print(f"   trace len={len(tr)}, start={tr[0]:.4f} -> end={tr[-1]:.4f}, worst step={worst_drop:.1e}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- B: exact unit modulus ----------------------------------------------
    print("\n[B] Unit modulus: |F| == 1 exactly")
    F, J = design_sensing_matrix(cov_bar, 4, sigma_e2, rng=np.random.default_rng(1))
    um_err = float(np.max(np.abs(np.abs(F) - 1.0)))
    ok = um_err < 1e-12
    all_pass &= ok
    print(f"   max||F|-1| = {um_err:.1e}  -> {'PASS' if ok else 'FAIL'}")

    # ---- C: bounded by unconstrained + captures most of it -------------------
    print("\n[C] I4: designed J <= water-filling bound, and captures a high fraction")
    ok = True
    for n_rf in [1, 2, 3, 4, 6]:
        Jwf, _, _ = optimal_unconstrained(cov_bar, n_rf, sigma_e2)   # Ptot = M*n_rf (unit-mod energy)
        Fd, Jd = design_sensing_matrix(cov_bar, n_rf, sigma_e2, rng=np.random.default_rng(2))
        frac = Jd / Jwf if Jwf > 0 else 1.0
        bounded = Jd <= Jwf + 1e-6
        ok = ok and bounded and (frac > 0.85)
        print(f"   n_rf={n_rf}: designed={Jd:7.4f} | bound={Jwf:7.4f} | frac={frac:5.3f}"
              f" {'ok' if bounded and frac > 0.85 else 'BAD'}")
    all_pass &= ok
    print(f"   -> {'PASS' if ok else 'FAIL'}")

    # ---- D: designed beats random unit-modulus, warm start helps -------------
    print("\n[D] I3: designed > best random unit-modulus; warm start >= random-only restarts")
    ok = True
    for n_rf in [2, 3, 4]:
        rng = np.random.default_rng(7)
        Jrand = max(sense_info(_rand_um(M, n_rf, rng), cov_bar, sigma_e2) for _ in range(400))
        Fd, Jd = design_sensing_matrix(cov_bar, n_rf, sigma_e2, rng=np.random.default_rng(3))
        # warm start OFF (random restarts only) as a reference
        _, Jrs = design_sensing_matrix(cov_bar, n_rf, sigma_e2, warm_start=False,
                                       rng=np.random.default_rng(3))
        better = Jd > Jrand + 1e-6 and Jd >= Jrs - 1e-6
        ok = ok and better
        print(f"   n_rf={n_rf}: designed={Jd:7.4f} | best-random={Jrand:7.4f} | random-restart-opt={Jrs:7.4f}"
              f" {'ok' if better else 'BAD'}")
    all_pass &= ok
    print(f"   -> {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 46)
    print(f"S2 STEP 4 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
