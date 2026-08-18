"""
S2 Step 2 verification -- the sensing-through-F_RF observation model is correct.

Gate before S2-3 (the info objective). Checks:

  A. I1 ANCHOR (the master reduction) -- F_RF = I_M, n_rf = M gives A = P_S, and the full
     pipeline sense(...) -> update_general(A, y) is BIT-FOR-BIT identical to S1 per-port sensing
     update(S, y). Same rng => same noise draw (both draw a (K, M) complex block).
  B. CONSISTENCY -- sense() clean part equals A applied to the true channel: y_clean == (A h)^T.
     Guards that observation_matrix and sense use the same F_RF^H convention.
  C. NOISE POWER -- empirical Var of the additive noise ~ sigma_e^2.
  D. SHARPENING -- a mixed (n_rf < M) read still reduces total uncertainty on the active ports.

Run:  python sim_version4/verify_s2_step2.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402
from belief import KalmanBelief                # noqa: E402
from sensing import observation_matrix, sense  # noqa: E402
import efe                                      # noqa: E402


def _fro_rel(A, B):
    return np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-300)


def main():
    rho, sigma_e2 = 0.9, 1e-2
    beta = np.array([1.0, 0.7, 1.3]); K = len(beta)
    S = (0, 4, 12, 20, 24); M = len(S)
    sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=7)
    N = sim.N
    H_t = sim.h                                        # (K, N) one slot
    print(f"N={N} (5x5), K={K}, active S={S} (M={M}), sigma_e^2={sigma_e2}")
    all_pass = True

    # ---- A: I1 anchor -- F_RF = I_M reproduces S1 per-port sensing bit-for-bit ----
    print("\n[A] I1 anchor: F_RF=I_M (n_rf=M) == S1 per-port sensing (pipeline bit-for-bit)")
    F_I = np.eye(M, dtype=complex)
    A = observation_matrix(F_I, S, N)
    P = np.zeros((M, N)); P[np.arange(M), list(S)] = 1.0
    errA_mat = _fro_rel(A, P.astype(complex))
    # same rng -> identical noise; S1 draws (K,M), sense draws (K, n_rf=M)
    r1 = np.random.default_rng(0); r2 = np.random.default_rng(0)
    y_s1 = H_t[:, list(S)] + np.sqrt(sigma_e2 / 2) * (
        r1.standard_normal((K, M)) + 1j * r1.standard_normal((K, M)))
    y_s2 = sense(H_t, F_I, S, sigma_e2, r2)
    err_y = _fro_rel(y_s2, y_s1)
    b1 = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2); b1.update(S, y_s1)
    b2 = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2); b2.update_general(A, y_s2)
    err_bel = max(_fro_rel(b1.Sigma[k], b2.Sigma[k]) for k in range(K))
    ok = errA_mat < 1e-12 and err_y < 1e-12 and err_bel < 1e-12
    all_pass &= ok
    print(f"   A vs P_S = {errA_mat:.1e} | y_S2 vs y_S1 = {err_y:.1e} | belief Sigma = {err_bel:.1e}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- B: consistency  y_clean == (A h)^T ------------------------------------
    print("\n[B] Consistency: sense() clean part == observation_matrix applied to h")
    rng = np.random.default_rng(1)
    F = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(M, 3)))     # unit-modulus, n_rf=3
    A3 = observation_matrix(F, S, N)
    y_clean = sense(H_t, F, S, 0.0, rng)               # sigma_e2=0 -> clean
    y_ref = (A3 @ H_t.T).T                             # (K, n_rf): row k = A3 @ h_k = F^H h_S[k]
    ok = _fro_rel(y_clean, y_ref) < 1e-12
    all_pass &= ok
    print(f"   ||y_clean - (A h)^T|| rel = {_fro_rel(y_clean, y_ref):.1e}  -> {'PASS' if ok else 'FAIL'}")

    # ---- C: noise power ~ sigma_e^2 -------------------------------------------
    print("\n[C] Noise power ~ sigma_e^2")
    rng = np.random.default_rng(2)
    n_rf = 4
    Fc = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(M, n_rf)))
    acc = 0.0; reps = 4000
    y0 = sense(H_t, Fc, S, 0.0, np.random.default_rng(99))      # clean reference
    for _ in range(reps):
        yn = sense(H_t, Fc, S, sigma_e2, rng)
        acc += np.mean(np.abs(yn - y0) ** 2)
    emp = acc / reps
    ok = abs(emp - sigma_e2) / sigma_e2 < 0.05
    all_pass &= ok
    print(f"   empirical noise var = {emp:.4e} (sigma_e^2={sigma_e2:.0e}), rel-err "
          f"{abs(emp-sigma_e2)/sigma_e2:.3f}  -> {'PASS' if ok else 'FAIL'}")

    # ---- D: a mixed (n_rf < M) read still sharpens the active ports ------------
    print("\n[D] Mixed read (n_rf=3 < M=5) reduces active-port uncertainty")
    bel = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
    covs0 = efe.active_covs(bel, S)
    tr0 = sum(np.real(np.trace(c)) for c in covs0)
    rng = np.random.default_rng(5)
    F = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(M, 3)))
    y = sense(H_t, F, S, sigma_e2, rng)
    bel.update_general(observation_matrix(F, S, N), y)
    covs1 = efe.active_covs(bel, S)
    tr1 = sum(np.real(np.trace(c)) for c in covs1)
    ok = tr1 < tr0
    all_pass &= ok
    print(f"   aggregate trace(Cov_S): {tr0:.3f} -> {tr1:.3f}  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 46)
    print(f"S2 STEP 2 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
