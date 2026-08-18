"""
S2 Step 1 verification -- the GENERALIZED (arbitrary-A) Kalman update is correct & calibrated.

Gate before S2-2 (sensing through the analog network). S2 replaces the per-port observation
y = P_S h + n with a mixed/compressed observation y = A h + n, A = F_RF^H P_S in C^{m x N}.
The Kalman engine must be UNCHANGED except for swapping the observation matrix. This gate proves:

  A. REGRESSION -- `bel.update(S, y)` (now delegating to update_general with A = P_S) reproduces
     the original selection-matrix Kalman recursion bit-for-bit (<= 1e-10). Guards the S1 path.
  B. I1 ANCHOR  -- update_general(P_S, y) is identical to update(S, y). The master reduction:
     A = P_S must behave exactly like per-port sensing.
  C. CALIBRATION (the gate) -- for a RANDOM complex A (m < N mixed measurements), the filter
     covariance Sigma equals the empirical error covariance Cov(h_true - mu) over Monte Carlo.
     If this holds for a general A, the S2 belief is trustworthy and every downstream EFE term is.
  D. I5 -- Sigma stays Hermitian PSD after a general-A update.

Run:  python sim_version4/verify_s2_step1.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402
from belief import KalmanBelief, selection_matrix   # noqa: E402


def _fro_rel(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(B)


def _ref_update_selection(Sigma, mu, S, y, sigma_e2, N):
    """The ORIGINAL S1 selection-matrix Kalman update (inlined reference for the regression gate).
    Operates on lists Sigma[k], mu[k]; returns new (Sigma, mu)."""
    P = selection_matrix(S, N)                         # (M, N) real
    M = P.shape[0]
    I_M, I_N = np.eye(M), np.eye(N)
    K = len(mu)
    Sig_out, mu_out = [], []
    for k in range(K):
        Sig = Sigma[k]
        SPt = Sig @ P.T                                # N x M
        Scov = P @ SPt + sigma_e2 * I_M                # M x M
        Kg = np.linalg.solve(Scov.T, SPt.T).T          # N x M
        innov = y[k] - P @ mu[k]
        mu_out.append(mu[k] + Kg @ innov)
        A = I_N - Kg @ P
        Sig = A @ Sig @ A.conj().T + sigma_e2 * (Kg @ Kg.conj().T)
        Sig_out.append(0.5 * (Sig + Sig.conj().T))
    return Sig_out, mu_out


def main():
    rho, sigma_e2 = 0.9, 1e-2
    beta = np.array([1.0, 0.7, 1.3]); K = len(beta)
    S = (0, 4, 12, 20, 24)
    sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=7)
    N = sim.N
    rng = np.random.default_rng(3)
    print(f"N={N} (5x5), K={K}, rho={rho}, sigma_e^2={sigma_e2}")
    all_pass = True

    # ---- A: regression vs the original selection-matrix recursion ----------
    print("\n[A] Regression: update(S,y) == original P_S Kalman recursion (bit-for-bit)")
    bel = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
    bel.update(S, np.zeros((K, len(S))))               # perturb off the stationary prior
    bel.predict()
    Sig0 = [bel.Sigma[k].copy() for k in range(K)]
    mu0 = [bel.mu[k].copy() for k in range(K)]
    y = (rng.standard_normal((K, len(S))) + 1j * rng.standard_normal((K, len(S)))) / np.sqrt(2)
    ref_Sig, ref_mu = _ref_update_selection(Sig0, mu0, S, y, sigma_e2, N)
    bel.update(S, y)
    errA = max(max(_fro_rel(bel.Sigma[k], ref_Sig[k]) for k in range(K)),
               max(np.linalg.norm(bel.mu[k] - ref_mu[k]) / (np.linalg.norm(ref_mu[k]) + 1e-30)
                   for k in range(K)))
    ok = errA < 1e-10; all_pass &= ok
    print(f"   max rel-err (Sigma & mu) = {errA:.2e} (<1e-10)  -> {'PASS' if ok else 'FAIL'}")

    # ---- B: I1 anchor -- update_general(P_S) == update(S) ------------------
    print("\n[B] I1 anchor: update_general(A=P_S) identical to update(S)")
    b1 = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2); b1.predict()
    b2 = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2); b2.predict()
    P = selection_matrix(S, N).astype(complex)
    b1.update(S, y)
    b2.update_general(P, y)
    errB = max(_fro_rel(b1.Sigma[k], b2.Sigma[k]) for k in range(K))
    ok = errB < 1e-12; all_pass &= ok
    print(f"   max rel-Fro(Sigma) = {errB:.2e} (<1e-12)  -> {'PASS' if ok else 'FAIL'}")

    # ---- C: calibration under a RANDOM complex A (the gate) ---------------
    m = 4                                               # mixed measurements, m < |S| < N
    A = (rng.standard_normal((m, N)) + 1j * rng.standard_normal((m, N))) / np.sqrt(2)
    print(f"\n[C] Calibration under general A (m={m} mixed reads of N={N}):  Sigma ~= Cov(h - mu)  [gate]")
    T, MC = 40, 3000
    err_acc = [np.zeros((N, N), complex) for _ in range(K)]
    bfin = None
    for mc in range(MC):
        sm = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=10_000 + mc)
        b = KalmanBelief(R=sm.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
        h = sm.h
        for t in range(T):
            if t > 0:
                h = sm.step(); b.predict()
            noise = np.sqrt(sigma_e2 / 2) * (sm.rng.standard_normal((K, m))
                                             + 1j * sm.rng.standard_normal((K, m)))
            yk = h @ A.T + noise                       # (K, m): y[k] = A h_k + noise
            b.update_general(A, yk)
        e = h - b.mu
        for k in range(K):
            err_acc[k] += np.outer(e[k], e[k].conj())
        bfin = b
    max_cov_err = 0.0
    for k in range(K):
        emp = err_acc[k] / MC
        rel = _fro_rel(emp, bfin.Sigma[k])
        max_cov_err = max(max_cov_err, rel)
        print(f"   user {k}: Cov vs Sigma rel-Fro = {rel:6.4f}")
    ok = max_cov_err < 0.05; all_pass &= ok
    print(f"   worst cov rel-Fro = {max_cov_err:.4f} (<0.05)  -> {'PASS' if ok else 'FAIL'}")

    # ---- D: I5 -- Hermitian PSD after a general-A update -------------------
    print("\n[D] I5: Sigma Hermitian PSD after general-A update")
    herm = max(np.max(np.abs(bfin.Sigma[k] - bfin.Sigma[k].conj().T)) for k in range(K))
    mineig = min(np.min(np.linalg.eigvalsh(bfin.Sigma[k])) for k in range(K))
    ok = herm < 1e-9 and mineig > -1e-9; all_pass &= ok
    print(f"   max|Sigma-Sigma^H| = {herm:.2e}, min eig = {mineig:.2e}  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 46)
    print(f"S2 STEP 1 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
