"""
S2 Step 3 verification -- the sensing info objective J_sense and its unconstrained bound.

Gate before S2-4 (the unit-modulus optimizer). Checks:

  A. ROTATION-INVARIANCE ANCHOR -- for a UNITARY F (n_rf = M), sensing y = F^H h + n is
     informationally equivalent to reading all M ports, so
        sense_info(F_unitary, Sigma_bar) == log2 det(I + sigma_e^{-2} Sigma_bar)
     independent of the unitary. (F = I_M is the per-port special case.)
  B. WATER-FILLING SELF-CONSISTENCY -- optimal_unconstrained returns (J, F_opt, p) with
     sense_info(F_opt, Sigma_bar) == J to machine precision.
  C. UPPER BOUND (I4) -- for the same total energy Ptot = M*n_rf, no random F and no
     gradient-ascent F exceeds the water-filling J; and a projected-gradient maximizer
     REACHES it (bound is tight -> it really is the optimum).
  D. MONOTONE IN n_rf (I2) + SATURATION -- J_opt(n_rf) is non-decreasing and saturates once
     n_rf reaches the numerical rank (spatial DoF) of Sigma_bar.

Run:  python sim_version4/verify_s2_step3.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402
from belief import KalmanBelief                # noqa: E402
import efe                                      # noqa: E402
from sensing import sense_info, optimal_unconstrained  # noqa: E402


def _make_cov_bar(seed=7):
    """A realistic aggregate active-port covariance Sigma_bar = sum_k Cov_k after a few slots."""
    rho, sigma_e2 = 0.9, 1e-2
    beta = np.array([1.0, 0.7, 1.3]); K = len(beta)
    sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=seed)
    S = (0, 2, 7, 12, 18, 24); M = len(S)
    bel = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
    # a couple of aged predicts so Sigma_bar is a nontrivial (low-rank) PSD matrix
    bel.predict(); bel.predict()
    covs = efe.active_covs(bel, S)
    cov_bar = np.sum(covs, axis=0)
    return 0.5 * (cov_bar + cov_bar.conj().T), M, sigma_e2


def _grad_ascent(cov_bar, n_rf, sigma_e2, Ptot, iters=600, restarts=5, rng=None):
    """Projected-gradient ascent maximising J_sense over F with ||F||_F^2 = Ptot.
    grad of log det(I + c F^H S F) wrt F* is  c S F (I + c F^H S F)^{-1}."""
    rng = np.random.default_rng(0) if rng is None else rng
    M = cov_bar.shape[0]; c = 1.0 / sigma_e2
    bestJ, bestF = -np.inf, None
    for _ in range(restarts):
        F = (rng.standard_normal((M, n_rf)) + 1j * rng.standard_normal((M, n_rf))) / np.sqrt(2)
        F *= np.sqrt(Ptot) / np.linalg.norm(F)
        eta = 0.5; Jprev = sense_info(F, cov_bar, sigma_e2)
        for _ in range(iters):
            G = np.eye(n_rf) + c * (F.conj().T @ cov_bar @ F)
            grad = c * cov_bar @ F @ np.linalg.inv(G)
            Fn = F + eta * grad
            Fn *= np.sqrt(Ptot) / np.linalg.norm(Fn)
            J = sense_info(Fn, cov_bar, sigma_e2)
            if J >= Jprev:
                F, Jprev = Fn, J; eta *= 1.1
            else:
                eta *= 0.5
            if eta < 1e-6:
                break
        if Jprev > bestJ:
            bestJ, bestF = Jprev, F
    return bestJ, bestF


def main():
    cov_bar, M, sigma_e2 = _make_cov_bar()
    rank = int(np.sum(np.linalg.eigvalsh(cov_bar) > 1e-9 * np.trace(cov_bar).real))
    print(f"Sigma_bar: M={M}, numerical rank ~ {rank}, sigma_e^2={sigma_e2}")
    all_pass = True
    inv = 1.0 / sigma_e2

    # ---- A: rotation-invariance anchor ------------------------------------
    print("\n[A] Unitary F (n_rf=M): sense_info == log2 det(I + Sigma_bar/sigma_e^2)")
    full = float(np.sum(np.log2(np.real(np.linalg.eigvalsh(np.eye(M) + inv * cov_bar)))))
    rng = np.random.default_rng(3)
    X = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    Q, _ = np.linalg.qr(X)                             # random unitary
    err_I = abs(sense_info(np.eye(M, dtype=complex), cov_bar, sigma_e2) - full)
    err_Q = abs(sense_info(Q, cov_bar, sigma_e2) - full)
    ok = err_I < 1e-9 and err_Q < 1e-9
    all_pass &= ok
    print(f"   full-read info = {full:.4f} bits | err(I_M)={err_I:.1e} err(random unitary)={err_Q:.1e}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- B: water-filling self-consistency --------------------------------
    print("\n[B] optimal_unconstrained self-consistency: sense_info(F_opt) == J_wf")
    worst = 0.0
    for n_rf in range(1, M + 1):
        J, F_opt, p = optimal_unconstrained(cov_bar, n_rf, sigma_e2)
        worst = max(worst, abs(sense_info(F_opt, cov_bar, sigma_e2) - J))
    ok = worst < 1e-9
    all_pass &= ok
    print(f"   worst |sense_info(F_opt) - J_wf| over n_rf=1..M = {worst:.1e}  -> {'PASS' if ok else 'FAIL'}")

    # ---- C: upper bound is valid AND tight --------------------------------
    print("\n[C] Bound valid (random & GA <= J_wf) and tight (GA reaches J_wf), Ptot=M*n_rf")
    ok = True
    for n_rf in [2, 3, 4]:
        Ptot = float(M * n_rf)
        Jwf, _, _ = optimal_unconstrained(cov_bar, n_rf, sigma_e2, Ptot=Ptot)
        rng = np.random.default_rng(10)
        Jrand = max(sense_info(
            (lambda Z: Z * np.sqrt(Ptot) / np.linalg.norm(Z))(
                (rng.standard_normal((M, n_rf)) + 1j * rng.standard_normal((M, n_rf)))),
            cov_bar, sigma_e2) for _ in range(200))
        Jga, _ = _grad_ascent(cov_bar, n_rf, sigma_e2, Ptot, rng=np.random.default_rng(1))
        valid = (Jrand <= Jwf + 1e-6) and (Jga <= Jwf + 1e-6)
        tight = Jga >= Jwf - 5e-3
        ok = ok and valid and tight
        print(f"   n_rf={n_rf}: J_wf={Jwf:.4f} | random_max={Jrand:.4f} | grad-ascent={Jga:.4f}"
              f"  {'ok' if valid and tight else 'BAD'}")
    all_pass &= ok
    print(f"   -> {'PASS' if ok else 'FAIL'}")

    # ---- D: monotone in n_rf + saturation ---------------------------------
    print("\n[D] J_opt(n_rf) non-decreasing and saturates at numerical rank")
    Js = [optimal_unconstrained(cov_bar, nr, sigma_e2)[0] for nr in range(1, M + 1)]
    mono = all(Js[i + 1] >= Js[i] - 1e-9 for i in range(len(Js) - 1))
    sat = abs(Js[min(rank, M) - 1] - Js[-1]) < 1e-6    # info at n_rf=rank ~ info at n_rf=M
    ok = mono and sat
    all_pass &= ok
    print("   J_opt vs n_rf: " + ", ".join(f"{j:.3f}" for j in Js))
    print(f"   monotone={mono}, saturates at n_rf={min(rank, M)} (== M value): {sat}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 46)
    print(f"S2 STEP 3 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
