"""
Step 3 verification -- the Kalman belief is correct and calibrated.

Gate before Step 4 (EFE terms). This is the CRITICAL calibration gate: if the belief
covariance Sigma does not match the true error, every downstream EFE term (which reads
Sigma) is meaningless.

Checks
------
  A. Predict step = CSI-aging law.  Predict-only for t slots reproduces the closed form
     Sigma_t = rho^{2t} Sigma_0 + (1 - rho^{2t}) beta_k R  to machine precision.
  B. Update sharpens observed ports.  One update from the stationary prior drives each
     observed port's variance below sigma_e^2 (and far below the prior power beta_k).
  C. Steady-state calibration (Monte Carlo, THE gate).  With a fixed active set, the filter
     covariance Sigma equals the empirical error covariance Cov(h_true - mu).
  D. Neighbour information sharing.  Observing one port lowers a strongly-correlated
     neighbour's variance more than a weakly-correlated far port's (info flows via R) --
     this is what makes partial observation pay off, and powers the epistemic term later.

Run:  python sim/verify_step3.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402  (run from sim/ dir)
from belief import KalmanBelief                # noqa: E402


def _fro_rel(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(B)


# fixed active set for the whole test: 4 corners + centre of the 5x5 grid
S = (0, 4, 12, 20, 24)


def main():
    rho = 0.9
    sigma_e2 = 1e-2
    beta = np.array([1.0, 0.7, 1.3])
    K = len(beta)

    sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=7)
    N = sim.N
    bel = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)

    print(f"N={N} ports (5x5), K={K}, rho={rho}, sigma_e^2={sigma_e2}, active set S={S} (M={len(S)})")
    all_pass = True

    # ---- A: predict step reproduces the aging law -------------------------
    print("\n[A] Predict step = CSI-aging law   Sigma_t = rho^2t Sigma_0 + (1-rho^2t) beta_k R")
    bel.reset()
    # perturb Sigma to a non-stationary start (a few updates), then predict-only
    bel.update(S, np.zeros((K, len(S))))          # Sigma update is data-independent
    Sigma0 = [bel.Sigma[k].copy() for k in range(K)]
    r2 = rho ** 2
    max_err = 0.0
    for t in [1, 3, 10, 50]:
        b2 = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
        for k in range(K):
            b2.Sigma[k] = Sigma0[k].copy()
        for _ in range(t):
            b2.predict()
        for k in range(K):
            closed = (r2 ** t) * Sigma0[k] + (1 - r2 ** t) * (beta[k] * sim.R)
            max_err = max(max_err, _fro_rel(b2.Sigma[k], closed))
    ok = max_err < 1e-10
    all_pass &= ok
    print(f"   max rel-Fro vs closed form (t in 1,3,10,50) = {max_err:.2e}  -> {'PASS' if ok else 'FAIL'}")

    # ---- B: update sharpens observed ports below sigma_e^2 ------------------
    print("\n[B] One update from the prior:  observed-port variance < sigma_e^2 and << beta_k")
    bel.reset()
    v_prior = bel.port_variances().copy()
    bel.update(S, np.zeros((K, len(S))))
    v_post = bel.port_variances()
    worst_obs = 0.0
    for k in range(K):
        obs_var = v_post[k, list(S)]
        worst_obs = max(worst_obs, obs_var.max() / sigma_e2)
        print(f"   user {k}: prior var@obs ~ {v_prior[k, list(S)].mean():.3f} (=beta) -> "
              f"post var@obs max = {obs_var.max():.4f}  (sigma_e^2={sigma_e2})")
    ok = worst_obs < 1.0                            # every observed port below sigma_e^2
    all_pass &= ok
    print(f"   max(observed post-var)/sigma_e^2 = {worst_obs:.3f} (<1)  -> {'PASS' if ok else 'FAIL'}")

    # ---- C: steady-state calibration (Monte Carlo) ------------------------
    print("\n[C] Steady-state calibration (MC):  Sigma_k  ~=  Cov(h_true - mu)   [THE gate]")
    T, MC = 40, 4000
    err_acc = [np.zeros((N, N), complex) for _ in range(K)]
    for m in range(MC):
        sim_m = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=1000 + m)
        b = KalmanBelief(R=sim_m.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
        h = sim_m.h                                 # slot 0 (stationary draw)
        for t in range(T):
            if t > 0:
                h = sim_m.step()
                b.predict()
            y = h[:, list(S)] + np.sqrt(sigma_e2 / 2) * (
                sim_m.rng.standard_normal((K, len(S))) + 1j * sim_m.rng.standard_normal((K, len(S))))
            b.update(S, y)
        e = h - b.mu                                # (K, N) final-slot error
        for k in range(K):
            err_acc[k] += np.outer(e[k], e[k].conj())
    max_cov_err, max_diag_err = 0.0, 0.0
    for k in range(K):
        emp = err_acc[k] / MC
        rel = _fro_rel(emp, b.Sigma[k])             # b.Sigma is the (data-independent) steady state
        diag_rel = np.max(np.abs(np.real(np.diag(emp)) - np.real(np.diag(b.Sigma[k])))) / \
            np.real(np.diag(b.Sigma[k])).max()
        max_cov_err = max(max_cov_err, rel)
        max_diag_err = max(max_diag_err, diag_rel)
        print(f"   user {k}: Cov vs Sigma rel-Fro = {rel:6.4f} | diag max rel-err = {diag_rel:6.4f}")
    ok = max_cov_err < 0.05 and max_diag_err < 0.05
    all_pass &= ok
    print(f"   worst cov rel-Fro = {max_cov_err:.4f}, worst diag rel-err = {max_diag_err:.4f} (<0.05)"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- D: neighbour information sharing ----------------------------------
    print("\n[D] Neighbour info sharing:  observing one port helps a correlated neighbour > a far port")
    n0 = 12                                         # centre port
    corr = np.abs(sim.R[n0])
    order = np.argsort(corr)                        # ascending correlation to n0
    far = int(order[0])                             # least correlated
    neigh = int(order[-2])                          # most correlated (excl. n0 itself = order[-1])
    bel.reset()
    v0 = bel.port_variances()[0].copy()
    bel.update((n0,), np.zeros((K, 1)))
    v1 = bel.port_variances()[0]
    drop_obs = v0[n0] - v1[n0]
    drop_neigh = v0[neigh] - v1[neigh]
    drop_far = v0[far] - v1[far]
    print(f"   observe only port {n0}:  |R[{n0},{neigh}]|={corr[neigh]:.2f} (neighbour), "
          f"|R[{n0},{far}]|={corr[far]:.2f} (far)")
    print(f"   variance drop:  observed={drop_obs:.3f} > neighbour={drop_neigh:.3f} > far={drop_far:.3f}")
    ok = drop_obs > drop_neigh > drop_far and drop_far >= -1e-9
    all_pass &= ok
    print(f"   ordering observed > neighbour > far, all >= 0  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + ("=" * 46))
    print(f"STEP 3 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)

    _try_plot(sim, beta, rho, sigma_e2)
    return 0 if all_pass else 1


def _try_plot(sim, beta, rho, sigma_e2):
    """Visualise (1) sharpen-then-age of port variances, (2) calibration scatter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return

    K, N = len(beta), sim.N
    # (1) variance-over-time: observe S for the first half, then stop (predict-only) -> aging.
    T = 60
    T_obs = 30
    bel = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
    track = {"port 0 (in S)": 0, "port 1 (neighbour of S)": 1,
             "port 7 (interior, off S)": 7, "port 12 (in S)": 12}
    hist = {name: [] for name in track}
    for t in range(T):
        if t > 0:
            bel.predict()
        if t < T_obs:
            bel.update(S, np.zeros((K, len(S))))     # Sigma is data-independent
        v = bel.port_variances()[0]                  # user 0
        for name, idx in track.items():
            hist[name].append(v[idx])

    # (2) calibration: quick MC of diag empirical var vs predicted, at steady state.
    T2, MC = 40, 1500
    diag_emp = np.zeros(N)
    bfin = None
    for m in range(MC):
        sm = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=5000 + m)
        b = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
        h = sm.h
        for t in range(T2):
            if t > 0:
                h = sm.step()
                b.predict()
            y = h[:, list(S)] + np.sqrt(sigma_e2 / 2) * (
                sm.rng.standard_normal((K, len(S))) + 1j * sm.rng.standard_normal((K, len(S))))
            b.update(S, y)
        diag_emp += np.abs(h[0] - b.mu[0]) ** 2
        bfin = b
    diag_emp /= MC
    diag_pred = np.real(np.diag(bfin.Sigma[0]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, series in hist.items():
        ax[0].plot(range(T), series, marker=".", ms=3, label=name)
    ax[0].axvline(T_obs - 0.5, color="k", ls="--", alpha=0.5)
    ax[0].axhline(sigma_e2, color="grey", ls=":", alpha=0.7)
    ax[0].text(T_obs + 0.5, beta[0] * 0.9, "observation OFF\n(CSI aging)", fontsize=8, va="top")
    ax[0].set_title("Belief variance: sharpen then age (user 0)")
    ax[0].set_xlabel("slot t"); ax[0].set_ylabel("posterior variance diag(Sigma)")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

    lim = max(diag_emp.max(), diag_pred.max()) * 1.05
    ax[1].plot([0, lim], [0, lim], "k--", alpha=0.6, label="y = x")
    ax[1].scatter(diag_pred, diag_emp, s=18, alpha=0.8)
    ax[1].set_title("Calibration: predicted vs empirical variance")
    ax[1].set_xlabel("filter diag(Sigma)  (predicted)")
    ax[1].set_ylabel("empirical Var(h - mu)")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = "step3_belief_check.png"
    fig.savefig(out, dpi=120)
    print(f"(saved plot -> sim/{out})")


if __name__ == "__main__":
    raise SystemExit(main())
