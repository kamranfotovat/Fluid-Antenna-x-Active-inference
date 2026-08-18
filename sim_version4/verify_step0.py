"""
Step 0 verification -- statistical checks on the channel generator.

Gate before Step 1. Checks (with tolerances):
  1. Spatial covariance   (1/T) sum_t h_t h_t^H  ~=  beta_k * R          (Eq. 2)
  2. Temporal autocorr    E[h(t) conj(h(t+L))]/E|h|^2  ~=  rho^L          (Eq. 3)
  3. Stationary power      diag(cov) ~= beta_k  (since diag(R) = 1)
  4. Circularity           pseudo-cov (1/T) sum_t h_t h_t^T ~= 0          (proper CN)
  5. Gaussianity           real/imag parts pass a normality sanity check

Run:  python sim/verify_step0.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator  # noqa: E402  (run from sim/ dir)


def _fro_rel(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(B)


def main():
    T = 200_000
    rho = 0.9
    beta = np.array([1.0, 0.7, 1.3])  # unequal powers to test per-user scaling
    K = len(beta)

    sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=1)
    N = sim.N
    print(f"N={N} ports (5x5), K={K}, rho={rho}, beta={beta.tolist()}")
    print(f"Generating T={T} slots ...")
    H = sim.generate(T)  # (T, K, N)

    all_pass = True
    tol_cov, tol_ac, tol_circ = 0.03, 0.02, 0.02

    # ---- 1 & 3: spatial covariance and stationary power (per user) --------
    print("\n[1/3] Spatial covariance  cov ~= beta_k * R   and   diag(cov) ~= beta_k")
    for k in range(K):
        hk = H[:, k, :]                      # (T, N)
        cov = (hk.conj().T @ hk) / T         # (N, N) empirical, E[h h^H]
        target = beta[k] * sim.R
        err = _fro_rel(cov, target)
        diag_err = np.max(np.abs(np.real(np.diag(cov)) - beta[k])) / beta[k]
        ok = err < tol_cov and diag_err < tol_cov
        all_pass &= ok
        print(f"   user {k}: cov rel-Fro err = {err:6.4f} | diag power err = {diag_err:6.4f}"
              f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- 2: temporal autocorrelation vs rho^L -----------------------------
    print("\n[2] Temporal autocorrelation  r(L) ~= rho^L   (averaged over users & ports)")
    lags = [1, 2, 3, 5, 10, 20]
    max_ac_err = 0.0
    for L in lags:
        num = np.mean(H[:-L] * np.conj(H[L:]), axis=(0, 1, 2))   # E[h(t) conj(h(t+L))]
        den = np.mean(np.abs(H) ** 2)
        r_emp = np.real(num) / den
        r_the = rho ** L
        e = abs(r_emp - r_the)
        max_ac_err = max(max_ac_err, e)
        print(f"   lag {L:2d}: empirical = {r_emp:6.4f} | rho^L = {r_the:6.4f} | err = {e:6.4f}")
    ok = max_ac_err < tol_ac
    all_pass &= ok
    print(f"   max autocorr err = {max_ac_err:6.4f}  -> {'PASS' if ok else 'FAIL'}")

    # ---- 4: circularity (pseudo-covariance ~ 0) ---------------------------
    print("\n[4] Circularity  pseudo-cov (E[h h^T]) ~= 0")
    hk = H[:, 0, :]
    pcov = (hk.T @ hk) / T                    # E[h h^T], should be ~0
    cov = (hk.conj().T @ hk) / T
    circ = np.linalg.norm(pcov) / np.linalg.norm(cov)
    ok = circ < tol_circ
    all_pass &= ok
    print(f"   ||pseudo-cov|| / ||cov|| = {circ:6.4f}  -> {'PASS' if ok else 'FAIL'}")

    # ---- 5: Gaussianity sanity (real & imag parts) ------------------------
    print("\n[5] Gaussianity sanity  (real/imag of port 0, user 0)")
    x = np.real(H[:, 0, 0])
    # excess kurtosis of a Gaussian is ~0; sample std of kurtosis ~ sqrt(24/T)
    kurt = np.mean(((x - x.mean()) / x.std()) ** 4) - 3.0
    tol_k = 8.0 * np.sqrt(24.0 / T)
    ok = abs(kurt) < tol_k
    all_pass &= ok
    print(f"   excess kurtosis = {kurt:+.4f} (|.| < {tol_k:.4f})  -> {'PASS' if ok else 'FAIL'}")

    # ---- report on the correlation matrix itself --------------------------
    eig = np.linalg.eigvalsh(sim.R)
    print(f"\nR: min eig = {eig.min():.3e}, max eig = {eig.max():.3f}, "
          f"cond = {eig.max() / max(eig.min(), 1e-15):.1f} "
          f"(off-diag strength: mean|R_ij, i!=j| = {np.mean(np.abs(sim.R - np.eye(N))):.3f})")

    print("\n" + ("=" * 46))
    print(f"STEP 0 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)

    _try_plot(sim, lags, rho)
    return 0 if all_pass else 1


def _try_plot(sim, lags, rho):
    """Optional: save R heatmap + autocorr curve for a visual sanity check."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return

    Ls = np.arange(0, 25)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im = ax[0].imshow(sim.R, cmap="viridis")
    ax[0].set_title("Spatial correlation R (Jakes)")
    ax[0].set_xlabel("port j"); ax[0].set_ylabel("port i")
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    ax[1].plot(Ls, rho ** Ls, "o-", label="rho^L (theory)")
    ax[1].set_title("Temporal autocorrelation")
    ax[1].set_xlabel("lag L"); ax[1].set_ylabel("correlation")
    ax[1].grid(True, alpha=0.3); ax[1].legend()

    fig.tight_layout()
    out = "step0_channel_check.png"
    fig.savefig(out, dpi=120)
    print(f"(saved plot -> sim/{out})")


if __name__ == "__main__":
    raise SystemExit(main())
