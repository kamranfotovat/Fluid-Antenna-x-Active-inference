"""
Step 4 verification -- the three EFE terms are correct, in isolation.

Gate before Step 5 (greedy assembly). Checks:

  E1  Pragmatic == genie rate when the belief is confident.  With Sigma -> 0 the robust-MMSE
      pragmatic value collapses exactly onto the full-CSI MMSE sum-rate evaluated on mu.
  E1b Pragmatic is conservative under uncertainty.  Same mu, larger Sigma -> lower pragmatic.
  E2  Epistemic == belief-entropy reduction (bits).  log2 det(I + Cov/sigma_e^2) equals
      (log det Sigma_prior - log det Sigma_post)/ln2 from an actual Kalman update.
  E3  Epistemic is monotone in S.  Adding a port never lowers information.
  E4  Epistemic is submodular in S.  Marginal gain of a port shrinks as the set grows.
  E5  Information sharing via R.  A port correlated with the observed set gives less marginal
      info than an uncorrelated far port (diminishing returns come from R, not distance alone).
  E6  Switching cost = symmetric-difference count.
  All rate/info terms are in BITS.

Run:  python sim/verify_step4.py
"""

from __future__ import annotations

import sys
import itertools
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402
from belief import KalmanBelief                # noqa: E402
from precoding import mmse_precoder, sinr_and_rates  # noqa: E402
import efe                                     # noqa: E402


S = (0, 4, 12, 20, 24)


def main():
    rho, sigma_e2 = 0.9, 1e-2
    sigma2, P = 1e-3, 1.0
    beta = np.array([1.0, 0.7, 1.3])
    K = len(beta)
    sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=beta, seed=7)
    N = sim.N
    rng = np.random.default_rng(0)
    all_pass = True
    print(f"N={N}, K={K}, rho={rho}, sigma_e^2={sigma_e2}, sigma^2={sigma2}")

    # ---- E1: pragmatic == genie MMSE rate when belief is confident --------
    print("\n[E1] Pragmatic == full-CSI MMSE sum-rate when Sigma -> 0")
    bel = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
    bel.mu = sim.h.copy()
    for k in range(K):
        bel.Sigma[k] = np.zeros((N, N), dtype=complex)     # perfectly confident
    prag = efe.pragmatic_value(bel, S, sigma2, P)
    Hhat = efe.active_mean(bel, S)
    W = mmse_precoder(Hhat, P=P, sigma2=sigma2)             # standard MMSE on mu (=true here)
    genie = float(sinr_and_rates(Hhat, W, sigma2)[1].sum())
    err = abs(prag - genie) / genie
    ok = err < 1e-9
    all_pass &= ok
    print(f"   pragmatic={prag:.6f}  genie-MMSE={genie:.6f}  rel-err={err:.2e}  -> {'PASS' if ok else 'FAIL'}")

    # ---- E1b: pragmatic conservative under uncertainty --------------------
    print("\n[E1b] Pragmatic decreases as belief uncertainty grows (same mu)")
    vals = []
    for scale in [0.0, 0.05, 0.2, 0.5]:
        b = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
        b.mu = sim.h.copy()
        for k in range(K):
            b.Sigma[k] = scale * (beta[k] * sim.R).astype(complex)
        vals.append(efe.pragmatic_value(b, S, sigma2, P))
    monotone = all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))
    all_pass &= monotone
    print(f"   pragmatic vs Sigma-scale {[0.0,0.05,0.2,0.5]}: {[round(v,3) for v in vals]}"
          f"  -> {'PASS' if monotone else 'FAIL'}")

    # ---- E2: epistemic == entropy reduction (bits) ------------------------
    print("\n[E2] Epistemic == (log det Sigma_prior - log det Sigma_post)/ln2   [bits]")
    b = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2, reg=0.1)  # reg -> invertible
    max_err = 0.0
    for Stest in [S, (1, 7, 13), (0, 1, 2, 3)]:
        epis_direct = efe.epistemic_value(b, Stest, return_per_user=True)[1]
        # actual Kalman update reduction per user
        import copy
        bpost = copy.deepcopy(b)
        bpost.update(Stest, np.zeros((K, len(Stest))))
        for k in range(K):
            _, ld_prior = np.linalg.slogdet(b.Sigma[k])
            _, ld_post = np.linalg.slogdet(bpost.Sigma[k])
            red_bits = np.real(ld_prior - ld_post) / np.log(2.0)
            max_err = max(max_err, abs(red_bits - epis_direct[k]))
    ok = max_err < 1e-8
    all_pass &= ok
    print(f"   max |I_k - entropy-drop|  over 3 sets, 3 users = {max_err:.2e}  -> {'PASS' if ok else 'FAIL'}")

    # ---- E3 & E4: monotone + submodular epistemic -------------------------
    print("\n[E3/E4] Epistemic monotone & submodular  (random A subset B, port n not in B)")
    b = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)  # stationary (singular ok)
    ports = list(range(N))
    mono_ok, sub_ok = True, True
    trials = 400
    for _ in range(trials):
        B = sorted(rng.choice(ports, size=rng.integers(2, 7), replace=False).tolist())
        A = sorted(rng.choice(B, size=rng.integers(1, len(B)), replace=False).tolist())
        rest = [n for n in ports if n not in B]
        n = int(rng.choice(rest))
        mA, mB = efe.epistemic_value(b, A), efe.epistemic_value(b, B)
        mAn, mBn = efe.epistemic_value(b, A + [n]), efe.epistemic_value(b, B + [n])
        if mAn < mA - 1e-9 or mBn < mB - 1e-9:
            mono_ok = False
        if (mAn - mA) < (mBn - mB) - 1e-9:               # marginal(A) >= marginal(B)
            sub_ok = False
    all_pass &= mono_ok and sub_ok
    print(f"   monotone over {trials} trials -> {'PASS' if mono_ok else 'FAIL'}")
    print(f"   submodular (marginal shrinks with set) over {trials} trials -> {'PASS' if sub_ok else 'FAIL'}")

    # ---- E5: information sharing via R -------------------------------------
    print("\n[E5] Info sharing via R: correlated neighbour's marginal info drops more than a far port's")
    n0 = 12
    corr = np.abs(sim.R[n0])
    order = np.argsort(corr)
    far, neigh = int(order[0]), int(order[-2])
    g_neigh_alone = efe.epistemic_value(b, (neigh,))
    g_neigh_given0 = efe.epistemic_value(b, (n0, neigh)) - efe.epistemic_value(b, (n0,))
    g_far_alone = efe.epistemic_value(b, (far,))
    g_far_given0 = efe.epistemic_value(b, (n0, far)) - efe.epistemic_value(b, (n0,))
    drop_neigh = g_neigh_alone - g_neigh_given0
    drop_far = g_far_alone - g_far_given0
    print(f"   neighbour (|R|={corr[neigh]:.2f}): marginal {g_neigh_alone:.3f} -> {g_neigh_given0:.3f} "
          f"(drop {drop_neigh:.3f})")
    print(f"   far       (|R|={corr[far]:.2f}): marginal {g_far_alone:.3f} -> {g_far_given0:.3f} "
          f"(drop {drop_far:.3f})")
    ok = drop_neigh > drop_far and drop_far >= -1e-9
    all_pass &= ok
    print(f"   neighbour diminished more than far  -> {'PASS' if ok else 'FAIL'}")

    # ---- E6: switching cost -----------------------------------------------
    print("\n[E6] Switching cost = symmetric-difference count")
    c = efe.switching_cost((0, 1, 2, 3, 4), (2, 3, 4, 5, 6), eta_sw=1.0, e_sw=1.0)
    c0 = efe.switching_cost((0, 1, 2), None)
    ok = (c == 4.0) and (c0 == 0.0)
    all_pass &= ok
    print(f"   |{{0,1,2,3,4}} XOR {{2,3,4,5,6}}| = {c} (expect 4); first-slot (S_prev=None) = {c0}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- units note (use a representative stationary belief) --------------
    b_ex = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
    b_ex.mu = sim.h.copy()
    G, terms = efe.expected_free_energy(b_ex, S, S_prev=(1, 2, 3, 4, 5), sigma2=sigma2, P=P)
    print(f"\n   Example G(S) = {G:.3f}  terms(bits) = "
          f"{{prag: {terms['pragmatic']:.3f}, epis: {terms['epistemic']:.3f}, switch: {terms['switching']:.1f}}}")

    print("\n" + ("=" * 46))
    print(f"STEP 4 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)

    _try_plot(sim, beta, rho, sigma_e2, sigma2, P)
    return 0 if all_pass else 1


def _try_plot(sim, beta, rho, sigma_e2, sigma2, P):
    """Visualise (1) epistemic diminishing returns as |S| grows, (2) pragmatic vs uncertainty."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return

    K, N = len(beta), sim.N
    b = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)

    # (1) greedily grow S by max marginal epistemic; record cumulative + marginal
    chosen, cum, marg = [], [], []
    remaining = set(range(N))
    prev = 0.0
    for _ in range(12):
        best_n, best_v = None, -np.inf
        for n in remaining:
            v = efe.epistemic_value(b, tuple(chosen + [n]))
            if v > best_v:
                best_v, best_n = v, n
        chosen.append(best_n); remaining.remove(best_n)
        cum.append(best_v); marg.append(best_v - prev); prev = best_v

    # (2) pragmatic vs uncertainty scale, at a fixed mu
    b2 = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
    b2.mu = sim.h.copy()
    scales = np.linspace(0.0, 0.8, 12)
    prag = []
    for s in scales:
        for k in range(K):
            b2.Sigma[k] = s * (beta[k] * sim.R).astype(complex)
        prag.append(efe.pragmatic_value(b2, S, sigma2, P))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(1, len(cum) + 1), cum, "o-", label="cumulative info (bits)")
    ax[0].plot(range(1, len(marg) + 1), marg, "s--", label="marginal gain (bits)")
    ax[0].set_title("Epistemic value: diminishing returns (submodular)")
    ax[0].set_xlabel("|S| (ports added greedily)"); ax[0].set_ylabel("bits")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

    ax[1].plot(scales, prag, "o-")
    ax[1].set_title("Pragmatic value: conservative under uncertainty")
    ax[1].set_xlabel("belief uncertainty scale (Sigma = s * beta R)")
    ax[1].set_ylabel("expected sum-rate (bits/s/Hz)")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = "step4_efe_terms_check.png"
    fig.savefig(out, dpi=120)
    print(f"(saved plot -> sim/{out})")


if __name__ == "__main__":
    raise SystemExit(main())
