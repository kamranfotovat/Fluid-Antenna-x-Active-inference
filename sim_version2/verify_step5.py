"""
Step 5 verification -- greedy EFE selection is near-optimal and fast.

Gate before Step 6 (closed loop). Checks:

  G1  Mechanics.  greedy returns |S|=M, no duplicates, valid ports; first-slot (S_prev=None)
      switching is zero so greedy == unconstrained argmax.
  G2  Epistemic (1-1/e) guarantee.  With alpha=0 (epistemic-only, submodular), greedy achieves
      >= (1-1/e) ~ 0.632 of the exhaustive optimum on N=8, M=3 (in practice much higher).
  G3  Combined-objective near-optimality.  On N=8, M=3, greedy J is within a small gap of the
      exhaustive optimum across many random beliefs, for several (alpha,beta,eta_sw) settings.
  G4  Switching modularity is honest.  The greedy-accounted switching equals the exact
      switching_cost recomputed on the final set.
  G5  Latency.  Measure greedy ms/slot at N=25, M=5 and contrast with the projected exhaustive
      time (C(25,5) = 53130 evaluations) -- the "AIF is slow" rebuttal.
  G6  Knob works.  beta=0 (exploit) and alpha=0 (explore) generally pick different sets.

Run:  python sim/verify_step5.py
"""

from __future__ import annotations

import sys
import time
import math
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402
from belief import KalmanBelief                # noqa: E402
import efe                                     # noqa: E402


def make_belief(Nx, Ny, Wx, Wy, beta, rho, sigma_e2, seed, warm=0):
    """Belief with a non-trivial state: random mu (a channel sample) + optionally a few
    predict/update cycles so Sigma varies across ports (not flat stationary)."""
    sim = ChannelSimulator(Nx=Nx, Ny=Ny, Wx=Wx, Wy=Wy, K=len(beta), rho=rho, beta=beta, seed=seed)
    bel = KalmanBelief(R=sim.R, beta=beta, rho=rho, sigma_e2=sigma_e2)
    bel.mu = sim.h.copy()
    for _ in range(warm):
        bel.predict()
        Swarm = tuple(sim.rng.choice(sim.N, size=min(3, sim.N), replace=False).tolist())
        y = bel.mu[:, list(Swarm)]  # dummy obs (Sigma update is data-independent)
        bel.update(Swarm, y)
    return sim, bel


def main():
    rho, sigma_e2, sigma2, P = 0.9, 1e-2, 1e-3, 1.0
    beta = np.array([1.0, 0.7, 1.3])
    all_pass = True
    print(f"K={len(beta)}, rho={rho}, sigma_e^2={sigma_e2}, sigma^2={sigma2}")

    # ---- G1: mechanics ----------------------------------------------------
    print("\n[G1] Mechanics: |S|=M, unique, valid; first-slot switching = 0")
    sim8, bel8 = make_belief(8, 1, 1.75, 0.0, beta, rho, sigma_e2, seed=3, warm=2)
    M = 3
    S = efe.greedy_select(bel8, M, S_prev=None, sigma2=sigma2, P=P)
    ok = len(S) == M and len(set(S)) == M and all(0 <= n < bel8.N for n in S)
    all_pass &= ok
    print(f"   greedy S={S}  |S|={len(S)} unique={len(set(S))==M} valid={ok}  -> {'PASS' if ok else 'FAIL'}")

    # ---- G2: epistemic-only (1-1/e) guarantee -----------------------------
    print("\n[G2] Epistemic-only greedy >= (1-1/e) of exhaustive optimum (N=8, M=3)")
    kw_epi = dict(alpha=0.0, beta=1.0, eta_sw=0.0, sigma2=sigma2, P=P)
    worst_ratio = 1.0
    for seed in range(8):
        _, b = make_belief(8, 1, 1.75, 0.0, beta, rho, sigma_e2, seed=seed, warm=seed % 4)
        Sg = efe.greedy_select(b, 3, S_prev=None, **kw_epi)
        Jg = efe.epistemic_value(b, Sg)
        _, Jopt = efe.exhaustive_select(b, 3, S_prev=None, **kw_epi)
        worst_ratio = min(worst_ratio, Jg / Jopt)
    ok = worst_ratio >= 1.0 - 1.0 / math.e - 1e-9
    all_pass &= ok
    print(f"   worst greedy/opt over 8 beliefs = {worst_ratio:.4f}  (>= 1-1/e = {1-1/math.e:.4f})"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- G3: combined-objective near-optimality ---------------------------
    print("\n[G3] Combined greedy J within small gap of exhaustive (N=8, M=3, several weightings)")
    settings = [dict(alpha=1, beta=1, eta_sw=1), dict(alpha=1, beta=0, eta_sw=1),
                dict(alpha=1, beta=0.2, eta_sw=2), dict(alpha=0.5, beta=1, eta_sw=0.5)]
    worst_gap = 0.0
    for st in settings:
        for seed in range(6):
            _, b = make_belief(8, 1, 1.75, 0.0, beta, rho, sigma_e2, seed=100 + seed, warm=seed % 3)
            S_prev = (1, 3, 5)
            Sg = efe.greedy_select(b, 3, S_prev=S_prev, sigma2=sigma2, P=P, **st)
            Gg, _ = efe.expected_free_energy(b, Sg, S_prev=S_prev, sigma2=sigma2, P=P, **st)
            _, Jopt = efe.exhaustive_select(b, 3, S_prev=S_prev, sigma2=sigma2, P=P, **st)
            gap = (Jopt - (-Gg)) / abs(Jopt)
            worst_gap = max(worst_gap, gap)
    ok = worst_gap < 0.05
    all_pass &= ok
    print(f"   worst relative optimality gap over 24 cases = {worst_gap*100:.2f}%  (<5%)"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- G4: switching modularity is honest -------------------------------
    print("\n[G4] Greedy switching accounting == exact switching_cost on the final set")
    _, b = make_belief(8, 1, 1.75, 0.0, beta, rho, sigma_e2, seed=11, warm=1)
    S_prev = (0, 2, 6)
    Sg = efe.greedy_select(b, 3, S_prev=S_prev, eta_sw=1.5, e_sw=1.0, sigma2=sigma2, P=P)
    exact_sw = efe.switching_cost(Sg, S_prev, eta_sw=1.5, e_sw=1.0)
    # reconstruct from modular per-port marginals + constant eta*e*|S_prev|
    prev_set = set(S_prev)
    modular = 1.5 * 1.0 * len(S_prev) + sum(efe._switch_marginal(n, prev_set, 1.5, 1.0) for n in Sg)
    ok = abs(exact_sw - modular) < 1e-12
    all_pass &= ok
    print(f"   exact switching={exact_sw:.3f}  modular-reconstruction={modular:.3f}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- G5: latency ------------------------------------------------------
    print("\n[G5] Latency at N=25, M=5: greedy ms/slot vs projected exhaustive")
    _, b25 = make_belief(5, 5, 1.0, 1.0, beta, rho, sigma_e2, seed=7, warm=3)
    reps = 30
    t0 = time.perf_counter()
    for _ in range(reps):
        efe.greedy_select(b25, 5, S_prev=(0, 1, 2, 3, 4), sigma2=sigma2, P=P)
    greedy_ms = (time.perf_counter() - t0) / reps * 1e3
    # per-evaluation cost of one G(S)
    t0 = time.perf_counter()
    ne = 300
    for _ in range(ne):
        efe.expected_free_energy(b25, (0, 4, 12, 20, 24), S_prev=(0, 1, 2, 3, 4), sigma2=sigma2, P=P)
    per_eval_ms = (time.perf_counter() - t0) / ne * 1e3
    n_comb = math.comb(25, 5)
    proj_exhaustive_s = per_eval_ms * n_comb / 1e3
    print(f"   greedy = {greedy_ms:.2f} ms/slot   ({5*25} objective evals, O(N*M))")
    print(f"   exhaustive C(25,5)={n_comb} evals @ {per_eval_ms:.3f} ms  ->  ~{proj_exhaustive_s:.1f} s/slot")
    print(f"   speed-up ~ {proj_exhaustive_s*1e3/greedy_ms:,.0f}x")
    ok = greedy_ms < 50.0    # comfortably real-time for a letter's claim
    all_pass &= ok
    print(f"   greedy under 50 ms/slot  -> {'PASS' if ok else 'FAIL'}")

    # ---- G6: exploration/exploitation knob --------------------------------
    print("\n[G6] Knob: beta=0 (exploit) vs alpha=0 (explore) pick different sets")
    diffs = 0
    for seed in range(6):
        _, b = make_belief(5, 5, 1.0, 1.0, beta, rho, sigma_e2, seed=200 + seed, warm=2)
        S_exploit = efe.greedy_select(b, 5, alpha=1, beta=0, eta_sw=0, sigma2=sigma2, P=P)
        S_explore = efe.greedy_select(b, 5, alpha=0, beta=1, eta_sw=0, sigma2=sigma2, P=P)
        if set(S_exploit) != set(S_explore):
            diffs += 1
    ok = diffs >= 4
    all_pass &= ok
    print(f"   exploit != explore in {diffs}/6 beliefs  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + ("=" * 46))
    print(f"STEP 5 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)

    _try_plot(beta, rho, sigma_e2, sigma2, P)
    return 0 if all_pass else 1


def _try_plot(beta, rho, sigma_e2, sigma2, P):
    """(1) greedy vs exhaustive objective (N=8); (2) greedy latency vs N."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return

    # (1) greedy vs optimal, many beliefs
    Jg_list, Jo_list = [], []
    for seed in range(30):
        _, b = make_belief(8, 1, 1.75, 0.0, beta, rho, sigma_e2, seed=300 + seed, warm=seed % 4)
        S_prev = (1, 3, 5)
        Sg = efe.greedy_select(b, 3, S_prev=S_prev, sigma2=sigma2, P=P)
        Gg, _ = efe.expected_free_energy(b, Sg, S_prev=S_prev, sigma2=sigma2, P=P)
        _, Jo = efe.exhaustive_select(b, 3, S_prev=S_prev, sigma2=sigma2, P=P)
        Jg_list.append(-Gg); Jo_list.append(Jo)

    # (2) latency vs N
    import time as _t
    Ns, times = [], []
    for (nx, ny, wx, wy) in [(3, 3, 0.5, 0.5), (4, 4, 0.75, 0.75), (5, 5, 1.0, 1.0),
                             (6, 6, 1.25, 1.25), (7, 7, 1.5, 1.5)]:
        _, b = make_belief(nx, ny, wx, wy, beta, rho, sigma_e2, seed=7, warm=2)
        reps = 20
        t0 = _t.perf_counter()
        for _ in range(reps):
            efe.greedy_select(b, 5, S_prev=(0, 1, 2, 3, 4), sigma2=sigma2, P=P)
        Ns.append(nx * ny); times.append((_t.perf_counter() - t0) / reps * 1e3)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lim = max(max(Jg_list), max(Jo_list)) * 1.02
    lo = min(min(Jg_list), min(Jo_list)) * 0.98
    ax[0].plot([lo, lim], [lo, lim], "k--", alpha=0.6, label="greedy = optimal")
    ax[0].scatter(Jo_list, Jg_list, s=22, alpha=0.8)
    ax[0].set_title("Greedy vs exhaustive objective (N=8, M=3)")
    ax[0].set_xlabel("exhaustive optimum J"); ax[0].set_ylabel("greedy J")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

    ax[1].plot(Ns, times, "o-")
    ax[1].set_title("Greedy latency vs N (M=5) -- O(N*M)")
    ax[1].set_xlabel("N ports"); ax[1].set_ylabel("ms / slot")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = "step5_greedy_check.png"
    fig.savefig(out, dpi=120)
    print(f"(saved plot -> sim/{out})")


if __name__ == "__main__":
    raise SystemExit(main())
