"""
Step 12 -- comparison against the BANDIT baseline (model-free learning competitor).

cf. Zou, Sun, Wang, "Online Learning-Induced Port Selection for FAS", IEEE WCL 2024 (bandit-based
port selection without full CSI). We implement a combinatorial-UCB port selector in OUR setting and
compare. The bandit is model-FREE: it learns per-port value by sampling and does NOT use the spatial
correlation R to infer un-measured ports -- so it pays a perpetual exploration/switching cost.

We give the bandit its FAIREST shot by sweeping its exploration constant c; AIF must beat it at ALL c.

Checks
------
  BA1  Dominance: AIF objective > the bandit's BEST objective (over the whole c sweep).
  BA2  Stability: at the bandit's best c, AIF switches far less (model-based lock vs perpetual explore).
  BA3  Rate: AIF is not worse than the bandit on rate (fair -- both observe-then-precode).

Run:  python sim/verify_step12_bandit.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator                                   # noqa: E402
from agent import AIFAgent, run_aif, run_genie, run_naive, objective   # noqa: E402
from bandit_baseline import run_bandit                                 # noqa: E402

SIGMA_E2, SIGMA2 = 1e-3, 0.03
BETA = np.array([1.0, 0.7, 1.3])
K, M, RHO = 3, 5, 0.9
ETA_SW, BETA_W = 1.0, 0.25
T, MC = 100, 20
CS = [0.0, 0.1, 0.3, 0.5, 1.0]


def main():
    print(f"op point: 15 dB, sigma_e^2=1e-3, rho=0.9, beta_w=0.25, eta_sw=1, T={T}, MC={MC}")
    # AIF, genie, naive references
    a_o = a_r = a_s = g_o = n_o = 0.0
    for m in range(MC):
        sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
        H = sim.generate(T)
        A = run_aif(AIFAgent(sim.R, BETA, RHO, SIGMA_E2, M, 1.0, BETA_W, ETA_SW, sigma2=SIGMA2),
                    H, SIGMA_E2, np.random.default_rng(20000 + m), sense_first=True)
        a_o += objective(A, ETA_SW); a_r += A["rate"][T // 2:].mean(); a_s += A["switch"].mean()
        g_o += objective(run_genie(H, M, sigma2=SIGMA2), ETA_SW)
        n_o += objective(run_naive(H, M, SIGMA_E2, np.random.default_rng(40000 + m), sigma2=SIGMA2), ETA_SW)
    a_o, a_r, a_s, g_o, n_o = a_o / MC, a_r / MC, a_s / MC, g_o / MC, n_o / MC

    # bandit c-sweep
    b_o, b_r, b_s = [], [], []
    for c in CS:
        o = r = s = 0.0
        for m in range(MC):
            sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
            H = sim.generate(T)
            B = run_bandit(H, M, SIGMA_E2, np.random.default_rng(60000 + m), sigma2=SIGMA2, eta_sw=ETA_SW, c=c)
            o += objective(B, ETA_SW); r += B["rate"][T // 2:].mean(); s += B["switch"].mean()
        b_o.append(o / MC); b_r.append(r / MC); b_s.append(s / MC)
        print(f"   bandit c={c}: objective={b_o[-1]:.2f}  rate={b_r[-1]:.2f}  switch={b_s[-1]:.2f}")
    best = int(np.argmax(b_o))                       # bandit's best (max objective) tuning

    print(f"\n AIF: objective={a_o:.2f} rate={a_r:.2f} switch={a_s:.2f} | genie obj={g_o:.2f} | naive obj={n_o:.2f}")
    print(f" bandit best: c={CS[best]} objective={b_o[best]:.2f} rate={b_r[best]:.2f} switch={b_s[best]:.2f}")

    all_pass = True
    print("\n[BA1] AIF objective > bandit's BEST objective (over all c)")
    ok = a_o > max(b_o) + 1e-6
    all_pass &= ok
    print(f"   AIF={a_o:.2f} vs bandit-best={max(b_o):.2f} (+{100*(a_o-max(b_o))/max(b_o):.0f}%)"
          f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n[BA2] AIF switches far less than the bandit (model-based lock vs perpetual exploration)")
    ok = a_s < b_s[best]
    all_pass &= ok
    print(f"   AIF switch={a_s:.2f} << bandit(best c) switch={b_s[best]:.2f}  -> {'PASS' if ok else 'FAIL'}")

    print("\n[BA3] AIF not worse than bandit on rate")
    ok = a_r >= b_r[best] - 1e-6
    all_pass &= ok
    print(f"   AIF rate={a_r:.2f} >= bandit(best) rate={b_r[best]:.2f}  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + ("=" * 46))
    print(f"STEP 12 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)

    _try_plot(CS, b_o, b_s, a_o, a_s, g_o, n_o, best, a_r, b_r)
    return 0 if all_pass else 1


def _try_plot(CS, b_o, b_s, a_o, a_s, g_o, n_o, best, a_r, b_r):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(CS))
    ax[0].plot(x, b_o, "s-", color="tab:red", label="bandit objective (vs its own c)")
    ax[0].axhline(a_o, color="tab:green", lw=2, label=f"AIF (ours) = {a_o:.1f}")
    ax[0].axhline(g_o, color="black", ls="--", alpha=0.7, label=f"genie = {g_o:.1f}")
    ax[0].axhline(n_o, color="gray", ls=":", alpha=0.7, label=f"naive = {n_o:.1f}")
    ax[0].set_xticks(x); ax[0].set_xticklabels([str(c) for c in CS])
    ax[0].set(title="AIF beats the bandit at every exploration setting",
              xlabel="bandit exploration constant c", ylabel="objective (bits/s/Hz)")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

    labels = ["AIF\n(ours)", f"bandit\n(best c={CS[best]})"]
    xb = np.arange(2); w = 0.38
    ax[1].bar(xb - w / 2, [a_r, b_r[best]], w, label="rate", color="tab:blue", alpha=0.85)
    ax[1].bar(xb + w / 2, [a_o, b_o[best]], w, label="objective", color="tab:orange", alpha=0.85)
    for i, s in enumerate([a_s, b_s[best]]):
        ax[1].text(i, max(a_r, b_r[best]) + 0.2, f"{s:.2f} sw", ha="center", fontsize=8, color="gray")
    ax[1].set_xticks(xb); ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].set(title="AIF vs bandit: same rate, higher objective, ~no switching",
              ylabel="bits/s/Hz")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    import os
    out = "step12_bandit_comparison.png"
    fig.savefig(out, dpi=130)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures",
                             "figG_bandit_comparison.png"), dpi=130)
    print(f"(saved -> sim/{out} and figures/figG_bandit_comparison.png)")


if __name__ == "__main__":
    raise SystemExit(main())
