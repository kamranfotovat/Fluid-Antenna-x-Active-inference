"""
Step 8 verification -- slow-loop learning of R, and objection-proofing under model mismatch (Fig E).

Step 7c established that R is the ONE parameter whose mismatch hurts. This step shows the agent can
LEARN R from its own partial observations and thereby (a) recover the wrong-R gap on a Jakes channel,
and (b) -- the real objection-proofing -- adapt when the true propagation is NOT Jakes at all.

Checks
------
  L1  Estimator consistency.  Pair coverage -> 1 and R_hat error decreases with warm-up length.
  L2  Recovery (Jakes truth).  oracle-R >= learned-R > wrong-R (uncorrelated) on the objective.
  L3  Model mismatch (Fig E).  True channel is EXPONENTIAL-correlated (non-Jakes). An agent that
      wrongly assumes Jakes loses; the learned-R agent adapts and recovers toward the oracle.

Run:  python sim/verify_step8_learning.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator, spatial_correlation, port_positions  # noqa: E402
from agent import AIFAgent, run_aif, objective                              # noqa: E402
from learning import (SpatialCorrEstimator, gather_correlation,             # noqa: E402
                      exponential_correlation, set_correlation)

SIGMA_E2, SIGMA2 = 1e-2, 0.03
BETA = np.array([1.0, 0.7, 1.3])
K, M, RHO = 3, 5, 0.9
ETA_SW, BETA_W = 1.0, 0.25
T, MC, T_WARM = 60, 12, 150


def _run_obj(R_belief, H, seed):
    a = run_aif(AIFAgent(R_belief, BETA, RHO, SIGMA_E2, M, 1.0, BETA_W, ETA_SW, sigma2=SIGMA2),
                H, SIGMA_E2, np.random.default_rng(20000 + seed), sense_first=True)
    return objective(a, ETA_SW)


def main():
    print(f"N=25, K={K}, M={M}, 15 dB, rho={RHO}, beta_w={BETA_W}, T={T}, MC={MC}, T_warm={T_WARM}")
    N = 25
    I = np.eye(N)
    all_pass = True

    # ---- L1: estimator consistency ----------------------------------------
    print("\n[L1] Estimator consistency: coverage -> 1, R_hat error falls with warm-up")
    prev_err = np.inf
    ok = True
    for Tw in [30, 80, 200]:
        errs, covs = [], []
        for m in range(MC):
            sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
            Rhat, cov = gather_correlation(sim, Tw, M, SIGMA_E2, np.random.default_rng(60000 + m))
            errs.append(np.linalg.norm(Rhat - sim.R) / np.linalg.norm(sim.R)); covs.append(cov)
        e, c = np.mean(errs), np.mean(covs)
        print(f"   T_warm={Tw:3d}: R_hat rel-Fro err={e:.3f}, pair coverage={c:.2f}")
        ok &= (e < prev_err); prev_err = e
    ok &= c > 0.99
    all_pass &= ok
    print(f"   error monotone-decreasing and coverage->1  -> {'PASS' if ok else 'FAIL'}")

    # ---- L2: recovery on Jakes truth --------------------------------------
    print("\n[L2] Recovery (Jakes truth): oracle-R >= learned-R > wrong-R")
    o = w = l = 0.0
    for m in range(MC):
        sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
        H = sim.generate(T)
        Rhat, _ = gather_correlation(ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO,
                                     beta=BETA, seed=m), T_WARM, M, SIGMA_E2, np.random.default_rng(60000 + m))
        o += _run_obj(sim.R, H, m); w += _run_obj(I, H, m); l += _run_obj(Rhat, H, m)
    o, w, l = o / MC, w / MC, l / MC
    recovered = (l - w) / (o - w) if o > w else 0.0
    ok = o >= l > w
    all_pass &= ok
    print(f"   objective: oracle={o:.2f}  learned={l:.2f}  wrong(I)={w:.2f}  "
          f"(learned recovers {recovered*100:.0f}% of the gap)  -> {'PASS' if ok else 'FAIL'}")

    # ---- L3: model mismatch (Fig E) ---------------------------------------
    print("\n[L3] Model mismatch (Fig E): true channel is EXPONENTIAL (non-Jakes)")
    pos = port_positions(5, 5, 1.0, 1.0)
    R_exp = exponential_correlation(pos, d0=0.3)     # the true (non-Jakes) correlation
    R_jakes = spatial_correlation(pos)               # what a naive agent wrongly assumes
    o = mism = l = 0.0
    for m in range(MC):
        sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
        set_correlation(sim, R_exp)                   # channel now non-Jakes
        H = sim.generate(T)
        simw = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
        set_correlation(simw, R_exp)
        Rhat, _ = gather_correlation(simw, T_WARM, M, SIGMA_E2, np.random.default_rng(70000 + m))
        o += _run_obj(R_exp, H, m)                    # oracle knows the true exponential R
        mism += _run_obj(R_jakes, H, m)              # mismatch: assumes Jakes
        l += _run_obj(Rhat, H, m)                    # learned from data
    o, mism, l = o / MC, mism / MC, l / MC
    recovered = (l - mism) / (o - mism) if o > mism else 0.0
    ok = l > mism + 1e-6 and l >= 0.97 * o        # beats the mismatched agent, reaches oracle (within MC noise)
    all_pass &= ok
    print(f"   objective: oracle(true R)={o:.2f}  assumes-Jakes(wrong)={mism:.2f}  learned={l:.2f}")
    print(f"   learning recovers {recovered*100:.0f}% of the mismatch gap (learned ~= oracle)"
          f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + ("=" * 46))
    print(f"STEP 8 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)

    _try_plot(R_exp, R_jakes, pos, o, mism, l)
    return 0 if all_pass else 1


def _try_plot(R_exp, R_jakes, pos, o, mism, l):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    # learn one R_hat for display
    sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=0)
    set_correlation(sim, R_exp)
    Rhat, _ = gather_correlation(sim, 400, M, SIGMA_E2, np.random.default_rng(1))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, Mtx, title in [(ax[0], R_exp, "True R (exponential)"),
                          (ax[1], Rhat, "Learned R_hat (from data)"),
                          (ax[2], R_jakes, "Assumed R (Jakes, wrong)")]:
        im = a.imshow(Mtx, cmap="viridis", vmin=-0.3, vmax=1.0)
        a.set_title(title, fontsize=10); a.set_xlabel("port j"); a.set_ylabel("port i")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"Model mismatch: learned R_hat tracks the true R "
                 f"(objective  oracle={o:.1f}  learned={l:.1f}  assumes-Jakes={mism:.1f})", fontsize=11)
    fig.tight_layout()
    out = "step8_learning_mismatch.png"
    fig.savefig(out, dpi=120)
    print(f"(saved plot -> sim/{out})")


if __name__ == "__main__":
    raise SystemExit(main())
