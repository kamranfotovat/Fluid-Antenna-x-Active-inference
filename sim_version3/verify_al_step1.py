r"""
AL-1 gate -- passive online learning of g(d), and the diagnosis that MOTIVATES active learning.

True channel = exponential (non-Jakes); agent starts assuming Jakes and learns g(d) online from its
own measurements. Key finding: the communication policy SPREADS active ports (diversity), so it
co-observes only LONG distances and STARVES the near-field bins -- exactly where g(d) is large and
load-bearing. So passive learning corrects the correlation where it doesn't matter and can't touch
it where it does -> ~0 objective recovery. This is the motivation for AL-2 (active learning).

Checks (means over MC seeds):
  A. the estimator is CORRECT where it gets data: in sampled bins, learned g-error < prior g-error.
  B. UNDER-SAMPLING confirmed: short-distance (d<0.4 lambda) co-observations are near-zero vs long.
  C. R_hat stays a valid correlation (PSD, unit diagonal).
Reported (not gated): passive-learn objective vs fixed/oracle -- expect ~0% gap recovery.

Run:  python verify_al_step1.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V3
from channel import ChannelSimulator, spatial_correlation
from agent import AIFAgent, run_aif, objective
from learning import exponential_correlation, set_correlation
from dist_profile import DistanceProfileEstimator
from active_learn import run_aif_learn

OP = OP_V3
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T = int(sys.argv[2]) if len(sys.argv) > 2 else 60
D0 = 0.3
RELEARN = 5
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def _agent(R):
    return AIFAgent(R=R, beta=OP.beta, rho=OP.rho, sigma_e2=OP.sigma_e2, M=OP.M,
                    alpha=1.0, beta_w=OP.beta_w, eta_sw=OP.eta_sw, e_sw=1.0,
                    sigma2=OP.sigma2, P=OP.P, positions=OP.positions(), d_min=OP.d_min)


def main():
    print(f"OP_V3: {OP.label()}\nMC={MC}, T={T}, TRUE=exp(d0={D0}), start=Jakes, relearn every {RELEARN}\n")
    pos = OP.positions()
    R_jakes = spatial_correlation(pos)
    R_true = exponential_correlation(pos, d0=D0)

    acc = {"oracle": [], "fixed": [], "learn": []}
    prior_err_s, learn_err_s = [], []       # g-error over SAMPLED bins: prior vs learned
    short_co, long_co = [], []
    psd_ok = True
    t0 = time.perf_counter()
    for s in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                               rho=OP.rho, beta=OP.beta, seed=100 + s)
        set_correlation(sim, R_true)
        H = sim.generate(T)
        acc["oracle"].append(objective(run_aif(_agent(R_true), H, OP.sigma_e2,
                             np.random.default_rng(200 + s), sense_first=True), OP.eta_sw))
        acc["fixed"].append(objective(run_aif(_agent(R_jakes), H, OP.sigma_e2,
                             np.random.default_rng(200 + s), sense_first=True), OP.eta_sw))
        est = DistanceProfileEstimator(pos, OP.sigma_e2, bin_width=0.1)
        true_g = est.true_g_bins(R_true)
        ag = _agent(R_jakes)
        res = run_aif_learn(ag, H, OP.sigma_e2, np.random.default_rng(200 + s), est,
                            relearn_every=RELEARN, sense_first=True, true_g=true_g)
        acc["learn"].append(objective(res, OP.eta_sw))
        m = est.mc > 0                                                # sampled bins
        prior_err_s.append(np.sqrt(np.mean((est.prior[m] - true_g[m]) ** 2)))
        learn_err_s.append(np.sqrt(np.mean((est.g_hat()[m] - true_g[m]) ** 2)))
        short_co.append(int(est.mc[est.centers < 0.4].sum()))
        long_co.append(int(est.mc[est.centers > 0.8].sum()))
        Rh = res["R_hat"]
        psd_ok &= (np.linalg.eigvalsh(Rh).min() >= -1e-8) and np.allclose(np.diag(Rh), 1.0)
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    O = {k: float(np.mean(v)) for k, v in acc.items()}
    pe, le = float(np.mean(prior_err_s)), float(np.mean(learn_err_s))
    sc, lc = float(np.mean(short_co)), float(np.mean(long_co))
    print(f"\n{'agent':>8} | {'objective':>10}")
    print("-" * 22)
    for k in ("oracle", "learn", "fixed"):
        print(f"{k:>8} | {O[k]:10.3f}")
    gap = O["oracle"] - O["fixed"]
    rec = (O["learn"] - O["fixed"]) / gap * 100 if abs(gap) > 1e-9 else 0.0
    print(f"\nmismatch gap (oracle-fixed) = {gap:+.3f};  PASSIVE learning recovers {rec:.0f}% of it")
    print(f"g-error over SAMPLED bins:  Jakes prior {pe:.3f}  ->  learned {le:.3f}")
    print(f"co-observations:  short d<0.4λ = {sc:.0f}   |   long d>0.8λ = {lc:.0f}")
    print("=> passive policy STARVES the near-field -> motivates ACTIVE learning (AL-2)")

    print("\nGates:")
    check("A  estimator correct where sampled (learned g-error < prior)", le < pe - 1e-3,
          f"{pe:.3f} -> {le:.3f}")
    check("B  under-sampling confirmed (short co-obs < 1% of long)", sc < 0.01 * max(lc, 1),
          f"short {sc:.0f} vs long {lc:.0f}")
    check("C  R_hat valid (PSD, unit diag)", psd_ok)

    print("\n" + "=" * 44)
    print(f"AL-1 GATE: {'ALL PASS' if ok else 'FAILURES ABOVE'}  ({time.perf_counter()-t0:.0f}s)")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
