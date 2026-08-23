r"""
AL-2 gate -- ACTIVE learning of g(d): the novelty term makes the agent co-observe the near-field.

True channel = exponential; agent starts from Jakes. Compare, on shared trajectories:
  oracle / fixed        : upper / lower reference (objective)
  passive               : learns from comm measurements only (AL-1 -> ~0 near-field, 0% recovery)
  random-probe          : undirected probing (random feasible sets) -> mostly long distances
  active(lam_model)      : novelty term targets under-sampled SHORT bins -> learns near-field

Gates (means over MC seeds):
  A. active co-observes the near-field (short d<0.4 co-obs > 0 AND >> passive's ~0).
  B. active learns the near-field: active short-bin g-error < Jakes-prior short-bin g-error.
  C. active recovers real objective vs fixed (obj_active > obj_fixed) -- learning finally pays.
  D. directedness beats blind: active short-bin co-obs > random-probe's.

Run:  python verify_al_step2.py [MC] [T] [lam_model]
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
from active_learn import run_aif_learn, run_random_probe

OP = OP_V3
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 2
T = int(sys.argv[2]) if len(sys.argv) > 2 else 50
LAM = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
D0, RELEARN = 0.3, 5
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def _agent(R):
    return AIFAgent(R=R, beta=OP.beta, rho=OP.rho, sigma_e2=OP.sigma_e2, M=OP.M,
                    alpha=1.0, beta_w=OP.beta_w, eta_sw=OP.eta_sw, e_sw=1.0,
                    sigma2=OP.sigma2, P=OP.P, positions=OP.positions(), d_min=OP.d_min)


def _short_metrics(est, true_g):
    short = est.centers < 0.4
    co = float(est.mc[short].sum())
    prior_e = np.sqrt(np.mean((est.prior[short] - true_g[short]) ** 2))
    learn_e = np.sqrt(np.mean((est.g_hat()[short] - true_g[short]) ** 2))
    return co, float(prior_e), float(learn_e)


def main():
    print(f"OP_V3: {OP.label()}\nMC={MC}, T={T}, lam_model={LAM}, TRUE=exp(d0={D0}), start=Jakes\n")
    pos = OP.positions()
    R_jakes = spatial_correlation(pos)
    R_true = exponential_correlation(pos, d0=D0)

    acc = {k: [] for k in ("oracle", "fixed", "passive", "random", "active")}
    co = {k: [] for k in ("passive", "random", "active")}
    perr = {k: [] for k in ("passive", "random", "active")}
    lerr = {k: [] for k in ("passive", "random", "active")}
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
        runs = {}
        est_p = DistanceProfileEstimator(pos, OP.sigma_e2, bin_width=0.1)
        true_g = est_p.true_g_bins(R_true)
        runs["passive"] = (run_aif_learn(_agent(R_jakes), H, OP.sigma_e2,
                           np.random.default_rng(200 + s), est_p, relearn_every=RELEARN), est_p)
        est_r = DistanceProfileEstimator(pos, OP.sigma_e2, bin_width=0.1)
        runs["random"] = (run_random_probe(_agent(R_jakes), H, OP.sigma_e2,
                          np.random.default_rng(300 + s), est_r, relearn_every=RELEARN), est_r)
        est_a = DistanceProfileEstimator(pos, OP.sigma_e2, bin_width=0.1)
        runs["active"] = (run_aif_learn(_agent(R_jakes), H, OP.sigma_e2,
                          np.random.default_rng(200 + s), est_a, relearn_every=RELEARN,
                          active=True, lam_model=LAM), est_a)
        for k, (res, est) in runs.items():
            acc[k].append(objective(res, OP.eta_sw))
            c, pe, le = _short_metrics(est, true_g)
            co[k].append(c); perr[k].append(pe); lerr[k].append(le)
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    O = {k: float(np.mean(v)) for k, v in acc.items()}
    print(f"\n{'agent':>8} | {'objective':>9} | {'short co-obs':>12} | {'short g-err':>11}")
    print("-" * 50)
    print(f"{'oracle':>8} | {O['oracle']:9.3f} |")
    for k in ("active", "random", "passive"):
        print(f"{k:>8} | {O[k]:9.3f} | {np.mean(co[k]):12.0f} | {np.mean(lerr[k]):11.3f}")
    print(f"{'fixed':>8} | {O['fixed']:9.3f} |")
    gap = O["oracle"] - O["fixed"]
    rec = (O["active"] - O["fixed"]) / gap * 100 if abs(gap) > 1e-9 else 0.0
    print(f"\nmismatch gap = {gap:+.3f};  ACTIVE recovers {rec:.0f}%  (passive "
          f"{(O['passive']-O['fixed'])/gap*100 if abs(gap)>1e-9 else 0:.0f}%)")
    pe_a = float(np.mean(perr["active"]))

    print("\nGates:")
    check("A  active co-observes near-field (>0 and >> passive)",
          np.mean(co["active"]) > 0 and np.mean(co["active"]) > 10 * (np.mean(co["passive"]) + 1),
          f"active {np.mean(co['active']):.0f} vs passive {np.mean(co['passive']):.0f}")
    check("B  active learns near-field (short g-err < prior)",
          np.mean(lerr["active"]) < pe_a - 1e-3, f"prior {pe_a:.3f} -> active {np.mean(lerr['active']):.3f}")
    check("C  active recovers objective vs fixed", O["active"] > O["fixed"] + 1e-3,
          f"active {O['active']:.3f} vs fixed {O['fixed']:.3f}")
    check("D  directed beats blind (active short co-obs > random)",
          np.mean(co["active"]) > np.mean(co["random"]),
          f"active {np.mean(co['active']):.0f} vs random {np.mean(co['random']):.0f}")

    print("\n" + "=" * 44)
    print(f"AL-2 GATE: {'ALL PASS' if ok else 'FAILURES ABOVE'}  ({time.perf_counter()-t0:.0f}s)")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
