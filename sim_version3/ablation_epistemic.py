r"""
EPISTEMIC ABLATION -- does the information-gain (beta_w) term earn its keep, or is
"active inference" just a relabel of rate-greedy selection?

The decisive test for novelty. We hold EVERYTHING fixed (belief, protocol, precoder,
channel trajectory, switching cost) and vary ONLY beta_w, the weight on the epistemic
(mutual-information) term in G(S) = -alpha*rate - beta_w*info + switching.

  beta_w = 0.0  ->  PURE rate-greedy selection (no active inference, no exploration).
  beta_w > 0    ->  adds the information-seeking term.

If the epistemic term is real, higher beta_w should IMPROVE the long-term objective
(rate net of switching cost) -- and improve it MORE under stress (noisy pilots / fast
aging), where the agent's belief degrades and knowing what it doesn't know matters.

If the objective is flat or worse in beta_w, the term is dressing and "active inference"
is rebranding. This script prints the numbers that settle it.

Protocol: observe-then-precode (sense_first), fully-digital transmit (isolates SELECTION).
Metrics (means over MC seeds, second-half slots to skip warm-up):
  rate      realized sum-rate on the TRUE channel (b/s/Hz)
  switch    ports changed per slot
  obj       rate - eta_sw*switch      (the Eq.7 long-term objective -- the honest score)
  %genie    rate as a fraction of the full-CSI genie
  Uglobal   mean posterior variance over ALL N ports (how well it knows the whole grid)

Run:  python ablation_epistemic.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator
from agent import AIFAgent, run_genie
from precoding import sinr_and_rates
import efe
from config import OP_V3

OP = OP_V3
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
T = int(sys.argv[2]) if len(sys.argv) > 2 else 48
HALF = slice(T // 2, None)


def _switch(S, Sp):
    return 0 if Sp is None else len(set(S) ^ set(Sp))


def run_one(H, R, beta_w, sigma_e2, rho, seed_obs):
    """One closed-loop run (observe-then-precode, digital). Returns per-slot arrays."""
    K, N = OP.K, OP.N
    agent = AIFAgent(R=R, beta=OP.beta, rho=rho, sigma_e2=sigma_e2, M=OP.M,
                     alpha=1.0, beta_w=beta_w, eta_sw=OP.eta_sw, e_sw=1.0,
                     sigma2=OP.sigma2, P=OP.P,
                     positions=OP.positions(), d_min=OP.d_min, n_rf=None)
    agent.reset()
    rng = np.random.default_rng(seed_obs)
    Tt = H.shape[0]
    rate = np.zeros(Tt); switch = np.zeros(Tt); uglob = np.zeros(Tt)
    for t in range(Tt):
        S = agent.select(first=(t == 0))
        idx = list(S)
        noise = np.sqrt(sigma_e2 / 2) * (rng.standard_normal((K, len(idx)))
                                         + 1j * rng.standard_normal((K, len(idx))))
        y = H[t][:, idx] + noise
        agent.bel.update(S, y)                          # sense-first: fresh belief
        uglob[t] = agent.bel.port_variances().mean()    # global uncertainty over ALL ports
        W = agent.precoder(S)
        Ht = H[t][:, idx].T
        rate[t] = float(sinr_and_rates(Ht, W, agent.sigma2)[1].sum())
        switch[t] = _switch(S, agent.S_prev)
        agent.S_prev = S
    return rate, switch, uglob


def sweep(betas, sigma_e2, rho, label):
    print(f"\n{'='*72}\n{label}   (sigma_e2={sigma_e2:g}, rho={rho:g})\n{'='*72}")
    print(f"{'beta_w':>7} | {'rate':>7} | {'switch':>7} | {'obj':>8} | "
          f"{'%genie':>7} | {'Uglobal':>8}")
    print("-" * 60)
    acc = {b: {"rate": [], "sw": [], "obj": [], "pg": [], "ug": []} for b in betas}
    for s in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                               rho=rho, beta=OP.beta, seed=300 + s)
        H = sim.generate(T)
        R = sim.R
        genie = run_genie(H, OP.M, sigma2=OP.sigma2, P=OP.P,
                          positions=OP.positions(), d_min=OP.d_min)["rate"]
        gmean = genie[HALF].mean()
        for b in betas:
            rate, switch, uglob = run_one(H, R, b, sigma_e2, rho, seed_obs=700 + s)
            r = rate[HALF].mean(); sw = switch[HALF].mean()
            acc[b]["rate"].append(r)
            acc[b]["sw"].append(sw)
            acc[b]["obj"].append(r - OP.eta_sw * sw)
            acc[b]["pg"].append(100 * r / gmean)
            acc[b]["ug"].append(uglob[HALF].mean())
    rows = {}
    for b in betas:
        r = np.mean(acc[b]["rate"]); sw = np.mean(acc[b]["sw"])
        obj = np.mean(acc[b]["obj"]); pg = np.mean(acc[b]["pg"]); ug = np.mean(acc[b]["ug"])
        rows[b] = (r, sw, obj, pg, ug)
        tag = "  <- rate-greedy (no AIF)" if b == 0.0 else ""
        print(f"{b:7.2f} | {r:7.3f} | {sw:7.3f} | {obj:8.3f} | {pg:7.1f} | {ug:8.4f}{tag}")
    return rows


def verdict(rows_nom):
    b0 = rows_nom[0.0]
    best_b = max(rows_nom, key=lambda b: rows_nom[b][2])   # max objective
    best = rows_nom[best_b]
    d_obj = best[2] - b0[2]
    d_rate = best[0] - b0[0]
    print(f"\n{'#'*72}\nVERDICT (nominal point)\n{'#'*72}")
    print(f"  rate-greedy (beta_w=0):   obj={b0[2]:.3f}  rate={b0[0]:.3f}  "
          f"switch={b0[1]:.3f}  Uglobal={b0[4]:.4f}")
    print(f"  best epistemic (beta_w={best_b:g}): obj={best[2]:.3f}  rate={best[0]:.3f}  "
          f"switch={best[1]:.3f}  Uglobal={best[4]:.4f}")
    print(f"  delta objective = {d_obj:+.3f} b/s/Hz   delta rate = {d_rate:+.3f} b/s/Hz")
    if d_obj > 0.05:
        print("  => epistemic term IMPROVES the objective: it is load-bearing, not dressing.")
    elif d_obj > -0.05:
        print("  => epistemic term ~ neutral on objective: WEAK novelty signal (rebranding risk).")
    else:
        print("  => epistemic term HURTS the objective: pure exploration is not paying off here.")


def main():
    t0 = time.perf_counter()
    print(f"OP_V3: {OP.label()}   |  MC={MC}, T={T}, second-half slots, digital transmit")
    betas = [0.0, 0.1, 0.25, 0.6, 1.0]

    rows_nom = sweep(betas, OP.sigma_e2, OP.rho, "A. NOMINAL")
    verdict(rows_nom)

    # Stress 1: noisy pilots (belief stays uncertain -> knowing what you don't know matters)
    print(f"\n\n{'*'*72}\nB. STRESS -- NOISY PILOTS  (sigma_e2 up: 1e-3 -> 1e-1)\n{'*'*72}")
    for se2 in (1e-2, 1e-1):
        sweep([0.0, 0.6], se2, OP.rho, f"noisy pilots sigma_e2={se2:g}")

    # Stress 2: fast aging (good ports drift -> must re-explore)
    print(f"\n\n{'*'*72}\nC. STRESS -- FAST AGING  (rho down: 0.9 -> 0.7)\n{'*'*72}")
    for r in (0.8, 0.7):
        sweep([0.0, 0.6], OP.sigma_e2, r, f"fast aging rho={r:g}")

    print(f"\n(total {time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
