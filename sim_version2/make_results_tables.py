r"""
OP_V2 results as TABLES (no figures). Regenerates the headline numbers at the rescaled
operating point (2 lambda, lambda/10, N=441, M=8, d_min=0.5) so the stale N=25 numbers
(20% budget, 84%/89% of genie) can be replaced.

Three tradeoff tables, all written to results_v2.md and echoed to stdout:

  T1  Baselines          -- genie / AIF / naive / random at the default point, plus AIF with
                            d_min OFF (the mutual-coupling comparison). rate | objective | switching.
  T2  Observation budget -- M sweep: how rate/objective trade against how many ports we sense.
  T3  Exploration weight -- beta_w sweep: the rate vs switching Pareto (the tunable knob).

Usage:  python sim_version2/make_results_tables.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V2
from channel import ChannelSimulator
from agent import (AIFAgent, run_aif, run_genie, run_naive,
                   run_random_partial, objective)

OP = OP_V2
POS, R = OP.positions(), OP.R()
MC, T = 6, 40
HALF = slice(T // 2, None)
HERE = os.path.dirname(os.path.abspath(__file__))
OUTMD = os.path.join(HERE, "results_v2.md")

_LINES = []
def emit(s=""):
    print(s, flush=True)
    _LINES.append(s)


def trajectories():
    """One channel trajectory per Monte-Carlo seed (shared across all tables)."""
    Hs = []
    for m in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy,
                               K=OP.K, rho=OP.rho, beta=OP.beta, seed=m)
        Hs.append(sim.generate(T))
    return Hs


def aif_run(H, m, M, beta_w, d_min):
    ag = AIFAgent(R, OP.beta, OP.rho, OP.sigma_e2, M, 1.0, beta_w, OP.eta_sw,
                  sigma2=OP.sigma2, positions=POS, d_min=d_min)
    return run_aif(ag, H, OP.sigma_e2, np.random.default_rng(20000 + m), sense_first=True)


def agg(runs):
    """Mean second-half rate, objective, switching over a list of per-seed result dicts."""
    rate = np.mean([r["rate"][HALF].mean() for r in runs])
    obj = np.mean([objective(r, OP.eta_sw) for r in runs])
    sw = np.mean([r["switch"].mean() for r in runs])
    return rate, obj, sw


# ------------------------------------------------------------------ T1: baselines
def table_baselines(Hs):
    emit("### T1 — Baselines at the default point "
         f"(M={OP.M}, {100*OP.M/OP.N:.0f}% of ports, beta_w={OP.beta_w}, d_min=0.5)\n")
    methods = {
        "genie (full CSI)":      lambda H, m: run_genie(H, OP.M, sigma2=OP.sigma2, positions=POS, d_min=OP.d_min),
        "AIF (ours)":            lambda H, m: aif_run(H, m, OP.M, OP.beta_w, OP.d_min),
        "AIF (d_min OFF)":       lambda H, m: aif_run(H, m, OP.M, OP.beta_w, None),
        "naive (no inference)":  lambda H, m: run_naive(H, OP.M, OP.sigma_e2, np.random.default_rng(40000 + m),
                                                        sigma2=OP.sigma2, positions=POS, d_min=OP.d_min),
        "random (partial CSI)":  lambda H, m: run_random_partial(H, OP.M, OP.sigma_e2, np.random.default_rng(50000 + m),
                                                                 sigma2=OP.sigma2, positions=POS, d_min=OP.d_min),
    }
    emit("| method | rate | % genie | objective | switch/slot |")
    emit("|---|---:|---:|---:|---:|")
    gr = None
    for name, fn in methods.items():
        runs = [fn(Hs[m], m) for m in range(MC)]
        rate, obj, sw = agg(runs)
        if gr is None:
            gr = rate
        emit(f"| {name} | {rate:.2f} | {100*rate/gr:.0f}% | {obj:.2f} | {sw:.2f} |")
    emit("")


# ------------------------------------------------------------------ T2: observation budget
def table_budget(Hs):
    emit("### T2 — Observation budget (M sweep, d_min=0.5, beta_w=0.25)\n")
    emit("| M | % ports | genie rate | AIF rate | % genie | AIF obj | naive rate |")
    emit("|---:|---:|---:|---:|---:|---:|---:|")
    for M in [4, 6, 8, 10, 12]:
        g = [run_genie(Hs[m], M, sigma2=OP.sigma2, positions=POS, d_min=OP.d_min) for m in range(MC)]
        a = [aif_run(Hs[m], m, M, OP.beta_w, OP.d_min) for m in range(MC)]
        n = [run_naive(Hs[m], M, OP.sigma_e2, np.random.default_rng(40000 + m),
                       sigma2=OP.sigma2, positions=POS, d_min=OP.d_min) for m in range(MC)]
        gr, _, _ = agg(g); ar, ao, _ = agg(a); nr, _, _ = agg(n)
        emit(f"| {M} | {100*M/OP.N:.1f}% | {gr:.2f} | {ar:.2f} | {100*ar/gr:.0f}% | {ao:.2f} | {nr:.2f} |")
    emit("")


# ------------------------------------------------------------------ T3: exploration weight / Pareto
def table_exploration(Hs):
    emit("### T3 — Exploration weight (beta_w sweep, M=8, d_min=0.5) — rate vs switching Pareto\n")
    emit("| beta_w | rate | % genie | objective | switch/slot |")
    emit("|---:|---:|---:|---:|---:|")
    gref = agg([run_genie(Hs[m], OP.M, sigma2=OP.sigma2, positions=POS, d_min=OP.d_min)
                for m in range(MC)])[0]
    for bw in [0.0, 0.1, 0.25, 0.5, 1.0]:
        a = [aif_run(Hs[m], m, OP.M, bw, OP.d_min) for m in range(MC)]
        rate, obj, sw = agg(a)
        emit(f"| {bw:g} | {rate:.2f} | {100*rate/gref:.0f}% | {obj:.2f} | {sw:.2f} |")
    emit("")


def main():
    t0 = time.perf_counter()
    emit(f"# OP_V2 results (no figures)\n")
    emit(f"**Operating point:** {OP.label()}")
    emit(f"**Monte-Carlo:** {MC} seeds, T={T} slots, second-half averaging. "
         f"All selection methods honor the >= 0.5 lambda min-spacing constraint.\n")
    Hs = trajectories()
    table_baselines(Hs)
    table_budget(Hs)
    table_exploration(Hs)
    emit(f"_generated in {time.perf_counter()-t0:.0f}s_")
    with open(OUTMD, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES) + "\n")
    print(f"\n-> wrote {OUTMD}")


if __name__ == "__main__":
    main()
