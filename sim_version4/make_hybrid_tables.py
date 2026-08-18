r"""
Version-3 hybrid-beamforming results as TABLES (no figures).

Operating point OP_V3: 2 lambda, lambda/10, N=441 candidate ports, M=10 active, K=3.
The M active ports are driven by n_rf RF chains through a fully-connected unit-modulus
analog network. The AIF loop (belief, EFE selection, per-port pilot sensing, Kalman update)
is UNCHANGED -- only the transmit precoder is factorized into F_RF W_BB (PE / coordinate
descent AltMin from the belief-based digital precoder).

Two tables, written to results_v3.md and echoed to stdout:

  H1  RF-chain sweep -- rate vs n_rf for genie / AIF / naive, all hybrid. Shows the
      near-lossless threshold (<= 2K = 6) and graceful degradation at n_rf = K. The
      n_rf = M rows are the fully-digital anchors.
  H2  Joint budget  -- (M active ports) x (n_rf RF chains) rate surface for AIF: how the
      two hardware budgets trade against each other.

Speed: selection is n_rf-independent, so each seed's greedy/belief pass is run ONCE and all
n_rf are scored from it (run_*_sweep helpers) -- ~6x faster than a naive per-n_rf loop.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V3
from channel import ChannelSimulator
from agent import AIFAgent, run_aif_sweep, run_genie_sweep, run_naive_sweep

OP = OP_V3
POS, R = OP.positions(), OP.R()
MC, T = 6, 40
HALF = slice(T // 2, None)
HERE = os.path.dirname(os.path.abspath(__file__))
OUTMD = os.path.join(HERE, "results_v3.md")

_LINES = []
def emit(s=""):
    print(s, flush=True)
    _LINES.append(s)


def trajectories():
    Hs = []
    for m in range(MC):
        sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy,
                               K=OP.K, rho=OP.rho, beta=OP.beta, seed=m)
        Hs.append(sim.generate(T))
    return Hs


def mean_over_seeds(per_seed_rate_dicts, n_rf):
    """Mean second-half rate over seeds for a given n_rf key."""
    return float(np.mean([d[n_rf][HALF].mean() for d in per_seed_rate_dicts]))


def aif_agent(M, n_rf=None):
    return AIFAgent(R, OP.beta, OP.rho, OP.sigma_e2, M, 1.0, OP.beta_w, OP.eta_sw,
                    sigma2=OP.sigma2, positions=POS, d_min=OP.d_min, n_rf=n_rf)


# --------------------------------------------------------------- H1: RF-chain sweep
def table_rf_sweep(Hs):
    n_rfs = [None, 3, 4, 5, 6, 8, 10]                     # None = fully-digital anchor
    emit(f"### H1 — RF-chain sweep (M={OP.M} active, K={OP.K}, 2K={2*OP.K}, beta_w={OP.beta_w})\n")
    emit("Rate (bits/slot) vs number of RF chains n_rf. 'digital' = one chain per active "
         "port (n_rf=M=10). n_rf >= 2K=6 recovers digital; the loss concentrates at n_rf=K.\n")

    aif = [run_aif_sweep(aif_agent(OP.M), Hs[m], OP.sigma_e2,
                         np.random.default_rng(20000 + m), n_rfs)[0] for m in range(MC)]
    gen = [run_genie_sweep(Hs[m], OP.M, n_rfs, sigma2=OP.sigma2, positions=POS, d_min=OP.d_min,
                           rng=np.random.default_rng(60000 + m)) for m in range(MC)]
    nai = [run_naive_sweep(Hs[m], OP.M, OP.sigma_e2, np.random.default_rng(40000 + m), n_rfs,
                           sigma2=OP.sigma2, positions=POS, d_min=OP.d_min) for m in range(MC)]

    dig_aif = mean_over_seeds(aif, None)
    emit("| n_rf | note | genie | AIF | naive | AIF % of digital |")
    emit("|---:|---|---:|---:|---:|---:|")
    for nr in n_rfs:
        g, a, nv = (mean_over_seeds(gen, nr), mean_over_seeds(aif, nr), mean_over_seeds(nai, nr))
        label = str(OP.M) + " (digital)" if nr is None else str(nr)
        note = "digital" if nr is None else ("= K" if nr == OP.K else ("= 2K" if nr == 2 * OP.K else ""))
        emit(f"| {label} | {note} | {g:.2f} | {a:.2f} | {nv:.2f} | {100*a/dig_aif:.0f}% |")
    emit("")


# --------------------------------------------------------------- H2: joint (M, n_rf) budget
def table_joint_budget(Hs):
    emit("### H2 — Joint budget: AIF rate for (M active ports) x (n_rf RF chains)\n")
    Ms = [6, 8, 10]
    rfs = [3, 4, 6, 8, 10]
    emit("| M \\ n_rf | " + " | ".join(str(r) for r in rfs) + " |")
    emit("|---:|" + "|".join(["---:"] * len(rfs)) + "|")
    for M in Ms:
        cols = [nr for nr in rfs if nr <= M]
        per_seed = [run_aif_sweep(aif_agent(M), Hs[m], OP.sigma_e2,
                                  np.random.default_rng(20000 + m), cols)[0] for m in range(MC)]
        cells = []
        for nr in rfs:
            cells.append("-" if nr > M else f"{mean_over_seeds(per_seed, nr):.2f}")
        emit(f"| {M} | " + " | ".join(cells) + " |")
    emit("")


def main():
    t0 = time.perf_counter()
    emit("# OP_V3 hybrid-beamforming results (no figures)\n")
    emit(f"**Operating point:** {OP.label()}")
    emit(f"**Monte-Carlo:** {MC} seeds, T={T} slots, second-half averaging. Fully-connected "
         f"unit-modulus analog network; hybrid factorized from the belief-based precoder by "
         f"coordinate-descent AltMin.\n")
    Hs = trajectories()
    table_rf_sweep(Hs)
    table_joint_budget(Hs)
    emit(f"_generated in {time.perf_counter()-t0:.0f}s_")
    with open(OUTMD, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES) + "\n")
    print(f"\n-> wrote {OUTMD}")


if __name__ == "__main__":
    main()
