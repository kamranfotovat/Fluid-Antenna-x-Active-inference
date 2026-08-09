"""
Step 2 verification -- reference-band baselines (full CSI).

Gate before Step 3. Thorough console report:
  A. Near-optimality of greedy: on a small problem (N=12, M=4) compare greedy vs the true
     exhaustive optimum -> greedy should recover ~99-100%.
  B. Reference band at the real dimensions (N=25, M=5, K=3): genie-greedy >= norm-topM >
     random. This is the band the AIF agent must land in.
  C. CSI-aging demo: build+select on a channel that is Delta slots stale, score on the true
     channel. Sum-rate collapses toward random as Delta grows -> this is WHY a belief that
     infers/refreshes the channel is worth building (Steps 3+).

Run from sim/:  python verify_step2.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator                          # noqa: E402
from selection import (                                        # noqa: E402
    evaluate, select_norm_topM, select_random, select_greedy, select_exhaustive,
)


def main():
    sigma2, P = 1e-3, 1.0
    all_pass = True

    print("=" * 70)
    print("STEP 2 -- REFERENCE SELECTION BASELINES  (full CSI)")
    print("=" * 70)

    # ---------------------------------------------------------------- A. greedy vs optimum
    print("\n[A] Greedy vs exhaustive optimum on a SMALL problem (N=12, M=4, K=3), 100 realizations.")
    print("    Sum-rate is not exactly submodular, but greedy is empirically near-optimal.")
    simA = ChannelSimulator(Nx=4, Ny=3, K=3, rho=0.9, beta=1.0, seed=11)
    ratios = []
    for _ in range(100):
        h = simA.reset()
        _, v_greedy = select_greedy(h, 4, sigma2=sigma2, P=P)
        _, v_opt = select_exhaustive(h, 4, sigma2=sigma2, P=P)
        ratios.append(v_greedy / v_opt)
    ratios = np.array(ratios)
    ok = ratios.mean() > 0.94
    all_pass &= ok
    print(f"    greedy/optimum ratio: mean = {ratios.mean():.4f}, min = {ratios.min():.4f} "
          f"(gate: mean > 0.94)  -> {'PASS' if ok else 'FAIL'}")
    print("    NOTE: the sum-RATE objective is NOT submodular, so greedy has no (1-1/e) guarantee")
    print("    here; ~95% of optimum is the honest, expected behaviour (worse when M is close to K")
    print(f"    and at high SNR). The (1-1/e) = {1 - 1/np.e:.4f} guarantee attaches to the EPISTEMIC")
    print("    info-gain term (Step 4), which IS submodular -- not to this rate baseline.")

    # ---------------------------------------------------------------- B. reference band
    print("\n[B] Reference band at N=25, M=5, K=3 (mean +/- std sum-rate over 300 realizations).")
    sim = ChannelSimulator(Nx=5, Ny=5, K=3, rho=0.9, beta=1.0, seed=21)
    rng = np.random.default_rng(123)
    M = 5
    acc = {"genie_greedy": [], "norm_topM": [], "random": []}
    for _ in range(300):
        h = sim.reset()
        acc["genie_greedy"].append(select_greedy(h, M, sigma2=sigma2, P=P)[1])
        acc["norm_topM"].append(evaluate(select_norm_topM(h, M), h, sigma2=sigma2, P=P))
        acc["random"].append(evaluate(select_random(h, M, rng), h, sigma2=sigma2, P=P))
    stats = {k: (np.mean(v), np.std(v)) for k, v in acc.items()}
    print(f"    {'strategy':>14} | {'mean':>7} | {'std':>6}")
    for k in ["genie_greedy", "norm_topM", "random"]:
        m, s = stats[k]
        print(f"    {k:>14} | {m:7.3f} | {s:6.3f}")
    band_ok = stats["genie_greedy"][0] >= stats["norm_topM"][0] >= stats["random"][0]
    all_pass &= band_ok
    gap = 100 * (stats["genie_greedy"][0] - stats["norm_topM"][0]) / stats["genie_greedy"][0]
    print(f"    ordering genie >= norm >= random: {'PASS' if band_ok else 'FAIL'}")
    print(f"    genie beats the norm-heuristic by {gap:.1f}% -> headroom for a smarter selector.")

    # ---------------------------------------------------------------- C. CSI aging
    print("\n[C] CSI-aging demo (rho=0.9): select+precode on a Delta-slot-stale channel, score on truth.")
    print("    Shows realized sum-rate decaying toward 'random' as CSI ages -> motivates the belief.")
    fresh, randfloor = stats["genie_greedy"][0], stats["random"][0]
    print(f"    {'Delta':>5} | {'stale sum-rate':>14} | {'% of fresh genie':>16}")
    prev = np.inf
    decay_ok = True
    for Delta in [0, 1, 2, 5, 10, 20]:
        vals = []
        for _ in range(300):
            sim.reset()
            traj = sim.generate(Delta + 1)                    # traj[0] = stale, traj[Delta] = true
            h_stale, h_true = traj[0], traj[Delta]
            S = select_norm_topM(h_stale, M)                  # decide on stale info
            vals.append(evaluate(S, h_score=h_true, h_build=h_stale, sigma2=sigma2, P=P))
        v = np.mean(vals)
        print(f"    {Delta:5d} | {v:14.3f} | {100 * v / fresh:15.1f}%")
        if v > prev + 1e-6:
            decay_ok = False
        prev = v
    all_pass &= decay_ok
    print(f"    monotone decay with staleness: {'PASS' if decay_ok else 'FAIL'}")
    print(f"    (fresh genie = {fresh:.2f}, random floor = {randfloor:.2f} bits/s/Hz)")
    print("    NOTE: even Delta=1 is catastrophic because at 30 dB the precoder built on a 10%-")
    print("    decorrelated channel leaves residual interference that dwarfs the noise (high-SNR")
    print("    CSI-error floor). This is precisely what belief-tracking + robust-MMSE must fix.")

    print("\n" + "=" * 70)
    print(f"STEP 2 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
