r"""
LIQUID-COLUMN GEOMETRY TRADE STUDY (sim_version5, Paper 2).

Goal: pick (N_t columns, N_p ports/column, column length L in lambda) and a physically
meaningful per-slot move limit Delta_max, for a liquid-metal FAS where each column holds ONE
droplet that slides along its 1-D tube.

We keep the S1 port pitch delta = 0.1 lambda (= 2 lambda / 20, the OP_V3 grid) so results are
comparable, and study one column in isolation (columns are independent, spaced > lambda/2).

Physics anchors (Jakes/Clarke, matching channel.spatial_correlation):
  within-column correlation  R(d) = J0(2*pi*d),  d in wavelengths
  first null of J0 at d ~ 0.383 lambda  -> the "grain" (decorrelation distance)

Two questions this script answers with numbers:
  Q1 (geometry): how does the best-port SNR / rate a column can offer grow with its length L?
                 -> diminishing returns set N_p, L.
  Q2 (Delta_max): how many PORTS/slot does the optimal droplet position drift under AR(1) rho?
                 -> Delta_max must be >= that to track, but small enough that crossing the column
                    takes several slots (so horizon planning matters). We report the drift
                    distribution and the "slots to cross" for candidate Delta_max.

Run:  python geometry_study.py
"""

from __future__ import annotations

import sys
import numpy as np
from scipy.special import j0

sys.stdout.reconfigure(encoding="utf-8")

PITCH = 0.1                     # port spacing in wavelengths (= S1 OP_V3 grid)
RHO = 0.9                       # AR(1) temporal correlation (OP_V3)
K = 3                           # users
BETA = np.array([1.0, 0.7, 1.3])
SIGMA2 = 0.03                   # 15 dB
T = 400                         # slots for drift statistics
SEED = 0
GRAIN = 0.3827                  # first null of J0 / lambda


def col_corr(N_p, pitch=PITCH):
    """1-D Jakes correlation for a single column of N_p ports at `pitch` spacing."""
    pos = np.arange(N_p) * pitch
    d = np.abs(pos[:, None] - pos[None, :])
    R = j0(2.0 * np.pi * d)
    return 0.5 * (R + R.T)


def herm_sqrt(A):
    w, V = np.linalg.eigh(0.5 * (A + A.conj().T))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.conj().T


def sim_column(N_p, T, rng):
    """AR(1) Jakes channel for ONE column, K users: h[t] in C^{K, N_p}."""
    R = col_corr(N_p)
    Lstat = [herm_sqrt(BETA[k] * R) for k in range(K)]
    a = np.sqrt(1.0 - RHO ** 2)
    def draw(L):
        z = (rng.standard_normal((K, N_p)) + 1j * rng.standard_normal((K, N_p))) / np.sqrt(2)
        return np.stack([z[k] @ L[k].conj().T for k in range(K)], axis=0)
    h = draw(Lstat)
    H = np.empty((T, K, N_p), dtype=complex)
    H[0] = h
    for t in range(1, T):
        innov = draw(Lstat)
        h = RHO * h + a * innov
        H[t] = h
    return H


def q1_length(rng):
    """Best-single-port channel power a column offers vs its length (diminishing returns)."""
    print("=" * 68)
    print("Q1  COLUMN LENGTH -> best-port gain (single-user proxy, user 0, beta=1)")
    print("=" * 68)
    print(f"{'N_p':>4} | {'L (lambda)':>10} | {'grains':>6} | {'E[max|h|^2]':>11} | {'gain vs N_p=1':>13}")
    print("-" * 60)
    base = None
    for N_p in (1, 6, 11, 21, 31, 41, 61, 81):
        H = sim_column(N_p, 200, np.random.default_rng(1))
        p = np.abs(H[:, 0, :]) ** 2               # (T, N_p) user-0 port powers
        emax = float(p.max(axis=1).mean())        # E[max over ports |h|^2]
        L = (N_p - 1) * PITCH
        if base is None:
            base = emax
        print(f"{N_p:>4} | {L:>10.1f} | {L/GRAIN:>6.1f} | {emax:>11.3f} | {emax/base:>12.2f}x")


def q2_drift(N_p, rng):
    """How far (in ports) does the best-port position move per slot? -> calibrate Delta_max."""
    H = sim_column(N_p, T, rng)
    # per-slot optimal position = argmax of the aggregate |h|^2 across users (a rate proxy)
    agg = np.sum(np.abs(H) ** 2, axis=1)          # (T, N_p)
    best = np.argmax(agg, axis=1)                 # (T,)
    drift = np.abs(np.diff(best))                 # ports moved by the optimum per slot
    return best, drift


def main():
    rng = np.random.default_rng(SEED)
    q1_length(rng)

    print("\n" + "=" * 68)
    print("Q2  OPTIMUM DRIFT / slot  ->  calibrate Delta_max   (N_p=41, L=4 lambda, rho=0.9)")
    print("=" * 68)
    N_p = 41
    best, drift = q2_drift(N_p, np.random.default_rng(7))
    pct = np.percentile(drift, [50, 75, 90, 95])
    print(f"optimal-port drift per slot (ports):  mean={drift.mean():.2f}  "
          f"median={pct[0]:.0f}  p75={pct[1]:.0f}  p90={pct[2]:.0f}  p95={pct[3]:.0f}  max={drift.max():.0f}")
    print(f"(port pitch {PITCH} lambda -> mean drift {drift.mean()*PITCH:.3f} lambda/slot; "
          f"one grain = {GRAIN:.3f} lambda = {GRAIN/PITCH:.0f} ports)")

    print("\n  Delta_max candidates (ports/slot):")
    print(f"  {'Dmax':>5} | {'lambda/slot':>11} | {'slots to cross col':>18} | {'tracks optimum?':>16}")
    print("  " + "-" * 60)
    for dmax in (1, 2, 3, 4, 6, 41):
        cross = (N_p - 1) / dmax
        tracks = f"{100*np.mean(drift <= dmax):.0f}% of slots"
        note = "  (teleport)" if dmax >= N_p else ""
        print(f"  {dmax:>5} | {dmax*PITCH:>11.2f} | {cross:>18.0f} | {tracks:>16}{note}")

    # How much rate does a myopic Delta_max-limited tracker lose vs an unconstrained jumper?
    print("\n  Rate retained by a GREEDY move-limited tracker vs unconstrained best port:")
    print("  (single column, user-0 proxy: follow argmax within +/-Dmax of current position)")
    H = sim_column(N_p, T, np.random.default_rng(11))
    agg = np.sum(np.abs(H) ** 2, axis=1)
    unc = np.log2(1 + agg.max(axis=1) / SIGMA2).mean()      # unconstrained: jump anywhere
    print(f"  {'Dmax':>5} | {'rate (b/s/Hz)':>13} | {'% of unconstrained':>18} | {'moves/slot':>11}")
    print("  " + "-" * 58)
    for dmax in (1, 2, 3, 4, 6, 41):
        pos = int(np.argmax(agg[0])); rate = 0.0; moves = 0.0
        for t in range(T):
            lo, hi = max(0, pos - dmax), min(N_p, pos + dmax + 1)
            local = int(lo + np.argmax(agg[t, lo:hi]))
            moves += abs(local - pos); pos = local
            rate += np.log2(1 + agg[t, pos] / SIGMA2)
        rate /= T; moves /= T
        print(f"  {dmax:>5} | {rate:>13.3f} | {100*rate/unc:>17.1f}% | {moves:>11.2f}")


if __name__ == "__main__":
    main()
