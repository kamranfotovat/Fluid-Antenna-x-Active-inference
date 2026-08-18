"""
Step 2 -- Reference port-selection baselines (full CSI).

These establish the band the AIF agent will live in. A "strategy" picks a set S of M ports
out of N; we then precode on those ports and score the realized sum-rate.

Strategies
----------
  genie_exhaustive : try every C(N, M) subset, keep the best -> true upper bound (small N only).
  genie_greedy     : start empty, repeatedly add the port with the largest marginal sum-rate
                     -> O(N*M) precoder solves, empirically near-optimal.
  norm_topM        : pick the M ports with the largest aggregate power sum_k |h_k[n]|^2
                     -> the classic full-CSI heuristic (Paper 3 style), ignores interference geometry.
  random           : M random ports -> lower reference.

All strategies use the SAME precoder (default MMSE) and the SAME scorer so the comparison is fair.
`evaluate` also lets us build the precoder from one channel (e.g. stale/estimated) and score it on
another (the true channel) -- used for the CSI-aging demo.
"""

from __future__ import annotations

import itertools
import numpy as np

from precoding import mmse_precoder, sum_rate
from channel import feasible_ports


def _active(h, S):
    """h: (K, N) -> (M, K) active-port channel (columns = users)."""
    return h[:, list(S)].T


def evaluate(S, h_score, h_build=None, precoder=mmse_precoder, sigma2=1e-3, P=1.0):
    """Sum-rate of port set S. Precoder is built from h_build (default = h_score) on the
    ports S and scored on h_score. Set h_build to a stale/estimated channel to model
    imperfect CSI."""
    h_build = h_score if h_build is None else h_build
    W = precoder(_active(h_build, S), P=P, sigma2=sigma2)
    return sum_rate(_active(h_score, S), W, sigma2)


# --------------------------------------------------------------------------- strategies
def select_norm_topM(h, M):
    """Top-M ports by aggregate channel power across users (full-CSI heuristic)."""
    power = np.sum(np.abs(h) ** 2, axis=0)                     # (N,)
    return tuple(np.sort(np.argsort(power)[-M:]))


def select_random(h, M, rng):
    N = h.shape[1]
    return tuple(np.sort(rng.choice(N, size=M, replace=False)))


def select_topM_feasible(power, M, positions=None, d_min=None):
    """Highest-power ports subject to the >= d_min min-spacing constraint (greedy by
    descending power). Falls back to unconstrained top-up if the constraint can't yield M."""
    order = [int(n) for n in np.argsort(power)[::-1]]
    if positions is None or d_min is None:
        return tuple(sorted(order[:M]))
    S = []
    for n in order:
        if len(S) == M:
            break
        if feasible_ports(positions, S, [n], d_min):
            S.append(n)
    for n in order:                                   # relax to reach M if constraint too tight
        if len(S) == M:
            break
        if n not in S:
            S.append(n)
    return tuple(sorted(S))


def select_random_feasible(N, M, rng, positions=None, d_min=None):
    """Random M-subset respecting >= d_min (add ports in a random order if feasible)."""
    if positions is None or d_min is None:
        return tuple(np.sort(rng.choice(N, size=M, replace=False)))
    order = [int(n) for n in rng.permutation(N)]
    S = []
    for n in order:
        if len(S) == M:
            break
        if feasible_ports(positions, S, [n], d_min):
            S.append(n)
    for n in order:
        if len(S) == M:
            break
        if n not in S:
            S.append(n)
    return tuple(sorted(S))


def select_greedy(h, M, precoder=mmse_precoder, sigma2=1e-3, P=1.0,
                  positions=None, d_min=None):
    """Greedily add the port giving the largest marginal sum-rate (MMSE handles |S|<K).

    positions + d_min impose the same hardware min-spacing constraint as the AIF selector,
    so the full-CSI genie is a FEASIBLE upper bound (it may not pack ports closer than
    d_min either). If the constraint empties the pool early, relax that pick to keep |S|=M.
    """
    N = h.shape[1]
    S = []
    remaining = set(range(N))
    best_val = -np.inf
    for _ in range(M):
        pool = feasible_ports(positions, S, remaining, d_min) if positions is not None else list(remaining)
        if not pool:
            pool = list(remaining)
        best_n, best_val = None, -np.inf
        for n in pool:
            val = evaluate(S + [n], h, precoder=precoder, sigma2=sigma2, P=P)
            if val > best_val:
                best_val, best_n = val, n
        S.append(best_n)
        remaining.remove(best_n)
    return tuple(sorted(S)), best_val


def select_exhaustive(h, M, precoder=mmse_precoder, sigma2=1e-3, P=1.0):
    """Brute force over all C(N, M) subsets -> true optimum (use only for small N)."""
    N = h.shape[1]
    best_S, best_val = None, -np.inf
    for S in itertools.combinations(range(N), M):
        val = evaluate(S, h, precoder=precoder, sigma2=sigma2, P=P)
        if val > best_val:
            best_val, best_S = val, S
    return best_S, best_val
