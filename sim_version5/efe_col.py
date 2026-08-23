r"""
Expected Free Energy over droplet configurations + the myopic (single-slot) selector.

G(i) = -alpha*Pragmatic(i) - beta_w*Epistemic(i) + eta_mv * sum_c |i_c - i_prev_c|

with Pragmatic/Epistemic the reused S1 EFE terms evaluated on S = pos_to_ports(i), and the
movement cost the metric generalization of the S1 switching cost (per port travelled; reduces to
the count of repositioned droplets when every move is <= 1 port).

Selection is myopic COORDINATE DESCENT: hold all columns but one fixed, pick that column's best
legal height (reachable from i_prev[c], spacing-feasible given the others), sweep to convergence.
Because rate couples all active ports through the precoder, each candidate re-evaluates the full G.
Multiple restarts guard against local minima. Horizon planning arrives in V5-6.
"""

from __future__ import annotations

import numpy as np

import efe
from columns import pos_to_ports
from feasibility import legal_heights, random_feasible_config


def movement_cost(i, i_prev, eta_mv):
    """eta_mv * sum_c |i_c - i_prev_c|  (0 on the first slot)."""
    if i_prev is None:
        return 0.0
    return float(eta_mv * np.sum(np.abs(np.asarray(i) - np.asarray(i_prev))))


def free_energy(bel, i, i_prev, op, return_terms=False):
    """G(i) = -alpha*prag - beta_w*epis + movement. Lower is better."""
    S = pos_to_ports(i, op.N_t)
    prag = efe.pragmatic_value(bel, S, sigma2=op.sigma2, P=op.P)
    epis = efe.epistemic_value(bel, S)
    mv = movement_cost(i, i_prev, op.eta_mv)
    G = -op.alpha * prag - op.beta_w * epis + mv
    if return_terms:
        return G, {"pragmatic": prag, "epistemic": epis, "movement": mv}
    return G


def _coord_descent_once(bel, i0, i_prev, op, pos, dmax, ref, max_sweeps):
    """One coordinate-descent run from init i0. ref[c] centres reachability (= i_prev[c], or the
    init itself on the first slot). Returns (i, G, g_trace) with g_trace = G after each sweep."""
    i = np.asarray(i0, dtype=int).copy()
    Gc = free_energy(bel, i, i_prev, op)
    g_trace = []
    for _ in range(max_sweeps):
        changed = False
        for c in range(op.N_t):
            L = legal_heights(c, i, ref, op, positions=pos, delta_max=dmax)
            best_p, best_g = i[c], Gc
            for p in L:
                if p == i[c]:
                    continue
                cand = i.copy(); cand[c] = p
                g = free_energy(bel, cand, i_prev, op)
                if g < best_g - 1e-12:
                    best_g, best_p = g, int(p)
            if best_p != i[c]:
                i[c] = best_p; Gc = best_g; changed = True
        g_trace.append(Gc)
        if not changed:
            break
    return i, Gc, g_trace


def select_myopic(bel, i_prev, op, rng, n_restart=3, max_sweeps=8, return_trace=False):
    """Myopic EFE selection by coordinate descent with restarts. On the first slot (i_prev is None)
    reach is unconstrained and movement cost is zero."""
    pos = op.positions()
    first = i_prev is None
    dmax = op.N_p if first else op.delta_max
    ref = (np.zeros(op.N_t, dtype=int) if first else np.asarray(i_prev, dtype=int))

    inits = []
    if not first:
        inits.append(np.asarray(i_prev, dtype=int).copy())      # "stay" as one init
    for _ in range(n_restart):
        inits.append(random_feasible_config(op, rng, i_prev=None if first else i_prev))

    best_i, best_G, best_trace = None, np.inf, None
    for i0 in inits:
        i, G, tr = _coord_descent_once(bel, i0, i_prev, op, pos, dmax, ref, max_sweeps)
        if G < best_G:
            best_i, best_G, best_trace = i, G, tr
    if return_trace:
        return best_i, best_G, best_trace
    return best_i, best_G
