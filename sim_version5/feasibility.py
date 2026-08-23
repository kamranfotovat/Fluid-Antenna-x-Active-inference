r"""
Feasibility for the liquid-column FAS (sim_version5): per-slot legal moves.

Two constraints on a droplet configuration i = (p_0,...,p_{N_t-1}):
  * REACHABILITY -- each droplet moves at most Delta_max ports from its previous height:
        |i_c(t) - i_c(t-1)| <= Delta_max
  * MIN-SPACING  -- no two ACTIVE droplets closer than d_min (lambda/2). With lambda/3 columns
    only ADJACENT columns can violate it (they need >= ~4 ports vertical sep); 2-columns-apart
    (0.667 lambda) never binds. Enforced generically here by pairwise Euclidean distance on the
    true port positions, so it stays correct for any geometry.

The coordinate-descent selector (V5-3) asks, for one column at a time, which heights are legal
given the previous config (reachability) and the other columns' current heights (spacing).
"""

from __future__ import annotations

import numpy as np

from columns import pos_to_ports

_TOL = 1e-9


def reachable_heights(p_prev, delta_max, N_p):
    """Heights column c may move to from p_prev this slot: {p : |p - p_prev| <= Delta_max}."""
    lo, hi = max(0, p_prev - delta_max), min(N_p - 1, p_prev + delta_max)
    return np.arange(lo, hi + 1)


def _dist(pa, pb):
    return np.hypot(pa[0] - pb[0], pa[1] - pb[1])


def config_feasible(i, op, positions=None):
    """True iff all pairs of active droplets are >= d_min apart."""
    pos = op.positions() if positions is None else positions
    S = pos_to_ports(i, op.N_t)
    P = pos[list(S)]
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            if _dist(P[a], P[b]) < op.d_min - _TOL:
                return False
    return True


def legal_heights(c, i, i_prev, op, positions=None, delta_max=None):
    """Heights column c can occupy: reachable from i_prev[c] AND keeping the WHOLE config feasible
    given every other column fixed at i[.]. Always non-empty when called from a feasible i with
    the column held at its current height (staying is legal) -- see I5.

    delta_max overrides op.delta_max (e.g. N_p for the first slot, where reach is unconstrained)."""
    pos = op.positions() if positions is None else positions
    N_t = op.N_t
    dmax = op.delta_max if delta_max is None else delta_max
    # positions of the OTHER droplets (fixed)
    others = [pos[h * N_t + cc] for cc, h in enumerate(i) if cc != c]
    out = []
    for p in reachable_heights(i_prev[c], dmax, op.N_p):
        pc = pos[p * N_t + c]
        if all(_dist(pc, o) >= op.d_min - _TOL for o in others):
            out.append(int(p))
    return np.array(out, dtype=int)


def random_feasible_config(op, rng, i_prev=None, max_tries=2000):
    """A random configuration that is feasible and (if i_prev given) reachable, built column by
    column so it always succeeds when a legal set exists."""
    N_t, N_p = op.N_t, op.N_p
    pos = op.positions()
    for _ in range(max_tries):
        i = np.full(N_t, -1, dtype=int)
        ok = True
        order = rng.permutation(N_t)
        for c in order:
            if i_prev is None:
                cand = np.arange(N_p)
            else:
                cand = reachable_heights(i_prev[c], op.delta_max, N_p)
            placed = [pos[h * N_t + cc] for cc, h in enumerate(i) if h >= 0]
            legal = [p for p in cand
                     if all(_dist(pos[p * N_t + c], o) >= op.d_min - _TOL for o in placed)]
            if not legal:
                ok = False
                break
            i[c] = int(rng.choice(legal))
        if ok:
            return i
    raise RuntimeError("could not sample a feasible config")
