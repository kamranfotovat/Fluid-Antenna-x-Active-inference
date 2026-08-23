r"""
Receding-horizon planner for the liquid-column FAS (sim_version5, V5-6).

Plans a droplet-trajectory over a horizon H that minimizes discounted cumulative Expected Free
Energy, then executes only the first step and re-plans next slot (MPC / receding horizon).

Tractability: the joint problem factorizes into per-column 1-D VITERBI over heights (transitions
limited to Delta_max), wrapped in COORDINATE DESCENT across columns (columns couple through the
shared precoder and the min-spacing constraint). See SIMULATION_PLAN_V5.md.

Belief prediction along the plan:
  * Covariance evolves DETERMINISTICALLY given the planned sensing positions (Kalman covariance
    update is data-independent) -- so the epistemic term over the horizon is exactly predictable.
  * Mean is aged (mu_k = rho^k mu) -- in expectation future measurements carry zero innovation,
    so sensing does not shift the predicted mean, only shrinks covariance. Pragmatic/epistemic are
    scored on this predicted (pre-sensing, aged) belief per step.

NOTE (honest): under a stationary AR(1) belief the future is predictable only through uniform
aging (no spatial anticipation), so planning ~ myopic apart from avoiding wasteful movement. The
anticipatory win needs a predictable spatial dynamic (drift-aware belief, next layer).
"""

from __future__ import annotations

import copy
import numpy as np

import efe
from columns import pos_to_ports
from feasibility import random_feasible_config

_INF = 1e18


class _Snap:
    """Lightweight belief snapshot for scoring (exposes what the EFE terms read)."""
    __slots__ = ("mu", "Sigma", "K", "N", "sigma_e2")

    def __init__(self, bel, op):
        self.mu = bel.mu.copy()
        self.Sigma = np.array([S.copy() for S in bel.Sigma])
        self.K, self.N, self.sigma_e2 = op.K, op.N, bel.sigma_e2


def _working(bel):
    """Shallow copy sharing read-only R/C_stat but with private mu/Sigma (mutated by predict/update)."""
    b = copy.copy(bel)
    b.mu = bel.mu.copy()
    b.Sigma = np.array([S.copy() for S in bel.Sigma])
    return b


def _predict_sequence(bel, traj, op):
    """H belief snapshots along the planned trajectory. seq[0] is the CURRENT belief as passed (the
    outer loop already aged it for this slot); aging happens BETWEEN steps, not before step 0."""
    b = _working(bel)
    seq = []
    H = len(traj)
    for k in range(H):
        seq.append(_Snap(b, op))                                            # belief for step k
        S = pos_to_ports(traj[k], op.N_t)
        yexp = np.stack([b.mu[kk][list(S)] for kk in range(op.K)], axis=0)  # expected obs
        b.update(S, yexp)                                                   # covariance-only update
        if k < H - 1:
            b.predict()                                                     # age to step k+1
    return seq


def _spacing_ok(c, p, traj_k, op, pos):
    pc = pos[p * op.N_t + c]
    for cc in range(op.N_t):
        if cc == c:
            continue
        if np.hypot(*(pc - pos[traj_k[cc] * op.N_t + cc])) < op.d_min - 1e-9:
            return False
    return True


def _viterbi_column(c, traj, seq, i_prev, op, pos, gamma):
    """Optimal height-trajectory for column c (others fixed at traj), by Viterbi over heights."""
    H, N_p, dmax = len(seq), op.N_p, op.delta_max
    first = i_prev is None
    center = None if first else int(i_prev[c])

    def stage(k, p):
        cfg = traj[k].copy(); cfg[c] = p
        S = pos_to_ports(cfg, op.N_t)
        prag = efe.pragmatic_value(seq[k], S, sigma2=op.sigma2, P=op.P)
        epis = efe.epistemic_value(seq[k], S)
        return -op.alpha * prag - op.beta_w * epis

    feas = [[p for p in range(N_p) if _spacing_ok(c, p, traj[k], op, pos)] for k in range(H)]

    # k = 0 (movement measured from i_prev[c]; free on the first slot)
    dp = {}; back = [dict() for _ in range(H)]
    for p in feas[0]:
        if not first and abs(p - center) > dmax:
            continue
        mv = 0.0 if first else op.eta_mv * abs(p - center)
        dp[p] = stage(0, p) + mv
        back[0][p] = None
    # k = 1..H-1
    for k in range(1, H):
        ndp = {}
        for p in feas[k]:
            best, bp = _INF, None
            sc = (gamma ** k) * stage(k, p)
            for pp, val in dp.items():
                if abs(p - pp) > dmax:
                    continue
                cost = val + sc + (gamma ** k) * op.eta_mv * abs(p - pp)
                if cost < best:
                    best, bp = cost, pp
            if bp is not None:
                ndp[p] = best; back[k][p] = bp
        if not ndp:                                   # dead end -> keep staying feasible
            return traj
        dp = ndp
    # backtrack
    p_end = min(dp, key=dp.get)
    path = [p_end]
    for k in range(H - 1, 0, -1):
        path.append(back[k][path[-1]])
    path = path[::-1]
    new = traj.copy()
    for k in range(H):
        new[k][c] = path[k]
    return new


def plan_horizon(bel, i_prev, op, H=3, gamma=0.9, n_sweeps=2, rng=None, return_traj=False):
    """Receding-horizon plan; returns the FIRST config to execute (and optionally the trajectory)."""
    rng = np.random.default_rng(0) if rng is None else rng
    first = i_prev is None
    if first:
        i0 = random_feasible_config(op, rng)
    else:
        i0 = np.asarray(i_prev, dtype=int)
    traj = np.tile(i0, (H, 1))
    pos = op.positions()
    for _ in range(n_sweeps):
        for c in range(op.N_t):
            seq = _predict_sequence(bel, traj, op)     # refresh belief sequence per column (consistent scoring)
            traj = _viterbi_column(c, traj, seq, i_prev, op, pos, gamma)
    return (traj[0], traj) if return_traj else traj[0]
