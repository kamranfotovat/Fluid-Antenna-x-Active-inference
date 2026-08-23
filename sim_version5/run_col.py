r"""
Closed-loop runners for the liquid-column FAS (sim_version5), all on a shared channel trajectory
H (T, K, N) for paired comparison.

  run_col_aif    : the active-inference column agent (predict -> myopic EFE select -> sense ->
                   update -> robust-MMSE precode).  Observe-then-precode (sense-first).
  run_col_genie  : full-CSI rate-maximizing feasible config each slot (no Delta_max, no movement
                   cost) -> realized-rate CEILING.
  run_col_naive  : partial-CSI, NO active inference -- memoryless per-port point estimate, power-
                   greedy feasible selection, precode on fresh pilots.  The baseline to beat.
  run_col_random : random feasible & reachable config, observe-then-precode -> lower bound.

All respect the one-droplet-per-column structure and the min-spacing constraint; the AIF/naive/
random agents also respect Delta_max reachability.  Realized rate is always scored on the TRUE
active channel.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace

import efe
from precoding import mmse_precoder, sinr_and_rates
from belief import KalmanBelief
from columns import pos_to_ports, sense
from feasibility import reachable_heights, random_feasible_config
from efe_col import select_myopic, movement_cost
from planner import plan_horizon
from agent_col import make_belief


def _rate(h_t, S, W, sigma2):
    return float(sinr_and_rates(h_t[:, list(S)].T, W, sigma2)[1].sum())


def _feasible_reachable(i, i_prev, op, pos):
    """Assert config i is min-spacing feasible and (if i_prev) within Delta_max of it."""
    from feasibility import config_feasible
    okf = config_feasible(i, op, pos)
    okr = True if i_prev is None else bool(np.all(np.abs(i - i_prev) <= op.delta_max))
    return okf and okr


# --------------------------------------------------------------------------- AIF column agent
def run_col_aif(op, H, rng, beta_w=None, R=None, track=False, sense_first=True,
                horizon=None, gamma=0.9, n_sweeps=2):
    """Closed-loop myopic AIF agent. beta_w/R override the operating point (for ablations).

    sense_first (protocol):
      True  (observe-then-precode) -- sense the active ports, Kalman-update, THEN precode from the
            fresh belief. Active-port CSI is fresh -> the correlation model barely affects rate.
      False (predict-then-precode) -- precode from the PREDICTED (aged/inferred) belief BEFORE
            sensing, then sense+update for next slot. Precoding quality now depends on the belief's
            inference, so cross-column correlation directly affects rate (the regime that tests B vs A).
    """
    opw = op if beta_w is None else replace(op, beta_w=beta_w)
    T, K, N = H.shape
    pos = op.positions()
    bel = make_belief(op, R=R)
    i_prev = None
    rate = np.zeros(T); move = np.zeros(T); info = np.zeros(T); feas = True
    for t in range(T):
        if t > 0:
            bel.predict()
        if horizon is None:
            i, _ = select_myopic(bel, i_prev, opw, rng)
        else:
            i = plan_horizon(bel, i_prev, opw, H=horizon, gamma=gamma, n_sweeps=n_sweeps, rng=rng)
        S = pos_to_ports(i, op.N_t)
        feas &= _feasible_reachable(i, i_prev, op, pos)
        info[t] = efe.epistemic_value(bel, S)             # info the measurement will carry
        if sense_first:
            y = sense(H[t], S, op.sigma_e2, rng)
            bel.update(S, y)                              # fresh belief before precoding
            W, _, _ = efe.robust_mmse_from_belief(bel, S, op.sigma2, op.P)
            rate[t] = _rate(H[t], S, W, op.sigma2)
        else:
            W, _, _ = efe.robust_mmse_from_belief(bel, S, op.sigma2, op.P)  # from PREDICTED belief
            rate[t] = _rate(H[t], S, W, op.sigma2)
            y = sense(H[t], S, op.sigma_e2, rng)
            bel.update(S, y)                              # update for next slot
        move[t] = movement_cost(i, i_prev, op.eta_mv)
        i_prev = i
    out = dict(rate=rate, move=move, info=info, feasible=feas)
    return out


# --------------------------------------------------------------------------- genie ceiling
def _perfect_belief(op, h_t):
    bel = make_belief(op)
    bel.mu = h_t.astype(complex).copy()
    for k in range(op.K):
        bel.Sigma[k] = 1e-9 * np.eye(op.N)
    return bel


def run_col_genie(op, H, rng):
    """Full-CSI, rate-max feasible config each slot (unconstrained reach, no movement cost)."""
    opg = replace(op, beta_w=0.0, eta_mv=0.0)
    T = H.shape[0]
    rate = np.zeros(T)
    for t in range(T):
        bel = _perfect_belief(op, H[t])
        i, _ = select_myopic(bel, None, opg, rng, n_restart=4)   # i_prev=None -> unconstrained
        S = pos_to_ports(i, op.N_t)
        W = mmse_precoder(H[t][:, list(S)].T, P=op.P, sigma2=op.sigma2)
        rate[t] = _rate(H[t], S, W, op.sigma2)
    return dict(rate=rate)


# --------------------------------------------------------------------------- naive (no AIF)
def _select_power_greedy(power, i_prev, op, pos, rng):
    """Sequential feasible selection by last-known port power (memoryless, no correlation model)."""
    order = rng.permutation(op.N_t)
    i = np.full(op.N_t, -1, dtype=int); placed = []
    for c in order:
        if i_prev is None:
            cand = np.arange(op.N_p)
        else:
            cand = reachable_heights(i_prev[c], op.delta_max, op.N_p)
        legal = [int(p) for p in cand
                 if all(np.hypot(*(pos[p * op.N_t + c] - q)) >= op.d_min - 1e-9 for q in placed)]
        if not legal:
            legal = [int(i_prev[c])] if i_prev is not None else [0]
        best = max(legal, key=lambda p: power[p * op.N_t + c])
        i[c] = best; placed.append(pos[best * op.N_t + c])
    return i


def run_col_naive(op, H, rng):
    T, K, N = H.shape
    pos = op.positions()
    est = np.zeros((K, N), dtype=complex)
    rate = np.zeros(T); move = np.zeros(T); i_prev = None
    for t in range(T):
        power = np.sum(np.abs(est) ** 2, axis=0)
        i = _select_power_greedy(power, i_prev, op, pos, rng)
        S = pos_to_ports(i, op.N_t)
        y = sense(H[t], S, op.sigma_e2, rng)
        est[:, list(S)] = y                                # raw held estimate (no filtering)
        W = mmse_precoder(y.T, P=op.P, sigma2=op.sigma2)
        rate[t] = _rate(H[t], S, W, op.sigma2)
        move[t] = movement_cost(i, i_prev, op.eta_mv)
        i_prev = i
    return dict(rate=rate, move=move)


# --------------------------------------------------------------------------- random lower bound
def run_col_random(op, H, rng):
    T = H.shape[0]
    rate = np.zeros(T); move = np.zeros(T); i_prev = None
    for t in range(T):
        i = random_feasible_config(op, rng, i_prev=i_prev)
        S = pos_to_ports(i, op.N_t)
        y = sense(H[t], S, op.sigma_e2, rng)
        W = mmse_precoder(y.T, P=op.P, sigma2=op.sigma2)
        rate[t] = _rate(H[t], S, W, op.sigma2)
        move[t] = movement_cost(i, i_prev, op.eta_mv)
        i_prev = i
    return dict(rate=rate, move=move)
