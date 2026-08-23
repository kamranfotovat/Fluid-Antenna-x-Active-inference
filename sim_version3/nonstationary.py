r"""
Non-stationary spatial correlation for the active-learning-of-R study (AL-3).

The true correlation DRIFTS over time -- physically, mobility / changing scattering shifts the
angular spread, so the correlation length changes. We model g_t(d) = exp(-d / d0(t)) with d0(t)
drifting smoothly. Because the truth keeps moving, the Kalman belief can never fully settle on it,
so a STATIC assumed R stays wrong every slot (persistent penalty) -- unlike the stationary case
where the belief becomes data-dominated and the wrong-R penalty decays away. Tracking g_t(d) with a
FORGETTING estimator (dist_profile) is therefore continuously valuable -> active learning's home.

Channel: AR(1) in time with per-slot colouring L_t = chol(beta_k R_t). hermitian_sqrt is cached by
(rounded) d0 so smooth drift needs only a few eigendecompositions.
"""

from __future__ import annotations

import numpy as np

from channel import port_positions, hermitian_sqrt


def exp_R(positions, d0):
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    R = np.exp(-dist / d0)
    return 0.5 * (R + R.T)


def d0_drift(t, lo=0.2, hi=0.5, period=40):
    """Smoothly oscillating correlation length in [lo, hi]."""
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.sin(2.0 * np.pi * t / period)


def generate_nonstationary(Nx, Ny, Wx, Wy, K, rho, beta, T, seed=0,
                           d0_fn=d0_drift, R_from_d0=None):
    """Return (H (T,K,N), R_seq (list of N x N), d0_seq). Correlation drifts per slot."""
    pos = port_positions(Nx, Ny, Wx, Wy)
    N = pos.shape[0]
    beta = np.full(K, float(beta)) if np.isscalar(beta) else np.asarray(beta, float)
    rng = np.random.default_rng(seed)
    a = np.sqrt(1.0 - rho ** 2)

    Rfun = (lambda d0: exp_R(pos, d0)) if R_from_d0 is None else R_from_d0
    _cache = {}

    def colour(d0):
        key = round(float(d0), 3)
        if key not in _cache:
            R = Rfun(key)
            _cache[key] = ([hermitian_sqrt(beta[k] * R) for k in range(K)], R)
        return _cache[key]

    def cn(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

    H = np.empty((T, K, N), dtype=complex)
    R_seq, d0_seq = [], []
    L0, R0 = colour(d0_fn(0))
    z = cn((K, N))
    h = np.stack([L0[k] @ z[k] for k in range(K)], axis=0)      # stationary init at R(0)
    H[0] = h; R_seq.append(R0); d0_seq.append(d0_fn(0))
    for t in range(1, T):
        Lt, Rt = colour(d0_fn(t))
        z = cn((K, N))
        e = np.stack([Lt[k] @ z[k] for k in range(K)], axis=0)
        h = rho * h + a * e
        H[t] = h; R_seq.append(Rt); d0_seq.append(d0_fn(t))
    return H, R_seq, d0_seq


def run_aif_track(agent, H, R_seq, sigma_e2, rng, sense_first=True):
    """Oracle-tracking baseline: the agent's belief adopts the TRUE R_t each slot (knows the drift)."""
    from agent import _obs, _switch_count
    from precoding import sinr_and_rates
    T, K, N = H.shape
    agent.reset()
    eye = np.eye(N)
    rate = np.zeros(T); switch = np.zeros(T)
    for t in range(T):
        agent.bel.C_stat = [agent.bel.beta[k] * R_seq[t] + agent.bel.reg * eye for k in range(K)]
        S = agent.select(first=(t == 0))
        idx = list(S)
        y = _obs(H[t], idx, K, sigma_e2, rng)
        if sense_first:
            agent.bel.update(S, y)
        W = agent.precoder(S)
        rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, agent.sigma2)[1].sum())
        switch[t] = _switch_count(S, agent.S_prev)
        if sense_first:
            agent.S_prev = S
        else:
            agent.update(S, y)
    return dict(rate=rate, switch=switch)
