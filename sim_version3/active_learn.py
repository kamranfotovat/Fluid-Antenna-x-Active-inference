r"""
Online self-calibrating AIF agent for S1 -- learns g(d) from its own measurements, and (AL-2) can
learn ACTIVELY by adding a model-epistemic (novelty) term to the selection so it deliberately
co-observes the under-sampled near-field.

Unified selection objective (maximize):
    J(S) = alpha*Pragmatic + beta_w*Epistemic + lam_model*Novelty - Switching
Novelty(S) rewards co-observing under-sampled distance bins:
    novelty marginal of adding port n to A  =  sum_{a in A} 1/sqrt(1 + count[bin(n,a)])
so pairing a new port with an existing one in a starved bin (the short distances the comm policy
never visits) scores high -> the agent occasionally CLUSTERS ports to learn g(d) at small d,
trading a little diversity/rate now for a better belief later. lam_model is the exploit/learn knob.

Requires d_min OFF (else min-spacing forbids the short pairs active learning needs).
"""

from __future__ import annotations

import numpy as np

from agent import _obs, _switch_count
from precoding import sinr_and_rates
from efe import pragmatic_value, epistemic_value, _switch_marginal
from channel import feasible_ports
from selection import select_random_feasible
from dist_profile import DistanceProfileEstimator


def _adopt_R(agent, R_hat):
    bel = agent.bel
    eye = np.eye(bel.N)
    bel.C_stat = [bel.beta[k] * R_hat + bel.reg * eye for k in range(bel.K)]


def greedy_select_active(bel, M, estimator, lam_model, S_prev=None, alpha=1.0, beta=1.0,
                         eta_sw=1.0, e_sw=1.0, sigma2=1e-3, P=1.0, positions=None, d_min=None):
    """Greedy selection with the added model-novelty term (AL-2)."""
    N = bel.N
    B, mc = estimator.B, estimator.mc
    prev_set = set() if S_prev is None else set(S_prev)
    A, remaining = [], set(range(N))
    prag_A, epis_A = 0.0, 0.0
    local = np.zeros(estimator.n_bins)                 # within-slot pair counts (submodular novelty)

    def w(b):
        return 1.0 / np.sqrt(1.0 + mc[b] + local[b])

    for _ in range(M):
        pool = feasible_ports(positions, A, remaining, d_min) if positions is not None else list(remaining)
        if not pool:
            pool = list(remaining)
        best = (-np.inf, None, None, None)
        for n in pool:
            cand = tuple(A + [n])
            prag_c = pragmatic_value(bel, cand, sigma2, P)
            epis_c = epistemic_value(bel, cand)
            nov = sum(w(B[n, a]) for a in A)           # novelty marginal (0 for the first pick)
            marg = (alpha * (prag_c - prag_A) + beta * (epis_c - epis_A)
                    + lam_model * nov - _switch_marginal(n, prev_set, eta_sw, e_sw))
            if np.isfinite(marg) and marg > best[0]:
                best = (marg, n, prag_c, epis_c)
        if best[1] is None:
            n_star = min(pool)
            best = (0.0, n_star, pragmatic_value(bel, tuple(A + [n_star]), sigma2, P),
                    epistemic_value(bel, tuple(A + [n_star])))
        _, n_star, prag_A, epis_A = best
        for a in A:
            local[B[n_star, a]] += bel.K               # book the co-observations this pick creates
        A.append(n_star); remaining.remove(n_star)
    return tuple(sorted(A))


def run_aif_learn(agent, H, sigma_e2, rng, estimator: DistanceProfileEstimator,
                  relearn_every=5, sense_first=True, true_g=None,
                  active=False, lam_model=0.0):
    """Closed-loop self-calibrating agent. active=True adds the novelty term (AL-2)."""
    T, K, N = H.shape
    agent.reset()
    rate = np.zeros(T); switch = np.zeros(T)
    gerr = []
    for t in range(T):
        if active:
            if t > 0:
                agent.bel.predict()
            S = greedy_select_active(agent.bel, agent.M, estimator, lam_model,
                                     S_prev=agent.S_prev, alpha=agent.alpha, beta=agent.beta_w,
                                     eta_sw=agent.eta_sw, e_sw=agent.e_sw, sigma2=agent.sigma2,
                                     P=agent.P, positions=agent.positions, d_min=agent.d_min)
        else:
            S = agent.select(first=(t == 0))
        idx = list(S)
        y = _obs(H[t], idx, K, sigma_e2, rng)
        if sense_first:
            agent.bel.update(S, y)
        estimator.update(idx, y)
        W = agent.precoder(S)
        rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, agent.sigma2)[1].sum())
        switch[t] = _switch_count(S, agent.S_prev)
        if sense_first:
            agent.S_prev = S
        else:
            agent.update(S, y)
        if (t + 1) % relearn_every == 0:
            _adopt_R(agent, estimator.R_hat())
            if true_g is not None:
                gerr.append((t, estimator.g_rmse(true_g)))
    return dict(rate=rate, switch=switch, gerr=gerr, R_hat=estimator.R_hat())


def run_random_probe(agent, H, sigma_e2, rng, estimator: DistanceProfileEstimator,
                     relearn_every=5, p_probe=0.3):
    """Baseline: UNDIRECTED probing -- with prob p_probe each slot, select a random feasible set
    (blind exploration) instead of the comm selection. Random ports are spread across the aperture,
    so it mostly co-observes LONG distances -> still starves the near-field (the contrast to active)."""
    T, K, N = H.shape
    agent.reset()
    rate = np.zeros(T); switch = np.zeros(T)
    for t in range(T):
        if rng.random() < p_probe:
            if t > 0:
                agent.bel.predict()
            S = select_random_feasible(N, agent.M, rng, agent.positions, agent.d_min)
        else:
            S = agent.select(first=(t == 0))
        idx = list(S)
        y = _obs(H[t], idx, K, sigma_e2, rng)
        agent.bel.update(S, y)
        estimator.update(idx, y)
        W = agent.precoder(S)
        rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, agent.sigma2)[1].sum())
        switch[t] = _switch_count(S, agent.S_prev)
        agent.S_prev = S
        if (t + 1) % relearn_every == 0:
            _adopt_R(agent, estimator.R_hat())
    return dict(rate=rate, switch=switch, R_hat=estimator.R_hat())
