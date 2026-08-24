r"""
PARTIAL SENSING on the AR(1) S1 model -- the middle ground between observe- and predict-then-precode.

Each slot the agent SERVES M active ports (usual EFE selection) but PILOTS only m_sense <= M of them
fresh (a realistic limited-pilot budget). The remaining M - m_sense served ports are precoded from the
belief -- their CSI comes from (a) their own aged measurement from a previous slot and (b) R-INFERENCE
from this slot's freshly-sensed ports (the Kalman update propagates through the spatial correlation).
Precode all M; score realized rate on the TRUE channel over all M served ports.

m_sense = M  -> observe-then-precode (all fresh).      m_sense = 0 -> predict-then-precode (none fresh).
Intermediate m_sense is where R-inference of the un-sensed served ports becomes load-bearing -> exactly
where learning R would pay. Which ports to pilot: the m_sense most-uncertain active ports.
"""

from __future__ import annotations

import numpy as np

from agent import _obs, _switch_count
from precoding import sinr_and_rates


def run_aif_partial(agent, H, sigma_e2, rng, m_sense):
    """Observe(partial)-then-precode: select M, pilot m_sense most-uncertain of them, update, precode all M."""
    T, K, N = H.shape
    agent.reset()
    rate = np.zeros(T); switch = np.zeros(T)
    for t in range(T):
        S = agent.select(first=(t == 0))                 # M served ports (predicted belief)
        idx = list(S)
        if m_sense >= len(idx):
            S_sense = list(idx)
        elif m_sense <= 0:
            S_sense = []
        else:
            var = agent.bel.port_variances()[:, idx].sum(axis=0)     # aggregate uncertainty per active port
            order = np.argsort(var)[::-1]                            # most-uncertain first
            S_sense = sorted(idx[i] for i in order[:m_sense])        # SORT so y and update agree on order
        if S_sense:
            y = _obs(H[t], S_sense, K, sigma_e2, rng)               # y ordered by S_sense (sorted)
            agent.bel.update(tuple(S_sense), y)                     # same order -> correct association
        W = agent.precoder(S)                                        # robust-MMSE on all M from belief
        rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, agent.sigma2)[1].sum())
        switch[t] = _switch_count(S, agent.S_prev)
        agent.S_prev = S
    return dict(rate=rate, switch=switch)
