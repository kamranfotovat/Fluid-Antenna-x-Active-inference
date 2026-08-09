r"""
Bandit baseline for FAS port selection -- the model-FREE learning competitor
(cf. Zou, Sun, Wang, "Online Learning-Induced Port Selection for FAS", IEEE WCL 2024).

That paper casts port selection without full CSI as a multi-armed bandit. Here we implement a
combinatorial UCB (CUCB) version in OUR setting (BS activates M of N ports, observe-then-precode,
switching cost, dynamic channel): each port is an arm; the agent keeps a per-port quality estimate,
picks the top-M by an upper-confidence-bound score, observes those ports (semi-bandit feedback),
and updates. Crucially it is MODEL-FREE: it does NOT use the spatial correlation R to infer
un-measured ports -- so it must EXPLORE each port to learn it, and cannot fill in the ports it
never activates. This is exactly the axis where our model-based active-inference belief wins:
zero-shot inference from R + rho vs. learn-by-sampling.
"""

from __future__ import annotations

import numpy as np

from precoding import mmse_precoder, sinr_and_rates


def run_bandit(H, M, sigma_e2, rng, sigma2=0.03, eta_sw=1.0, e_sw=1.0, c=1.0, P=1.0):
    """Combinatorial-UCB port selection, observe-then-precode. Returns per-slot rate & switching.

    c : UCB exploration constant. Per-port quality = running mean of observed aggregate power
        (what the agent can actually measure); unseen ports get an optimistic (infinite) score so
        they are explored first.
    """
    T, K, N = H.shape
    cnt = np.zeros(N)
    val = np.zeros(N)                              # running-mean per-port observed quality
    rate = np.zeros(T); switch = np.zeros(T); S_prev = None
    for t in range(T):
        bonus = np.where(cnt > 0, c * np.sqrt(np.log(t + 2.0) / np.maximum(cnt, 1.0)), np.inf)
        ucb = val + bonus                          # unseen ports -> inf -> forced exploration
        S = tuple(sorted(np.argpartition(ucb, -M)[-M:]))
        idx = list(S)
        y = H[t][:, idx] + np.sqrt(sigma_e2 / 2) * (rng.standard_normal((K, len(idx)))
                                                    + 1j * rng.standard_normal((K, len(idx))))
        W = mmse_precoder(y.T, P=P, sigma2=sigma2)
        rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, sigma2)[1].sum())
        switch[t] = 0 if S_prev is None else len(set(idx) ^ set(S_prev)); S_prev = idx
        # semi-bandit update: observed aggregate power per activated port
        for p, n in enumerate(idx):
            q = float(np.sum(np.abs(y[:, p]) ** 2) / K)
            cnt[n] += 1.0
            val[n] += (q - val[n]) / cnt[n]
    return dict(rate=rate, switch=switch)
