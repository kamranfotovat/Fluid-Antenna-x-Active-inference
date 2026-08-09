r"""
Step 6 -- the closed-loop Active-Inference agent (perception + action together).

Per slot t the agent runs:
    1. PREDICT  belief (aging)                 bel.predict()      [skip at t=0: prior = h(0) dist]
    2. SELECT   S_t = argmin_S G(S)            efe.greedy_select
    3. ACT      robust-MMSE precode on S_t     W from belief (mu, Sigma)
    4. OBSERVE  noisy channel at S_t only      y = P_S h_true + CN(0, sigma_e^2 I)
    5. UPDATE   belief                         bel.update(S_t, y)
    6. LOG      realized rate (on the TRUE channel) + switching cost

The precoder is built from the BELIEF (the agent never sees the true channel), but the
realized rate is scored on the TRUE active channel -- that is the honest throughput.

Also provides two references that share the same channel trajectory for paired comparison:
    run_genie  : full-CSI greedy selection + full-CSI precoder  (rate upper bound)
    run_random : random selection + full-CSI precoder           (selection lower bound)
"""

from __future__ import annotations

import numpy as np

from belief import KalmanBelief
from precoding import mmse_precoder, sinr_and_rates
from selection import select_greedy
import efe


# --------------------------------------------------------------------------- the agent
class AIFAgent:
    def __init__(self, R, beta, rho, sigma_e2, M,
                 alpha=1.0, beta_w=1.0, eta_sw=1.0, e_sw=1.0, sigma2=1e-3, P=1.0):
        self.bel = KalmanBelief(R=R, beta=beta, rho=rho, sigma_e2=sigma_e2)
        self.M = M
        self.alpha, self.beta_w = alpha, beta_w
        self.eta_sw, self.e_sw = eta_sw, e_sw
        self.sigma2, self.P = sigma2, P
        self.K = self.bel.K
        self.S_prev = None

    def reset(self):
        self.bel.reset()
        self.S_prev = None
        return self

    def select(self, first: bool):
        if not first:
            self.bel.predict()                      # aging
        return efe.greedy_select(
            self.bel, self.M, S_prev=self.S_prev,
            alpha=self.alpha, beta=self.beta_w, eta_sw=self.eta_sw, e_sw=self.e_sw,
            sigma2=self.sigma2, P=self.P)

    def precoder(self, S):
        W, _, _ = efe.robust_mmse_from_belief(self.bel, S, self.sigma2, self.P)
        return W

    def update(self, S, y):
        self.bel.update(S, y)
        self.S_prev = S


# --------------------------------------------------------------------------- runners (shared trajectory)
def _switch_count(S, S_prev):
    return 0 if S_prev is None else len(set(S) ^ set(S_prev))


def run_aif(agent: AIFAgent, H, sigma_e2, rng, track_belief=False):
    """Run the closed-loop agent over a fixed channel trajectory H (T,K,N).
    Returns dict with per-slot realized rate, switching count, and (optional) belief error."""
    T, K, N = H.shape
    agent.reset()
    rate = np.zeros(T); switch = np.zeros(T)
    obs_err = np.zeros(T)                            # mean posterior var on served ports (calibration)
    real_err = np.zeros(T)                           # realized |h-mu|^2 on served ports
    for t in range(T):
        S = agent.select(first=(t == 0))
        W = agent.precoder(S)
        idx = list(S)
        Ht = H[t][:, idx].T                          # M x K true active channel
        rate[t] = float(sinr_and_rates(Ht, W, agent.sigma2)[1].sum())
        switch[t] = _switch_count(S, agent.S_prev)
        # observe activated ports (noisy), then update
        noise = np.sqrt(sigma_e2 / 2) * (rng.standard_normal((K, len(idx)))
                                         + 1j * rng.standard_normal((K, len(idx))))
        y = H[t][:, idx] + noise
        if track_belief:
            served = idx[:K]                         # first K activated ~ the served ports
            pv = agent.bel.port_variances()[:, served]
            obs_err[t] = pv.mean()
            real_err[t] = np.mean(np.abs(H[t][:, served] - agent.bel.mu[:, served]) ** 2)
        agent.update(S, y)
    out = dict(rate=rate, switch=switch)
    if track_belief:
        out.update(post_var=obs_err, real_err=real_err)
    return out


def run_genie(H, M, sigma2=1e-3, P=1.0):
    """Full-CSI greedy selection + full-CSI MMSE precoder -> realized-rate upper bound."""
    T, K, N = H.shape
    rate = np.zeros(T); switch = np.zeros(T); S_prev = None
    for t in range(T):
        h = H[t]
        S, _ = select_greedy(h, M, sigma2=sigma2, P=P)
        W = mmse_precoder(h[:, list(S)].T, P=P, sigma2=sigma2)
        rate[t] = float(sinr_and_rates(h[:, list(S)].T, W, sigma2)[1].sum())
        switch[t] = _switch_count(S, S_prev); S_prev = S
    return dict(rate=rate, switch=switch)


def run_random(H, M, rng, sigma2=1e-3, P=1.0):
    """Random selection + full-CSI precoder on the chosen ports -> selection lower bound."""
    T, K, N = H.shape
    rate = np.zeros(T); switch = np.zeros(T); S_prev = None
    for t in range(T):
        h = H[t]
        S = tuple(sorted(rng.choice(N, size=M, replace=False).tolist()))
        W = mmse_precoder(h[:, list(S)].T, P=P, sigma2=sigma2)
        rate[t] = float(sinr_and_rates(h[:, list(S)].T, W, sigma2)[1].sum())
        switch[t] = _switch_count(S, S_prev); S_prev = S
    return dict(rate=rate, switch=switch)


def objective(res, eta_sw=1.0, e_sw=1.0):
    """Eq.7 long-term objective: mean over slots of (sum-rate - eta_sw e_sw * switch)."""
    return float(np.mean(res["rate"] - eta_sw * e_sw * res["switch"]))
