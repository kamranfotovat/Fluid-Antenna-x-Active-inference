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
from selection import select_greedy, select_topM_feasible, select_random_feasible
from channel import feasible_ports
from hybrid import hybridize
from sensing import observation_matrix, sense, design_sensing_matrix, sense_info
import efe


# --------------------------------------------------------------------------- the agent
class AIFAgent:
    def __init__(self, R, beta, rho, sigma_e2, M,
                 alpha=1.0, beta_w=1.0, eta_sw=1.0, e_sw=1.0, sigma2=1e-3, P=1.0,
                 positions=None, d_min=None, n_rf=None):
        self.bel = KalmanBelief(R=R, beta=beta, rho=rho, sigma_e2=sigma_e2)
        self.M = M
        self.alpha, self.beta_w = alpha, beta_w
        self.eta_sw, self.e_sw = eta_sw, e_sw
        self.sigma2, self.P = sigma2, P
        self.positions, self.d_min = positions, d_min   # min-spacing constraint (None = off)
        self.n_rf = n_rf                                 # RF chains (None = fully-digital)
        self.hyb_rng = np.random.default_rng(0)          # restarts for hybrid factorization
        self.K = self.bel.K
        self.S_prev = None

    def reset(self):
        self.bel.reset()
        self.S_prev = None
        return self

    def select(self, first: bool, n_rf_sense=None):
        """Greedy EFE port selection on the predicted belief. If n_rf_sense is given, the epistemic
        term values COMPRESSED sensing info (S2-6 sensing-aware selection) instead of full per-port."""
        if not first:
            self.bel.predict()                      # aging
        return efe.greedy_select(
            self.bel, self.M, S_prev=self.S_prev,
            alpha=self.alpha, beta=self.beta_w, eta_sw=self.eta_sw, e_sw=self.e_sw,
            sigma2=self.sigma2, P=self.P,
            positions=self.positions, d_min=self.d_min, n_rf_sense=n_rf_sense)

    def precoder(self, S):
        """Belief-based robust-MMSE digital precoder F* (M x K). If n_rf is set, F* is the
        TARGET that gets factorized into a fully-connected hybrid precoder F_RF W_BB; we
        return the effective M x K precoder so the rate scorer is identical to the digital
        case. The belief/selection/sensing are untouched -- hybrid only reshapes the beam."""
        W, _, _ = efe.robust_mmse_from_belief(self.bel, S, self.sigma2, self.P)
        return hybridize(W, self.n_rf, P=self.P, rng=self.hyb_rng)

    def update(self, S, y):
        self.bel.update(S, y)
        self.S_prev = S


# --------------------------------------------------------------------------- runners (shared trajectory)
def _switch_count(S, S_prev):
    return 0 if S_prev is None else len(set(S) ^ set(S_prev))


def run_aif(agent: AIFAgent, H, sigma_e2, rng, track_belief=False, sense_first=False):
    """Run the closed-loop agent over a fixed channel trajectory H (T,K,N).

    sense_first controls the intra-slot protocol -- the single biggest lever on rate:
      False (predict-then-act): precode from the PREDICTED (aged) belief, then observe.
             The beam is aimed on a stale guess -> aging error caps the rate.
      True  (observe-then-precode): send pilots on the activated ports, Kalman-update,
             THEN precode from the FRESH belief. Served-port CSI error ~ sigma_e^2, so
             the rate approaches the genie. Selection still uses the predicted belief
             (we must choose which ports to activate before we can sense them).

    Returns per-slot realized rate, switching count, and (optional) belief-quality traces
    measured at the moment the precoder is built (the CSI the beam actually uses).
    """
    T, K, N = H.shape
    agent.reset()
    rate = np.zeros(T); switch = np.zeros(T)
    post_var = np.zeros(T); real_err = np.zeros(T)
    for t in range(T):
        S = agent.select(first=(t == 0))            # predict (if t>0) + greedy on predicted belief
        idx = list(S)
        noise = np.sqrt(sigma_e2 / 2) * (rng.standard_normal((K, len(idx)))
                                         + 1j * rng.standard_normal((K, len(idx))))
        y = H[t][:, idx] + noise
        if sense_first:
            agent.bel.update(S, y)                   # fresh belief BEFORE precoding
        if track_belief:
            served = idx[:K]                         # first K activated ~ the served ports
            post_var[t] = agent.bel.port_variances()[:, served].mean()
            real_err[t] = np.mean(np.abs(H[t][:, served] - agent.bel.mu[:, served]) ** 2)
        W = agent.precoder(S)                        # fresh if sense_first, predicted otherwise
        Ht = H[t][:, idx].T                          # M x K true active channel
        rate[t] = float(sinr_and_rates(Ht, W, agent.sigma2)[1].sum())
        switch[t] = _switch_count(S, agent.S_prev)
        if sense_first:
            agent.S_prev = S                         # belief already updated above
        else:
            agent.update(S, y)                       # precode-then-observe: update belief now
    out = dict(rate=rate, switch=switch)
    if track_belief:
        out.update(post_var=post_var, real_err=real_err)
    return out


def run_aif_s2(agent: AIFAgent, H, sigma_e2, rng, n_rf_sense, sense_mode="designed",
               sense_aware_select=False, track_belief=True):
    """S2 closed loop -- sense the M active ports THROUGH the analog network (observe-then-precode).

    Each slot: predict -> greedy EFE select S -> read the M active ports with a sensing budget of
    n_rf_sense measurements -> Kalman-update (general A) -> transmit-hybrid precode from the fresh
    belief -> score realized rate. Selection and transmit are IDENTICAL to S1; only the sensing
    read changes with `sense_mode`:

      'designed' -- n_rf_sense EFE-designed unit-modulus analog combinations of all M active ports
                    (design_sensing_matrix on the predicted aggregate covariance)          [S2]
      'random'   -- n_rf_sense random unit-modulus combinations                       [ablation]
      'subset'   -- read the n_rf_sense highest-variance individual active ports  [S1 @ same budget]
      'perport'  -- read all M active ports individually (n_rf_sense ignored)  [S1 full-read ceiling]

    Returns per-slot rate, switch, and (optional) belief-quality traces (post_var, real_err) on the
    served ports at the moment the precoder is built.
    """
    T, K, N = H.shape
    agent.reset()
    d_rng = np.random.default_rng(12345)                  # restarts for the sensing-matrix design
    rate = np.zeros(T); switch = np.zeros(T)
    info = np.zeros(T); post_var = np.zeros(T); real_err = np.zeros(T)
    # S2-6 selection epistemic: None -> S1 per-user full-read; a budget k -> compressed info bound.
    #   sense_aware_select True  -> budget-aware, sel_nrs = n_rf_sense
    #                       "full"-> same aggregate objective but budget-UNAWARE (sel_nrs = M), the
    #                                clean same-scale control for the aware-vs-unaware comparison
    #                       False -> the ordinary S1 selector
    if sense_aware_select == "full":
        sel_nrs = agent.M
    elif sense_aware_select:
        sel_nrs = n_rf_sense
    else:
        sel_nrs = None
    for t in range(T):
        S = agent.select(first=(t == 0), n_rf_sense=sel_nrs)   # predict (t>0) + greedy on belief
        idx = list(S); M = len(idx)
        nrs = M if (n_rf_sense is None or sense_mode == "perport") else int(min(n_rf_sense, M))
        cov_bar = np.sum(efe.active_covs(agent.bel, S), axis=0)   # PREDICTED aggregate uncertainty

        if sense_mode in ("designed", "random"):
            if sense_mode == "designed":
                F_RF, _ = design_sensing_matrix(cov_bar, nrs, sigma_e2, rng=d_rng)
            else:
                F_RF = np.exp(1j * d_rng.uniform(0, 2 * np.pi, size=(M, nrs)))
            info[t] = sense_info(F_RF, cov_bar, sigma_e2)  # bits of info this measurement carries
            y = sense(H[t], F_RF, S, sigma_e2, rng)        # (K, nrs) mixed reads
            agent.bel.update_general(observation_matrix(F_RF, S, N), y)
        else:                                              # per-port reads (subset or full)
            if sense_mode == "subset" and nrs < M:
                var_agg = agent.bel.port_variances()[:, idx].sum(axis=0)   # (M,) predicted variance
                loc = list(np.argsort(var_agg)[::-1][:nrs])   # highest-uncertainty active ports
            else:
                loc = list(range(M))
            read = [idx[j] for j in loc]
            F_sel = np.zeros((M, len(loc)), dtype=complex)     # equivalent selection "combiner"
            F_sel[loc, range(len(loc))] = 1.0
            info[t] = sense_info(F_sel, cov_bar, sigma_e2)
            yb = H[t][:, read] + np.sqrt(sigma_e2 / 2) * (
                rng.standard_normal((K, len(read))) + 1j * rng.standard_normal((K, len(read))))
            agent.bel.update(tuple(read), yb)

        if track_belief:
            served = idx[:K]
            post_var[t] = agent.bel.port_variances()[:, served].mean()
            real_err[t] = np.mean(np.abs(H[t][:, served] - agent.bel.mu[:, served]) ** 2)
        W = agent.precoder(S)                              # transmit hybrid from the FRESH belief
        Ht = H[t][:, idx].T
        rate[t] = float(sinr_and_rates(Ht, W, agent.sigma2)[1].sum())
        switch[t] = _switch_count(S, agent.S_prev)
        agent.S_prev = S
    out = dict(rate=rate, switch=switch, info=info)
    if track_belief:
        out.update(post_var=post_var, real_err=real_err)
    return out


def run_aif_sweep(agent: AIFAgent, H, sigma_e2, rng, n_rf_list):
    """Run the closed-loop AIF agent ONCE (observe-then-precode protocol) but score every
    n_rf in n_rf_list from the SAME belief trajectory. Selection/sensing/Kalman update are
    n_rf-independent, so this is ~len(n_rf_list)x cheaper than calling run_aif per n_rf.

    n_rf_list entries may include None (fully-digital). Returns {n_rf: rate_array(T)} plus a
    shared 'switch' array (identical across n_rf). The per-slot digital target F* is
    factorized once per n_rf with the agent's own hybrid rng for reproducibility.
    """
    from hybrid import hybridize
    T, K, N = H.shape
    agent.reset()
    rates = {nr: np.zeros(T) for nr in n_rf_list}
    switch = np.zeros(T)
    for t in range(T):
        S = agent.select(first=(t == 0))
        idx = list(S)
        noise = np.sqrt(sigma_e2 / 2) * (rng.standard_normal((K, len(idx)))
                                         + 1j * rng.standard_normal((K, len(idx))))
        y = H[t][:, idx] + noise
        agent.bel.update(S, y)                            # sense-first: fresh belief
        W_dig, _, _ = efe.robust_mmse_from_belief(agent.bel, S, agent.sigma2, agent.P)
        Ht = H[t][:, idx].T                               # M x K true active channel
        for nr in n_rf_list:
            W = hybridize(W_dig, nr, P=agent.P, rng=agent.hyb_rng)
            rates[nr][t] = float(sinr_and_rates(Ht, W, agent.sigma2)[1].sum())
        switch[t] = _switch_count(S, agent.S_prev)
        agent.S_prev = S
    return rates, switch


def run_genie_sweep(H, M, n_rf_list, sigma2=1e-3, P=1.0, positions=None, d_min=None, rng=None):
    """Full-CSI genie scored for every n_rf from one greedy-selection pass (selection is
    n_rf-independent). Returns {n_rf: rate_array(T)}."""
    T, K, N = H.shape
    rng = np.random.default_rng(0) if rng is None else rng
    rates = {nr: np.zeros(T) for nr in n_rf_list}
    for t in range(T):
        h = H[t]
        S, _ = select_greedy(h, M, sigma2=sigma2, P=P, positions=positions, d_min=d_min)
        W_dig = mmse_precoder(h[:, list(S)].T, P=P, sigma2=sigma2)
        Ht = h[:, list(S)].T
        for nr in n_rf_list:
            W = hybridize(W_dig, nr, P=P, rng=rng)
            rates[nr][t] = float(sinr_and_rates(Ht, W, sigma2)[1].sum())
    return rates


def run_naive_sweep(H, M, sigma_e2, rng, n_rf_list, sigma2=1e-3, P=1.0, refresh=1,
                    positions=None, d_min=None):
    """Naive partial-CSI baseline scored for every n_rf from one selection pass."""
    T, K, N = H.shape
    est = np.zeros((K, N), dtype=complex)
    rates = {nr: np.zeros(T) for nr in n_rf_list}
    S_prev = None; rr = 0
    for t in range(T):
        power = np.sum(np.abs(est) ** 2, axis=0)
        S = list(select_topM_feasible(power, M - refresh, positions, d_min))
        c = 0
        while c < refresh:
            n = rr % N
            feas = positions is None or d_min is None or bool(feasible_ports(positions, S, [n], d_min))
            if n not in S and feas:
                S.append(n); c += 1
            rr += 1
            if rr > 4 * N:
                break
        S = tuple(sorted(S)); idx = list(S)
        y = _obs(H[t], idx, K, sigma_e2, rng)
        est[:, idx] = y
        W_dig = mmse_precoder(y.T, P=P, sigma2=sigma2)
        Ht = H[t][:, idx].T
        for nr in n_rf_list:
            W = hybridize(W_dig, nr, P=P, rng=rng)
            rates[nr][t] = float(sinr_and_rates(Ht, W, sigma2)[1].sum())
        S_prev = S
    return rates


def run_genie(H, M, sigma2=1e-3, P=1.0, positions=None, d_min=None, n_rf=None, rng=None):
    """Full-CSI greedy selection + full-CSI MMSE precoder -> realized-rate upper bound.
    Honors the same min-spacing constraint (positions + d_min) so it stays a feasible bound.
    If n_rf is set, the full-CSI digital precoder is factorized into the same fully-connected
    hybrid structure -> a fair full-CSI *hybrid* upper bound."""
    T, K, N = H.shape
    rng = np.random.default_rng(0) if rng is None else rng
    rate = np.zeros(T); switch = np.zeros(T); S_prev = None
    for t in range(T):
        h = H[t]
        S, _ = select_greedy(h, M, sigma2=sigma2, P=P, positions=positions, d_min=d_min)
        W = mmse_precoder(h[:, list(S)].T, P=P, sigma2=sigma2)
        W = hybridize(W, n_rf, P=P, rng=rng)
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


def _obs(h_slice, idx, K, sigma_e2, rng):
    """Noisy pilots on the active ports: y = h[idx] + CN(0, sigma_e^2 I). Shape (K, |idx|)."""
    return h_slice[:, idx] + np.sqrt(sigma_e2 / 2) * (
        rng.standard_normal((K, len(idx))) + 1j * rng.standard_normal((K, len(idx))))


def run_naive(H, M, sigma_e2, rng, sigma2=1e-3, P=1.0, refresh=1, positions=None, d_min=None,
              n_rf=None):
    """Fair partial-CSI competitor -- NO active inference (the key baseline to beat).

    Same partial access + observe-then-precode as the agent, but replaces the generative
    Kalman belief with a memoryless point estimate: last measured value held per port (no
    temporal prediction, no spatial inference, no uncertainty). Selection = top-M by
    last-known aggregate power (a Paper-3-style heuristic) plus `refresh` round-robin ports
    for PASSIVE (non-active) sensing. Precoding uses the fresh pilots on the chosen ports.
    Isolates exactly what active inference buys: model-based belief + EFE-unified selection.
    """
    T, K, N = H.shape
    est = np.zeros((K, N), dtype=complex)
    rate = np.zeros(T); switch = np.zeros(T); S_prev = None; rr = 0
    for t in range(T):
        power = np.sum(np.abs(est) ** 2, axis=0)
        S = list(select_topM_feasible(power, M - refresh, positions, d_min))
        c = 0
        while c < refresh:                            # passive round-robin refresh of stale ports
            n = rr % N
            feas = positions is None or d_min is None or bool(
                feasible_ports(positions, S, [n], d_min))
            if n not in S and feas:
                S.append(n); c += 1
            rr += 1
            if rr > 4 * N:                            # safety: constraint too tight for a refresh port
                break
        S = tuple(sorted(S)); idx = list(S)
        y = _obs(H[t], idx, K, sigma_e2, rng)
        est[:, idx] = y                               # raw held estimate (no filtering)
        W = mmse_precoder(y.T, P=P, sigma2=sigma2)    # precode on fresh pilots
        W = hybridize(W, n_rf, P=P, rng=rng)          # factorize to hybrid if n_rf set
        rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, sigma2)[1].sum())
        switch[t] = _switch_count(S, S_prev); S_prev = S
    return dict(rate=rate, switch=switch)


def run_random_partial(H, M, sigma_e2, rng, sigma2=1e-3, P=1.0, positions=None, d_min=None):
    """Partial-CSI lower bound: random selection, observe-then-precode on fresh pilots.
    Random draw respects the min-spacing constraint when positions + d_min are given."""
    T, K, N = H.shape
    rate = np.zeros(T); switch = np.zeros(T); S_prev = None
    for t in range(T):
        S = select_random_feasible(N, M, rng, positions, d_min); idx = list(S)
        y = _obs(H[t], idx, K, sigma_e2, rng)
        W = mmse_precoder(y.T, P=P, sigma2=sigma2)
        rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, sigma2)[1].sum())
        switch[t] = _switch_count(S, S_prev); S_prev = S
    return dict(rate=rate, switch=switch)


def run_fixed(H, S, sigma_e2, rng, sigma2=1e-3, P=1.0):
    """Hold a FIXED port set S forever (observe-then-precode on fresh pilots, never switch).
    Demonstrates why tracking is needed when the good ports move: rate decays as the hotspot
    drifts away from S. Zero switching by construction."""
    T, K, N = H.shape
    idx = list(S)
    rate = np.zeros(T); switch = np.zeros(T)
    for t in range(T):
        y = _obs(H[t], idx, K, sigma_e2, rng)
        W = mmse_precoder(y.T, P=P, sigma2=sigma2)
        rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, sigma2)[1].sum())
    return dict(rate=rate, switch=switch)


def objective(res, eta_sw=1.0, e_sw=1.0):
    """Eq.7 long-term objective: mean over slots of (sum-rate - eta_sw e_sw * switch)."""
    return float(np.mean(res["rate"] - eta_sw * e_sw * res["switch"]))
