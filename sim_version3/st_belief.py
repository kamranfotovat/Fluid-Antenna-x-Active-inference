r"""
Exact AR(p) SPACE-TIME Kalman belief (TM-2) -- augmented p-lag temporal state per port, spatial R.

State per user: x_k = [h(t); h(t-1); ...; h(t-p+1)] in C^{pN}. Space-time separable:
  transition  F   = companion(a) (x) I_N            (shared temporal AR(p), a from Yule-Walker)
  process Q_k     = E1 (x) (err_var * beta_k R)      (innovation only on h(t), spatially R-correlated)
  stationary P0_k = Gamma (x) (beta_k R),  Gamma[i,j] = J0(2 pi fd |i-j|)
  observe S       = [P_S | 0 ... 0] x_k + noise      (current-time channel at active ports)

Exposes the CURRENT-TIME marginal as .mu (K,N) and .Sigma (K,N,N) so the S1 EFE terms
(pragmatic/epistemic/precoder) work unchanged. AR(1) (p=1) reduces EXACTLY to the current
KalmanBelief (F=rho I, Q=(1-rho^2)beta R, P0=beta R). Exact but O((pN)^3) -> use at reduced N.
"""

from __future__ import annotations

import numpy as np

from belief import selection_matrix
from temporal import ar_coeffs_yw, companion, jakes_autocorr


class STKalmanBelief:
    def __init__(self, R, beta, fd_ts, p, sigma_e2):
        self.R = np.asarray(R, float)
        self.N = self.R.shape[0]
        self.beta = np.atleast_1d(np.asarray(beta, float))
        self.K = self.beta.shape[0]
        self.sigma_e2 = float(sigma_e2)
        self.p = int(p)
        self.fd_ts = float(fd_ts)

        a, ev = ar_coeffs_yw(self.p, self.fd_ts)
        self.a, self.ev = a, ev
        I_N = np.eye(self.N)
        self.F = np.kron(companion(a), I_N)                          # pN x pN
        E1 = np.zeros((self.p, self.p)); E1[0, 0] = 1.0
        self.Q = [np.kron(E1, ev * self.beta[k] * self.R) for k in range(self.K)]
        Gamma = jakes_autocorr(np.abs(np.subtract.outer(np.arange(self.p), np.arange(self.p))),
                               self.fd_ts)                            # p x p stationary temporal corr
        self.P0 = [np.kron(Gamma, self.beta[k] * self.R) for k in range(self.K)]
        self._I = np.eye(self.p * self.N)
        self.reset()

    def reset(self):
        self.X = np.zeros((self.K, self.p * self.N), dtype=complex)
        self.P = [P.astype(complex).copy() for P in self.P0]
        self._sync()
        return self

    def _sync(self):
        """Current-time marginal for the EFE terms."""
        self.mu = self.X[:, :self.N].copy()                          # (K, N)
        self.Sigma = np.array([self.P[k][:self.N, :self.N] for k in range(self.K)])  # (K,N,N)

    def predict(self):
        self.X = self.X @ self.F.T
        for k in range(self.K):
            Pk = self.F @ self.P[k] @ self.F.T + self.Q[k]
            self.P[k] = 0.5 * (Pk + Pk.conj().T)
        self._sync()
        return self

    def update(self, S, y):
        idx = list(S); m = len(idx)
        P_S = selection_matrix(S, self.N)                            # m x N
        Hobs = np.hstack([P_S, np.zeros((m, (self.p - 1) * self.N))])  # m x pN
        I_m = np.eye(m)
        y = np.asarray(y, dtype=complex)
        for k in range(self.K):
            Pk = self.P[k]
            PHt = Pk @ Hobs.T                                        # pN x m
            Scov = Hobs @ PHt + self.sigma_e2 * I_m
            Kg = np.linalg.solve(Scov.T, PHt.T).T                    # pN x m
            innov = y[k] - Hobs @ self.X[k]
            self.X[k] = self.X[k] + Kg @ innov
            J = self._I - Kg @ Hobs
            Pk = J @ Pk @ J.conj().T + self.sigma_e2 * (Kg @ Kg.conj().T)
            self.P[k] = 0.5 * (Pk + Pk.conj().T)
        self._sync()
        return self

    def port_variances(self):
        return np.stack([np.real(np.diag(self.Sigma[k])) for k in range(self.K)], axis=0)

    def set_ar(self, a, ev):
        """Adopt learned AR(p) coefficients: rebuild transition F and process noise Q (keep X, P)."""
        self.a, self.ev = a, ev
        I_N = np.eye(self.N)
        self.F = np.kron(companion(a), I_N)
        E1 = np.zeros((self.p, self.p)); E1[0, 0] = 1.0
        self.Q = [np.kron(E1, ev * self.beta[k] * self.R) for k in range(self.K)]


# --------------------------------------------------------------------------- precoding helper
def _precode(bel, S, op, rng_h):
    """Belief-based robust-MMSE precoder, projected onto the hardware-feasible set if the operating
    point specifies RF chains. Hybrid is pure POST-PROCESSING of the AIF precoder (hybrid.py), so
    the belief / EFE selection / sensing are untouched. op.n_rf = None -> fully digital, which is
    what OP_V2 uses, so every previously-recorded digital result is bit-for-bit unaffected."""
    import efe
    W, _, _ = efe.robust_mmse_from_belief(bel, S, op.sigma2, op.P)
    n_rf = getattr(op, "n_rf", None)
    if n_rf is not None and n_rf < W.shape[0]:
        from hybrid import hybridize
        W = hybridize(W, n_rf, P=op.P, rng=rng_h)
    return W


# --------------------------------------------------------------------------- closed-loop runner
def choose_pilots(bel, idx, m, rule="variance"):
    """Pick which m of the |S| ACTIVATED ports to spend pilots on.

    rule="variance"  -- top-m by summed per-user posterior variance. Cheap, and the
                        intuition is right (leave unpiloted what the belief is already
                        confident about), but it scores each port in ISOLATION.
    rule="epistemic" -- greedily maximize the EFE's own epistemic term,
                        Epis(Q) = sum_k log2 det(I + Cov_k(Q)/sigma_e^2).
                        Differs from "variance" by accounting for REDUNDANCY: two
                        activated ports that are both uncertain AND strongly correlated
                        with each other carry nearly the same information, so the second
                        is nearly worthless. Marginal variance cannot see that; a log-det
                        can. They coincide only if the candidates are uncorrelated, which
                        on a sub-wavelength grid is exactly what we may not assume.

    Note the first greedy pick is identical under both rules -- for a single port
    log2(1 + var/sigma_e^2) is monotone in var. They diverge only from the second pick on.
    """
    if rule == "variance":
        var = bel.port_variances()[:, idx].sum(axis=0)
        return sorted(idx[i] for i in np.argsort(var)[::-1][:m])
    if rule != "epistemic":
        raise ValueError(f"unknown pilot rule {rule!r}")

    import efe
    chosen, remaining = [], list(idx)
    while len(chosen) < m and remaining:
        best_gain, best_p = -np.inf, None
        for p in remaining:
            gain = efe.epistemic_value(bel, tuple(sorted(chosen + [p])))
            if gain > best_gain:
                best_gain, best_p = gain, p
        chosen.append(best_p)
        remaining.remove(best_p)
    return sorted(chosen)


def run_st(bel, H, op, rng, protocol="observe", m_sense=None, pilot_rule="variance"):
    """Closed-loop with the ST belief. protocol in {observe, predict, partial}. Returns rate/switch.
      observe : select -> sense all M -> update -> precode (fresh)
      predict : select -> precode (from predicted belief) -> sense all M -> update (for next slot)
      partial : select -> sense m_sense most-uncertain of M -> update -> precode (rest R/AR-inferred)
    """
    import efe
    from agent import _obs, _switch_count
    from precoding import sinr_and_rates
    T, K, N = H.shape
    bel.reset()
    rate = np.zeros(T); switch = np.zeros(T); S_prev = None
    pos = op.positions()
    rng_h = np.random.default_rng(12345)          # separate stream: never perturbs the pilot noise
    for t in range(T):
        if t > 0:
            bel.predict()
        S = efe.greedy_select(bel, op.M, S_prev=S_prev, alpha=1.0, beta=op.beta_w,
                              eta_sw=op.eta_sw, e_sw=1.0, sigma2=op.sigma2, P=op.P,
                              positions=pos, d_min=op.d_min)
        idx = list(S)
        if protocol == "predict":
            W = _precode(bel, S, op, rng_h)                                    # from predicted belief
            rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, op.sigma2)[1].sum())
            y = _obs(H[t], idx, K, op.sigma_e2, rng); bel.update(S, y)         # sense for next slot
        else:
            if protocol == "partial" and m_sense is not None and m_sense < len(idx):
                S_sense = choose_pilots(bel, idx, m_sense, rule=pilot_rule)
            else:
                S_sense = idx
            y = _obs(H[t], S_sense, K, op.sigma_e2, rng); bel.update(tuple(S_sense), y)
            W = _precode(bel, S, op, rng_h)
            rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, op.sigma2)[1].sum())
        switch[t] = _switch_count(S, S_prev); S_prev = S
    return dict(rate=rate, switch=switch)


def run_st_learn(bel, H, op, rng, acf, protocol="predict", relearn_every=5, m_sense=None,
                 ev_inflate=1.0, r_shrink=1.0):
    """Closed-loop with ONLINE Doppler/AR learning: estimate temporal autocorr from the measurement
    stream, refit AR(p), and adopt it every relearn_every slots (bel starts from a wrong Doppler)."""
    import efe
    from agent import _obs, _switch_count
    from precoding import sinr_and_rates
    from temporal import ar_from_acf
    T, K, N = H.shape
    bel.reset()
    rate = np.zeros(T); switch = np.zeros(T); S_prev = None
    pos = op.positions()
    for t in range(T):
        if t > 0:
            bel.predict()
        S = efe.greedy_select(bel, op.M, S_prev=S_prev, alpha=1.0, beta=op.beta_w,
                              eta_sw=op.eta_sw, e_sw=1.0, sigma2=op.sigma2, P=op.P,
                              positions=pos, d_min=op.d_min)
        idx = list(S)
        if protocol == "predict":
            W, _, _ = efe.robust_mmse_from_belief(bel, S, op.sigma2, op.P)
            rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, op.sigma2)[1].sum())
            S_sense = idx
            y = _obs(H[t], S_sense, K, op.sigma_e2, rng); bel.update(S, y)
        else:
            if protocol == "partial" and m_sense is not None and m_sense < len(idx):
                var = bel.port_variances()[:, idx].sum(axis=0)
                S_sense = sorted(idx[i] for i in np.argsort(var)[::-1][:m_sense])
            else:
                S_sense = idx
            y = _obs(H[t], S_sense, K, op.sigma_e2, rng); bel.update(tuple(S_sense), y)
            W, _, _ = efe.robust_mmse_from_belief(bel, S, op.sigma2, op.P)
            rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, op.sigma2)[1].sum())
        acf.update(t, list(S_sense), y)                       # feed the temporal-autocorr estimator
        if (t + 1) % relearn_every == 0:
            r = acf.rhat().copy()
            r[1:] *= r_shrink                                 # conservative: shrink correlation (=> more Doppler)
            a, ev = ar_from_acf(r)
            bel.set_ar(a, ev * ev_inflate)                    # conservative: never underestimate process noise
        switch[t] = _switch_count(S, S_prev); S_prev = S
    return dict(rate=rate, switch=switch, rhat=acf.rhat())


def run_st_learn_probe(bel, H, op, rng, acf, protocol="predict", relearn_every=5, m_sense=None,
                       kappa=0.0, ev_inflate=1.0, probe=True, robust=True, method="nonparam"):
    """TM-4 -- PRINCIPLED online Doppler learning: unbiased ACF + risk-aware point estimate.

    Two fixes over run_st_learn (which needed hand-tuned r_shrink / ev_inflate):
      (1) UNBIASED SAMPLING. One of the M ports each slot is a RANDOM PROBE -- drawn uniformly at
          random, INDEPENDENT of the belief, and held for p+1 consecutive slots so it yields lag
          1..p pairs. Only the probe's measurements feed the ACF, so the estimator never sees the
          policy's survivorship selection. This is a MODEL-epistemic action (information about the
          parameters), distinct from S1's STATE-epistemic term. Cost: exactly 1/M of the pilots.
      (2) RISK-AWARE ESTIMATE. Use the one-sided lower confidence bound r-hat - kappa*se instead of
          an arbitrary shrink. Justified by the loss asymmetry (too-slow catastrophic, too-fast mild);
          it self-annihilates as samples accumulate. kappa=0 recovers the plain estimate.

    probe=False keeps the policy-fed ACF (the biased baseline) with the same risk-aware estimate,
    isolating how much of the fix is the sampling vs the hedging.
    """
    import efe
    from agent import _obs, _switch_count
    from precoding import sinr_and_rates
    from temporal import (ar_from_acf, ar_from_acf_robust, ar_from_acf_parametric,
                          last_order)
    T, K, N = H.shape
    p = bel.p
    orders = []
    bel.reset()
    rate = np.zeros(T); switch = np.zeros(T); S_prev = None
    pos = op.positions()
    probe_port, probe_age = None, 0
    for t in range(T):
        if t > 0:
            bel.predict()
        if probe:
            if probe_port is None or probe_age > p:                  # fresh probe every p+1 slots
                probe_port, probe_age = int(rng.integers(N)), 0
            probe_age += 1
        S, tr = efe.greedy_select(bel, op.M, S_prev=S_prev, alpha=1.0, beta=op.beta_w,
                                  eta_sw=op.eta_sw, e_sw=1.0, sigma2=op.sigma2, P=op.P,
                                  positions=pos, d_min=op.d_min, return_trace=True)
        if probe and probe_port not in S:                            # force the probe in, evicting
            drop = tr[-1][0]                                         # the lowest-marginal greedy pick
            S = tuple(sorted(set(S) - {drop} | {probe_port}))
        idx = list(S)
        if protocol == "predict":
            W, _, _ = efe.robust_mmse_from_belief(bel, S, op.sigma2, op.P)
            rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, op.sigma2)[1].sum())
            S_sense = idx
            y = _obs(H[t], S_sense, K, op.sigma_e2, rng); bel.update(S, y)
        else:
            if protocol == "partial" and m_sense is not None and m_sense < len(idx):
                var = bel.port_variances()[:, idx].sum(axis=0)
                keep = set(np.argsort(var)[::-1][:m_sense])
                if probe and probe_port in idx:                      # the probe must always be sensed
                    keep = set(list(keep)[:max(m_sense - 1, 0)]) | {idx.index(probe_port)}
                S_sense = sorted(idx[i] for i in keep)
            else:
                S_sense = idx
            y = _obs(H[t], S_sense, K, op.sigma_e2, rng); bel.update(tuple(S_sense), y)
            W, _, _ = efe.robust_mmse_from_belief(bel, S, op.sigma2, op.P)
            rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, op.sigma2)[1].sum())
        if probe:
            if probe_port in S_sense:                                # UNBIASED: probe stream only
                j = S_sense.index(probe_port)
                acf.update(t, [probe_port], y[:, [j]])
        else:
            acf.update(t, list(S_sense), y)                          # biased baseline
        if (t + 1) % relearn_every == 0:
            r = acf.rhat_lcb(kappa)
            if method == "param":
                a, ev = ar_from_acf_parametric(r, acf.stderr())  # fit ONE Doppler -> full order
                orders.append(last_order())
            elif robust:
                a, ev = ar_from_acf_robust(r, acf.stderr())   # order selection + ridge + honest ev
                orders.append(last_order())
            else:
                a, ev = ar_from_acf(r)
            bel.set_ar(a, ev * ev_inflate)
        switch[t] = _switch_count(S, S_prev); S_prev = S
    return dict(rate=rate, switch=switch, rhat=acf.rhat(), rlcb=acf.rhat_lcb(kappa),
                se=acf.stderr(), cnt=acf.counts(), orders=orders)
