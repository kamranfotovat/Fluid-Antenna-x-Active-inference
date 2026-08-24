r"""
Predictable temporal channel model (AR(1) -> Jakes/Doppler) for the "learning that pays" upgrade.

Truth: a band-limited Jakes process via sum-of-sinusoids, temporal autocorrelation
    r(tau) = J0(2*pi * f_D T_s * tau)          (f_D T_s = normalized Doppler; mobility knob)
Belief/predictor: AR(p) fit to r(tau) by Yule-Walker. AR(1) (p=1) reproduces the current model
(rho = r(1)); higher p exploits the band-limited memory that AR(1) throws away -> smaller prediction
error -> smaller effective "aging" error in predict/partial-sensing. The AR coefficients encode the
Doppler, so learning them = learning mobility = touching the DOMINANT (temporal) error.

Space-time separable elsewhere: temporal AR(p) (shared across ports) x spatial R (unchanged).
This module is purely temporal (1-D) -- the fast, decisive premise check.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0
from scipy.linalg import toeplitz, solve_toeplitz


def jakes_autocorr(tau, fd_ts):
    """Temporal autocorrelation r(tau) = J0(2*pi f_D T_s tau)."""
    return j0(2.0 * np.pi * fd_ts * np.asarray(tau, float))


def jakes_series(T, fd_ts, n_sin=64, rng=None):
    """Unit-power complex Jakes time series via sum-of-sinusoids: E[h(t)h*(t+tau)] -> J0(...)."""
    rng = np.random.default_rng() if rng is None else rng
    alpha = rng.uniform(0, 2 * np.pi, n_sin)
    phi = rng.uniform(0, 2 * np.pi, n_sin)
    t = np.arange(T)[:, None]
    doppler = 2.0 * np.pi * fd_ts * np.cos(alpha)[None, :]        # (1, n_sin)
    h = np.exp(1j * (doppler * t + phi[None, :]))                 # (T, n_sin)
    return h.mean(axis=1) * np.sqrt(n_sin) / np.sqrt(n_sin)       # (T,), unit power (mean of unit phasors)


def ar_coeffs_yw(p, fd_ts):
    """Yule-Walker AR(p) coefficients fit to the Jakes autocorrelation, plus prediction-error var.

    Solves R a = r  with R = Toeplitz(r[0..p-1]), r = [r(1)..r(p)].
    Returns (a (p,), pred_err_var). AR(1): a=[r(1)], err = 1 - r(1)^2."""
    r = jakes_autocorr(np.arange(p + 1), fd_ts)                   # r[0..p]
    if p == 0:
        return np.array([]), 1.0
    a = solve_toeplitz((r[:p], r[:p]), r[1:p + 1])               # Hermitian Toeplitz (real, symmetric)
    err = float(r[0] - a @ r[1:p + 1])
    return a, max(err, 1e-12)


def companion(a):
    """Companion transition matrix A (p x p) for AR(p): x(t) = A x(t-1) + e, top row = coeffs."""
    p = len(a)
    A = np.zeros((p, p))
    A[0, :] = a
    if p > 1:
        A[1:, :-1] = np.eye(p - 1)
    return A


def ar_kalman_track(obs, sensed, a, err_var, sigma_e2, beta=1.0):
    """Per-port AR(p) Kalman over noisy pilots. obs[t] = noisy measurement (used only where sensed[t]).
    Returns (pred, post): pred[t] = one-step-ahead estimate of h(t) BEFORE this slot's obs
    (predict-then-precode uses this); post[t] = estimate after the obs (observe-then-precode uses this).
    AR(1) (p=1) reduces to the current model."""
    p = len(a)
    T = len(obs)
    A = companion(a)
    Q = np.zeros((p, p)); Q[0, 0] = err_var * beta
    x = np.zeros(p, dtype=complex)
    P = beta * np.eye(p)
    pred = np.zeros(T, dtype=complex); post = np.zeros(T, dtype=complex)
    for t in range(T):
        x = A @ x                                   # predict
        P = A @ P @ A.T + Q
        pred[t] = x[0]
        if sensed[t]:
            s = P[0, 0] + sigma_e2
            k = P[:, 0] / s                         # Kalman gain (p,)
            x = x + k * (obs[t] - x[0])
            P = P - np.outer(k, P[0, :])
        post[t] = x[0]
    return pred, post


def generate_spacetime_jakes(R, beta, fd_ts, T, K, n_sin=64, seed=0):
    """Space-time separable channel: spatial corr beta_k R x temporal Jakes J0(2 pi fd tau).
    h_k(t) = L_k @ g(t), g = N iid unit-power Jakes series, L_k = chol(beta_k R). Returns H (T,K,N).
    E[h(t)[n] h*(t')[m]] = beta_k R[n,m] J0(2 pi fd (t-t'))."""
    from channel import hermitian_sqrt
    R = np.asarray(R, float); N = R.shape[0]
    beta = np.full(K, float(beta)) if np.isscalar(beta) else np.asarray(beta, float)
    rng = np.random.default_rng(seed)
    L = [hermitian_sqrt(beta[k] * R) for k in range(K)]
    H = np.empty((T, K, N), dtype=complex)
    for k in range(K):
        g = np.stack([jakes_series(T, fd_ts, n_sin, rng) for _ in range(N)], axis=1)  # (T, N) iid Jakes
        g = g / np.sqrt(np.mean(np.abs(g) ** 2, axis=0, keepdims=True))                # unit power per port
        H[:, k, :] = g @ L[k].T                                                        # spatial colouring
    return H


def ar_from_acf(r):
    """Yule-Walker AR(p) from a given autocorrelation vector r[0..p] (r[0]=1). Returns (a, err_var).

    Plain (unregularized) Levinson. On an ESTIMATED r this is exactly the fragile path TM-4
    diagnosed -- at low Doppler the Toeplitz system is near-singular and Levinson can fail outright
    ("Singular principal minor"); we fall back to a least-squares solve so the naive baseline stays
    measurable instead of crashing. Use ar_from_acf_robust for estimated ACFs."""
    from scipy.linalg import solve_toeplitz
    r = np.asarray(r, float)
    p = len(r) - 1
    if p == 0:
        return np.array([]), 1.0
    try:
        a = solve_toeplitz((r[:p], r[:p]), r[1:p + 1])
    except (np.linalg.LinAlgError, ValueError):
        a = np.linalg.lstsq(toeplitz(r[:p]), r[1:p + 1], rcond=None)[0]
    err = float(r[0] - a @ r[1:p + 1])
    return a, max(err, 1e-6)


def ar_from_acf_robust(r, se, n_draws=64, ridge=None, seed=0):
    """Yule-Walker that KNOWS IT IS ESTIMATED -- the principled replacement for TM-3's hand hedges.

    The problem it solves (TM-4). At low Doppler the Jakes ACF is smooth, so the p x p Toeplitz
    system is near-singular: at fd=0.10, p=4 -> cond(Gamma) = 3e4, ev = 1.6e-4, |a|_1 = 13.1. A
    0.01 error in r-hat then produces coefficients whose ACTUAL one-step error variance is ~17
    while plain Yule-Walker still REPORTS ~0.18 -- 94x overconfident. Feeding that to the Kalman
    makes it trust a stale prediction, and predict-then-precode craters. This, not ACF bias, is
    what r_shrink=0.95 / ev_inflate=3.0 were secretly compensating for.

    Three data-driven corrections, all keyed to the estimator's OWN standard errors `se`:
      ORDER SELECTION -- the dominant one. A learner cannot afford the order an oracle can. Measured
                actual 1-step error at fd=0.10: with a perfect ACF, AR(6) reaches 0.0000 and AR(1)
                only 0.1833; but with se=0.010, AR(4) degrades to 0.0229 while the far humbler AR(2)
                gives 0.0214 and WINS. So we pick q by bootstrap: resample r ~ N(r-hat, diag(se^2)),
                refit each order, and score it by the error it ACTUALLY incurs. INFORMATION BUYS
                MODEL ORDER -- sharper ACF estimates unlock higher orders, which is exactly what an
                epistemic probe purchases. Coefficients are zero-padded back to p so the pN-state
                Kalman is untouched (a companion matrix with trailing zeros IS an AR(q) in a p-lag
                state).
      RIDGE  -- diagonal loading delta = sum_tau se(tau)^2 on the Toeplitz system. It is exactly
                the ACF uncertainty, so it shrinks |a| when the data are thin and vanishes as
                samples accumulate. No constant to tune.
      POSTERIOR-PREDICTIVE ev -- instead of reporting the in-sample residual, report the error the
                resampled coefficients actually incur (median, since the tail is heavy). The process
                noise then reflects how badly our own uncertainty could hurt us, which is precisely
                what the Kalman needs to stay calibrated rather than overconfident.

    Returns (a, ev) zero-padded to length p, same contract as ar_from_acf. `q` is exposed via
    the module-level last_order() for diagnostics.
    """
    global _LAST_ORDER
    r = np.asarray(r, float).copy()
    se = np.asarray(se, float)
    p = len(r) - 1
    if p == 0:
        return np.array([]), 1.0
    delta = float(np.sum(se[1:] ** 2)) if ridge is None else float(ridge)
    rng = np.random.default_rng(seed)
    draws = np.clip(r[None, 1:] + rng.normal(0.0, 1.0, (n_draws, p)) * se[None, 1:], -0.999, 0.999)

    best = (np.inf, None, None)
    for q in range(1, p + 1):
        G_hat = toeplitz(r[:q])
        errs = []
        for j in range(n_draws):
            rp = np.r_[1.0, draws[j]]
            try:
                ap = np.linalg.solve(toeplitz(rp[:q]) + delta * np.eye(q), rp[1:q + 1])
            except np.linalg.LinAlgError:
                errs.append(np.inf); continue
            errs.append(float(r[0] - 2.0 * ap @ r[1:q + 1] + ap @ G_hat @ ap))
        score = float(np.median(errs))
        if score < best[0]:
            aq = np.linalg.solve(toeplitz(r[:q]) + delta * np.eye(q), r[1:q + 1])
            best = (score, q, aq)
    score, q, aq = best
    _LAST_ORDER = q
    a = np.zeros(p)
    a[:q] = aq
    return a, max(score, 1e-6)


def fit_fd_jakes(r, se=None):
    """Least-squares fit of the single Doppler parameter f_D to a measured autocorrelation.

    WEIGHTED by 1/se^2 when standard errors are given. This is not cosmetic: TemporalACF.rhat()
    reports r=1.0 for lags it has NO samples for (flagged by se=1.0), and at the first relearn the
    high lags are always unsampled. An unweighted fit reads those 1.0's as "perfectly correlated"
    and returns a wildly wrong f_D -- measured 0.3040 against a true 0.1000 at t=4, which wrecks the
    belief before any real data arrives. Weighting by 1/se^2 makes an unsampled lag ~250x less
    influential than a well-sampled one, so the fit follows the lags actually measured."""
    from scipy.optimize import minimize_scalar
    r = np.asarray(r, float)
    taus = np.arange(1, len(r))
    if se is None:
        w = np.ones(len(taus))
    else:
        w = 1.0 / np.maximum(np.asarray(se, float)[1:], 1e-3) ** 2
    obj = lambda fd: float(np.sum(w * (r[1:] - jakes_autocorr(taus, fd)) ** 2))
    return float(minimize_scalar(obj, bounds=(0.005, 0.45), method="bounded").x)


def ar_from_acf_parametric(r, se, n_draws=24, seed=0):
    """PARAMETRIC alternative to ar_from_acf_robust: fit ONE physical parameter, not p free lags.

    ar_from_acf_robust must estimate p free autocorrelation values and then survive an
    ill-conditioned Yule-Walker solve, which forces it down to order ~2 and caps recovery. But the
    physics says those p values are not free -- they all follow from the Doppler via
    r(tau) = J0(2 pi f_D T_s tau). Fitting 1 parameter to p noisy lags is far better conditioned, so
    the FULL order p becomes affordable: TM-6 measured f_D pinned to +-0.0045 even at se = 0.08.

    TM-6 also measured the cost. This trades estimator variance for MODEL-MISMATCH BIAS: on
    non-Jakes spectra the error is flat in se (it stops improving with more data) because the
    residual is bias. It still beat the nonparametric fit in every cell at realistic sample sizes
    (se >= 0.02), but with enough data the assumption-free estimator must eventually win.

    Process noise is bootstrapped exactly as in ar_from_acf_robust -- resample the ACF, refit f_D,
    and report the error those coefficients ACTUALLY incur -- so a slightly wrong f_D cannot make
    the Kalman overconfident (the failure mode that craters predict-then-precode).
    """
    global _LAST_ORDER
    r = np.asarray(r, float)
    se = np.asarray(se, float)
    p = len(r) - 1
    if p == 0:
        return np.array([]), 1.0
    fd = fit_fd_jakes(r, se)
    a, _ = ar_coeffs_yw(p, fd)
    # Evaluate the process noise against a SANITISED reference: keep the lags we actually measured,
    # and fill unsampled ones (se ~ 1) from the fitted Jakes curve instead of rhat()'s 1.0 default.
    # Scoring against the raw r would either read garbage as a perfect predictor (ev -> the 1e-6
    # floor, i.e. catastrophic overconfidence) or read shape mismatch as enormous noise (ev ~ 2).
    r_eval = np.where(se >= 0.99, jakes_autocorr(np.arange(p + 1), fd), r)
    r_eval[0] = 1.0
    G = toeplitz(r_eval[:p])
    rng = np.random.default_rng(seed)
    evs = []
    for _ in range(n_draws):
        rp = r.copy()
        rp[1:] = np.clip(r[1:] + rng.normal(0.0, 1.0, p) * se[1:], -0.999, 0.999)
        ap, _ = ar_coeffs_yw(p, fit_fd_jakes(rp, se))
        evs.append(float(r_eval[0] - 2.0 * ap @ r_eval[1:p + 1] + ap @ G @ ap))
    _LAST_ORDER = p                                   # parametric always affords the full order
    return a, max(float(np.median(evs)), 1e-6)


_LAST_ORDER = None


def last_order():
    """Order chosen by the most recent ar_from_acf_robust call (diagnostics only)."""
    return _LAST_ORDER


class TemporalACF:
    """Online temporal autocorrelation estimator from same-port measurements across slots.
    E[y_n(t) conj(y_n(t-tau))] = beta_n r(tau) (tau>0); |y|^2 gives beta+sigma_e2. Normalized -> r(tau).
    Temporal samples are AVAILABLE (the policy holds ports across slots) -- unlike the near-field.

    BIAS WARNING (TM-3 / TM-4). If fed the POLICY's measurements, the lag-tau pairs only exist for
    ports the policy chose to hold at BOTH t and t-tau. That conditions on both endpoints being
    strong, which preferentially samples temporally-coherent realizations -> r-hat biased HIGH ->
    Doppler underestimated -> overconfident stale prediction -> predict-then-precode craters.
    Feed it a RANDOM PROBE port instead (see st_belief.run_st_learn_probe) for an unbiased sample.

    Estimator uncertainty: rhat_lcb(kappa) returns a one-sided LOWER confidence bound
    r-hat(tau) - kappa * se(tau) with Bartlett's standard error se(tau)^2 ~ (1 + 2 sum_{j<tau} r(j)^2)/n_tau.
    Justified by the proven ASYMMETRY of the loss (underestimating Doppler is catastrophic,
    overestimating is mild), and it vanishes as samples accumulate -- no tuned shrink constant.
    """

    def __init__(self, N, p, sigma_e2, forget=1.0, matched=True):
        self.N, self.p, self.sigma_e2, self.forget = N, p, float(sigma_e2), float(forget)
        self.matched = bool(matched)
        self.acc = np.zeros(p + 1); self.cnt = np.zeros(p + 1)
        self.den = np.zeros(p + 1)                 # matched-window power for each lag
        self.hist = [dict() for _ in range(N)]

    def update(self, t, idx, y):
        if self.forget < 1.0:
            self.acc *= self.forget; self.cnt *= self.forget; self.den *= self.forget
        for a, n in enumerate(idx):
            yn = y[:, a]
            K = len(yn)
            pn = np.abs(yn) ** 2
            self.acc[0] += np.sum(pn); self.cnt[0] += K
            self.den[0] += np.sum(pn) - self.sigma_e2 * K
            for tau in range(1, self.p + 1):
                yp = self.hist[n].get(t - tau)
                if yp is not None:
                    self.acc[tau] += np.real(np.sum(yn * np.conj(yp)))
                    # MATCHED normalization: the power of the very same pair of samples
                    self.den[tau] += 0.5 * np.sum(pn + np.abs(yp) ** 2) - self.sigma_e2 * K
                    self.cnt[tau] += K
            self.hist[n][t] = yn
            self.hist[n].pop(t - self.p - 1, None)

    def rhat(self):
        """r-hat(tau). matched=True normalizes each lag by the power of the SAME sample pairs.

        Why matched matters (TM-4). The pooled form r-hat = A/B (numerator over lag-tau pairs,
        denominator over ALL samples) is a ratio of two random sums drawn from DIFFERENT sample
        sets. Under a sparse sensing schedule the effective sample count is the number of coherent
        windows (~tens, not hundreds), and Jensen's inequality on E[1/B] inflates the ratio by
        ~r/n_eff -- measured +0.034 at n_eff ~ 36, which is the ENTIRE bias TM-3 was hand-hedging.
        Normalizing each lag by its own pairs' power makes numerator and denominator share the
        same fading realizations, so the fade cancels and the ratio bias largely disappears.
        """
        r = np.ones(self.p + 1)
        for tau in range(1, self.p + 1):
            if self.cnt[tau] > 0:
                if self.matched:
                    d = max(self.den[tau] / self.cnt[tau], 1e-6)
                else:
                    d = max(self.acc[0] / max(self.cnt[0], 1) - self.sigma_e2, 1e-6)
                r[tau] = np.clip((self.acc[tau] / self.cnt[tau]) / d, -0.999, 0.999)
        return r

    def counts(self):
        return self.cnt.copy()

    def stderr(self, r=None):
        """Bartlett standard error of r-hat(tau) from the ACTUAL sample counts."""
        r = self.rhat() if r is None else np.asarray(r, float)
        se = np.zeros(self.p + 1)
        for tau in range(1, self.p + 1):
            n = self.cnt[tau]
            if n > 1:
                se[tau] = np.sqrt(max(1.0 + 2.0 * np.sum(r[1:tau] ** 2), 1.0) / n)
            else:
                se[tau] = 1.0                      # no data -> maximal uncertainty
        return se

    def rhat_lcb(self, kappa=1.0):
        """Risk-aware ACF: one-sided lower confidence bound (hedges the catastrophic direction)."""
        r = self.rhat()
        r_lo = r.copy()
        r_lo[1:] = np.clip(r[1:] - kappa * self.stderr(r)[1:], -0.999, 0.999)
        return r_lo


def ar_predict_1step_empirical(series, a):
    """Empirical 1-step prediction MSE of AR(p) predictor a on a series (normalized by signal power)."""
    p = len(a)
    if p == 0:
        return 1.0
    T = len(series)
    # predict series[t] from the p most recent samples (lag 1 first): a[0]*s[t-1]+...+a[p-1]*s[t-p]
    pred = np.array([a @ series[t - p:t][::-1] for t in range(p, T)])
    true = series[p:T]
    return float(np.mean(np.abs(true - pred) ** 2) / np.mean(np.abs(series) ** 2))
