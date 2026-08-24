r"""
EXACT reduced-rank AR(p) space-time Kalman belief -- makes TM-2/TM-3 tractable at FULL SCALE (N=441).

Why this is exact, not an approximation
---------------------------------------
The exact ST filter (st_belief.STKalmanBelief) carries a pN x pN covariance; at OP_V2 (N=441, p=4)
that is 1764 x 1764 per user and the Joseph update costs O((pN)^3) ~ 5.5e9 per user per slot.

But the spatial correlation R = J0(2 pi d) on a dense sub-lambda grid is STRONGLY rank-deficient --
its numerical rank is the number of spatial degrees of freedom of the aperture, not the number of
ports (measured: N=441 -> rank 22 at 99.99% energy, N=25 -> rank 14). The channel lives EXACTLY in
range(R):  h_k(t) = hermitian_sqrt(beta_k R) g(t)  in  range(R)  for every t. So writing

    R = B Lam B^T   (B: N x r orthonormal, Lam: r x r diagonal),      h_k(t) = B c_k(t)

loses nothing: the belief on c_k in C^r is a sufficient statistic for the belief on h_k in C^N.
The AR(p) dynamics act identically on every port, so they carry over to c unchanged:

    state  z_k = [c(t); c(t-1); ...; c(t-p+1)] in C^{pr}
    F      = companion(a) (x) I_r
    Q_k    = E1 (x) (err_var * beta_k Lam)
    P0_k   = Gamma (x) (beta_k Lam)
    Hobs   = [B[S,:] | 0 ... 0]                       (observe current-time channel at ports S)
    mu_k   = B c_k(t),      Sigma_k = B P_k[:r,:r] B^T

Cost drops from O((pN)^3) to O((pr)^3) -- 1764^3 -> 96^3, ~6e5x -- while agreeing with the exact
filter to the truncation energy (verified against STKalmanBelief in verify_tm_lr.py).

Interface is identical to STKalmanBelief (.mu, .Sigma, predict/update/set_ar), so run_st /
run_st_learn / run_st_learn_probe accept it unchanged.
"""

from __future__ import annotations

import numpy as np

from temporal import ar_coeffs_yw, companion, jakes_autocorr


class STKalmanBeliefLR:
    def __init__(self, R, beta, fd_ts, p, sigma_e2, rank=None, energy=1 - 1e-6):
        R = np.asarray(R, float)
        self.N = R.shape[0]
        w, V = np.linalg.eigh(R)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        w = np.clip(w, 0.0, None)
        if rank is None:
            rank = int(np.searchsorted(np.cumsum(w) / w.sum(), energy) + 1)
        self.r = int(min(rank, self.N))
        self.B = V[:, :self.r]                                      # N x r, orthonormal
        self.lam = w[:self.r]                                       # r
        self.R = R

        self.beta = np.atleast_1d(np.asarray(beta, float))
        self.K = self.beta.shape[0]
        self.sigma_e2 = float(sigma_e2)
        self.p = int(p)
        self.fd_ts = float(fd_ts)

        a, ev = ar_coeffs_yw(self.p, self.fd_ts)
        self.a, self.ev = a, ev
        self._build(a, ev)
        Gamma = jakes_autocorr(np.abs(np.subtract.outer(np.arange(self.p), np.arange(self.p))),
                               self.fd_ts)
        L = np.diag(self.lam)
        self.P0 = [np.kron(Gamma, self.beta[k] * L) for k in range(self.K)]
        self._I = np.eye(self.p * self.r)
        self.reset()

    def _build(self, a, ev):
        I_r = np.eye(self.r)
        self.F = np.kron(companion(a), I_r)
        E1 = np.zeros((self.p, self.p)); E1[0, 0] = 1.0
        L = np.diag(self.lam)
        self.Q = [np.kron(E1, ev * self.beta[k] * L) for k in range(self.K)]

    def reset(self):
        self.X = np.zeros((self.K, self.p * self.r), dtype=complex)
        self.P = [P.astype(complex).copy() for P in self.P0]
        self._sync()
        return self

    def _sync(self):
        """Lift the reduced belief back to the N-port marginal the EFE terms consume."""
        r, B = self.r, self.B
        self.mu = self.X[:, :r] @ B.T                                # (K, N)
        self.Sigma = np.array([B @ self.P[k][:r, :r] @ B.T for k in range(self.K)])  # (K,N,N)

    def predict(self):
        self.X = self.X @ self.F.T
        for k in range(self.K):
            Pk = self.F @ self.P[k] @ self.F.T + self.Q[k]
            self.P[k] = 0.5 * (Pk + Pk.conj().T)
        self._sync()
        return self

    def update(self, S, y):
        idx = list(S); m = len(idx)
        r = self.r
        Hobs = np.hstack([self.B[idx, :], np.zeros((m, (self.p - 1) * r))])   # m x pr
        I_m = np.eye(m)
        y = np.asarray(y, dtype=complex)
        for k in range(self.K):
            Pk = self.P[k]
            PHt = Pk @ Hobs.T
            Scov = Hobs @ PHt + self.sigma_e2 * I_m
            Kg = np.linalg.solve(Scov.T, PHt.T).T
            self.X[k] = self.X[k] + Kg @ (y[k] - Hobs @ self.X[k])
            J = self._I - Kg @ Hobs
            Pk = J @ Pk @ J.conj().T + self.sigma_e2 * (Kg @ Kg.conj().T)
            self.P[k] = 0.5 * (Pk + Pk.conj().T)
        self._sync()
        return self

    def port_variances(self):
        return np.stack([np.real(np.diag(self.Sigma[k])) for k in range(self.K)], axis=0)

    def set_ar(self, a, ev):
        self.a, self.ev = a, ev
        self._build(a, ev)
