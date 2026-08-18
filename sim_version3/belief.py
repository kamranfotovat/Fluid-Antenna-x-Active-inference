r"""
Step 3 -- Kalman belief over the channel (perception only, no decisions).

The agent never sees the full channel. Each slot it maintains a Gaussian belief
per user, q(h_k) = CN(mu_k, Sigma_k) over ALL N ports, and refines it from noisy
observations of only the M activated ports. This module is the PERCEPTION block;
action selection (EFE) comes in Steps 4-5.

Generative model (linear-Gaussian -> exact Kalman; RESEARCH_PLAN Sec. 4, EFE_DESIGN Sec. 6)
-------------------------------------------------------------------------------------------
    state      h_k(t) in C^N
    transition h_k(t) = rho h_k(t-1) + CN(0, (1-rho^2) beta_k R)     (AR(1), given by physics)
    likelihood y_k(t) = P_S h_k(t) + CN(0, sigma_e^2 I_M)            (observe only activated ports)
    prior D    h_k(0) ~ CN(0, beta_k R)                              (stationary distribution)

P_S in {0,1}^{M x N} is the selection matrix of the active port set S (rows = active ports).

Kalman recursion (per user; complex circular Gaussian)
------------------------------------------------------
    predict:  mu    <- rho mu
              Sigma <- rho^2 Sigma + (1 - rho^2) beta_k R            (uncertainty grows -> CSI aging)
    update:   Scov  = P_S Sigma P_S^H + sigma_e^2 I_M                (M x M, ALWAYS positive definite)
              Kgain = Sigma P_S^H Scov^{-1}
              mu    <- mu + Kgain (y - P_S mu)
              Sigma <- (I - Kgain P_S) Sigma (I - Kgain P_S)^H + sigma_e^2 Kgain Kgain^H   (Joseph)

On the singular prior (Step 0 finding). With dense sub-lambda/2 spacing R is strongly
rank-deficient, so Sigma0 = beta_k R is singular. This is harmless here: the update only
inverts the M x M innovation covariance Scov, which is regularized by sigma_e^2 I and is
therefore always invertible; Sigma itself is never inverted. The Joseph form keeps Sigma
Hermitian PSD across iterations. Hence no artificial jitter is needed -- the measurement
noise IS the regularizer. (A `reg` knob is exposed for stress tests but defaults to 0.)
"""

from __future__ import annotations

import numpy as np


def selection_matrix(S, N: int) -> np.ndarray:
    """P_S in {0,1}^{M x N}: row m is the indicator of the m-th active port."""
    idx = list(S)
    P = np.zeros((len(idx), N))
    P[np.arange(len(idx)), idx] = 1.0
    return P


class KalmanBelief:
    """Bank of K independent complex Kalman filters, one per user's N-port channel.

    Parameters
    ----------
    R        : (N, N) real spatial correlation (Jakes). PSD, unit diagonal.
    beta     : (K,) per-user channel power (scalar broadcast allowed).
    rho      : temporal correlation coefficient in [0, 1].
    sigma_e2 : estimation/pilot noise variance on an observed port.
    reg      : optional diagonal jitter added to the process/target covariance
               (default 0 -- not needed; see module docstring).
    """

    def __init__(self, R, beta, rho, sigma_e2, reg: float = 0.0):
        self.R = np.asarray(R, float)
        self.N = self.R.shape[0]
        beta = np.atleast_1d(np.asarray(beta, float))
        self.K = beta.shape[0]
        self.beta = beta
        self.rho = float(rho)
        self.sigma_e2 = float(sigma_e2)
        self.reg = float(reg)

        eye = np.eye(self.N)
        # Stationary covariance per user = the D prior and the aging target.
        self.C_stat = [self.beta[k] * self.R + self.reg * eye for k in range(self.K)]
        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def reset(self):
        """Belief = stationary prior: mu = 0, Sigma = beta_k R (the D prior)."""
        self.mu = np.zeros((self.K, self.N), dtype=complex)
        self.Sigma = np.array([C.astype(complex).copy() for C in self.C_stat])  # (K,N,N)
        return self

    # -- Kalman steps --------------------------------------------------------
    def predict(self):
        """Time update (aging): mu <- rho mu; Sigma <- rho^2 Sigma + (1-rho^2) C_stat."""
        r2 = self.rho ** 2
        self.mu *= self.rho
        for k in range(self.K):
            self.Sigma[k] = r2 * self.Sigma[k] + (1.0 - r2) * self.C_stat[k]
            self.Sigma[k] = 0.5 * (self.Sigma[k] + self.Sigma[k].conj().T)
        return self

    def update(self, S, y):
        """Measurement update from observations of the active ports S.

        y : (K, M) noisy observations, y[k] = P_S h_k + CN(0, sigma_e^2 I).
        """
        P = selection_matrix(S, self.N)                 # (M, N), real
        M = P.shape[0]
        I_M = np.eye(M)
        I_N = np.eye(self.N)
        y = np.asarray(y, dtype=complex)
        for k in range(self.K):
            Sig = self.Sigma[k]
            SPt = Sig @ P.T                             # N x M  (= Sigma P_S^H)
            Scov = P @ SPt + self.sigma_e2 * I_M        # M x M, positive definite
            Kg = np.linalg.solve(Scov.T, SPt.T).T       # N x M  (= SPt @ inv(Scov))
            innov = y[k] - P @ self.mu[k]               # M
            self.mu[k] = self.mu[k] + Kg @ innov
            A = I_N - Kg @ P                            # N x N
            Sig = A @ Sig @ A.conj().T + self.sigma_e2 * (Kg @ Kg.conj().T)  # Joseph
            self.Sigma[k] = 0.5 * (Sig + Sig.conj().T)
        return self

    # -- read-outs (used by verification now, by EFE later) ------------------
    def port_variances(self) -> np.ndarray:
        """(K, N) real posterior variance per port = diag(Sigma_k)."""
        return np.stack([np.real(np.diag(self.Sigma[k])) for k in range(self.K)], axis=0)

    def total_variance(self) -> np.ndarray:
        """(K,) trace(Sigma_k) -- total per-user uncertainty."""
        return np.array([np.real(np.trace(self.Sigma[k])) for k in range(self.K)])

    def logdet(self) -> np.ndarray:
        """(K,) log det Sigma_k via Cholesky-safe eigen route (belief entropy up to const)."""
        out = np.empty(self.K)
        for k in range(self.K):
            w = np.linalg.eigvalsh(self.Sigma[k])
            out[k] = np.sum(np.log(np.clip(w, 1e-300, None)))
        return out
