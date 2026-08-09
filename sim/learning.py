r"""
Step 8 -- slow-loop learning of the spatial correlation R (the one parameter that matters).

The model-mismatch probe (Step 7c) showed the agent is near-immune to wrong rho/beta but loses
~1.5 objective under a wrong spatial correlation R. In reality the true propagation may not be
the idealized Jakes model, so R is the parameter worth LEARNING from the observation stream.

Estimator (partial-observation covariance -> normalized correlation)
-------------------------------------------------------------------
Each slot we observe noisy pilots y_k = h_k[S] + CN(0, sigma_e^2 I) on the active ports S.
Every marginal h_k is a draw from CN(0, beta_k R), so for a CO-OBSERVED pair (i, j):
    E[ y_{k,i} conj(y_{k,j}) ] = beta_k R_ij            (i != j; pilot noise is independent)
    E[ |y_{k,i}|^2 ]           = beta_k R_ii + sigma_e^2 = beta_k + sigma_e^2   (R_ii = 1)
Accumulating outer products over users & slots and NORMALIZING to unit diagonal cancels the
unknown per-user beta_k and yields an unbiased estimate of the (real, Jakes) correlation R.
Temporal AR(1) correlation only inflates the estimator variance, not its mean.
"""

from __future__ import annotations

import numpy as np


class SpatialCorrEstimator:
    """Online estimator of the spatial correlation R from partial noisy observations."""

    def __init__(self, N: int):
        self.N = N
        self.S = np.zeros((N, N), dtype=complex)   # sum of y_i conj(y_j) over co-observations
        self.C = np.zeros((N, N))                  # co-observation counts

    def update(self, idx, y):
        """idx: list of active ports; y: (K, |idx|) noisy observations."""
        idx = list(idx)
        ix = np.ix_(idx, idx)
        for k in range(y.shape[0]):
            self.S[ix] += np.outer(y[k], y[k].conj())
            self.C[ix] += 1.0

    def coverage(self) -> float:
        """Fraction of port pairs observed at least once (need ~full for a good estimate)."""
        return float(np.mean(self.C > 0))

    def estimate(self, sigma_e2: float, reg: float = 1e-6) -> np.ndarray:
        """Return R_hat: real, symmetric, unit-diagonal, PSD."""
        cnt = np.maximum(self.C, 1.0)
        cov = self.S / cnt                          # ~ (avg beta) R  (+ sigma_e^2 on diagonal)
        diag = np.real(np.diag(cov)) - sigma_e2     # remove pilot noise from the diagonal
        diag = np.clip(diag, 1e-6, None)
        d = 1.0 / np.sqrt(diag)
        Rhat = np.real(cov) * np.outer(d, d)        # normalize -> unit diagonal, cancel beta
        Rhat = 0.5 * (Rhat + Rhat.T)
        np.fill_diagonal(Rhat, 1.0)
        # project to PSD and renormalize the diagonal to 1
        w, V = np.linalg.eigh(Rhat)
        w = np.clip(w, 0.0, None)
        Rhat = (V * w) @ V.T + reg * np.eye(self.N)
        dd = np.sqrt(np.clip(np.diag(Rhat), 1e-9, None))
        Rhat = Rhat / np.outer(dd, dd)
        return Rhat


def exponential_correlation(positions, d0=0.3):
    """A NON-Jakes spatial correlation: R_ij = exp(-||p_i - p_j|| / d0) (distances in wavelengths).
    Valid PSD unit-diagonal correlation with monotone (no Bessel oscillation) decay -- used to
    stress-test the agent under model mismatch (it assumes Jakes but the truth is exponential)."""
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    R = np.exp(-dist / d0)
    return 0.5 * (R + R.T)


def set_correlation(sim, R):
    """Override a ChannelSimulator's spatial correlation with a custom R (rebuilds the colouring
    matrices and re-draws the state). Lets us generate non-Jakes channels from the same class."""
    from channel import hermitian_sqrt
    sim.R = 0.5 * (R + R.conj().T)
    sim._Lstat = [hermitian_sqrt(sim.beta[k] * sim.R) for k in range(sim.K)]
    sim._Linnov = [np.sqrt(1.0 - sim.rho ** 2) * sim._Lstat[k] for k in range(sim.K)]
    sim.reset()
    return sim


def gather_correlation(sim, T_warm, M, sigma_e2, rng):
    """Probe the array for T_warm slots with random port sets (good pair coverage), accumulating
    the correlation statistics. Returns (R_hat, coverage). Uses the simulator's own dynamics."""
    est = SpatialCorrEstimator(sim.N)
    h = sim.h
    for t in range(T_warm):
        if t > 0:
            h = sim.step()
        S = rng.choice(sim.N, size=M, replace=False)
        y = h[:, S] + np.sqrt(sigma_e2 / 2) * (rng.standard_normal((sim.K, M))
                                               + 1j * rng.standard_normal((sim.K, M)))
        est.update(S, y)
    return est.estimate(sigma_e2), est.coverage()
