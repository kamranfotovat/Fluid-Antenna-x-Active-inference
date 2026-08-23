r"""
Isotropic correlation-vs-distance estimator g(d) for online learning of the FAS spatial
correlation (S1 active-learning extension, AL-1).

Model: R_ij = g(||p_i - p_j||), a single 1-D profile (Jakes is g(d)=J0(2*pi*d)). Estimating a
profile instead of a full N x N matrix is what makes learning feasible at N=441 -- a handful of
distance bins vs ~10^5 port pairs.

Estimator (co-observation, per distance bin b):
    E[ y_{k,i} conj(y_{k,j}) ] = beta_k R_ij           (i != j; pilot noise independent)
    E[ |y_{k,i}|^2 ]           = beta_k + sigma_e^2
Summing over users AND over pairs in a bin, the (unknown, per-user) beta cancels on normalization:
    g_hat(b) = mean_pairs,users Re(y_i conj y_j)  /  mean_ports (|y|^2 - sigma_e^2)
Bins never co-observed keep the PRIOR g (Jakes) -> the agent starts from Jakes and corrects bins as
data arrives. bin_counts() exposes per-bin coverage for the active-learning novelty term (AL-2).
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0


def jakes_g(d):
    return j0(2.0 * np.pi * np.asarray(d))


class DistanceProfileEstimator:
    def __init__(self, positions, sigma_e2, bin_width=0.1, max_dist=None, prior_g=jakes_g):
        self.pos = np.asarray(positions, float)
        self.N = self.pos.shape[0]
        self.sigma_e2 = float(sigma_e2)
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        self.D = np.sqrt(np.sum(diff ** 2, axis=-1))            # (N,N) pairwise distance
        self.bw = float(bin_width)
        self.max_dist = float(self.D.max()) if max_dist is None else float(max_dist)
        self.n_bins = int(np.ceil(self.max_dist / self.bw)) + 1
        self.B = np.minimum((self.D / self.bw).astype(int), self.n_bins - 1)   # (N,N) bin index
        self.centers = (np.arange(self.n_bins) + 0.5) * self.bw
        self.prior = np.clip(prior_g(self.centers), -1.0, 1.0)  # prior profile per bin
        # accumulators
        self.xc = np.zeros(self.n_bins)      # sum Re(y_i conj y_j) over co-observed pairs & users
        self.mc = np.zeros(self.n_bins)      # count of (pair, user) contributions
        self.pw = np.zeros(self.N)           # sum |y_i|^2 over slots & users
        self.nc = np.zeros(self.N)           # count of (slot, user) per port

    def update(self, idx, y):
        """idx: active ports this slot; y: (K, |idx|) noisy pilots."""
        idx = list(idx); K = y.shape[0]
        p = np.sum(np.abs(y) ** 2, axis=0)                     # (|idx|,) power summed over users
        for a, i in enumerate(idx):
            self.pw[i] += p[a]; self.nc[i] += K
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                bn = self.B[idx[a], idx[b]]
                self.xc[bn] += np.real(np.sum(y[:, a] * np.conj(y[:, b])))
                self.mc[bn] += K

    def avg_var(self):
        m = self.nc > 0
        if not np.any(m):
            return 1.0
        return max(float(np.mean(self.pw[m] / self.nc[m]) - self.sigma_e2), 1e-6)

    def g_hat(self):
        g = self.prior.copy()
        m = self.mc > 0
        g[m] = (self.xc[m] / self.mc[m]) / self.avg_var()
        return np.clip(g, -1.0, 1.0)

    def bin_counts(self):
        return self.mc.copy()

    def R_hat(self, reg=1e-6):
        """Reconstruct a valid (PSD, unit-diagonal) correlation matrix from g_hat."""
        g = self.g_hat()
        R = g[self.B]
        np.fill_diagonal(R, 1.0)
        R = 0.5 * (R + R.T)
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 0.0, None)
        R = (V * w) @ V.T + reg * np.eye(self.N)
        d = np.sqrt(np.clip(np.diag(R), 1e-9, None))
        return R / np.outer(d, d)

    def true_g_bins(self, R_true):
        """Bin-average a known R (for error diagnostics)."""
        g = self.prior.copy()
        iu = np.triu_indices(self.N, k=1)
        for b in range(self.n_bins):
            sel = self.B[iu] == b
            if np.any(sel):
                g[b] = np.mean(R_true[iu][sel])
        return g

    def g_rmse(self, true_g, observed_only=True):
        gh = self.g_hat()
        m = (self.mc > 0) if observed_only else np.ones(self.n_bins, bool)
        if not np.any(m):
            return float("nan")
        return float(np.sqrt(np.mean((gh[m] - true_g[m]) ** 2)))
