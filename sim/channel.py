"""
Step 0 -- Channel generator (ground truth, no agent).

Implements the shared physics used by every downstream module:
  - Jakes spatial correlation R   (RESEARCH_PLAN Eq. 2)
  - AR(1) temporal evolution rho  (RESEARCH_PLAN Eq. 3)

Model
-----
Per-user channel over all N ports:  h_k(t) in C^N.
    h_k(t) = rho * h_k(t-1) + sqrt(1 - rho^2) * e_k(t),   e_k(t) ~ CN(0, beta_k * R)
    rho    = J0(2*pi*f_D*T_s)
Stationary distribution:  h_k ~ CN(0, beta_k * R)   (used for the initial prior / reset).

Conventions
-----------
Circularly-symmetric complex Gaussian:  z ~ CN(0, I) drawn as (a + 1j b)/sqrt(2),
a,b ~ N(0, I) real, so E[|z_i|^2] = 1 and E[z z^H] = I, E[z z^T] = 0 (proper/circular).
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0


# ----------------------------------------------------------------------------- geometry / correlation
def port_positions(Nx: int, Ny: int, Wx: float, Wy: float) -> np.ndarray:
    """Return (N, 2) port coordinates in wavelengths for an Nx-by-Ny grid
    spanning an aperture Wx-by-Wy (also in wavelengths). N = Nx*Ny.

    Spacing per axis = W/(n-1); e.g. Nx=5, Wx=1.0 -> 0.25 lambda (dense, sub-lambda/2).
    """
    xs = np.linspace(0.0, Wx, Nx) if Nx > 1 else np.array([0.0])
    ys = np.linspace(0.0, Wy, Ny) if Ny > 1 else np.array([0.0])
    grid = np.array([(x, y) for y in ys for x in xs])  # row-major (x fastest)
    return grid


def spatial_correlation(positions: np.ndarray) -> np.ndarray:
    """Jakes / Clarke isotropic spatial correlation matrix (Eq. 2).

    R[i, j] = J0(2*pi * ||p_i - p_j||)   with distances in wavelengths.
    R is real, symmetric, unit-diagonal, PSD.
    """
    diff = positions[:, None, :] - positions[None, :, :]      # (N, N, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))                # (N, N) in wavelengths
    R = j0(2.0 * np.pi * dist)
    return 0.5 * (R + R.T)  # symmetrize against float noise


def rho_from_doppler(fD: float, Ts: float) -> float:
    """Temporal correlation coefficient rho = J0(2*pi*f_D*T_s) (Eq. 3)."""
    return float(j0(2.0 * np.pi * fD * Ts))


def hermitian_sqrt(M: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """Hermitian PSD square root L with L @ L.conj().T = M.

    Uses eigendecomposition and clips tiny negative eigenvalues (Jakes R is PSD
    in theory but can be marginally indefinite numerically).
    """
    M = 0.5 * (M + M.conj().T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, jitter, None)
    return (V * np.sqrt(w)) @ V.conj().T


# ----------------------------------------------------------------------------- simulator
class ChannelSimulator:
    """Generates ground-truth FAS channels h_k(t) for K users over N ports.

    Parameters
    ----------
    Nx, Ny      : grid size (N = Nx*Ny)
    Wx, Wy      : aperture in wavelengths
    K           : number of users
    rho         : temporal correlation (or pass fD, Ts to derive it)
    beta        : per-user channel power, scalar or length-K array (default 1)
    seed        : RNG seed
    """

    def __init__(
        self,
        Nx: int = 5,
        Ny: int = 5,
        Wx: float = 1.0,
        Wy: float = 1.0,
        K: int = 3,
        rho: float | None = 0.9,
        fD: float | None = None,
        Ts: float | None = None,
        beta: float | np.ndarray = 1.0,
        seed: int | None = 0,
    ):
        self.Nx, self.Ny = Nx, Ny
        self.N = Nx * Ny
        self.K = K
        self.positions = port_positions(Nx, Ny, Wx, Wy)
        self.R = spatial_correlation(self.positions)

        if rho is None:
            assert fD is not None and Ts is not None, "give rho, or both fD and Ts"
            rho = rho_from_doppler(fD, Ts)
        self.rho = float(rho)

        self.beta = np.full(K, float(beta)) if np.isscalar(beta) else np.asarray(beta, float)
        assert self.beta.shape == (K,)

        # Per-user innovation colour matrix: sqrt(beta_k * R).
        self._Lstat = [hermitian_sqrt(self.beta[k] * self.R) for k in range(K)]  # stationary sqrt
        self._Linnov = [np.sqrt(1.0 - self.rho ** 2) * self._Lstat[k] for k in range(K)]

        self.rng = np.random.default_rng(seed)
        self.h = None  # (K, N) current channel
        self.reset()

    # -- CN(0, I) draws ------------------------------------------------------
    def _cn(self, shape) -> np.ndarray:
        a = self.rng.standard_normal(shape)
        b = self.rng.standard_normal(shape)
        return (a + 1j * b) / np.sqrt(2.0)

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Draw h from the stationary prior CN(0, beta_k R). This is the D prior."""
        z = self._cn((self.K, self.N))
        self.h = np.stack([self._Lstat[k] @ z[k] for k in range(self.K)], axis=0)
        return self.h

    def step(self) -> np.ndarray:
        """Advance one slot via AR(1) and return the new (K, N) channel."""
        z = self._cn((self.K, self.N))
        e = np.stack([self._Linnov[k] @ z[k] for k in range(self.K)], axis=0)
        self.h = self.rho * self.h + e
        return self.h

    def generate(self, T: int) -> np.ndarray:
        """Return a (T, K, N) trajectory starting from the current state (inclusive)."""
        out = np.empty((T, self.K, self.N), dtype=complex)
        out[0] = self.h
        for t in range(1, T):
            out[t] = self.step()
        return out


class MovingHotspotSimulator(ChannelSimulator):
    """Channel with a spatial power 'sweet spot' that DRIFTS across the aperture over time.

    On top of the base correlated AR(1) fading f_k(t), a deterministic spatial power envelope
    a_n(t) is applied:  h_k(t)[n] = a_n(t) * f_k(t)[n],  with

        a_n(t) = sqrt( hs_base + hs_peak * exp(-||p_n - c(t)||^2 / (2 hs_width^2)) )

    and the hotspot centre c(t) moving on a circle of radius hs_radius about the aperture centre,
    one lap every hs_period slots. This makes some ports intrinsically better (high SNR) AND makes
    the good region MOVE, so a fixed port set decays and the agent must actively track it. The
    agent's belief still assumes the stationary uniform-power model, so the envelope is unknown to
    it -> it must SENSE to find where the hotspot went (the point of active inference).
    """

    def __init__(self, *args, hs_peak=1.0, hs_base=0.1, hs_width=0.25,
                 hs_radius=0.3, hs_period=40, **kw):
        self.hs_peak, self.hs_base, self.hs_width = hs_peak, hs_base, hs_width
        self.hs_radius, self.hs_period = hs_radius, hs_period
        super().__init__(*args, **kw)          # sets positions/colouring, then calls our reset()

    def _envelope(self, t) -> np.ndarray:
        c0 = self.positions.mean(axis=0)
        ang = 2.0 * np.pi * t / self.hs_period
        c = c0 + self.hs_radius * np.array([np.cos(ang), np.sin(ang)])
        d2 = np.sum((self.positions - c) ** 2, axis=1)
        return np.sqrt(self.hs_base + self.hs_peak * np.exp(-d2 / (2.0 * self.hs_width ** 2)))

    def reset(self) -> np.ndarray:
        self.t = 0
        z = self._cn((self.K, self.N))
        self.f = np.stack([self._Lstat[k] @ z[k] for k in range(self.K)], axis=0)   # base fading
        self.h = self.f * self._envelope(self.t)[None, :]
        return self.h

    def step(self) -> np.ndarray:
        z = self._cn((self.K, self.N))
        e = np.stack([self._Linnov[k] @ z[k] for k in range(self.K)], axis=0)
        self.f = self.rho * self.f + e
        self.t += 1
        self.h = self.f * self._envelope(self.t)[None, :]
        return self.h
