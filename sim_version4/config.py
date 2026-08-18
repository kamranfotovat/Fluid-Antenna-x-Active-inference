r"""
Central operating point for the version-2 experiments.

Everything that used to be copy-pasted across the driver scripts (grid size, aperture,
M, correlation, noise, weights) lives here ONCE so the geometry rescale (2 lambda /
lambda-10) becomes a one-line change instead of an eight-file edit.

Two named operating points are provided:

  OP_V1  -- the original locked point: 1 lambda aperture, 5x5 = 25 ports, 0.25 lambda
            spacing, M=5, NO min-distance constraint. Reproduces the current paper numbers.
            Used first to smoke-test the min-distance machinery at small N.

  OP_V2  -- the target regime discussed with Zijun: 2 lambda aperture, 21x21 = 441 ports,
            lambda/10 spacing, M=8, min-distance d_min = 0.5 lambda (>= half wavelength
            between any two activated ports -> mutual-coupling / decorrelation constraint).

`ACTIVE` selects which one the drivers use. Flip it to OP_V2 for the rescaled run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from channel import port_positions, spatial_correlation


@dataclass(frozen=True)
class OperatingPoint:
    # geometry
    Nx: int
    Ny: int
    Wx: float                 # aperture (wavelengths)
    Wy: float
    # selection
    M: int
    d_min: float | None       # min spacing (wavelengths) between activated ports; None = off
    # beamforming hardware
    n_rf: int | None = None   # TX RF chains for hybrid beamforming; None = fully-digital (n_rf = M)
    # sensing hardware (S2): read the M active ports through n_rf_sense mixed analog measurements
    n_rf_sense: int | None = None   # sensing RF chains; None = S1 per-port reads (one ADC per active port)
    # scenario
    K: int = 3
    rho: float = 0.9
    sigma2: float = 0.03      # data-noise power (15 dB)
    sigma_e2: float = 1e-3    # pilot/estimation noise (~20 dB)
    eta_sw: float = 1.0       # switching-cost weight
    beta_w: float = 0.25      # epistemic (exploration) weight
    P: float = 1.0            # tx power budget

    @property
    def N(self) -> int:
        return self.Nx * self.Ny

    @property
    def beta(self) -> np.ndarray:
        """Per-user channel powers (length K)."""
        base = np.array([1.0, 0.7, 1.3])
        if self.K == len(base):
            return base
        # generic fallback for other K
        return np.linspace(0.7, 1.3, self.K)

    def positions(self) -> np.ndarray:
        return port_positions(self.Nx, self.Ny, self.Wx, self.Wy)

    def R(self) -> np.ndarray:
        return spatial_correlation(self.positions())

    def label(self) -> str:
        dm = "off" if self.d_min is None else f"{self.d_min:g} lambda"
        rf = "digital" if self.n_rf is None else f"n_rf={self.n_rf}"
        sns = "per-port" if self.n_rf_sense is None else f"n_rf_sense={self.n_rf_sense}"
        return (f"N={self.N} ({self.Nx}x{self.Ny}), {self.Wx:g}x{self.Wy:g} lambda, "
                f"K={self.K}, M={self.M}, {rf}, sense={sns}, d_min={dm}, "
                f"{10*np.log10(1/self.sigma2):.0f} dB, rho={self.rho}, beta_w={self.beta_w}")


# --- original locked point (reproduces current paper) ------------------------
OP_V1 = OperatingPoint(Nx=5, Ny=5, Wx=1.0, Wy=1.0, M=5, d_min=None)

# --- target rescaled regime (2 lambda, lambda/10) ----------------------------
# d_min left OFF: the correlation-aware belief already spreads ports to the Jakes
# decorrelation null on its own, so the hard >= lambda/2 filter is near-redundant
# (AIF rate with it on vs off was identical). Kept as an optional knob, not hardcoded.
# M raised to 10 for the hybrid study (more active ports -> a real RF-chain budget to compress).
OP_V2 = OperatingPoint(Nx=21, Ny=21, Wx=2.0, Wy=2.0, M=10, d_min=None)

# --- version-3 hybrid-beamforming point --------------------------------------
# Same geometry as OP_V2, but the M=10 active ports are now driven by n_rf < M RF chains
# through a fully-connected unit-modulus analog network (hybrid beamforming). n_rf is the
# knob we sweep: n_rf >= 2K reproduces the digital precoder (near-lossless); below that the
# analog approximation bites. The AIF loop (belief, selection, sensing) is unchanged --
# only the transmit precoder is factorized into F_RF * W_BB.
OP_V3 = OperatingPoint(Nx=21, Ny=21, Wx=2.0, Wy=2.0, M=10, d_min=None, n_rf=6)

# --- version-4 sensing-through-the-analog-network point (S2) ------------------
# Same geometry / transmit hybrid as OP_V3, but the M=10 active ports are now SENSED through
# n_rf_sense < M mixed analog measurements (y = F_RF^H P_S h + noise) instead of one ADC per
# active port. n_rf_sense = None recovers S1 (per-port reads) exactly (Invariant I1). The agent
# DESIGNS the unit-modulus sensing combiner F_RF to maximize information gain about its belief.
OP_V4 = OperatingPoint(Nx=21, Ny=21, Wx=2.0, Wy=2.0, M=10, d_min=None, n_rf=6, n_rf_sense=6)

# Which one the drivers use. Flip to OP_V4 for the S2 sensing run.
ACTIVE = OP_V4
