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
        return (f"N={self.N} ({self.Nx}x{self.Ny}), {self.Wx:g}x{self.Wy:g} lambda, "
                f"K={self.K}, M={self.M}, d_min={dm}, "
                f"{10*np.log10(1/self.sigma2):.0f} dB, rho={self.rho}, beta_w={self.beta_w}")


# --- original locked point (reproduces current paper) ------------------------
OP_V1 = OperatingPoint(Nx=5, Ny=5, Wx=1.0, Wy=1.0, M=5, d_min=None)

# --- target rescaled regime (2 lambda, lambda/10) ----------------------------
# d_min left OFF: the correlation-aware belief already spreads ports to the Jakes
# decorrelation null on its own, so the hard >= lambda/2 filter is near-redundant
# (AIF rate with it on vs off was identical). Kept as an optional knob, not hardcoded.
OP_V2 = OperatingPoint(Nx=21, Ny=21, Wx=2.0, Wy=2.0, M=8, d_min=None)

# Which one the drivers use. Flip to OP_V2 for the rescaled run.
ACTIVE = OP_V2
