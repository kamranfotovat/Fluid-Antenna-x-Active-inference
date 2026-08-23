r"""
Operating point for sim_version5 -- the LIQUID-COLUMN FAS (Paper 2, Option B).

N_t liquid columns, each an N_p-port 1-D tube holding ONE droplet. Columns are spaced lambda/3
(dense) so the WHOLE array is one correlated 2-D Jakes field -- the agent exploits inter-column
correlation for informed repositioning (vs IDET, which spaces > lambda/2 to discard it).

Geometry maps onto the existing ChannelSimulator grid:
    Nx = N_t columns,  Wx = (N_t-1)*col_spacing   -> x-pitch = col_spacing (lambda/3)
    Ny = N_p ports,    Wy = (N_p-1)*pitch          -> y-pitch = pitch       (lambda/10)
Grid is row-major with x (column) fastest, so global port index:
    n = p * N_t + c     (port height p in {0..N_p-1}, column c in {0..N_t-1})

See SIMULATION_PLAN_V5.md for the gated build and invariants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from channel import port_positions, spatial_correlation


@dataclass(frozen=True)
class ColumnOperatingPoint:
    # geometry
    N_t: int = 10                 # columns  (= M active elements, one droplet each)
    N_p: int = 21                 # ports per column
    pitch: float = 0.1            # within-column port spacing (wavelengths) = lambda/10
    col_spacing: float = 1.0 / 3  # inter-column spacing (wavelengths) = lambda/3

    # action constraints
    delta_max: int = 7            # max ports a droplet may move per slot (0.7 lambda); sweep 2-7
    d_min: float = 0.5            # min spacing (lambda) between ANY two active droplets (lambda/2)

    # scenario (inherits OP_V3)
    K: int = 3
    rho: float = 0.9
    sigma2: float = 0.03          # 15 dB data noise
    sigma_e2: float = 1e-3        # ~20 dB pilot noise
    alpha: float = 1.0            # pragmatic weight
    beta_w: float = 0.25          # epistemic (exploration) weight
    eta_mv: float = 0.5           # movement-cost weight (per port travelled); sweepable
    P: float = 1.0

    # -- derived -------------------------------------------------------------
    @property
    def M(self) -> int:
        return self.N_t                       # exactly one active port per column

    @property
    def N(self) -> int:
        return self.N_t * self.N_p

    @property
    def Nx(self) -> int:
        return self.N_t

    @property
    def Ny(self) -> int:
        return self.N_p

    @property
    def Wx(self) -> float:
        return (self.N_t - 1) * self.col_spacing

    @property
    def Wy(self) -> float:
        return (self.N_p - 1) * self.pitch

    @property
    def beta(self) -> np.ndarray:
        base = np.array([1.0, 0.7, 1.3])
        return base if self.K == len(base) else np.linspace(0.7, 1.3, self.K)

    def positions(self) -> np.ndarray:
        """(N, 2) port coordinates in wavelengths; row-major, column (x) fastest."""
        return port_positions(self.Nx, self.Ny, self.Wx, self.Wy)

    def R(self) -> np.ndarray:
        """Full 2-D Jakes correlation over all N ports (Option B -- coupled columns)."""
        return spatial_correlation(self.positions())

    def R_block(self) -> np.ndarray:
        """Block-diagonal correlation: within-column Jakes, cross-column ZEROED (Option A
        ablation -- independent columns). PSD (blkdiag of PSD blocks)."""
        pos = self.positions()
        R = spatial_correlation(pos)
        cols = np.arange(self.N) % self.N_t          # column index of each global port
        same = cols[:, None] == cols[None, :]        # True where two ports share a column
        return R * same                              # keep intra-column, zero inter-column

    def label(self) -> str:
        return (f"N_t={self.N_t} cols x N_p={self.N_p} ports (N={self.N}), "
                f"pitch={self.pitch:g}λ, col_spacing={self.col_spacing:.3f}λ, "
                f"footprint {self.Wx:.2f}x{self.Wy:.2f}λ, M={self.M}, "
                f"Δmax={self.delta_max}, d_min={self.d_min:g}λ, "
                f"K={self.K}, ρ={self.rho}, β_w={self.beta_w}, η_mv={self.eta_mv}")


OP_B = ColumnOperatingPoint()                        # main model (dense, coupled columns)
ACTIVE = OP_B
