r"""
Position algebra + sensing for the liquid-column FAS (sim_version5).

A droplet configuration is a POSITION VECTOR i = (p_0, ..., p_{N_t-1}) with p_c in {0..N_p-1}
the height of column c's droplet. It maps to a global port-index set S (one port per column):

    global index of (column c, height p)  ->  n = p * N_t + c      (grid is column-fastest)
    column(n) = n % N_t                          height(n) = n // N_t

These helpers convert between the two views and provide the one-per-column noisy sensing used to
drive the Kalman belief. All heavy lifting (correlation, belief, precoding) is reused from the S1
modules unchanged.
"""

from __future__ import annotations

import numpy as np


def column_of(n, N_t):
    """Column index (x) of global port n."""
    return np.asarray(n) % N_t


def height_of(n, N_t):
    """Height index (y) of global port n."""
    return np.asarray(n) // N_t


def pos_to_ports(i, N_t):
    """Position vector i (length N_t, heights) -> sorted tuple of N_t global port indices."""
    i = np.asarray(i, dtype=int)
    assert i.shape == (N_t,), f"position vector must have length N_t={N_t}"
    assert np.all((i >= 0)), "heights must be >= 0"
    ports = i * N_t + np.arange(N_t)                 # n = p*N_t + c
    return tuple(int(n) for n in np.sort(ports))


def ports_to_pos(S, N_t):
    """Global port set S (exactly one port per column) -> position vector (heights per column)."""
    S = list(S)
    pos = -np.ones(N_t, dtype=int)
    for n in S:
        c = n % N_t
        assert pos[c] == -1, f"more than one active port in column {c} -- not one-per-column"
        pos[c] = n // N_t
    assert np.all(pos >= 0), "every column must have exactly one active port"
    return pos


def sense(h_t, S, sigma_e2, rng):
    """Noisy pilots on the active ports S: y_k = h_k[S] + CN(0, sigma_e^2 I). Shape (K, |S|).

    h_t : (K, N) true channel this slot.  Mirrors the S1 observation model exactly."""
    idx = list(S)
    K = h_t.shape[0]
    noise = np.sqrt(sigma_e2 / 2) * (rng.standard_normal((K, len(idx)))
                                     + 1j * rng.standard_normal((K, len(idx))))
    return h_t[:, idx] + noise
