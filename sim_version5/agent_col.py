r"""
Belief wiring for the liquid-column FAS (sim_version5).

Thin layer over the reused S1 KalmanBelief: build a belief on the full 2-D column correlation, and
run the one-per-column sense->update for a droplet configuration. The heavy Kalman math is unchanged
from S1 (belief.py); this only adapts the interface to the position-vector view. Selection / EFE /
planning arrive in later steps.
"""

from __future__ import annotations

import numpy as np

from belief import KalmanBelief
from columns import pos_to_ports, sense


def make_belief(op, R=None) -> KalmanBelief:
    """KalmanBelief on the column array. Pass R=op.R_block() for the independent-column ablation."""
    R = op.R() if R is None else R
    return KalmanBelief(R=R, beta=op.beta, rho=op.rho, sigma_e2=op.sigma_e2)


def sense_and_update(bel, h_t, i, op, rng):
    """Sense the droplet ports of configuration i and Kalman-update the belief. Returns S."""
    S = pos_to_ports(i, op.N_t)
    y = sense(h_t, S, op.sigma_e2, rng)
    bel.update(S, y)
    return S
