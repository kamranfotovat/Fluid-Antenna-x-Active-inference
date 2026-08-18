r"""
S2 -- sensing THROUGH the analog network (the observation side; belief update lives in belief.py).

Observation model (replaces the S1 per-port read y = P_S h):
    y = F_RF^H P_S h + n ,   F_RF in C^{M x n_rf}, |F_RF| = 1 (unit modulus),   n ~ CN(0, sigma_e^2 I)
i.e. the M = |S| activated ports are read through n_rf < M mixed analog measurements. The composite
observation matrix A = F_RF^H P_S in C^{n_rf x N} is what belief.update_general consumes, so the
Kalman engine is untouched -- only the measurement operator changes.

Design objective (aggregate, shared across users -- SIMULATION_PLAN_S2 locked decision #1):
    J_sense(F_RF; S) = log2 det( I_{n_rf} + sigma_e^{-2} F_RF^H Sigma_bar_S F_RF )   [bits]
with Sigma_bar_S = sum_k Cov_k(S) the aggregate active-port belief covariance (the same aggregate
E the robust-MMSE precoder uses). This module provides:
  - observation_matrix / sense : build A and draw y  (S2-2)
  - sense_info                 : the objective J_sense                              (S2-3)
  - optimal_unconstrained      : the energy-constrained UPPER BOUND (water-filling)  (S2-3)
The unit-modulus optimizer that maximises J_sense under |F_RF|=1 is S2-4 (design_sensing_matrix).
"""

from __future__ import annotations

import numpy as np

from belief import selection_matrix


# --------------------------------------------------------------------------- S2-2: observation model
def observation_matrix(F_RF, S, N) -> np.ndarray:
    """Composite sensing matrix A = F_RF^H P_S in C^{n_rf x N}.

    F_RF : (M, n_rf) analog combiner over the M = len(S) active ports.
    Setting F_RF = I_M (n_rf = M) gives A = P_S -> per-port sensing (Invariant I1).
    """
    F_RF = np.asarray(F_RF, dtype=complex)
    P = selection_matrix(S, N)                        # (M, N) real 0/1
    return F_RF.conj().T @ P                           # (n_rf, N)


def sense(H_t, F_RF, S, sigma_e2, rng) -> np.ndarray:
    """Draw noisy mixed reads of the active ports: y[k] = F_RF^H h_S[k] + CN(0, sigma_e^2 I).

    H_t : (K, N) true channel this slot. Returns y of shape (K, n_rf).
    Consistent with observation_matrix: y_clean == h @ A.T restricted to the active ports.
    """
    F_RF = np.asarray(F_RF, dtype=complex)
    idx = list(S)
    hS = H_t[:, idx]                                   # (K, M)
    y_clean = hS @ F_RF.conj()                         # (K, n_rf): (F_RF^H h_S)^T per user
    K, n_rf = y_clean.shape
    noise = np.sqrt(sigma_e2 / 2) * (rng.standard_normal((K, n_rf))
                                     + 1j * rng.standard_normal((K, n_rf)))
    return y_clean + noise


# --------------------------------------------------------------------------- S2-3: info objective + bound
def sense_info(F_RF, cov_bar, sigma_e2) -> float:
    """J_sense = log2 det( I + sigma_e^{-2} F_RF^H cov_bar F_RF )  [bits].

    cov_bar : (M, M) aggregate active-port covariance Sigma_bar_S (Hermitian PSD).
    Computed in the n_rf x n_rf form so it stays finite when cov_bar is rank-deficient.
    """
    F_RF = np.asarray(F_RF, dtype=complex)
    G = F_RF.conj().T @ cov_bar @ F_RF                 # (n_rf, n_rf) Hermitian PSD
    n = G.shape[0]
    w = np.linalg.eigvalsh(np.eye(n) + (1.0 / sigma_e2) * G)
    return float(np.sum(np.log2(np.clip(np.real(w), 1e-300, None))))


def _waterfill(lam, sigma_e2, Ptot) -> np.ndarray:
    """Water-filling powers p_i maximising sum log(1 + lam_i p_i / sigma_e^2) s.t. sum p_i = Ptot.
    lam : eigenvalues (>= 0). Directions with lam_i = 0 get zero power."""
    lam = np.asarray(lam, float)
    thr = np.where(lam > 0, sigma_e2 / lam, np.inf)    # inverse-eigenvalue thresholds
    order = np.argsort(thr)                            # ascending threshold
    thr_s = thr[order]
    p_s = np.zeros(len(lam))
    for kk in range(len(lam), 0, -1):
        sub = thr_s[:kk]
        if not np.isfinite(sub).all():
            continue
        mu = (Ptot + sub.sum()) / kk                   # water level for kk active directions
        if mu > sub[-1]:                               # all kk get strictly positive power
            p_s[:kk] = mu - sub
            break
    p = np.zeros(len(lam))
    p[order] = p_s
    return p


def optimal_unconstrained(cov_bar, n_rf, sigma_e2, Ptot=None):
    """UPPER BOUND on J_sense over F with ||F||_F^2 <= Ptot (energy relaxation of |F|=1).

    Any unit-modulus F_RF (M x n_rf) has ||F||_F^2 = M * n_rf, so the natural bound uses that
    total energy (default). The optimum diagonalises in the eigenbasis of cov_bar: put the energy
    (water-filling) on the top-n_rf eigen-directions. Returns (J_bits, F_opt, powers).
    """
    cov_bar = np.asarray(cov_bar, dtype=complex)
    M = cov_bar.shape[0]
    if Ptot is None:
        Ptot = float(M * n_rf)
    w, U = np.linalg.eigh(cov_bar)                     # ascending real eigs, unitary U
    order = np.argsort(np.real(w))[::-1]
    take = order[:n_rf]
    lam = np.real(w[take]).clip(min=0.0)               # top-n_rf eigenvalues
    Ur = U[:, take]                                    # (M, n_rf) eigenvectors
    p = _waterfill(lam, sigma_e2, Ptot)                # (n_rf,)
    F_opt = Ur * np.sqrt(p)[None, :]                   # scale each eigen-column
    J = float(np.sum(np.log2(1.0 + (1.0 / sigma_e2) * lam * p)))
    return J, F_opt, p


# --------------------------------------------------------------------------- S2-4: unit-modulus optimizer
def _egrad(F, cov_bar, c):
    """Euclidean gradient d/dF* of ln det(I + c F^H cov_bar F) = c cov_bar F (I + c F^H cov_bar F)^{-1}.
    (The log2 objective is this over ln2 -- an irrelevant positive scale for ascent.)"""
    G = np.eye(F.shape[1]) + c * (F.conj().T @ cov_bar @ F)
    return c * cov_bar @ F @ np.linalg.inv(G)


def design_sensing_matrix(cov_bar, n_rf, sigma_e2, n_iter=300, n_restart=4,
                          rng=None, warm_start=True, return_trace=False):
    """Maximise J_sense = log2 det(I + sigma_e^{-2} F^H cov_bar F) over UNIT-MODULUS F (M x n_rf).

    Riemannian gradient ascent on the complex-circle (torus) manifold: project the Euclidean
    gradient onto the per-entry tangent (remove the radial part), take a backtracked step, and
    RETRACT by exp(i*angle(.)) so |F| = 1 holds exactly at every iterate. Backtracking accepts a
    step only if it increases J -> the objective is monotone non-decreasing (Invariant I7).

    Warm-started from the phases of the unconstrained water-filling optimum (the S2-3 bound), plus
    random unit-modulus restarts; the best restart is returned. Returns (F, J[, trace]) where
    trace is the accepted-J history of the winning restart (for the monotonicity gate).
    """
    cov_bar = np.asarray(cov_bar, dtype=complex)
    cov_bar = 0.5 * (cov_bar + cov_bar.conj().T)
    M = cov_bar.shape[0]
    c = 1.0 / sigma_e2
    rng = np.random.default_rng(0) if rng is None else rng

    inits = []
    if warm_start:
        _, F_opt, p = optimal_unconstrained(cov_bar, n_rf, sigma_e2)
        F0 = np.exp(1j * np.angle(F_opt))              # keep the phases of the eigen-solution
        thr = 1e-12 * max(float(p.max()), 1e-30)
        for r in range(n_rf):                          # zero-power (unused) columns -> random phase
            if p[r] <= thr:
                F0[:, r] = np.exp(1j * rng.uniform(0, 2 * np.pi, size=M))
        inits.append(F0)
    while len(inits) < max(1, n_restart):
        inits.append(np.exp(1j * rng.uniform(0, 2 * np.pi, size=(M, n_rf))))

    bestJ, bestF, bestTr = -np.inf, None, None
    for F in inits:
        F = F.astype(complex)
        J = sense_info(F, cov_bar, sigma_e2)
        step = 1.0
        trace = [J]
        for _ in range(n_iter):
            eg = _egrad(F, cov_bar, c)
            rg = eg - np.real(eg * np.conj(F)) * F     # tangent projection on the torus
            accepted = False
            for _bt in range(40):                      # backtracking line search (monotone accept)
                Fn = np.exp(1j * np.angle(F + step * rg))   # retraction -> exact unit modulus
                Jn = sense_info(Fn, cov_bar, sigma_e2)
                if Jn > J + 1e-12:
                    F, J = Fn, Jn
                    step *= 1.5
                    accepted = True
                    break
                step *= 0.5
            trace.append(J)
            if not accepted or step < 1e-10:
                break
        if J > bestJ:
            bestJ, bestF, bestTr = J, F, trace
    if return_trace:
        return bestF, bestJ, bestTr
    return bestF, bestJ
