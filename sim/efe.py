r"""
Step 4 -- Expected Free Energy (EFE) terms for port selection under partial CSI.

Consumes the Kalman belief q(h_k) = CN(mu_k, Sigma_k) from `belief.py` and scores a
candidate active port set S with the three terms of EFE_DESIGN.md:

    G(S) = - alpha * PragmaticValue(S)     (prefer high expected rate)
           - beta  * EpistemicValue(S)     (prefer resolving channel uncertainty)
           + SwitchingCost(S)              (prefer not churning ports)

Select S* = argmin_S G(S) with |S| = M. This module defines the three terms in
isolation (Step 4); the greedy submodular selector that assembles them is Step 5.

Notation on the active ports (|S| = M):
    m_k   = P_S mu_k    in C^M     active-port belief mean for user k  (columns of Hhat)
    Cov_k = P_S Sigma_k P_S^H in C^{MxM}   active-port belief covariance for user k
    W = [w_1,...,w_K] in C^{MxK}   robust-MMSE precoder built from (Hhat, sum_k Cov_k)

All information/rate quantities are in BITS (log2) so alpha, beta, eta_sw are
dimensionless trade-off weights.
"""

from __future__ import annotations

import itertools
import numpy as np

from precoding import mmse_precoder


# --------------------------------------------------------------------------- belief -> active-port views
def active_mean(bel, S) -> np.ndarray:
    """Hhat in C^{MxK}: column k = m_k = P_S mu_k (active-port mean for user k)."""
    idx = list(S)
    return np.stack([bel.mu[k][idx] for k in range(bel.K)], axis=1)


def active_covs(bel, S) -> list[np.ndarray]:
    """List of K matrices Cov_k = P_S Sigma_k P_S^H in C^{MxM}."""
    idx = list(S)
    return [bel.Sigma[k][np.ix_(idx, idx)] for k in range(bel.K)]


def robust_mmse_from_belief(bel, S, sigma2=1e-3, P=1.0):
    """Robust-MMSE precoder from the belief: uses the active means Hhat and the
    AGGREGATE CSI-error covariance E = sum_k Cov_k (EFE_DESIGN Sec. 3). Returns
    (W, Hhat, covs)."""
    Hhat = active_mean(bel, S)
    covs = active_covs(bel, S)
    E = np.sum(covs, axis=0)                          # M x M aggregate error cov
    W = mmse_precoder(Hhat, P=P, sigma2=sigma2, error_cov=E)
    return W, Hhat, covs


# --------------------------------------------------------------------------- (1) pragmatic value
def pragmatic_value(bel, S, sigma2=1e-3, P=1.0, return_rates=False):
    """Expected robust-MMSE sum-rate under the belief (imperfect-CSI lower bound).

        SINR_k = |m_k^H w_k|^2
                 ---------------------------------------------------------------
                 sum_{j!=k} |m_k^H w_j|^2  +  sum_j w_j^H Cov_k w_j  +  sigma^2

    The middle term (grows with Cov_k) is the CSI-error penalty: when the agent is
    unsure about user k's active channel, its effective SINR drops -> the value is
    automatically conservative. Bits.
    """
    if len(S) == 0:
        return (0.0, np.zeros(bel.K)) if return_rates else 0.0
    W, Hhat, covs = robust_mmse_from_belief(bel, S, sigma2, P)
    eff = Hhat.conj().T @ W                            # K x K, eff[k,j] = m_k^H w_j
    power = np.abs(eff) ** 2
    signal = np.diag(power)
    interf = power.sum(axis=1) - signal                # sum_{j!=k} |m_k^H w_j|^2
    # CSI-error term e_k = sum_j w_j^H Cov_k w_j = trace(W^H Cov_k W)
    ek = np.array([np.real(np.trace(W.conj().T @ covs[k] @ W)) for k in range(bel.K)])
    sinr = signal / (interf + ek + sigma2)
    rates = np.log2(1.0 + sinr)
    return (float(rates.sum()), rates) if return_rates else float(rates.sum())


# --------------------------------------------------------------------------- (2) epistemic value
def epistemic_value(bel, S, return_per_user=False):
    """Expected information gain from observing S = mutual info I(h_k; y_k) (bits):

        I_k(S) = log2 det( I_M + (1/sigma_e^2) Cov_k )

    Computed in the M x M form so it stays finite even when Sigma is singular (our R
    is rank-deficient). Monotone & submodular in S -> underpins the greedy guarantee.
    """
    idx = list(S)
    M = len(idx)
    if M == 0:
        return (0.0, np.zeros(bel.K)) if return_per_user else 0.0
    I_M = np.eye(M)
    inv_se2 = 1.0 / bel.sigma_e2
    vals = np.empty(bel.K)
    for k in range(bel.K):
        Cov = bel.Sigma[k][np.ix_(idx, idx)]
        w = np.linalg.eigvalsh(I_M + inv_se2 * Cov)    # Hermitian PD -> real eigs >= 1
        vals[k] = np.sum(np.log2(np.clip(np.real(w), 1e-300, None)))
    return (float(vals.sum()), vals) if return_per_user else float(vals.sum())


# --------------------------------------------------------------------------- (3) switching cost
def switching_cost(S, S_prev, eta_sw=1.0, e_sw=1.0) -> float:
    """eta_sw * e_sw * |S XOR S_prev| (symmetric difference = number of ports changed)."""
    if S_prev is None:
        return 0.0
    return float(eta_sw * e_sw * len(set(S) ^ set(S_prev)))


# --------------------------------------------------------------------------- combined EFE
def expected_free_energy(bel, S, S_prev=None, alpha=1.0, beta=1.0,
                         eta_sw=1.0, e_sw=1.0, sigma2=1e-3, P=1.0):
    """G(S) = -alpha*Pragmatic -beta*Epistemic +Switching. Returns (G, terms-dict).
    Lower is better. The greedy minimiser over |S|=M is Step 5."""
    prag = pragmatic_value(bel, S, sigma2, P)
    epis = epistemic_value(bel, S)
    swc = switching_cost(S, S_prev, eta_sw, e_sw)
    G = -alpha * prag - beta * epis + swc
    return G, {"pragmatic": prag, "epistemic": epis, "switching": swc}


# --------------------------------------------------------------------------- greedy selection (Step 5)
def _switch_marginal(n, S_prev_set, eta_sw, e_sw) -> float:
    """Marginal switching cost of adding port n. Switching is MODULAR:
    |S XOR S_prev| = |S_prev| + sum_{n in S} (1 - 2*[n in S_prev]). So adding a
    NEW port costs +eta*e, re-using a previous port credits -eta*e."""
    return eta_sw * e_sw * (1.0 - 2.0 * (1.0 if n in S_prev_set else 0.0))


def greedy_select(bel, M, S_prev=None, alpha=1.0, beta=1.0, eta_sw=1.0, e_sw=1.0,
                  sigma2=1e-3, P=1.0, return_trace=False):
    """Greedily build S (|S|=M) minimising G(S) = -alpha*prag -beta*epis +switch, i.e.
    maximising J(S) = alpha*prag + beta*epis - switch. Each step adds the port with the
    largest marginal J. O(N*M) objective evaluations. The epistemic part is monotone
    submodular -> (1-1/e) guarantee on that component; pragmatic is a marginal heuristic.
    """
    N = bel.N
    prev_set = set() if S_prev is None else set(S_prev)
    A, remaining = [], set(range(N))
    prag_A, epis_A = 0.0, 0.0
    trace = []
    for _ in range(M):
        best = (-np.inf, None, None, None)  # (marginal, n, prag_cand, epis_cand)
        for n in remaining:
            cand = tuple(A + [n])
            prag_c = pragmatic_value(bel, cand, sigma2, P)
            epis_c = epistemic_value(bel, cand)
            marg = (alpha * (prag_c - prag_A) + beta * (epis_c - epis_A)
                    - _switch_marginal(n, prev_set, eta_sw, e_sw))
            if marg > best[0]:
                best = (marg, n, prag_c, epis_c)
        _, n_star, prag_A, epis_A = best
        A.append(n_star); remaining.remove(n_star)
        trace.append((n_star, best[0]))
    S = tuple(sorted(A))
    return (S, trace) if return_trace else S


def exhaustive_select(bel, M, S_prev=None, alpha=1.0, beta=1.0, eta_sw=1.0, e_sw=1.0,
                      sigma2=1e-3, P=1.0):
    """Brute force over all C(N, M) sets -> the S minimising G (use only for small N).
    Returns (S*, J*) with J* = -G* (the maximised objective)."""
    best_S, best_J = None, -np.inf
    for S in itertools.combinations(range(bel.N), M):
        G, _ = expected_free_energy(bel, S, S_prev=S_prev, alpha=alpha, beta=beta,
                                    eta_sw=eta_sw, e_sw=e_sw, sigma2=sigma2, P=P)
        if -G > best_J:
            best_J, best_S = -G, S
    return best_S, best_J
