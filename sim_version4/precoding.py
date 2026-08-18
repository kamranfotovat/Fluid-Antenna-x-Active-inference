r"""
Step 1 -- Precoder + rate module.

Downlink FAS: the BS activates M ports (= M transmit antennas this slot) and serves
K single-antenna users. For each user k we have the channel on the active ports
h_k in C^M. Stack them as the columns of the effective channel:

    H = [h_1, ..., h_K]  in  C^{M x K}     (column k = user k's active-port channel)

The BS sends  x = W s = sum_j w_j s_j,  with unit-power symbols s_j and precoder
W = [w_1,...,w_K] in C^{M x K}. User k receives

    y_k = h_k^H x + n_k = (h_k^H w_k) s_k  +  sum_{j!=k} (h_k^H w_j) s_j  +  n_k,
                          \-- desired --/     \------- inter-user interf. -------/   noise CN(0, sigma^2)

    SINR_k = |h_k^H w_k|^2 / ( sum_{j!=k} |h_k^H w_j|^2 + sigma^2 )
    R_k    = log2(1 + SINR_k)                                            (Eq. 5)

Two precoders are provided:
  - Zero-Forcing (ZF): forces h_k^H w_j = 0 for j!=k -> zero interference, but amplifies
    noise when the users' active channels are near-collinear.
  - (robust) MMSE / regularized-ZF: balances interference vs noise; strictly better at
    finite SNR and numerically safe even when M < K. Passing an error covariance makes it
    the *robust* MMSE that will consume the belief covariance Sigma in later steps.

All precoders are scaled to a total power budget  tr(W W^H) = P  for a fair comparison.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- precoders
def zf_precoder(H: np.ndarray, P: float = 1.0) -> np.ndarray:
    """Zero-forcing precoder for the M x K active channel H (columns = users).

    Direction  G = H (H^H H)^{-1}  satisfies  H^H G = I  (=> zero inter-user interference).
    Then scale so tr(W W^H) = P. Requires M >= K (else H^H H is singular).
    """
    M, K = H.shape
    assert M >= K, f"ZF needs M >= K, got M={M}, K={K}"
    G = H @ np.linalg.inv(H.conj().T @ H)          # M x K, H^H G = I exactly
    scale = np.sqrt(P / np.real(np.trace(G @ G.conj().T)))
    return scale * G


def mmse_precoder(
    H: np.ndarray,
    P: float = 1.0,
    sigma2: float = 1e-3,
    error_cov: np.ndarray | None = None,
) -> np.ndarray:
    """(Robust) transmit-MMSE / regularized-ZF precoder.

    W ~ (H H^H + E + (K*sigma^2/P) I)^{-1} H,  then scaled to tr(W W^H) = P.

    - With error_cov = None this is the standard MMSE precoder (regularized ZF). The
      regularization (K*sigma^2/P) I trades interference-nulling against noise, so it is
      never worse than ZF and stays invertible even for M < K.
    - error_cov E = sum_k P_S Sigma_k P_S^H is the aggregate channel-estimate-error
      covariance on the active ports. Adding it makes the design *robust*: the more the
      agent is unsure (large Sigma), the more conservative W becomes. This is the hook the
      later EFE pragmatic term uses; here (Step 1, full CSI) we call it with error_cov=None.
    """
    M, K = H.shape
    A = H @ H.conj().T + (K * sigma2 / P) * np.eye(M)
    if error_cov is not None:
        A = A + error_cov
    G = np.linalg.solve(A, H)                        # M x K
    denom = np.real(np.trace(G @ G.conj().T))
    if not np.isfinite(denom) or denom < 1e-300:     # zero channel estimate (e.g. mu=0 cold start)
        return np.zeros_like(G)                       # -> no beam, rate 0 (correct, not NaN)
    scale = np.sqrt(P / denom)
    return scale * G


# --------------------------------------------------------------------------- rate / SINR
def sinr_and_rates(H_true: np.ndarray, W: np.ndarray, sigma2: float = 1e-3):
    """Per-user SINR and rate for precoder W evaluated on the *true* active channel H_true.

    H_true and W are both M x K. Crucially H_true may differ from the channel W was built
    from (stale or estimated) -- then the off-diagonal terms are nonzero and show up as
    residual interference. Returns (sinr[K], rate[K]).
    """
    eff = H_true.conj().T @ W                        # K x K, eff[k, j] = h_k^H w_j
    power = np.abs(eff) ** 2
    signal = np.diag(power)                           # |h_k^H w_k|^2
    interf = power.sum(axis=1) - signal              # sum_{j!=k} |h_k^H w_j|^2
    sinr = signal / (interf + sigma2)
    rate = np.log2(1.0 + sinr)
    return sinr, rate


def sum_rate(H_true: np.ndarray, W: np.ndarray, sigma2: float = 1e-3) -> float:
    """Convenience: total sum-rate sum_k R_k."""
    return float(sinr_and_rates(H_true, W, sigma2)[1].sum())


def max_offdiag_leakage(H_true: np.ndarray, W: np.ndarray) -> float:
    """Largest |h_k^H w_j|, j != k -- the residual inter-user interference amplitude.
    ~0 for ZF under perfect CSI; a direct check that interference is nulled."""
    eff = H_true.conj().T @ W
    K = eff.shape[0]
    off = np.abs(eff) * (1.0 - np.eye(K))
    return float(off.max())
