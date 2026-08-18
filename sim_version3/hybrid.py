r"""
Step 13 (version 3) -- Hybrid beamforming: factorize a digital precoder into an analog
RF network + a small baseband precoder.

Motivation
----------
Fully-digital beamforming needs one RF chain per active port (n_rf = M). That is expensive
when M is large. Hybrid beamforming keeps only n_rf < M RF chains and drives the M active
ports through a fully-connected network of unit-modulus phase shifters:

    x = F_RF W_BB s ,      F_RF in C^{M x n_rf}  (|[F_RF]_{m,r}| = 1),   W_BB in C^{n_rf x K}

The effective precoder is  W_eff = F_RF W_BB  in C^{M x K}. We reuse the tutorial's
"factorize a target digital precoder" route (main(3)(1).pdf, Sec. 4, step 2b):

    min_{F_RF, W_BB}  || F* - F_RF W_BB ||_F     s.t.  |[F_RF]_{m,r}| = 1 .

CRUCIAL for our paper: the target F* is the *belief-based robust-MMSE precoder* we already
build (efe.robust_mmse_from_belief). So hybrid is a pure POST-PROCESSING of the existing
AIF precoder -- the belief, EFE selection and Kalman sensing are all untouched. Only the
transmit stage is projected onto the hardware-feasible set.

Algorithm (block coordinate AltMin)
-----------------------------------
Alternating minimization, fully-connected, infinite-resolution phase shifters:
  1) fix F_RF -> W_BB = argmin ||F* - F_RF W_BB||  = pinv(F_RF) F*        (least squares)
  2) fix W_BB -> update every phase shifter [F_RF]_{m,r} by its 1-D optimum. Holding all
     other entries fixed, the residual e = f*_m - sum_{r'!=r} [F_RF]_{m,r'} wbb_{r'} makes
     the best unit-modulus entry  [F_RF]_{m,r} = exp( j * angle( e . wbb_r^H ) ). Sweeping
     all (m,r) is monotone coordinate descent -- each step can only lower ||F* - F_RF W_BB||.
Repeat until the error stops improving; rescale so tr(W_eff W_eff^H) = P.

(The naive "F_RF = exp(j*angle(F* pinv(W_BB)))" phase-extraction step is degenerate once
n_rf > K -- W_BB W_BB^H is rank-K singular -- so we use the exact per-entry update instead.)

Design fact that anchors the experiments: for the fully-connected, infinite-resolution
structure, n_rf >= 2K RF chains suffice to represent ANY digital precoder exactly
(Sohrabi & Yu 2016; Zhang, Molisch & Kung 2005) -- any complex entry is the sum of two
unit-modulus phasors. So at n_rf >= 2K the hybrid rate matches the digital rate; below 2K
the approximation error grows and the rate degrades gracefully. That threshold (= 6 for
K=3) is the headline of the n_rf sweep.
"""

from __future__ import annotations

import numpy as np


def _update_FRF(F_star, F_RF, W_BB):
    """One monotone coordinate-descent sweep over all phase shifters: for each RF chain r,
    set the whole column F_RF[:, r] to its per-entry unit-modulus optimum given the rest.
    Vectorized over antennas m. Lowers ||F* - F_RF W_BB|| (or leaves it unchanged)."""
    resid = F_star - F_RF @ W_BB                          # M x K current residual
    for r in range(F_RF.shape[1]):
        wr = W_BB[r]                                      # K  (chain r's baseband row)
        e = resid + np.outer(F_RF[:, r], wr)             # add chain r back in -> M x K
        c = e @ wr.conj()                                 # M   correlation with chain r
        new = np.where(np.abs(c) > 0, np.exp(1j * np.angle(c)), F_RF[:, r])
        resid = e - np.outer(new, wr)                     # subtract the updated chain
        F_RF[:, r] = new
    return F_RF


def factorize_hybrid(F_star, n_rf, P=1.0, n_iter=50, tol=1e-10, rng=None):
    r"""Approximate a target digital precoder F* (M x K) by a fully-connected hybrid
    precoder F_RF (M x n_rf, unit modulus) * W_BB (n_rf x K), power-normalized to P.

    Returns (F_RF, W_BB, W_eff) with W_eff = F_RF @ W_BB scaled so tr(W_eff W_eff^H) = P.
    Block coordinate AltMin: LS for W_BB, exact per-entry unit-modulus updates for F_RF.
    Warm-started from the phases of F* (so n_rf >= K starts near a good basin), which makes
    the >= 2K exact-representation regime reachable. If F* is all-zero (cold-start belief ->
    no beam), returns zeros -> rate 0, not NaN.
    """
    F_star = np.asarray(F_star, dtype=complex)
    M, K = F_star.shape
    rng = np.random.default_rng(0) if rng is None else rng

    nf = float(np.linalg.norm(F_star))
    if not np.isfinite(nf) or nf < 1e-300:
        z = np.zeros((M, K), dtype=complex)
        return np.ones((M, n_rf), dtype=complex), np.zeros((n_rf, K), dtype=complex), z

    # warm start: first min(n_rf,K) chains = phases of F*'s columns; any extra chains random
    F_RF = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(M, n_rf)))
    ncopy = min(n_rf, K)
    F_RF[:, :ncopy] = np.exp(1j * np.angle(F_star[:, :ncopy]))

    prev = np.inf
    for _ in range(n_iter):
        W_BB = np.linalg.lstsq(F_RF, F_star, rcond=None)[0]         # n_rf x K  (fix F_RF)
        F_RF = _update_FRF(F_star, F_RF, W_BB)                       # per-entry (fix W_BB)
        err = float(np.linalg.norm(F_star - F_RF @ W_BB))
        if abs(prev - err) < tol * max(nf, 1.0):
            break
        prev = err
    W_BB = np.linalg.lstsq(F_RF, F_star, rcond=None)[0]             # final W_BB for latest F_RF

    W_eff = F_RF @ W_BB
    denom = float(np.real(np.trace(W_eff @ W_eff.conj().T)))
    if not np.isfinite(denom) or denom < 1e-300:
        z = np.zeros((M, K), dtype=complex)
        return F_RF, np.zeros_like(W_BB), z
    scale = np.sqrt(P / denom)
    return F_RF, W_BB * scale, W_eff * scale


def hybridize(F_star, n_rf, P=1.0, rng=None):
    """Convenience: return only the M x K effective hybrid precoder W_eff = F_RF W_BB
    (power P) that approximates the digital target F*. n_rf=None -> return F* unchanged
    (fully-digital), so callers can pass a single knob through."""
    if n_rf is None:
        return np.asarray(F_star, dtype=complex)
    return factorize_hybrid(F_star, n_rf, P=P, rng=rng)[2]


def factorization_loss_db(F_star, W_eff):
    """Relative approximation error 10 log10( min_c ||F* - c W_eff||^2 / ||F*||^2 ) in dB.
    The optimal complex scalar c removes the arbitrary gain/phase left by power normalization,
    so this measures how well the hybrid STRUCTURE represents F*. ~ -inf when n_rf >= 2K
    (exact representation), rising toward 0 dB as n_rf shrinks below 2K."""
    F_star = np.asarray(F_star, dtype=complex)
    nf2 = float(np.vdot(F_star, F_star).real)
    we2 = float(np.vdot(W_eff, W_eff).real)
    if nf2 < 1e-300 or we2 < 1e-300:
        return float("nan")
    c = np.vdot(W_eff, F_star) / we2                      # LS scalar: min_c ||F* - c W_eff||
    err2 = float(np.vdot(F_star - c * W_eff, F_star - c * W_eff).real)
    return float(10.0 * np.log10(max(err2, 1e-300) / nf2))
