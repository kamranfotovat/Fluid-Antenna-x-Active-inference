"""
Step 1 verification -- precoder + rate module (full CSI).

Gate before Step 2. Thorough console report:
  A. Effective channel structure: ZF nulls inter-user interference; MMSE trades a little
     leakage for higher SINR.
  B. K=1 closed-form sanity: SINR must equal P * ||h||^2 / sigma^2 (matched filter).
  C. SNR sweep: MMSE >= ZF at every SNR, gap closes as SNR -> inf.
  D. Array gain: sum-rate grows as we activate more ports M.

Run from sim/:  python verify_step1.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator                         # noqa: E402
from precoding import (                                       # noqa: E402
    zf_precoder, mmse_precoder, sinr_and_rates, sum_rate, max_offdiag_leakage,
)

np.set_printoptions(precision=3, suppress=True)


def active_channel(h, S):
    """h: (K, N) full channel -> H: (M, K) on the active ports S (columns = users)."""
    return h[:, S].T                                           # (M, K)


def main():
    K, N, M = 3, 25, 5
    sigma2 = 1e-3
    P = 1.0
    sim = ChannelSimulator(Nx=5, Ny=5, K=K, rho=0.9, beta=1.0, seed=7)

    all_pass = True
    print("=" * 70)
    print("STEP 1 -- PRECODER + RATE MODULE  (full CSI)")
    print(f"K={K} users, N={N} ports, M={M} active, sigma^2={sigma2}, P={P}")
    print("=" * 70)

    # ---------------------------------------------------------------- A. structure
    print("\n[A] Effective channel  eff[k,j] = h_k^H w_j  (diagonal = signal, off = interference)")
    h = sim.reset()
    S = np.arange(M)                                           # first M ports
    H = active_channel(h, S)                                   # (M, K)

    W_zf = zf_precoder(H, P=P)
    W_mmse = mmse_precoder(H, P=P, sigma2=sigma2)

    eff_zf = np.abs(H.conj().T @ W_zf)
    eff_mmse = np.abs(H.conj().T @ W_mmse)
    print("   |eff| under ZF   :\n", eff_zf)
    print("   |eff| under MMSE :\n", eff_mmse)

    leak_zf = max_offdiag_leakage(H, W_zf)
    leak_mmse = max_offdiag_leakage(H, W_mmse)
    ok = leak_zf < 1e-9
    all_pass &= ok
    print(f"   ZF   max interference leakage = {leak_zf:.2e}  (expect ~0)  -> {'PASS' if ok else 'FAIL'}")
    print(f"   MMSE max interference leakage = {leak_mmse:.2e}  (nonzero by design)")

    r_zf = sum_rate(H, W_zf, sigma2)
    r_mmse = sum_rate(H, W_mmse, sigma2)
    print(f"   sum-rate  ZF = {r_zf:6.3f}  |  MMSE = {r_mmse:6.3f} bits/s/Hz  "
          f"(MMSE >= ZF: {'PASS' if r_mmse >= r_zf - 1e-9 else 'FAIL'})")
    all_pass &= (r_mmse >= r_zf - 1e-9)

    # ---------------------------------------------------------------- B. K=1 closed form
    print("\n[B] K=1 closed-form sanity: SINR must equal P*||h||^2/sigma^2 (matched filter, no interf.)")
    sim1 = ChannelSimulator(Nx=5, Ny=5, K=1, rho=0.9, beta=1.0, seed=3)
    h1 = sim1.reset()
    H1 = active_channel(h1, S)                                 # (M, 1)
    W1 = mmse_precoder(H1, P=P, sigma2=sigma2)                 # reduces to scaled matched filter
    sinr1, _ = sinr_and_rates(H1, W1, sigma2)
    closed = P * np.linalg.norm(H1) ** 2 / sigma2
    rel = abs(sinr1[0] - closed) / closed
    ok = rel < 1e-6
    all_pass &= ok
    print(f"   measured SINR = {sinr1[0]:.4f} | closed form P||h||^2/sigma^2 = {closed:.4f} "
          f"| rel-err = {rel:.2e}  -> {'PASS' if ok else 'FAIL'}")

    # ---------------------------------------------------------------- C. SNR sweep
    print("\n[C] SNR sweep (mean sum-rate over 400 realizations). Expect MMSE >= ZF; gap -> 0 at high SNR.")
    print(f"   {'SNR(dB)':>8} | {'ZF':>8} | {'MMSE':>8} | {'gain':>7}")
    mc = 400
    monotone_ok = True
    prev_gap = None
    for snr_db in [-10, 0, 10, 20, 30]:
        s2 = P / 10 ** (snr_db / 10)                          # fix P, vary noise
        acc_zf, acc_mmse = 0.0, 0.0
        for _ in range(mc):
            hh = sim.reset()
            HH = active_channel(hh, S)
            acc_zf += sum_rate(HH, zf_precoder(HH, P), s2)
            acc_mmse += sum_rate(HH, mmse_precoder(HH, P, s2), s2)
        rz, rm = acc_zf / mc, acc_mmse / mc
        gap = rm - rz
        print(f"   {snr_db:8d} | {rz:8.3f} | {rm:8.3f} | {gap:7.3f}")
        if rm < rz - 1e-6:
            monotone_ok = False
        prev_gap = gap
    all_pass &= monotone_ok
    print(f"   MMSE >= ZF at all SNR: {'PASS' if monotone_ok else 'FAIL'}")

    # ---------------------------------------------------------------- D. array gain
    print("\n[D] Array gain: activate more ports M -> more sum-rate (mean over 400, MMSE, sigma^2=1e-3).")
    print(f"   {'M':>3} | {'sum-rate':>9}")
    prev = -np.inf
    mono = True
    for Mtest in [3, 4, 5, 7, 10]:
        St = np.arange(Mtest)
        acc = 0.0
        for _ in range(mc):
            hh = sim.reset()
            HH = active_channel(hh, St)
            acc += sum_rate(HH, mmse_precoder(HH, P, sigma2), sigma2)
        val = acc / mc
        print(f"   {Mtest:3d} | {val:9.3f}")
        if val < prev - 1e-3:
            mono = False
        prev = val
    all_pass &= mono
    print(f"   monotincreasing in M: {'PASS' if mono else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"STEP 1 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
