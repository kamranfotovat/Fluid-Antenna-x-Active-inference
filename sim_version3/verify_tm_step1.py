r"""
TM-1a -- realistic (noise-limited) aging-error reduction from AR(p) prediction, per port.

Per-port AR(p) Kalman over a Jakes truth with NOISY pilots (sigma_e2). This is the honest version of
TM-0: the achievable prediction error is now limited by pilot noise and model order, not the
noiseless autocorrelation. Two errors that map directly to rate:
  PRED error  = |predicted h(t) before this slot's obs - true|^2   (caps PREDICT-then-precode / the
                aging of un-piloted served ports in PARTIAL sensing)
  POST error  = |estimate after obs - true|^2                       (observe-then-precode; ~sigma_e2)

We report both vs order p and Doppler. p=1 is the current AR(1) model. Gate: AR(p>1) PRED error is
well below AR(1) at moderate Doppler (the aging error the temporal upgrade removes).

Run:  python verify_tm_step1.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from temporal import jakes_series, ar_coeffs_yw, ar_kalman_track

SIGMA_E2 = 1e-3
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def pred_post_err(fd, p, T=2000, reps=30, sense_every=1, seed0=0):
    a, ev = ar_coeffs_yw(p, fd)
    pe, po = [], []
    rng = np.random.default_rng(seed0)
    for _ in range(reps):
        h = jakes_series(T, fd, 64, rng)
        h = h / np.sqrt(np.mean(np.abs(h) ** 2))
        noise = np.sqrt(SIGMA_E2 / 2) * (rng.standard_normal(T) + 1j * rng.standard_normal(T))
        obs = h + noise
        sensed = (np.arange(T) % sense_every == 0)
        pred, post = ar_kalman_track(obs, sensed, a, ev, SIGMA_E2, beta=1.0)
        warm = T // 4
        pe.append(np.mean(np.abs(pred[warm:] - h[warm:]) ** 2))
        po.append(np.mean(np.abs(post[warm:] - h[warm:]) ** 2))
    return float(np.mean(pe)), float(np.mean(po))


def main():
    print(f"TM-1a: per-port AR(p) Kalman, noisy pilots sigma_e2={SIGMA_E2}, sense every slot\n")
    print("PRED error (predict-then-precode aging error), fraction of signal power:")
    print(f"{'f_D T_s':>8} | " + " | ".join(f"p={p}" for p in (1, 2, 4, 8)) + " | AR(1) floor 1-rho^2")
    print("-" * 62)
    res = {}
    for fd in (0.05, 0.10, 0.15, 0.20):
        row = []
        for p in (1, 2, 4, 8):
            pe, po = pred_post_err(fd, p)
            res[(fd, p)] = (pe, po); row.append(pe)
        from temporal import jakes_autocorr
        floor = 1 - jakes_autocorr(1, fd) ** 2
        print(f"{fd:>8} | " + " | ".join(f"{e:.3f}" for e in row) + f" |  {floor:.3f}")

    print("\nPOST error (observe-then-precode, ~sigma_e2) at fd=0.10:")
    for p in (1, 2, 4, 8):
        print(f"   p={p}: {res[(0.10, p)][1]:.4f}")

    pe1 = res[(0.10, 1)][0]; pe4 = res[(0.10, 4)][0]
    print(f"\nAt fd=0.10: PRED error AR(1) {pe1:.3f} -> AR(4) {pe4:.3f}  ({pe1/pe4:.1f}x reduction)")
    check("AR(4) PRED error << AR(1) at fd=0.10 (>=3x)", pe1 / pe4 >= 3.0)
    pe1f = res[(0.20, 1)][0]; pe4f = res[(0.20, 4)][0]
    check("AR(4) helps at fast Doppler too (fd=0.20, >=2x)", pe1f / pe4f >= 2.0,
          f"AR(1) {pe1f:.3f} -> AR(4) {pe4f:.3f}")

    print("\n" + "=" * 44)
    print(f"TM-1a: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    print("=" * 44)
    print("PRED error is the aging error capping predict/partial; AR(p) shrinks it toward sigma_e2.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
