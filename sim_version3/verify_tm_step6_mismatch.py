r"""
TM-6 -- MODEL-MISMATCH TEST for the parametric (Jakes) Doppler fit. The deciding experiment.

TM-4/TM-5 concluded a LEARNER CANNOT AFFORD AR(4): estimating 4 free autocorrelation values
r(1..4) and feeding them to an ill-conditioned Yule-Walker solve forces order ~2, capping recovery.

A parametric fit removes that cap. The physics says those 4 numbers are not free -- they are all
determined by ONE parameter via r(tau) = J0(2 pi f_D T_s tau). Fitting 1 parameter to 4 noisy lags
is far better conditioned, and in a first check it recovered the ORACLE AR(4) prediction error at
every noise level (0.00016 vs the nonparametric 0.018-0.185), with f_D pinned to +-0.0045 even at
se = 0.08.

THAT CHECK WAS SELF-FULFILLING: our channel generator IS Jakes, so it fitted a Jakes model to a
Jakes channel. This script measures what survives when the channel is NOT Jakes.

Four Doppler spectra, each with the SAME lag-1 correlation (so the comparison is not confounded by
one channel simply decorrelating faster):
    jakes      r(tau) = J0(2 pi f tau)                      <- MATCHED (control)
    flat       r(tau) = sinc(2 f tau)                       <- uniform Doppler spectrum
    gaussian   r(tau) = exp(-2 (pi s tau)^2)                <- Gaussian spectrum (aeronautical)
    rician     r(tau) = [K cos(2 pi f cos(th0) tau) + J0(2 pi f tau)] / (1+K)   <- specular + diffuse

Methods compared, both fed the SAME noisy ACF estimate:
    NONPARAM   ar_from_acf_robust  -- bootstrap order selection, assumes nothing  (current)
    PARAM      fit f_D by least squares to J0, then exact Yule-Walker at full order p
Scored by the ACTUAL 1-step prediction error variance each incurs under the spectrum's TRUE r,
against that spectrum's own ORACLE (Yule-Walker on the exact r).

Run:  python verify_tm_step6_mismatch.py [DRAWS]
"""

from __future__ import annotations

import sys
import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize_scalar, brentq

sys.stdout.reconfigure(encoding="utf-8")

from temporal import jakes_autocorr, ar_from_acf, ar_from_acf_robust

P = 4
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SE_LIST = [0.005, 0.01, 0.02, 0.04]
R1_TARGET = float(jakes_autocorr(1, 0.10))          # 0.9037 -- match every spectrum to this
K_RICE, TH0 = 5.0, 0.6
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


# --------------------------------------------------------------------- true ACFs
def acf_jakes(tau, f):
    return jakes_autocorr(tau, f)


def acf_flat(tau, f):
    x = 2.0 * f * np.asarray(tau, float)
    return np.sinc(x)                                # np.sinc(x) = sin(pi x)/(pi x)


def acf_gauss(tau, s):
    return np.exp(-2.0 * (np.pi * s * np.asarray(tau, float)) ** 2)


def acf_rice(tau, f):
    tau = np.asarray(tau, float)
    return (K_RICE * np.cos(2 * np.pi * f * np.cos(TH0) * tau) + jakes_autocorr(tau, f)) / (1 + K_RICE)


def calibrate(fn, lo, hi):
    """Solve for the spectrum parameter giving r(1) = R1_TARGET."""
    return brentq(lambda p: float(fn(1, p)) - R1_TARGET, lo, hi)


def main():
    print(f"TM-6 model mismatch -- parametric (Jakes) fit vs nonparametric, p={P}, "
          f"{DRAWS} draws/cell")
    print(f"all spectra calibrated to r(1) = {R1_TARGET:.4f}\n")

    specs = {
        "jakes (MATCHED)": (acf_jakes, calibrate(acf_jakes, 0.001, 0.35)),
        "flat": (acf_flat, calibrate(acf_flat, 0.001, 0.44)),
        "gaussian": (acf_gauss, calibrate(acf_gauss, 0.001, 0.35)),
        "rician K=5": (acf_rice, calibrate(acf_rice, 0.001, 0.35)),
    }
    taus = np.arange(P + 1)
    rng = np.random.default_rng(0)

    print(f"{'spectrum':>16} | " + " ".join(f"r({t})" for t in range(1, P + 1)))
    print("-" * 52)
    truths = {}
    for name, (fn, par) in specs.items():
        r = np.asarray(fn(taus, par), float); r[0] = 1.0
        truths[name] = r
        print(f"{name:>16} | " + " ".join(f"{v:+.3f}" for v in r[1:]))

    def actual(a, r_true):
        """Actual 1-step error variance of zero-padded coefficients a under r_true."""
        q = len(a)
        return float(r_true[0] - 2.0 * a @ r_true[1:q + 1] + a @ toeplitz(r_true[:q]) @ a)

    def fit_fd(rh):
        f = lambda fd: float(np.sum((rh[1:] - jakes_autocorr(np.arange(1, P + 1), fd)) ** 2))
        return minimize_scalar(f, bounds=(0.005, 0.45), method="bounded").x

    results = {}
    for name, r_true in truths.items():
        orc = actual(ar_from_acf(r_true)[0], r_true)          # this spectrum's own oracle
        print(f"\n{name}   (oracle AR({P}) error = {orc:.5f})")
        print(f"    {'se':>6} | {'NONPARAM':>10} | {'PARAM(Jakes)':>13} | {'winner':>10}")
        print("    " + "-" * 49)
        for se_lv in SE_LIST:
            se = np.r_[0.0, np.full(P, se_lv)]
            npe, pae = [], []
            for _ in range(DRAWS):
                rh = r_true.copy()
                rh[1:] = np.clip(r_true[1:] + rng.normal(0, se_lv, P), -0.999, 0.999)
                a_np, _ = ar_from_acf_robust(rh, se, n_draws=24)
                npe.append(actual(a_np, r_true))
                a_pa, _ = ar_from_acf(jakes_autocorr(taus, fit_fd(rh)))
                pae.append(actual(a_pa, r_true))
            mn, mp = float(np.median(npe)), float(np.median(pae))
            win = "PARAM" if mp < mn else "nonparam"
            print(f"    {se_lv:6.3f} | {mn:10.5f} | {mp:13.5f} | {win:>10}")
            results[(name, se_lv)] = (mn, mp, orc)

    # ------------------------------------------------------------------ gates
    print("\nGates:")
    m = "jakes (MATCHED)"
    mn, mp, orc = results[(m, 0.02)]
    check("MATCHED: parametric ~ reaches the oracle (the self-fulfilling case)",
          mp < 2 * orc + 1e-4, f"param {mp:.5f} vs oracle {orc:.5f} (nonparam {mn:.5f})")

    mism = [k for k in truths if k != m]
    wins = sum(results[(k, se)][1] < results[(k, se)][0] for k in mism for se in SE_LIST)
    tot = len(mism) * len(SE_LIST)
    check("MISMATCHED: parametric still beats nonparametric in most cells", wins > tot / 2,
          f"{wins}/{tot} cells")

    # NB: a RATIO to the oracle is meaningless when the oracle is ~1e-5 (rician). What matters is
    # the ABSOLUTE error, and whether it beats the alternative at the sample sizes we actually have.
    real = [se for se in SE_LIST if se >= 0.02]          # se we actually see (TM-5: 0.022-0.063)
    bad = [(k, se) for k in mism for se in real if results[(k, se)][1] >= results[(k, se)][0]]
    check("MISMATCHED: parametric beats nonparametric at REALISTIC sample sizes (se>=0.02)",
          not bad, "no losing cells" if not bad else f"loses at {bad}")

    floor = {k: float(np.median([results[(k, se)][1] for se in SE_LIST])) for k in mism}
    spread = {k: float(np.std([results[(k, se)][1] for se in SE_LIST]) / max(floor[k], 1e-12))
              for k in mism}
    check("parametric error is BIAS-dominated (flat in se) -> it has a mismatch FLOOR",
          all(v < 0.05 for v in spread.values()),
          "; ".join(f"{k}: {floor[k]:.4f} (spread {spread[k]*100:.1f}%)" for k in mism))

    check("that floor still sits below the nonparametric error at realistic se",
          all(floor[k] < results[(k, 0.02)][0] for k in mism),
          "; ".join(f"{k}: floor {floor[k]:.4f} vs nonparam {results[(k,0.02)][0]:.4f}"
                    for k in mism))

    # the honest limit: enough data and the assumption-free estimator wins
    cross = {k: results[(k, SE_LIST[0])][0] < floor[k] for k in mism}
    print(f"\n  Bias-variance crossover: at the SMALLEST se tested ({SE_LIST[0]}), nonparametric "
          f"already wins on {sum(cross.values())}/{len(mism)} mismatched spectra "
          f"({[k for k,v in cross.items() if v]}).\n  With enough data the assumption-free "
          f"estimator must win -- the parametric fit trades variance for MISMATCH BIAS, and only\n"
          f"  wins because at realistic horizons the variance dominates.")

    print("\n" + "=" * 62)
    print(f"TM-6: {'ALL PASS' if ok else 'SEE FAILURES ABOVE'}")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
