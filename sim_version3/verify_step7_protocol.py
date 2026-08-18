"""
Step 7a -- intra-slot protocol ablation + Doppler robustness (the headline lever).

Two ways to run the SAME agent within a slot:
  predict-then-act    : aim the beam from the PREDICTED (aged) belief, then observe.
  observe-then-precode : send pilots on the activated ports, Kalman-update, THEN aim.

This is the biggest single lever on realized rate. observe-then-precode makes the served-port
CSI fresh (error ~ sigma_e^2 instead of the aging error (1-rho^2)beta), so the rate jumps
toward the genie and -- crucially -- stops depending on the Doppler rho.

Checks (operating point: 15 dB, beta_w=0.5, eta_sw=1.0, M/N = 5/25 = 20% observation budget):
  P1  Fresh vs aged CSI.  observe-then-precode served-port error ~= sigma_e^2; predict-then-act
      error ~= the aging floor. Both remain CALIBRATED (belief var ~= realized error).
  P2  Rate.  observe-then-precode >> predict-then-act, and reaches a high fraction of the genie.
  P3  Doppler robustness.  observe-then-precode's fraction-of-genie stays high across rho, while
      predict-then-act degrades as the channel speeds up (rho down).

Run:  python sim/verify_step7_protocol.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402
from agent import AIFAgent, run_aif, run_genie  # noqa: E402

SIGMA_E2, SIGMA2 = 1e-2, 0.03                  # 15 dB
BETA = np.array([1.0, 0.7, 1.3])
K, M = 3, 5
ETA_SW, BETA_W = 1.0, 0.5
RHOS = [0.95, 0.9, 0.8, 0.7, 0.6]
T, MC = 70, 10


def _agent(R, rho):
    return AIFAgent(R, BETA, rho, SIGMA_E2, M, alpha=1.0, beta_w=BETA_W,
                    eta_sw=ETA_SW, e_sw=1.0, sigma2=SIGMA2)


def sweep():
    """Return per-rho means: genie/predict/observe rate, their served-CSI err & belief var."""
    res = {k: [] for k in ["genie", "pa", "sf", "err_pa", "var_pa", "err_sf", "var_sf"]}
    for rho in RHOS:
        acc = {k: 0.0 for k in res}
        for m in range(MC):
            sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=rho, beta=BETA, seed=m)
            H = sim.generate(T)
            G = run_genie(H, M, sigma2=SIGMA2)
            PA = run_aif(_agent(sim.R, rho), H, SIGMA_E2, np.random.default_rng(20000 + m),
                         track_belief=True, sense_first=False)
            SF = run_aif(_agent(sim.R, rho), H, SIGMA_E2, np.random.default_rng(20000 + m),
                         track_belief=True, sense_first=True)
            h = slice(T // 2, None)
            acc["genie"] += G["rate"][h].mean()
            acc["pa"] += PA["rate"][h].mean(); acc["sf"] += SF["rate"][h].mean()
            acc["err_pa"] += PA["real_err"][h].mean(); acc["var_pa"] += PA["post_var"][h].mean()
            acc["err_sf"] += SF["real_err"][h].mean(); acc["var_sf"] += SF["post_var"][h].mean()
        for k in res:
            res[k].append(acc[k] / MC)
    return {k: np.array(v) for k, v in res.items()}


def main():
    print(f"N=25, K={K}, M={M} (obs budget {M}/{25}={100*M/25:.0f}%), SNR={-10*np.log10(SIGMA2):.0f}dB, "
          f"beta_w={BETA_W}, eta_sw={ETA_SW}, T={T}, MC={MC}")
    print(f"sweeping rho over {RHOS} ...")
    r = sweep()
    frac_pa = r["pa"] / r["genie"]
    frac_sf = r["sf"] / r["genie"]
    all_pass = True

    print("\n rho  | genie | predict-then-act (%) | observe-then-precode (%)")
    for i, rho in enumerate(RHOS):
        print(f"  {rho:.2f} | {r['genie'][i]:5.2f} | {r['pa'][i]:5.2f} ({100*frac_pa[i]:3.0f}%)"
              f"            | {r['sf'][i]:5.2f} ({100*frac_sf[i]:3.0f}%)")

    # ---- P1: fresh vs aged CSI, both calibrated ---------------------------
    print("\n[P1] observe-then-precode CSI is fresh (~sigma_e^2); both protocols calibrated")
    fresh_ok = np.all(r["err_sf"] < 3 * SIGMA_E2)
    cal_ok = (np.max(np.abs(r["err_sf"] - r["var_sf"]) / r["var_sf"]) < 0.15 and
              np.max(np.abs(r["err_pa"] - r["var_pa"]) / r["var_pa"]) < 0.15)
    ok = fresh_ok and cal_ok
    all_pass &= ok
    print(f"   observe-then-precode served err range = [{r['err_sf'].min():.3f}, {r['err_sf'].max():.3f}] "
          f"(sigma_e^2={SIGMA_E2}) | predict-then-act err range = [{r['err_pa'].min():.3f}, {r['err_pa'].max():.3f}]")
    print(f"   fresh(<3 sigma_e^2)={fresh_ok}  both calibrated(<15%)={cal_ok}  -> {'PASS' if ok else 'FAIL'}")

    # ---- P2: rate improvement + high fraction of genie --------------------
    print("\n[P2] observe-then-precode >> predict-then-act, and near the genie")
    ratio = r["sf"] / r["pa"]
    ok = np.all(ratio > 1.3) and np.all(frac_sf > 0.75)
    all_pass &= ok
    print(f"   observe/predict rate ratio range = [{ratio.min():.2f}, {ratio.max():.2f}] (>1.3)")
    print(f"   observe-then-precode fraction of genie range = [{100*frac_sf.min():.0f}%, {100*frac_sf.max():.0f}%] (>75%)"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- P3: Doppler robustness -------------------------------------------
    print("\n[P3] Doppler robustness: observe-then-precode holds; predict-then-act degrades as rho falls")
    # variation of fraction across rho: observe-then-precode should be much flatter
    spread_sf = frac_sf.max() - frac_sf.min()
    spread_pa = frac_pa.max() - frac_pa.min()
    # predict-then-act should clearly worsen from high rho to low rho
    degrades = frac_pa[0] - frac_pa[-1] > 0.10
    ok = (spread_sf < spread_pa) and degrades
    all_pass &= ok
    print(f"   fraction spread across rho: observe-then-precode={spread_sf:.2f} < predict-then-act={spread_pa:.2f}")
    print(f"   predict-then-act drops {100*(frac_pa[0]-frac_pa[-1]):.0f} pts from rho={RHOS[0]} to {RHOS[-1]}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + ("=" * 46))
    print(f"STEP 7a OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)

    _try_plot(r, frac_pa, frac_sf)
    return 0 if all_pass else 1


def _try_plot(r, frac_pa, frac_sf):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    rhos = np.array(RHOS)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(rhos, r["genie"], "k--o", label="genie (full CSI, ceiling)")
    ax[0].plot(rhos, r["sf"], "o-", color="tab:green", label="observe-then-precode (ours)")
    ax[0].plot(rhos, r["pa"], "s-", color="tab:red", label="predict-then-act")
    ax[0].set_title("Realized rate vs Doppler correlation")
    ax[0].set_xlabel("temporal correlation rho (higher = slower channel)")
    ax[0].set_ylabel("sum-rate (bits/s/Hz)")
    ax[0].invert_xaxis(); ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

    ax[1].plot(rhos, 100 * frac_sf, "o-", color="tab:green", label="observe-then-precode")
    ax[1].plot(rhos, 100 * frac_pa, "s-", color="tab:red", label="predict-then-act")
    ax[1].axhline(100, color="k", ls="--", alpha=0.5, label="genie")
    ax[1].set_title("Fraction of genie captured (robustness)")
    ax[1].set_xlabel("temporal correlation rho"); ax[1].set_ylabel("% of genie rate")
    ax[1].invert_xaxis(); ax[1].set_ylim(0, 105); ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = "step7_protocol_doppler.png"
    fig.savefig(out, dpi=120)
    print(f"(saved plot -> sim/{out})")


if __name__ == "__main__":
    raise SystemExit(main())
