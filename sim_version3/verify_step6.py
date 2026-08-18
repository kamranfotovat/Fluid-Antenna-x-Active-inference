"""
Step 6 verification -- the closed-loop agent works, is calibrated, and the epistemic term earns its place.

Gate before Step 7 (sweeps/figures). Operating point follows Step 0-2 findings: MODERATE SNR
(15 dB; at 30 dB the system is interference-limited and predicted-CSI precoding collapses) and a
balanced epistemic weight (beta_w = 0.5; beta_w = 1 over-explores, beta_w = 0 never explores).

Loop per slot (RESEARCH_PLAN Sec. 5): predict -> select -> precode from belief -> transmit ->
observe activated ports -> Kalman update. The precoder uses the PREDICTED belief (we act before
we see this slot's feedback), so temporal aging directly sets the achievable rate.

Checks
------
  C1  Numerical stability.  Over T slots: finite rates, belief Sigma stays PSD, no divergence.
  C2  Closed-loop calibration (THE gate).  Served-port posterior variance ~= realized squared
      error -- the belief stays honest despite the selection<->observation coupling.
  C3  Learning curve trends up.  Rate rises from the cold start (mu=0 -> rate 0) to a plateau.
  C4  Epistemic earns its place.  AIF(beta=0.5) steady-state RATE beats AIF(beta=0) by a clear
      margin, and its switching-aware OBJECTIVE is no worse.
  C5  Trends toward genie.  genie >= AIF on rate; AIF captures a meaningful fraction of genie,
      and closes most of the gap on the switching-aware objective (AIF respects switching cost).

Run:  python sim/verify_step6.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from channel import ChannelSimulator          # noqa: E402
from agent import AIFAgent, run_aif, run_genie, objective  # noqa: E402

# operating point
RHO, SIGMA_E2, SIGMA2 = 0.9, 1e-2, 0.03       # 15 dB
BETA = np.array([1.0, 0.7, 1.3])
K, M = 3, 5
ETA_SW, E_SW = 1.0, 1.0
ALPHA, BETA_W = 1.0, 0.5
T, MC = 100, 14


def _make_agent(R, beta_w):
    return AIFAgent(R, BETA, RHO, SIGMA_E2, M, alpha=ALPHA, beta_w=beta_w,
                    eta_sw=ETA_SW, e_sw=E_SW, sigma2=SIGMA2)


def main():
    print(f"N={5*5}, K={K}, M={M}, rho={RHO}, SNR={-10*np.log10(SIGMA2):.0f}dB, "
          f"sigma_e^2={SIGMA_E2}, beta_w={BETA_W}, eta_sw={ETA_SW}, T={T}, MC={MC}")
    all_pass = True

    curve = np.zeros(T)
    pv_acc = re_acc = 0.0
    min_eig = np.inf
    obj = {"genie": 0.0, "aif": 0.0, "aif0": 0.0}
    rate2 = {"genie": 0.0, "aif": 0.0, "aif0": 0.0}
    sw = {"genie": 0.0, "aif": 0.0, "aif0": 0.0}
    finite_ok = True

    for m in range(MC):
        sim = ChannelSimulator(Nx=5, Ny=5, Wx=1.0, Wy=1.0, K=K, rho=RHO, beta=BETA, seed=m)
        H = sim.generate(T)
        G = run_genie(H, M, sigma2=SIGMA2)
        agent = _make_agent(sim.R, BETA_W)
        A = run_aif(agent, H, SIGMA_E2, np.random.default_rng(20000 + m), track_belief=True)
        A0 = run_aif(_make_agent(sim.R, 0.0), H, SIGMA_E2, np.random.default_rng(20000 + m))

        # C1: finiteness + PSD of the final belief
        for res in (G, A, A0):
            finite_ok &= np.all(np.isfinite(res["rate"]))
        for k in range(K):
            min_eig = min(min_eig, np.linalg.eigvalsh(agent.bel.Sigma[k]).min())

        curve += A["rate"]
        pv_acc += A["post_var"][T // 2:].mean()
        re_acc += A["real_err"][T // 2:].mean()
        for nm, res in [("genie", G), ("aif", A), ("aif0", A0)]:
            obj[nm] += objective(res, ETA_SW, E_SW)
            rate2[nm] += res["rate"][T // 2:].mean()
            sw[nm] += res["switch"].mean()

    curve /= MC
    for d in (obj, rate2, sw):
        for nm in d:
            d[nm] /= MC
    pv, re = pv_acc / MC, re_acc / MC

    # ---- C1 -------------------------------------------------------------
    print("\n[C1] Numerical stability over the closed loop")
    ok = finite_ok and (min_eig > -1e-8)
    all_pass &= ok
    print(f"   all rates finite = {finite_ok} | min belief eig = {min_eig:.2e} (>=0)  -> {'PASS' if ok else 'FAIL'}")

    # ---- C2 -------------------------------------------------------------
    print("\n[C2] Closed-loop calibration: served-port post-var ~= realized error   [THE gate]")
    rel = abs(pv - re) / re
    ok = rel < 0.10
    all_pass &= ok
    print(f"   posterior var = {pv:.4f} | realized |h-mu|^2 = {re:.4f} | rel-err = {rel:.3f} (<0.10)"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- C3 -------------------------------------------------------------
    print("\n[C3] Learning curve trends up from the cold start")
    cold = curve[:2].mean()
    steady = curve[T // 2:].mean()
    ok = curve[0] < 1e-9 and steady > cold and steady > 2.0 * cold
    all_pass &= ok
    print(f"   rate[0]={curve[0]:.2f} (cold) | first-2 avg={cold:.2f} | steady(2nd half)={steady:.2f}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- C4 -------------------------------------------------------------
    print("\n[C4] Epistemic earns its place: AIF(beta=0.5) vs AIF(beta=0)")
    rate_margin = (rate2["aif"] - rate2["aif0"]) / rate2["aif0"]
    ok = rate_margin > 0.10 and obj["aif"] >= obj["aif0"] - 1e-6
    all_pass &= ok
    print(f"   steady RATE: aif(.5)={rate2['aif']:.2f} vs aif(0)={rate2['aif0']:.2f}  (+{rate_margin*100:.1f}%)")
    print(f"   OBJECTIVE:   aif(.5)={obj['aif']:.2f} vs aif(0)={obj['aif0']:.2f}"
          f"  -> {'PASS' if ok else 'FAIL'}")

    # ---- C5 -------------------------------------------------------------
    print("\n[C5] Trends toward genie (rate ceiling) and closes the gap on the switching-aware objective")
    rate_frac = rate2["aif"] / rate2["genie"]
    obj_frac = obj["aif"] / obj["genie"]
    ok = (rate2["genie"] >= rate2["aif"]) and (rate_frac > 0.40) and (obj_frac > rate_frac)
    all_pass &= ok
    print(f"   RATE: genie={rate2['genie']:.2f} aif={rate2['aif']:.2f} ({rate_frac*100:.0f}% of genie)"
          f" | switch/slot genie={sw['genie']:.1f} aif={sw['aif']:.1f}")
    print(f"   OBJECTIVE: genie={obj['genie']:.2f} aif={obj['aif']:.2f} ({obj_frac*100:.0f}% of genie)"
          f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + ("=" * 46))
    print(f"STEP 6 OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 46)

    _try_plot(curve, rate2, obj, sw)
    return 0 if all_pass else 1


def _try_plot(curve, rate2, obj, sw):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(len(curve)), curve, lw=1.5, label="AIF (beta=0.5)")
    ax[0].axhline(rate2["genie"], color="k", ls="--", alpha=0.7, label="genie (full CSI)")
    ax[0].axhline(rate2["aif0"], color="tab:red", ls=":", alpha=0.7, label="AIF (beta=0, no explore)")
    ax[0].set_title("Closed-loop learning curve (rate)")
    ax[0].set_xlabel("slot t"); ax[0].set_ylabel("realized sum-rate (bits/s/Hz)")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

    labels = ["genie\n(full CSI)", "AIF\n(beta=0.5)", "AIF\n(beta=0)"]
    x = np.arange(3)
    w = 0.38
    rates = [rate2["genie"], rate2["aif"], rate2["aif0"]]
    objs = [obj["genie"], obj["aif"], obj["aif0"]]
    ax[1].bar(x - w / 2, rates, w, label="rate", color="tab:blue", alpha=0.85)
    ax[1].bar(x + w / 2, objs, w, label="objective (rate - switch)", color="tab:orange", alpha=0.85)
    for i in range(3):
        ax[1].text(i - w / 2, rates[i] + 0.1, f"{sw[list(sw)[i]]:.1f} sw", ha="center", fontsize=7, color="gray")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_title("Rate vs switching-aware objective")
    ax[1].set_ylabel("bits/s/Hz"); ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = "step6_closed_loop_check.png"
    fig.savefig(out, dpi=120)
    print(f"(saved plot -> sim/{out})")


if __name__ == "__main__":
    raise SystemExit(main())
