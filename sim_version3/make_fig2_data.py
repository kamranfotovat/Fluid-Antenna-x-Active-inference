r"""Fig. 2 DATA -- sum rate vs pilot budget m, at the paper's operating point.

This is the paper's headline figure: x = number of piloted ports m, y = sum rate.
Everything is AR(4) with hybrid transmit (OP_V3, n_rf = 6 = 2K) -- the configuration
the paper actually claims. There is deliberately NO AR(1) arm: comparing our own
model against a weaker version of our own model is self-benchmarking, and Kian cut
it. The comparison that remains is against the full-CSI genie and against the
m = M end of our own budget curve, which is a statement about the PILOT BUDGET
rather than about the model.

Unlike verify_paper1_config.py this dumps PER-SEED rates to JSON so the figure can
carry error bars. Mean alone at MC=6 was never going to be publishable.

Run:  python make_fig2_data.py [MC] [T]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V2, OP_V3
from channel import spatial_correlation
from temporal import generate_spacetime_jakes
from st_belief_lr import STKalmanBeliefLR
from st_belief import run_st
from agent import run_genie

MC = int(sys.argv[1]) if len(sys.argv) > 1 else 20
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FD = 0.10
P_AR = 4
M_LIST = [2, 4, 6, 8, 10]
HALF = slice(T // 2, None)
OUT = Path(__file__).resolve().parent.parent / "results_tm" / "fig2_pilot_sweep.json"


def main() -> int:
    OP = OP_V3
    print(f"FIG-2 DATA  MC={MC}, T={T}, fd={FD}, AR({P_AR}), m sweep {M_LIST}")
    print(f"  {OP.label()}")
    print(f"  n_rf = {OP.n_rf} = 2K -> digital-exact threshold\n", flush=True)

    R = spatial_correlation(OP_V2.positions())        # same geometry as the digital point
    pos = OP.positions()
    rates = {m: [] for m in M_LIST}
    genie = []
    t0 = time.perf_counter()

    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD, T, OP.K, seed=100 + s)
        genie.append(float(run_genie(H, OP.M, sigma2=OP.sigma2, P=OP.P, positions=pos,
                                     d_min=OP.d_min, n_rf=OP.n_rf)["rate"][HALF].mean()))
        for m in M_LIST:
            bel = STKalmanBeliefLR(R, OP.beta, FD, P_AR, OP.sigma_e2)
            proto = "observe" if m >= OP.M else "partial"
            out = run_st(bel, H, OP, np.random.default_rng(200 + s),
                         protocol=proto, m_sense=m)
            rates[m].append(float(out["rate"][HALF].mean()))
        print(f"  seed {s} done ({time.perf_counter()-t0:.0f}s)  "
              + "  ".join(f"m={m}:{rates[m][-1]:.2f}" for m in M_LIST), flush=True)

        # checkpoint every seed -- a long run must survive an interrupt
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps({
            "meta": {"MC_done": s + 1, "MC_target": MC, "T": T, "fd": FD, "p": P_AR,
                     "m_list": M_LIST, "op": OP.label(), "n_rf": OP.n_rf,
                     "half": f"slots {T//2}..{T-1}"},
            "genie": genie,
            "rates": {str(m): v for m, v in rates.items()},
        }, indent=1), encoding="utf-8")

    g = float(np.mean(genie))
    print(f"\n{'m':>4} | {'pilots':>7} | {'mean':>7} | {'std':>6} | {'% genie':>8} | {'% of m=M':>9}")
    print("-" * 58)
    full = float(np.mean(rates[OP.M]))
    for m in M_LIST:
        v = np.array(rates[m])
        print(f"{m:4d} | {m/OP.M*100:6.0f}% | {v.mean():7.3f} | {v.std():6.3f} | "
              f"{v.mean()/g*100:7.1f}% | {v.mean()/full*100:8.1f}%")
    print(f"{'genie':>4} | {'':>7} | {g:7.3f} | {np.std(genie):6.3f}")
    print(f"\nwrote {OUT}  ({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
