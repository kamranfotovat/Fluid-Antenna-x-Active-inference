r"""Fig. 3 DATA -- the rate/switching frontier, i.e. the OBJECTIVE.

Every figure so far reports sum rate alone, but the objective (6) is rate NET of
reconfiguration cost, and the switching term sits inside the EFE. Nothing yet shows it
doing anything. A reviewer is entitled to ask why it is there.

This sweeps the switching weight eta_sw at the paper's operating point (m = 6) and
records, per arm, the mean sum rate AND the mean number of ports moved per slot. Plotting
one against the other traces the frontier the agent can operate on; the genie is a single
point on the same axes. The genie re-selects greedily from perfect CSI every slot with no
movement penalty, so it should buy its rate with heavy reconfiguration -- which is exactly
what the objective is meant to charge for.

Run:  python make_fig3_data.py [MC] [T]
      python make_fig3_data.py --smoke     (1 seed, T=20, 2 eta values)
"""

from __future__ import annotations

import dataclasses
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

SMOKE = "--smoke" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
MC = 1 if SMOKE else (int(args[0]) if args else 8)
T = 20 if SMOKE else (int(args[1]) if len(args) > 1 else 40)
ETA_LIST = [0.0, 2.0] if SMOKE else [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
M_SENSE = 6
FD, P_AR = 0.10, 4
PILOT_RULE = "epistemic"
HALF = slice(T // 2, None)
OUT = Path(__file__).resolve().parent.parent / "results_tm" / "fig3_frontier.json"


def main() -> int:
    print(f"FIG-3 DATA (objective frontier)  MC={MC}, T={T}, m={M_SENSE}, "
          f"eta sweep {ETA_LIST}")
    print(f"  {OP_V3.label()}\n", flush=True)
    R = spatial_correlation(OP_V2.positions())
    pos = OP_V3.positions()

    rate = {e: [] for e in ETA_LIST}
    swit = {e: [] for e in ETA_LIST}
    g_rate, g_swit = [], []
    t0 = time.perf_counter()

    for s in range(MC):
        H = generate_spacetime_jakes(R, OP_V3.beta, FD, T, OP_V3.K, seed=100 + s)
        gen = run_genie(H, OP_V3.M, sigma2=OP_V3.sigma2, P=OP_V3.P, positions=pos,
                        d_min=OP_V3.d_min, n_rf=OP_V3.n_rf)
        g_rate.append(float(gen["rate"][HALF].mean()))
        g_swit.append(float(gen["switch"][HALF].mean()))
        for e in ETA_LIST:
            op = dataclasses.replace(OP_V3, eta_sw=e)
            bel = STKalmanBeliefLR(R, op.beta, FD, P_AR, op.sigma_e2)
            out = run_st(bel, H, op, np.random.default_rng(200 + s),
                         protocol="partial", m_sense=M_SENSE, pilot_rule=PILOT_RULE)
            rate[e].append(float(out["rate"][HALF].mean()))
            swit[e].append(float(out["switch"][HALF].mean()))
        print(f"  seed {s} ({time.perf_counter()-t0:.0f}s)  " + "  ".join(
            f"eta={e:g}: {rate[e][-1]:.2f}r/{swit[e][-1]:.2f}s" for e in ETA_LIST),
            flush=True)

        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps({
            "meta": {"MC_done": s + 1, "MC_target": MC, "T": T, "m_sense": M_SENSE,
                     "p": P_AR, "fd": FD, "eta_list": ETA_LIST, "op": OP_V3.label(),
                     "pilot_rule": PILOT_RULE, "smoke": SMOKE},
            "rate": {str(e): v for e, v in rate.items()},
            "switch": {str(e): v for e, v in swit.items()},
            "genie_rate": g_rate, "genie_switch": g_swit,
        }, indent=1), encoding="utf-8")

    gr, gs = float(np.mean(g_rate)), float(np.mean(g_swit))
    print(f"\n{'eta_sw':>7} | {'rate':>7} | {'switches/slot':>14} | "
          f"{'objective (eta=1)':>18}")
    print("-" * 56)
    for e in ETA_LIST:
        r, w = float(np.mean(rate[e])), float(np.mean(swit[e]))
        print(f"{e:7g} | {r:7.3f} | {w:14.2f} | {r - 1.0*w:18.3f}")
    print(f"{'genie':>7} | {gr:7.3f} | {gs:14.2f} | {gr - 1.0*gs:18.3f}")

    spread = max(np.mean(swit[e]) for e in ETA_LIST) - min(np.mean(swit[e]) for e in ETA_LIST)
    print(f"\nswitching spread across the eta sweep: {spread:.2f} ports/slot")
    if spread < 0.5:
        print("  WARNING: eta_sw barely moves the operating point -- a frontier plot "
              "would be a flat blob and this figure is not worth making.")
    print(f"\nwrote {OUT}  ({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
