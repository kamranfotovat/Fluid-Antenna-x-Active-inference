r"""EPISTEMIC ABLATION DATA -- is the information-gain term load-bearing, or is
"active inference" a relabel of rate-greedy port selection?

This is the novelty test, and it is the one Zijun pushed on. Hold EVERYTHING fixed --
belief, AR(4) model, pilot rule, precoder, switching weight, channel trajectory -- and
vary ONLY beta_w, the weight on the epistemic term of the EFE:

    G(S,Q) = -[Prag(S) - eta_sw |S xor S_prev|] - beta_w Epis(Q)

    beta_w = 0   ->  pure rate-greedy selection. No exploration, no active inference.
    beta_w > 0   ->  the agent also values what a port would TELL it.

If the objective is flat in beta_w, the term is dressing. If it peaks at some interior
beta_w, then there is a genuine exploration sweet spot that neither a rate-only selector
(beta_w = 0) nor an over-eager explorer (large beta_w) finds -- which is the paper's
actual claim.

WHY THIS SUPERSEDES ablation_epistemic.py. That script answered the same question at the
OLD operating point: fully digital transmit, observe-then-precode, and pilots on ALL M
activated ports. The paper is now hybrid (n_rf = 6 = 2K) with partial sensing at m = 6,
so its numbers cannot be quoted in Section IV without contradicting Table I. This driver
goes through exactly the run_st path that produced Fig. 2 and Table I, so every number
here is directly comparable to them -- in particular the beta_w = 0.25 row must reproduce
the m = 6 row of Table I block (a), which is a free correctness check.

WHY THIS SWEEPS TWO SWITCHING PRICES. The smoke run exposed a confound that would
have made a beta_w-only figure misleading. At eta_sw = 1, beta_w = 0 scores a DECENT
objective -- not because rate-greedy is good (it strands the agent at ~57% of genie)
but because it never moves, so it pays no switching cost at all. It looks competitive
on the objective for the wrong reason. Comparing it against an exploring agent that
does pay to move is not an apples-to-apples test of the epistemic term.

So we sweep beta_w at eta_sw = 1 (the paper's operating point) AND at eta_sw = 4,
where Table I block (b) shows every arm is driven to zero switching. At eta_sw = 4 the
switching cost is equalised at ~0 by construction, so any remaining difference is
attributable to the epistemic term alone. That is the comparison the figure should
make, and it is the one a referee will ask for.

We also track Uglobal, the mean posterior variance over ALL N ports (not just the
activated ones). It is the mechanism made visible: a rate-greedy agent keeps re-selecting
from the corner of the grid it already knows and stays ignorant of the rest, so its
Uglobal stays high even while its instantaneous rate looks acceptable.

Run:  python make_ablation_data.py [MC] [T]
      python make_ablation_data.py --smoke     (1 seed, T=20, 3 beta values)
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
# 0 is the rate-greedy control; 0.25 is the paper's operating point and must
# reproduce Table I; the tail probes whether over-exploration is punished.
BETA_LIST = [0.0, 0.25, 1.0] if SMOKE else [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
# eta_sw = 1 is the paper's point; eta_sw = 4 equalises switching at ~0 across all
# arms (Table I block (b)), isolating the epistemic term from the movement budget.
ETA_LIST = [1.0] if SMOKE else [1.0, 4.0]
M_SENSE = 6
FD, P_AR = 0.10, 4
PILOT_RULE = "epistemic"
HALF = slice(T // 2, None)
OUT = Path(__file__).resolve().parent.parent / "results_tm" / "ablation_beta.json"


def run_one(bel, H, op, rng):
    """run_st, but also tracking global posterior variance.

    run_st does not report Uglobal, and wrapping it would mean re-running the loop.
    Monkey-patching the belief's update is uglier than it looks -- instead we let
    run_st do its work and sample the variance afterwards is NOT possible (the loop
    ends), so we wrap bel.update to record after every assimilation."""
    trace = []
    orig = bel.update

    def spy(S, y):
        orig(S, y)
        trace.append(float(bel.port_variances().mean()))

    bel.update = spy
    try:
        out = run_st(bel, H, op, rng, protocol="partial",
                     m_sense=M_SENSE, pilot_rule=PILOT_RULE)
    finally:
        bel.update = orig
    out["uglobal"] = np.asarray(trace)
    return out


def main() -> int:
    print(f"EPISTEMIC ABLATION  MC={MC}, T={T}, m={M_SENSE}, beta_w sweep {BETA_LIST}")
    print(f"  {OP_V3.label()}")
    print(f"  beta_w=0 is rate-greedy (no AIF); beta_w=0.25 is the paper's point\n",
          flush=True)
    R = spatial_correlation(OP_V2.positions())
    pos = OP_V3.positions()

    cells = [(b, e) for e in ETA_LIST for b in BETA_LIST]
    key = lambda b, e: f"{b:g}|{e:g}"
    rate = {key(b, e): [] for b, e in cells}
    swit = {key(b, e): [] for b, e in cells}
    ugl = {key(b, e): [] for b, e in cells}
    g_rate, g_swit = [], []
    t0 = time.perf_counter()

    for s in range(MC):
        H = generate_spacetime_jakes(R, OP_V3.beta, FD, T, OP_V3.K, seed=100 + s)
        gen = run_genie(H, OP_V3.M, sigma2=OP_V3.sigma2, P=OP_V3.P, positions=pos,
                        d_min=OP_V3.d_min, n_rf=OP_V3.n_rf)
        g_rate.append(float(gen["rate"][HALF].mean()))
        g_swit.append(float(gen["switch"][HALF].mean()))
        for bw, e in cells:
            op = dataclasses.replace(OP_V3, beta_w=bw, eta_sw=e)
            bel = STKalmanBeliefLR(R, op.beta, FD, P_AR, op.sigma_e2)
            out = run_one(bel, H, op, np.random.default_rng(200 + s))
            k = key(bw, e)
            rate[k].append(float(out["rate"][HALF].mean()))
            swit[k].append(float(out["switch"][HALF].mean()))
            ugl[k].append(float(out["uglobal"][HALF].mean()))
        print(f"  seed {s} ({time.perf_counter()-t0:.0f}s)  " + "  ".join(
            f"b={bw:g}/e={e:g}: {rate[key(bw,e)][-1]:.2f}r/{swit[key(bw,e)][-1]:.1f}s"
            for bw, e in cells), flush=True)

        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps({
            "meta": {"MC_done": s + 1, "MC_target": MC, "T": T, "m_sense": M_SENSE,
                     "p": P_AR, "fd": FD, "beta_list": BETA_LIST,
                     "eta_list": ETA_LIST, "op": OP_V3.label(),
                     "pilot_rule": PILOT_RULE, "smoke": SMOKE},
            "rate": rate, "switch": swit, "uglobal": ugl,
            "genie_rate": g_rate, "genie_switch": g_swit,
        }, indent=1), encoding="utf-8")

    gr = float(np.mean(g_rate))
    for e in ETA_LIST:
        tail = " (paper's point)" if e == 1.0 else " (switching equalised at ~0)"
        print(f"\n  eta_sw = {e:g}{tail}")
        print(f"{'beta_w':>7} | {'rate':>7} | {'sw/slot':>8} | {'objective':>10} | "
              f"{'% genie':>8} | {'Uglobal':>8}")
        print("  " + "-" * 62)
        for bw in BETA_LIST:
            k = key(bw, e)
            r, w, u = np.mean(rate[k]), np.mean(swit[k]), np.mean(ugl[k])
            tag = "   <- rate-greedy" if bw == 0 else ("   <- paper" if bw == 0.25 else "")
            print(f"{bw:7g} | {r:7.3f} | {w:8.2f} | {r - e*w:10.3f} | "
                  f"{100*r/gr:7.1f}% | {u:8.4f}{tag}")
    print(f"\n  genie: {gr:.3f} b/s/Hz, {np.mean(g_swit):.2f} sw/slot")

    # The claim the paper makes stands or falls on this line. Needs the
    # switching-equalised arm, which --smoke does not run.
    if 4.0 not in ETA_LIST:
        print("\n(skipping matched-switching test: eta_sw = 4 not in this sweep)")
        print(f"\nwrote {OUT}  ({time.perf_counter()-t0:.0f}s)")
        return 0
    r0 = np.mean(rate[key(0.0, 4.0)]); w0 = np.mean(swit[key(0.0, 4.0)])
    u0 = np.mean(ugl[key(0.0, 4.0)])
    bb = max(BETA_LIST, key=lambda x: np.mean(rate[key(x, 4.0)]) - 4.0 * np.mean(swit[key(x, 4.0)]))
    rb, wb = np.mean(rate[key(bb, 4.0)]), np.mean(swit[key(bb, 4.0)])
    ub = np.mean(ugl[key(bb, 4.0)])
    print(f"\nMATCHED-SWITCHING TEST (eta_sw = 4):")
    print(f"  rate-greedy  beta_w=0     : {r0:6.3f} b/s/Hz ({100*r0/gr:.1f}% genie), "
          f"{w0:.2f} sw/slot, Uglobal {u0:.4f}")
    print(f"  epistemic    beta_w={bb:g}  : {rb:6.3f} b/s/Hz ({100*rb/gr:.1f}% genie), "
          f"{wb:.2f} sw/slot, Uglobal {ub:.4f}")
    print(f"  => +{rb-r0:.3f} b/s/Hz ({100*(rb-r0)/gr:+.1f} pts of genie) at "
          f"{'equal' if abs(wb-w0) < 0.5 else 'UNEQUAL'} switching; "
          f"Uglobal {u0/max(ub,1e-9):.1f}x lower")
    if rb - r0 > 0.5 and abs(wb - w0) < 0.5:
        print("  => the epistemic term is LOAD-BEARING: it buys rate, not just movement.")
    else:
        print("  => WEAK or confounded -- do not claim the term is load-bearing.")

    print(f"\nwrote {OUT}  ({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
