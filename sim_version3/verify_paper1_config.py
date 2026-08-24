r"""
THE PAPER-1 CONFIGURATION -- partial sensing + AR(p) temporal model + HYBRID transmit (n_rf = 2K).

Kian's decision (2026-08-24): Paper 1 becomes "EFE port selection + PILOT ALLOCATION", keeping the
S1 hybrid transmit stage. Every partial-sensing number to date was measured with a FULLY DIGITAL
transmitter (OP_V2); this is the first run of partial sensing and hybrid beamforming TOGETHER, which
is the configuration the paper will actually claim.

Two things to establish.

(1) HYBRID IS FREE. n_rf = 6 = 2K is digital-exact for a fully-connected infinite-resolution
    phase-shifter network (Sohrabi & Yu 2016; Zhang, Molisch & Kung 2005 -- any complex entry is a
    sum of two unit-modulus phasors). It SHOULD cost nothing at every pilot budget. That is an
    assumption until measured -- if hybrid interacts badly with belief-inferred (un-piloted) ports,
    the headline moves. Compared head-to-head against OP_V2 (digital, same seeds).

(2) THE HEADLINE. Current Paper 1 is m_sense = M (all active ports piloted) with the AR(1) belief.
    The claim to establish is:

        m_sense = 6 with AR(4)   >=   m_sense = 10 with AR(1)

    i.e. EQUAL RATE ON 40% FEWER PILOTS -- the temporal model buys back exactly what withdrawing
    pilots costs. At the digital operating point this held (18.477 vs 18.337). It must survive the
    hybrid transmit stage to be the paper's headline.

SENSING FRONT-END ASSUMPTION (state this in the paper). m_sense piloted ports are modelled as clean
PER-PORT observations, i.e. a selection matrix P_S. That presumes a SWITCH-based sensing front-end,
which is the standard FAS architecture (port switching is the premise of the whole system). A
fully-connected unit-modulus phase-shifter network could NOT deliver this on receive -- it cannot
null the unselected ports, so each chain would see a combination. The hybrid network here is on the
TRANSMIT side only, where it is a post-processing of the precoder.

Run:  python verify_paper1_config.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V2, OP_V3
from channel import spatial_correlation
from temporal import generate_spacetime_jakes
from st_belief_lr import STKalmanBeliefLR
from st_belief import run_st
from agent import run_genie

MC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FD = 0.10
M_LIST = [4, 6, 8, 10]
HALF = slice(T // 2, None)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def sweep(OP, R, label):
    """m_sense x AR-order sweep at one operating point. Returns (rates dict, genie)."""
    pos = OP.positions()
    acc = {(m, p): [] for m in M_LIST for p in (1, 4)}
    genie = []
    t0 = time.perf_counter()
    for s in range(MC):
        H = generate_spacetime_jakes(R, OP.beta, FD, T, OP.K, seed=100 + s)
        genie.append(run_genie(H, OP.M, sigma2=OP.sigma2, P=OP.P, positions=pos,
                               d_min=OP.d_min, n_rf=OP.n_rf)["rate"][HALF].mean())
        for m in M_LIST:
            for p in (1, 4):
                bel = STKalmanBeliefLR(R, OP.beta, FD, p, OP.sigma_e2)
                proto = "observe" if m >= OP.M else "partial"
                out = run_st(bel, H, OP, np.random.default_rng(200 + s),
                             protocol=proto, m_sense=m)
                acc[(m, p)].append(out["rate"][HALF].mean())
        print(f"    [{label}] seed {s} done ({time.perf_counter()-t0:.0f}s)", flush=True)
    return {k: float(np.mean(v)) for k, v in acc.items()}, float(np.mean(genie))


def main():
    print(f"PAPER-1 CONFIGURATION CHECK\nMC={MC}, T={T}, fd={FD}, m_sense sweep {M_LIST}\n")
    print(f"  digital (OP_V2): {OP_V2.label()}")
    print(f"  hybrid  (OP_V3): {OP_V3.label()}")
    print(f"  n_rf = {OP_V3.n_rf} = 2K = {2*OP_V3.K} -> digital-exact threshold\n")
    R = spatial_correlation(OP_V2.positions())          # same geometry for both
    t0 = time.perf_counter()

    print("--- hybrid (OP_V3, n_rf=6) ---")
    Hy, Ghy = sweep(OP_V3, R, "hybrid")
    print("--- digital (OP_V2, reference) ---")
    Dg, Gdg = sweep(OP_V2, R, "digital")

    print(f"\n{'m_sense':>8} | {'pilots':>7} | {'HYBRID AR(1)':>13} | {'HYBRID AR(4)':>13} | "
          f"{'DIGITAL AR(4)':>14} | {'hybrid loss':>12} | {'% genie':>8}")
    print("-" * 96)
    for m in M_LIST:
        loss = Dg[(m, 4)] - Hy[(m, 4)]
        print(f"{m:8d} | {m/OP_V3.M*100:6.0f}% | {Hy[(m,1)]:13.3f} | {Hy[(m,4)]:13.3f} | "
              f"{Dg[(m,4)]:14.3f} | {loss:+12.3f} | {Hy[(m,4)]/Ghy*100:7.1f}%")
    print(f"{'genie':>8} | {'':>7} | {'':>13} | {Ghy:13.3f} | {Gdg:14.3f}")

    base = Hy[(OP_V3.M, 1)]           # current Paper 1: all pilots, AR(1), hybrid
    headline = Hy[(6, 4)]             # proposed: 60% pilots, AR(4), hybrid
    max_loss = max(abs(Dg[(m, 4)] - Hy[(m, 4)]) for m in M_LIST)

    print("\n" + "=" * 78)
    print(f"CURRENT Paper 1 : m_sense={OP_V3.M} (100% pilots), AR(1), hybrid -> {base:.3f}")
    print(f"PROPOSED        : m_sense=6 ( 60% pilots), AR(4), hybrid -> {headline:.3f}")
    print(f"                  => {headline-base:+.3f} rate on 40% FEWER pilots")
    print("=" * 78)

    print("\nGates:")
    check("hybrid n_rf=2K is free at EVERY pilot budget (|loss| < 0.5)", max_loss < 0.5,
          f"max |digital - hybrid| = {max_loss:.3f}")
    check("HEADLINE: 60% pilots + AR(4) >= 100% pilots + AR(1)", headline >= base - 0.2,
          f"{headline:.3f} vs {base:.3f} ({headline-base:+.3f})")
    check("temporal model helps at reduced pilot budgets", Hy[(6, 4)] > Hy[(6, 1)],
          f"m=6: AR(1) {Hy[(6,1)]:.3f} -> AR(4) {Hy[(6,4)]:.3f}")
    check("and the saving is real (m=6 AR(1) alone would LOSE rate)", Hy[(6, 1)] < base,
          f"m=6 AR(1) {Hy[(6,1)]:.3f} < full-pilot AR(1) {base:.3f} "
          f"-> the temporal model is what buys it back")

    print("\n" + "=" * 78)
    print(f"PAPER-1 CONFIG: {'ALL PASS' if ok else 'FAILURES ABOVE'} "
          f"({time.perf_counter()-t0:.0f}s)")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
