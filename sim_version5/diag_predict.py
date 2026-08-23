r"""
Diagnostic: does cross-column correlation (B) beat independence (A) under PREDICT-then-precode,
where the belief's inference (not fresh pilots) drives precoding quality?  Compares both protocols
so the contrast is explicit. B = full 2-D R belief; A = block-diagonal (independent-column) belief.
Same dense channel throughout.
"""
from __future__ import annotations
import sys, io, time
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from dataclasses import replace
from config import OP_B
from channel import ChannelSimulator
from run_col import run_col_aif

MC = int(sys.argv[1]) if len(sys.argv) > 1 else 4
T = int(sys.argv[2]) if len(sys.argv) > 2 else 24
HALF = slice(T // 2, None)
t0 = time.perf_counter()


def sweep(sense_first, rho):
    op = replace(OP_B, rho=rho, delta_max=7)
    acc = {"B": {"r": [], "m": []}, "A": {"r": [], "m": []}}
    for s in range(MC):
        sim = ChannelSimulator(Nx=op.Nx, Ny=op.Ny, Wx=op.Wx, Wy=op.Wy, K=op.K,
                               rho=rho, beta=op.beta, seed=400 + s)
        H = sim.generate(T)
        for tag, R in [("B", op.R()), ("A", op.R_block())]:
            r = run_col_aif(op, H, np.random.default_rng(500 + s), R=R, sense_first=sense_first)
            acc[tag]["r"].append(r["rate"][HALF].mean())
            acc[tag]["m"].append(r["move"][HALF].mean())
    out = {}
    for tag in ("B", "A"):
        rr, mm = np.mean(acc[tag]["r"]), np.mean(acc[tag]["m"])
        out[tag] = (rr, mm, rr - op.eta_mv * mm)
    return out


print(f"MC={MC}, T={T}, Delta_max=7, second-half slots\n")
print(f"{'protocol':>20} | {'rho':>4} | {'B obj':>7} | {'A obj':>7} | {'B-A':>7} | "
      f"{'B rate':>7} | {'A rate':>7}")
print("-" * 78)
for sf, name in [(True, "observe-then-precode"), (False, "predict-then-precode")]:
    for rho in (0.9, 0.7):
        o = sweep(sf, rho)
        d = o["B"][2] - o["A"][2]
        print(f"{name:>20} | {rho:>4} | {o['B'][2]:7.3f} | {o['A'][2]:7.3f} | {d:+7.3f} | "
              f"{o['B'][0]:7.3f} | {o['A'][0]:7.3f}")
print(f"\n(total {time.perf_counter()-t0:.0f}s)")
