r"""
V5-0 gate G0 -- geometry, correlation, position algebra.

Checks:
  A. Full 2-D Jakes R is PSD and unit-diagonal.
  B. Named correlations match the hand calc (adjacent col ~ +0.17, 2 cols ~ -0.38, within-col
     first-neighbour ~ +0.998).
  C. Geometry: column c sits at x = c*col_spacing; height p at y = p*pitch.
  D. Position algebra: pos_to_ports gives N_t distinct indices with correct column membership,
     and round-trips through ports_to_pos.
  E. Block-diagonal R (Option-A ablation) is truly block-diagonal (zero cross-column) and PSD.

Run:  python verify_v5_step0.py
"""

from __future__ import annotations

import sys
import numpy as np
from scipy.special import j0

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_B
from columns import pos_to_ports, ports_to_pos, column_of, height_of

OP = OP_B
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print(f"OP_B: {OP.label()}\n")
    pos = OP.positions()
    R = OP.R()
    N, N_t, N_p = OP.N, OP.N_t, OP.N_p

    print("A. correlation matrix health")
    w = np.linalg.eigvalsh(0.5 * (R + R.T))
    check("R PSD (min eig >= -1e-9)", w.min() >= -1e-9, f"min eig={w.min():.2e}")
    check("R unit diagonal", np.allclose(np.diag(R), 1.0))
    check("R symmetric", np.allclose(R, R.T))

    print("\nB. named correlations vs hand calc")
    # ports by (column c, height p): n = p*N_t + c
    n00 = 0 * N_t + 0                 # col 0, height 0
    n_adj = 0 * N_t + 1              # col 1, height 0   (adjacent column, same height, d=lambda/3)
    n_2col = 0 * N_t + 2            # col 2, height 0   (2 cols, d=2 lambda/3)
    n_up1 = 1 * N_t + 0             # col 0, height 1   (within column, d=pitch)
    for name, na, nb, want in [
        ("adjacent col same height (lambda/3)", n00, n_adj, j0(2 * np.pi * OP.col_spacing)),
        ("2 columns apart (2 lambda/3)", n00, n_2col, j0(2 * np.pi * 2 * OP.col_spacing)),
        ("within column, 1 port (lambda/10)", n00, n_up1, j0(2 * np.pi * OP.pitch)),
    ]:
        got = R[na, nb]
        check(name, abs(got - want) < 1e-9, f"R={got:+.3f} (want {want:+.3f})")

    print("\nC. geometry (coords match column/height indices)")
    xs_ok = np.allclose(pos[:, 0], column_of(np.arange(N), N_t) * OP.col_spacing)
    ys_ok = np.allclose(pos[:, 1], height_of(np.arange(N), N_t) * OP.pitch)
    check("x-coord = column * col_spacing", xs_ok)
    check("y-coord = height * pitch", ys_ok)

    print("\nD. position algebra")
    rng = np.random.default_rng(0)
    all_rt = True; all_cols = True; all_distinct = True
    for _ in range(200):
        i = rng.integers(0, N_p, size=N_t)
        S = pos_to_ports(i, N_t)
        all_distinct &= (len(set(S)) == N_t)
        all_cols &= (sorted(column_of(np.array(S), N_t).tolist()) == list(range(N_t)))
        all_rt &= np.array_equal(ports_to_pos(S, N_t), i)
    check("pos_to_ports gives N_t distinct indices", all_distinct)
    check("exactly one port per column", all_cols)
    check("ports_to_pos round-trips", all_rt)

    print("\nE. block-diagonal R (Option-A ablation)")
    Rb = OP.R_block()
    cols = np.arange(N) % N_t
    cross = cols[:, None] != cols[None, :]
    check("cross-column entries are zero", np.allclose(Rb[cross], 0.0))
    check("within-column entries preserved", np.allclose(Rb[~cross], R[~cross]))
    wb = np.linalg.eigvalsh(0.5 * (Rb + Rb.T))
    check("block R PSD", wb.min() >= -1e-9, f"min eig={wb.min():.2e}")

    print("\n" + "=" * 44)
    print(f"V5-0 GATE G0: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
