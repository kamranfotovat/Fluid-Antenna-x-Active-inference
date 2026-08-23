r"""
V5-1 gate G1 -- belief + one-per-column sensing, and the Option-B justification (I3).

Checks:
  A. After sensing the 10 droplet ports, their posterior variance collapses to ~sigma_e^2
     (belief actually learns the sensed ports).
  B. I3 (the reason for Option B): sensing ONE port lowers the posterior variance on an
     adjacent-column port under the full 2-D correlation, but NOT under the block-diagonal
     (independent-column) model. Cross-column inference is real and is what informed jumps exploit.
     (Kalman variance update is data-independent, so this needs no channel realization.)
  C. I7: every Sigma_k stays Hermitian PSD after the update.
  D. Aging (predict) re-inflates variance toward the stationary prior.

Run:  python verify_v5_step1.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_B
from channel import ChannelSimulator
from agent_col import make_belief, sense_and_update
from belief import KalmanBelief

OP = OP_B
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print(f"OP_B: {OP.label()}\n")
    N_t, N_p = OP.N_t, OP.N_p
    rng = np.random.default_rng(0)

    # --- A. sensed-port variance collapses ---
    print("A. sensed-port variance collapse")
    sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                           rho=OP.rho, beta=OP.beta, seed=1)
    H = sim.generate(1)
    bel = make_belief(OP)
    i = rng.integers(0, N_p, size=N_t)                 # a random droplet configuration
    prior_v = bel.port_variances().mean()
    S = sense_and_update(bel, H[0], i, OP, rng)
    sensed_v = bel.port_variances()[:, list(S)].mean()
    check("prior port variance ~ beta scale", prior_v > 0.5, f"prior mean var={prior_v:.3f}")
    check("sensed-port variance -> ~sigma_e^2", sensed_v < 0.05,
          f"sensed mean var={sensed_v:.4f} (sigma_e^2={OP.sigma_e2:g})")

    # --- B. I3: cross-column inference, full vs block-diagonal ---
    print("\nB. I3 -- cross-column inference (full 2-D R vs independent-column block R)")
    # observe one port (column 0, height 10); measure variance drop on col 1, same height.
    h_obs = 10 * N_t + 0
    h_adj = 10 * N_t + 1                                # adjacent column, same height (lambda/3)
    dummy_y = np.zeros((OP.K, 1), dtype=complex)        # variance update is data-independent
    drops = {}
    for tag, R in [("full 2-D R", OP.R()), ("block-diag R", OP.R_block())]:
        b = KalmanBelief(R=R, beta=OP.beta, rho=OP.rho, sigma_e2=OP.sigma_e2)
        v_before = b.port_variances()[0, h_adj]
        b.update((h_obs,), dummy_y)
        v_after = b.port_variances()[0, h_adj]
        drops[tag] = v_before - v_after
        print(f"    {tag:14s}: var(adj) {v_before:.4f} -> {v_after:.4f}  (drop {drops[tag]:+.4f})")
    check("full-R lowers adjacent-column variance", drops["full 2-D R"] > 1e-3,
          f"drop={drops['full 2-D R']:.4f}")
    check("block-R does NOT (independent columns)", abs(drops["block-diag R"]) < 1e-9,
          f"drop={drops['block-diag R']:.2e}")
    check("=> cross-column inference is real (Option-B justified)",
          drops["full 2-D R"] > drops["block-diag R"] + 1e-3)

    # --- C. I7: belief health ---
    print("\nC. I7 -- belief stays Hermitian PSD")
    herm = all(np.allclose(bel.Sigma[k], bel.Sigma[k].conj().T) for k in range(OP.K))
    psd = all(np.linalg.eigvalsh(bel.Sigma[k]).min() >= -1e-8 for k in range(OP.K))
    check("Sigma Hermitian", herm)
    check("Sigma PSD", psd)

    # --- D. aging re-inflates ---
    print("\nD. aging (predict) re-inflates variance")
    v_post = bel.port_variances()[:, list(S)].mean()
    bel.predict()
    v_aged = bel.port_variances()[:, list(S)].mean()
    check("predict grows sensed-port variance", v_aged > v_post,
          f"{v_post:.4f} -> {v_aged:.4f}")

    print("\n" + "=" * 44)
    print(f"V5-1 GATE G1: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
