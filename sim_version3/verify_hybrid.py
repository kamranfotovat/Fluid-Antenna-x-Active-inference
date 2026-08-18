r"""Smoke test for the hybrid factorization (version 3).

(A) Factorization loss vs n_rf on a random M x K digital target: check the >= 2K lossless
    threshold (loss should crash toward -inf once n_rf >= 2K).
(B) One closed-loop AIF slot-run at OP_V3, digital vs hybrid, to confirm the wiring works
    and the hybrid rate approaches the digital rate at n_rf = 2K.
"""
from __future__ import annotations

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_V3
from channel import ChannelSimulator
from agent import AIFAgent, run_aif, run_genie
from hybrid import factorize_hybrid, factorization_loss_db

OP = OP_V3
POS, R = OP.positions(), OP.R()
K, M = OP.K, OP.M


def part_a():
    print(f"(A) Factorization loss vs n_rf  (random target M={M} x K={K}, 2K={2*K})")
    rng = np.random.default_rng(1)
    F = (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))) / np.sqrt(2)
    print("| n_rf | approx loss (dB) |")
    print("|---:|---:|")
    for n_rf in range(K, M + 1):
        _, _, W_eff = factorize_hybrid(F, n_rf, P=1.0, rng=np.random.default_rng(7))
        print(f"| {n_rf} | {factorization_loss_db(F, W_eff):.1f} |")
    print()


def part_b():
    print(f"(B) Closed-loop AIF at OP_V3 (M={M}, K={K}), one seed, T=30 -- digital vs hybrid")
    sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy,
                           K=OP.K, rho=OP.rho, beta=OP.beta, seed=0)
    H = sim.generate(30)
    HALF = slice(15, None)

    def aif(n_rf):
        ag = AIFAgent(R, OP.beta, OP.rho, OP.sigma_e2, M, 1.0, OP.beta_w, OP.eta_sw,
                      sigma2=OP.sigma2, positions=POS, d_min=OP.d_min, n_rf=n_rf)
        return run_aif(ag, H, OP.sigma_e2, np.random.default_rng(100), sense_first=True)

    dig = aif(None)["rate"][HALF].mean()
    g_dig = run_genie(H, M, sigma2=OP.sigma2, positions=POS, d_min=OP.d_min)["rate"][HALF].mean()
    print(f"  genie (digital) : {g_dig:.2f}")
    print(f"  AIF   (digital) : {dig:.2f}")
    for n_rf in [3, 4, 6, 8, 10]:
        r = aif(n_rf)["rate"][HALF].mean()
        print(f"  AIF   (hybrid n_rf={n_rf:2d}) : {r:.2f}   ({100*r/dig:.0f}% of digital)")


if __name__ == "__main__":
    part_a()
    part_b()
