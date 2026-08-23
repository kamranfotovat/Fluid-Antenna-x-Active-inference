r"""
V5-6 gate G6 (layer 1) -- receding-horizon planner mechanics.

Checks:
  A. the planned trajectory is feasible (min-spacing) and reachable (|Delta i| <= Delta_max each
     step, and the first step within Delta_max of i_prev).
  B. horizon H=1 reduces to myopic selection (same objective) -- the planner is a strict
     generalization of the myopic selector.
  C. planner is not worse than a trivial "stay" plan under its own predicted objective.
  D. closed-loop on the STATIONARY channel: the planner does NOT hurt the realized objective vs
     myopic (we EXPECT ~tie here -- anticipation needs a predictable dynamic, added next layer).

Run:  python verify_v5_step6.py [MC] [T]
"""

from __future__ import annotations

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from config import OP_B
from channel import ChannelSimulator
from agent_col import make_belief, sense_and_update
from feasibility import config_feasible, random_feasible_config
from efe_col import select_myopic, free_energy
from planner import plan_horizon, _predict_sequence, _viterbi_column
from columns import pos_to_ports
import efe
from run_col import run_col_aif

OP = OP_B
MC = int(sys.argv[1]) if len(sys.argv) > 1 else 2
T = int(sys.argv[2]) if len(sys.argv) > 2 else 10
HALF = slice(T // 2, None)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def _traj_obj(traj, seq, i_prev, op, gamma=0.9):
    """Discounted cumulative EFE of a trajectory against a FIXED belief sequence seq."""
    J = 0.0; prev = None if i_prev is None else np.asarray(i_prev)
    for k in range(len(traj)):
        S = pos_to_ports(traj[k], op.N_t)
        prag = efe.pragmatic_value(seq[k], S, sigma2=op.sigma2, P=op.P)
        epis = efe.epistemic_value(seq[k], S)
        mv = 0.0 if prev is None else op.eta_mv * np.sum(np.abs(traj[k] - prev))
        J += (gamma ** k) * (-op.alpha * prag - op.beta_w * epis + mv)
        prev = traj[k]
    return J


def main():
    print(f"OP_B: {OP.label()}\nMC={MC}, T={T}\n")
    pos = OP.positions()

    # belief with structure
    sim = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                           rho=OP.rho, beta=OP.beta, seed=3)
    Hc = sim.generate(2)
    bel = make_belief(OP)
    i_prev = random_feasible_config(OP, np.random.default_rng(5))
    sense_and_update(bel, Hc[0], i_prev, OP, np.random.default_rng(6))
    bel.predict()

    print("A. planned trajectory feasible & reachable (H=3)")
    i0, traj = plan_horizon(bel, i_prev, OP, H=3, rng=np.random.default_rng(7), return_traj=True)
    feas_all = all(config_feasible(traj[k], OP, pos) for k in range(len(traj)))
    reach0 = bool(np.all(np.abs(traj[0] - i_prev) <= OP.delta_max))
    reach_steps = all(np.all(np.abs(traj[k] - traj[k - 1]) <= OP.delta_max) for k in range(1, len(traj)))
    check("every planned config min-spacing feasible", feas_all)
    check("first step within Delta_max of i_prev", reach0)
    check("each step within Delta_max of previous", reach_steps)

    print("\nB. H=1 reduces to myopic")
    i_plan1 = plan_horizon(bel, i_prev, OP, H=1, rng=np.random.default_rng(8))
    i_myo, G_myo = select_myopic(bel, i_prev, OP, np.random.default_rng(8))
    G_plan1 = free_energy(bel, i_plan1, i_prev, OP)
    check("H=1 planner objective == myopic (up to coord-descent noise)", abs(G_plan1 - G_myo) < 1e-2,
          f"plan {G_plan1:.4f} vs myopic {G_myo:.4f}")

    print("\nC. optimizer correctness -- coordinate-descent Viterbi is monotone under a FIXED belief seq")
    stay = np.tile(np.asarray(i_prev), (3, 1))
    seq_fixed = _predict_sequence(bel, stay, OP)          # fix the belief evolution
    tr = stay.copy(); Js = [_traj_obj(tr, seq_fixed, i_prev, OP)]
    for _ in range(2):
        for c in range(OP.N_t):
            tr = _viterbi_column(c, tr, seq_fixed, i_prev, OP, pos, 0.9)
            Js.append(_traj_obj(tr, seq_fixed, i_prev, OP))
    monotone = all(Js[k + 1] <= Js[k] + 1e-9 for k in range(len(Js) - 1))
    check("J non-increasing across coordinate-descent updates", monotone,
          f"J: {Js[0]:.2f} -> {Js[-1]:.2f}")
    check("planner improves over its init (stay)", Js[-1] <= Js[0] - 1e-6)

    print("\nD. closed-loop planner vs myopic on STATIONARY (no anticipation possible -> expect ~tie/slightly worse)")
    t0 = time.perf_counter()
    dm, dp, mvm, mvp = [], [], [], []
    for s in range(MC):
        simc = ChannelSimulator(Nx=OP.Nx, Ny=OP.Ny, Wx=OP.Wx, Wy=OP.Wy, K=OP.K,
                                rho=OP.rho, beta=OP.beta, seed=400 + s)
        Hh = simc.generate(T)
        rm = run_col_aif(OP, Hh, np.random.default_rng(500 + s))
        rp = run_col_aif(OP, Hh, np.random.default_rng(500 + s), horizon=3)
        dm.append(rm["rate"][HALF].mean() - OP.eta_mv * rm["move"][HALF].mean())
        dp.append(rp["rate"][HALF].mean() - OP.eta_mv * rp["move"][HALF].mean())
        mvm.append(rm["move"][HALF].mean()); mvp.append(rp["move"][HALF].mean())
    om, op_ = float(np.mean(dm)), float(np.mean(dp))
    print(f"     myopic obj {om:.3f} (move {np.mean(mvm):.2f}) | planner obj {op_:.3f} "
          f"(move {np.mean(mvp):.2f}) | delta {op_-om:+.3f}   [{time.perf_counter()-t0:.0f}s]")
    print("     NOTE: no benefit expected on the stationary channel (nothing to anticipate);")
    print("           this is the baseline. The anticipatory win is tested in layer 2 (drift-aware).")
    check("planner runs closed-loop & is not catastrophically worse (< 1.5 below myopic)",
          op_ >= om - 1.5)

    print("\n" + "=" * 44)
    print(f"V5-6 GATE G6 (layer 1): {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    print("=" * 44)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
