# Progress Since S1 — Catch-Up for Another Device / AI

This documents everything explored **after Paper 1 (S1)** was in good shape, so a fresh session
(here or on another machine) can understand the current state and decisions without re-deriving them.
Read `PROJECT_STATUS.md` first for the original S1 project; this file continues from there.

Last updated: 2026-08-23.

---

## 0. TL;DR — the one thing to internalize

Across ~6 independent experiments we established a **unifying fact**:

> **Active inference's *model/information* value only realizes when the belief drives the rate** —
> i.e. under **predict-then-precode** or **partial/compressed sensing**. Under **observe-then-precode**
> (sense every active port fresh each slot, then precode) you get the best absolute rate, but the
> belief barely touches the rate, so cross-column correlation, horizon info-planning, and online
> R-learning are all **largely decorative**.

Important nuance that PROTECTS Paper 1: this applies to **model**-information (the correlation R),
NOT **state**-information. The **state-epistemic / exploration term is load-bearing** (S1 ablation:
62.6% → 83.5% of genie). Only the *R-learning* extension dies under observe-then-precode.

---

## 1. Paper strategy (decided)

- **Paper 1 (S1, `sim_version3`)** — EFE port selection under partial CSI, 2-D pixel FAS. Essentially
  done; a WCL-sized letter. Its novelty was pressure-tested and **survives** (see §2).
- **Paper 2 (flagship, `sim_version5`)** — liquid-metal **column** FAS + AIF trajectory control. Under
  active construction (see §4). Myopic half built and gated; horizon planning layer 1 built.
- **Extensions** (preferences/QoS, per-port power, active R-learning) — slot into whichever paper;
  active R-learning was explored and shelved (see §5).
- Split into **2 papers, possibly 3**. Don't cram. Paper 1 = protect from being too thin; Paper 2 =
  protect from being over-stuffed.

---

## 2. S1 novelty stress-test (the "rebranding" worry) — RESOLVED

Doubt: is "active inference" just a relabel of rate + mutual-info + switching (Bayesian experimental
design)? Two ablations settled it (`sim_version3/ablation_epistemic.py`, `ablation_naive_headtohead.py`):

- **Epistemic term is load-bearing.** Turning it off (β_w=0) strands the agent at **62.6% of genie**;
  a small weight (β_w≈0.1–0.25) recovers to **83.5% at zero switching**. Advantage grows under stress
  (noisy pilots, fast aging). Not decoration.
- **Information-directed > blind exploration.** AIF (β_w=0.25) **strictly dominates** passive
  round-robin exploration — higher rate, zero switching, higher objective (nominal +5.3 objective vs
  best naive, widening under stress). The claim is "*knowing where to look*", not "exploration helps".

Honest positioning for the paper: cite the Bayesian-experimental-design / submodular-sensing lineage,
lead with the *result* (training-free info-seeking selector beating DRL/naive at low CSI overhead),
frame AIF as the lens. Optional strengthener: **preference-prior (QoS/fairness)** pragmatic term.

---

## 3. Related work (7 papers assessed, in repo root)

None threaten the core EFE-selection novelty (all are CSI-estimation or continuous-position-opt).
Closest models: the **Diffusion-Models** and **AGMAE** papers use the same `y=Sh+n` switch-matrix /
Jakes-2D setup but for *estimation*, not a *selection policy*. The **IDET (JSAC'26)** paper is the key
reference for Paper 2: it uses **liquid columns + movement delay/energy + a DRL (C-SAC) long-term
policy**, but assumes independent FAs (> λ/2) and known CSI. Our Paper-2 wedge vs IDET: **partial CSI
+ belief**, **model-based AIF planning (vs their DRL)**, and **exploiting inter-column correlation**
(they discard it). Borrow: two-timescale framing (Pan/Ren uplink), learned-prior upgrade path
(diffusion/AGMAE), oversampling min-ports theory (New/Wong) for the intro.

---

## 4. Paper 2 — liquid-column FAS (`sim_version5`)

### Model (locked, Option B "dense/coupled")
- **N_t = 10 columns**, each an **N_p = 21**-port 1-D tube, **one droplet per column** → M = N_t = 10
  active elements. pitch = λ/10, **column spacing = λ/3** (dense → FULL 2-D Jakes correlation across
  the array; columns are NOT independent). Footprint ≈ 3.3λ × 2λ, N = 210. Grid maps onto the existing
  `ChannelSimulator` (Nx=N_t, Ny=N_p); global index `n = p·N_t + c`.
- **Action = position vector** i(t) ∈ {0..20}^10; per-slot move limit **Δ_max = 7 ports (0.7λ)**;
  **min-spacing** ≥ λ/2 between active droplets (only adjacent columns bind, need ≥4 ports vertical).
- **Movement cost** η_mv·Σ_c|Δi_c| (ports; lumps delay+energy; reduces to S1 switch count when |Δi|≤1).
- **EFE** G(i) = −α·prag − β_w·epis + movement, over feasible & reachable i.
- **Protocol = observe-then-precode** (primary). Belief = full S1 Kalman over N=210 (reuses `belief.py`).
- Rationale for Δ_max=7 and λ/3: geometry study (`geometry_study.py`) + the informed-jump idea (λ/3
  coupling makes long jumps land on ports already informed by neighbors). Independent-column λ/2 model
  kept only as an ablation baseline ("Option A").

### Gated build (see `sim_version5/SIMULATION_PLAN_V5.md`) — status
- **V5-0..V5-5 (myopic) — ALL BUILT & GATED.** geometry/position algebra, belief+one-per-column
  sensing (I3: cross-column inference is real — sensing one port drops an adjacent-column port's
  variance 0.029 under full R, 0 under block R), feasibility (Δ_max + min-spacing), movement cost +
  **myopic coordinate-descent selector (verified against exhaustive, zero gap)**, closed-loop agent
  (**84% of genie**), and the two ablations.
- **V5-6 horizon planner — LAYER 1 BUILT & GATED** (`planner.py`): receding-horizon per-column
  Viterbi + coordinate descent. Optimizer verified correct (H=1 == myopic; monotone).
- **V5-6 LAYER 2 (drift-aware anticipation) — PARKED, needs discussion.**

### Key Paper-2 findings
- **B (full R) ≈ A (independent-column belief)** under observe-then-precode (myopic): tied on
  objective, A even slightly higher rate; B moves ~4× less. Cross-column correlation modeling does NOT
  win here. Under **predict-then-precode** it DOES (B−A obj +0.81 at ρ=0.9, +1.34 at ρ=0.7,
  `diag_predict.py`) — but predict-then-precode's absolute rate is far lower (~11.5 vs 18.4), so **Kian
  rejected it; observe-then-precode stays primary.** A-vs-B verdict deferred to horizon planning.
- **Horizon planner over-explores** on the stationary channel: moves 15× more than myopic for a
  *worse* objective, because it credits multi-slot epistemic info that doesn't realize under
  observe-then-precode. Optimizer is correct; the model over-values information. → the anticipatory win
  (layer 2) must come from **pragmatic pre-positioning** (foresee where the good channel is heading, on
  a MovingHotspot with a drift-aware belief), NOT epistemic gathering. This is the parked discussion.

---

## 5. Active learning of R for S1 (`sim_version3`, AL-0..AL-3) — SHELVED (clean negative)

Idea: self-calibrating agent learns the spatial correlation online (isotropic profile g(d)) and
ACTIVELY (novelty term) co-observes the near-field the comm policy under-samples. Built end-to-end:
- `dist_profile.py` (g(d) estimator, verified unbiased), `active_learn.py` (`run_aif_learn`,
  `greedy_select_active` with novelty term = Σ 1/√(1+count[bin]) rewarding under-sampled distances,
  `run_random_probe`), `nonstationary.py` (drifting-R channel), `verify_al_step0..3`.

Findings:
- **AL-0:** wrong R costs objective but the gap **decays with T** (+0.72@T24 → +0.087@T120) — the
  penalty is **transient** (belief becomes data-dominated).
- **AL-1:** passive learning recovers **0%** — the comm policy spreads ports ≥λ/2, logging **0**
  near-field co-observations (the informative short distances).
- **AL-2:** the active novelty term *works* (clusters, learns near-field g-err 0.19→0.12) but nets
  **negative** — probing costs rate and there's almost nothing to recover at steady state.
- **AL-3 (non-stationary, Kian's chosen last hope):** oracle-track (knows true drifting R_t) ==
  fixed-static, **gap 0.000** — knowing the exact drifting R gives ZERO benefit under observe-then-precode.

**Conclusion:** active R-learning is not viable under observe-then-precode (state-info matters,
model-info doesn't). Paper 1 unaffected. R-learning's only home = predict-then-precode / partial sensing.

---

## 6. Open decisions / next steps

1. **Paper 2 layer 2** (drift-aware MovingHotspot anticipation) — parked, "needs a lot of discussion."
   The one regime where horizon planning could genuinely beat myopic (pragmatic pre-positioning).
2. **Predict-then-precode variant** — the principled home for the information machinery (correlation,
   learning). Trade-off: much lower absolute rate. Open whether to build a copy version for it.
3. Paper 1: optionally fold in a **preference-prior (QoS/fairness)** subsection; finalize & submit.
4. Notation agreement with Zijun (tutorial M=candidate/N=active vs our N=candidate/M=active).

---

## 7. File map (added since S1)

```
PROGRESS_SINCE_S1.md                     <- this file
results_v5/                              <- Paper-2 figures/results (new folder convention)
sim_version5/                            <- Paper 2 (liquid-column FAS)
  SIMULATION_PLAN_V5.md                   gated plan + locked decisions + invariants
  config.py (ColumnOperatingPoint OP_B)   geometry, Δ_max, spacing, movement weight
  columns.py                              position algebra (pos<->ports) + sensing
  agent_col.py                            belief wiring (reuses KalmanBelief on N=210)
  feasibility.py                          Δ_max reachability + min-spacing mask
  efe_col.py                              movement cost + myopic coordinate-descent selector
  planner.py                              receding-horizon Viterbi planner (layer 1)
  run_col.py                              closed-loop runners (aif/genie/naive/random) + sense_first
  geometry_study.py                       Δ_max / column-length trade study
  diag_predict.py                         observe- vs predict-then-precode B-vs-A diagnostic
  ablation_v5.py                          B-vs-A and B-vs-S1-free ablations
  verify_v5_step0..4, step6.py            gates G0..G4, G6
sim_version3/  (S1 additions)
  ablation_epistemic.py                   epistemic-term ablation (load-bearing proof)
  ablation_naive_headtohead.py            AIF vs blind exploration
  dist_profile.py                         g(d) isotropic correlation estimator
  active_learn.py                         online + active R-learning agent (SHELVED result)
  nonstationary.py                        drifting-R channel generator + oracle-tracking
  verify_al_step0..3_premise.py           active-learning gates (AL thread, shelved)
```

Memory files (auto-memory, not in repo): `epistemic-ablation-result`, `v5-column-protocol-finding`,
`active-learning-S1` capture the same findings for the assistant's recall.
