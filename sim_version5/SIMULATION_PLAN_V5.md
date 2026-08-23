# Simulation Plan — sim_version5 (Paper 2: Liquid-Column FAS + AIF trajectory control)

Gated, debuggable build of the **liquid-metal column** model: a dense fluid-antenna array where
each of `N_t` columns holds ONE droplet that slides along its 1-D tube. The agent selects, every
slot, a **position vector** (one port per column) under a per-slot move limit and a movement cost,
minimizing Expected Free Energy — and it **exploits inter-column correlation** to make informed
long-range repositioning (the decision that separates this from IDET, which discards that
correlation by spacing antennas > λ/2).

Build the same way we built S2: one step, one gate. Do NOT start a step until the previous gate
passes. New code in new files; never regress the S1/S3 or S2 paths.

---

## Locked decisions (agreed 2026-08-22)

- **Model = Option B (dense, coupled).** Column spacing **λ/3** → full 2-D Jakes correlation across
  the whole array (columns are NOT independent). Option A (λ/2, block-diagonal, independent
  columns) is implemented ONLY as an ablation baseline, never the base model.
- **Geometry:** `N_t = 10` columns, `N_p = 21` ports/column, length `L = 2λ`, pitch `λ/10`,
  column spacing `λ/3`. Footprint ≈ 3.3λ × 2λ. Total `N = 210` candidate ports; `M = N_t = 10`
  active (exactly one droplet per column).
- **Belief:** full S1-style Kalman over `N = 210` (reuse `belief.py`; R is the full 2-D Jakes).
  No block-diagonal shortcut in the base model — the cross-column correlation is the point.
- **Sensing:** 1 read per column per slot (the droplet's current port) → 10 reads = an M=10 S1
  budget. Un-sensed ports (incl. other columns') inferred through R.
- **Action:** position vector `i(t) ∈ {1..N_p}^{N_t}`, constrained by:
  - **Δ_max = 7 ports (0.7λ)** per column per slot (physics ceiling; matched to the inference
    radius so long jumps land on informed ports; sweep 2–7).
  - **Min-spacing:** no two active droplets < λ/2 apart. Only ADJACENT columns bind → they need
    **≥ 4 ports vertical separation**; 2-columns-apart (0.667λ) never binds. Enforced by a per-slot
    feasibility mask (reuse `feasible_ports`).
- **Movement cost (replaces S1 switching):** `η_mv · Σ_c |i_c(t) − i_c(t−1)|` in ports (lumps
  delay + energy, both ∝ index difference; reduces to the S1 switch count when |Δi|∈{0,1}).
- **EFE:** `G(i) = −α·rate(i) − β_w·info(i) + η_mv·Σ_c|Δi_c|`, over feasible & reachable `i`.
  Pragmatic = robust-MMSE sum-rate (reuse `efe.pragmatic_value`); epistemic = mutual info (reuse
  `efe.epistemic_value`). Δ_max is the hard limit; η_mv is the economic discipline.
- **Selection:** MYOPIC coordinate-descent first (build path de-risk), horizon Viterbi later.
- **Protocol: OBSERVE-then-precode is primary** (sense active ports fresh, then precode). Decided
  2026-08-23: predict-then-precode was tested (it makes B beat A, gap growing with mobility — see
  `diag_predict.py`) but REJECTED because its absolute rate is far lower (~11.5 vs 18.4 at ρ=0.9).
  Under observe-then-precode A ≈ B myopically; the A-vs-B verdict is DEFERRED to after horizon
  planning (V5-6). `run_col_aif(..., sense_first=True)` (default).
- **Scenario params inherit OP_V3:** K=3, ρ=0.9, σ²=0.03 (15 dB), σ_e²=1e-3, β=[1,0.7,1.3],
  α=1.0, β_w sweepable, η_sw→η_mv.

---

## Invariants (regression anchors — check these hold at every step)

- **I1 (master anchor → S1).** With column spacing large enough to be independent, Δ_max = ∞,
  η_mv = 0, min-spacing off, and the one-per-column constraint relaxed to free selection, the
  belief update and realized rate must match the S1 (sim_version3) path bit-for-bit on a shared
  trajectory. Proves the new code reuses the validated core.
- **I2 (movement ↔ switching).** With |Δi_c| ∈ {0,1}, `η_mv·Σ|Δi_c|` equals the S1 switching count
  `|S △ S_prev|`. The metric cost is the strict generalization of the S1 switch cost.
- **I3 (correlation is real).** At λ/3, cross-column posterior-variance coupling is present
  (sensing column c lowers variance on adjacent-column ports); in the block-diagonal ablation it
  is absent. Validates the reason to choose Option B.
- **I4 (planner sanity).** Coordinate descent is monotone-improving in G per sweep and converges;
  horizon Viterbi ≥ myopic on long-term objective.
- **I5 (feasibility).** Every selected config respects Δ_max and min-spacing; the feasible set
  never empties (there is always a legal move — at worst "stay").
- **I6 (design > correlation-blind).** At long Δ_max, the coupled agent (B) ≥ the independent-column
  agent (A) on objective — informed jumps beat blind jumps.
- **I7 (belief health).** Every Σ_k stays Hermitian PSD (Joseph form, inherited from S1).

---

## Gated steps

### V5-0 — Scaffold, geometry, position algebra  → `config.py`, `columns.py`, `verify_v5_step0.py`
Build `ColumnOperatingPoint` (all locked params). Build column port coordinates (2-D) and the full
2-D Jakes R via `channel.spatial_correlation`. Implement the position algebra:
`pos_to_ports(i) → global port-index set S` (one per column) and back. Also build the
block-diagonal R for the Option-A ablation.
**Gate G0:** R PSD & unit-diag; adjacent-column correlation ≈ +0.17 (λ/3), 2-cols ≈ −0.38 (matches
hand calc); `pos_to_ports` yields 10 distinct global indices with correct column membership;
round-trips. Independent-R constructible.

### V5-1 — Belief + one-per-column sensing  → `agent_col.py` (belief wiring), `verify_v5_step1.py`
Wire `KalmanBelief` (full N=210 R) + observe the 10 droplet ports each slot.
**Gate G1:** sensed-port posterior variance → ~σ_e²; **I3** — variance also drops on λ/3
adjacent-column ports (quantify the drop vs the block-diagonal case where it does not). **I7** holds.

### V5-2 — Feasibility: reachability + min-spacing mask  → `feasibility.py`, `verify_v5_step2.py`
`reachable(i_prev, Δ_max)` (per-column ports within Δ_max) and `feasible(i)` (min-spacing across
adjacent columns via `feasible_ports`). Combined per-slot legal-move mask.
**Gate G2:** unit tests — reachable sizes correct; min-spacing forbids exactly the <λ/2 adjacent
pairs and nothing else; a known-good/known-bad config classified right; **I5** (never empties).

### V5-3 — Movement cost + myopic EFE selection  → `efe_col.py`, `verify_v5_step3.py`
`G(i)` over feasible∩reachable positions; myopic **coordinate-descent** selector (init at i_prev,
sweep columns, each picks best legal port given others fixed, iterate to convergence).
**Gate G3:** **I2** (movement=switching when |Δi|≤1); **I4** (coord-descent monotone & converges);
myopic beats random-feasible on objective; **I1** regression (independent + Δ_max=∞ + η_mv=0 + free
→ matches S1).

### V5-4 — Closed-loop myopic agent + baselines  → `run_col.py`, `verify_v5_step4.py`
Loop: predict → myopic select → sense droplet ports → update → robust-MMSE precode → log
rate/movement/info. Baselines on the shared trajectory: **S1-free** (no column constraint, upper
ref), **naive/random** (partial-CSI lower refs), **genie** (full-CSI ceiling).
**Gate G4:** runs closed-loop; column-AIF beats naive/random on objective; sensible %genie.

### V5-5 — The two justifying ablations (paper results)  → `ablation_v5.py`
(a) **B (λ/3 coupled) vs A (λ/2 independent)** at Δ_max=7 → value of exploiting cross-column
correlation (informed jumps). (b) **B vs S1-free** → cost of the one-droplet-per-column hardware
constraint.
**Gate G5:** **I6** — B > A on objective at long Δ_max; B within a stated gap of S1-free. Both are
figures for the paper.

### V5-6 — Horizon planning (flagship core)  → `planner.py`, `verify_v5_step6.py`
Per-column **Viterbi** over horizon H (predicted belief evolution, Δ_max transitions), wrapped in
coordinate descent across columns; discounted EFE.
**Gate G6:** **I4** — horizon ≥ myopic on long-term objective; planner pre-positions before a
predictable channel change (show a trajectory); captures the 7–10% headroom the geometry study
found at small Δ_max.

### V5-7 — Frontiers, trajectory GIF, results doc  → `make_*_v5.py`, `results_v5.md`
Sweep η_mv, Δ_max, β_w → objective/rate/movement frontiers; a droplet-trajectory animation;
write-up. **Gate G7:** frontiers coherent; sweet-spots match geometry-study predictions.

### V5-8 (Paper-2 second engine, PARKED) — active online learning of R, ρ + model-epistemic term
Deferred. Build only after V5-6. Nothing above may block it.

---

## Build order & de-risking
Myopic (V5-0..V5-5) is a complete, publishable baseline on its own — coordinate-descent single-step
selection over the reachable/feasible set, essentially S1 greedy adapted to Δ_max + min-spacing.
Horizon planning (V5-6) is the enhancement, not a prerequisite. So the flagship is reachable
incrementally; we never have to solve coupled horizon planning before we have working results.
