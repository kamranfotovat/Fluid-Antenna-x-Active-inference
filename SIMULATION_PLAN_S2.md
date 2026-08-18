# S2 build plan — sensing THROUGH the analog network (`sim_version4/`)

**What S2 is.** In S1 (the letter, `sim_version3/`) the agent reads each activated port individually:
`y = P_S h + n`. In **S2** the RF-chain budget bites at *sensing* too — the agent reads only `n_rf_sense`
**mixed** measurements through a unit-modulus analog combiner it gets to **design**:

```
    y = F_RF^H P_S h + n ,     F_RF ∈ C^{M×n_rf_sense}, |F_RF[i,j]| = 1 ,   y ∈ C^{n_rf_sense}
```

So the observation matrix becomes `A = F_RF^H P_S` (n_rf_sense × N). The model stays **linear-Gaussian**, so
the Kalman engine is UNCHANGED in structure — we only swap the observation matrix. What is new is that the
**epistemic EFE term becomes a function of `F_RF`**, so the agent *designs its sensing beams* to probe where
its belief is most uncertain. This is `FUTURE_WORK.md §0`. We build the three tiers there in order.

This plan is deliberately **incremental and gated**: every step ships one small piece + one `verify_s2_*.py`
smoke test + a hard numeric GATE. Nothing proceeds until the gate is green. See the Invariants list (§I) for
the checks reused everywhere — they are the debugging backbone.

---

## Locked design decisions (read before coding)

1. **Sensing objective = information about the AGGREGATE uncertainty.** The analog net is *shared* across the
   K users, so we design one `F_RF` against the aggregate active-port covariance `Σ̄_S = Σ_k Cov_k(S)`
   (M×M) — the same aggregate `E` the robust-MMSE precoder already uses (`efe.robust_mmse_from_belief`):
   ```
       J_sense(F_RF; S) = logdet( I_{n_rf} + σ_e^{-2} · F_RF^H Σ̄_S F_RF )      (bits, maximize)
   ```
   This is ONE clean matrix objective whose unconstrained optimum is exact (eigen/water-filling), which makes
   the reference bound and all gates well-defined. (Per-user `Σ_k logdet(...)` is a documented variant, not
   the default — its aggregate-eigenvector solution is only a heuristic.)

2. **Sensing and transmit analog matrices are SEPARATE** in Light/Medium-S2 (`F_RF^sense` designed for info,
   `F_RF^tx` = the v3 transmit factorization). This deliberately **avoids** the explore/exploit coupling so we
   can validate the sensing design in isolation. Coupling them into ONE shared network is **Full-S2 only**.

3. **S1 must stay green (regression rule).** `sim_version4` is a *superset* of `sim_version3`. Every existing
   v3 verify script must still pass unchanged after every step. The clean reduction `F_RF = I_M, n_rf = M`
   must reproduce S1 bit-for-bit (Invariant I1) — this is our single most important correctness anchor.

4. **New code goes in NEW files / additive methods**, never by rewriting S1 paths in place:
   - `belief.py`  → ADD `update_general(A, y)`; make the existing `update(S, y)` delegate to it.
   - `sensing.py` (NEW) → observation matrix, info objective, unconstrained bound, unit-modulus optimizer.
   - `agent.py`   → ADD `run_aif_s2(...)`; leave all S1 runners untouched.
   - `config.py`  → ADD `n_rf_sense` field + `OP_V4`.
   - `verify_s2_*.py` (NEW per step); `make_s2_tables.py` → `results_v4.md`.

---

## Steps (each: deliverable → verify script → GATE)

### S2-0 — scaffold & regression baseline
- **Do:** `OP_V4` in `config.py` = OP_V3 geometry (N=441, M=10, K=3) + new field `n_rf_sense` (None = S1
  per-port sensing). `ACTIVE = OP_V4`. No behavioural change yet.
- **Verify:** `verify_hybrid.py` (v3) runs; a 1-seed closed-loop AIF at OP_V4.
- **GATE:** S1 numbers reproduce (AIF digital rate ≈ 18.6 at OP_V4, unchanged from v3). Proves the copy is clean.

### S2-1 — generalized (arbitrary-A) Kalman update  *(perception, no decisions)*
- **Do:** add `KalmanBelief.update_general(A, y)` for a complex observation matrix `A ∈ C^{m×N}`, per-user
  `y[k] = A h_k + CN(0, σ_e² I_m)`. Innovation `Scov = A Σ_k A^H + σ_e² I_m`; Joseph form as today but with
  `A` (conjugate-transpose) in place of the real `P`. Rewrite `update(S, y)` to call
  `update_general(selection_matrix(S,N).astype(complex), y)`.
- **Verify:** `verify_s2_step1.py`.
- **GATE:** (a) **regression** — `update(S,y)` gives Σ,μ identical (≤1e-10) to the pre-change method on a
  fixed seed; (b) with `A = P_S` (complex), `update_general` ≡ `update`; (c) **calibration** — for a random
  complex `A`, Monte-Carlo empirical posterior cov of `h|y` matches filter `Σ` within <5% (same test style as
  S1 Step 3); (d) `Σ` stays Hermitian PSD (Invariant I5).

### S2-2 — sensing-through-F_RF observation model
- **Do:** `sensing.py`: `observation_matrix(F_RF, S, N)` → `A = F_RF.conj().T @ P_S`;
  `sense(H_t, F_RF, S, σ_e², rng)` → `y ∈ C^{K×n_rf}` = `A @ h_k + noise`.
- **Verify:** `verify_s2_step2.py`.
- **GATE (I1, the anchor):** with `n_rf = M` and `F_RF = I_M`, `A = P_S`, and the whole pipeline
  (sense → `update_general`) is **bit-identical to S1** (`update(S,y)`); the resulting epistemic info equals
  `efe.epistemic_value`. Also: shapes correct, noise power ≈ σ_e², `trace(Σ)` on active ports drops after update.

### S2-3 — info-gain objective + unconstrained reference bound
- **Do:** `sensing.sense_info(F_RF, cov_bar, σ_e²)` = the `J_sense` logdet above;
  `sensing.optimal_unconstrained(cov_bar, n_rf, σ_e²)` = top-`n_rf` eigenvectors of `Σ̄` (water-filling),
  the **upper bound** any unit-modulus `F_RF` must respect.
- **Verify:** `verify_s2_step3.py`.
- **GATE:** (a) a black-box numeric maximizer (unconstrained, e.g. gradient ascent on `F_RF`) reaches the
  eigen-formula value (≤1e-6); (b) `J_sense` **monotone increasing in n_rf** (I2) and **saturates at
  n_rf = rank(Σ̄)** (≈ spatial DoF, small for dense sub-λ arrays — a nice sanity check).

### S2-4 — unit-modulus sensing-matrix optimizer  *(the core new solver)*
- **Do:** `sensing.design_sensing_matrix(cov_bar, n_rf, σ_e², n_iter, n_restart, rng)` — maximize `J_sense`
  over unit-modulus `F_RF` by **coordinate / Riemannian ascent** (reuse the per-entry closed-form spirit of
  `hybrid._update_FRF`, but the objective is logdet, not Frobenius). Warm-start from the phases of the
  unconstrained optimum; multi-restart; return best `F_RF`.
- **Verify:** `verify_s2_step4.py`.
- **GATE:** (a) objective **non-decreasing every iteration** (assert monotonicity — the #1 debug signal);
  (b) `|F_RF| = 1` exactly; (c) `J_designed ≤ J_unconstrained` (I4) and the gap **shrinks as n_rf → M**;
  (d) `J_designed > J_random` unit-modulus by a clear margin (I3); (e) restart-stable (spread across restarts small).

### S2-5 — Light-S2 closed loop (FIXED active set)  ← first real milestone
- **Do:** `agent.run_aif_s2(agent, H, σ_e², rng, n_rf_sense, select_mode)`. Per slot: `predict`; pick S
  (`select_mode='s1'` reuses the v3 greedy, or `'fixed'` holds a set); build `Σ̄_S`; `F_RF^sense =
  design_sensing_matrix(...)`; `y = sense(...)`; `update_general(observation_matrix(F_RF^sense,S,N), y)`;
  precode with the v3 transmit hybrid; score rate. Selection & transmit are the SAME as S1 — only the sensing
  read changes.
- **Verify:** `verify_s2_step5.py` — closed loop, one seed, T=30.
- **GATE (the Light-S2 headline):** at an **equal sensing budget** of `n_rf_sense` ADC reads/slot, the
  designed combiner beats reading `n_rf_sense` *individual* ports (an "S1-subset" baseline) on **belief quality
  (lower `real_err`/`trace Σ`)** and **rate**; and it approaches the `n_rf_sense = M` full-per-port ceiling as
  `n_rf_sense → M`. Calibration still holds in closed loop (<5%). Message: *smart compression of M ports into
  n_rf_sense reads ≈ reading all M.*

### S2-6 — Medium-S2: joint selection + sensing design
- **Do:** the greedy epistemic term uses the **F_RF-designed** info gain (design a small `F_RF` per candidate
  set, or a fast proxy — e.g. sum of top-`n_rf` eigenvalues of `Σ̄_cand`) so port selection is aware that
  sensing is compressed. New runner / flag.
- **GATE:** joint ≥ fixed-set (S2-5) at equal budget; log the per-slot compute overhead (expect the proxy to
  be needed at N=441 — the eigenvalue proxy is O(M³) per candidate vs a full optimizer).

### S2-7 — Full-S2: ONE shared analog network (explore/exploit)  ← flagship / may defer
- **Do:** sensing and transmit share `F_RF` across a coherence block. Model the pilot-vs-data split: option A
  = two configs (sensing `F_RF` in pilot sub-slot, transmit `F_RF` in data sub-slot — collapses toward S1);
  option B = one compromise `F_RF` scored by a combined EFE (info + rate); option C = a schedule. Prototype A
  vs B; define the combined objective.
- **GATE:** define + prototype; quantify the explore/exploit loss of sharing vs separate. Likely its own paper;
  strongest in the continuous movable-antenna limit (`FUTURE_WORK.md §2e`) as *active spatial sampling*.

### S2-8 — tables/figures → `results_v4.md`
- **Do:** `make_s2_tables.py`. Core figure: **sensing-efficiency curve** — belief RMSE and rate vs `n_rf_sense`
  for {S2-designed combiner, S1 per-port subset, random combiner, full-per-port ceiling}. Plus the I1
  reduction check and a compute-cost note.
- **Headline target:** "a designed analog sensing net recovers ≈X% of full per-port CSI quality using only
  n_rf_sense/M of the ADC reads — the agent designs *how it looks*, not just *where*."

---

## §I — Invariants tested everywhere (the debugging backbone)
- **I1 (reduction):** `F_RF = I_M, n_rf = M` ⟹ S2 ≡ S1 bit-for-bit. The master anchor; test it at every layer.
- **I2 (monotonicity in budget):** `J_sense` and closed-loop belief quality improve (weakly) as `n_rf_sense` ↑.
- **I3 (design beats random):** designed `F_RF` > random unit-modulus `F_RF`.
- **I4 (bounded by unconstrained):** designed `J` ≤ eigen/water-filling optimum.
- **I5 (belief sanity):** `Σ_k` Hermitian PSD after every update (max|Σ−Σ^H|, min eig ≥ −1e-9).
- **I6 (calibration):** filter `Σ` matches Monte-Carlo empirical posterior within <5% (as in S1 Step 3).
- **I7 (optimizer monotone):** `design_sensing_matrix` objective never decreases across iterations.

## Notation note (carry from S1)
Our code: `N` = candidate ports, `M` = active, `K` = users, `n_rf` = transmit chains, **`n_rf_sense`** = sense
chains (new). Zijun's tutorial swaps M/N — settle one manuscript convention with him before writing S2 up.
