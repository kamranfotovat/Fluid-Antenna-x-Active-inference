# Simulation Plan — Active Inference for FAS Port Selection (v1: implicit sensing)

Companion to `RESEARCH_PLAN.md` (what/why) and `EFE_DESIGN.md` (the decision rule).
This document is the **build plan**: how the agent's objects are **initialized**, the **timescales**
they run at, the concrete **system dimensions** (K, N, M) and why, and the **staged, verify-as-you-go**
coding steps. Notation kept in plain text so it renders in any markdown preview.

---

## 1. Generative-model objects — how each is initialized (priors, preference, habit)

We do NOT use a discrete `pymdp` A/B/C/D/E setup — this is a **linear-Gaussian** agent. But the canonical
AIF objects map onto it cleanly, and each must be given a concrete value before we can simulate.

| Canonical AIF | Our object | What we set it to (v1) |
|---|---|---|
| **D** — initial state prior | initial belief `q(h_k) = CN(mu0, Sigma0)` | **mu0 = 0**, **Sigma0 = beta_k * R** — the *stationary* distribution of the AR(1) process. Best belief before any observation = zero-mean with the channel's own steady-state covariance. Physics-given, not a tuning knob. |
| **B** — transition | AR(1): `rho`, process noise `(1 - rho^2) * beta_k * R` | Given by physics in v1. Its *parameters* (`rho, beta_k`) are what the SLOW loop learns later (Sec. 2). |
| **A** — likelihood | `y = P_S h + CN(0, sigma_e^2 I)` | Selection matrix `P_S` (which ports are observed) + estimation-noise variance `sigma_e^2`. Fixed in v1; a slow-learned parameter later. |
| **C** — preferences | (the "preference") | **No separate preferred-observation vector — we use a utility.** The log-preference over outcomes = the **rate**: `PragmaticValue = E[ sum_k log2(1 + SINR_k) ]`. The **switching penalty** is a negative preference over *transitions*. So preference = "prefer high-rate, low-churn outcomes" and it IS the Eq. 7 objective. No arbitrary preference tuning. |
| **E** — habit | prior over which port set to pick | **v1: flat (no habit).** Stickiness toward the last set is already provided by the switching cost, so a learned habit is unnecessary. Habit-learning (Dirichlet over frequently-good ports) is an optional later ablation, off by default. |

**Why preference = utility (not a hand-set vector):** in textbook AIF `C` is a vector you tune. Here the
preference is *derived* from the communication objective (rate − switching). Cleaner and reviewer-friendly:
the preference is the physics/system objective, nothing to hand-pick.

---

## 2. The three timescales (perception / action / learning)

Three loops run at different rates. Keeping them separate is what lets us get the core loop correct
before adding the complication of learning the model.

```
FAST  (every slot t):   PERCEPTION  -> Kalman predict/update of (mu_k, Sigma_k)     [state inference]
FAST  (every slot t):   ACTION      -> greedy EFE selection of S_t                  [policy]
SLOW  (over many slots): LEARNING   -> estimate params rho, beta_k, sigma_e^2 (R)   [parameter inference]
```

- **Fast loops (v1 focus):** perception + action treat the generative-model parameters
  (`rho, beta, R, sigma_e^2`) as **fixed and known** — we hand the agent the true physics values.
- **Slow loop (deferred to Step 8):** the agent does NOT actually know the Doppler `rho` or power `beta_k`;
  it estimates them from the observation stream with a **forgetting factor / slow EMA** so it tracks drift
  without chasing per-slot noise. Estimate `rho` from empirical autocorrelation of successive observations,
  `beta_k` from sample power, `sigma_e^2` from residuals. This slow loop **is** the "learning at a different
  timescale," and it powers **Fig E (model-mismatch / online estimation)**.

Rationale for fixing params in v1: isolate whether the **decision rule** works before adding model learning.

---

## 3. System dimensions — K, N, M (and the best values for v1)

The constraint from the system model is **N >= M >= K**. Within that, here is what each dimension does and
the value we recommend for the version we are actually building (v1: implicit sensing, partial CSI).

### K — number of users (served streams)
- **Need K >= 2** or there is no inter-user interference (IUI) and the ZF / robust-MMSE precoder story
  collapses (with K=1 there is nothing to null). IUI suppression is half the point of the precoder.
- Cost scales with K (we run **K independent Kalman filters**, one per user's channel).
- **Recommended: K = 3.** Enough IUI to make precoding meaningful, still light. Use **K = 2 for debugging**.

### N — number of ports (belief dimension)
- N is the dimension of each user's channel `h_k in C^N`, so the covariance `Sigma_k` is **N x N**. Kalman
  cost grows with N; keep it modest for fast iteration and a reproducible DQN baseline.
- The **partial-CSI headline** ("matches genie while observing a fraction of ports") is stronger when the
  observed fraction `M/N` is small -> argues for N not too small.
- The **neighbor-inference / epistemic story** needs ports **densely packed (sub-lambda/2 spacing)** so the
  spatial correlation `R` has strong off-diagonals — observing one port then informs its neighbors. FAS is
  exactly this sub-wavelength regime, which is favorable for us.
- Too small (N=9): observing M=5 covers most ports -> partial-CSI advantage muted. Too large (N=100):
  Sigma is 100x100, slower convergence, heavier DQN baseline.
- **Recommended: N = 25 (5x5 grid), dense sub-lambda/2 spacing.** Primary/default for development and main
  figures. **Sweep N in {16, 25, 36, 49}** for the scalability/robustness figure.

### M — number of activated ports (RF chains this slot)
- **Need M >= K** to have enough spatial degrees of freedom to separate K streams (ZF/MMSE needs >= K).
- In v1 **observation is tied to activation** — the only ports we measure are the M we activate. So M is
  simultaneously "serving DoF" AND "observation budget per slot."
- **Set M modestly larger than K** so there is **sensing/array slack**: with M = K every activated port is
  fully committed to serving, leaving no room to spend a port on a promising-but-uncertain candidate. The
  exploration tension still exists at M = K (choose *which* K to activate), but M > K gives clean room for
  active sensing plus array gain from the extra ports. The M − K "extra" ports are not idle — they add
  array gain and get observed.
- Larger M -> better precoding + faster belief refinement, but more hardware and a **less "partial"** story
  (higher M/N). Smaller M -> strongest partial-CSI story but no sensing slack.
- **Recommended: M = 5** (with K = 3 -> **2 DoF of sensing/array slack**, observation budget **M/N = 20%**).
  **Sweep M in {3, 4, 5, 6, 7}** for the rate-vs-observation-budget figure.

### Summary recommendation for v1
```
K = 3 users,  N = 25 ports (5x5, sub-lambda/2),  M = 5 activated
-> observation budget M/N = 20%,  sensing slack M - K = 2,  N >= M >= K satisfied
```
These match the existing `RESEARCH_PLAN.md` defaults (consistency), are the sweet spot for our v1 story,
and stay small enough that the DQN baseline is reproducible.

---

## 4. Full default parameter table

| Symbol | Meaning | Default (v1) | Notes / sweep |
|---|---|---|---|
| `K` | users / streams | 3 | 2 for debugging |
| `N` | candidate ports (belief dim) | 25 (5x5) | sweep {16,25,36,49} |
| `M` | activated ports / slot | 5 | sweep {3,4,5,6,7}; M >= K |
| `M/N` | observation budget | 20% | the headline fraction |
| aperture / spacing | port geometry -> sets `R` | sub-lambda/2 (dense) | denser = stronger neighbor correlation; pin exact value from Paper 3 |
| `beta_k` | per-user channel power | 1 (equal) | later: unequal for near/far users |
| `rho` | temporal corr `= J0(2 pi f_D T_s)` | 0.9 | sweep {0.5,...,0.99}; low rho = fast aging = harder |
| `sigma^2` | data/receiver noise | 1e-3 | from Paper 3 |
| `sigma_e^2` | estimation/pilot noise on observed port | ~1e-2 | set by pilot SNR; sweep for robustness (Fig) |
| `e_sw` | per-port switching energy | 1 | from Paper 3 |
| `eta_sw` | switching-cost weight in `G(S)` | tune | balances rate vs churn |
| `alpha` | pragmatic weight in EFE | 1 | canonical |
| `beta`  | epistemic weight in EFE | 1 | **swept; beta=0 = exploitation-only ablation** (Fig D) |
| `T` | slots per episode | 100 | trajectory length |
| MC seeds | episodes to average | 200-1000 | for smooth curves |

---

## 5. Staged simulation steps (each with a verify-gate — do not advance until it passes)

- **Step 0 — Channel generator (ground truth, no agent).** Jakes `R` (Eq. 2) + AR(1) `rho` (Eq. 3) -> `h_k(t)`.
  **Verify:** empirical spatial cov ~= `beta_k R`; empirical temporal autocorr(lag) ~= `rho^lag`; samples proper
  complex Gaussian. *If this is wrong, everything downstream is wrong.*
- **Step 1 — Precoder + rate module.** Given true `h` and set `S`: ZF (then robust MMSE), SINR, rate.
  **Verify:** full-CSI ZF kills IUI; rate = `log2(1 + SNR)`; sane scaling with M.
- **Step 2 — Reference baselines.** Full-CSI genie (greedy/exhaustive), random, stale-CSI.
  **Verify:** genie >= everything; establishes the upper/lower band the AIF agent will live between.
- **Step 3 — Kalman belief alone (perception, no decisions).** Drive with a FIXED port set.
  **Verify (critical): belief calibration** — observed ports' `Sigma` shrinks to ~`sigma_e^2`; unobserved
  ports' `Sigma` inflates back toward `beta_k R` at the `rho`-governed rate (CSI aging, visualized);
  predicted error cov ~= empirical error.
- **Step 4 — EFE terms in isolation.** Pragmatic (expected rate from belief), epistemic (log-det info gain),
  switching. **Verify:** epistemic submodular and observing one port lowers *neighbors'* `Sigma` (via `R`);
  pragmatic ~= realized rate when belief is confident; all terms in bits.
- **Step 5 — Greedy selection.** Assemble `G(S)`, greedy add. **Verify:** on small `N=8, M=3`, greedy vs
  exhaustive — confirm near-optimality / `(1 - 1/e)`; record ms/slot (latency story).
- **Step 6 — Closed loop (full agent).** predict -> select -> act -> observe -> update over `T` slots.
  **Verify:** objective trends up toward genie; **beta = 0 ablation does worse** (sanity that the epistemic
  term earns its place).
- **Step 7 — Sweeps -> figures.** Observation budget (M), Doppler `rho`, noise `sigma_e`, sample efficiency
  (Figs A-D).
- **Step 8 — Objection-proofing.** Model mismatch + the **slow-timescale online parameter learning**
  (Fig E), latency (Fig F).

Priors/preference land in **Steps 0 & 3** (mu0 = 0, Sigma0 = beta_k R, fixed-param physics); the
slow-timescale learning is deferred to **Step 8** so it does not complicate getting the core loop correct.

---

## 5b. Progress log & findings

- **Step 0 DONE** (`sim/channel.py`, `sim/verify_step0.py`). Spatial cov ~= beta_k R (2% err),
  temporal autocorr ~= rho^L (<0.002 err), proper/circular, Gaussian. **Finding:** with dense
  0.25-lambda spacing `R` is strongly **rank-deficient** (min eig ~= 0, top eig ~= 4 of trace 25) ->
  the channel is compressible (few ports pin down many = our story) but `Sigma0 = beta_k R` is
  singular, so the Kalman filter (Step 3) needs mild regularization.
- **Step 1 DONE** (`sim/precoding.py`, `sim/verify_step1.py`). ZF nulls interference (leak 1e-15);
  MMSE >= ZF at all SNR (big at low SNR, ties at high SNR); K=1 matches P||h||^2/sigma^2; array gain
  monotone in M.
- **Step 2 DONE** (`sim/selection.py`, `sim/verify_step2.py`). Reference band at N=25,M=5,K=3:
  genie-greedy 31.2 > norm-topM 28.9 > random 27.8 bits/s/Hz (**7.4% headroom** for a smarter
  selector). Greedy = **95.5%** of exhaustive optimum (rate is NOT submodular -> no (1-1/e) here;
  that guarantee is for the epistemic term). CSI-aging: 1 stale slot at rho=0.9 crushes 28.6 -> 10.0.
- **Step 3 DONE** (`sim/belief.py`, `sim/verify_step3.py`). Complex per-user Kalman bank over
  N ports; predict = aging, update = Joseph form. All 4 gates pass: (A) predict reproduces
  the aging law `Sigma_t = rho^2t Sigma_0 + (1-rho^2t) beta R` to 2e-16; (B) one update drives
  observed-port var to ~0.0099 < sigma_e^2=1e-2; (C) **calibration gate** — steady-state filter
  Sigma matches empirical Cov(h-mu) to <4% rel-Fro (4000 MC); (D) observing port 12 drops a
  correlated neighbour (|R|=0.47) by 0.22 vs a far port (|R|=0.10) by 0.01 -> info flows via R.
  **Finding:** the singular `Sigma0=beta R` needs NO artificial jitter — the update only inverts the
  M×M innovation cov (regularized by sigma_e^2 I), never Sigma; Joseph form keeps Sigma PSD.
  `reg` knob exposed but defaults 0. Plot `step3_belief_check.png` shows sharpen-then-age + calibration.
- **Step 4 DONE** (`sim/efe.py`, `sim/verify_step4.py`). The three EFE terms in isolation, all in
  BITS. Pragmatic = robust-MMSE expected sum-rate with imperfect-CSI lower bound (CSI-error term
  `sum_j w_j^H Cov_k w_j`); epistemic = `sum_k log2 det(I + Cov_k/sigma_e^2)` (M×M form, finite even
  for singular Sigma); switching = `eta_sw e_sw |S XOR S_prev|`. All 8 gates pass: (E1) pragmatic ==
  full-CSI MMSE rate at Sigma->0 to 0e0; (E1b) pragmatic decreases with uncertainty; (E2) epistemic ==
  entropy drop (logdet reduction)/ln2 to 2e-14; (E3) monotone & (E4) submodular over 400 random
  A⊆B trials; (E5) info sharing via R — conditioning on correlated n0 cuts a neighbour's marginal
  info (drop 1.07) far more than a far port's (0.04); (E6) switching = symdiff count. Plot
  `step4_efe_terms_check.png`. **Note:** single-port epistemic is identical across ports (diag Sigma =
  beta_k), so the R-driven selectivity lives entirely in the *conditional/marginal* gains — greedy (Step 5)
  is what exploits it.
- **Step 5 DONE** (`efe.greedy_select` / `efe.exhaustive_select`, `sim/verify_step5.py`). Greedy builds
  S by max marginal J = alpha*prag + beta*epis - switch, O(N*M) evals. All 6 gates pass: (G1) mechanics;
  (G2) epistemic-only greedy = **99.15%** of exhaustive opt (>> 1-1/e floor); (G3) combined greedy = exhaustive
  to **0% gap** over 24 N=8,M=3 cases (machinery validated by G2<100%); (G4) switching modular reconstruction
  exact; (G5) latency **~20 ms/slot** at N=25,M=5 vs projected **~9.4 s** exhaustive = **~477x** speed-up;
  (G6) alpha/beta knob flips exploit vs explore sets 6/6. Plot `step5_greedy_check.png`. **Caveat:** absolute
  ms/slot is wall-clock/load-sensitive (saw 20-100ms across runs) — robust claims are O(N*M) scaling +
  ~500x vs exhaustive; re-benchmark cleanly for the paper. Pragmatic MMSE re-solve dominates cost
  (warm-start/rank-1 approx is the future optimization noted in EFE_DESIGN Sec.5).
- **Step 6 DONE** (`sim/agent.py`, `sim/verify_step6.py`). Full closed loop predict->select->precode-from-
  belief->transmit->observe->update. `AIFAgent` + `run_aif`/`run_genie`/`run_random` share a channel
  trajectory for paired MC. **Operating point: 15 dB (sigma2=0.03), beta_w=0.5, eta_sw=1.0** (30 dB collapses
  precoding on predicted CSI = Step-2 interference-limited finding; beta_w=1 over-explores, 0 never explores).
  All 5 gates pass: (C1) stable, belief PSD; (C2) **closed-loop calibration** served-port post-var 0.245 ~=
  realized err 0.242 (1.3%); (C3) learning curve 0->8.4 plateau in ~2 slots; (C4) **epistemic earns its place**
  AIF(beta=.5) rate 8.39 vs beta=0 6.81 (+23%), objective 7.09 vs 6.85; (C5) genie rate 16.5 (AIF=51%), but on
  switching-aware objective genie 10.6 vs AIF 7.1 (67%) since AIF churns 1.3 sw/slot vs genie's 6.
  Plot `step6_closed_loop_check.png`. **Cold-start fix:** mu=0 at t=0 made MMSE normalization divide-by-zero;
  guarded `mmse_precoder` (zero estimate -> zero precoder, rate 0) + `greedy_select` (NaN-safe, fallback pick).
- **KEY RESEARCH FINDINGS from Step 6 (shape S7):**
  1. **Predict-then-act protocol** (precode on PREDICTED belief, per RESEARCH_PLAN Sec.5) caps AIF rate at ~51%
     of genie because served-port CSI carries aging error ((1-rho^2)beta ~ 0.19 at rho=0.9). If we instead
     observe-then-precode (pilots first, same slot), served-port error ~ sigma_e^2=0.01 -> rate near genie.
     This ordering is a MAJOR design lever to expose/ablate in S7 (it decides whether the headline is
     "matches genie" or "graceful partial-CSI tradeoff").
  2. **Selection headroom is small in dense-correlated regime** (Step-2 7.4%), so exploration's benefit on
     *rate* is real but modest (+23% here comes mostly from beta=0 getting stuck on cold-start ports). AIF's
     bigger structural win is **switching-awareness** (1.3 vs genie 6 sw/slot). S7 should also test regimes
     where active sensing pays more (sparser/directional channels, unequal port SNR, larger aperture).
  3. **Fair baselines needed for S7:** current run_genie/run_random use FULL-CSI precoding -> too strong as
     "lower" bounds. Need a partial-CSI naive baseline (raw noisy obs, no Kalman) to isolate AIF's value.
- **Findings that shape later steps:**
  1. **Operating SNR is a design choice.** 30 dB (sigma^2=1e-3) makes ZF~=MMSE and stale CSI
     catastrophic (interference-limited). Run headline experiments at a **moderate SNR (~10-20 dB)**
     where robust-MMSE/belief quality is legible; also show high-SNR. Lock at Step 6/7.
  2. **Genie reference:** greedy-genie is a ~95% proxy; use **exhaustive** (feasible offline at
     N=25,M=5) for the true upper-bound curve in final figures.
  3. Rank-deficient `R` + high-SNR CSI floor **justify robust-MMSE** (uses Sigma) -- confirms
     `EFE_DESIGN.md`.

## 6. Open items to lock during implementation
- Exact aperture/spacing from Paper 3 (sets the strength of `R`'s off-diagonals).
- `sigma_e^2` calibration to a stated pilot SNR.
- `eta_sw` and the `alpha/beta` balance (default 1/1; `beta` swept).
- Number of MC seeds for smooth curves.
- (From `EFE_DESIGN.md` Sec. 8) unit reconciliation, pragmatic rate lower bound, pragmatic marginal MMSE
  re-solve cost.
