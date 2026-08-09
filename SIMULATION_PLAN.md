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
- **Step 7a DONE** (`agent.run_aif(sense_first=...)`, `sim/verify_step7_protocol.py`). Added the intra-slot
  protocol switch (finding #1 below) + verified Doppler robustness. `sense_first=True` (observe-then-precode):
  pilots on activated ports -> Kalman update -> THEN precode from fresh belief. All 3 gates pass. **Headline
  numbers (15dB, 20% obs budget):** observe-then-precode reaches **79-89% of genie** and is **flat across
  rho** (0.95->0.6), while predict-then-act falls **64%->25%**; rate ratio up to **3.17x** at rho=0.6; served
  CSI fresh (~0.009 = sigma_e^2) and calibrated. Figure `step7_protocol_doppler.png` = strong Fig C candidate
  (this is the "matches genie while observing 20% of ports, robust to Doppler" story). **observe-then-precode
  is now the DEFAULT headline protocol; predict-then-act is the ablation.** Finding #1 below is RESOLVED.
- **Step 7b DONE — hyperparam tuning + fair baselines** (`agent.run_naive`, `agent.run_random_partial`,
  `sim/verify_step7_baselines.py`). **Tuning: optimal beta_w = 0.1-0.25** (NOT 0.5 — that was for predict-then-act);
  a little curiosity lifts rate +13% with ZERO extra switching (explores early, locks). **Locked op point:
  beta_w=0.25.** Fair partial-CSI competitors (same 20% budget, observe-then-precode): genie(full CSI) obj 10.6
  rate 16.5 sw 6.0 | **AIF obj 13.2 rate 13.3 sw 0.0** | naive(no inference) obj 9.8 rate 12.9 sw 3.1 | random
  obj 4.5 rate 12.4 sw 8.0. All 4 gates pass. **THE STORY:** on RATE, AIF~=naive (selection headroom small,
  both get fresh pilots) at 80% of genie; on the switching-aware OBJECTIVE (Eq.7, the real metric — same one
  the switching-cost FAS papers care about) **AIF beats naive +35% AND beats the genie**, because EFE unifies
  rate+info-switching -> AIF finds a good set and STOPS moving (0 vs naive 3 vs genie 6 sw/slot). AIF's decisive
  edge is STABILITY/switching-awareness, not raw selection. Plot `step7_baselines.png`.
- **Step 7c: LEARNING question ANSWERED (model-mismatch probe, MC=12).** True channel rho=0.9/Jakes R; vary the
  agent's ASSUMED params: correct 13.19 | wrong rho=0.5 -> 13.17 (~no harm!) | wrong rho=0.99 -> 12.15 | wrong R
  (uncorrelated I) -> 11.70 | wrong beta -> 13.20. **Conclusions:** (1) with correct params we're at the Bayesian
  OPTIMUM — learning can only tie (confirms AIF's zero-training selling point); (2) observe-then-precode makes us
  nearly IMMUNE to wrong rho/beta (served ports re-measured each slot) — big robustness win for free; (3) **only R
  (spatial structure) matters** — wrong R costs ~1.5. So the slow-loop's targeted job = learn R for the
  non-Jakes/model-mismatch case (Fig E), NOT a headline booster.
- **Step 8 DONE — slow-loop learning of R + Fig E** (`sim/learning.py`, `sim/verify_step8_learning.py`).
  `SpatialCorrEstimator`: accumulates y_i conj(y_j) over co-observed pairs, subtracts sigma_e^2 from the
  diagonal, normalizes to unit-diagonal correlation (cancels per-user beta), PSD-projects -> R_hat from
  PARTIAL noisy obs. All 3 gates pass: (L1) coverage->1, R_hat err falls with warm-up (0.74@30 -> 0.43@200);
  (L2) on Jakes truth, learned recovers **100%** of the wrong-R(=I) gap (oracle 13.15 = learned 13.15 > wrong
  11.67); (L3 = **Fig E**) with a genuinely non-Jakes (exponential) true channel, an agent that assumes Jakes
  gets 12.83 while **learned R_hat matches the oracle (~13.2)**. Even a rough R_hat (43% Fro err) suffices —
  capturing WHICH ports correlate matters, not exact values. Plot `step8_learning_mismatch.png` (3 R heatmaps:
  true/learned/assumed-Jakes; Jakes shows false negative Bessel bands the true channel lacks). **This closes the
  "learning" question: not a headline booster (matched case already optimal = zero-training strength), but a
  working robustness tool that adapts to real (non-Jakes) propagation.**
- **Step 9 — MOVING-HOTSPOT scenario (dynamic tracking): HONEST NEGATIVE RESULT** (`channel.MovingHotspotSimulator`,
  `agent.run_fixed`). Built a channel with a Gaussian power sweet-spot that DRIFTS (circles the aperture every
  ~40 slots) so good ports move & a fixed set decays. Purpose: make switching non-trivial & show active tracking.
  **Findings:** (1) switching IS now necessary — fixed-hold decays (rate 8.7), genie churns ~5 sw/slot to track
  (rate 12.5). This CONFIRMS the earlier 0-switching was regime-specific & CORRECT for the static-power world,
  not a bug. (2) BUT our current AIF does NOT track well: at η_sw=1/β_w=0.25 it locks (0 sw, rate 8.9 ≈ fixed
  8.8) and is BEATEN by naive round-robin (9.6); cranking β_w just makes it thrash (9-10 sw) with LOWER rate.
  Even on an easy big/slow hotspot AIF only marginally beats naive (12.2 vs 11.9) at 7.7 sw. **ROOT CAUSE:** the
  generative model assumes STATIONARY equal-power (Jakes+AR1) fading — it's blind to the moving power envelope,
  so it attributes hotspot power to random fading (decays via ρ) & can only REACT by blind sensing, no better
  than round-robin. To win at tracking, the belief must MODEL the smooth moving power structure (a latent
  hotspot-location state / per-port power that drifts) so it can PREDICT where the hotspot goes & sense ahead.
  **DECISION POINT (asked user):** (A) extend generative model with a moving-power-envelope belief = proper AIF
  tracking, strong contribution but real work / journal-scale; (B) keep the static-regime story as the letter
  (84% genie, wins switching-aware objective, 0-switching = correct minimal-movement, zero-training, robust,
  learns R) & cite dynamic tracking as future work. Infrastructure committed for whichever path.
  **DECISION (2026-08-10): PATH B chosen** — ship the static-regime letter now. Path A (moving-envelope tracking
  belief) AND the scaling-up study (N=64/100+) are deferred and specified in `FUTURE_WORK.md` (separate file so
  THIS plan stays the current-paper spec). Moving-hotspot infra (MovingHotspotSimulator, run_fixed) stays in repo.
- **Step 10 DONE — DRL baseline comparison** (`sim/drl_baseline.py` Transformer port-selector trained by
  policy gradient on the Eq.7 objective; `sim/verify_step10_drl.py`). Answers "did we compare to literature
  methods?" — yes now (Paper-3-style DRL). All 3 gates pass (MC=20, GPU, ~5min train). **Results (objective,
  eta_sw=1):** genie(full CSI) 10.4 | DRL full-CSI 12.96 (locked, 0 sw) | DRL partial 12.03 | **AIF partial
  13.54** | naive 10.5. **(L1) Sample efficiency (Fig B):** DRL climbs 10.6→13.0 over ~100-200 iters; AIF flat
  13.54 from iter 0 (zero training), above DRL at every stage. **(L2) No full-CSI advantage:** AIF partial (20%
  CSI, no training) 13.54 > DRL full-CSI+full-training 12.96. **(L3) Competence gate:** DRL(eta=0) hits 85% of
  genie RATE → a fair baseline, not a straw man (85% is the top-M-scoring ceiling; can't capture full
  combinatorial interference like greedy-on-true-CSI). Figure `figB_drl_sample_efficiency.png`. **Honest note:**
  under observe-then-precode selection headroom is small so DRL/naive/AIF rates are close; AIF's edges are
  zero-training + switching-awareness + partial-CSI robustness, NOT raw selection. torch 2.7+CUDA available.
- **Step 11 — PARETO FRONTIER (best-results sweep)** (`sim/make_frontier_figure.py`, `figF_pareto_frontier.png`).
  User pushed for the best operating point. Fine beta_w sweep (MC=15-20, sigma_e^2=1e-3) shows AIF traces a
  frontier that DOMINATES the genie: **max-objective beta_w~0.3 (rate 13.96=84%, obj 13.80, ~0 sw)**;
  **max-rate beta_w~0.6 (rate 14.80=89%, obj 12.30, 2.5 sw)**; genie (16.61 rate, 10.61 obj, 6 sw). The ENTIRE
  frontier (beta_w 0.1-0.7) beats the genie objective while switching << genie. **89% is the partial-obs rate
  ceiling at M=5** — beyond beta_w=0.6 rate flatlines & objective falls; higher needs bigger M (Fig A). So the
  headline is a frontier, not a point: "84% rate at ~0 switching up to 89% rate, all beating the switching-blind
  genie." Two reportable anchors depending on throughput-first vs efficiency-first. (Main single-point figures
  keep beta_w~0.25-0.3 = balanced/max-objective; figF shows the full range.)
- **Step 12 DONE — BANDIT baseline comparison** (`sim/bandit_baseline.py` combinatorial-UCB, model-free, cf.
  Zou et al. WCL'24; `sim/verify_step12_bandit.py`). Completes the "model-based AIF > model-free learning" story
  (DRL in S10, bandit here). Gave the bandit its fairest shot by sweeping exploration c. All 3 gates pass (MC=20):
  bandit objective by c: 12.23(c=0) / 12.08 / 11.74 / 10.89 / 9.59(c=1) — **AIF 13.69 beats the bandit at EVERY c
  (+12% over its best)**; AIF switch 0.00 vs bandit >=1.0 (bandit never locks: per-port power is ~equal so UCB
  explores perpetually); rates comparable (AIF 13.71 >= bandit 12.95, honest — observe-then-precode makes
  selection headroom small). **Root cause:** model-free bandit has no per-port persistent signal to learn in the
  equal-average-power FAS regime → pure exploration cost; AIF's model-based belief + switching-aware EFE locks
  immediately. Figure `figG_bandit_comparison.png`. Note: a structured/correlated bandit (using R) would be
  stronger = moving toward our model-based approach (see FUTURE_WORK §5).
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
