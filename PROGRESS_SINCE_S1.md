# Progress Since S1 — Catch-Up for Another Device / AI

This documents everything explored **after Paper 1 (S1)** was in good shape, so a fresh session
(here or on another machine) can understand the current state and decisions without re-deriving them.
Read `PROJECT_STATUS.md` first for the original S1 project; this file continues from there.

Last updated: 2026-08-24.

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

### ⚠️ Second, sharper fact (added 2026-08-24) — read this before building anything on §5

The statement above is about **which protocol**. There is a second axis — **which model** — and it
turned out to matter more:

> **Learning pays in the TEMPORAL domain (Doppler / AR dynamics), NOT the spatial one (R).**

Wrong spatial R costs ≈ 0–0.6 in rate; wrong temporal model costs **+11.3 at full scale**. The whole
§5 negative result is real but it was aimed at the wrong parameter. §6 is the positive counterpart
and is now the main line of the post-S1 work. If you are picking this project up, **§6 supersedes
the "learning doesn't pay" mood of §5 and §0**.

---

## 1. Paper strategy (decided)

- **Paper 1 (S1, `sim_version3`)** — ~~EFE port selection under partial CSI~~ **RESCOPED 2026-08-24
  (Kian's decision): EFE port selection *and pilot allocation*.** See §9. Novelty pressure-tested
  and survives (§2); the rescope is about making it *stronger*, not repairing it.
- **Paper 2 (flagship, `sim_version5`)** — liquid-metal **column** FAS + AIF trajectory control. Under
  active construction (see §4). Myopic half built and gated; horizon planning layer 1 built.
- **Extensions** (preferences/QoS, per-port power, active R-learning) — slot into whichever paper;
  active R-learning was explored and shelved (see §5).
- Split into **2 papers**. The temporal-model work is **no longer a candidate third paper** — it ships
  inside Paper 1 as the enabler of the pilot reduction (§9). Paper 1 = protect from being too thin
  (now handled); Paper 2 = protect from being over-stuffed.

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

**Postscript (2026-08-24):** we later tested that "only home" directly. In partial sensing, *having*
a reasonable spatial model is essential (Jakes-R vs identity-R ≈ **+5 to +7**, roughly doubles rate),
but R **accuracy** — what learning actually buys over a good fixed prior — is worth only **+0.6**
(`verify_partial_mismatch.py`), and part of that is the same transient. So R-learning is dead in all
three regimes. A fixed Jakes prior suffices. **The lever is temporal, not spatial → §6.**

---

## 6. Partial sensing + the TEMPORAL model (`sim_version3`, TM-0..TM-5) — the positive result

This is where the post-S1 work actually landed. Two banked wins.

### 6.1 Partial sensing (`partial_sense.py`, `verify_partial.py`) — a real pilot-savings result

Serve M ports, pilot only `m_sense < M` of them fresh, precode all M from the belief (fresh on
piloted, R-inferred on the rest). Graceful curve, unlike predict-then-precode's cliff:
`m_sense=4 → 78%`, `6 → 87%`, `8 → 94%` of full-pilot rate. Spatial inference is essential here
(see §5 postscript). *(Gotcha fixed: `y` must be built in the SAME port order `bel.update` uses —
sort `S_sense`, or the port↔measurement association is scrambled.)*

### 6.2 Why predict-then-precode was bad, and the fix

Under AR(1) the effective CSI error when precoding from a *predicted* belief is `(1−ρ²)β ≈ 0.19β`
at ρ=0.9, vs `σ_e² = 0.001β` — **190× worse**. That aging floor is temporal; no amount of R-learning
touches it. Fix: make the channel **predictable**. Truth = band-limited **Jakes** sum-of-sinusoids
(`r(τ) = J₀(2π f_D T_s τ)`, normalized-Doppler knob); belief = **AR(p)** via Yule-Walker, space-time
separable (temporal AR(p) ⊗ spatial R).

- **TM-0/TM-1a** — AR(4) cuts 1-step prediction error **20×** with realistic pilot noise
  (0.186 → 0.009 at f_D T_s = 0.10). *(Noiseless figures overstate this — quote the noisy one.)*
- **TM-2** (`st_belief.py` `STKalmanBelief`, exact AR(p) space-time filter) — AR(1) ≡ the old model
  exactly (invariant). AR(4) helps **every** protocol, most in predict.

### 6.3 Learning the Doppler online — and the TM-3 story that TM-4 OVERTURNED

**Do not build on TM-3's explanation.** TM-3 got 79% recovery but via two hand-tuned constants
(`r_shrink=0.95`, `ev_inflate=3.0`) and a **wrong diagnosis** (claimed survivorship bias in the
autocorrelation estimate). TM-4 disproved it:

1. Not survivorship — a uniformly random, belief-independent probe port shows the *same* bias.
2. Not the generator — its realized ACF is 0.9023 vs J₀'s 0.9037.
3. It *is* a **ratio bias**: fed every port the estimator is exact, but under sparse sensing
   `r̂ = A/B` pools numerator and denominator over *different* sample sets, and with a
   time-correlated channel n_eff = number of coherent windows (~36), so Jensen inflates `E[A/B]` by
   ≈ r/n_eff ≈ 0.025. **Matched-window normalization** (normalize each lag by its own pairs' power)
   removes 93% of it.
4. **But de-biasing alone made rate WORSE** — so bias was never the mechanism.

**The real mechanism is ill-conditioned Yule-Walker.** At f_D T_s = 0.10 the Jakes ACF is smooth:
`cond(Γ) = 3e4`, `ev = 1.6e-4`, `|a|₁ = 13.1`. A 0.01 ACF error yields coefficients whose *actual*
1-step error variance is **17.2** while plain YW still reports 0.18 — **94× overconfident**. The
Kalman then trusts a stale prediction and the rate craters. (One seed crashed Levinson outright:
`Singular principal minor`.) *That* is what the two constants were secretly compensating for.

**Principled fix = data-driven AR ORDER SELECTION** (`temporal.ar_from_acf_robust`): bootstrap
`r ~ N(r̂, diag(se²))`, refit each order, score by the error it *actually* incurs; ridge `δ = Σse²`;
process noise = the posterior-predictive error. **Zero tuned constants** — everything keys off the
estimator's own Bartlett standard errors.

### 6.4 TM-5 — "information buys model order" is FALSE; the true story is better

Hypothesis failed: selected order went **2.67 → 2.00** as data grew (down, not up). Arithmetic:
order 4 only wins once `se(1) ≲ 0.005`, and `se ~ √((1+2r²)/n)` with `n ~ T·M·K` needs **T ≈ 7000
slots**. So **a learner can never afford AR(4) in this regime — AR(2) is the practical ceiling.**

That is fine, because **closed-loop rate saturates long before prediction error does** (σ_e² and
multiuser interference dominate the last decade): AR(2)-oracle = **92.9%** of AR(4)-oracle. The
learner reaches **97.8% of the AR(4) oracle using only q≈2**. What information buys is *coefficient
accuracy at a low safe order*; the selector's **refusal** to buy an unaffordable order is the safety
mechanism — **forcing p=4 craters at every horizon (2.2–6.5)**.

> **Consequence for the paper:** TM-2's headline AR(4) numbers are *oracle* results. Quote the
> learner-achievable figures, not the oracle ones.

### 6.5 Full scale (N=441) — enabled by an EXACT reduced-rank filter

`st_belief_lr.py` `STKalmanBeliefLR`: R at N=441 has numerical **rank 26**, and the channel lives
*exactly* in range(R), so `h = Bc` loses nothing — state **1764 → 104**, **~4880× cheaper**, 16
ms/slot. Verified bit-identical to the exact filter at full rank (`max|Δr| = 2e-11`) and matching to
6e-7 truncated (`verify_tm_lr.py`). The "separable approximation" deferred earlier is **not needed**.
*(Gotcha: greedy selection is DISCRETE — a 1e-7 belief difference flips a tie and changes the port
set, so compare MEANS over seeds, never trajectories.)*

**Results at OP_V2 (N=441, M=10, K=3), MC=3, T=40 — all gates pass** (`results_tm/fullscale.txt`):

| protocol | AR(1) | AR(4) | gain | % genie |
|---|---|---|---|---|
| observe-then-precode | 18.289 | 20.214 | +1.93 | 91.1% |
| predict-then-precode | 11.524 | 17.715 | **+6.19** | 79.8% |
| partial (m=4) | 13.951 | 14.931 | +0.98 | 67.3% |
| genie | — | 22.200 | | |

Learning from a **wrong** Doppler (assumed 0.05, true 0.10); gap = **+11.301**:

| arm | rate | recovery | order q |
|---|---|---|---|
| oracle | 17.715 | 100% | 4 |
| wrong-fixed | 6.414 | 0% | — |
| TM-3 naive | 3.295 | −28% | — |
| TM-3 tuned hedge | 10.826 | 39% | — |
| **PRINCIPLED (matched + order-sel)** | **15.713** | **82%** | 2.7 |

**The key cross-scale result: hand-tuned constants DO NOT TRANSFER.** TM-3's hedge scores 79% → 60%
at N=25 and collapses to **39%** at N=441; the principled estimator holds (87% → **82%**). Only
visible at full scale.

**Deployable headline (learner-achievable, not oracle):** starting from a wrong Doppler and learning
online, **15.713 vs the current AR(1) model's 11.524 = +36%**, at 71% of genie.

**MC=10 rerun** (`results_tm/fullscale_mc10.txt`): Part 1 confirmed — predict **+5.880** (was +6.19),
observe **+1.856** (was +1.93); both stable under 3× the seeds, so figure-grade. *Part 2 at MC=10 was
deliberately killed mid-run to save battery (not a crash — the file says so); rerun
`python verify_tm_fullscale.py 10 40` to finish it. The MC=3 Part 2 table stands in the meantime.*

### 6.6 Full-scale pilot-savings curve — and a refuted hypothesis worth keeping

`verify_tm_fullscale_partial.py`, MC=8, T=40 (`results_tm/fullscale_partial.txt`):

| m_sense | pilots | AR(1) | AR(4) | gain | % full-pilot | % genie |
|---|---|---|---|---|---|---|
| 2 | 20% | 8.865 | 9.581 | +0.716 | 47.5% | 43.3% |
| 4 | 40% | 14.038 | 14.701 | +0.663 | 72.9% | 66.5% |
| 6 | 60% | 16.070 | 18.477 | **+2.407** | 91.7% | 83.5% |
| 8 | 80% | 17.223 | 19.715 | **+2.492** | 97.8% | 89.1% |
| 10 | 100% | 18.298 | 20.159 | +1.861 | 100% | 91.2% |

The script's hypothesis — that the AR(4)−AR(1) gap **widens** as pilots are withdrawn, since more
served ports are carried by the belief — is **REFUTED**. It *narrows* (+0.716 at m=2 vs +1.861 at
m=10), and the gain **peaks at intermediate pilots** (m=6–8, ≈+2.5).

> **Why:** with 2 of 10 ports sensed, the dominant error is **spatial** inference of the 8 unsensed
> ports — the state is poorly known at every instant, so propagating it more accurately *in time*
> buys little. **The temporal model needs a reasonably-informed belief to be worth propagating.**

This also explains why the m=4 gain kept shrinking as seeds were added (+0.98 → +0.622 → +0.663):
m=4 sits in the low-information regime. **Practical consequence: m=4 was the worst point to showcase
partial sensing. Use m=6 — 83.5% of genie at 60% of the pilots, with a +2.4 temporal gain.**

---

## 7. Open decisions / next steps

1. **Paper 2 layer 2** (drift-aware MovingHotspot anticipation) — parked, "needs a lot of discussion."
   The one regime where horizon planning could genuinely beat myopic (pragmatic pre-positioning).
2. ~~**Predict-then-precode variant** — open whether to build it.~~ **DONE (§6).** Built, and it is
   no longer "much lower absolute rate": with the temporal model it reaches 17.7 vs observe's 20.2
   (80% of genie), and it is the regime where learning demonstrably pays.
3. Paper 1: optionally fold in a **preference-prior (QoS/fairness)** subsection; finalize & submit.
4. Notation agreement with Zijun (tutorial M=candidate/N=active vs our N=candidate/M=active).
5. ~~**Where the temporal-model work goes** — undecided.~~ **DECIDED 2026-08-24 → into Paper 1 (§9),**
   together with partial sensing, keeping the S1 hybrid transmit stage. The two cannot be split: at
   m=6 partial sensing alone loses ~10 points of genie and the temporal model is what buys it back.
6. **Before any figure:** MC=10 done for Part 1 and the m_sense sweep (§6.6); **Part 2 at MC=10 still
   owed** (`python verify_tm_fullscale.py 10 40`, ~30 min). Quote **m=6**, not m=4, for partial
   sensing (§6.6).

---

## 8. File map (added since S1)

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
  -- partial sensing (§6.1) --
  partial_sense.py                        run_aif_partial: pilot m_sense of M, infer the rest
  verify_partial.py                       pilot-savings curve + R-inference value
  verify_partial_mismatch.py              FAIR R-accuracy test (the +0.6 deflation)
  -- temporal model (§6.2-6.5) --
  temporal.py                             Jakes generator, Yule-Walker, TemporalACF (matched norm,
                                          Bartlett se, LCB), ar_from_acf_robust (ORDER SELECTION)
  st_belief.py                            STKalmanBelief (exact AR(p) space-time) + runners:
                                          run_st / run_st_learn / run_st_learn_probe
  st_belief_lr.py                         EXACT reduced-rank ST filter -> full N=441 (§6.5)
  verify_tm_step0/1/2.py                  AR(p) predicts Jakes; noisy 20x; closed-loop by protocol
  verify_tm_step3_premise.py, step3.py    wrong-Doppler cost; TM-3 tuned-hedge learning (SUPERSEDED)
  verify_tm_step4.py                      TM-4 ablation: matched norm + order selection (87%)
  verify_tm_step5.py                      TM-5 order/horizon study (AR(2) is the ceiling)
  verify_tm_lr.py                         reduced-rank filter gate (exactness + speed)
  verify_tm_fullscale.py                  full-scale N=441 TM-2 + TM-3/TM-4
  verify_tm_fullscale_partial.py          full-scale m_sense sweep (pilot-savings curve)
results_tm/                              <- temporal-model results (new folder convention)
```

Memory files (auto-memory, not in repo): `epistemic-ablation-result`, `v5-column-protocol-finding`,
`active-learning-S1` capture the same findings for the assistant's recall.

---

## 9. Paper 1 RESCOPE (decided 2026-08-24) — "port selection + pilot allocation"

**Decision (Kian):** fold partial sensing and the AR(p) temporal model into Paper 1, and **keep the
S1 hybrid transmit stage**. The temporal-model work is no longer a separate paper.

### Why — the rebranding exposure this closes

The real reviewer risk was never "active inference vs Bayesian inference." It is this: under
**observe-then-precode**, the belief only chooses *where to look*; the rate comes from fresh CSI. A
reviewer can then say this is **uncertainty-aware sensor selection** (Krause & Guestrin 2008) with
new vocabulary. The existing defence — the pragmatic term ties selection to the *communication*
objective, and the switching cost makes it sequential — is true but sounds incremental.

**Partial sensing removes the exposure structurally.** Once only `m_sense < M` ports are piloted,
the belief's covariance enters the *achieved rate* through the precoder on the un-piloted ports. The
belief stops being a selection heuristic and becomes load-bearing. It also lands on the field's
actual pain point — 3 of the 7 related papers (§3) are about FAS channel-estimation overhead.

### The headline claim

| config | pilots | rate | % genie |
|---|---|---|---|
| m=10, AR(1) — *current* Paper 1 | 100% | 18.337 | 82.9% |
| **m=6, AR(4)** — *proposed* | **60%** | **18.477** | 83.5% |

**Equal rate on 40% fewer pilots.** Partial sensing alone at m=6 *loses* ~10 points of genie
(82.9% → 72.7%); **the temporal model is what buys it back.** That is why the two must ship
together — splitting them across papers weakens both.

### Hardware story — get the two 6's right

They are a **coincidence**, and conflating them is a claim a reviewer can break:
- **n_rf = 6** on transmit because **6 = 2K** (K=3). Fully-connected infinite-resolution phase
  shifters represent *any* digital precoder exactly at n_rf ≥ 2K (Sohrabi & Yu 2016; Zhang, Molisch
  & Kung 2005). Verified free here — measured **bit-identical** to digital.
- **m_sense = 6** because that is where the pilot-savings curve knees (§6.6).

At K=4 transmit would need n_rf=8 while m_sense would stay ~6. In TDD the same chains serve both, so
`max(6, 6) = 6` — but say *why* each 6 arises, separately.

**Sensing front-end assumption (must be stated in the paper).** The `m_sense` piloted ports are
modelled as clean **per-port** observations (a selection matrix `P_S`). That presumes a
**switch-based** sensing front-end — standard for FAS, where port switching is the premise. A
fully-connected unit-modulus phase-shifter network could *not* do this on receive: it cannot null
the unselected ports, so each chain would see a *combination*. The hybrid network here is
**transmit-only**, where it is post-processing of the precoder. Generalising the observation model
from `P_S` to an arbitrary analog combiner `A` (≈3 lines in `STKalmanBelief.update`) would be a
stronger, more novel result — **deliberately not done**; closed by assumption instead.

### What this costs

This un-finishes a paper that was "essentially done": new figures, new baselines, new text, and the
temporal model has no write-up yet. Weeks, not days. The conservative alternative — ship Paper 1
as-is and make partial+temporal a second paper — remains defensible; it just leaves Paper 1
answering the sensor-selection objection with an argument rather than a result.

### Status

`verify_paper1_config.py` runs the actual paper configuration (partial sensing × AR order × **hybrid
n_rf=6**) — the first time partial sensing and hybrid have been simulated together. Smoke test:
hybrid is **exactly** digital at n_rf=2K. → `results_tm/paper1_config.txt`.
