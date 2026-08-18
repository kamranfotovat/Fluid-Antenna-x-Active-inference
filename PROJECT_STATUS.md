# PROJECT STATUS — read this first (cross-device handoff)

**Purpose.** A single catch-up file so a fresh AI (or the user on another device) can `git pull` and
immediately understand where the project stands — especially on the **code**. For the full narrative and
design rationale, read the deeper docs listed at the bottom; this file is the map and the recent-work log.

**Last updated:** 2026-08-18.

---

## 1. What this project is (one paragraph)

Applying **Active Inference / the Expected-Free-Energy (EFE) principle** to **fluid-antenna systems (FAS)**.
The agent activates only a small subset of M ports out of N candidates, senses the channel *only* on those M
ports (partial CSI), infers the un-sensed ports through a generative spatial-correlation belief (a complex
Kalman filter over a Jakes correlation R + AR(1) time dynamics), and **selects ports by minimizing EFE** —
which unifies three terms: pragmatic (rate), epistemic (information gain), and switching cost. The pitch is
**active channel *acquisition* under partial CSI** (not "cost-aware selection," which DRL work already owns
under full CSI). We compete on sample-efficiency / partial-CSI robustness / zero-training, not peak throughput.

Authors: **Kian Fotovat** (Univ. of Tehran) + **Kamran Fotovat** (Iran Univ. of Science & Tech); collaborator
**Zijun Wang** (PhD, SUNY Buffalo) advising on beamforming. Repo is shared with Kamran (owner) — **never
force-push `main`**, forward commits only.

---

## 2. Repo map — where things live

```
RESEARCH_PLAN.md      what/why of the CURRENT letter (static-regime partial-CSI acquisition)
EFE_DESIGN.md         the EFE decision rule (the selection objective)
SIMULATION_PLAN.md    build plan, Steps 0–8, progress log in §5b (the authoritative numbers)
PAPER_OUTLINE.md      per-section bulleted claims for the draft
FUTURE_WORK.md        extensions beyond the letter — READ §0 (S1-vs-S2 map) and §6 (liquid-metal) first
PROJECT_STATUS.md     <-- this file (recent work + code map)
README.md             repo intro

sim/            ORIGINAL simulation (N=25, 5x5 sub-λ/2). The letter's locked results live here. UNTOUCHED.
sim_version2/   Aperture-scaled operating point (2x2 λ, N=441=21x21, M=10). Copy of sim + rescaled config.
sim_version3/   HYBRID BEAMFORMING build. Copy of sim_version2 + hybrid.py. THIS is where recent work is.
figures/        The 6 paper figures + diagnostics (from sim/make_paper_figures.py), all at ONE op point.
animation/      EFE port-selection GIFs across eta_sw / beta_w sweeps.
paper/          IEEE two-column .docx draft + make_paper_docx.py generator (no LaTeX on the machine).
meeting prep/   presentation script + meeting-questions docs.
main (3) (1).pdf   Zijun's 4-page hybrid-beamforming tutorial (reference; full-CSI AO cookbook).
```

**Copy-not-edit rule (standing):** each new direction is a NEW `sim_versionN/` folder copied from the
previous one; originals are never edited. So `sim/` and `sim_version2/` stay frozen; active work is in
`sim_version3/`.

---

## 3. The code — module by module (applies to sim / sim_version2 / sim_version3)

All three folders share the same core modules (version2/3 differ only in `config.py` values and, for v3, the
added `hybrid.py` + hybrid plumbing in `agent.py`):

| File | What it does |
|---|---|
| `config.py` | `OperatingPoint` dataclass + named op points. v3 adds `n_rf` field (None = fully digital). `ACTIVE` picks the live op point. |
| `channel.py` | `ChannelSimulator` — Jakes spatial R over the aperture + AR(1) time dynamics. Also `MovingHotspotSimulator` (drifting power envelope, for the deferred dynamic-tracking study). |
| `belief.py` | Complex Kalman filter q(h)=CN(μ,Σ) over ports. Joseph-form update; σ_e²I regularizes the innovation (Σ never inverted, no jitter needed). |
| `efe.py` | The EFE terms: pragmatic = robust-MMSE rate w/ CSI-error term (`robust_mmse_from_belief`), epistemic = Σ logdet(I+Cov/σ_e²), switching. Plus `greedy_select`. |
| `precoding.py` | MMSE / ZF precoders (returns M×K). |
| `selection.py` | Selection helpers / exhaustive reference. |
| `agent.py` | `AIFAgent` (the closed loop) + `run_aif` / `run_genie` / `run_naive` / `run_random_partial` + (v3) the hybrid sweep helpers. |
| `learning.py` | R-estimator (learns the correlation kernel when the channel is non-Jakes). |
| `bandit_baseline.py`, `drl_baseline.py` | Model-free competitors (both beaten by the model-based AIF agent). |
| `verify_step*.py` | Per-step smoke tests / gates (Steps 0–12). |
| `make_*_figures.py` / `make_*_tables.py` | Figure and table generators. |

---

## 4. RECENT WORK (this is the "what were we doing" part)

### 4a. Aperture scale-up → `sim_version2/`
Rescaled the array to a **2λ × 2λ** aperture, **N=441 (21×21)** candidate ports, **M=10** active, K=3. This
gives genuinely more spatial DoF (more distinct "good" ports) so selection matters more. Results in
`sim_version2/results_v2.md`. This is the geometry the hybrid work builds on.

### 4b. HYBRID BEAMFORMING → `sim_version3/` (the main recent effort)
**Goal:** make the transmit precoder hardware-feasible with fewer RF chains than active ports, WITHOUT
touching the active-inference loop.

**Design decision (important):** hybrid beamforming is applied as a **hardware-feasible projection of the
belief-based digital precoder**. The AIF loop — belief, EFE selection, per-port pilot sensing, Kalman update —
is **100% unchanged**. Only the transmit precoder changes: our robust-MMSE precoder `F*` (from
`efe.robust_mmse_from_belief`) is factorized as `F* ≈ F_RF · W_BB`, where `F_RF` is a fully-connected
**unit-modulus** analog network (M_active × n_rf) and `W_BB` is the digital baseband part (n_rf × K).

**The new file — `sim_version3/hybrid.py`:**
  - `_update_FRF(F_star, F_RF, W_BB)` — one monotone **coordinate-descent** sweep: for each RF chain, a
    per-entry closed-form unit-modulus phase update on the residual. (This replaced a first-try bug: the naive
    `F_RF = exp(j·angle(F* · pinv(W_BB)))` is degenerate when n_rf > K — rate went DOWN with n_rf. Fixed.)
  - `factorize_hybrid(F_star, n_rf, ...)` — warm-start from F*'s phases, then alternate LS for `W_BB` and
    `_update_FRF`; returns power-normalized `(F_RF, W_BB, W_eff)`.
  - `hybridize(F_star, n_rf, ...)` — convenience wrapper; `n_rf=None` returns F* unchanged (digital).
  - `factorization_loss_db(...)` — diagnostic (relative error with the optimal complex scalar removed).

**Plumbing in `sim_version3/agent.py`:** `AIFAgent` gained `n_rf`; `precoder()` now returns
`hybridize(robust_mmse_from_belief(...), n_rf)`. `run_genie` / `run_naive` also hybridized (fair comparison).
Added **sweep helpers** `run_aif_sweep` / `run_genie_sweep` / `run_naive_sweep` that run selection ONCE and
score all n_rf from it (~6× faster, since selection is n_rf-independent).

**Config:** `OP_V3` = OP_V2 geometry with M=10 active and the new `n_rf` field; `ACTIVE = OP_V3`.

**Verify / tables:** `verify_hybrid.py` (smoke test: factorization loss vs n_rf + closed-loop digital vs
hybrid); `make_hybrid_tables.py` → `results_v3.md` (H1 RF-chain sweep, H2 joint M×n_rf budget; 6 seeds, T=40).

**RESULTS (`sim_version3/results_v3.md`, 6 seeds, T=40):**
  - AIF **digital** rate 18.63 bits/slot. Hybrid at **n_rf=4 → 18.58 (100%)**; n_rf=3(=K) → 16.83 (90%, the
    only real penalty); n_rf≥5 → full.
  - The near-lossless threshold is **n_rf=4**, *below* the worst-case 2K=6 bound — our rank-K=3 precoder is
    easier to factor than a random target. Factorization is exact (−76 dB) at n_rf≥2K, machine-exact at n_rf=M.
  - AIF beats naive by ~2 bits/slot at **every** n_rf → hybrid hardware does NOT erode the AIF gain (orthogonal).
  - **HEADLINE: drive 10 active ports with only 4 RF chains at ~zero rate cost (60% fewer RF chains).** Two
    decoupled budgets: M active ports raises the rate ceiling; n_rf only needs ~4 to reach it — a framing the
    full-CSI tutorial cannot make.
  - Anchor citations: Sohrabi & Yu 2016; Zhang/Molisch/Kung 2005 (n_rf ≥ 2K reproduces any digital precoder).

**Sensing model chosen = S1** (separate pilot/data phase, per-port reads, Kalman unchanged). The more
ambitious **S2** (sense *through* the analog net, y = F_RF^H h) is scoped in `FUTURE_WORK.md §0`, not built.

---

## 5. Current decisions / open threads (see FUTURE_WORK.md for detail)

- **S1 vs S2 scope** (`FUTURE_WORK.md §0`): S1 = "sense-per-port, transmit-hybrid" = THIS paper (done).
  S2 = "sense-through-the-analog-network" = next paper; has **three tiers** (Light ~1–2 wk / Medium ~1 mo /
  Full = a paper). Ship S1; make S2 (ideally its continuous movable-antenna limit) the flagship follow-up.
- **Pixel vs liquid-metal hardware** (`FUTURE_WORK.md §6`): current paper stays **pixel/RF-switching**
  (activation = a set, no "which antenna goes where" question). Liquid-metal / movable FAS = a future paper
  where the switching cost becomes an **optimal-transport / assignment** cost (Hungarian algorithm, exact
  O(M³); collision-aware planning is the hard, novel version). Bridges to the continuous movable-antenna limit.
- **Notation clash to settle with Zijun:** his tutorial uses M=candidate / N=active ports; our code uses
  N=candidate / M=active. Agree on ONE convention for the manuscript.
- Draft prose iteration with Zijun; verify all citations flagged `[VERIFY]`.

---

## 6. Deeper reading (in priority order for a fresh catch-up)
1. `FUTURE_WORK.md` — §0 (S1/S2 map + tiers), §6 (liquid-metal/transport), §1–§3 (dynamic tracking, scaling, GP belief).
2. `SIMULATION_PLAN.md` §5b — the authoritative locked numbers and per-step progress log.
3. `RESEARCH_PLAN.md` / `EFE_DESIGN.md` — the framing and the exact selection objective.
4. `sim_version3/hybrid.py` + `results_v3.md` — the most recent code and its results.
