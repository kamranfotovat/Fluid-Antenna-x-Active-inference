# Research Plan — Active Inference for Port Selection in Fluid Antenna Systems under Partial CSI

**Status:** planning · **Home field:** wireless (AIF = enabling method) · **Anchor:** Paper 3 (Switching-Cost-Aware DRL, Comm. Letters) · **Target venue:** IEEE Comm. Letters / ICC / Globecom (letter/conference scope, ~5–6 pages)

---

## 1. Thesis / headline

> You don't need full CSI. An active-inference agent probes only the ports it activates, infers the rest from the known spatio-temporal channel correlation (Jakes `R` + AR(1) `ρ`), and selects ports by minimizing expected free energy — jointly maximizing rate, resolving channel uncertainty, and minimizing switching cost. It matches a full-CSI genie while observing a fraction of ports, and is more sample-efficient and robust than DRL baselines that assume full CSI.

**Why it wins (the axis choice):** we do NOT compete on peak throughput under full CSI (we'd tie/lose there). We compete on **partial CSI, sample efficiency, robustness, latency** — the realistic axis where AIF's belief + active sensing structurally dominate.

## 2. Contributions (paper bullets)

1. **First formulation of FAS port selection as partial-observability active inference** — removes the unrealistic full-CSI assumption baked into prior work (Paper 3 computes `‖h_n‖²` for *all* N ports).
2. **Model-based agent** that folds spatio-temporal channel correlation (Eq. 2 spatial `R`, Eq. 3 temporal `ρ`) into a **Kalman belief** over all N ports, and selects ports by minimizing **expected free energy (EFE) = expected rate + information gain − switching cost**.
3. **Greedy submodular EFE selection** with a **(1 − 1/e) near-optimality guarantee** → low decision latency (pre-empts the latency objection).
4. **Simulations:** matches full-CSI upper bound while observing only a fraction of ports; far more sample-efficient and robust to Doppler/model shift than DRL baselines.

## 3. System model (inherit from Paper 3, add partial observability)

Keep identical so the comparison is fair:
- 2D FAS at BS: `N = Nx×Ny` candidate ports over `Wxλ×Wyλ`. **Default N = 25 (5×5).**
- Activate **M of N** ports/slot to serve **K** single-antenna users. `N ≥ M ≥ K`. **Default M = 5, K = 3.**
- Channel `h_k(t) ∈ C^N` over all ports. Spatial correlation `R` (Bessel, Eq. 2). Temporal AR(1): `h_k(t) = ρ h_k(t−1) + √(1−ρ²) e_k(t)`, `e_k(t) ∼ CN(0, β_k R)` (Eq. 3), `ρ = J₀(2π f_D T_s)`.
- ZF precoding → `R_k(t) = log₂(1 + SNR_k(t))`, `SNR_k = |h_k^H w_k|²/σ²` (Eq. 5).
- Switching cost: `C_t = |S_t △ S_{t−1}|`, `E_sw = e_sw·C_t`. Objective (Eq. 7): `max Σ_t [ Σ_k R_k(t) − η_sw E_sw(t) ]`.
- Sim params to reuse: `σ² = 1e−3`, `e_sw = 1`, `T = 100` slots.

**THE KEY DEPARTURE — partial observability:**
- Full CSI is **not** available. Each slot the agent obtains **noisy channel estimates only for the M activated ports**: `y_k(t) = S(a_t) h_k(t) + n`, `n ∼ CN(0, σ_e² I)`. The other N−M ports are **hidden**.
- Observation is tied to activation → to *learn* a port you must *spend a serving slot* on it. This is what creates the explore-vs-exploit tension AIF resolves.
- **CSI aging** emerges for free: a port measured at slot t decays in belief-confidence at rate `ρ` per slot (Eq. 3 predict step). The agent must decide when to **re-activate/refresh** stale ports (in version 1 the only way to re-measure a port is to activate it).

## 4. Active-inference formulation

**Generative model (linear-Gaussian → tractable):**
- Hidden state: per-user channel `h_k(t) ∈ C^N`.
- Transition `B`: AR(1) with `ρ`, process noise cov `(1−ρ²)β_k R` — **given by physics.**
- Likelihood `A`: `y_k = S(a) h_k + noise` — partial linear observation of activated ports.
- Preferences `C`: prefer high-rate outcomes; penalize switching.

**Perception = Kalman filter** (complex/circular Gaussian). Maintain mean `μ_k(t)` and covariance `Σ_k(t)` for each user's N-dim channel.
- *Predict:* `μ ← ρμ`, `Σ ← ρ²Σ + (1−ρ²)β_k R` (this inflates uncertainty on unobserved ports → CSI aging).
- *Update:* standard Kalman update using the M new observations; spatial `R` propagates info to neighbors.

**Action selection = minimize EFE.** For a candidate port set `S`:
```
G(S) = − PragmaticValue(S) − EpistemicValue(S) + SwitchingCost(S)
```
- **Pragmatic:** expected sum-rate under current belief, `E[ Σ_k log₂(1+SNR_k) | μ, Σ, S ]` (use belief mean/variance; a rate lower bound is fine).
- **Epistemic:** expected information gain = belief-entropy reduction from observing `S` = `½(log|Σ_prior| − log|Σ_post(S)|)` (closed form for Gaussians, log-det).
- **Switching:** `η_sw e_sw |S △ S_{t−1}|`.
- Exploration/exploitation is unified in one objective — the epistemic term is principled, not an ε-greedy hack. Expose a weight on epistemic value as an ablation knob.

**Tractability — greedy submodular selection.** `choose M of N` = C(25,5) ≈ 53k; don't enumerate. Build `S_t` greedily: start empty, repeatedly add the port with max marginal EFE until `|S|=M`. Info gain is submodular ⇒ greedy gives **(1 − 1/e)** guarantee and runs in `O(N·M)` → low latency + a theory nugget.

## 5. Algorithm (per slot)
1. **Predict** belief (μ, Σ) via AR(1) `ρ` + process noise `R` (CSI aging).
2. **Greedily select** `S_t` maximizing marginal EFE (pragmatic + epistemic − switching).
3. **Activate** `S_t`, apply ZF using belief mean, transmit.
4. **Observe** noisy channel at `S_t`.
5. **Update** belief (Kalman).
6. **Log** realized rate, switching cost.

## 6. Baselines
1. **Full-CSI genie** + exhaustive/greedy — upper bound.
2. **Paper 3 DQN** (dueling DQN + Transformer) — run BOTH under full CSI (their setting) and starved partial CSI (show it collapses).
3. **Greedy on last-known / stale CSI** — no inference, no active sensing (naive).
4. **Random** port selection — lower bound.
5. **Ablation — AIF with epistemic weight = 0** (pragmatic-only) — isolates the active-sensing benefit.

## 7. Metrics
- Long-term objective (Eq. 7): sum-rate − switching penalty.
- Sum-rate vs **observation/pilot budget** (fraction of ports observed).
- **Sample efficiency:** performance vs. number of environment interactions (AIF ≈ zero training vs DQN episodes).
- **Robustness:** vs Doppler `ρ`, vs N, vs estimation noise `σ_e`.
- Switching frequency/cost.
- **Decision latency** (ms/slot).
- Belief calibration (predicted vs actual channel).

## 8. Key figures (each engineered to expose one advantage)
- **Fig A (headline):** objective vs observation budget → AIF ≈ genie, DQN collapses.
- **Fig B:** performance vs training samples → AIF high from start, DQN slow ramp.
- **Fig C:** robustness to Doppler / distribution shift → AIF stable, DQN needs retrain.
- **Fig D:** epistemic-weight sweep → active sensing contributes.
- **Fig E (pre-empt objection):** model-mismatch — true channel ≠ assumed Jakes+AR(1) → AIF degrades gracefully (optional online estimation of ρ/β).
- **Fig F (pre-empt objection):** latency comparison → greedy-EFE competitive with DQN forward pass.
- **Fig G (optional):** CSI-aging / belief-calibration visualization.

## 9. Pre-empting the two known objections
- **Model mismatch (AIF's Achilles heel):** Fig E. Generate the true channel with a different model (Rician / non-Jakes spatial) while the agent assumes Jakes+AR(1). Show graceful degradation; optionally add lightweight online estimation of `ρ`, `β_k`. Turns a weakness into a robustness result.
- **Latency:** greedy submodular selection is `O(NM)`; report ms/slot and the (1−1/e) guarantee. Contrast with DQN's fast forward pass but note DQN needs full CSI + heavy training first.

## 10. Paper structure (~5–6 pages)
- **I. Introduction** — FAS, port selection, the full-CSI assumption is unrealistic, contributions.
- **II. System Model & Problem** — inherit Paper 3, add partial observability + estimation noise.
- **III. Active-Inference Port Selection** — generative model, Kalman belief, EFE, greedy algorithm + guarantee.
- **IV. Simulation Results** — figures above.
- **V. Conclusion** — and tease the IDET/Option-2 follow-up.

## 11. Tooling
- Python. `numpy`/`scipy` for the complex Kalman filter + log-det info gain (roll our own linear-Gaussian AIF — `pymdp` is discrete-only and not a fit).
- `PyTorch` only for the DQN baseline (reimplement a dueling DQN; Transformer optional if time-limited).
- Reuse Paper 3 sim params for a clean comparison.

## 12. Milestones / steps
- **Phase 0 — Foundation:** implement the shared channel simulator (Jakes `R`, AR(1) `ρ`, ZF, rate, switching cost). Reproduce a Paper-3-style DQN baseline (or a simpler DQN if time-constrained). *Deliverable: reproducible sim + baseline numbers.*
- **Phase 1 — AIF agent:** complex Kalman belief over N ports; EFE terms (pragmatic expected rate, epistemic log-det, switching); greedy selection. *Deliverable: working agent, sanity plots.*
- **Phase 2 — Experiments:** implement remaining baselines + metrics; run budget/sample/robustness sweeps; produce Figs A–D.
- **Phase 3 — Objection-proofing:** model-mismatch (Fig E) + latency (Fig F) + ablation (Fig D).
- **Phase 4 — Writing:** draft, related-work positioning, iterate.

## 13. Open decisions / risks
- **Observation model:** tie observation to activation (chosen) vs. a separate pilot/probe budget. Chosen = simpler, cleaner explore/exploit story. Revisit if reviewers want a dedicated sensing phase.
- **DQN baseline fidelity:** full Transformer-dueling-DQN reproduction is costly; a solid dueling-DQN may suffice for a letter. Decide by time budget.
- **Per-user vs joint belief:** start per-user (K independent Kalman filters); revisit if inter-user structure matters.
- **Beamforming (decided):** fully-digital **robust MMSE** as the main precoder (uses belief `μ, Σ`); fixed ZF as a baseline; joint `W` optimization and **hybrid** architecture are future/journal work (hybrid valid under `N_RF ≥ 2K`, narrowband; wideband hybrid adds beam squint + reduced observability). EFE pragmatic term uses the robust-MMSE rate for consistency — see `EFE_DESIGN.md`.
