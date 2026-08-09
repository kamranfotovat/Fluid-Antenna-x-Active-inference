# Paper outline — bulleted CLAIMS per section (not prose yet)

This is the logical skeleton to react to before we write any polished sentences. Each bullet is a
*claim/thing-the-paragraph-must-say*, grounded in results we actually have (figures in `figures/`).
Target: IEEE letter (~5-6 pages, e.g. IEEE Wireless Communications Letters). Once we agree on this
skeleton, we fill in prose section by section.

> **Reminders while drafting:** (1) every citation below is a PLACEHOLDER — we/Zijun must VERIFY each
> reference exists and says what we claim (AI-suggested refs are a hallucination risk). (2) Disclose any
> AI writing assistance per the venue's policy (usually in Acknowledgments). (3) Authors are fully
> accountable for every number — all figures here come from our own verified sims.

---

## Title (options)
1. **Active Inference for Port Selection in Fluid Antenna Systems under Partial CSI**
2. Active Channel Acquisition for Fluid Antennas: A Free-Energy Approach to Port Selection
3. You Don't Need Full CSI: Active-Inference Port Selection for Fluid Antenna Systems

## Authors / affiliations
- **Kian Fotovat**, Dept. of Electrical and Computer Engineering, University of Tehran, Tehran, Iran. *(first author)*
- **Kamran Fotovat**, Iran University of Science and Technology, Tehran, Iran. *(second author)*
- *(Open item: corresponding author + who submits/pays — see the publishing notes. Consider adding a
  senior co-author, e.g. Zijun Wang / a UT faculty advisor, for corresponding-author + funding; your call.)*

---

## Abstract (claims to compress into ~6 sentences)
- FAS/movable antennas add spatial DoF via port switching, but choosing ports needs channel knowledge
  over many candidate ports -> large CSI-acquisition overhead.
- Prior port-selection methods assume (near-)full CSI and/or require training; the sensing cost and
  partial observability are not modeled.
- We formulate FAS port selection as **active inference under partial CSI**: the agent measures only the
  M ports it activates, infers the rest from a spatio-temporal generative model (Jakes spatial
  correlation R + AR(1) temporal ρ) via a **Kalman belief**, and selects ports by **minimizing expected
  free energy (EFE) = − expected rate − information gain + switching cost**.
- A greedy submodular selector gives low decision latency; an observe-then-precode protocol keeps the
  served-port CSI fresh.
- Results: matches **84–89% of a full-CSI genie while observing only 20% of ports**, **beats** naive,
  DRL, and bandit baselines on the switching-aware objective, needs **zero training**, and is robust to
  Doppler and to model mismatch (it can learn R online).

---

## I. Introduction (claims)
- **Para 1 — FAS context.** Fluid/movable antennas reconfigure position to exploit spatial diversity
  beyond fixed MIMO; a small number of RF chains activate a few of many candidate ports. [cite Wong FAS/FAMA]
- **Para 2 — the problem.** Port selection is the key control problem, but scoring ports needs CSI over
  all N ports -> prohibitive training/pilot overhead, worsening as arrays grow. [cite]
- **Para 3 — prior work & its gap.**
  - Learning-based selection: DRL (Transformer-DQN) [switching-cost DRL, Comm. Lett.] and bandit
    [Zou et al., WCL'24] — but assume full CSI and/or need heavy training; ignore acquisition cost.
  - Channel estimation for FAS: S-BAR [Zhang–Dai–Heath, TWC'25], sequential LMMSE
    [Skouroumounis–Krikidis, TComm'23] — solve *estimation*, not the *decision* (selection + serving +
    movement) problem.
  - **Gap:** no method treats port selection as *active acquisition under partial CSI*, jointly trading
    throughput, information gain, and antenna-movement cost.
- **Para 4 — contributions (bullet list in the paper):**
  1. First formulation of FAS port selection as **active inference under partial observability** (drops
     the full-CSI assumption).
  2. A **model-based agent**: complex Kalman belief over all N ports (spatio-temporal prior), and an EFE
     objective that **unifies expected rate + information gain − switching cost** in one decision rule.
  3. A **greedy submodular** selector (epistemic term is submodular → (1−1/e) guarantee) with low,
     O(NM) latency; and an **observe-then-precode** protocol that makes performance robust to Doppler.
  4. Simulations: **84–89% of the full-CSI genie at a 20% observation budget**, **beating** naive/DRL/
     bandit on the switching-aware objective with **zero training**, plus **online R-learning** for
     model mismatch.

## II. System Model and Problem Formulation (claims)
- **Geometry.** BS with an N = Nx×Ny port FAS over a sub-λ/2 aperture; activate M ports/slot (M RF
  chains) to serve K single-antenna users; N ≥ M ≥ K. Default N=25, M=5, K=3.
- **Channel.** Per-user h_k(t) ∈ C^N. Spatial correlation R (Jakes, R_ij = J0(2π d_ij/λ)); temporal
  AR(1): h_k(t)=ρ h_k(t−1)+√(1−ρ²) e_k, e_k~CN(0,β_k R), ρ = J0(2π f_D T_s). [give the two equations]
- **Partial observability (the key departure).** Each slot the agent obtains noisy pilots only on the M
  activated ports, y_k = P_S h_k + CN(0, σ_e² I); the other N−M ports are hidden.
- **Precoding + rate.** Robust MMSE precoder from the belief; per-user SINR and R_k = log2(1+SINR_k).
- **Switching cost.** Moving the fluid element costs delay/energy; C_t = |S_t △ S_{t−1}|, E_sw = e_sw C_t.
- **Objective (Eq. 7).** maximize Σ_t [ Σ_k R_k(t) − η_sw E_sw(t) ].

## III. Active-Inference Port Selection (claims)
- **Generative model → AIF objects.** State = h_k; transition B = AR(1) (physics); likelihood A =
  partial noisy observation; preference C = rate (utility, not a hand-set vector); EFE selects the action.
- **Perception = complex Kalman belief.** Predict step = CSI aging (μ←ρμ, Σ←ρ²Σ+(1−ρ²)β_kR); update =
  Joseph form. Belief is calibrated (predicted Σ matches realized error) — [Fig: diagnostics].
- **EFE decision rule (in bits).** G(S) = −α·Pragmatic − β·Epistemic + Switching:
  - Pragmatic = expected robust-MMSE sum-rate (imperfect-CSI lower bound; uses Σ → honest under
    uncertainty).
  - Epistemic = Σ_k log2 det(I + P_SΣ_kP_S^H/σ_e²) — mutual information; **monotone submodular**;
    observing a port also sharpens correlated neighbours via R.
  - Switching = η_sw e_sw |S △ S_prev| (modular).
- **Observe-then-precode protocol.** Select on the predicted belief, then sense the activated ports and
  precode on the fresh belief → served-CSI error ~ σ_e² instead of the aging floor. *The single biggest
  lever on rate and on Doppler robustness.*
- **Greedy submodular selection.** Build S by max marginal G; O(NM) evaluations; (1−1/e) on the epistemic
  part; ~20 ms/slot vs ~9 s exhaustive (≈500× faster).
- **(Optional) online model learning.** If R is unknown/non-Jakes, estimate it from co-observed pilots
  (normalized correlation) — recovers the mismatch gap.

## IV. Simulation Results (claims → each maps to a figure we already have)
- **Setup.** N=25 (5×5, sub-λ/2), K=3, M=5 (20% budget), 15 dB, σ_e²=1e-3, ρ=0.9, η_sw=1; averaged over
  MC seeds. Baselines: full-CSI genie (ceiling), naive (no inference), random, DRL (Transformer, policy
  gradient), bandit (combinatorial UCB).
- **Result 1 — observation budget [figA].** AIF tracks 73→91% of genie rate as budget grows 12→40%, and
  *exceeds* the genie on the switching-aware objective for budgets ≥ ~18%.
- **Result 2 — protocol & Doppler robustness [figC].** observe-then-precode reaches ~84–89% of genie and
  is flat across ρ; predict-then-act collapses (64%→25%) as the channel speeds up.
- **Result 3 — Pareto frontier [figF].** Sweeping the exploration weight traces a frontier that dominates
  the genie: from 84% rate at ~0 switching to 89% rate; the whole frontier beats the genie objective.
- **Result 4 — vs baselines [figR].** Same rate as naive but +34% objective and ~0 vs 3 (naive) / 6
  (genie) switches/slot: EFE’s switching-awareness is the edge.
- **Result 5 — vs DRL [figB].** AIF (zero training, 20% CSI) ≥ a fully-trained full-CSI DRL on the
  objective; DRL needs ~100–200 episodes to reach its plateau (sample efficiency).
- **Result 6 — vs bandit [figG].** AIF beats a combinatorial-UCB bandit at every exploration setting
  (+12% over its best), same rate but ~100× less switching (model-based lock vs perpetual exploration).
- **Result 7 — model mismatch / learning [figE].** On a non-Jakes (exponential-correlation) channel, an
  agent assuming Jakes loses; learning R online recovers the oracle performance.

## V. Conclusion (claims)
- Recap: partial-CSI active acquisition matches most of full-CSI performance while measuring a fraction
  of ports, wins the switching-aware objective vs learning baselines, needs no training, and is robust.
- Future work (from FUTURE_WORK.md): moving-hotspot tracking via a moving-power-envelope belief; scaling
  to large N (compressibility + pilot-economics); a GP/S-BAR belief front-end; distance-weighted
  switching cost; continuous movable-antenna (gradient-EFE) limit.

## References (PLACEHOLDERS — verify every one)
- [FAS] K. K. Wong et al., "Fluid antenna systems," IEEE TWC, 2021; f-/s-FAMA papers.
- [DRL] Switching-Cost-Aware Deep RL for Dynamic Port Selection in FAS, IEEE Comm. Lett., 2026.
- [Bandit] J. Zou, S. Sun, C. Wang, "Online Learning-Induced Port Selection for FAS," IEEE WCL, 2024.
- [S-BAR] Z. Zhang, J. Zhu, L. Dai, R. W. Heath, "Successive Bayesian Reconstructor for Channel
  Estimation in FAS," IEEE TWC, 2025.
- [LMMSE] C. Skouroumounis, I. Krikidis, "Fluid Antenna with Linear MMSE Channel Estimation for
  Large-Scale Cellular Networks," IEEE TComm, 2023.
- [IDET] L. Zhang et al., "Energy-Efficient Port Selection and Beamforming for IDET Assisted by Fluid
  Antennas," IEEE JSAC, 2026.
- [AIF] T. Parr, G. Pezzulo, K. Friston, "Active Inference: The Free Energy Principle in Mind, Brain,
  and Behavior," MIT Press, 2022; K. Friston et al., process-theory paper, 2017.
