# Meeting Notes — Draft Review, Open Problems & Questions for the PhD Advisor

Prepared for the weekly guidance meeting. Two parts: (A) what to fix/add in the current draft, and
(B) the questions to ask the PhD student (grouped, with priorities marked ★). Also records the answer
to the "which type of FAS are we assuming?" question.

---

## A. Review of the current draft (`paper/Active_Inference_FAS_draft.docx`)

**Overall:** strong, near-submittable IEEE letter — full structure (Abstract, Intro, System Model,
Method, 6 result figures, Conclusion, Acknowledgment, References), clean prose, contributions crisp,
every reference honestly tagged `[VERIFY]`.

**What to fix / add, in priority order:**

1. **★ The "surpasses the genie on the objective" claim — biggest risk.** Abstract, contribution #4,
   and Figs 1 & 3 lean on "exceeds/dominates the genie." That genie is *switching-blind*, so beating it
   on the switching-aware objective is a soft claim a reviewer will attack.
   **Fix:** add a *switching-aware* genie and reframe as "approaches the full-CSI rate ceiling / beats
   the *fair* baselines." We cannot beat a full-CSI genie on **rate** (it's an upper bound) — we reach
   84–89% of it. We only "beat" the current genie on the **objective** because that genie ignores the
   movement cost it then gets charged for.

2. **★ FAS hardware type & switching-cost model is unspecified / inconsistent** (see Section B below and
   the note in Section C). The prose says "moving the fluid element costs delay and energy" (sounds
   mechanical), but the math uses a *uniform* per-port cost (pixel-like). Make them consistent.

3. **Parameter consistency.** The draft states `sigma_e^2 = 1e-3`, but closed-loop runs used `0.01`.
   Every figure must use one locked operating point and the text must match it. Verify all numbers.

4. **Pre-empt "is it really partial CSI?"** Because observe-then-precode gives *fresh* pilots on the
   served ports, state clearly that the novelty is inferring the **N−M hidden ports for selection**
   (we still never see most ports). One sentence.

5. **Position the AIF novelty.** Half a sentence on whether active inference has been used for
   antenna/beam/port selection before (reviewers will ask "what's new vs prior AIF-in-wireless?").

6. **Optional but nice:** a small system diagram; a one-line complexity/latency statement (~20 ms/slot).

---

## B. The FAS hardware type — what our simulation actually assumes

There are two families of fluid/movable antenna hardware, and they imply very different switching costs:

- **Pixel / RF-switch type** ("ports like buttons"): a grid of elements connected by electronic
  switches; activating a port is ~nanosecond and cheap. Switching cost ∝ **number** of ports toggled.
- **Mechanically-movable / liquid-metal type**: the radiating element physically **moves** a distance;
  switching takes time and energy that **grow with the movement distance**. (This is Paper 1's model,
  delay `tau ∝ |i - j|`.)

**Our simulation currently uses a *uniform* switching cost:** `e_sw × |S XOR S_prev|` — every changed
port costs the same, independent of distance. That best matches the **pixel/RF-switch type** (or is a
first-order abstraction). It does **not** capture distance-dependent mechanical movement (which is
listed in `FUTURE_WORK.md` as "distance-weighted switching").

**Why it matters:**
- Pixel-based → switching is cheap → our switching-awareness edge is smaller (but method still valid).
- Mechanical → switching is expensive **and** distance-dependent → our edge is *bigger*, but we'd need
  the distance-weighted cost to be realistic and comparable to the anchor papers.
- There is a prose/model mismatch to resolve (see A2).

---

## C. Questions for the PhD advisor (★ = ask first if short on time)

### FAS hardware & switching cost
1. **★** Should we target **pixel/RF-switch FAS** (uniform, cheap switching) or
   **mechanically-movable/liquid-metal** (distance-dependent, costly)? Which is more valued / expected
   by reviewers in this area?
2. Should the switching cost be **distance-weighted** to match mechanical movement (like the anchor
   papers), or is uniform-count acceptable? Does Paper 3 use distance-based movement delay — must we
   match it for a fair comparison?

### Belief / estimation algorithm
3. **★** Is a **linear-Gaussian Kalman belief** the right tool, or should we use / compare against
   **S-BAR (Bayesian-GP)** or **sequential LMMSE** from the FAS literature? Is Kalman seen as too
   simple, or fine because it's principled and fast?
4. Is the **Jakes-spatial + AR(1)-temporal** generative model standard and accepted for FAS, or is
   there a more realistic/expected channel model we should adopt?

### Novelty & venue
5. **★** Is "**active inference for FAS port selection under partial CSI**" novel and substantial enough
   for a letter? Has AIF been applied to antenna/beam/port selection before (positioning risk)?
6. Which venue fits best — IEEE **WCL**, **Comm. Letters**, or a conference?

### Baselines & fairness
7. Is comparing to a **switching-blind genie** acceptable, or must we add a **switching-aware genie**?
   (We think we must.)
8. Is our **DRL baseline** (Transformer + policy gradient) a fair stand-in for Paper 3's
   dueling-DQN+Transformer, or do reviewers require a faithful reproduction of the anchor's method?
9. Should we add **S-BAR / LMMSE as channel-estimation baselines** (estimate-then-select) to show the
   *joint* decision wins?

### Method assumptions / likely objections
10. **Observe-then-precode:** is getting fresh pilots on the served ports a fair "partial-CSI" claim,
    and is pilot-then-data-within-a-slot a standard assumption?
11. Is **fully-digital robust-MMSE** beamforming fine for a letter, or do reviewers expect **hybrid**?
12. Are our simplifications — **per-user independent channels sharing R**, and **known rho/beta** —
    acceptable?

### Scope / what to include
13. Should the **moving-hotspot tracking** scenario and **online R-learning** be in the main letter, or
    kept as future work (space limits)?
14. Is "**84–89% of genie at a 20% budget**" a compelling headline, or what result would strengthen it
    most?

---

## D. First fix regardless of the meeting outcome
Implement the **switching-aware genie** baseline, re-measure the true gap, and correct the figures/claims
so "beats the genie" becomes "approaches the ceiling / beats the fair baselines."
