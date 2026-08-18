# Future Work — extensions beyond the current letter

This file is DELIBERATELY separate from `RESEARCH_PLAN.md` / `SIMULATION_PLAN.md`, which
specify the **current letter** (static-regime partial-CSI active acquisition). The items here are
follow-ups / journal-scale extensions so they are not lost. Nothing here gates the current paper.

Current-paper decision (2026-08-10): **Path B** — ship the static-regime results
(84% of genie rate observing 20% of ports; wins the switching-aware objective; zero training;
robust to Doppler & model mismatch; learns R). The two extensions below are the roadmap after it.

---

## 0. Paper scope map — S1 (this letter) vs S2 (next paper)

The single decision that organizes the whole roadmap: **where does the RF-chain budget bite, and does the
agent design its own measurement operator?** This splits into two papers.

| Axis | **S1 — THIS paper (done)** | **S2 — next paper (designed, not built)** |
|---|---|---|
| Name | "sense-per-port, transmit-hybrid" active **acquisition** | "sense-through-the-analog-network" active **sensing** |
| Sensing observation | `y = P_S h + n` — per-port pilot reads on the activated ports | `y = F_RF^H h + n` — N_RF **mixed/compressed** reads through the analog net |
| Belief / Kalman | linear-Gaussian, obs matrix = selection `P_S` | linear-Gaussian, obs matrix = combiner `F_RF^H` — **same engine**, swap the matrix |
| Epistemic EFE term | depends only on **which discrete ports** S: `logdet(I+Σ_S/σ_e²)` | depends on the **continuous analog weights**: `logdet(I+F_RF^H Σ F_RF/σ_e²)` |
| What the agent controls | *which ports* to activate (discrete) | *which ports* **+ how to combine them for sensing** (discrete + continuous manifold) |
| Transmit | hybrid `F_RF W_BB` factorization of belief precoder (version 3, done) | same, but sensing-`F_RF` and transmit-`F_RF` now **compete** (explore/exploit in the analog domain) |
| Extra machinery vs S1 | — | a **unit-modulus manifold optimizer** inside the EFE loop; a sensing-vs-transmit `F_RF` schedule |
| Feedback needed | partial CSI on M activated ports | even LESS — N_RF mixed numbers (a selling point) |
| Novelty | solid, self-contained (AIF acquisition + hybrid transmit under partial CSI) | **medium** in discrete form (overlaps active-sensing lit — Sohrabi/Chen/Yu JSAC'22), **HIGH** in the continuous/movable limit (§2e) = active *spatial sampling* |
| Status | **COMPLETE & validated** (`sim_version3/`, `results_v3.md`) | scoped only — Kalman unchanged is the key de-risker |

**Decision rule for what belongs where:** S1 assumes the sensing hardware can read the activated ports
(time-multiplexed pilots, ⌈M/N_RF⌉ sub-slots) — the RF budget bites only at *transmission*. The moment we
say the RF budget bites at *sensing too* (single-shot compressed reads) and let the agent **choose the
combiner to maximize information gain**, we are in S2. S2 is where "active inference" becomes literal — the
agent designs its own perception. **Ship S1; make S2 (ideally its continuous form, §2e) the flagship
follow-up.**

**S2 is not one thing — it has three complexity tiers.** Pick deliberately; the effort and novelty scale
steeply. The key de-risker across all three: the model stays **linear-Gaussian**, so the Kalman engine is
untouched — what changes is the *action space* (continuous analog weights) and the *optimizer inside the EFE
loop*. It is a new action space, not a new inference engine.

  - **Light-S2 (~1–2 weeks) — design the sensing combiner on a FIXED active set.** Keep discrete port
    selection as-is; add one step that picks `F_RF` to maximize `logdet(I + F_RF^H Σ F_RF/σ_e²)`. Unconstrained
    the optimum is closed-form (align `F_RF` with the top eigenvectors of Σ — water-filling on *uncertainty*);
    with the unit-modulus constraint there is no closed form ⇒ add a small **manifold / coordinate-descent
    optimizer** (same spirit as the AltMin already in `hybrid.py`). Kalman unchanged. Deliverable: one figure,
    "EFE-designed sensing beams beat fixed per-port sensing by X%." This is the optional S1-hardening figure —
    but be aware the gain may be modest and it invites the "this is just known info-max active sensing" comparison.
  - **Medium-S2 (~a month) — jointly choose port-connections AND the combining weights.** The agent optimizes
    a continuous matrix *inside* the greedy selection loop; compute grows (a continuous optimization nested in
    the discrete one, roughly 10–50× the per-slot cost — starts to matter at N=441).
  - **Full-S2 (a whole second paper) — sensing AND transmit share the analog network under EFE.** The real
    research problem: the `F_RF` best for *sensing* (spread beams to probe uncertainty) is **not** the `F_RF`
    best for *transmission* (focus energy on known-good directions). That explore/exploit tension now lives in
    the **analog domain**, across a shared coherence block — resolve it via two configs (pilot vs data
    sub-slot), one compromise config, or a schedule. Likely needs a manifold or learning-based solver. This is
    the flagship; strongest in the **continuous / movable-antenna limit** (§2e), where it becomes *active
    spatial sampling* — genuinely FAS-native and differentiated from the mmWave active-sensing crowd.

Novelty caution (applies to all tiers): the *mechanism* — max-mutual-information measurement design — is NOT
new (Bayesian D-optimal experiment design; Sohrabi/Chen/Yu "Active Sensing for Communications by Learning,"
JSAC 2022). Our delta is the **EFE framing that unifies port-selection + sensing-beam design + precoding +
switching cost in a FAS setting** — a medium-strength "novel combination," strongest in the continuous limit.

---

## 1. Dynamic tracking — the moving-power-envelope belief (Path A)

**Why:** In the standard FAS model every port has equal average power (Jakes, unit-diagonal R), so the
best port SET is static and the optimal policy is "find a good set and hold" (our agent does this,
~0 switching — correct, not a bug). The interesting *active* story appears when the good region MOVES
(user mobility, a drifting scattering cluster). We built that scenario (`MovingHotspotSimulator`) and
found an **honest negative result**: the current agent tracks poorly and loses to naive round-robin,
because its stationary Jakes+AR(1) belief is **blind to the moving power envelope** — it attributes
hotspot power to random fading (decays via rho) and can only REACT by blind sensing.

**The fix (the contribution):** add a second, slow latent layer to the generative model — a belief over
the **spatial power envelope** a_n(t) (equivalently a latent hotspot location / per-port power that drifts
smoothly). Then the agent can:
  - INFER the envelope from the |y|^2 it already measures on activated ports,
  - exploit spatial smoothness (a GP / correlation prior) to interpolate the envelope over UN-measured ports,
  - PREDICT where the hotspot is heading (a motion model on the latent location),
  - drive EFE to **sense ahead of the hotspot** and serve the predicted-best ports, switching only as the
    hotspot moves (few, purposeful switches vs naive's blind round-robin and genie's constant churn).

**Concrete build:**
  - State: per-port log-power b_n(t) (or a low-dim hotspot-location state z(t) in R^2 + width).
  - Dynamics: b(t) = smooth spatial field + slow AR(1)/random-walk motion of the peak.
  - Observation: power estimate from |y|^2 on activated ports (note: nonlinear -> use a log/EKF or a
    moment-matching update, or estimate power via short pilot averaging).
  - Two-timescale inference: fast fading Kalman (existing) + slow envelope belief (new).
  - EFE epistemic term now values sensing that reduces envelope/location uncertainty (find the hotspot),
    not just fading uncertainty.

**Verify/figures to add:** rate-over-time tracking curves (genie / AIF-with-envelope / naive / fixed);
AIF should track near genie with FAR fewer switches than genie and beat naive decisively; a
"lead/lag" plot showing AIF's belief predicts the hotspot location; sweep hotspot speed (hs_period) and
size (hs_width) to map where tracking is feasible. Infra already in repo:
`channel.MovingHotspotSimulator`, `agent.run_fixed`.

**Risk:** nonlinear power observation + two-timescale inference is real work; journal-scale.

---

## 2. Scaling up — N = 64, 100, ... (and larger apertures / more users)

**The question (user, 2026-08-10):** what does active inference do if we go from N=25 ports to 64 or 100
(and scale K, M, aperture)? **Short answer: the partial-CSI advantage should get STRONGER, but HOW you
scale matters — density vs aperture behave very differently.** Two distinct regimes:

### 2a. Denser array, SAME aperture (more ports packed into the same physical size)
  - Jakes R over a fixed aperture is strongly **rank-deficient**; its effective rank ~ the spatial DoF of
    the aperture (~pi * area_in_wavelengths^2), which **barely grows** as you add ports. (We saw N=25 over
    1x1 lambda has only ~3-5 significant eigenvalues.)
  - => The channel becomes MORE compressible as N grows: a handful of measurements pin down all N ports.
  - **Prediction:** the observation fraction M/N needed to reach X% of genie **DECREASES** with N. At
    N=100 dense, we might need to observe only ~5% of ports for the same performance. This is the
    STRONGEST version of the "you don't need full CSI" thesis, and where AIF's model-based inference beats
    naive/DQN (which must effectively probe far more).
  - **Caveat:** selection headroom stays small (ports are redundant), so the win is about *acquisition
    efficiency* (few pilots), not about picking cleverly among near-identical ports.

### 2b. LARGER aperture, same spacing (more ports = physically bigger array)
  - Effective rank / spatial DoF **grows ~ proportional to aperture area** => genuinely more distinct,
    resolvable directions and more independent "good" ports.
  - => Selection and (if dynamic) tracking matter MORE — there are real choices to make. More DoF also
    supports more users K and sharper beams.
  - **Prediction:** active selection (EFE) opens a bigger gap over random/naive here than in the dense
    same-aperture case; this is the regime to showcase smart acquisition.

### 2c. The baseline economics flip in our favor at scale
  - Full-CSI / genie / Paper-3-style methods must measure ~all N ports each coherence time -> pilot
    overhead scales with N and becomes prohibitive at N=100. Our agent's budget is a FIXED small M.
  - **This is the strongest scaling argument:** as N grows, "measure everything" gets impractical, so a
    method that measures a fixed handful and infers the rest wins by an increasing margin. Report
    performance-per-pilot (or per-switching-energy), not just rate, and the gap widens with N.

### 2d. Computation — keep it tractable at large N
  - Belief covariance is N x N; Kalman ~O(N^2) update, greedy EFE ~O(N*M) evals. Fine to N~100-200.
  - For large N exploit the **low rank of R**: carry the belief in the reduced eigenbasis of R (dimension
    r = effective rank << N). Predict/update/log-det become O(r^2) instead of O(N^2). This makes the agent
    scale to very dense arrays and is itself a nice efficiency contribution.
  - Greedy selection with rank-1 log-det updates (already submodular) stays cheap; warm-start the MMSE
    re-solve for the pragmatic marginal.

### 2e. The natural limit: continuous aperture / movable antennas
  - As density -> infinity, discrete port selection becomes CONTINUOUS antenna positioning (movable
    antennas / MA). The belief becomes a **Gaussian process over the aperture** (a continuous field), and
    EFE becomes a **gradient-based** objective on antenna position (the continuous form of active inference
    we discussed at the very start). This unifies the discrete-FAS and continuous-MA stories and is a
    compelling standalone direction.

**Experiments to run (when we pick this up):**
  1. Fix aperture 1x1 lambda; sweep N in {25, 49, 64, 100}; fix M. Plot AIF % of genie vs N (expect flat/up)
     and the M/N needed for 90% of genie vs N (expect down). Confirms compressibility (regime 2a).
  2. Fix spacing lambda/4; grow aperture with N in {25,64,100}; K in {3,6,8}. Show selection/EFE gap over
     naive grows (regime 2b) and more users are served.
  3. Performance-per-pilot and per-switching-energy vs N -> the baseline-economics flip (2c).
  4. Low-rank (eigenbasis) belief: verify same performance at O(r^2) cost; plot runtime vs N with/without.
  5. (Stretch) continuous-aperture GP belief + gradient EFE (2e).

**Predicted headline:** "The advantage of active-inference acquisition GROWS with array size — at N=100 it
matches the full-CSI genie while measuring only a few percent of ports, because the sub-wavelength channel
is low-rank and full-CSI pilot overhead is prohibitive." This is likely a stronger paper than the N=25 letter.

---

## 3. Inference front-end upgrade — GP / S-BAR belief instead of the linear-Gaussian Kalman

**Why (from the related-work read):** our belief is a linear-Gaussian Kalman filter over the ports with a
FIXED Jakes correlation R as its (implicit) kernel. The **Successive Bayesian Reconstructor, S-BAR**
(Zhang, Zhu, Dai, Heath, IEEE TWC 2025) does essentially the same *reconstruction* but as **Gaussian-process
regression** with a **learned/experiential kernel**, and is explicitly robust to **model mismatch** (does not
assume Jakes/sparsity/slow-variation). It beats model-based estimators on estimation accuracy. So S-BAR is a
strictly stronger version of our PERCEPTION block.

**Idea:** swap our Kalman belief for a **GP / S-BAR-style reconstructor** as the front-end that the EFE
consumes. Benefits:
  - Better partial-CSI inference under real (non-Jakes) propagation -> directly improves the wrong-R case we
    saw in Step 8 (learning R) and the model-mismatch figure (Fig E). This is the principled version of our
    hand-rolled R-learning.
  - The GP kernel is **learned from data** (kernel learning), unifying "estimate the channel" and "learn R".
  - GP over a CONTINUOUS input naturally gives the **continuous-aperture / movable-antenna** belief in §2e
    (belief becomes a field over the aperture; EFE becomes gradient-based positioning).
  - The GP posterior variance plugs straight into our EPISTEMIC term (info gain = entropy reduction), and the
    posterior mean into the PRAGMATIC term — the active-inference machinery is unchanged; only the belief
    representation upgrades.
**Positioning note (important):** S-BAR / GP is CHANNEL ESTIMATION only — no port SELECTION, no switching
cost, no rate/serving objective. Our contribution stays the **decision layer (EFE: what to sense, what to
serve, when to move)** built ON TOP of Bayesian reconstruction. Cite S-BAR as the inference front-end we build
on; do NOT claim to beat it on estimation. (See also the LMMSE estimator, Skouroumounis & Krikidis, TComm 2023
— sequential LMMSE + pick-strongest = essentially our `naive` baseline + a stochastic-geometry outage analysis;
we beat its selection heuristic, cite its analysis.)

## 4. Physical (distance-weighted) switching cost  →  superseded by the transport-cost view in §6

Currently switching cost = number of ports changed, |S XOR S_prev|. But the fluid-antenna moving delay is
proportional to the DISTANCE moved (Eq. 2 of the IDET paper: tau ∝ |i-j|). A **distance-weighted** switching
cost would make SMALL exploratory moves cheap and large jumps expensive -> the agent could sense/track nearby
ports almost for free, potentially shifting the whole rate/switching Pareto frontier up (esp. relevant for the
moving-hotspot tracking in §1, where cheap local moves = cheap tracking). Worth modelling and re-running the
frontier. **NOTE:** in 2D with multiple moving elements this is not a scalar per-port weight but a *matching*
cost — see §6, which is the principled version of this idea.

## 6. Physical realization — pixel/RF-switching vs liquid-metal, and the optimal-transport switching cost

**The distinction (user, 2026-08-18).** There are two physically different ways to build a FAS, and they are
*different research objects*, not the same model with a tweaked cost:

  - **Pixel / RF-switching FAS (what S1 assumes).** A fixed grid of N candidate ports (radiating pixels);
    M RF chains connect to a chosen subset via switches. "Moving" = electronically re-energizing a different
    port — **no mass travels**. Activation is a *set* S; the RF chains are interchangeable, so **there is no
    "which antenna goes where" question** — any chain can drive any activated port. Switching cost ≈ number
    of ports toggled. This is the dominant FAS model (Wong et al.; reconfigurable-pixel antennas) and all our
    baselines (DRL, bandit, LMMSE) live here.
  - **Liquid-metal / movable FAS (the "literal fluid" model).** A small number of *physical* radiators —
    liquid-metal blobs (EGaIn/galinstan) pumped through microchannels, or mechanically movable antennas (MA)
    on rails — that **physically translate** to new positions. Now "moving" has a real continuous trajectory
    and a travel cost **∝ distance**. This is the model behind the 1D "pace/distance/time" framing the user
    saw (IDET τ ∝ |i−j|; the movable-antenna line, Zhu/Ma/Zhang).

**The user's key observation is correct and is the crux.** In the liquid/MA model, when the M physical
elements move from configuration `P_prev = {p_1..p_M}` to `P_new = {q_1..q_M}`, the reconfiguration cost is
**not** a port-toggle count — it is the total physical travel, which requires **matching** old positions to
new ones ("should the element at a go to c or to d?"). Minimising the summed travel over all matchings is
exactly the **linear assignment problem / discrete optimal transport (1-Wasserstein / earth-mover distance)**:
```
    C_switch(P_prev → P_new) = min over matchings π  Σ_i dist(p_i, q_{π(i)})
```
Good news the user was unsure about: the clean version is **NOT combinatorially hard** — identical,
interchangeable elements ⇒ min-cost bipartite matching, solved EXACTLY by the **Hungarian algorithm in
O(M³)** (instant for M=5–10). It only becomes hard (NP-hard, multi-agent path-finding / motion planning) if
we add **collision avoidance / no-crossing / shared-channel constraints** — which is itself a legitimate,
novel sub-problem to own. The distance metric is a hardware choice: Euclidean (free 2D motion), Manhattan
(X-then-Y rails), or geodesic on a microchannel graph (confined fluid).

**Why this is valuable — and where it belongs.** This upgrades the ad-hoc "distance-weighted" idea of §4 into
a *principled* switching cost with the right mathematical object (optimal transport), and it plugs straight
into EFE: replace the `|S XOR S_prev|` switching term with `C_switch` above. Consequences:
  - small **local** moves become cheap ⇒ the agent tracks/explores nearby ports almost for free ⇒ directly
    strengthens the moving-hotspot tracking story (§1) and reshapes the rate/switching Pareto frontier;
  - it is the natural bridge to the **continuous movable-antenna / GP-over-aperture limit** (§2e, §3):
    discrete ports → transport/assignment cost → continuous positioning with **gradient** EFE. Liquid is the
    *physical narrative* that unifies the entire discrete-FAS ↔ continuous-MA roadmap;
  - a fresh contribution nobody in the AIF-FAS space has: **"EFE port selection with an optimal-transport
    switching cost for physically-moving FAS,"** and its multi-element assignment/trajectory sub-problem.

**Does the multi-element moving-fluid setup physically exist?** Partly. Single liquid-metal reconfigurable
antennas are demonstrated in the lab; multi-element MA arrays on 2D rails exist as a research concept; but
*many* independent liquid radiators sliding fast over a 2D surface simultaneously is at/beyond the current
hardware frontier — so treat it as an **idealized-but-legitimate model**, stated honestly as such.

**RECOMMENDATION.** Keep S1 **pixel-based** — it is done, coherent, matches every baseline and reviewer
expectation, and the acquisition story is hardware-agnostic; switching mid-stream would force re-deriving and
re-running everything for no gain to the current claim. Adopt the **liquid-metal / physical-motion** model as a
*defining feature of a future paper*, where the transport-cost switching + the multi-element assignment
(Hungarian → collision-aware planning) + the continuous-MA limit become the contribution. It composes cleanly
with either S1 or S2 but is most naturally the journal-scale "physical motion" version. This subsumes §4.

## 5. Other extensions (parking lot)
  - Multi-step (planning) EFE instead of myopic one-slot selection (RESEARCH_PLAN Sec. 7).
  - Hybrid beamforming / wideband (beam squint) — ties to the THz paper studied earlier.
  - DONE: DQN baseline (Step 10) and bandit baseline (Step 12) -> model-based AIF beats model-free learning.
    A **structured/correlated bandit** (exploiting R, like the Online-Learning paper's use of J) would be a
    fairer, stronger bandit — narrowing the gap by moving the bandit TOWARD our model-based approach (itself a
    nice framing: "the better a bandit does, the more it has to become model-based").
  - Per-user (rather than shared) hotspots and joint multi-user tracking.
