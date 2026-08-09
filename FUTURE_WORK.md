# Future Work — extensions beyond the current letter

This file is DELIBERATELY separate from `RESEARCH_PLAN.md` / `SIMULATION_PLAN.md`, which
specify the **current letter** (static-regime partial-CSI active acquisition). The items here are
follow-ups / journal-scale extensions so they are not lost. Nothing here gates the current paper.

Current-paper decision (2026-08-10): **Path B** — ship the static-regime results
(84% of genie rate observing 20% of ports; wins the switching-aware objective; zero training;
robust to Doppler & model mismatch; learns R). The two extensions below are the roadmap after it.

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

## 4. Physical (distance-weighted) switching cost

Currently switching cost = number of ports changed, |S XOR S_prev|. But the fluid-antenna moving delay is
proportional to the DISTANCE moved (Eq. 2 of the IDET paper: tau ∝ |i-j|). A **distance-weighted** switching
cost would make SMALL exploratory moves cheap and large jumps expensive -> the agent could sense/track nearby
ports almost for free, potentially shifting the whole rate/switching Pareto frontier up (esp. relevant for the
moving-hotspot tracking in §1, where cheap local moves = cheap tracking). Worth modelling and re-running the
frontier.

## 5. Other extensions (parking lot)
  - Multi-step (planning) EFE instead of myopic one-slot selection (RESEARCH_PLAN Sec. 7).
  - Hybrid beamforming / wideband (beam squint) — ties to the THz paper studied earlier.
  - DONE: DQN baseline (Step 10) and bandit baseline (Step 12) -> model-based AIF beats model-free learning.
    A **structured/correlated bandit** (exploiting R, like the Online-Learning paper's use of J) would be a
    fairer, stronger bandit — narrowing the gap by moving the bandit TOWARD our model-based approach (itself a
    nice framing: "the better a bandit does, the more it has to become model-based").
  - Per-user (rather than shared) hotspots and joint multi-user tracking.
