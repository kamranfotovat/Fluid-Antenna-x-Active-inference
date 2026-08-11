# Presentation Script — Active Inference for Port Selection in Fluid Antenna Systems

**Goal:** explain the whole project in ~20 minutes to someone who knows *nothing* about antennas,
wireless, or active inference. Every idea is built up in plain language first, then named. Say the
intuition, not the jargon.

**Suggested timing (20 min):**
| Part | Topic | Minutes |
|---|---|---|
| 1 | The one-sentence pitch | 1 |
| 2 | What is a Fluid Antenna System | 3 |
| 3 | The problem: knowing the channel is expensive | 3 |
| 4 | Our idea: only look where you point | 2 |
| 5 | The two patterns we exploit (space & time) | 2 |
| 6 | Active inference (the philosophy) | 1.5 |
| 7 | Perception: the Kalman filter | 3 |
| 8 | Decision: Expected Free Energy (3 terms) | 2.5 |
| 9 | Making it fast + the fresh-CSI trick | 1 |
| 10 | Results & baselines | 1.5 |
| 11 | Honest limits + takeaway | 0.5 |

---

## Part 1 — The one-sentence pitch (say this first)

> "We have an antenna that can pick a few good spots out of many to transmit from. Normally you'd
> have to measure the signal quality at *all* the spots to choose well — which is very expensive. We
> built an agent that only measures the few spots it actually uses, *guesses* the rest using the
> physics of how signals behave, and still performs almost as well as if it knew everything."

Then: "Let me explain each piece."

---

## Part 2 — What is a Fluid Antenna System (FAS)?

**Start with ordinary antennas.** A wireless transmitter has an antenna at a fixed location. The
quality of the wireless signal at any point in space is not constant — because signals bounce off
walls and objects and add up, in some spots they reinforce (strong signal) and in others they cancel
out (a "dead zone"). This is called **fading**. With a fixed antenna, if you happen to sit in a dead
zone, you're stuck.

**Now the fluid antenna.** A *fluid* (or *movable*) antenna can reposition its radiating element
within a small area — imagine a small grid of candidate positions, and the antenna can sit at any of
them. We call each candidate position a **port**. So instead of one fixed spot, we have, say, **25
ports** arranged in a 5×5 grid, and we can choose which ones to use.

- Because signal quality varies from spot to spot, some ports are much better than others *right now*.
- The antenna hardware can only actively use a **few** ports at once (each active port needs its own
  expensive radio unit). Say we light up **5 ports out of 25**.
- We use those 5 active ports to serve **3 users** at the same time.

**So the core job — "port selection" — is: which 5 of the 25 ports should I turn on this moment to
give my 3 users the best signal?** That's the whole control problem.

*(Analogy: you have 25 possible places to stand in a room to get the best phone reception for 3
friends at once, but you can only stand in 5 of them, and you must pick which 5.)*

---

## Part 3 — The problem: to choose well, you must know the channel, and that's expensive

To decide which ports are good, you need to know **how strong and in what phase the signal is at each
port, for each user**. This information is called **Channel State Information (CSI)**. Think of CSI as
a "signal quality map" over all the ports.

Here's the catch that the whole paper is about:

- The obvious approach is to **measure the CSI at all 25 ports** and then pick the best 5. Prior work
  basically assumes you can do this ("full CSI").
- But **measuring a port costs resources** — you have to send known test signals (**pilots**) and use
  radio hardware to read each port. Measuring *all* ports every moment is a huge overhead, and it gets
  worse as antennas get denser (more ports).
- Worse, you can only physically read the ports you actually turn on. You don't get the others for free.

So "full CSI" is unrealistic. **The real question is: can we choose good ports well while measuring
only a few of them?**

---

## Part 4 — Our idea: only look where you point, and *infer* the rest

Our agent does **not** measure all ports. Each moment it:
1. Turns on 5 ports, and gets a (slightly noisy) measurement of the channel **only at those 5**.
2. The other 20 ports are **hidden** — never directly measured this moment.

This creates a tension, which is the heart of the problem:
- You want to turn on ports you already know are good (to serve users well **now**).
- But you also want to occasionally turn on **uncertain** ports to *learn* whether they're good (so you
  can use them later).
- And you don't want to keep changing ports, because changing them has a cost (next parts).

This is a classic **explore vs. exploit** problem, and it's exactly what our method balances
automatically.

**But how can we possibly say anything about ports we never measured?** Because the physical world has
structure — two patterns we can exploit.

---

## Part 5 — The two patterns that let us guess the unseen ports

**Pattern 1 — Space: nearby ports are similar (spatial correlation).**
The ports sit very close together (closer than half a wavelength). Physics says two ports that close
have *correlated* signals: if one port is good, its neighbors are probably decent too; if one is in a
dead zone, nearby ones likely are as well. This "how similar are two ports as a function of their
distance" relationship follows a known curve (mathematically, a **Bessel function** — it starts at
"identical" for zero distance and wiggles down as distance grows). The practical upshot:

> **If I measure one port, I automatically learn something about its neighbors — without measuring
> them.** So I don't need to measure everything.

**Pattern 2 — Time: the channel changes slowly (temporal correlation).**
The signal map doesn't reset randomly every instant. From one moment to the next it changes only a
little; how fast it changes depends on movement (the **Doppler** effect). So a port I measured a moment
ago is still *mostly* valid now — but a little less trustworthy with each passing moment. We call this
**CSI aging**: knowledge decays over time.

These two patterns — *space says which ports relate to which*, *time says how fast knowledge goes
stale* — are the "physics prior" our agent uses to fill in everything it didn't measure.

---

## Part 6 — Active inference: the philosophy behind the agent

Now, how should the agent combine "serve well now," "learn about uncertain ports," and "don't move too
much"? We borrow a framework from neuroscience called **active inference**.

The one-line idea: an active-inference agent keeps a **belief** about the hidden state of the world,
and it **acts to make the world match what it prefers while reducing its own uncertainty** — both at
once, from a single principle. It doesn't have two separate rules ("a rule for getting reward" and "a
bolt-on rule for exploring"); it has *one* quantity it tries to minimize, and that quantity naturally
contains both "get what you want" and "resolve what you're unsure about."

Why this fits our problem perfectly: port selection *is* exactly "act (choose ports) to get good
service **and** to learn the hidden channel **and** at acceptable cost." Active inference gives us one
clean objective for all three. That's the conceptual contribution.

The agent has two jobs each moment: **perceive** (update its belief from the new measurement) and
**act** (choose the next ports). Let's do perception first.

---

## Part 7 — Perception: the Kalman filter (explained from scratch)

We need something that keeps a running best-guess of the *entire* channel map — including the ports we
didn't measure — and knows *how confident* it is about each part. The tool that does exactly this is
the **Kalman filter**. Let me explain what it does before saying why we use it.

**What a Kalman filter maintains: a belief = a guess + an uncertainty.**
For each user it keeps two things:
- a **mean** — its current best guess of the channel at every port (the "estimated signal map"), and
- an **uncertainty** (technically a **covariance**) — how unsure it is about each port, *and* how the
  ports relate to each other.

*(Analogy: a weather forecaster doesn't just say "20°C"; it says "20°C, and I'm quite sure" or "20°C
but very unsure." The Kalman filter carries both the number and the confidence, for every port.)*

**How it works — two alternating steps:**

1. **Predict (time passes → knowledge ages).** Before each new moment, the filter rolls its guess
   forward using the "channel changes slowly" rule, and — crucially — it **increases its uncertainty**,
   because time has passed and old measurements are now less trustworthy. Ports it hasn't measured in a
   while become "fuzzier." *This is CSI aging, captured automatically.*

2. **Update (a measurement arrives → sharpen).** When it measures the 5 active ports, it corrects its
   guess toward what it just saw, and *reduces* its uncertainty there. And here's the magic: because it
   knows the **spatial correlation** (which ports relate to which), **measuring those 5 ports also
   sharpens its guess about their unmeasured neighbors.** Information spreads to ports it never touched.

**Why we use it here (the "so we use it for that" part):**
Because the Kalman filter (a) keeps a full guess of *all* ports including unseen ones, (b) automatically
grows uncertainty as knowledge ages, and (c) spreads each measurement to correlated neighbors — it is
*exactly* the machine we need to turn "measure 5 ports" into "an informed guess about all 25, with
honest confidence levels." We checked that its stated confidence actually matches its real error (it's
**calibrated** — when it says it's 90% sure, it's right about 90% of the time).

One nice bonus: the filter is **exact and cheap** here because our channel model is "linear-Gaussian"
(nice mathematical structure) — no training, no neural network, just running formulas.

---

## Part 8 — Decision: Expected Free Energy, the single score with three parts

Now the agent must **choose** the next 5 ports. It scores every candidate set of ports with one number,
called the **Expected Free Energy (EFE)**. Lower is better. That number is the sum of three things —
and this is the core of the method, so explain each plainly:

**Term 1 — "Will this give good service now?" (pragmatic value / expected rate).**
Using its current belief, the agent predicts the data rate the users would get if it activated this
port set. High predicted rate → good. Importantly, when the agent is *unsure* about a port, this term
automatically becomes *cautious* (it doesn't over-trust a shaky guess). *(This uses a signal-shaping
method called robust MMSE precoding — it aims each user's signal and, when unsure, "backs off" so it
doesn't cause interference. You can just call it "smart aiming that's careful when unsure.")*

**Term 2 — "Will this teach me something?" (epistemic value / information gain).**
The agent also values turning on ports it's *uncertain* about, because measuring them will shrink its
uncertainty — not just for those ports but for their correlated neighbors too. This is a **built-in
curiosity term.** It's what makes the agent occasionally probe unknown ports instead of blindly reusing
known-good ones. In ordinary methods, "exploration" is a hand-tuned hack; here it falls out of the math.

**Term 3 — "How much will this cost me to move?" (switching cost).**
Changing which ports are active is not free — the antenna hardware has to reconfigure, which costs time
and energy. So every port you *change* from last moment adds a penalty. This makes the agent **stable**:
it doesn't thrash between port sets for tiny gains.

**The whole decision rule:**
> pick the port set that **maximizes (good service + information gained) − (cost of moving)**.

One number, three drivers, balanced automatically. That unification is the paper's technical heart.

---

## Part 9 — Two practical touches: making it fast, and keeping CSI fresh

**Making it fast (greedy selection).** Choosing 5 ports out of 25 has ~53,000 possible combinations —
too many to check each moment. Instead the agent builds the set **one port at a time**: start empty,
add the single port that improves the score the most, repeat until you have 5. This is called **greedy**
selection. It works remarkably well because the "information gained" term has **diminishing returns**
(the 5th port teaches you less than the 1st) — a property that mathematically guarantees greedy is
near-optimal. In practice it's within a fraction of a percent of the best possible, and runs in about
**20 milliseconds** instead of seconds.

**Keeping CSI fresh (observe-then-precode).** A subtle but important ordering trick. Within one moment:
first the agent *chooses* the 5 ports (using its aged guess), then it **sends a quick test signal on
those 5 ports to measure them freshly**, and only *then* aims the users' data using that fresh
measurement. So the ports it actually serves on always have up-to-date CSI, not stale guesses. This one
choice is what makes performance stay high even when the channel changes quickly.

*(Note to preempt a question: "isn't that full CSI then?" No — we still only ever measure the 5 active
ports. The other 20 are still guessed. We just make sure the 5 we serve on are measured fresh.)*

---

## Part 10 — Does it work? Results and the baselines we compare against

To judge it, we compare against reference methods on the same channels:

- **Genie (the cheater):** knows the true channel at *all* ports. This is the **best-possible ceiling**,
  not a fair opponent — it's there to show how close we get.
- **Naive:** same limited access as us, but with *no* smart inference (it just reuses last-known good
  ports). This isolates what our belief actually buys.
- **DRL (a trained neural network):** a learning-based competitor in the spirit of prior work.
- **Bandit:** a model-free learner that must *sample* every port to learn it (it can't infer neighbors).

**Headline results (say these plainly):**
- We reach **84–89% of the all-knowing genie's rate while measuring only 20% of the ports** — and with
  **zero training** (no dataset, no learning phase; it works from moment one).
- We **beat the naive, DRL, and bandit** competitors on the overall objective (rate minus switching
  cost), largely because we **move the antenna far less** — the model-based belief lets us "lock on" to
  good ports instead of constantly re-exploring.
- We're **robust**: performance stays flat as the channel speeds up (Doppler), and if our assumed
  physics model is wrong, the agent can **learn the spatial correlation online** and recover.

*(Honest framing for the genie: we do not claim to beat an all-knowing genie on rate — that's
impossible. We approach it. Where our earlier figures said we "beat the genie," that was against a genie
that ignores movement cost; we're fixing that comparison to be fully fair.)*

---

## Part 11 — Honest limitations & the takeaway

**Limitations to mention (shows maturity):**
- We assume a specific, idealized channel model (though we can learn part of it).
- We use fully-digital signal aiming; a cheaper "hybrid" hardware version is future work.
- The switching-cost model and the exact antenna hardware type (electronic "pixel" switching vs.
  physically moving elements) affect how big the movement cost really is — something we're clarifying.

**One-sentence takeaway to end on:**
> "You don't need to measure everything. By combining a physics-based belief with a single objective
> that balances serving, learning, and moving, an antenna can pick great ports while looking at only a
> fraction of them — no training required."

---

## Cheat-sheet: jargon → plain word (in case you're asked)
- **Port** → a candidate position/spot for the antenna.
- **CSI** → the signal-quality map (how strong/what phase at each port).
- **Full / partial CSI** → knowing all ports vs. only the few you measured.
- **Fading** → signal is strong in some spots, dead in others.
- **Spatial correlation** → nearby ports behave similarly.
- **Temporal correlation / CSI aging** → the map changes slowly; old info goes stale.
- **Doppler** → how fast the channel changes (movement).
- **Belief (mean + covariance)** → best guess + how confident, for every port.
- **Kalman filter** → the machine that keeps and updates that belief.
- **Precoding / MMSE** → aiming each user's signal so they don't interfere.
- **Expected Free Energy (EFE)** → the single score = good service + info gained − moving cost.
- **Epistemic value** → the curiosity term (value of reducing uncertainty).
- **Greedy selection** → build the port set one best-port-at-a-time (fast, near-optimal).
- **Genie** → an all-knowing reference (upper bound), not a fair opponent.
