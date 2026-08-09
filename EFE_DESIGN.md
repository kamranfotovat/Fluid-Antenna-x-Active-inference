# Expected Free Energy (EFE) Design — Port Selection under Partial CSI

Companion to `RESEARCH_PLAN.md` (version 1: implicit sensing, fully-digital robust-MMSE beamforming).
This document specifies the **decision rule**: how the agent scores and selects a port set each slot.

Notation kept in plain text so it renders in any markdown preview.

---

## 1. Setup & interfaces

- `N` candidate ports, activate `M`, serve `K` users. Slot index `t`.
- Per-user channel `h_k in C^N` (over all ports). Belief is Gaussian:
  `q(h_k) = CN(mu_k, Sigma_k)` — mean `mu_k in C^N`, covariance `Sigma_k in C^{NxN}`.
- Action = port set `S` with `|S| = M`. Selection matrix `P_S in {0,1}^{MxN}` picks the active rows.
- Active-port belief: `g_k = P_S h_k`, with `mean = P_S mu_k`, `cov = P_S Sigma_k P_S^H`.
- Beamformer `W = [w_1,...,w_K] in C^{MxK}` = **robust MMSE** computed from the active-port belief.
- Estimation noise on an observed port: variance `sigma_e^2`. Data noise: `sigma^2`.

The belief `(mu_k, Sigma_k)` is produced by the Kalman filter (Section 6). The EFE below consumes it.

---

## 2. The objective (what we minimize)

Active inference selects the action that **minimizes Expected Free Energy**. For a candidate port set `S`:

```
G(S) = - alpha * PragmaticValue(S)      (want high rate)
       - beta  * EpistemicValue(S)      (want to reduce channel uncertainty)
       + SwitchingCost(S)               (want to avoid churn)
```

Select `S* = argmin_S G(S)`, subject to `|S| = M`.

- `alpha, beta >= 0` are weights. In canonical AIF both terms are in nats and `alpha = beta = 1`.
  We **expose `beta`** as the exploration knob for an ablation (beta = 0 recovers a pure-exploitation,
  greedy-on-belief baseline).
- **Units caveat (design detail):** PragmaticValue is in bits (rate), EpistemicValue in nats/bits (info).
  Fix one convention (e.g., convert info gain to bits via `log2`) so the two terms are commensurable,
  then `alpha, beta` are dimensionless trade-off weights.

The three terms are defined next.

---

## 3. Pragmatic value — expected (robust-MMSE) sum-rate

Prefer outcomes with high sum-rate. Practical form = expected achievable sum-rate under the belief,
using the **same robust-MMSE precoder we will actually apply** (consistency — closes the loop):

```
PragmaticValue(S) = E_q[ sum_k R_k(S; h) ] ~= sum_k Rbar_k(S; mu, Sigma)
```

where `Rbar_k` is an **uncertainty-aware rate** built from an effective SINR that accounts for
imperfect CSI. Because the precoder uses the *estimate* `P_S mu_k` (not the true channel), the
residual after MMSE contains an estimation-error term that scales with the belief covariance:

```
SINR_k(S) ~= |mu_k^H P_S^H w_k|^2
             ------------------------------------------------------------------
             sum_{j != k} |mu_k^H P_S^H w_j|^2  +  e_k(Sigma)  +  sigma^2

  e_k(Sigma) = residual interference + self-noise from CSI error, grows with P_S Sigma_k P_S^H
```

Then `Rbar_k = log2(1 + SINR_k(S))` and `PragmaticValue(S) = sum_k Rbar_k`.

**Why this matters:** `Sigma` enters the *rate itself*. When the agent is unsure about the active
ports, the effective SINR drops -> the pragmatic value is automatically conservative. This is exactly
the "robust MMSE" behavior and it makes the selection honest under partial CSI.

> Note: the exact `E[log2(1+SINR)]` is intractable; use the plug-in / lower-bound form above.
> Specify the chosen bound in the paper.

---

## 4. Epistemic value — expected information gain

Prefer actions that **reduce uncertainty** about the channel. For Gaussian beliefs, the expected
information gain from observing the ports in `S` is the drop in differential entropy = the mutual
information between `h_k` and the observation `y_k = P_S h_k + noise`:

```
EpistemicValue(S) = sum_k I_k(S)

I_k(S) = log det( I_M + (1/sigma_e^2) * P_S Sigma_k P_S^H )     (complex circular Gaussian)
```

Equivalently `I_k(S) = log|Sigma_prior| - log|Sigma_post(S)|` up to the observation model.

**Key properties:**
- Ports with **large uncertainty** (stale, per the aging predict step) give high info gain.
- Because `Sigma_k` carries the **spatial correlation `R`**, observing one port also lowers
  uncertainty on its correlated neighbors -> info gain is *shared*, so you don't need to probe everything.
- `I_k(S)` is **monotone submodular** in `S` -> enables the greedy guarantee (Section 5).

---

## 5. Switching cost & the combined rule

```
SwitchingCost(S) = eta_sw * e_sw * |S XOR S_{t-1}|      (symmetric difference = # ports changed)
```

Combined per-slot objective:

```
minimize_S  G(S) = - alpha * sum_k Rbar_k(S)
                   - beta  * sum_k I_k(S)
                   + eta_sw * e_sw * |S XOR S_{t-1}|
```

### Greedy submodular selection (tractable, low-latency)
`choose M of N` is combinatorial (C(25,5) ~ 53k). Instead build `S` greedily:

```
A <- {}                                  # partial active set
for m = 1..M:
    for each port n not in A:
        marginal(n) =  alpha * dRbar(n | A)      # pragmatic gain of adding n
                     + beta  * dI(n | A)         # epistemic gain (rank-1 log-det update, cheap)
                     - dSwitch(n | A)            # extra switching if n not in S_{t-1}
    n* = argmax_n marginal(n)
    A <- A + {n*}
S_t <- A
```

- **Complexity:** `O(N * M * c)`, `c` = cost of the marginal (log-det update is rank-1/cheap;
  the MMSE re-solve for the pragmatic marginal is the main cost — can be warm-started/approximated).
- **Guarantee:** the epistemic part is monotone submodular -> greedy achieves `(1 - 1/e)` of optimal
  on that component; combined with the modular switching cost this is a strong, principled heuristic.
- **Latency:** this is what disarms the "AIF is slow" objection — report ms/slot vs the DQN forward pass.

---

## 6. Belief update interface (Kalman) — what feeds the EFE

Each slot, before selection, run the **predict** step; after serving/observing, run **update**.

```
Predict (time passes -> aging):
    mu_k    <- rho * mu_k
    Sigma_k <- rho^2 * Sigma_k + (1 - rho^2) * beta_k * R        # uncertainty grows on all ports

Select S_t by minimizing G(S) using predicted (mu_k, Sigma_k).   # Sections 2-5

Update (observe activated ports S_t, noisy):
    standard Kalman update of (mu_k, Sigma_k) using y_k = P_S h_k + CN(0, sigma_e^2 I)
    -> observed ports sharpen; neighbors sharpen via off-diagonal Sigma (from R)
```

So the EFE **selection** uses the *predicted* (aged) belief; the **update** refines it after acting.

---

## 7. Myopic vs. planning (scope note)
- We use **one-slot (myopic) EFE** per slot: pick `S_t` optimizing the current-slot objective.
- Dynamics are NOT ignored — the temporal model `rho` enters through the predict step, so uncertainty
  and staleness are anticipated. We simply don't roll the policy forward multiple slots.
- **Full multi-step AIF planning** (policies over a horizon) is a natural extension / journal item.

---

## 8. Open design details to lock during implementation
1. **Unit reconciliation** between pragmatic (bits) and epistemic (nats) — pick `log2` throughout.
2. **Complex-Gaussian constants** in the info-gain / entropy (circular complex vs real factor).
3. **Pragmatic rate bound** — exact `E[log2(1+SINR)]` is intractable; choose and state the lower bound.
4. **Per-user aggregation** — sum rates and sum info gains across `k` (K independent Kalman filters).
5. **Pragmatic marginal cost** — the MMSE re-solve inside greedy; decide exact vs approximate/warm-start.
6. **Weights (alpha, beta)** — default balanced; sweep `beta` for the exploration ablation (Fig. D).

---

## 9. What each term buys the paper
- **Pragmatic (Sec. 3):** high throughput + honest robustness to partial CSI (uses `Sigma`).
- **Epistemic (Sec. 4):** principled active sensing / re-activation of stale ports (uses `Sigma` + `R`).
- **Switching (Sec. 5):** stability, matches the anchor papers' cost model.
- **Greedy (Sec. 5):** tractability + `(1-1/e)` guarantee + low latency.
All three are unified in **one objective** driven by a single belief `(mu, Sigma)` — that unification
is the contribution.
