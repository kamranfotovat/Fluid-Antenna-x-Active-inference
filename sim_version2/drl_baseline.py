r"""
DRL baseline for FAS port selection -- a representative learning competitor (in the spirit of
Paper 3's Transformer-DQN), so we compare against the literature, not just genie/naive.

Policy: per-port features -> Transformer self-attention over ports (captures inter-port
interference/correlation, like Paper 3's Transformer) -> per-port score -> select top-M.
Trained by policy gradient (REINFORCE with a moving baseline) to maximise the SAME Eq.7
objective the AIF agent uses: sum-rate - eta_sw * switching.

FAIRNESS GATE: under FULL CSI the trained policy must approach the genie. Only then is showing
its collapse under partial/stale CSI a fair comparison (not just a badly-trained net).

Two evaluation regimes (same trained weights):
  full CSI  : policy sees the true channel on ALL ports, selects, precodes on full CSI (their setting).
  partial   : policy sees only last-known/stale CSI (no generative inference), selects, then
              observe-then-precode on fresh pilots of the chosen ports (realistic deployment).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from precoding import mmse_precoder, sinr_and_rates


# --------------------------------------------------------------------------- features
def port_features(h_est, prev_mask, K):
    """Per-port features from a channel estimate h_est (K,N) complex and prev-selection mask (N,).
    Returns (N, F) with F = 2K + 2: [Re(h_k), Im(h_k) for k, aggregate power, prev-selected flag]."""
    re = h_est.real
    im = h_est.imag
    pw = np.sum(np.abs(h_est) ** 2, axis=0, keepdims=True)
    feats = np.concatenate([re, im, pw, prev_mask[None, :]], axis=0).T   # (N, 2K+2)
    return feats.astype(np.float32)


F_DIM = lambda K: 2 * K + 2


# --------------------------------------------------------------------------- policy net
class PortPolicy(nn.Module):
    def __init__(self, K, d=64, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(F_DIM(K), d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):                       # x: (B, N, F) -> (B, N) logits
        return self.head(self.tr(self.embed(x))).squeeze(-1)


def sample_topM(logits, M, greedy=False):
    """Select M ports without replacement. Returns (S:(B,M) long, logprob:(B,)).
    greedy=True -> deterministic top-M (for evaluation)."""
    B, N = logits.shape
    if greedy:
        S = torch.topk(logits, M, dim=1).indices
        return S, torch.zeros(B, device=logits.device)
    avail = torch.ones(B, N, dtype=torch.bool, device=logits.device)
    logp = torch.zeros(B, device=logits.device)
    chosen = []
    for _ in range(M):
        masked = logits.masked_fill(~avail, -1e9)
        p = torch.softmax(masked, dim=1)
        idx = torch.multinomial(p, 1).squeeze(1)
        logp = logp + torch.log(p.gather(1, idx[:, None]).squeeze(1) + 1e-12)
        avail = avail.scatter(1, idx[:, None], False)
        chosen.append(idx)
    return torch.stack(chosen, dim=1), logp


# --------------------------------------------------------------------------- reward (numpy)
def objective_of_selection(h_true, S, prev_S, sigma2, eta_sw, e_sw, P=1.0):
    """Realized Eq.7 objective for selecting set S (full-CSI precoder on h_true[S])."""
    idx = list(S)
    W = mmse_precoder(h_true[:, idx].T, P=P, sigma2=sigma2)
    rate = float(sinr_and_rates(h_true[:, idx].T, W, sigma2)[1].sum())
    sw = 0 if prev_S is None else len(set(idx) ^ set(prev_S))
    return rate - eta_sw * e_sw * sw, rate, sw


def _mask(prev_S, N):
    m = np.zeros(N, dtype=np.float32)
    if prev_S is not None:
        m[list(prev_S)] = 1.0
    return m


# --------------------------------------------------------------------------- training (REINFORCE)
def train_policy(sims, K, M, sigma2=0.03, eta_sw=1.0, e_sw=1.0, device=None,
                 iters=400, L=16, lr=1e-3, gamma=0.9, seed=0, log_every=100, snapshots=None):
    """Train the port-selection policy under FULL CSI by policy gradient. `sims` is a list of
    ChannelSimulator (batch = len(sims)); reused each iter (fresh noise via reset)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    B, N = len(sims), sims[0].N
    policy = PortPolicy(K).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    hist = []
    snaps = {} if snapshots else None
    for it in range(iters):
        for s in sims:
            s.reset()
        Hs = [s.generate(L) for s in sims]                       # each (L,K,N)
        prev = [None] * B
        logps, rews = [], []
        for t in range(L):
            feats = np.stack([port_features(Hs[b][t], _mask(prev[b], N), K) for b in range(B)])
            logits = policy(torch.from_numpy(feats).to(device))
            S, logp = sample_topM(logits, M)
            Snp = S.cpu().numpy()
            r = np.empty(B, np.float32)
            for b in range(B):
                obj, _, _ = objective_of_selection(Hs[b][t], Snp[b].tolist(), prev[b], sigma2, eta_sw, e_sw)
                r[b] = obj; prev[b] = Snp[b].tolist()
            logps.append(logp); rews.append(torch.from_numpy(r).to(device))
        # discounted reward-to-go with a per-timestep baseline
        R = torch.zeros(B, device=device); returns = []
        for t in reversed(range(L)):
            R = rews[t] + gamma * R; returns.insert(0, R.clone())
        returns = torch.stack(returns); logps = torch.stack(logps)
        adv = returns - returns.mean(dim=1, keepdim=True)
        loss = -(adv.detach() * logps).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        mean_obj = torch.stack(rews).mean().item()
        hist.append(mean_obj)
        if snapshots and it in snapshots:
            import copy
            snaps[it] = copy.deepcopy(policy).eval()
        if it % log_every == 0:
            print(f"   iter {it:4d}: mean objective/slot = {mean_obj:.2f}")
    return (policy.eval(), hist, snaps) if snapshots else (policy.eval(), hist)


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_fullcsi(policy, H, M, sigma2=0.03, eta_sw=1.0, e_sw=1.0, device=None):
    """Policy sees true full CSI, selects greedily (top-M), precodes on full CSI (their setting)."""
    device = device or next(policy.parameters()).device
    T, K, N = H.shape
    rate = np.zeros(T); switch = np.zeros(T); prev = None
    for t in range(T):
        x = torch.from_numpy(port_features(H[t], _mask(prev, N), K)[None]).to(device)
        S = sample_topM(policy(x), M, greedy=True)[0][0].cpu().numpy().tolist()
        _, r, sw = objective_of_selection(H[t], S, prev, sigma2, eta_sw, e_sw)
        rate[t] = r; switch[t] = sw; prev = S
    return dict(rate=rate, switch=switch)


@torch.no_grad()
def eval_partial(policy, H, M, sigma_e2, rng, sigma2=0.03, eta_sw=1.0, e_sw=1.0, device=None):
    """Realistic deployment: policy only has last-known (stale) CSI -- no generative inference.
    It selects from stale features, then observe-then-precode on fresh pilots of the chosen ports."""
    device = device or next(policy.parameters()).device
    T, K, N = H.shape
    est = np.zeros((K, N), dtype=complex)                        # last-known estimate (stale)
    rate = np.zeros(T); switch = np.zeros(T); prev = None
    for t in range(T):
        x = torch.from_numpy(port_features(est, _mask(prev, N), K)[None]).to(device)
        S = sorted(sample_topM(policy(x), M, greedy=True)[0][0].cpu().numpy().tolist())
        idx = list(S)
        y = H[t][:, idx] + np.sqrt(sigma_e2 / 2) * (rng.standard_normal((K, len(idx)))
                                                    + 1j * rng.standard_normal((K, len(idx))))
        est[:, idx] = y                                          # refresh only observed ports
        W = mmse_precoder(y.T, P=1.0, sigma2=sigma2)
        rate[t] = float(sinr_and_rates(H[t][:, idx].T, W, sigma2)[1].sum())
        switch[t] = 0 if prev is None else len(set(idx) ^ set(prev)); prev = idx
    return dict(rate=rate, switch=switch)
