"""
PPO Model for HPRL Meta-Policy Training
======================================

This file mirrors the style and layout of *ddpg_model.py* so the rest of
your codebase can be re-used with minimal edits.  Key differences:

•  Actor now produces **mean + log σ** for a diagonal Gaussian and uses
   tanh to squash actions into [-1, 1].  
•  Critic is replaced by a **Value network** V(s).  
•  PPO-specific on-policy logic is implemented in `PPOAgent`
   (roll-out buffer, GAE-λ, clipped-surrogate optimisation).

All hyper-parameters (γ = 0.95, λ = 0.95, εclip = 0.10, etc.) match the
official HPRL repository.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional, Dict, Any, List


# ──────────────────────────────────────────────────────────────
#   CNN trunk (identical to ddpg_model.Actor / Critic)
# ──────────────────────────────────────────────────────────────
class ConvTrunk(nn.Module):
    """8×8×8 grid  → 512-D feature vector via 3×Conv + GAP."""
    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),         # 4×4
            nn.Flatten()
        )
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x) if x.dim() == 4 else self.net(x.unsqueeze(0))
# ──────────────────────────────────────────────────────────────
#   Actor network  πθ(a|s)
# ──────────────────────────────────────────────────────────────
class Actor(nn.Module):
    """
    Maps Karel states to latent program embeddings.

    Output: mean μ and log-std ϖ for a 64-D diagonal Gaussian; the final
    action is tanh-squashed to stay in (-1, 1).
    """
    def __init__(
        self,
        state_shape: Tuple[int, int, int] = (8, 8, 8),
        latent_dim: int = 64,
        hidden_dims: List[int] = [256, 256],
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        dropout: float = 0.0
    ):
        super().__init__()
        self.trunk = ConvTrunk(state_shape[2])
        trunk_out = 32 * 4 * 4                      # 512

        layers: List[nn.Module] = []
        prev_dim = trunk_out
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        self.mlp = nn.Sequential(*layers)

        self.mu_head = nn.Linear(prev_dim, latent_dim)
        self.log_std_head = nn.Linear(prev_dim, latent_dim)
        self.log_std_min, self.log_std_max = log_std_bounds

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Accept NHWC or NCHW
        if state.dim() == 4 and state.shape[-1] == 8:
            state = state.permute(0, 3, 1, 2)
        h = self.trunk(state)
        h = self.mlp(h)
        mu = torch.tanh(self.mu_head(h))                  # in (-1,1)
        log_std = torch.clamp(self.log_std_head(h),
                              self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std


# ──────────────────────────────────────────────────────────────
#   Value network  Vϕ(s)
# ──────────────────────────────────────────────────────────────
class ValueNet(nn.Module):
    """State-value function."""
    def __init__(
        self,
        state_shape: Tuple[int, int, int] = (8, 8, 8),
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.0
    ):
        super().__init__()
        self.trunk = ConvTrunk(state_shape[2])
        trunk_out = 32 * 4 * 4

        layers: List[nn.Module] = []
        prev_dim = trunk_out
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 4 and state.shape[-1] == 8:
            state = state.permute(0, 3, 1, 2)
        h = self.trunk(state)
        return self.net(h).squeeze(-1)           # (B,)


# ──────────────────────────────────────────────────────────────
#   Roll-out buffer for PPO
# ──────────────────────────────────────────────────────────────
class RolloutBuffer:
    """Stores one full on-policy trajectory until an update."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.states, self.actions, self.logps = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def add(self, state, action, logp, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.logps.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def is_full(self) -> bool:
        return len(self.rewards) >= self.capacity

    def tensors(self, device: torch.device):
        s = torch.FloatTensor(np.array(self.states)).to(device)
        a = torch.FloatTensor(np.array(self.actions)).to(device)
        logp = torch.FloatTensor(self.logps).to(device)
        r = torch.FloatTensor(self.rewards).to(device)
        d = torch.FloatTensor(self.dones).to(device)
        v = torch.FloatTensor(self.values).to(device)
        return s, a, logp, r, d, v


# ──────────────────────────────────────────────────────────────
#   PPO Agent (on-policy, clipped surrogate)
# ──────────────────────────────────────────────────────────────
class PPOAgent:
    """
    Handles action selection, roll-out storage, advantage estimation,
    and PPO updates. Hyper-parameters follow the HPRL repo.
    """
    def __init__(
        self,
        state_shape: Tuple[int, int, int] = (8, 8, 8),
        latent_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.10,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 2048,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        device: str = "auto"
    ):
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.actor = Actor(state_shape, latent_dim).to(self.device)
        self.critic = ValueNet(state_shape).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr, eps=1e-5
        )

        # Hyper-parameters
        self.gamma, self.lam = gamma, gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.max_grad_norm = max_grad_norm

        # Roll-out settings
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.buffer = RolloutBuffer(rollout_steps)

        self.training = True
        self.total_updates = 0

    # ── sampling helpers ──────────────────────────────────────
    @staticmethod
    def _tanh_sample(mu: torch.Tensor, std: torch.Tensor, deterministic: bool):
        dist = torch.distributions.Normal(mu, std)
        z = mu if deterministic else dist.rsample()
        a = torch.tanh(z)
        logp = dist.log_prob(z).sum(-1) - torch.log(torch.clamp(1.0 - a.pow(2), 1e-6)).sum(-1)
        return a, logp

    # ── public API ────────────────────────────────────────────
    def select_action(self, state: np.ndarray, add_noise: bool = True):
        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, std = self.actor(s_t)
        a_t, logp_t = self._tanh_sample(mu, std, deterministic=not add_noise)
        v_t = self.critic(s_t)
        return a_t.detach().cpu().numpy()[0], logp_t.item(), v_t.item()

    def store(self, state, action, logp, reward, done, value):
        self.buffer.add(state, action, logp, reward, done, value)

    # ── PPO update ───────────────────────────────────────────
    def update(self):
        if not self.buffer.is_full():
            return {}  # not enough data yet

        states, actions, old_logps, rewards, dones, values = self.buffer.tensors(self.device)
        self.buffer.reset()

        # GAE-λ advantages & returns
        adv, ret, gae = torch.zeros_like(rewards), torch.zeros_like(rewards), 0.0
        for t in reversed(range(rewards.size(0))):
            next_v = values[t + 1] if t + 1 < rewards.size(0) else 0.0
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_v - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
            ret[t] = adv[t] + values[t]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO optimisation
        n = rewards.size(0)
        idx = np.arange(n)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                s_mb, a_mb, old_mb, adv_mb, ret_mb = states[mb], actions[mb], old_logps[mb], adv[mb], ret[mb]

                mu, std = self.actor(s_mb)
                dist = torch.distributions.Normal(mu, std)
                z = torch.atanh(torch.clamp(a_mb, -0.999, 0.999))
                logp = dist.log_prob(z).sum(-1) - torch.log(torch.clamp(1 - a_mb.pow(2), 1e-6)).sum(-1)
                ratio = torch.exp(logp - old_mb)

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -dist.entropy().sum(-1).mean()
                value_loss = F.mse_loss(self.critic(s_mb), ret_mb)

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         self.max_grad_norm)
                self.optimizer.step()

        self.total_updates += 1
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_loss.item()),
            "updates": self.total_updates
        }

    # ── mode switches ────────────────────────────────────────
    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()

    # ── checkpoint helpers ───────────────────────────────────
    def save_models(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optim": self.optimizer.state_dict(),
            "updates": self.total_updates
        }, path)

    def load_models(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.total_updates = ckpt.get("updates", 0)


# ──────────────────────────────────────────────────────────────
#   Quick smoke-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("✓  PPO model smoke-test")

    agent = PPOAgent()  # only to access fields
    dummy_state = np.random.randn(8, 8, 8).astype(np.float32)
    a, lp, v = agent.select_action(dummy_state)
    print(f"  action   : shape={a.shape}, min={a.min():.2f}, max={a.max():.2f}")
    print(f"  log-prob : {lp:.4f}")
    print(f"  value    : {v:.4f}")

    # fill buffer quickly for one update
    for _ in range(agent.rollout_steps):
        a, lp, v = agent.select_action(dummy_state)
        agent.store(dummy_state, a, lp, reward=0.1, done=False, value=v)
    stats = agent.update()
    print("  update() :", stats)