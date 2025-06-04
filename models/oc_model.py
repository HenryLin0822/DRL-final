"""
Option‑Critic Networks for HPRL Meta‑Policy
==========================================
This module contains a *minimal but complete* implementation of an **Option‑Critic**
agent that outputs 64‑dimensional latent vectors (decoded by the VAE into Karel
programmes).  It is purposely designed to slot into the `oc_trainer.py` file I
shared earlier and uses the same hyper‑parameter names as your existing DDPG
implementation so you can swap agents with *one* import change.

Main classes
------------
* **FeatureExtractor** – light CNN that turns an 8×8×8 Karel state into a flat
  feature vector.
* **OptionCriticNet** – a *single* network with three heads:
    1. `q_options(s)`            → value of each option.
    2. `intra_mean(s)` & `intra_log_std(s)` (both shaped *(N_opts, latent_dim)*)
       → parameters of a Gaussian intra‑option policy.
    3. `termination_logits(s)`   → logits for β(s, o)   (option termination).
* **OptionCriticAgent** – wraps the network, replay buffer, and the full
  update rule (critic‑loss, policy‑loss, termination‑loss) from Bacon *et al.*

Assumptions
-----------
* Initiation set is the full state‑space (standard OC simplification).
* Options share the same feature extractor – only heads are option‑specific.
* Latent embeddings are squashed with **tanh** to keep them in [‑1, 1].  This
  matches your current VAE expectations.
* A continuous OU noise process is added to the *mean* during exploration when
  `eps_greedy == False` (so you can still ε‑greedy over the option Id).

If you need to customise anything – e.g. separate target networks, twin Q‑net,
entropy regulation of the Gaussian – you can start from this scaffold.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Dict, Any, Optional

# -------------------------------------------------------------
#  Feature extractor (shared by all heads)                     
# -------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """Tiny CNN identical to the one used in your DDPG actor/critic."""

    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.out_dim = 32 * 4 * 4  # 512
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Accept either NHWC or NCHW
        if s.dim() == 4 and s.shape[-1] == 8:
            s = s.permute(0, 3, 1, 2)
        return self.net(s.float())

# -------------------------------------------------------------
#  Option–Critic master network                                 
# -------------------------------------------------------------
class OptionCriticNet(nn.Module):
    """One network → 3 heads (Q, π, β)."""

    def __init__(
        self,
        n_options: int = 4,
        latent_dim: int = 64,
        feature_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_options = n_options
        self.latent_dim = latent_dim

        # Shared trunk
        self.feature_extractor = FeatureExtractor()
        trunk_out = feature_dim
        if trunk_out != self.feature_extractor.out_dim:
            trunk_out = self.feature_extractor.out_dim

        # Q‑option head  (critic)
        self.q_head = nn.Linear(trunk_out, n_options)

        # Intra‑option policy heads (actor)
        self.intra_mean = nn.Linear(trunk_out, n_options * latent_dim)
        self.intra_log_std = nn.Linear(trunk_out, n_options * latent_dim)

        # Termination head  β(s,o)
        self.beta_head = nn.Linear(trunk_out, n_options)

        # Log‑std bounds (follow SAC style)
        self.LOG_STD_MIN, self.LOG_STD_MAX = -5, 2
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    # ------------------------------------------------------------------
    def forward(self, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns a dict with *all* heads (used for efficient batch updates)."""
        phi = self.feature_extractor(s)
        q_opts = self.q_head(phi)                                # (B, N)

        mean = self.intra_mean(phi).view(-1, self.n_options, self.latent_dim)
        log_std = self.intra_log_std(phi).view(-1, self.n_options, self.latent_dim)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        beta_logits = self.beta_head(phi)                        # (B, N)
        return {"q": q_opts, "mean": mean, "log_std": log_std, "beta_logits": beta_logits}

    # ------------------------------------------------------------------
    def sample_action(self, s: torch.Tensor, option: torch.Tensor,
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or return mean) latent embedding for a *given* option id."""
        out = self.forward(s)
        idx = option.long()                                     # (B,)
        batch = torch.arange(len(idx), device=s.device)

        mean = out["mean"][batch, idx]                          # (B, latent_dim)
        if deterministic:
            return torch.tanh(mean), torch.zeros_like(mean)

        std = out["log_std"][batch, idx].exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        logp = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, logp.sum(1, keepdim=True)

    # ------------------------------------------------------------------
    def termination_prob(self, s: torch.Tensor) -> torch.Tensor:
        """β(s, o) for **all** options (after sigmoid)."""
        return torch.sigmoid(self.forward(s)["beta_logits"])    # (B, N)

# -------------------------------------------------------------
#  Helper classes (OU noise & Replay)                            
# -------------------------------------------------------------
class OUNoise:
    def __init__(self, size: int, sigma: float = 0.2, theta: float = 0.15, dt: float = 1e-2):
        self.size, self.sigma, self.theta, self.dt = size, sigma, theta, dt
        self.reset()

    def reset(self):
        self.x_prev = np.zeros(self.size)

    def __call__(self):
        dx = self.theta * -self.x_prev * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.x_prev += dx
        return self.x_prev

class ReplayBuffer:
    """Stores (s, o, a, r, s′, done, β) tuples."""
    def __init__(self, capacity: int = 100000):
        self.buf = deque(maxlen=capacity)

    def add(self, *exp):
        self.buf.append(tuple(exp))

    def sample(self, batch: int):
        import random, torch
        batch = random.sample(self.buf, batch)
        s, o, a, r, sn, d, beta = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(o),
            torch.FloatTensor(np.array(a)),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(np.array(sn)),
            torch.FloatTensor(d).unsqueeze(1),
            torch.FloatTensor(beta).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buf)
    def size(self):
        return len(self.buf)

# -------------------------------------------------------------
#  Option‑Critic Agent                                           
# -------------------------------------------------------------
class OptionCriticAgent:
    def __init__(
        self,
        state_shape: Tuple[int, int, int] = (8, 8, 8),
        n_options: int = 4,
        latent_dim: int = 64,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        beta_lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        ou_sigma: float = 0.2,
        device: str = "auto",
    ):
        self.state_shape = state_shape
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        self.n_options, self.latent_dim = n_options, latent_dim
        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size

        self.net = OptionCriticNet(n_options, latent_dim).to(self.device)
        self.target = OptionCriticNet(n_options, latent_dim).to(self.device)
        self.target.load_state_dict(self.net.state_dict())

        # Separate optimisers for each head so you can tune them differently.
        self.opt_q = torch.optim.Adam(self.net.q_head.parameters(), lr=critic_lr)
        self.opt_pi = torch.optim.Adam(list(self.net.intra_mean.parameters()) + list(self.net.intra_log_std.parameters()), lr=actor_lr)
        self.opt_beta = torch.optim.Adam(self.net.beta_head.parameters(), lr=beta_lr)

        # Replay & exploration
        self.replay = ReplayBuffer(buffer_size)
        self.ou_noise = OUNoise(latent_dim, sigma=ou_sigma)
        self.training = True

        # Stats
        self.stats: Dict[str, float] = {}

    # ---------------- interact -----------------------------------------
    def select_option(self, s: np.ndarray, eps: float = 0.1) -> int:
        """ϵ‑greedy over Q‑options."""
        if np.random.rand() < eps:
            return np.random.randint(self.n_options)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.net.forward(s_t)["q"]
        return int(q.argmax(1).item())

    def select_action(self, s: np.ndarray, option: int, deterministic: bool = False) -> np.ndarray:
        s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a, _ = self.net.sample_action(s_t, torch.tensor([option], device=self.device), deterministic)
        a = a.cpu().numpy()[0]
        if self.training and not deterministic:
            a += self.ou_noise()
            a = np.clip(a, -1., 1.)
        return a

    def option_termination(self, s: np.ndarray, option: int, training: bool = True) -> bool:
        s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        with torch.no_grad():
            beta = self.net.termination_prob(s_t)[0, option].item()
        if training:
            return np.random.rand() < beta
        return beta > 0.5

    # ---------------- learning -----------------------------------------
    def store(self, *transition):
        self.replay.add(*transition)

    def update(self):
        if len(self.replay) < self.batch_size:
            return self.stats

        s, o, a, r, sn, d, beta = self.replay.sample(self.batch_size)
        s = s.to(self.device); sn = sn.to(self.device)
        o = o.to(self.device); a = a.to(self.device)
        r = r.to(self.device); d = d.to(self.device)

        # --- Critic loss (Q‑options) ----------------------------------
        with torch.no_grad():
            q_next = self.target.forward(sn)["q"]              # (B, N)
            # Greedy option value for target
            target_val = q_next.max(1, keepdim=True)[0]
            y = r + (1 - d) * self.gamma * target_val          # (B, 1)
        q_pred = self.net.forward(s)["q"].gather(1, o.unsqueeze(1))
        critic_loss = F.mse_loss(q_pred, y)

        self.opt_q.zero_grad(); critic_loss.backward(); self.opt_q.step()

        # --- Intra‑option policy loss -------------------------------
        action_tanh, logp = self.net.sample_action(s, o, deterministic=False)  # (B, latent), (B,1)
        q_pi = self.net.forward(s)["q"][torch.arange(self.batch_size, device=self.device), o]  # (B,)
        actor_loss = (logp.squeeze(1) - q_pi).mean()  # maximise Q, minimise entropy (same sign as OC paper)

        self.opt_pi.zero_grad(); actor_loss.backward(); self.opt_pi.step()

        # --- Termination loss ---------------------------------------
        beta_probs = self.net.termination_prob(s)[torch.arange(self.batch_size, device=self.device), o]  # (B,)
        with torch.no_grad():
            q_all = self.net.forward(s)["q"]              # fresh Q for advantage
            adv = q_all.max(1)[0] - q_all.gather(1, o.unsqueeze(1)).squeeze(1)  # A_β
        term_loss = (beta_probs * adv.detach()).mean()

        self.opt_beta.zero_grad(); term_loss.backward(); self.opt_beta.step()

        # --- Soft‑update target -------------------------------------
        with torch.no_grad():
            for tgt, src in zip(self.target.parameters(), self.net.parameters()):
                tgt.data.mul_(1 - self.tau).add_(self.tau * src.data)

        self.stats = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "termin_loss": term_loss.item(),
        }
        return self.stats

    # ---------------------------------------------------------------
    def train(self):
        self.training = True
        self.net.train()

    def eval(self):
        self.training = False
        self.net.eval()

    # ---------------------------------------------------------------
    def save_model(self, path: str):
        torch.save({
            "net": self.net.state_dict(),
            "target": self.target.state_dict(),
            "opt_q": self.opt_q.state_dict(),
            "opt_pi": self.opt_pi.state_dict(),
            "opt_beta": self.opt_beta.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.target.load_state_dict(ckpt["target"])
        self.opt_q.load_state_dict(ckpt["opt_q"])
        self.opt_pi.load_state_dict(ckpt["opt_pi"])
        self.opt_beta.load_state_dict(ckpt["opt_beta"])

