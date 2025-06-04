
"""SAC Model components for HPRL meta‑policy.

This mirrors the structure of `ddpg_model.py` but implements the
Soft Actor‑Critic algorithm with:
  • Gaussian policy + Tanh squashing
  • Twin Q‑networks
  • Entropy temperature α learned online
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Tuple, Optional, Dict

# ---------- Shared CNN backbone identical to DDPG -----------
def _build_cnn(in_channels: int = 8):
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4,4)),  # 4×4
        nn.Flatten()
    )

class GaussianActor(nn.Module):
    """Actor that outputs mean & log‑std of latent embedding."""
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self,
                 state_shape: Tuple[int,int,int]=(8,8,8),
                 latent_dim: int = 64,
                 hidden_dims = [256,256],
                 dropout: float = 0.1):
        super().__init__()
        self.cnn = _build_cnn(state_shape[2])
        cnn_out = 32*4*4
        layers = []
        prev = cnn_out
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.mean = nn.Linear(prev, latent_dim)
        self.log_std = nn.Linear(prev, latent_dim)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        if state.dim()==4 and state.shape[-1]==8:
            state = state.permute(0,3,1,2)
        x = self.cnn(state)
        x = self.mlp(x)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        # log‑prob
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return y_t, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    """Single Q‑network Q(s,a)."""
    def __init__(self,
                 state_shape: Tuple[int,int,int]=(8,8,8),
                 latent_dim: int = 64,
                 hidden_dims=[256,256],
                 dropout: float=0.1):
        super().__init__()
        self.cnn = _build_cnn(state_shape[2])
        cnn_out = 32*4*4
        self.latent_layers = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(dropout)
        )
        combined_in = cnn_out + 128
        layers=[]
        prev = combined_in
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev,1)]
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init)

    def _init(self,m):
        if isinstance(m,(nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias,0)

    def forward(self, state, action):
        if state.dim()==4 and state.shape[-1]==8:
            state = state.permute(0,3,1,2)
        s_feat = self.cnn(state)
        a_feat = self.latent_layers(action)
        x = torch.cat([s_feat, a_feat], dim=1)
        return self.mlp(x)


class ReplayBuffer:
    def __init__(self, capacity:int=100000):
        self.buffer=deque(maxlen=capacity)

    def add(self,s,a,r,sn,d):
        self.buffer.append((s,a,r,sn,d))

    def sample(self,batch:int):
        import random, torch, numpy as np
        batch=random.sample(self.buffer,batch)
        s,a,r,sn,d=zip(*batch)
        s=torch.FloatTensor(np.array(s))
        a=torch.FloatTensor(np.array(a))
        r=torch.FloatTensor(r).unsqueeze(1)
        sn=torch.FloatTensor(np.array(sn))
        d=torch.FloatTensor(d).unsqueeze(1)
        return s,a,r,sn,d

    def size(self): return len(self.buffer)


class SACAgent:
    """SAC agent with twin critics & auto‑entropy."""
    def __init__(self,
                 state_shape=(8,8,8),
                 latent_dim=64,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 tau=0.005,
                 gamma=0.99,
                 buffer_size=100000,
                 batch_size=256,
                 init_temperature=0.2,
                 device='auto'):
        self.latent_dim=latent_dim
        self.tau=tau
        self.gamma=gamma
        self.batch_size=batch_size
        self.device=torch.device('cuda' if (device=='auto' and torch.cuda.is_available()) else device)

        self.actor=GaussianActor(state_shape, latent_dim).to(self.device)
        self.critic1=Critic(state_shape, latent_dim).to(self.device)
        self.critic2=Critic(state_shape, latent_dim).to(self.device)
        self.critic1_t=Critic(state_shape, latent_dim).to(self.device)
        self.critic2_t=Critic(state_shape, latent_dim).to(self.device)
        self.critic1_t.load_state_dict(self.critic1.state_dict())
        self.critic2_t.load_state_dict(self.critic2.state_dict())

        self.actor_opt=torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt=torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.q2_opt=torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha=torch.tensor(np.log(init_temperature), device=self.device, requires_grad=True)
        self.alpha_opt=torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy=-latent_dim  # can scale later

        self.replay=ReplayBuffer(buffer_size)
        self.training=True

        # stats
        self.stats={'actor_loss':0., 'critic_loss':0., 'alpha':init_temperature}

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self,state,evaluate=False):
        import numpy as np, torch
        if isinstance(state,np.ndarray):
            state=torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state=state.unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                mu, _ = self.actor(state)
                action=torch.tanh(mu)[0]
            else:
                a,_,_=self.actor.sample(state)
                action=a[0]
        return action.cpu().numpy()

    def store_transition(self,*args):
        self.replay.add(*args)

    def update(self):
        if self.replay.size() < self.batch_size: return self.stats
        s,a,r,sn,d = self.replay.sample(self.batch_size)
        s=s.to(self.device); a=a.to(self.device); r=r.to(self.device); sn=sn.to(self.device); d=d.to(self.device)

        # ------- Critic update -------
        with torch.no_grad():
            a2, logp2, _ = self.actor.sample(sn.to(self.device))
            q1_next = self.critic1_t(sn,a2)
            q2_next = self.critic2_t(sn,a2)
            q_next = torch.min(q1_next, q2_next) - self.alpha*logp2
            target_q = r + (1-d)*self.gamma*q_next
        q1 = self.critic1(s,a)
        q2 = self.critic2(s,a)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # ------- Actor update -------
        new_a, logp, _ = self.actor.sample(s)
        q1_new = self.critic1(s,new_a)
        q2_new = self.critic2(s,new_a)
        q_new = torch.min(q1_new,q2_new)
        actor_loss = (self.alpha*logp - q_new).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # ------- Alpha update -------
        alpha_loss = -(self.log_alpha*(logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # ------- Soft update targets -------
        with torch.no_grad():
            for t, s_param in zip(self.critic1_t.parameters(), self.critic1.parameters()):
                t.data.mul_(1-self.tau).add_(self.tau*s_param.data)
            for t, s_param in zip(self.critic2_t.parameters(), self.critic2.parameters()):
                t.data.mul_(1-self.tau).add_(self.tau*s_param.data)

        self.stats={'actor_loss':actor_loss.item(),
                    'critic_loss':(q1_loss+q2_loss).item()/2,
                    'alpha':self.alpha.item()}
        return self.stats

    def train(self): self.training=True; self.actor.train(); self.critic1.train(); self.critic2.train()
    def eval(self): self.training=False; self.actor.eval(); self.critic1.eval(); self.critic2.eval()

    def save_models(self,fpath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_t': self.critic1_t.state_dict(),
            'critic2_t': self.critic2_t.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'q1_opt': self.q1_opt.state_dict(),
            'q2_opt': self.q2_opt.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_opt': self.alpha_opt.state_dict()
        }, fpath)

    def load_models(self,fpath):
        ckpt=torch.load(fpath, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])
        self.critic1_t.load_state_dict(ckpt['critic1_t'])
        self.critic2_t.load_state_dict(ckpt['critic2_t'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.q1_opt.load_state_dict(ckpt['q1_opt'])
        self.q2_opt.load_state_dict(ckpt['q2_opt'])
        self.log_alpha = ckpt['log_alpha']
        self.alpha_opt.load_state_dict(ckpt['alpha_opt'])
