"""
DDPG Model for HPRL Meta-Policy Training

This module implements DDPG (Deep Deterministic Policy Gradient) for training
the meta-policy that outputs 64-dimensional latent program embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Tuple, Optional, Dict, Any


class Actor(nn.Module):
    """
    Actor network that maps Karel states to latent program embeddings.
    
    Input: Karel environment state (8x8x8 grid)
    Output: 64-dimensional latent program embedding
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int, int] = (8, 8, 8),
        latent_dim: int = 64,
        hidden_dims: list = [256, 256],
        dropout: float = 0.0
    ):
        super(Actor, self).__init__()
        
        self.state_shape = state_shape
        self.latent_dim = latent_dim
        
        # CNN for processing Karel grid states
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce to 4x4
            nn.Flatten()
        )
        
        # Calculate conv output size
        conv_output_size = 32 * 4 * 4  # 512
        
        # Fully connected layers
        layers = []
        prev_dim = conv_output_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer with tanh to bound the latent embeddings
        layers.append(nn.Linear(prev_dim, latent_dim))
        layers.append(nn.Tanh())  # Bound outputs to [-1, 1]
        
        self.fc_layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network
        
        Args:
            state: [batch_size, H, W, C] or [batch_size, C, H, W] - Karel states
            
        Returns:
            latent: [batch_size, latent_dim] - program embeddings
        """
        # Handle different input formats
        if state.dim() == 4 and state.shape[-1] == self.state_shape[2]:
            # Convert from [B, H, W, C] to [B, C, H, W]
            state = state.permute(0, 3, 1, 2)
        elif state.dim() == 3 and state.shape[-1] == self.state_shape[2]:
            # Add batch dimension and convert format
            state = state.unsqueeze(0).permute(0, 3, 1, 2)
        
        # CNN feature extraction
        conv_features = self.conv_layers(state)
        
        # FC layers to latent embedding
        latent = self.fc_layers(conv_features)
        
        return latent
    
    def get_action(self, state: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get action for inference with optional exploration noise
        
        Args:
            state: Karel state
            noise: Optional noise to add for exploration
            
        Returns:
            action: Latent program embedding
        """
        with torch.no_grad():
            action = self.forward(state)
            
            if noise is not None:
                action = action + noise
                action = torch.clamp(action, -1.0, 1.0)  # Keep bounded
        
        return action


class Critic(nn.Module):
    """
    Critic network that estimates Q-values Q(s, a) where a is latent embedding.
    
    Input: Karel state + 64-dimensional latent embedding
    Output: Q-value (scalar)
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int, int] = (8, 8, 8),
        latent_dim: int = 64,
        hidden_dims: list = [256, 256],
        dropout: float = 0.0
    ):
        super(Critic, self).__init__()
        
        self.state_shape = state_shape
        self.latent_dim = latent_dim
        
        # CNN for processing Karel grid states
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        conv_output_size = 32 * 4 * 4  # 512
        
        # Separate processing for latent embeddings
        self.latent_layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined processing
        combined_input_size = conv_output_size + 128
        
        layers = []
        prev_dim = combined_input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output single Q-value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.combined_layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network
        
        Args:
            state: [batch_size, H, W, C] or [batch_size, C, H, W] - Karel states
            action: [batch_size, latent_dim] - latent embeddings
            
        Returns:
            q_value: [batch_size, 1] - Q-values
        """
        # Handle different input formats for state
        if state.dim() == 4 and state.shape[-1] == self.state_shape[2]:
            state = state.permute(0, 3, 1, 2)
        elif state.dim() == 3 and state.shape[-1] == self.state_shape[2]:
            state = state.unsqueeze(0).permute(0, 3, 1, 2)
        
        # Process state through CNN
        state_features = self.conv_layers(state)
        
        # Process action through FC layers
        action_features = self.latent_layers(action)
        
        # Combine state and action features
        combined = torch.cat([state_features, action_features], dim=1)
        
        # Get Q-value
        q_value = self.combined_layers(combined)
        
        return q_value


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck noise process for exploration in continuous action spaces.
    """
    
    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        x0: Optional[np.ndarray] = None
    ):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def reset(self):
        """Reset the noise process"""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
    
    def sample(self) -> np.ndarray:
        """Sample noise"""
        dx = self.theta * (self.mu - self.x_prev) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = self.x_prev + dx
        return self.x_prev


class ReplayBuffer:
    """
    Experience replay buffer for DDPG training.
    """
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)


class DDPGAgent:
    """
    DDPG Agent for training meta-policy in HPRL framework.
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int, int] = (8, 8, 8),
        latent_dim: int = 64,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        tau: float = 0.005,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 64,
        noise_std: float = 0.1,
        noise_decay: float = 0.995,
        device: str = 'auto'
    ):
        self.state_shape = state_shape
        self.latent_dim = latent_dim
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.actor = Actor(state_shape, latent_dim).to(self.device)
        self.critic = Critic(state_shape, latent_dim).to(self.device)
        
        # Target networks
        self.actor_target = Actor(state_shape, latent_dim).to(self.device)
        self.critic_target = Critic(state_shape, latent_dim).to(self.device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Noise process
        self.noise = OrnsteinUhlenbeckNoise(latent_dim, sigma=noise_std)
        
        # Training statistics
        self.training_stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'q_values': 0.0,
            'noise_std': noise_std
        }
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action given state
        
        Args:
            state: Karel environment state
            add_noise: Whether to add exploration noise
            
        Returns:
            action: 64-dimensional latent embedding
        """
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device)
        
        # Get action from actor
        action = self.actor.get_action(state_tensor).cpu().numpy()[0]
        
        # Add exploration noise
        if add_noise and self.training:
            noise = self.noise.sample() * self.noise_std
            action = action + noise
            action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """
        Update actor and critic networks using DDPG algorithm
        
        Returns:
            Dictionary with training statistics
        """
        if self.replay_buffer.size() < self.batch_size:
            return self.training_stats
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (self.gamma * target_q * ~dones)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update_targets()
        
        # Decay noise
        self.noise_std *= self.noise_decay
        self.noise_std = max(self.noise_std, 0.01)  # Minimum noise
        
        # Update statistics
        self.training_stats = {
            'actor_loss': float(actor_loss.item()),
            'critic_loss': float(critic_loss.item()),
            'q_values': float(current_q.mean().item()),
            'noise_std': self.noise_std
        }
        
        return self.training_stats
    
    def soft_update_targets(self):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self):
        """Set agent to training mode"""
        self.training = True
        self.actor.train()
        self.critic.train()
    
    def eval(self):
        """Set agent to evaluation mode"""
        self.training = False
        self.actor.eval()
        self.critic.eval()
    
    def save_models(self, filepath: str):
        """Save model checkpoints"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'noise_std': self.noise_std,
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, filepath)
    
    def load_models(self, filepath: str):
        """Load model checkpoints"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.noise_std = checkpoint.get('noise_std', self.noise_std)
        self.training_stats = checkpoint.get('training_stats', self.training_stats)


# Example usage and testing
if __name__ == "__main__":
    print("Testing DDPG Model...")
    
    # Create DDPG agent
    agent = DDPGAgent(
        state_shape=(8, 8, 8),
        latent_dim=64,
        batch_size=32
    )
    
    # Test action selection
    test_state = np.random.randn(8, 8, 8)
    action = agent.select_action(test_state, add_noise=True)
    print(f"Action shape: {action.shape}, Action range: [{action.min():.3f}, {action.max():.3f}]")
    
    # Test storing transitions and updates
    agent.train()
    
    for i in range(100):
        state = np.random.randn(8, 8, 8)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(8, 8, 8)
        done = np.random.random() > 0.9
        
        agent.store_transition(state, action, reward, next_state, done)
        
        if i > 32:  # Start updating after enough samples
            stats = agent.update()
            if i % 20 == 0:
                print(f"Step {i}: Actor Loss: {stats['actor_loss']:.4f}, "
                      f"Critic Loss: {stats['critic_loss']:.4f}, "
                      f"Q-value: {stats['q_values']:.4f}")
    
    print("DDPG Model tests completed!")