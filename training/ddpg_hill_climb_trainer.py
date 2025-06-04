

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import logging
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ddpg_model import DDPGAgent
from models.vae import ProgramVAE
from models.program_executor import ProgramExecutor
from environments.karel_env import KarelEnvironment
from environments.karel_world import KarelWorld
from dsl.tokens import karel_tokens
from training.ddpg_config import get_config, print_config, save_config


class DDPGTrainer:
    """
    DDPG Trainer for HPRL Meta-Policy
    
    Orchestrates the training of the meta-policy that outputs latent embeddings
    which are decoded by the VAE into executable Karel programs.
    """
    
    def __init__(
        self,
        vae_checkpoint_path: str,
        task: str = 'harvester',
        save_dir: str = './checkpoints/ddpg',
        config_override: Optional[Dict] = None,
        device: str = 'auto'
    ):
        self.save_dir = save_dir
        self.task = task
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load configuration using the new config system
        self.config = get_config(task, config_override)
        hc_defaults = {
            'start_episode': 400,     # wait until buffer well-filled & critic sane
            'K': 5,                   # gradient ascent steps
            'step_size': 1e-3,        # α
            'beta': 1e-2,             # trust-region weight
            'loss_threshold': 0.02,   # critic MSE must fall below this
            'dual_critic': False,      # use  min(Q1,Q2)  if you have two critics
        }
        self.config.setdefault('hill_climb', hc_defaults)
        for k, v in hc_defaults.items():
            self.config['hill_climb'].setdefault(k, v)
        # Setup logging
        self._setup_logging()
        
        # Print configuration
        print_config(self.config)
        save_config(self.config, os.path.join(save_dir, 'config.json'))
        
        # Load pre-trained VAE
        self.logger.info(f"Loading VAE from {vae_checkpoint_path}")
        self._load_vae(vae_checkpoint_path)
        
        # Initialize DDPG agent
        self.logger.info("Initializing DDPG agent")
        self._init_ddpg_agent()
        
        # Initialize Karel environment and program executor
        self.logger.info(f"Setting up Karel environment for task: {self.task}")
        self._setup_environment()
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        
        # Statistics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_stats = defaultdict(list)
        self._critic_loss_window = deque(maxlen=200)     # track recent critic losses
        self._hc_enabled = False
        self.logger.info("DDPG Trainer initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.save_dir, 'training.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_vae(self, checkpoint_path: str):
        """Load pre-trained VAE model"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        vae_config = checkpoint.get('config', {})
        
        # Initialize VAE
        self.vae = ProgramVAE(
            vocab_size=35,  # From karel_tokens
            embedding_dim=64,
            hidden_size=256,
            latent_dim=64,
            state_shape=(8, 8, 8),
            num_actions=6,
            max_program_length=50,
            max_demo_length=10,
            dropout=0.0,
            rnn_type='GRU'
        ).to(self.device)
        
        # Load state dict
        self.vae.load_state_dict(checkpoint['program_vae_state_dict'])
        self.vae.eval()  # Set to evaluation mode (frozen)
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.logger.info(f"VAE loaded successfully with {sum(p.numel() for p in self.vae.parameters()):,} parameters")
    
    def _init_ddpg_agent(self):
        """Initialize DDPG agent"""
        ddpg_config = self.config['ddpg']
        network_config = self.config['networks']
        
        self.agent = DDPGAgent(
            state_shape=(8, 8, 8),
            latent_dim=64,
            actor_lr=ddpg_config['actor_lr'],
            critic_lr=ddpg_config['critic_lr'],
            tau=ddpg_config['tau'],
            gamma=ddpg_config['gamma'],
            buffer_size=ddpg_config['buffer_size'],
            batch_size=ddpg_config['batch_size'],
            noise_std=ddpg_config['noise_std'],
            noise_decay=ddpg_config['noise_decay'],
            device=self.device
        )
        
        self.macro_steps = ddpg_config['macro_steps']
        self.max_program_steps = ddpg_config['max_program_steps']
        self.warmup_episodes = self.config['training']['warmup_episodes']
        
        self.logger.info(f"DDPG agent initialized with {self.macro_steps} macro steps")
        self.logger.info(f"Warmup period: {self.warmup_episodes} episodes")
    
    def _setup_environment(self):
        """Setup Karel environment and program executor"""
        # Karel environment for state management
        self.karel_env = KarelEnvironment(task=self.task, grid_size=(8, 8))
        
        # Program executor for running decoded programs
        self.program_executor = ProgramExecutor(
            max_execution_steps=self.max_program_steps,
            device='cpu'  # Executor runs on CPU
        )
        
        # Token handler
        self.tokens = karel_tokens
        
        self.logger.info(f"Karel environment ready for task '{self.task}' and program executor initialized")
    
    def latent_to_program_tokens(self, latent_embedding: torch.Tensor) -> torch.Tensor:
        """Convert latent embedding to program tokens using VAE decoder"""
        with torch.no_grad():
            if latent_embedding.dim() == 1:
                latent_embedding = latent_embedding.unsqueeze(0)
            
            # Apply VAE configuration settings
            vae_config = self.config['vae']
            scaled_latent = latent_embedding * vae_config['latent_scaling']
            
            if vae_config['use_latent_clipping']:
                clip_range = vae_config['latent_clip_range']
                scaled_latent = torch.clamp(scaled_latent, clip_range[0], clip_range[1])
            
            # Decode using VAE decoder
            program_logits, _ = self.vae.vae.decode(
                scaled_latent, 
                target_programs=None, 
                deterministic=True
            )
            
            # Get tokens via argmax
            program_tokens = torch.argmax(program_logits, dim=-1)
            
            # 🆕 NEW: Limit sequence length to reasonable program size
            max_program_length = 50  # Reasonable limit
            if program_tokens.size(1) > max_program_length:
                program_tokens = program_tokens[:, :max_program_length]
            
            # Debug occasionally
            if np.random.random() < 0.1:
                token_indices = program_tokens[0].cpu().numpy().tolist()
                print(f"🔍 VAE OUTPUT: {token_indices[:15]}...")
                
                # Look for first m) to see if truncation will work
                for i, idx in enumerate(token_indices):
                    if idx == 3:  # m)
                        print(f"   First m) found at position {i}")
                        break
                else:
                    print("   ⚠️  No m) found in VAE output!")
            
            return program_tokens.squeeze(0) if latent_embedding.size(0) == 1 else program_tokens
    
    def execute_latent_program(
        self, 
        latent_embedding: torch.Tensor, 
        karel_world: KarelWorld
    ) -> Tuple[float, bool, int]:
        """
        Execute a program decoded from latent embedding with improved reward calculation
        """
        reward_config = self.config['rewards']
        
        # Decode latent to program tokens
        program_tokens = self.latent_to_program_tokens(latent_embedding)
        
        # Initialize flags for reward shaping
        is_executable = False
        execution_attempted = False
        
        try:
            # Execute program using DSL executor
            result = self.program_executor.execute_with_dsl(
                program_tokens, 
                karel_world, 
                return_traces=False
            )
            
            # If we get here, the program was at least executable (no parsing errors)
            execution_attempted = True
            
            # Extract basic results
            base_reward = result.get('total_reward', 0.0)
            success = result.get('success', False)
            step_count = result.get('action_length', 0)
            error = result.get('error', None)
            
            # Check if program was executable (no parsing/syntax errors)
            if error is None:
                is_executable = True
            elif not any(err_type in str(error).lower() for err_type in ['parse', 'invalid', 'syntax', 'format']):
                is_executable = True
            else:
                is_executable = False
            
            # Calculate shaped reward
            shaped_reward = base_reward
            
            # Reward for executable programs (even if they fail)
            if is_executable:
                shaped_reward += reward_config['executable_bonus']
            
            if success:
                # Success bonus
                shaped_reward += reward_config['success_bonus']
                
                # Efficiency reward (reward shorter programs)
                if step_count > 0 and step_count < 30:
                    efficiency_bonus = reward_config['efficiency_reward'] * (30 - step_count) / 30
                    shaped_reward += efficiency_bonus
                    
            else:
                # Handle different types of failures
                if error and 'timeout' in str(error).lower():
                    shaped_reward += reward_config['timeout_penalty']
                elif error and ('invalid' in str(error).lower() or 'parse' in str(error).lower()):
                    shaped_reward += reward_config['invalid_program_penalty']
                    is_executable = False  # Override if it's actually a parsing error
                else:
                    shaped_reward += reward_config['failure_penalty']
            
            # Step penalty (encourage efficiency)
            shaped_reward += step_count * reward_config['step_penalty']
            
            # 🔥 LOG PROGRAM AND REWARDS
            program_str = self.tokens.indices_to_string(program_tokens.cpu().numpy().tolist()[:20])
            self.logger.info(f"PROGRAM: {program_str} | BASE: {base_reward:.3f} | SHAPED: {shaped_reward:.3f}")
            
            return shaped_reward, success, step_count
            
        except Exception as e:
            # Handle execution errors
            error_str = str(e).lower()
            
            if any(err_type in error_str for err_type in ['parse', 'invalid', 'syntax', 'format']):
                # Parsing/syntax error - program not executable
                penalty = reward_config['invalid_program_penalty']
            else:
                # Other execution error - might still be syntactically correct
                penalty = reward_config['failure_penalty']
                # Give executable bonus even if execution failed for other reasons
                if execution_attempted:
                    penalty += reward_config['executable_bonus']
            
            # 🔥 LOG FAILED PROGRAM
            try:
                program_str = self.tokens.indices_to_string(program_tokens.cpu().numpy().tolist()[:20])
                self.logger.info(f"PROGRAM: {program_str} | BASE: 0.000 | SHAPED: {penalty:.3f} | ERROR: {e}")
            except:
                self.logger.info(f"PROGRAM: [DECODE_FAILED] | BASE: 0.000 | SHAPED: {penalty:.3f} | ERROR: {e}")
            
            return penalty, False, 0
    
    def train_episode(self, episode_num: int) -> Dict[str, Any]:
        """
        Train a single episode with macro-step structure
        
        Args:
            episode_num: Current episode number
            
        Returns:
            Dictionary with episode statistics
        """
        # Reset environment
        obs, info = self.karel_env.reset()
        current_state = self._get_state_array(obs)
        
        episode_reward = 0.0
        episode_steps = 0
        successful_programs = 0
        total_updates = 0  # Track number of DDPG updates this episode
        
        # Episode statistics
        episode_stats = {
            'macro_rewards': [],
            'program_successes': [],
            'program_lengths': [],
            'latent_norms': [],
            'ddpg_updates': []  # Track updates per macro step
        }
        
        for macro_step in range(self.macro_steps):
            # Get action (latent embedding) from meta-policy
            #latent_action = self.agent.select_action(current_state, add_noise=True)
            #latent_tensor = torch.FloatTensor(latent_action).to(self.device)
            # ---- 1) Actor proposal -------------------------------------------------
            actor_action = self.agent.select_action(current_state, add_noise=True)
            latent_tensor = torch.FloatTensor(actor_action).unsqueeze(0).to(self.device)

            # ---- 2) Optional HC refinement ----------------------------------------
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
            latent_tensor = self._hill_climb_action(state_tensor, latent_tensor).squeeze(0)
            latent_action = latent_tensor.cpu().numpy()       # keep numpy version for replay
            # Track latent statistics
            latent_norm = float(np.linalg.norm(latent_action))
            episode_stats['latent_norms'].append(latent_norm)
            
            # Create fresh Karel world for program execution
            karel_world = KarelWorld(task=self.task, grid_size=(8, 8), timeout_steps=self.max_program_steps)
            karel_world.reset()
            
            # Copy current state to the fresh world
            karel_world.state = current_state.copy()
            
            # Execute program decoded from latent embedding
            macro_reward, success, program_steps = self.execute_latent_program(latent_tensor, karel_world)
            
            # Get next state from karel world
            next_state = karel_world.get_state()
            
            # Track statistics
            episode_stats['macro_rewards'].append(macro_reward)
            episode_stats['program_successes'].append(success)
            episode_stats['program_lengths'].append(program_steps)
            
            if success:
                successful_programs += 1
            
            # Check if episode should terminate (task completed or failed)
            done = karel_world.done or macro_step == self.macro_steps - 1
            
            # Store transition in replay buffer
            self.agent.store_transition(
                current_state, latent_action, macro_reward, next_state, done
            )
            
            # 🔥 UPDATE DDPG AFTER EACH MACRO STEP (if enough samples)
            macro_step_updates = 0
            if (self.current_episode >= self.warmup_episodes and 
                self.agent.replay_buffer.size() > self.agent.batch_size):
                
                training_stats = self.agent.update()
                if training_stats and 'critic_loss' in training_stats:
                    self._critic_loss_window.append(training_stats['critic_loss'])
                    # When the running mean drops below threshold we enable HC once
                    if (not self._hc_enabled and
                        len(self._critic_loss_window) == self._critic_loss_window.maxlen and
                        np.mean(self._critic_loss_window) < self.config['hill_climb']['loss_threshold']):
                        self._hc_enabled = True
                        self.logger.info('⚡ Hill-climb unlocked: critic looks stable')
                macro_step_updates = 1
                total_updates += 1
                
                # Log training statistics
                for key, value in training_stats.items():
                    self.training_stats[key].append(value)
            
            episode_stats['ddpg_updates'].append(macro_step_updates)
            
            # Update statistics
            episode_reward += macro_reward
            episode_steps += program_steps
            self.total_steps += 1
            
            # Update current state for next macro step
            current_state = next_state
            
            # Early termination if task is completed
            if karel_world.done:
                break
        
        # Compile episode results
        episode_results = {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'successful_programs': successful_programs,
            'success_rate': successful_programs / self.macro_steps,
            'avg_program_length': np.mean(episode_stats['program_lengths']),
            'avg_latent_norm': np.mean(episode_stats['latent_norms']),
            'macro_rewards': episode_stats['macro_rewards'],
            'total_ddpg_updates': total_updates,  # New metric
            'updates_per_macro_step': episode_stats['ddpg_updates']
        }
        
        return episode_results
    
    def _hill_climb_action(self, state_tensor: torch.Tensor,
                       z_base: torch.Tensor) -> torch.Tensor:
        """
        Perform in-graph gradient ascent on the critic Q(s,z) to refine the latent.
        z_base: detached output of the actor  (shape [1,64])
        Returns z*  (detached, clipped to [-1,1])
        """
        if not self._hc_enabled:
            return z_base                    # HC still locked

        cfg = self.config['hill_climb']
        z = z_base.clone().detach().requires_grad_(True)
        state_tensor = state_tensor.detach()   # no grad wrt state

        for _ in range(cfg['K']):
            q1 = self.agent.critic(state_tensor, z)
            if cfg['dual_critic'] and hasattr(self.agent, 'critic_target'):
                # if you keep a second critic in self.agent.critic2 change here
                q2 = self.agent.critic_target(state_tensor, z)
                q  = torch.min(q1, q2)
            else:
                q  = q1

            # trust-region penalty
            penalty = cfg['beta'] * ((z - z_base) ** 2).sum()
            obj = q - penalty
            obj.backward()

            with torch.no_grad():
                if torch.isnan(z.grad).any() or torch.isinf(z.grad).any():
                    self.logger.warning('HC aborted: gradient nan/inf')
                    return z_base
                z += cfg['step_size'] * z.grad
                z.grad.zero_()

        z_clipped = torch.clamp(z.detach(), -1.0, 1.0)
        return z_clipped
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the current policy without exploration noise
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.agent.eval()
        
        eval_rewards = []
        eval_success_rates = []
        eval_program_lengths = []
        
        for episode in range(num_episodes):
            # Reset environment
            obs, info = self.karel_env.reset()
            current_state = self._get_state_array(obs)
            
            episode_reward = 0.0
            successful_programs = 0
            total_program_steps = 0
            
            for macro_step in range(self.macro_steps):
                # Get action without exploration noise
                latent_action = self.agent.select_action(current_state, add_noise=False)
                latent_tensor = torch.FloatTensor(latent_action).to(self.device)
                
                # Create fresh Karel world
                karel_world = KarelWorld(task=self.task, grid_size=(8, 8), timeout_steps=self.max_program_steps)
                karel_world.reset()
                karel_world.state = current_state.copy()
                
                # Execute program
                macro_reward, success, program_steps = self.execute_latent_program(latent_tensor, karel_world)
                
                episode_reward += macro_reward
                total_program_steps += program_steps
                
                if success:
                    successful_programs += 1
                
                current_state = karel_world.get_state()
                
                if karel_world.done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_success_rates.append(successful_programs / self.macro_steps)
            eval_program_lengths.append(total_program_steps / self.macro_steps)
        
        self.agent.train()  # Return to training mode
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_success_rate': np.mean(eval_success_rates),
            'avg_program_length': np.mean(eval_program_lengths),
            'eval_episodes': num_episodes
        }
    
    def train(
        self, 
        max_episodes: int = 1000,
        eval_frequency: int = 50,
        save_frequency: int = 100,
        log_frequency: int = 10
    ):
        """
        Main training loop with minimal logging
        """
        self.logger.info(f"Starting DDPG training for {max_episodes} episodes on task '{self.task}'")
        
        self.agent.train()
        start_time = time.time()
        
        for episode in range(max_episodes):
            self.current_episode = episode
            
            self.logger.info(f"\n=== EPISODE {episode} ===")
            
            # Train one episode
            episode_results = self.train_episode(episode)
            
            # Track episode statistics
            self.episode_rewards.append(episode_results['episode_reward'])
            self.episode_lengths.append(episode_results['episode_steps'])
            
            # Update best reward
            if episode_results['episode_reward'] > self.best_reward:
                self.best_reward = episode_results['episode_reward']
                self.save_checkpoint('best_model.pt')
            
            # 🔥 LOG EPISODE TOTAL
            self.logger.info(f"EPISODE {episode} TOTAL REWARD: {episode_results['episode_reward']:.3f}")
            
            # Store episode statistics
            if 'episodes' not in self.training_stats:
                self.training_stats['episodes'] = []
            self.training_stats['episodes'].append(episode_results)
            
            # Evaluation (minimal logging)
            if episode % eval_frequency == 0 and episode > 0:
                eval_results = self.evaluate()
                self.logger.info(f"EVALUATION | Avg Reward: {eval_results['avg_reward']:.3f}")
                
                # Store evaluation results
                if 'evaluations' not in self.training_stats:
                    self.training_stats['evaluations'] = []
                eval_results['episode'] = episode
                self.training_stats['evaluations'].append(eval_results)
            
            # Save checkpoint
            if episode % save_frequency == 0 and episode > 0:
                self.save_checkpoint(f'episode_{episode}.pt')
                self.save_training_stats()
        
        # Final summary
        final_eval = self.evaluate(num_episodes=50)
        total_time = time.time() - start_time
        
        self.logger.info(f"\nTRAINING COMPLETED!")
        self.logger.info(f"Final Avg Reward: {final_eval['avg_reward']:.3f}")
        self.logger.info(f"Training Time: {total_time/3600:.2f} hours")
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        self.save_training_stats()
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'episode': self.current_episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'task': self.task,
            'config': self.config,
            'training_stats': dict(self.training_stats)  # Convert defaultdict to dict
        }
        
        # Save DDPG agent
        self.agent.save_models(filepath)
        
        # Save additional training info
        torch.save(checkpoint, filepath.replace('.pt', '_training_info.pt'))
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        # Load DDPG agent
        self.agent.load_models(filepath)
        
        # Load training info
        info_path = filepath.replace('.pt', '_training_info.pt')
        if os.path.exists(info_path):
            checkpoint = torch.load(info_path, map_location=self.device)
            
            self.current_episode = checkpoint.get('episode', 0)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.best_reward = checkpoint.get('best_reward', -float('inf'))
            self.training_stats = defaultdict(list, checkpoint.get('training_stats', {}))
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def save_training_stats(self):
        """Save training statistics to JSON"""
        stats_path = os.path.join(self.save_dir, 'training_stats.json')
        
        # Convert numpy arrays to lists for JSON serialization
        stats_to_save = {}
        for key, value in self.training_stats.items():
            if isinstance(value, list):
                stats_to_save[key] = [float(v) if isinstance(v, np.floating) else v for v in value]
            else:
                stats_to_save[key] = value
        
        with open(stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress"""
        if not self.training_stats['episodes']:
            self.logger.warning("No training statistics to plot")
            return
        
        episodes = list(range(len(self.training_stats['episodes'])))
        episode_rewards = [ep['episode_reward'] for ep in self.training_stats['episodes']]
        success_rates = [ep['success_rate'] for ep in self.training_stats['episodes']]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(episodes, episode_rewards, alpha=0.7, label='Episode Reward')
        if len(episode_rewards) > 10:
            # Moving average
            window = min(50, len(episode_rewards) // 10)
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True)
        
        # Success rates
        ax2.plot(episodes, success_rates, 'g-', alpha=0.7, label='Success Rate')
        if len(success_rates) > 10:
            window = min(50, len(success_rates) // 10)
            moving_avg = np.convolve(success_rates, np.ones(window)/window, mode='valid')
            ax2.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Program Success Rate')
        ax2.legend()
        ax2.grid(True)
        
        # Training losses (if available)
        if 'actor_loss' in self.training_stats and self.training_stats['actor_loss']:
            ax3.plot(self.training_stats['actor_loss'], label='Actor Loss')
            ax3.plot(self.training_stats['critic_loss'], label='Critic Loss')
            ax3.set_xlabel('Update Step')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Losses')
            ax3.legend()
            ax3.grid(True)
        
        # Q-values
        if 'q_values' in self.training_stats and self.training_stats['q_values']:
            ax4.plot(self.training_stats['q_values'], label='Q-values')
            ax4.set_xlabel('Update Step')
            ax4.set_ylabel('Q-value')
            ax4.set_title('Q-values')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training progress plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _log_program_samples(self):
        """Log sample programs generated by the current policy"""
        self.logger.info("=== PROGRAM SAMPLES ===")
        
        # Generate a few sample latent embeddings
        for i in range(3):
            # Sample random state
            obs, _ = self.karel_env.reset()
            state = self._get_state_array(obs)
            
            # Get latent from policy
            latent_action = self.agent.select_action(state, add_noise=False)
            latent_tensor = torch.FloatTensor(latent_action).to(self.device)
            
            # Decode to program
            program_tokens = self.latent_to_program_tokens(latent_tensor)
            
            try:
                # Convert to string
                token_list = program_tokens.cpu().numpy().tolist()[:15]  # First 15 tokens
                program_str = self.tokens.indices_to_string(token_list)
                self.logger.info(f"Sample {i+1}: {program_str}")
            except Exception as e:
                self.logger.info(f"Sample {i+1}: Failed to decode - {e}")
        
        self.logger.info("=====================")
    
    def _get_state_array(self, obs) -> np.ndarray:
        """Extract state array from environment observation"""
        if isinstance(obs, tuple):
            # Find numpy array in tuple
            for item in obs:
                if isinstance(item, np.ndarray) and len(item.shape) == 3:
                    return item
            return obs[0]  # Fallback to first item
        else:
            return obs


def main():
    """Main function for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DDPG Training for HPRL')
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                       help='Path to pre-trained VAE checkpoint')
    parser.add_argument('--task', type=str, default='harvester',
                       choices=['harvester', 'cleanHouse', 'fourCorners', 'randomMaze', 'stairClimber', 'topOff'],
                       help='Karel task to train on')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/ddpg',
                       help='Directory to save checkpoints')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes (overrides config)')
    parser.add_argument('--eval-freq', type=int, default=None,
                       help='Evaluation frequency (overrides config)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true',
                       help='Use debug configuration (fast training, detailed logging)')
    parser.add_argument('--config-override', type=str, default=None,
                       help='JSON string with config overrides')
    
    args = parser.parse_args()
    
    # Parse config overrides
    config_override = {}
    if args.config_override:
        import json
        config_override = json.loads(args.config_override)
    
    # Apply command line overrides
    if args.episodes is not None:
        config_override.setdefault('training', {})['max_episodes'] = args.episodes
    
    if args.eval_freq is not None:
        config_override.setdefault('training', {})['eval_frequency'] = args.eval_freq
    
    # Use debug config if requested
    if args.debug:
        from training.ddpg_config import get_debug_config
        config_override.update({
            'training': {
                'max_episodes': 200,
                'eval_frequency': 20,
                'save_frequency': 500,
                'log_frequency': 5,
                'warmup_episodes': 10,
            },
            'ddpg': {
                'batch_size': 16,
                'buffer_size': 5000,
                'noise_std': 0.4,
            },
            'debug': {
                'log_program_samples': True,
                'log_latent_stats': True,
                'verbose_evaluation': True,
            }
        })
    
    # Create trainer
    trainer = DDPGTrainer(
        vae_checkpoint_path=args.vae_checkpoint,
        task=args.task,
        save_dir=args.save_dir,
        config_override=config_override,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed training from {args.resume}")
    
    # Get training parameters from config
    training_config = trainer.config['training']
    
    # Start training
    trainer.train(
        max_episodes=training_config['max_episodes'],
        eval_frequency=training_config['eval_frequency'],
        save_frequency=training_config['save_frequency'],
        log_frequency=training_config['log_frequency']
    )
    
    # Plot training progress
    trainer.plot_training_progress()
    
    print("Training completed!")


if __name__ == "__main__":
    main()