
"""SAC Trainer for HPRL Meta‑Policy.

It copies nearly all logic from `ddpg_trainer.py` but swaps the
agent initialisation to use `SACAgent` (defined in `sac_model.py`)
and pulls hyper‑parameters from `sac_config.py`.
"""

import os, sys, logging, time, json
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Optional, Any
from collections import defaultdict, deque

# project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sac_model import SACAgent
from models.vae import ProgramVAE
from models.program_executor import ProgramExecutor
from environments.karel_env import KarelEnvironment
from environments.karel_world import KarelWorld
from dsl.tokens import karel_tokens
from training.sac_config import get_config, print_config, save_config


class SACTrainer:
    """Everything identical to DDPGTrainer except the agent."""

    def __init__(self, vae_checkpoint_path: str,
                 task: str = 'harvester',
                 save_dir: str = './checkpoints/sac',
                 config_override: Optional[Dict]=None,
                 device: str='auto'):
        self.save_dir=save_dir; self.task=task
        os.makedirs(save_dir, exist_ok=True)
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.config=get_config(task, config_override)
        self._setup_logging()
        print_config(self.config)
        save_config(self.config, os.path.join(save_dir,'config.json'))
        self.logger.info(f'Loading VAE from {vae_checkpoint_path}')
        self._load_vae(vae_checkpoint_path)
        self.logger.info('Initialising SAC agent')
        self._init_sac_agent()
        self._setup_environment()

        # training state
        self.current_episode=0
        self.total_steps=0
        self.best_reward=-float('inf')
        self.episode_rewards=deque(maxlen=100)
        self.episode_lengths=deque(maxlen=100)
        self.training_stats=defaultdict(list)

    # ------- identical helper methods from DDPG trainer -----------
    def _setup_logging(self):
        log_file=os.path.join(self.save_dir,'training.log')
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
        self.logger=logging.getLogger(__name__)

    def _load_vae(self, ckpt):
        if not os.path.exists(ckpt): raise FileNotFoundError(ckpt)
        checkpoint=torch.load(ckpt, map_location=self.device)
        self.vae=ProgramVAE(
            vocab_size=35, embedding_dim=64, hidden_size=256,
            latent_dim=64, state_shape=(8,8,8), num_actions=6,
            max_program_length=50, max_demo_length=10,
            dropout=0., rnn_type='GRU').to(self.device)
        self.vae.load_state_dict(checkpoint['program_vae_state_dict'])
        self.vae.eval()
        for p in self.vae.parameters(): p.requires_grad=False

    def _init_sac_agent(self):
        sac_cfg=self.config['sac']; net_cfg=self.config['networks']
        self.agent=SACAgent(
            state_shape=(8,8,8), latent_dim=64,
            actor_lr=sac_cfg['actor_lr'],
            critic_lr=sac_cfg['critic_lr'],
            alpha_lr=sac_cfg['alpha_lr'],
            tau=sac_cfg['tau'],
            gamma=sac_cfg['gamma'],
            buffer_size=sac_cfg['buffer_size'],
            batch_size=sac_cfg['batch_size'],
            init_temperature=sac_cfg['init_temperature'],
            device=self.device)
        self.macro_steps=sac_cfg['macro_steps']
        self.max_program_steps=sac_cfg['max_program_steps']
        self.warmup_episodes=self.config['training']['warmup_episodes']

        self.logger.info(f"SAC agent initialized with {self.macro_steps} macro steps")
        self.logger.info(f"Warmup period: {self.warmup_episodes} episodes")
    def _setup_environment(self):
        self.karel_env=KarelEnvironment(task=self.task, grid_size=(8,8))
        self.program_executor=ProgramExecutor(max_execution_steps=self.max_program_steps, device='cpu')
        self.tokens=karel_tokens
        self.logger.info(f"Karel environment ready for task '{self.task}' and program executor initialized")

    # ------- methods for program execution identical to DDPG -------
    def latent_to_program_tokens(self, latent):
        with torch.no_grad():
            if latent.dim()==1: latent=latent.unsqueeze(0)
            vae_cfg=self.config['vae']
            scaled=latent*vae_cfg['latent_scaling']
            if vae_cfg['use_latent_clipping']:
                lo,hi=vae_cfg['latent_clip_range']
                scaled=torch.clamp(scaled, lo, hi)
            logits,_=self.vae.vae.decode(scaled, target_programs=None, deterministic=True)
            tokens=torch.argmax(logits, dim=-1)
            if self.config['debug']['log_program_samples'] and np.random.random() < 0.1:
                # Occasionally log the decoded program for debugging
                token_list = tokens[0].cpu().numpy().tolist()[:10]  # First 10 tokens
                program_str = self.tokens.indices_to_string(token_list)
                self.logger.debug(f"Decoded program sample: {program_str}")
            
            return tokens.squeeze(0) if latent.size(0) == 1 else tokens

    def execute_latent_program(self, latent, karel_world):
        reward_config=self.config['rewards']
        program_tokens=self.latent_to_program_tokens(latent)
        if self.config['debug']['log_latent_stats'] and np.random.random() < 0.05:
            latent_np = latent.cpu().numpy()
            self.logger.debug(f"Latent stats - norm: {np.linalg.norm(latent_np):.3f}, "
                            f"mean: {np.mean(latent_np):.3f}, std: {np.std(latent_np):.3f}")
        try:
            res=self.program_executor.execute_with_dsl(program_tokens, karel_world, return_traces=False)
            base=res.get('total_reward',0.0)
            success=res.get('success',False)
            step_count=res.get('action_length',0)
            error = res.get('error', None)
            shaped_reward = base
            
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
                else:
                    shaped_reward += reward_config['failure_penalty']
            
            # Step penalty (encourage efficiency)
            shaped_reward += step_count * reward_config['step_penalty']
            
            # Log execution results occasionally
            if self.config['debug']['log_program_samples'] and np.random.random() < 0.05:
                self.logger.debug(f"Execution - Success: {success}, Steps: {step_count}, "
                                f"Base reward: {base:.3f}, Shaped reward: {shaped_reward:.3f}, "
                                f"Error: {error}")
            
            return shaped_reward, success, step_count
        except Exception as e:
            self.logger.warning(f'Execution failed: {e}')
            return reward_config['failure_penalty'], False, 0

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
        
        # Episode statistics
        episode_stats = {
            'macro_rewards': [],
            'program_successes': [],
            'program_lengths': [],
            'latent_norms': []
        }
        
        for macro_step in range(self.macro_steps):
            # Get action (latent embedding) from meta-policy
            latent_action = self.agent.select_action(current_state)
            latent_tensor = torch.FloatTensor(latent_action).to(self.device)
            
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
            
            # Update statistics
            episode_reward += macro_reward
            episode_steps += program_steps
            self.total_steps += 1
            
            # Update current state
            current_state = next_state
            
            # Early termination if task is completed
            if karel_world.done:
                break
        
        # Update DDPG agent (only after warmup period)
        if self.current_episode >= self.warmup_episodes and self.agent.replay.size() > self.agent.batch_size:
            training_stats = self.agent.update()
            
            # Log training statistics
            for key, value in training_stats.items():
                self.training_stats[key].append(value)
        
        # Compile episode results
        episode_results = {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'successful_programs': successful_programs,
            'success_rate': successful_programs / self.macro_steps,
            'avg_program_length': np.mean(episode_stats['program_lengths']),
            'avg_latent_norm': np.mean(episode_stats['latent_norms']),
            'macro_rewards': episode_stats['macro_rewards']
        }
        
        return episode_results
    
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
                latent_action = self.agent.select_action(current_state)
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
        Main training loop
        
        Args:
            max_episodes: Maximum number of training episodes
            eval_frequency: How often to run evaluation
            save_frequency: How often to save checkpoints
            log_frequency: How often to log progress
        """
        self.logger.info(f"Starting DDPG training for {max_episodes} episodes on task '{self.task}'")
        self.logger.info(f"Device: {self.device}, Macro steps: {self.macro_steps}")
        
        self.agent.train()
        start_time = time.time()
        max_episodes = int(max_episodes)
        for episode in range(max_episodes):
            self.current_episode = episode
            
            # Train one episode
            episode_results = self.train_episode(episode)
            
            # Track episode statistics
            self.episode_rewards.append(episode_results['episode_reward'])
            self.episode_lengths.append(episode_results['episode_steps'])
            
            # Update best reward
            if episode_results['episode_reward'] > self.best_reward:
                self.best_reward = episode_results['episode_reward']
                self.save_checkpoint('best_model.pt')
            
            # Logging
            if episode % log_frequency == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-log_frequency:])
                recent_episodes = self.training_stats.get('episodes', [])[-log_frequency:]
                avg_success = np.mean([ep['success_rate'] for ep in recent_episodes]) if recent_episodes else 0.0
                
                # Additional debug info
                warmup_status = "WARMUP" if episode < self.warmup_episodes else "TRAINING"
                buffer_size = self.agent.replay.size()
                
                self.logger.info(
                    f"Episode {episode:4d} [{warmup_status}] | "
                    f"Reward: {episode_results['episode_reward']:7.3f} | "
                    f"Avg Reward: {avg_reward:7.3f} | "
                    f"Success Rate: {episode_results['success_rate']:5.3f} | "
                    f"Successful Programs: {episode_results['successful_programs']}/{self.macro_steps} | "
                    f"Avg Program Length: {episode_results['avg_program_length']:5.1f} | "
                    f"Buffer Size: {buffer_size:6d} | "
                )
                
                # Log program samples if enabled
                if self.config['debug']['log_program_samples'] and episode % (log_frequency * 5) == 0:
                    self._log_program_samples()
            
            # Store episode statistics
            if 'episodes' not in self.training_stats:
                self.training_stats['episodes'] = []
            self.training_stats['episodes'].append(episode_results)
            
            # Evaluation
            if episode % eval_frequency == 0 and episode > 0:
                self.logger.info(f"Running evaluation at episode {episode}")
                eval_results = self.evaluate()
                
                self.logger.info(
                    f"Evaluation | "
                    f"Avg Reward: {eval_results['avg_reward']:7.3f} ± {eval_results['std_reward']:5.3f} | "
                    f"Avg Success Rate: {eval_results['avg_success_rate']:5.3f} | "
                    f"Avg Program Length: {eval_results['avg_program_length']:5.1f}"
                )
                
                # Store evaluation results
                if 'evaluations' not in self.training_stats:
                    self.training_stats['evaluations'] = []
                eval_results['episode'] = episode
                self.training_stats['evaluations'].append(eval_results)
            
            # Save checkpoint
            if episode % save_frequency == 0 and episode > 0:
                self.save_checkpoint(f'episode_{episode}.pt')
                self.save_training_stats()
                self.logger.info(f"Checkpoint saved at episode {episode}")
        
        # Final evaluation and save
        self.logger.info("Training completed! Running final evaluation...")
        final_eval = self.evaluate(num_episodes=50)
        
        self.logger.info(
            f"Final Evaluation | "
            f"Avg Reward: {final_eval['avg_reward']:7.3f} ± {final_eval['std_reward']:5.3f} | "
            f"Success Rate: {final_eval['avg_success_rate']:5.3f}"
        )
        
        # Save final model and statistics
        self.save_checkpoint('final_model.pt')
        self.save_training_stats()
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

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
            latent_action = self.agent.select_action(state)
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
# ------------- CLI -------------
def main():
    import argparse, json
    p=argparse.ArgumentParser(description='SAC Training for HPRL')
    p.add_argument('--vae-checkpoint', required=True)
    p.add_argument('--task', default='harvester')
    p.add_argument('--save-dir', default='./checkpoints/sac')
    p.add_argument('--episodes', default=None)
    p.add_argument('--eval-freq', default=None)
    p.add_argument('--device', default='auto')
    p.add_argument('--resume', default=None)
    p.add_argument('--debug')
    p.add_argument('--config-override', default=None,
                   help='JSON string with overrides')
    config_override = {}
    args = p.parse_args()
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
    trainer=SACTrainer(vae_checkpoint_path=args.vae_checkpoint,
                       task=args.task,
                       save_dir=args.save_dir,
                       config_override=config_override,
                       device=args.device)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed training from {args.resume}")
    train_cfg=trainer.config['training']
    trainer.train(max_episodes=train_cfg['max_episodes'],
                  eval_frequency=train_cfg['eval_frequency'],
                  save_frequency=train_cfg['save_frequency'],
                  log_frequency=train_cfg['log_frequency'])
    trainer.plot_training_progress()

if __name__=='__main__':
    main()
