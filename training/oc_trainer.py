"""
Option-Critic Trainer for HPRL Meta-Policy
=========================================

This is a *minimal* drop-in replacement for your current `ddpg_trainer.py`.
It re-uses almost all of that file’s helper functions (logging, plotting,
reward shaping, etc.) and swaps in an `OptionCriticAgent`.

Key changes
-----------
*   **Two-level control-flow** in every episode:
    1. High-level option π<sub>o</sub> is (re-)selected when  
       – the previous option terminates *or*  
       – a fixed macro-step limit `macro_steps` is reached.
    2. Intra-option actor π(a | s, o) runs one latent “primitive action”
       *per environment step* until termination.

*   **Replay** now stores `(s, o, a, r, s′, β)` tuples where `β` is the
    termination flag supplied by the option-critic termination head.

*   **Losses** – critic, intra-option actor and termination losses – are
    returned by `agent.update()` and logged into `self.training_stats`.

This script is intentionally thin; all option-critic logic lives in
`models/option_critic_model.py`.
"""

import os, sys, time, json, logging
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# ─── Project imports ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.oc_model import OptionCriticAgent          # NEW
from models.vae                import ProgramVAE
from models.program_executor    import ProgramExecutor
from environments.karel_env     import KarelEnvironment
from environments.karel_world   import KarelWorld
from dsl.tokens                 import karel_tokens

from training.oc_config   import get_config, print_config, save_config  # NEW


# ═════════════════════════════════════════════════════════════════════════════
#                              TRAINER CLASS
# ═════════════════════════════════════════════════════════════════════════════
class OCTrainer:
    """
    Option-Critic trainer for HPRL. 99 % identical to your DDPG trainer; only
    agent initialisation, per-step control-flow and logging differ.
    """

    def __init__(self,
                 vae_checkpoint_path: str,
                 task: str = "harvester",
                 save_dir: str = "./checkpoints/oc",
                 config_override: Optional[Dict[str, Any]] = None,
                 device: str = "auto"):

        self.task, self.save_dir = task, save_dir
        os.makedirs(save_dir, exist_ok=True)

        # ── device ────────────────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
        else:
            self.device = torch.device(device)

        # ── config & logging ─────────────────────────────────────────────────
        self.config = get_config(task, config_override)
        self._setup_logging()
        print_config(self.config)
        save_config(self.config, os.path.join(save_dir, "config.json"))

        # ── VAE (frozen) ─────────────────────────────────────────────────────
        self.logger.info(f"Loading VAE from {vae_checkpoint_path}")
        self._load_vae(vae_checkpoint_path)

        # ── option-critic agent ──────────────────────────────────────────────
        self.logger.info("Initialising Option-Critic agent")
        self._init_oc_agent()

        # ── Karel env / executor ─────────────────────────────────────────────
        self._setup_environment()

        # ── trainer state ────────────────────────────────────────────────────
        self.current_episode = 0
        self.total_steps     = 0
        self.best_reward     = -float("inf")

        self.episode_rewards = deque(maxlen=100)
        self.training_stats  = defaultdict(list)

    # ─────────────────────────────────────────────────────────────────────────
    #                        Helper setup functions
    # ─────────────────────────────────────────────────────────────────────────
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

    def _load_vae(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.vae = ProgramVAE(
            vocab_size=35, embedding_dim=64, hidden_size=256,
            latent_dim=64, state_shape=(8, 8, 8), num_actions=6,
            max_program_length=50, max_demo_length=10,
            dropout=0.0, rnn_type="GRU").to(self.device)
        self.vae.load_state_dict(ckpt["program_vae_state_dict"])
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    def _init_oc_agent(self):
        oc_cfg = self.config["oc"]
        self.agent = OptionCriticAgent(
            state_shape=(8, 8, 8),
            latent_dim=64,
            n_options=oc_cfg["num_options"],
            actor_lr=oc_cfg["actor_lr"],
            critic_lr=oc_cfg["critic_lr"],
            beta_lr=oc_cfg["beta_lr"],
            tau=oc_cfg["tau"],
            gamma=oc_cfg["gamma"],
            buffer_size=oc_cfg["buffer_size"],
            batch_size=oc_cfg["batch_size"],
            # init_epsilon=oc_cfg["eps_start"],
            # eps_decay=oc_cfg["eps_decay"],
            device=self.device)

        self.macro_steps       = oc_cfg["macro_steps"]
        self.max_program_steps = oc_cfg["max_program_steps"]
        self.warmup_episodes   = self.config["training"]["warmup_episodes"]

    def _setup_environment(self):
        self.karel_env       = KarelEnvironment(task=self.task, grid_size=(8, 8))
        self.program_executor = ProgramExecutor(self.max_program_steps, "cpu")
        self.tokens          = karel_tokens

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
        
        # Log latent statistics if enabled
        if self.config['debug']['log_latent_stats'] and np.random.random() < 0.05:
            latent_np = latent_embedding.cpu().numpy()
            self.logger.debug(f"Latent stats - norm: {np.linalg.norm(latent_np):.3f}, "
                            f"mean: {np.mean(latent_np):.3f}, std: {np.std(latent_np):.3f}")
        
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
            
            # DEBUG: Log the raw result for debugging
            if np.random.random() < 0.1:  # 10% chance to log
                program_str = self.tokens.indices_to_string(program_tokens.cpu().numpy().tolist()[:10])
                # self.logger.info(f"DEBUG EXEC: Program='{program_str}', Success={success}, Error='{error}', Steps={step_count}, BaseReward={base_reward}")
            
            # Check if program was executable (no parsing/syntax errors)
            if error is None:
                is_executable = True
            elif not any(err_type in str(error).lower() for err_type in ['parse', 'invalid', 'syntax', 'format']):
                is_executable = True
            else:
                is_executable = False
                
            # DEBUG: Log executable status
            # if np.random.random() < 0.1:
                # self.logger.info(f"DEBUG REWARD: Executable={is_executable}, Error='{error}'")
            
            # Calculate shaped reward
            shaped_reward = base_reward
            
            # NEW: Reward for executable programs (even if they fail)
            if is_executable:
                shaped_reward += reward_config['executable_bonus']
                # if np.random.random() < 0.1:
                    # self.logger.info(f"DEBUG: Applied executable bonus: +{reward_config['executable_bonus']:.3f}")
            
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
                    # if np.random.random() < 0.1:
                    #     self.logger.info(f"DEBUG: Applied invalid program penalty: {reward_config['invalid_program_penalty']:.3f}")
                else:
                    shaped_reward += reward_config['failure_penalty']
                    # if np.random.random() < 0.1:
                    #     self.logger.info(f"DEBUG: Applied failure penalty: {reward_config['failure_penalty']:.3f}")
            
            # Step penalty (encourage efficiency)
            shaped_reward += step_count * reward_config['step_penalty']
            
            # DEBUG: Final reward calculation
            # if np.random.random() < 0.1:
            #     self.logger.info(f"DEBUG FINAL: BaseReward={base_reward:.3f}, ShapedReward={shaped_reward:.3f}, Components: executable={is_executable}, success={success}")
            
            return shaped_reward, success, step_count
            
        except Exception as e:
            # DEBUG: Log exception details
            error_str = str(e).lower()
            #self.logger.info(f"DEBUG EXCEPTION: {e}, ErrorStr='{error_str}'")
            
            if any(err_type in error_str for err_type in ['parse', 'invalid', 'syntax', 'format']):
                # Parsing/syntax error - program not executable
                penalty = reward_config['invalid_program_penalty']
                #self.logger.info(f"DEBUG: Non-executable program penalty: {penalty:.3f}")
            else:
                # Other execution error - might still be syntactically correct
                penalty = reward_config['failure_penalty']
                # Give executable bonus even if execution failed for other reasons
                if execution_attempted:
                    penalty += reward_config['executable_bonus']
                    #self.logger.info(f"DEBUG: Executable but failed program: {penalty:.3f}")
                '''else:
                    self.logger.info(f"DEBUG: Execution failed penalty: {penalty:.3f}")'''
            
            #self.logger.warning(f"Program execution failed: {e}")
            return penalty, False, 0

    # ─────────────────────────────────────────────────────────────────────────
    #                             Core logic
    # ─────────────────────────────────────────────────────────────────────────
    def train_episode(self) -> Dict[str, Any]:
        obs, _      = self.karel_env.reset()
        state       = self._get_state_array(obs)

        option      = self.agent.select_option(state)
        option_len  = 0  # steps since option was chosen

        ep_reward, ep_steps, successes = 0.0, 0, 0
        ep_stats = {"opt_rewards": defaultdict(float),
                    "opt_lengths": defaultdict(int),
                    "program_lengths": []}

        # ── unconstrained loop – env stops internally ───────────────────────
        done = False
        karel_world   = KarelWorld(task=self.task, grid_size=(8, 8),
                                       timeout_steps=self.max_program_steps)
        karel_world.reset()
        while not done and ep_steps < self.config["training"]["max_episode_steps"]:
            # self.logger.info(f"ep_steps={ep_steps}, option={option}, option_len={option_len}, total_steps={self.total_steps}")
            # 1. Intra-option action
            action_latent = self.agent.select_action(state, option)            
            karel_world.state = state.copy()

            reward, success, program_length = self.execute_latent_program(
                torch.FloatTensor(action_latent).to(self.device), karel_world)

            next_state = karel_world.get_state()
            done       = karel_world.done

            # 2. Termination flag β
            beta = self.agent.option_termination(next_state, option, training=True)

            # 3. Store transition
            self.agent.store(state, option, action_latent,
                                        reward, next_state, beta, done)

            # 4. Statistics
            ep_reward                   += reward
            ep_steps                    += 1
            ep_stats["opt_rewards"][option] += reward
            ep_stats["opt_lengths"][option] += 1
            ep_stats["program_lengths"].append(program_length)
            if success:
                successes += 1

            # 5. Option switch?
            option_len += 1
            if beta or option_len >= self.macro_steps:
                option      = self.agent.select_option(next_state)
                option_len  = 0

            state = next_state
            self.total_steps += 1

            # 6. Learn
            if (self.current_episode >= self.warmup_episodes
                    and self.agent.replay.size() > self.agent.batch_size):
                stats = self.agent.update()
                for k, v in stats.items():
                    self.training_stats[k].append(v)

        return {"reward":       ep_reward,
                "length":       ep_steps,
                "success_rate": successes / max(ep_steps, 1),
                "avg_program_length": np.mean(ep_stats["program_lengths"]),
                **ep_stats}

    

    # ─────────────────────────────────────────────────────────────────────────
    #                          Checkpoint / CLI etc.
    # ─────────────────────────────────────────────────────────────────────────

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
                opt = self.agent.select_option(current_state)
                latent_action = self.agent.select_action(current_state, opt)
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
        """
        self.logger.info(f"Starting Option-Critic training for {max_episodes} episodes on task '{self.task}'")
        self.logger.info(f"Device: {self.device}, Macro steps: {self.macro_steps}")
        
        self.agent.train()
        start_time = time.time()
        
        for episode in range(max_episodes):
            # self.logger.info(f"=== Episode {episode + 1}/{max_episodes} ===")
            self.current_episode = episode
            
            # Train one episode
            episode_results = self.train_episode()
            
            # Track episode statistics
            self.episode_rewards.append(episode_results['reward'])
            
            # Update best reward
            if episode_results['reward'] > self.best_reward:
                self.best_reward = episode_results['reward']
                self.save_checkpoint('best_model.pt')
            
            # Store episode statistics BEFORE logging
            if 'episodes' not in self.training_stats:
                self.training_stats['episodes'] = []
            self.training_stats['episodes'].append(episode_results)
            
            # Logging
            if episode % log_frequency == 0:
                # Calculate moving averages
                recent_rewards = list(self.episode_rewards)[-log_frequency:]
                avg_reward = np.mean(recent_rewards)
                
                recent_episodes = self.training_stats['episodes'][-log_frequency:]
                avg_success = np.mean([ep['success_rate'] for ep in recent_episodes]) if recent_episodes else 0.0
                
                # Get current exploration parameters
                warmup_status = "WARMUP" if episode < self.warmup_episodes else "TRAINING"
                buffer_size = self.agent.replay.size()
                
                # For Option-Critic, we can show epsilon instead of noise std
                current_epsilon = getattr(self.agent, 'epsilon', 0.0)  # Current epsilon value
                
                self.logger.info(
                    f"Episode {episode:4d} [{warmup_status}] | "
                    f"Reward: {episode_results['reward']:7.3f} | "
                    f"Avg Reward: {avg_reward:7.3f} | "
                    f"Success Rate: {avg_success:5.3f} | "
                    f"Successful Programs: {avg_success * self.macro_steps:.1f}/{self.macro_steps} | "
                    f"Avg Program Length: {episode_results['avg_program_length']:5.1f} | "
                    f"Buffer Size: {buffer_size:6d} | "
                    f"Epsilon: {current_epsilon:6.4f}"
                )
                
                # Log program samples if enabled
                if self.config['debug']['log_program_samples'] and episode % (log_frequency) == 0:
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
        self.agent.save_model(filepath)
        
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
            opt = self.agent.select_option(state)
            latent_action = self.agent.select_action(state, opt)
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

# ─────────────────────────────────────────────────────────────────────────────
#                                CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
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
    trainer   = OCTrainer(args.vae_checkpoint, args.task, args.save_dir,
                          config_override, args.device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    training_config = trainer.config["training"]
    trainer.train(
        max_episodes=training_config['max_episodes'],
        eval_frequency=training_config['eval_frequency'],
        save_frequency=training_config['save_frequency'],
        log_frequency=training_config['log_frequency']
    )
    

    trainer.save_checkpoint("final_model.pt")