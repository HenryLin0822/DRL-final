"""
ppo_trainer.py ─ PPO Trainer for HPRL Meta-Policy
================================================

Structure and logging style mirror *ddpg_trainer.py* so that scripts,
CLI flags, checkpoints, and plotting utilities remain interchangeable.

Major changes vs. DDPGTrainer
─────────────────────────────
• Uses **on-policy** `PPOAgent` (no replay buffer / warm-up).
• Stores `(state, action, logp, reward, done, value)` in the roll-out
  buffer each macro-step.
• Calls `agent.update()` automatically whenever the roll-out buffer is
  full (≈ every `rollout_steps / macro_steps` episodes).
• Logs policy-loss / value-loss / entropy instead of actor/critic/Q.

Everything else (VAE decoding, Karel environment, reward shaping, plots,
checkpoint layout, CLI) is kept **identical** to the DDPG trainer so you
can swap algorithms with a single import/flag change.
"""

from __future__ import annotations
import os, sys, time, json, logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque

import numpy as np
import torch
import matplotlib.pyplot as plt

# add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ppo_model import PPOAgent                       # ← PPO model
from models.vae          import ProgramVAE
from models.program_executor import ProgramExecutor
from environments.karel_env   import KarelEnvironment
from environments.karel_world import KarelWorld
from dsl.tokens import karel_tokens

from training.ppo_config import get_config, print_config, save_config


# ─────────────────────────────────────────────────────────────
#   PPOTrainer
# ─────────────────────────────────────────────────────────────
class PPOTrainer:
    """
    PPO Trainer for the HPRL meta-policy that outputs latent program
    embeddings which the frozen VAE decodes into executable Karel code.
    """

    def __init__(
        self,
        vae_checkpoint_path: str,
        task: str = "harvester",
        save_dir: str = "./checkpoints/ppo",
        config_override: Optional[Dict[str, Any]] = None,
        device: str = "auto",
    ):
        self.save_dir, self.task = save_dir, task
        os.makedirs(save_dir, exist_ok=True)

        # device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # cfg
        self.config = get_config(task, config_override)

        # logging
        self._setup_logging()
        print_config(self.config)
        save_config(self.config, os.path.join(save_dir, "config.json"))

        # VAE
        self.logger.info("Loading VAE from %s", vae_checkpoint_path)
        self._load_vae(vae_checkpoint_path)

        # PPO agent
        self.logger.info("Initializing PPO agent")
        self._init_ppo_agent()

        # env + executor
        self.logger.info("Setting up Karel environment for task: %s", self.task)
        self._setup_environment()

        # bookkeeping
        self.current_episode = 0
        self.total_steps = 0
        self.best_reward = -float("inf")

        # stats
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        self.training_stats: defaultdict = defaultdict(list)

        self.logger.info("PPO Trainer initialized successfully")

    # ──────────────────────────────────────────────────────
    #   helpers
    # ──────────────────────────────────────────────────────
    def _setup_logging(self):
        """Setup logging configuration (same format as DDPG trainer)."""
        log_file = os.path.join(self.save_dir, "training.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    # ---------- VAE ----------
    def _load_vae(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        _ = ckpt.get("config", {})                     # not used but kept

        self.vae = ProgramVAE(
            vocab_size=35,
            embedding_dim=64,
            hidden_size=256,
            latent_dim=64,
            state_shape=(8, 8, 8),
            num_actions=6,
            max_program_length=50,
            max_demo_length=10,
            dropout=0.0,
            rnn_type="GRU",
        ).to(self.device)

        self.vae.load_state_dict(ckpt["program_vae_state_dict"])
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.logger.info(
            "VAE loaded with %,d parameters",
            sum(p.numel() for p in self.vae.parameters()),
        )

    # ---------- PPO agent ----------
    def _init_ppo_agent(self):
        ppo_cfg = self.config["ppo"]
        self.agent = PPOAgent(
            state_shape=(8, 8, 8),
            latent_dim=64,
            lr=ppo_cfg["lr"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_eps=ppo_cfg["clip_eps"],
            entropy_coef=ppo_cfg["entropy_coef"],
            value_coef=ppo_cfg["value_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            rollout_steps=ppo_cfg["rollout_steps"],
            ppo_epochs=ppo_cfg["ppo_epochs"],
            minibatch_size=ppo_cfg["minibatch_size"],
            device=self.device,
        )

        self.macro_steps = ppo_cfg["macro_steps"]
        self.max_program_steps = ppo_cfg["max_program_steps"]

        self.logger.info("PPO agent initialized with %d macro-steps", self.macro_steps)

    # ---------- Env / executor ----------
    def _setup_environment(self):
        self.karel_env = KarelEnvironment(task=self.task, grid_size=(8, 8))
        self.program_executor = ProgramExecutor(
            max_execution_steps=self.max_program_steps, device="cpu"
        )
        self.tokens = karel_tokens
        self.logger.info("Karel environment & executor ready")

    # ---------- utils ----------
    def _get_state_array(self, obs):
        if isinstance(obs, tuple):
            for x in obs:
                if isinstance(x, np.ndarray) and x.ndim == 3:
                    return x
            return obs[0]
        return obs

    def latent_to_program_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            cfg = self.config["vae"]
            latent = latent * cfg["latent_scaling"]
            if cfg["use_latent_clipping"]:
                lo, hi = cfg["latent_clip_range"]
                latent = torch.clamp(latent, lo, hi)

            logits, _ = self.vae.vae.decode(latent, None, deterministic=True)
            program_tokens = torch.argmax(logits, dim=-1)
            if self.config['debug']['log_program_samples'] and np.random.random() < 0.1:
                # Occasionally log the decoded program for debugging
                token_list = program_tokens[0].cpu().numpy().tolist()[:10]  # First 10 tokens
                program_str = self.tokens.indices_to_string(token_list)
                self.logger.debug(f"Decoded program sample: {program_str}")

            return program_tokens.squeeze(0) if latent.size(0) == 1 else program_tokens
        
    def execute_latent_program(
        self, 
        latent_embedding: torch.Tensor, 
        karel_world: KarelWorld
    ) -> Tuple[float, bool, int]:
        """
        Execute a program decoded from latent embedding with improved reward calculation
        
        Args:
            latent_embedding: [64] - latent program embedding
            karel_world: Karel world instance
            
        Returns:
            Tuple of (reward, success, step_count)
        """
        reward_config = self.config['rewards']
        
        # Decode latent to program tokens
        program_tokens = self.latent_to_program_tokens(latent_embedding)
        
        # Log latent statistics if enabled
        if self.config['debug']['log_latent_stats'] and np.random.random() < 0.05:
            latent_np = latent_embedding.cpu().numpy()
            self.logger.debug(f"Latent stats - norm: {np.linalg.norm(latent_np):.3f}, "
                            f"mean: {np.mean(latent_np):.3f}, std: {np.std(latent_np):.3f}")
        
        try:
            # Execute program using DSL executor
            result = self.program_executor.execute_with_dsl(
                program_tokens, 
                karel_world, 
                return_traces=False
            )
            
            # Extract basic results
            base_reward = result.get('total_reward', 0.0)
            success = result.get('success', False)
            step_count = result.get('action_length', 0)
            error = result.get('error', None)
            
            # Calculate shaped reward
            shaped_reward = base_reward
            
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
                                f"Base reward: {base_reward:.3f}, Shaped reward: {shaped_reward:.3f}, "
                                f"Error: {error}")
            
            return shaped_reward, success, step_count
            
        except Exception as e:
            self.logger.warning(f"Program execution failed: {e}")
            return reward_config['failure_penalty'], False, 0

    # ──────────────────────────────────────────────────────
    #   train / eval
    # ──────────────────────────────────────────────────────
    def train_episode(self, ep: int) -> Dict[str, Any]:
        obs, _ = self.karel_env.reset()
        state = self._get_state_array(obs)

        ep_reward, ep_steps, successes = 0.0, 0, 0
        stats = {"macro_rewards": [], "program_successes": [], "program_lengths": []}

        for _ in range(self.macro_steps):
            act, logp, value = self.agent.select_action(state, add_noise=True)
            latent_t = torch.FloatTensor(act).to(self.device)

            # new world copy
            world = KarelWorld(self.task, (8, 8), timeout_steps=self.max_program_steps)
            world.reset()
            world.state = state.copy()

            r, success, prog_steps = self.execute_latent_program(latent_t, world)
            next_state = world.get_state()

            # store transition
            self.agent.store(state, act, logp, r, success, value)

            ep_reward += r
            ep_steps += prog_steps
            successes += int(success)

            stats["macro_rewards"].append(r)
            stats["program_successes"].append(success)
            stats["program_lengths"].append(prog_steps)

            state = next_state
            self.total_steps += 1
            if world.done:
                break

        # PPO update (runs only when rollout buffer is full)
        train_stats = self.agent.update()
        if train_stats:
            for k, v in train_stats.items():
                self.training_stats[k].append(v)

        return {
            "episode_reward": ep_reward,
            "episode_steps": ep_steps,
            "successful_programs": successes,
            "success_rate": successes / self.macro_steps,
            "avg_program_length": np.mean(stats["program_lengths"]),
            "macro_rewards": stats["macro_rewards"],
        }

    def evaluate(self, n: int = 10) -> Dict[str, float]:
        self.agent.eval()

        R, S, L = [], [], []
        for _ in range(n):
            obs, _ = self.karel_env.reset()
            state = self._get_state_array(obs)

            ep_r, succ = 0.0, 0
            prog_steps_sum = 0
            for _ in range(self.macro_steps):
                a, _, _ = self.agent.select_action(state, add_noise=False)
                r, success, prog_steps = self.execute_latent_program(
                    torch.FloatTensor(a).to(self.device),
                    KarelWorld(self.task, (8, 8), timeout_steps=self.max_program_steps),
                )
                ep_r += r
                prog_steps_sum += prog_steps
                succ += int(success)
            R.append(ep_r)
            S.append(succ / self.macro_steps)
            L.append(prog_steps_sum / self.macro_steps)

        self.agent.train()
        return {
            "avg_reward": np.mean(R),
            "std_reward": np.std(R),
            "avg_success_rate": np.mean(S),
            "avg_program_length": np.mean(L),
            "eval_episodes": n,
        }

    def _log_program_samples(self):
        """Log three example programs decoded from the current policy."""
        self.logger.info("=== PROGRAM SAMPLES ===")
        for i in range(3):
            obs, _ = self.karel_env.reset()
            state = self._get_state_array(obs)
            latent, _, _ = self.agent.select_action(state, add_noise=False)
            latent_t = torch.FloatTensor(latent).to(self.device)
            tokens = self.latent_to_program_tokens(latent_t)
            try:
                prog = self.tokens.indices_to_string(tokens.cpu().numpy().tolist()[:15])
                self.logger.info(f"Sample {i+1}: {prog}")
            except Exception as e:                         # very rare decode error
                self.logger.info(f"Sample {i+1}: <decode-error> {e}")
        self.logger.info("=====================")

    def train(
        self,
        max_episodes: int = 1000,
        eval_frequency: int = 50,
        save_frequency: int = 100,
        log_frequency: int = 10,
    ):
        self.logger.info(
            "Starting PPO training for %d episodes on task '%s'", max_episodes, self.task
        )
        self.logger.info("Device: %s | Macro-steps: %d", self.device, self.macro_steps)

        start = time.time()
        for ep in range(max_episodes):
            self.current_episode = ep
            ep_res = self.train_episode(ep)

            self.episode_rewards.append(ep_res["episode_reward"])
            self.episode_lengths.append(ep_res["episode_steps"])

            if ep_res["episode_reward"] > self.best_reward:
                self.best_reward = ep_res["episode_reward"]
                self.save_checkpoint("best_model.pt")

            # logging
            if ep % log_frequency == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-log_frequency:])
                pol_loss = (np.mean(self.training_stats["policy_loss"][-log_frequency:])
                            if self.training_stats["policy_loss"] else 0.0)
                val_loss = (np.mean(self.training_stats["value_loss"][-log_frequency:])
                            if self.training_stats["value_loss"] else 0.0)
                entropy  = (np.mean(self.training_stats["entropy"][-log_frequency:])
                            if self.training_stats["entropy"] else 0.0)

                self.logger.info(
                    f"Episode {ep:4d} [TRAINING] | "
                    f"Reward: {ep_res['episode_reward']:7.3f} | "
                    f"Avg Reward: {avg_reward:7.3f} | "
                    f"Success Rate: {ep_res['success_rate']:5.3f} | "
                    f"Successful Programs: {ep_res['successful_programs']}/{self.macro_steps} | "
                    f"Avg Program Length: {ep_res['avg_program_length']:5.1f} | "
                    f"Buffer Size: {len(self.agent.buffer.states):6d} | "
                    f"π_loss: {pol_loss:6.4f} | "
                    f"V_loss: {val_loss:6.4f} | "
                    f"Entropy: {entropy:6.4f} | "
                )
            if self.config["debug"]["log_program_samples"] and ep % (log_frequency * 5) == 0:
                self._log_program_samples()
            # stats bookkeeping
            self.training_stats.setdefault("episodes", []).append(ep_res)

            # eval
            if ep % eval_frequency == 0 and ep > 0:
                ev = self.evaluate()
                self.logger.info(
                    "Eval | AvgR %.3f ± %.3f | Succ %.3f",
                    ev["avg_reward"],
                    ev["std_reward"],
                    ev["avg_success_rate"],
                )
                self.training_stats.setdefault("evaluations", []).append({**ev, "episode": ep})

            # checkpoint
            if ep % save_frequency == 0 and ep > 0:
                self.save_checkpoint(f"ep_{ep}.pt")
                self.save_training_stats()

        # final
        final = self.evaluate(50)
        self.logger.info(
            "Final Eval | AvgR %.3f ± %.3f | Succ %.3f",
            final["avg_reward"],
            final["std_reward"],
            final["avg_success_rate"],
        )
        self.save_checkpoint("final_model.pt")
        self.save_training_stats()
        self.logger.info("Training finished in %.2f h", (time.time() - start) / 3600)

    # ──────────────────────────────────────────────────────
    #   checkpoint / plots  (identical logic to DDPGTrainer)
    # ──────────────────────────────────────────────────────
    def save_checkpoint(self, fname: str):
        path = os.path.join(self.save_dir, fname)
        torch.save({"agent": "ppo"}, path)               # model stored below
        self.agent.save_models(path)
        torch.save(
            {
                "episode": self.current_episode,
                "total_steps": self.total_steps,
                "best_reward": self.best_reward,
                "task": self.task,
                "config": self.config,
                "training_stats": dict(self.training_stats),
            },
            path.replace(".pt", "_training_info.pt"),
        )

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.agent.load_models(path)
        info = torch.load(path.replace(".pt", "_training_info.pt"), map_location=self.device)
        self.current_episode = info.get("episode", 0)
        self.total_steps = info.get("total_steps", 0)
        self.best_reward = info.get("best_reward", -float("inf"))
        self.training_stats = defaultdict(list, info.get("training_stats", {}))
        self.logger.info("Checkpoint loaded from %s", path)

    def save_training_stats(self):
        p = os.path.join(self.save_dir, "training_stats.json")
        with open(p, "w") as f:
            json.dump(self.training_stats, f, indent=2)

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

# ─────────────────────────────────────────────────────────────
#   CLI
# ─────────────────────────────────────────────────────────────
def main():
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="PPO Training for HPRL")
    parser.add_argument("--vae-checkpoint", required=True, type=str)
    parser.add_argument("--task", default="harvester", type=str)
    parser.add_argument("--save-dir", default="./checkpoints/ppo", type=str)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config-override", type=str, default=None)
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

    trainer = PPOTrainer(
        vae_checkpoint_path=args.vae_checkpoint,
        task=args.task,
        save_dir=args.save_dir,
        config_override=config_override,
        device=args.device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")

    t_cfg = trainer.config["training"]
    trainer.train(
        t_cfg["max_episodes"],
        t_cfg["eval_frequency"],
        t_cfg["save_frequency"],
        t_cfg["log_frequency"],
    )
    trainer.plot_training_progress()

    print("Training completed!")

if __name__ == "__main__":
    main()