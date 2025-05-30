"""
DDPG Inference for HPRL Meta-Policy

This script loads a trained DDPG meta-policy and runs inference on Karel tasks,
outputting the generated programs and their performance scores.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ddpg_model import DDPGAgent
from models.vae import ProgramVAE
from models.program_executor import ProgramExecutor
from environments.karel_env import KarelEnvironment
from environments.karel_world import KarelWorld
from dsl.tokens import karel_tokens
from training.ddpg_config import get_config, load_config_from_file


class DDPGInference:
    """
    DDPG Inference Engine for HPRL
    
    Loads trained models and runs inference on Karel tasks, generating
    programs and evaluating their performance.
    """
    
    def __init__(
        self,
        ddpg_checkpoint_path: str,
        vae_checkpoint_path: str,
        task: str = 'harvester',
        device: str = 'auto',
        verbose: bool = True
    ):
        self.task = task
        self.verbose = verbose
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Initializing DDPG Inference for task '{task}' on {self.device}")
        
        # Load configuration
        self._load_config(ddpg_checkpoint_path)
        
        # Load models
        self._load_vae(vae_checkpoint_path)
        self._load_ddpg_agent(ddpg_checkpoint_path)
        
        # Setup environment and executor
        self._setup_environment()
        
        print(f"✅ DDPG Inference ready!")
    
    def _load_config(self, ddpg_checkpoint_path: str):
        """Load configuration from checkpoint directory"""
        checkpoint_dir = os.path.dirname(ddpg_checkpoint_path)
        config_path = os.path.join(checkpoint_dir, 'config.json')
        
        if os.path.exists(config_path):
            try:
                self.config = load_config_from_file(config_path)
                print(f"📋 Config loaded from {config_path}")
            except Exception as e:
                print(f"⚠️  Failed to load config, using defaults: {e}")
                self.config = get_config(self.task)
        else:
            print(f"⚠️  Config file not found, using defaults")
            self.config = get_config(self.task)
    
    def _load_vae(self, checkpoint_path: str):
        """Load pre-trained VAE model"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
        
        print(f"📦 Loading VAE from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize VAE
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
            rnn_type='GRU'
        ).to(self.device)
        
        # Load state dict
        self.vae.load_state_dict(checkpoint['program_vae_state_dict'])
        self.vae.eval()
        
        # Freeze parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        
        print(f"✅ VAE loaded successfully")
    
    def _load_ddpg_agent(self, checkpoint_path: str):
        """Load trained DDPG agent"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"DDPG checkpoint not found: {checkpoint_path}")
        
        print(f"🧠 Loading DDPG agent from {checkpoint_path}")
        
        # Initialize DDPG agent with config
        ddpg_config = self.config['ddpg']
        
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
        
        # Load checkpoint
        self.agent.load_models(checkpoint_path)
        self.agent.eval()  # Set to evaluation mode
        
        self.macro_steps = ddpg_config['macro_steps']
        self.max_program_steps = ddpg_config['max_program_steps']
        
        print(f"✅ DDPG agent loaded successfully")
    
    def _setup_environment(self):
        """Setup Karel environment and program executor"""
        self.karel_env = KarelEnvironment(task=self.task, grid_size=(8, 8))
        self.program_executor = ProgramExecutor(
            max_execution_steps=self.max_program_steps,
            device='cpu'
        )
        self.tokens = karel_tokens
        
        print(f"🌍 Environment setup for task '{self.task}'")
    
    def latent_to_program_tokens(self, latent_embedding: torch.Tensor) -> torch.Tensor:
        """Convert latent embedding to program tokens"""
        with torch.no_grad():
            if latent_embedding.dim() == 1:
                latent_embedding = latent_embedding.unsqueeze(0)
            
            # Apply VAE configuration
            vae_config = self.config['vae']
            scaled_latent = latent_embedding * vae_config['latent_scaling']
            
            if vae_config['use_latent_clipping']:
                clip_range = vae_config['latent_clip_range']
                scaled_latent = torch.clamp(scaled_latent, clip_range[0], clip_range[1])
            
            # Decode using VAE
            program_logits, _ = self.vae.vae.decode(
                scaled_latent,
                target_programs=None,
                deterministic=True
            )
            
            program_tokens = torch.argmax(program_logits, dim=-1)
            return program_tokens.squeeze(0) if latent_embedding.size(0) == 1 else program_tokens
    
    def tokens_to_program_string(self, tokens: torch.Tensor, max_length: int = 20) -> str:
        """Convert program tokens to readable string"""
        try:
            if isinstance(tokens, torch.Tensor):
                token_list = tokens.cpu().numpy().tolist()
            else:
                token_list = tokens
            
            # Truncate for readability
            token_list = token_list[:max_length]
            
            # Convert to string
            program_str = self.tokens.indices_to_string(token_list)
            
            # Clean up the string
            if len(program_str) > 200:
                program_str = program_str[:200] + "..."
            
            return program_str
            
        except Exception as e:
            return f"<DECODE_ERROR: {str(e)}>"
    
    def execute_program_tokens(
        self, 
        program_tokens: torch.Tensor, 
        karel_world: KarelWorld
    ) -> Dict[str, Any]:
        """Execute program tokens and return detailed results"""
        try:
            result = self.program_executor.execute_with_dsl(
                program_tokens,
                karel_world,
                return_traces=False
            )
            
            return {
                'success': result.get('success', False),
                'total_reward': result.get('total_reward', 0.0),
                'step_count': result.get('action_length', 0),
                'error': result.get('error', None),
                'execution_successful': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'total_reward': -0.5,
                'step_count': 0,
                'error': str(e),
                'execution_successful': False
            }
    
    def calculate_shaped_reward(self, execution_result: Dict[str, Any]) -> float:
        """Calculate shaped reward using same logic as training"""
        reward_config = self.config['rewards']
        
        base_reward = execution_result['total_reward']
        success = execution_result['success']
        step_count = execution_result['step_count']
        error = execution_result['error']
        
        shaped_reward = base_reward
        
        if success:
            shaped_reward += reward_config['success_bonus']
            if step_count > 0 and step_count < 30:
                efficiency_bonus = reward_config['efficiency_reward'] * (30 - step_count) / 30
                shaped_reward += efficiency_bonus
        else:
            if error and 'timeout' in str(error).lower():
                shaped_reward += reward_config['timeout_penalty']
            elif error and ('invalid' in str(error).lower() or 'parse' in str(error).lower()):
                shaped_reward += reward_config['invalid_program_penalty']
            else:
                shaped_reward += reward_config['failure_penalty']
        
        shaped_reward += step_count * reward_config['step_penalty']
        
        return shaped_reward
    
    def run_single_episode(self, episode_num: int = 0, add_noise: bool = False) -> Dict[str, Any]:
        """Run a single episode and return detailed results"""
        # Reset environment
        obs, info = self.karel_env.reset()
        current_state = self._get_state_array(obs)
        
        episode_results = {
            'episode_num': episode_num,
            'task': self.task,
            'macro_steps': [],
            'total_reward': 0.0,
            'total_steps': 0,
            'successful_programs': 0,
            'programs_generated': [],
            'overall_success': False
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎮 Episode {episode_num} - Task: {self.task}")
            print(f"{'='*60}")
        
        for macro_step in range(self.macro_steps):
            if self.verbose:
                print(f"\n--- Macro Step {macro_step + 1}/{self.macro_steps} ---")
            
            # Get latent action from meta-policy
            latent_action = self.agent.select_action(current_state, add_noise=add_noise)
            latent_tensor = torch.FloatTensor(latent_action).to(self.device)
            
            # Decode to program tokens
            program_tokens = self.latent_to_program_tokens(latent_tensor)
            program_string = self.tokens_to_program_string(program_tokens)
            
            # Create fresh Karel world
            karel_world = KarelWorld(task=self.task, grid_size=(8, 8), timeout_steps=self.max_program_steps)
            karel_world.reset()
            karel_world.state = current_state.copy()
            
            # Execute program
            execution_result = self.execute_program_tokens(program_tokens, karel_world)
            shaped_reward = self.calculate_shaped_reward(execution_result)
            
            # Store macro step results
            macro_step_result = {
                'macro_step': macro_step + 1,
                'latent_embedding': latent_action.tolist(),
                'latent_norm': float(np.linalg.norm(latent_action)),
                'program_tokens': program_tokens.cpu().numpy().tolist()[:15],
                'program_string': program_string,
                'execution_result': execution_result,
                'shaped_reward': shaped_reward,
                'next_state_available': not karel_world.done
            }
            
            episode_results['macro_steps'].append(macro_step_result)
            episode_results['programs_generated'].append(program_string)
            episode_results['total_reward'] += shaped_reward
            episode_results['total_steps'] += execution_result['step_count']
            
            if execution_result['success']:
                episode_results['successful_programs'] += 1
            
            if self.verbose:
                print(f"  🤖 Latent norm: {macro_step_result['latent_norm']:.3f}")
                print(f"  📝 Program: {program_string}")
                print(f"  ✅ Success: {execution_result['success']}")
                print(f"  📊 Base reward: {execution_result['total_reward']:.3f}")
                print(f"  🎯 Shaped reward: {shaped_reward:.3f}")
                print(f"  ⏱️  Steps: {execution_result['step_count']}")
                if execution_result['error']:
                    print(f"  ❌ Error: {execution_result['error']}")
            
            # Update state
            current_state = karel_world.get_state()
            
            # Check if task is completed
            if karel_world.done:
                episode_results['overall_success'] = True
                if self.verbose:
                    print(f"  🎉 Task completed early!")
                break
        
        # Calculate final statistics
        episode_results['success_rate'] = episode_results['successful_programs'] / self.macro_steps
        episode_results['avg_program_length'] = episode_results['total_steps'] / self.macro_steps if self.macro_steps > 0 else 0
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 Episode Summary:")
            print(f"  Total Reward: {episode_results['total_reward']:.3f}")
            print(f"  Success Rate: {episode_results['success_rate']:.3f}")
            print(f"  Successful Programs: {episode_results['successful_programs']}/{self.macro_steps}")
            print(f"  Total Steps: {episode_results['total_steps']}")
            print(f"  Avg Program Length: {episode_results['avg_program_length']:.1f}")
            print(f"  Overall Success: {episode_results['overall_success']}")
            print(f"{'='*60}")
        
        return episode_results
    
    def run_evaluation(self, num_episodes: int = 10, add_noise: bool = False) -> Dict[str, Any]:
        """Run multiple episodes and return aggregate statistics"""
        print(f"\n🔬 Running evaluation: {num_episodes} episodes on task '{self.task}'")
        print(f"   Exploration noise: {'ON' if add_noise else 'OFF'}")
        
        all_episodes = []
        total_rewards = []
        success_rates = []
        program_lengths = []
        overall_successes = 0
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_result = self.run_single_episode(episode, add_noise=add_noise)
            
            all_episodes.append(episode_result)
            total_rewards.append(episode_result['total_reward'])
            success_rates.append(episode_result['success_rate'])
            program_lengths.append(episode_result['avg_program_length'])
            
            if episode_result['overall_success']:
                overall_successes += 1
            
            # Progress indicator
            if not self.verbose and (episode + 1) % max(1, num_episodes // 10) == 0:
                print(f"  Progress: {episode + 1}/{num_episodes} episodes completed")
        
        elapsed_time = time.time() - start_time
        
        # Calculate aggregate statistics
        evaluation_results = {
            'num_episodes': num_episodes,
            'task': self.task,
            'total_time': elapsed_time,
            'episodes': all_episodes,
            
            # Aggregate metrics
            'avg_total_reward': np.mean(total_rewards),
            'std_total_reward': np.std(total_rewards),
            'min_total_reward': np.min(total_rewards),
            'max_total_reward': np.max(total_rewards),
            
            'avg_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            
            'avg_program_length': np.mean(program_lengths),
            'std_program_length': np.std(program_lengths),
            
            'overall_success_rate': overall_successes / num_episodes,
            'task_completion_rate': overall_successes / num_episodes,
            
            # Best episode
            'best_episode_idx': np.argmax(total_rewards),
            'best_episode_reward': np.max(total_rewards),
            
            # Program diversity
            'unique_programs': len(set([p for ep in all_episodes for p in ep['programs_generated']])),
            'total_programs_generated': sum([len(ep['programs_generated']) for ep in all_episodes])
        }
        
        return evaluation_results
    
    def print_evaluation_summary(self, eval_results: Dict[str, Any]):
        """Print a comprehensive evaluation summary"""
        print(f"\n{'🎯 EVALUATION RESULTS ':=^80}")
        print(f"Task: {eval_results['task']}")
        print(f"Episodes: {eval_results['num_episodes']}")
        print(f"Total Time: {eval_results['total_time']:.2f}s")
        print(f"{'':=^80}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  Average Total Reward:    {eval_results['avg_total_reward']:8.3f} ± {eval_results['std_total_reward']:.3f}")
        print(f"  Reward Range:           [{eval_results['min_total_reward']:7.3f}, {eval_results['max_total_reward']:7.3f}]")
        print(f"  Average Success Rate:    {eval_results['avg_success_rate']:8.3f} ± {eval_results['std_success_rate']:.3f}")
        print(f"  Task Completion Rate:    {eval_results['task_completion_rate']:8.3f}")
        print(f"  Avg Program Length:      {eval_results['avg_program_length']:8.1f} ± {eval_results['std_program_length']:.1f}")
        
        print(f"\n🔍 PROGRAM DIVERSITY:")
        print(f"  Total Programs Generated: {eval_results['total_programs_generated']}")
        print(f"  Unique Programs:          {eval_results['unique_programs']}")
        print(f"  Diversity Ratio:          {eval_results['unique_programs']/eval_results['total_programs_generated']:.3f}")
        
        # Show best episode
        best_idx = eval_results['best_episode_idx']
        best_episode = eval_results['episodes'][best_idx]
        print(f"\n🏆 BEST EPISODE (#{best_idx}):")
        print(f"  Total Reward: {best_episode['total_reward']:.3f}")
        print(f"  Success Rate: {best_episode['success_rate']:.3f}")
        print(f"  Programs Generated:")
        for i, program in enumerate(best_episode['programs_generated']):
            success = best_episode['macro_steps'][i]['execution_result']['success']
            status = "✅" if success else "❌"
            print(f"    {i+1}. {status} {program}")
        
        print(f"{'':=^80}")
    
    def save_results(self, eval_results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file"""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(eval_results)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {output_path}")
    
    def _get_state_array(self, obs) -> np.ndarray:
        """Extract state array from environment observation"""
        if isinstance(obs, tuple):
            for item in obs:
                if isinstance(item, np.ndarray) and len(item.shape) == 3:
                    return item
            return obs[0]
        else:
            return obs


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='DDPG Inference for HPRL')
    
    parser.add_argument('--ddpg-checkpoint', type=str, required=True,
                       help='Path to trained DDPG checkpoint')
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                       help='Path to pre-trained VAE checkpoint')
    parser.add_argument('--task', type=str, default='harvester',
                       choices=['harvester', 'cleanHouse', 'fourCorners', 'randomMaze', 'stairClimber', 'topOff'],
                       help='Karel task to evaluate on')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results (JSON format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output (show detailed episode information)')
    parser.add_argument('--add-noise', action='store_true',
                       help='Add exploration noise during inference')
    parser.add_argument('--single-episode', action='store_true',
                       help='Run only a single episode with detailed output')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = DDPGInference(
        ddpg_checkpoint_path=args.ddpg_checkpoint,
        vae_checkpoint_path=args.vae_checkpoint,
        task=args.task,
        device=args.device,
        verbose=args.verbose or args.single_episode
    )
    
    if args.single_episode:
        # Run single episode with detailed output
        print(f"\n🎮 Running single episode demonstration")
        episode_result = inference.run_single_episode(0, add_noise=args.add_noise)
        
        # Save single episode if output specified
        if args.output:
            inference.save_results({'single_episode': episode_result}, args.output)
    
    else:
        # Run full evaluation
        eval_results = inference.run_evaluation(
            num_episodes=args.episodes,
            add_noise=args.add_noise
        )
        
        # Print summary
        inference.print_evaluation_summary(eval_results)
        
        # Save results if output specified
        if args.output:
            inference.save_results(eval_results, args.output)
        else:
            # Default output file
            default_output = f"./results/ddpg_inference_{args.task}_{args.episodes}eps.json"
            inference.save_results(eval_results, default_output)


if __name__ == "__main__":
    main()