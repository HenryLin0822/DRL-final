"""
Minimal DDPG Inference for HPRL - Final Program and Rewards Only

This script provides a clean, minimal interface that only outputs:
1. The final concatenated program P = ⟨ρ_1, ρ_2, ..., ρ_{|H|}⟩
2. Base reward and shaped reward for the final program
3. Success status
"""

import os
import sys
import torch
import numpy as np
import argparse
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ddpg_model import DDPGAgent
from models.vae import ProgramVAE
from models.program_executor import ProgramExecutor
from environments.karel_env import KarelEnvironment
from environments.karel_world import KarelWorld
from dsl.tokens import karel_tokens
from training.ddpg_config import get_config, load_config_from_file


class MinimalDDPGInference:
    """Minimal DDPG Inference - Only Final Program and Rewards"""
    
    def __init__(self, ddpg_checkpoint_path: str, vae_checkpoint_path: str, task: str = 'harvester', device: str = 'auto'):
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        
        # Load models and config
        self._load_config(ddpg_checkpoint_path)
        self._load_vae(vae_checkpoint_path)
        self._load_ddpg_agent(ddpg_checkpoint_path)
        self._setup_environment()
    
    def _load_config(self, checkpoint_path: str):
        """Load configuration"""
        config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
        if os.path.exists(config_path):
            self.config = load_config_from_file(config_path)
        else:
            self.config = get_config(self.task)
    
    def _load_vae(self, checkpoint_path: str):
        """Load VAE model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.vae = ProgramVAE(
            vocab_size=35, embedding_dim=64, hidden_size=256, latent_dim=64,
            state_shape=(8, 8, 8), num_actions=6, max_program_length=50,
            max_demo_length=10, dropout=0.0, rnn_type='GRU'
        ).to(self.device)
        self.vae.load_state_dict(checkpoint['program_vae_state_dict'])
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def _load_ddpg_agent(self, checkpoint_path: str):
        """Load DDPG agent"""
        ddpg_config = self.config['ddpg']
        self.agent = DDPGAgent(
            state_shape=(8, 8, 8), latent_dim=64,
            actor_lr=ddpg_config['actor_lr'], critic_lr=ddpg_config['critic_lr'],
            tau=ddpg_config['tau'], gamma=ddpg_config['gamma'],
            buffer_size=ddpg_config['buffer_size'], batch_size=ddpg_config['batch_size'],
            noise_std=ddpg_config['noise_std'], noise_decay=ddpg_config['noise_decay'],
            device=self.device
        )
        self.agent.load_models(checkpoint_path)
        self.agent.eval()
        self.macro_steps = ddpg_config['macro_steps']
        self.max_program_steps = ddpg_config['max_program_steps']
    
    def _setup_environment(self):
        """Setup environment"""
        self.karel_env = KarelEnvironment(task=self.task, grid_size=(8, 8))
        self.program_executor = ProgramExecutor(max_execution_steps=self.max_program_steps, device='cpu')
        self.tokens = karel_tokens
    
    def _get_state_array(self, obs) -> np.ndarray:
        """Extract state array from observation"""
        if isinstance(obs, tuple):
            for item in obs:
                if isinstance(item, np.ndarray) and len(item.shape) == 3:
                    return item
            return obs[0]
        return obs
    
    def truncate_at_program_end(self, indices: List[int]) -> List[int]:
        """Truncate program at first m)"""
        if len(indices) < 4 or indices[0] != 0 or indices[1] != 1 or indices[2] != 2:
            return indices
        for i in range(3, len(indices)):
            if indices[i] == 3:  # Found m)
                return indices[:i+1]
        return indices
    
    def latent_to_program_tokens(self, latent_embedding: torch.Tensor) -> torch.Tensor:
        """Convert latent to program tokens with truncation"""
        with torch.no_grad():
            if latent_embedding.dim() == 1:
                latent_embedding = latent_embedding.unsqueeze(0)
            
            # Apply VAE config
            vae_config = self.config['vae']
            scaled_latent = latent_embedding * vae_config['latent_scaling']
            if vae_config['use_latent_clipping']:
                clip_range = vae_config['latent_clip_range']
                scaled_latent = torch.clamp(scaled_latent, clip_range[0], clip_range[1])
            
            # Decode and truncate
            program_logits, _ = self.vae.vae.decode(scaled_latent, target_programs=None, deterministic=True)
            program_tokens = torch.argmax(program_logits, dim=-1)
            
            # Truncate at first program end
            raw_indices = program_tokens[0].cpu().numpy().tolist()
            padding_idx = self.tokens.get_padding_index()
            vocab_size = self.tokens.vocab_size
            valid_indices = [i for i in raw_indices if i != padding_idx and i < vocab_size]
            truncated_indices = self.truncate_at_program_end(valid_indices)
            
            return torch.tensor(truncated_indices, dtype=torch.long, device=program_tokens.device)
    
    def tokens_to_program_string(self, tokens: torch.Tensor) -> str:
        """Convert tokens to program string"""
        try:
            token_list = tokens.cpu().numpy().tolist() if isinstance(tokens, torch.Tensor) else tokens
            padding_idx = self.tokens.get_padding_index()
            token_list = [t for t in token_list if t != padding_idx]
            return self.tokens.indices_to_string(token_list)
        except Exception as e:
            return f"<DECODE_ERROR: {str(e)}>"
    
    def concatenate_programs(self, program_list: List[str]) -> str:
        """Concatenate programs: P = ⟨ρ_1, ρ_2, ..., ρ_{|H|}⟩"""
        if not program_list:
            return "DEF run m( m)"
        
        statements = []
        for program in program_list:
            try:
                tokens = self.tokens.string_to_tokens(program.strip())
                if (len(tokens) >= 4 and tokens[0] == 'DEF' and tokens[1] == 'run' and 
                    tokens[2] == 'm(' and tokens[-1] == 'm)'):
                    inner_tokens = tokens[3:-1]
                    if inner_tokens:
                        statements.append(' '.join(inner_tokens))
            except:
                continue
        
        if statements:
            combined_statements = ' '.join(statements)
            return f"DEF run m( {combined_statements} m)"
        else:
            return "DEF run m( m)"
    
    def calculate_rewards(self, execution_result: Dict) -> Dict:
        """Calculate both base and shaped rewards"""
        reward_config = self.config['rewards']
        base_reward = execution_result['total_reward']
        success = execution_result['success']
        step_count = execution_result['step_count']
        error = execution_result['error']
        execution_successful = execution_result['execution_successful']
        
        shaped_reward = base_reward
        
        # Check if executable
        is_executable = (execution_successful and 
                        (error is None or not any(err_type in str(error).lower() 
                         for err_type in ['parse', 'invalid', 'syntax', 'format'])))
        
        # Apply shaping
        if is_executable:
            shaped_reward += reward_config['executable_bonus']
        
        if success:
            shaped_reward += reward_config['success_bonus']
            if 0 < step_count < 30:
                shaped_reward += reward_config['efficiency_reward'] * (30 - step_count) / 30
        else:
            if error and 'timeout' in str(error).lower():
                shaped_reward += reward_config['timeout_penalty']
            elif error and ('invalid' in str(error).lower() or 'parse' in str(error).lower()):
                shaped_reward += reward_config['invalid_program_penalty']
            else:
                shaped_reward += reward_config['failure_penalty']
        
        shaped_reward += step_count * reward_config['step_penalty']
        
        return {
            'base_reward': base_reward,
            'shaped_reward': shaped_reward,
            'is_executable': is_executable
        }
    
    def execute_final_program(self, final_program: str) -> Dict:
        """Execute final concatenated program"""
        try:
            program_tokens = torch.tensor(self.tokens.string_to_indices(final_program), 
                                        dtype=torch.long, device=self.device)
            karel_world = KarelWorld(task=self.task, grid_size=(8, 8), 
                                   timeout_steps=self.max_program_steps * self.macro_steps)
            karel_world.reset()
            
            result = self.program_executor.execute_with_dsl(program_tokens, karel_world, return_traces=False)
            
            return {
                'success': result.get('success', False),
                'total_reward': result.get('total_reward', 0.0),
                'step_count': result.get('action_length', 0),
                'error': result.get('error', None),
                'execution_successful': True
            }
        except Exception as e:
            return {
                'success': False, 'total_reward': -1.0, 'step_count': 0,
                'error': str(e), 'execution_successful': False
            }
    
    def run_inference(self) -> Dict:
        """
        Run complete HPRL inference and return final program with rewards
        
        Returns:
            Dict with final_program, base_reward, shaped_reward, success
        """
        # Step 1: Get initial state
        obs, _ = self.karel_env.reset()
        current_state = self._get_state_array(obs)
        
        # Step 2: Generate individual programs ρ_1, ρ_2, ..., ρ_{|H|}
        individual_programs = []
        for macro_step in range(self.macro_steps):
            # Meta-policy predicts latent
            latent_action = self.agent.select_action(current_state, add_noise=False)
            latent_tensor = torch.FloatTensor(latent_action).to(self.device)
            
            # Decode to program
            program_tokens = self.latent_to_program_tokens(latent_tensor)
            program_string = self.tokens_to_program_string(program_tokens)
            individual_programs.append(program_string)
            
            # Execute to get next state
            karel_world = KarelWorld(task=self.task, grid_size=(8, 8), timeout_steps=self.max_program_steps)
            karel_world.reset()
            karel_world.state = current_state.copy()
            
            try:
                self.program_executor.execute_with_dsl(program_tokens, karel_world, return_traces=False)
                current_state = karel_world.get_state()
            except:
                pass  # Continue with current state if execution fails
        
        # Step 3: Create final program P = ⟨ρ_1, ρ_2, ..., ρ_{|H|}⟩
        final_program = self.concatenate_programs(individual_programs)
        
        # Execute final program
        execution_result = self.execute_final_program(final_program)
        reward_info = self.calculate_rewards(execution_result)
        
        return {
            'individual_programs': individual_programs,
            'final_program': final_program,
            'base_reward': reward_info['base_reward'],
            'shaped_reward': reward_info['shaped_reward'],
            'success': execution_result['success'],
            'step_count': execution_result['step_count'],
            'is_executable': reward_info['is_executable'],
            'error': execution_result.get('error', None)
        }


def main():
    """Minimal main function"""
    parser = argparse.ArgumentParser(description='Minimal DDPG Inference for HPRL')
    parser.add_argument('--ddpg-checkpoint', type=str, required=True, help='Path to DDPG checkpoint')
    parser.add_argument('--vae-checkpoint', type=str, required=True, help='Path to VAE checkpoint')
    parser.add_argument('--task', type=str, default='harvester', 
                       choices=['harvester', 'cleanHouse', 'fourCorners', 'randomMaze', 'stairClimber', 'topOff'])
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--verbose', action='store_true', help='Show individual programs')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = MinimalDDPGInference(
        ddpg_checkpoint_path=args.ddpg_checkpoint,
        vae_checkpoint_path=args.vae_checkpoint,
        task=args.task,
        device=args.device
    )
    
    print(f"🎯 HPRL Minimal Inference - Task: {args.task}")
    print("=" * 60)
    
    total_base_reward = 0.0
    total_shaped_reward = 0.0
    successes = 0
    
    for episode in range(args.episodes):
        result = inference.run_inference()
        
        total_base_reward += result['base_reward']
        total_shaped_reward += result['shaped_reward']
        if result['success']:
            successes += 1
        
        print(f"\nEpisode {episode + 1}/{args.episodes}:")
        print(f"  Final Program: {result['final_program']}")
        print(f"  Base Reward:   {result['base_reward']:.3f}")
        print(f"  Shaped Reward: {result['shaped_reward']:.3f}")
        print(f"  Success:       {result['success']}")
        print(f"  Steps:         {result['step_count']}")
        print(f"  Executable:    {result['is_executable']}")
        
        if args.verbose:
            print(f"  Individual Programs:")
            for i, prog in enumerate(result['individual_programs']):
                print(f"    ρ_{i+1}: {prog}")
        
        if result['error']:
            print(f"  Error:         {result['error']}")
    
    # Summary
    if args.episodes > 1:
        print("\n" + "=" * 60)
        print(f"SUMMARY ({args.episodes} episodes):")
        print(f"  Average Base Reward:   {total_base_reward / args.episodes:.3f}")
        print(f"  Average Shaped Reward: {total_shaped_reward / args.episodes:.3f}")
        print(f"  Success Rate:          {successes / args.episodes:.3f} ({successes}/{args.episodes})")
        print("=" * 60)


if __name__ == "__main__":
    main()