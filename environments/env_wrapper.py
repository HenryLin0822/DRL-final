"""
Environment Wrapper for HPRL

This module provides environment wrappers that handle both VAE training
and PPO meta-policy training modes, bridging the clean Karel implementation
with the HPRL framework requirements.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List, Union
import copy

from .karel_env import KarelEnvironment, HPRLKarelEnvironment
from .karel_world import KarelWorld
from .state_generator import KarelStateGenerator


class HPRLEnvironmentWrapper(gym.Env):
    """
    Main HPRL Environment Wrapper
    
    This wrapper handles both training phases:
    1. VAE pretraining: Executes complete programs and matches execution traces
    2. Meta-policy training: Takes program embeddings and executes them hierarchically
    """
    
    def __init__(
        self,
        task: str = 'harvester',
        grid_size: Tuple[int, int] = (8, 8),
        max_episode_steps: int = 5,
        max_program_length: int = 40,
        vocab_size: int = 35,
        mode: str = 'hprl',  # 'vae' or 'hprl'
        latent_dim: int = 64,
        timeout_steps: int = 100,
        use_fixed_states: bool = True,
        vae_decoder = None,
        dsl_parser = None,
        reward_type: str = 'dense',
        **kwargs
    ):
        """
        Initialize HPRL Environment Wrapper
        
        Args:
            task: Karel task name
            grid_size: (height, width) of Karel grid
            max_episode_steps: Max macro steps (program executions)
            max_program_length: Max tokens per program
            vocab_size: Size of program vocabulary
            mode: 'vae' for VAE training, 'hprl' for meta-policy training
            latent_dim: Dimension of program embeddings
            timeout_steps: Max primitive steps per program
            use_fixed_states: Whether to use fixed observation states
            vae_decoder: VAE decoder for converting embeddings to programs
            dsl_parser: DSL parser for executing programs
            reward_type: 'dense' or 'sparse'
        """
        super().__init__()
        
        self.task = task
        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self.max_program_length = max_program_length
        self.vocab_size = vocab_size
        self.mode = mode
        self.latent_dim = latent_dim
        self.timeout_steps = timeout_steps
        self.use_fixed_states = use_fixed_states
        self.vae_decoder = vae_decoder
        self.dsl_parser = dsl_parser
        self.reward_type = reward_type
        
        # Initialize Karel environment
        self.karel_env = KarelEnvironment(
            task=task,
            grid_size=grid_size,
            max_episode_steps=max_episode_steps,
            max_program_length=max_program_length,
            vocab_size=vocab_size,
            timeout_steps=timeout_steps,
            reward_type=reward_type,
            use_fixed_states=use_fixed_states
        )
        
        # Initialize state generator for fixed states
        self.state_generator = KarelStateGenerator(
            grid_size=grid_size,
            task=task
        )
        
        # Define observation space (always Karel state)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(*grid_size, 8),
            dtype=np.uint8
        )
        
        # Define action space based on mode
        if mode == 'vae':
            # VAE training: program tokens
            self.action_space = spaces.Box(
                low=0,
                high=vocab_size,
                shape=(max_program_length,),
                dtype=np.int32
            )
        elif mode == 'hprl':
            # Meta-policy training: program embeddings
            self.action_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(latent_dim,),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.programs_executed = []
        
        # Fixed states for consistent input (as in original)
        self.fixed_states = []
        if use_fixed_states:
            self._generate_fixed_states()
        
        # VAE-specific tracking (for program execution matching)
        self.target_execution = None
        self.current_execution = []
        self.execution_progress = 0.0
    
    def _generate_fixed_states(self):
        """Generate fixed states for consistent observation"""
        self.fixed_states = []
        for i in range(self.max_episode_steps + 1):
            state = self.state_generator.generate_program_instruction_state(
                self.grid_size[0], self.grid_size[1], wall_prob=0.1, idx=i
            )
            self.fixed_states.append(state)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.programs_executed = []
        
        # Reset Karel environment
        if self.mode == 'vae':
            # For VAE training, start with task-specific state
            obs, info = self.karel_env.reset(seed=seed, options=options)
            self.target_execution = None  # Will be set by external trainer
        else:
            # For HPRL training, use fixed states for consistency
            obs, info = self.karel_env.reset(seed=seed, options=options)
        
        # Get observation (fixed state if enabled)
        if self.use_fixed_states and self.fixed_states:
            observation = self.fixed_states[0]
        else:
            observation = obs
        
        # VAE-specific reset
        self.current_execution = []
        self.execution_progress = 0.0
        
        info.update({
            'mode': self.mode,
            'task': self.task,
            'step': self.current_step
        })
        
        return observation, info
    
    def step(
        self,
        action: Union[np.ndarray, List[int]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step"""
        if self.mode == 'vae':
            return self._step_vae(action)
        elif self.mode == 'hprl':
            return self._step_hprl(action)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _step_vae(
        self,
        action: Union[np.ndarray, List[int]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """VAE training step: execute program and match target execution"""
        if isinstance(action, list):
            action = np.array(action)
        
        self.current_step += 1
        
        # Parse and execute program
        try:
            execution_trace, program_info = self._execute_program_tokens(action)
            self.current_execution.extend(execution_trace)
        except Exception as e:
            # Invalid program
            execution_trace = []
            program_info = {
                'valid': False,
                'error': str(e),
                'program_string': '',
                'execution_steps': 0
            }
        
        # Calculate reward based on execution matching
        if self.target_execution is not None:
            reward = self._calculate_vae_reward(execution_trace)
        else:
            reward = 0.0  # No target set
        
        self.episode_reward += reward
        
        # Get next observation
        if self.use_fixed_states and self.current_step < len(self.fixed_states):
            observation = self.fixed_states[self.current_step % len(self.fixed_states)]
        else:
            observation = self.karel_env.get_state()
        
        # Check termination
        terminated = (self.target_execution is not None and 
                     len(self.current_execution) >= len(self.target_execution))
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            'mode': 'vae',
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'program_valid': program_info.get('valid', False),
            'program_string': program_info.get('program_string', ''),
            'execution_length': len(execution_trace),
            'target_length': len(self.target_execution) if self.target_execution else 0,
            'execution_match': self.execution_progress
        }
        
        return observation, reward, terminated, truncated, info
    
    def _step_hprl(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """HPRL meta-policy step: decode embedding and execute program"""
        self.current_step += 1
        
        # Decode program embedding to tokens
        try:
            program_tokens = self._decode_embedding(action)
            program_info = {'valid': True, 'embedding': action.copy()}
        except Exception as e:
            # Invalid embedding
            program_tokens = np.array([0, 1, 2, 3])  # Empty program: DEF run m( m)
            program_info = {'valid': False, 'error': str(e), 'embedding': action.copy()}
        
        # Execute program in Karel environment
        obs, reward, terminated, truncated, env_info = self.karel_env.step(program_tokens)
        
        self.episode_reward += reward
        self.programs_executed.append({
            'step': self.current_step,
            'embedding': action.copy(),
            'tokens': program_tokens,
            'reward': reward,
            'info': program_info
        })
        
        # Get next observation (fixed state)
        if self.use_fixed_states and self.current_step < len(self.fixed_states):
            observation = self.fixed_states[self.current_step % len(self.fixed_states)]
        else:
            observation = obs
        
        info = {
            'mode': 'hprl',
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'program_reward': reward,
            'program_valid': program_info.get('valid', False),
            'programs_executed': len(self.programs_executed),
            'karel_info': env_info
        }
        
        return observation, reward, terminated, truncated, info
    
    def _execute_program_tokens(self, tokens: np.ndarray) -> Tuple[List[np.ndarray], Dict]:
        """Execute program tokens and return execution trace"""
        if self.dsl_parser is None:
            # Simple fallback execution
            return self._execute_simple_tokens(tokens)
        else:
            # Use proper DSL parser
            return self._execute_dsl_program(tokens)
    
    def _execute_simple_tokens(self, tokens: np.ndarray) -> Tuple[List[np.ndarray], Dict]:
        """Simple token execution (fallback when no DSL parser)"""
        # Filter valid tokens
        valid_tokens = tokens[tokens < self.vocab_size - 1]
        
        if len(valid_tokens) == 0:
            return [], {'valid': False, 'program_string': '', 'execution_steps': 0}
        
        # Simple action mapping
        action_map = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4}  # move, turnLeft, turnRight, pick, put
        
        execution_trace = []
        execution_trace.append(self.karel_env.karel_world.get_state())  # Initial state
        
        steps = 0
        for token in valid_tokens:
            if token in action_map and steps < self.timeout_steps:
                action = action_map[token]
                _, _, done, info = self.karel_env.karel_world.step(action)
                execution_trace.append(self.karel_env.karel_world.get_state())
                steps += 1
                
                if done:
                    break
        
        program_string = ' '.join(str(t) for t in valid_tokens)
        return execution_trace, {
            'valid': True,
            'program_string': program_string,
            'execution_steps': steps
        }
    
    def _execute_dsl_program(self, tokens: np.ndarray) -> Tuple[List[np.ndarray], Dict]:
        """Execute program using DSL parser"""
        try:
            program_string = self.dsl_parser.tokens_to_string(tokens)
            execution_trace = self.dsl_parser.execute(program_string, self.karel_env.karel_world)
            return execution_trace, {
                'valid': True,
                'program_string': program_string,
                'execution_steps': len(execution_trace) - 1
            }
        except Exception as e:
            return [], {
                'valid': False,
                'program_string': '',
                'execution_steps': 0,
                'error': str(e)
            }
    
    def _decode_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Decode program embedding to tokens using VAE decoder"""
        if self.vae_decoder is None:
            raise ValueError("VAE decoder not provided for HPRL mode")
        
        import torch
        
        # Convert to torch tensor
        if isinstance(embedding, np.ndarray):
            embedding_tensor = torch.from_numpy(embedding).float()
        else:
            embedding_tensor = embedding
        
        # Add batch dimension if needed
        if embedding_tensor.dim() == 1:
            embedding_tensor = embedding_tensor.unsqueeze(0)
        
        # Decode using VAE
        with torch.no_grad():
            self.vae_decoder.eval()
            
            # Apply compression decoder if exists
            if hasattr(self.vae_decoder, 'compression_decoder'):
                embedding_tensor = self.vae_decoder.compression_decoder(embedding_tensor)
            
            # Decode to program tokens
            decoder_output = self.vae_decoder.decode(
                embedding_tensor,
                max_length=self.max_program_length,
                deterministic=True
            )
            
            # Extract tokens
            if isinstance(decoder_output, tuple):
                program_tokens = decoder_output[0][0].cpu().numpy()
            else:
                program_tokens = decoder_output[0].cpu().numpy()
        
        return program_tokens
    
    def _calculate_vae_reward(self, execution_trace: List[np.ndarray]) -> float:
        """Calculate reward for VAE training based on execution matching"""
        if self.target_execution is None or len(execution_trace) == 0:
            return -0.01
        
        # Compare execution traces
        matches = 0
        for i, state in enumerate(execution_trace):
            if i < len(self.target_execution):
                if np.array_equal(state, self.target_execution[i]):
                    matches += 1
                else:
                    break  # Stop at first mismatch
        
        # Calculate progress
        max_length = max(len(execution_trace), len(self.target_execution))
        if max_length == 0:
            return 0.0
        
        current_progress = matches / max_length
        reward = current_progress - self.execution_progress
        self.execution_progress = current_progress
        
        # Bonus for complete match
        if matches == len(self.target_execution) and len(execution_trace) == len(self.target_execution):
            reward += 0.1
        
        return reward
    
    def set_target_execution(self, target_execution: List[np.ndarray]):
        """Set target execution for VAE training"""
        self.target_execution = target_execution
        self.execution_progress = 0.0
    
    def set_vae_decoder(self, vae_decoder):
        """Set VAE decoder for HPRL mode"""
        self.vae_decoder = vae_decoder
    
    def set_dsl_parser(self, dsl_parser):
        """Set DSL parser for program execution"""
        self.dsl_parser = dsl_parser
    
    def get_karel_state(self) -> np.ndarray:
        """Get current Karel world state"""
        return self.karel_env.karel_world.get_state()
    
    def render(self, mode: str = 'human'):
        """Render the environment"""
        return self.karel_env.render(mode)
    
    def close(self):
        """Clean up resources"""
        self.karel_env.close()


class VAETrainingWrapper(HPRLEnvironmentWrapper):
    """Specialized wrapper for VAE training"""
    
    def __init__(self, **kwargs):
        kwargs['mode'] = 'vae'
        super().__init__(**kwargs)


class MetaPolicyTrainingWrapper(HPRLEnvironmentWrapper):
    """Specialized wrapper for meta-policy training"""
    
    def __init__(self, **kwargs):
        kwargs['mode'] = 'hprl'
        super().__init__(**kwargs)


# Factory functions
def make_vae_training_env(task: str = 'harvester', **kwargs) -> VAETrainingWrapper:
    """Create environment for VAE training"""
    return VAETrainingWrapper(task=task, **kwargs)


def make_hprl_training_env(
    task: str = 'harvester',
    vae_decoder = None,
    **kwargs
) -> MetaPolicyTrainingWrapper:
    """Create environment for HPRL meta-policy training"""
    return MetaPolicyTrainingWrapper(
        task=task,
        vae_decoder=vae_decoder,
        **kwargs
    )


# Gymnasium compatibility wrapper
class GymnasiumCompatWrapper(gym.Wrapper):
    """
    Wrapper to handle old gym vs new gymnasium API differences
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._old_gym_api = False
    
    def step(self, action):
        """Handle both old and new step API"""
        result = self.env.step(action)
        
        if self._old_gym_api:
            # Convert new API (obs, reward, terminated, truncated, info) to old API
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            if truncated:
                info['TimeLimit.truncated'] = True
            return obs, reward, done, info
        else:
            # Return new API as-is
            return result
    
    def reset(self, **kwargs):
        """Handle both old and new reset API"""
        result = self.env.reset(**kwargs)
        
        if self._old_gym_api:
            # Convert new API (obs, info) to old API (obs)
            if isinstance(result, tuple):
                return result[0]
            else:
                return result
        else:
            # Return new API as-is
            return result
    
    def enable_old_gym_api(self):
        """Enable old gym API compatibility"""
        self._old_gym_api = True