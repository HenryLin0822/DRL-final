"""
Modern Karel Environment for HPRL

This module provides a clean Gymnasium-compatible environment for Karel tasks
that can be used with the HPRL framework.
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
from gymnasium import spaces

from .karel_world import KarelWorld
from .state_generator import KarelStateGenerator


class KarelEnvironment(gym.Env):
    """
    Modern Karel Environment compatible with Gymnasium API.
    
    This environment handles the execution of Karel programs and provides
    rewards based on the specified task. It supports both single program
    execution and hierarchical program composition (HPRL).
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        task: str = 'harvester',
        grid_size: Tuple[int, int] = (8, 8),
        max_episode_steps: int = 5,
        max_program_length: int = 40,
        vocab_size: int = 35,
        timeout_steps: int = 100,
        reward_type: str = 'dense',
        use_fixed_states: bool = True,
        **kwargs
    ):
        """
        Initialize Karel Environment
        
        Args:
            task: Karel task type ('harvester', 'cleanHouse', 'fourCorners', etc.)
            grid_size: (height, width) of the Karel grid
            max_episode_steps: Maximum number of macro steps (program executions)
            max_program_length: Maximum length of a program in tokens
            vocab_size: Size of the program vocabulary
            timeout_steps: Maximum primitive steps per program
            reward_type: 'dense' or 'sparse' rewards
            use_fixed_states: Whether to use fixed states for consistent training
        """
        super().__init__()
        
        self.task = task
        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self.max_program_length = max_program_length
        self.vocab_size = vocab_size
        self.timeout_steps = timeout_steps
        self.reward_type = reward_type
        self.use_fixed_states = use_fixed_states
        
        # Initialize Karel world and state generator
        self.karel_world = KarelWorld(
            task=task,
            grid_size=grid_size,
            timeout_steps=timeout_steps
        )
        
        self.state_generator = KarelStateGenerator(
            grid_size=grid_size,
            task=task
        )
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(*grid_size, 8),  # Karel state channels
            dtype=np.uint8
        )
        
        # Action space: program tokens
        self.action_space = spaces.Box(
            low=0,
            high=vocab_size,
            shape=(max_program_length,),
            dtype=np.int32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_programs = []
        
        # Fixed states for consistent training (if enabled)
        self.fixed_states = []
        if use_fixed_states:
            self._generate_fixed_states()
    
    def _generate_fixed_states(self):
        """Generate fixed states for consistent training"""
        self.fixed_states = []
        for i in range(self.max_episode_steps + 1):
            state = self.state_generator.generate_state(task_specific=True)
            self.fixed_states.append(state)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state
        
        Returns:
            observation: Initial Karel state
            info: Additional information
        """
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_programs = []
        
        # Get initial state
        if self.use_fixed_states and self.fixed_states:
            initial_state, metadata = self.fixed_states[0]
        else:
            initial_state, metadata = self.state_generator.generate_state(task_specific=True)
        
        # Reset Karel world
        observation = self.karel_world.reset(initial_state, metadata)
        
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'task': self.task
        }
        
        return observation, info
    
    def step(
        self, 
        action: Union[np.ndarray, List[int]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute a program in the Karel environment
        
        Args:
            action: Program tokens (array of integers)
            
        Returns:
            observation: New Karel state
            reward: Reward for this step
            terminated: Whether episode is done due to task completion
            truncated: Whether episode is done due to step limit
            info: Additional information
        """
        if isinstance(action, list):
            action = np.array(action)
        
        self.current_step += 1
        
        # Parse and execute the program
        program_reward, program_info = self._execute_program(action)
        
        self.episode_reward += program_reward
        self.episode_programs.append({
            'tokens': action.copy(),
            'reward': program_reward,
            'info': program_info
        })
        
        # Get next observation (either next fixed state or current world state)
        if self.use_fixed_states and self.current_step < len(self.fixed_states):
            observation = self.fixed_states[self.current_step % len(self.fixed_states)][0]
        else:
            observation = self.karel_world.get_state()
        
        # Check termination conditions
        terminated = self.karel_world.done or program_info.get('task_completed', False)
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'program_reward': program_reward,
            'program_valid': program_info.get('valid', False),
            'program_string': program_info.get('program_string', ''),
            'execution_steps': program_info.get('execution_steps', 0),
            'task_completed': program_info.get('task_completed', False),
            'timeout': program_info.get('timeout', False)
        }
        
        return observation, program_reward, terminated, truncated, info
    
    def _execute_program(self, program_tokens: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Execute a program represented as tokens
        
        Args:
            program_tokens: Array of program tokens
            
        Returns:
            reward: Reward for executing this program
            info: Execution information
        """
        # Filter out padding tokens (assuming vocab_size-1 is padding)
        valid_tokens = program_tokens[program_tokens < self.vocab_size - 1]
        
        if len(valid_tokens) == 0:
            return -0.01, {  # Small penalty for empty programs
                'valid': False,
                'program_string': '',
                'execution_steps': 0,
                'task_completed': False,
                'timeout': False
            }
        
        try:
            # For now, we'll use a simple program execution strategy
            # In a full implementation, this would use the DSL parser
            reward, steps, completed = self._execute_simple_program(valid_tokens)
            
            return reward, {
                'valid': True,
                'program_string': self._tokens_to_string(valid_tokens),
                'execution_steps': steps,
                'task_completed': completed,
                'timeout': steps >= self.timeout_steps
            }
            
        except Exception as e:
            return -0.01, {  # Penalty for invalid programs
                'valid': False,
                'program_string': self._tokens_to_string(valid_tokens),
                'execution_steps': 0,
                'task_completed': False,
                'timeout': False,
                'error': str(e)
            }
    
    def _execute_simple_program(self, tokens: np.ndarray) -> Tuple[float, int, bool]:
        """
        Execute a simple program (placeholder implementation)
        
        In a full implementation, this would parse the DSL and execute properly.
        For now, we'll execute basic action sequences.
        
        Args:
            tokens: Valid program tokens
            
        Returns:
            reward: Total reward from execution
            steps: Number of primitive steps executed
            completed: Whether task was completed
        """
        # Token mapping (simplified)
        # This would be replaced by proper DSL parsing
        action_tokens = {
            4: 0,   # 'move' -> MOVE
            5: 1,   # 'turnLeft' -> TURN_LEFT  
            6: 2,   # 'turnRight' -> TURN_RIGHT
            7: 3,   # 'pickMarker' -> PICK_MARKER
            8: 4,   # 'putMarker' -> PUT_MARKER
        }
        
        total_reward = 0.0
        steps = 0
        
        # Simple execution: just execute basic actions in sequence
        for token in tokens:
            if token in action_tokens and steps < self.timeout_steps:
                action = action_tokens[token]
                _, reward, done, info = self.karel_world.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
        
        # Check if task is completed
        completed = self.karel_world.done and not info.get('timeout', False)
        
        return total_reward, steps, completed
    
    def _tokens_to_string(self, tokens: np.ndarray) -> str:
        """
        Convert program tokens to string representation
        
        Args:
            tokens: Program tokens
            
        Returns:
            String representation of the program
        """
        # Token to string mapping (simplified)
        token_map = {
            0: 'DEF', 1: 'run', 2: 'm(', 3: 'm)',
            4: 'move', 5: 'turnLeft', 6: 'turnRight', 
            7: 'pickMarker', 8: 'putMarker',
            9: 'REPEAT', 10: 'r(', 11: 'r)', 
            12: 'R=2', 13: 'R=3', 14: 'R=4', 15: 'R=5',
            16: 'IF', 17: 'IFELSE', 18: 'ELSE',
            19: 'i(', 20: 'i)', 21: 'e(', 22: 'e)',
            23: 'frontIsClear', 24: 'leftIsClear', 25: 'rightIsClear',
            26: 'markersPresent', 27: 'noMarkersPresent',
            28: 'not', 29: 'c(', 30: 'c)',
            31: 'WHILE', 32: 'w(', 33: 'w)',
        }
        
        return ' '.join(token_map.get(token, f'UNK_{token}') for token in tokens)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment"""
        return self.karel_world.render(mode)
    
    def close(self):
        """Clean up resources"""
        pass
    
    def get_episode_info(self) -> Dict[str, Any]:
        """Get comprehensive episode information"""
        return {
            'episode_reward': self.episode_reward,
            'episode_length': self.current_step,
            'programs_executed': len(self.episode_programs),
            'task': self.task,
            'programs': self.episode_programs,
            'task_completed': self.karel_world.done
        }


class HPRLKarelEnvironment(KarelEnvironment):
    """
    HPRL-specific Karel Environment
    
    This extends the base Karel environment with HPRL-specific features:
    - Program embedding action space
    - Integration with VAE decoder
    - Hierarchical program composition
    """
    
    def __init__(
        self,
        vae_decoder=None,
        latent_dim: int = 64,
        deterministic_decoder: bool = True,
        **kwargs
    ):
        """
        Initialize HPRL Karel Environment
        
        Args:
            vae_decoder: Pre-trained VAE decoder for converting embeddings to programs
            latent_dim: Dimension of program embeddings
            deterministic_decoder: Whether to use deterministic decoding
            **kwargs: Additional arguments for base KarelEnvironment
        """
        super().__init__(**kwargs)
        
        self.vae_decoder = vae_decoder
        self.latent_dim = latent_dim
        self.deterministic_decoder = deterministic_decoder
        
        # Override action space to be program embeddings
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(latent_dim,),
            dtype=np.float32
        )
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute a program embedding in the Karel environment
        
        Args:
            action: Program embedding (latent vector)
            
        Returns:
            observation: New Karel state
            reward: Reward for this step
            terminated: Whether episode is done due to task completion
            truncated: Whether episode is done due to step limit
            info: Additional information
        """
        if self.vae_decoder is None:
            raise ValueError("VAE decoder not provided for HPRL environment")
        
        # Convert embedding to program tokens using VAE decoder
        program_tokens = self._decode_program(action)
        
        # Execute the decoded program
        return super().step(program_tokens)
    
    def _decode_program(self, embedding: np.ndarray) -> np.ndarray:
        """
        Decode program embedding to tokens using VAE decoder
        
        Args:
            embedding: Program embedding vector
            
        Returns:
            Program tokens
        """
        import torch
        
        # Convert to torch tensor
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        
        # Add batch dimension if needed
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        # Decode using VAE
        with torch.no_grad():
            self.vae_decoder.eval()
            
            # Apply tanh if needed (as in original code)
            if hasattr(self.vae_decoder, 'tanh') and self.vae_decoder.tanh is not None:
                embedding = self.vae_decoder.tanh(embedding)
            
            # Decode to program tokens
            decoder_output = self.vae_decoder.decoder(
                None, 
                embedding, 
                teacher_enforcing=False,
                deterministic=self.deterministic_decoder,
                evaluate=False
            )
            
            # Extract program tokens from decoder output
            _, pred_programs, pred_programs_len, *_ = decoder_output
            
            # Convert to numpy
            program_tokens = pred_programs[0].cpu().numpy()
            program_length = pred_programs_len[0].cpu().numpy()
            
            # Truncate to actual length
            program_tokens = program_tokens[:program_length]
        
        return program_tokens
    
    def set_vae_decoder(self, vae_decoder):
        """Set the VAE decoder for program decoding"""
        self.vae_decoder = vae_decoder


# Factory function for creating environments
def make_karel_env(
    task: str = 'harvester',
    env_type: str = 'standard',
    **kwargs
) -> KarelEnvironment:
    """
    Factory function for creating Karel environments
    
    Args:
        task: Karel task type
        env_type: 'standard' or 'hprl'
        **kwargs: Additional environment arguments
        
    Returns:
        Karel environment instance
    """
    if env_type == 'standard':
        return KarelEnvironment(task=task, **kwargs)
    elif env_type == 'hprl':
        return HPRLKarelEnvironment(task=task, **kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


# Register environments with Gymnasium
def register_karel_envs():
    """Register Karel environments with Gymnasium"""
    from gymnasium.envs.registration import register
    
    # Standard Karel environments
    tasks = ['harvester', 'cleanHouse', 'fourCorners', 'stairClimber', 'topOff', 'randomMaze']
    
    for task in tasks:
        register(
            id=f'Karel-{task}-v0',
            entry_point='karel_env:KarelEnvironment',
            kwargs={'task': task}
        )
        
        register(
            id=f'Karel-{task}-HPRL-v0',
            entry_point='karel_env:HPRLKarelEnvironment',
            kwargs={'task': task}
        )


# Auto-register environments when module is imported
try:
    register_karel_envs()
except:
    pass  # Ignore registration errors