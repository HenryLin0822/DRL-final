"""
Karel Program Executor

This module provides execution capabilities for Karel programs, integrating with
the VAE model for the latent behavior reconstruction loss in HPRL training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl.karel_dsl import KarelDSL
from dsl.parser import KarelDSLParser, ParseError, ExecutionError
from dsl.tokens import karel_tokens
from environments.karel_env import KarelEnvironment


class ProgramExecutor(nn.Module):
    """
    Karel Program Executor for HPRL
    
    This class handles:
    1. Executing programs from token sequences
    2. Generating execution traces for the latent behavior reconstruction loss
    3. Integration with Karel environments
    4. Batch execution for training efficiency
    """
    
    def __init__(
        self,
        vocab_size: int = 35,
        max_program_length: int = 50,
        max_execution_steps: int = 100,
        timeout_penalty: float = -0.1,
        device: str = 'cpu'
    ):
        """
        Initialize Program Executor
        
        Args:
            vocab_size: Size of the DSL vocabulary
            max_program_length: Maximum program length in tokens
            max_execution_steps: Maximum steps per program execution
            timeout_penalty: Penalty for programs that timeout
            device: Device to run on
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_program_length = max_program_length
        self.max_execution_steps = max_execution_steps
        self.timeout_penalty = timeout_penalty
        self.device = device
        
        # Initialize DSL and parser
        self.dsl = KarelDSL()
        self.parser = KarelDSLParser()
        self.tokens = karel_tokens
        
        # Action mapping from DSL tokens to environment actions
        self.action_mapping = {
            'move': 0,
            'turnLeft': 1, 
            'turnRight': 2,
            'pickMarker': 3,
            'putMarker': 4
        }
        
        # Token to action mapping
        self.token_to_action = {
            4: 0,   # 'move'
            5: 1,   # 'turnLeft'
            6: 2,   # 'turnRight'
            7: 3,   # 'pickMarker'
            8: 4,   # 'putMarker'
        }
        
        # Padding token index
        self.padding_idx = self.tokens.get_padding_index()
    
    def execute_program_batch(
        self,
        program_tokens: torch.Tensor,
        program_lengths: torch.Tensor,
        karel_environments: List[KarelEnvironment],
        return_traces: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Execute a batch of programs in parallel
        
        Args:
            program_tokens: [batch_size, max_seq_len] - tokenized programs
            program_lengths: [batch_size] - actual program lengths
            karel_environments: List of Karel environments for execution
            return_traces: Whether to return full execution traces
            
        Returns:
            Dictionary containing execution results
        """
        batch_size = program_tokens.size(0)
        device = program_tokens.device
        
        # Initialize outputs
        execution_traces = []
        action_sequences = []
        action_lengths = []
        rewards = []
        success_flags = []
        
        for i in range(batch_size):
            # Get individual program
            program = program_tokens[i]
            prog_len = program_lengths[i].item()
            
            # Execute program
            result = self.execute_single_program(
                program[:prog_len],
                karel_environments[i % len(karel_environments)],
                return_traces=return_traces
            )
            
            execution_traces.append(result['states'])
            action_sequences.append(result['actions'])
            action_lengths.append(result['action_length'])
            rewards.append(result['total_reward'])
            success_flags.append(result['success'])
        
        # Convert to tensors
        max_action_len = max(action_lengths) if action_lengths else 1
        
        # Pad action sequences
        padded_actions = []
        for actions in action_sequences:
            if len(actions) < max_action_len:
                padding = [self.vocab_size - 1] * (max_action_len - len(actions))
                actions = actions + padding
            else:
                actions = actions[:max_action_len]
            padded_actions.append(actions)
        
        return {
            'action_sequences': torch.tensor(padded_actions, dtype=torch.long, device=device),
            'action_lengths': torch.tensor(action_lengths, dtype=torch.long, device=device),
            'rewards': torch.tensor(rewards, dtype=torch.float32, device=device),
            'success_flags': torch.tensor(success_flags, dtype=torch.bool, device=device),
            'execution_traces': execution_traces,
            'max_action_length': max_action_len
        }
    
    def execute_single_program(
        self,
        program_tokens: torch.Tensor,
        karel_env: KarelEnvironment,
        return_traces: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a single program in a Karel environment
        
        Args:
            program_tokens: [seq_len] - tokenized program
            karel_env: Karel environment instance
            return_traces: Whether to return execution traces
            
        Returns:
            Dictionary with execution results
        """
        # Convert tokens to CPU numpy for DSL processing
        if isinstance(program_tokens, torch.Tensor):
            token_list = program_tokens.cpu().numpy().tolist()
        else:
            token_list = program_tokens
        
        # Filter padding tokens
        valid_tokens = [t for t in token_list if t != self.padding_idx and t < self.vocab_size]
        
        if not valid_tokens:
            return {
                'states': [],
                'actions': [],
                'action_length': 0,
                'total_reward': self.timeout_penalty,
                'success': False,
                'error': 'Empty program'
            }
        
        try:
            # Reset environment
            initial_state, _ = karel_env.reset()
            
            # Execute program using simple action execution
            actions, total_reward, success, states = self._execute_simple_actions(
                valid_tokens, karel_env, return_traces
            )
            
            return {
                'states': states,
                'actions': actions,
                'action_length': len(actions),
                'total_reward': total_reward,
                'success': success,
                'error': None
            }
            
        except Exception as e:
            return {
                'states': [],
                'actions': [],
                'action_length': 0,
                'total_reward': self.timeout_penalty,
                'success': False,
                'error': str(e)
            }
    
    def _execute_simple_actions(
        self,
        token_list: List[int],
        karel_env: KarelEnvironment,
        return_traces: bool = True
    ) -> Tuple[List[int], float, bool, List[np.ndarray]]:
        """
        Execute simple action sequences (for basic VAE training)
        
        Args:
            token_list: List of program tokens
            karel_env: Karel environment
            return_traces: Whether to collect state traces
            
        Returns:
            Tuple of (actions, total_reward, success, states)
        """
        actions = []
        total_reward = 0.0
        step_count = 0
        states = []
        
        if return_traces:
            states.append(karel_env.karel_world.get_state().copy())
        
        for token in token_list:
            if step_count >= self.max_execution_steps:
                break
                
            if token in self.token_to_action:
                action = self.token_to_action[token]
                
                try:
                    # Execute action in environment
                    obs, reward, terminated, truncated, info = karel_env.step([token])
                    
                    actions.append(action)
                    total_reward += reward
                    step_count += 1
                    
                    if return_traces:
                        states.append(karel_env.karel_world.get_state().copy())
                    
                    if terminated or truncated:
                        break
                        
                except Exception as e:
                    # If action fails, continue to next token
                    continue
        
        success = karel_env.karel_world.done and not karel_env.karel_world.timeout
        
        return actions, total_reward, success, states
    
    def execute_with_dsl(
        self,
        program_tokens: torch.Tensor,
        karel_env: KarelEnvironment,
        return_traces: bool = True
    ) -> Dict[str, Any]:
        """
        Execute program using full DSL parsing (for advanced programs)
        
        Args:
            program_tokens: [seq_len] - tokenized program
            karel_env: Karel environment
            return_traces: Whether to return execution traces
            
        Returns:
            Dictionary with execution results
        """
        # Convert tokens to indices
        if isinstance(program_tokens, torch.Tensor):
            indices = program_tokens.cpu().numpy().tolist()
        else:
            indices = program_tokens
        
        # Filter padding
        valid_indices = [i for i in indices if i != self.padding_idx and i < self.vocab_size]
        
        if not valid_indices:
            return {
                'states': [],
                'actions': [],
                'action_length': 0,
                'total_reward': self.timeout_penalty,
                'success': False,
                'error': 'Empty program'
            }
        
        try:
            # Convert indices to program string
            program_string = self.dsl.intseq2str(valid_indices)
            
            # Validate program
            valid, error_msg = self.dsl.validate_program(program_string)
            if not valid:
                return {
                    'states': [],
                    'actions': [],
                    'action_length': 0,
                    'total_reward': self.timeout_penalty,
                    'success': False,
                    'error': f'Invalid program: {error_msg}'
                }
            
            # Reset environment
            karel_env.reset()
            
            # Execute using DSL
            execution_trace = self.dsl.run(karel_env.karel_world, program_string)
            
            # Extract actions from execution trace
            actions = self._extract_actions_from_trace(execution_trace)
            
            # Calculate reward
            total_reward = karel_env.karel_world.get_reward()
            success = karel_env.karel_world.done and not karel_env.karel_world.timeout
            
            return {
                'states': execution_trace if return_traces else [],
                'actions': actions,
                'action_length': len(actions),
                'total_reward': total_reward,
                'success': success,
                'error': None
            }
            
        except (ParseError, ExecutionError, RuntimeError) as e:
            return {
                'states': [],
                'actions': [],
                'action_length': 0,
                'total_reward': self.timeout_penalty,
                'success': False,
                'error': str(e)
            }
    
    def _extract_actions_from_trace(self, execution_trace: List[np.ndarray]) -> List[int]:
        """
        Extract action sequence from execution trace
        
        Args:
            execution_trace: List of state arrays
            
        Returns:
            List of action indices
        """
        # This is a simplified version - in practice, you'd need to 
        # track the actual actions executed during the trace
        # For now, return empty list
        return []
    
    def compute_latent_behavior_loss(
        self,
        program_embeddings: torch.Tensor,
        target_actions: torch.Tensor,
        target_action_lengths: torch.Tensor,
        states: torch.Tensor,
        neural_executor: nn.Module
    ) -> torch.Tensor:
        """
        Compute latent behavior reconstruction loss (L^L)
        
        This is the key loss from the paper that ensures behavioral smoothness
        in the learned program embedding space.
        
        Args:
            program_embeddings: [batch_size, latent_dim] - program latent codes
            target_actions: [batch_size, max_seq_len] - target action sequences
            target_action_lengths: [batch_size] - actual sequence lengths
            states: [batch_size, state_dim] - environment states
            neural_executor: Neural network that executes latent codes
            
        Returns:
            Latent behavior reconstruction loss
        """
        batch_size, max_seq_len = target_actions.shape
        device = program_embeddings.device
        
        # Get predicted action probabilities from neural executor
        predicted_action_logits = neural_executor(states, program_embeddings)
        
        # Create mask for valid positions
        mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < target_action_lengths.unsqueeze(1)
        
        # Compute cross-entropy loss only on valid positions
        loss = F.cross_entropy(
            predicted_action_logits.view(-1, predicted_action_logits.size(-1)),
            target_actions.view(-1),
            reduction='none'
        )
        
        # Apply mask and average
        masked_loss = loss.view(batch_size, max_seq_len) * mask.float()
        total_loss = masked_loss.sum() / mask.sum().float()
        
        return total_loss
    
    def generate_execution_dataset(
        self,
        programs: List[str],
        num_demos_per_program: int = 5,
        karel_task: str = 'harvester'
    ) -> Dict[str, List]:
        """
        Generate execution dataset for training the condition policy
        
        Args:
            programs: List of program strings
            num_demos_per_program: Number of execution demos per program
            karel_task: Karel task type
            
        Returns:
            Dataset dictionary with programs, states, and actions
        """
        dataset = {
            'programs': [],
            'program_tokens': [],
            'states': [],
            'actions': [],
            'rewards': []
        }
        
        for program_str in programs:
            # Convert to tokens
            program_tokens = self.dsl.str2intseq(program_str)
            
            for demo_idx in range(num_demos_per_program):
                # Create fresh environment
                karel_env = KarelEnvironment(task=karel_task)
                
                # Execute program
                result = self.execute_single_program(
                    torch.tensor(program_tokens),
                    karel_env,
                    return_traces=True
                )
                
                if result['success'] and result['states']:
                    dataset['programs'].append(program_str)
                    dataset['program_tokens'].append(program_tokens)
                    dataset['states'].append(result['states'])
                    dataset['actions'].append(result['actions'])
                    dataset['rewards'].append(result['total_reward'])
        
        return dataset
    
    def forward(
        self,
        program_tokens: torch.Tensor,
        program_lengths: torch.Tensor,
        karel_environments: Optional[List[KarelEnvironment]] = None,
        return_traces: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for batch execution
        
        Args:
            program_tokens: [batch_size, max_seq_len] - tokenized programs
            program_lengths: [batch_size] - actual program lengths
            karel_environments: List of Karel environments (optional)
            return_traces: Whether to return execution traces
            
        Returns:
            Dictionary with execution results
        """
        batch_size = program_tokens.size(0)
        
        # Create environments if not provided
        if karel_environments is None:
            karel_environments = [KarelEnvironment() for _ in range(batch_size)]
        
        # Execute batch
        return self.execute_program_batch(
            program_tokens,
            program_lengths,
            karel_environments,
            return_traces
        )


class NeuralProgramExecutor(nn.Module):
    """
    Neural Program Executor for computing latent behavior reconstruction loss
    
    This network takes program embeddings and states as input and predicts
    the action sequence that would be generated by executing the program.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        state_dim: int = 64,  # Flattened state dimension
        hidden_dim: int = 256,
        num_actions: int = 5,
        max_sequence_length: int = 100,
        dropout: float = 0.0
    ):
        """
        Initialize Neural Program Executor
        
        Args:
            latent_dim: Dimension of program embeddings
            state_dim: Dimension of flattened state
            hidden_dim: Hidden dimension for RNN
            num_actions: Number of possible actions
            max_sequence_length: Maximum action sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.max_sequence_length = max_sequence_length
        
        # Input projection: latent + state -> hidden
        self.input_projection = nn.Linear(latent_dim + state_dim, hidden_dim)
        
        # Recurrent network for sequential prediction
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection: hidden -> action_probabilities
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        states: torch.Tensor,
        program_embeddings: torch.Tensor,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass to predict action sequences
        
        Args:
            states: [batch_size, state_dim] - initial states
            program_embeddings: [batch_size, latent_dim] - program embeddings
            max_length: Maximum sequence length to generate
            
        Returns:
            action_logits: [batch_size, max_length, num_actions] - predicted actions
        """
        batch_size = states.size(0)
        device = states.device
        max_length = max_length or self.max_sequence_length
        
        # Combine state and program embedding
        combined_input = torch.cat([states, program_embeddings], dim=-1)
        
        # Project to hidden dimension
        hidden_input = self.input_projection(combined_input)
        hidden_input = self.dropout(hidden_input)
        
        # Prepare for RNN: repeat input for sequence length
        rnn_input = hidden_input.unsqueeze(1).repeat(1, max_length, 1)
        
        # Run through RNN
        rnn_output, _ = self.rnn(rnn_input)
        
        # Project to action space
        action_logits = self.output_projection(rnn_output)
        
        return action_logits


# Example usage and testing
if __name__ == "__main__":
    # Test the program executor
    print("Testing Program Executor...")
    
    # Create executor
    executor = ProgramExecutor(device='cpu')
    
    # Test program (simple move sequence)
    test_program = torch.tensor([0, 1, 2, 4, 4, 4, 3])  # DEF run m( move move move m)
    program_length = torch.tensor([7])
    
    # Create test environment
    from environments.karel_env import KarelEnvironment
    karel_env = KarelEnvironment(task='harvester')
    
    # Execute single program
    result = executor.execute_single_program(test_program, karel_env)
    print(f"Execution result: {result}")
    
    # Test batch execution
    batch_programs = test_program.unsqueeze(0).repeat(3, 1)
    batch_lengths = program_length.repeat(3)
    
    batch_result = executor.execute_program_batch(
        batch_programs,
        batch_lengths,
        [karel_env, karel_env, karel_env]
    )
    print(f"Batch execution result keys: {batch_result.keys()}")
    print(f"Batch action sequences shape: {batch_result['action_sequences'].shape}")
    
    # Test neural executor
    neural_executor = NeuralProgramExecutor(
        latent_dim=64,
        state_dim=512,  # 8*8*8 flattened state
        num_actions=5
    )
    
    # Test forward pass
    dummy_states = torch.randn(3, 512)
    dummy_embeddings = torch.randn(3, 64)
    
    action_logits = neural_executor(dummy_states, dummy_embeddings, max_length=10)
    print(f"Neural executor output shape: {action_logits.shape}")
    
    print("Program Executor tests completed!")