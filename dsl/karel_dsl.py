"""
Karel Domain Specific Language

This module provides the main Karel DSL interface, combining token management,
parsing, and execution into a unified system compatible with the HPRL framework.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import random

from .tokens import karel_tokens, TokenType
from .parser import KarelDSLParser, KarelProgram, ParseError, ExecutionError


class KarelDSL:
    """
    Main Karel DSL Interface
    
    This class provides the primary interface for working with Karel programs,
    including token management, parsing, execution, and program generation.
    Compatible with the original HPRL codebase.
    """
    
    def __init__(self, seed: Optional[int] = None, environment: str = 'karel'):
        """
        Initialize Karel DSL
        
        Args:
            seed: Random seed for program generation
            environment: Environment name (for compatibility)
        """
        self.environment = environment
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        
        # Initialize components
        self.tokens = karel_tokens
        self.parser = KarelDSLParser()
        
        # Compatibility with original codebase
        self.int2token = self.tokens.all_tokens
        self.token2int = self.tokens.token_to_idx
        self.action_functions = self.tokens.action_tokens
        self.conditional_functions = self.tokens.condition_tokens
        
        # Execution settings
        self.max_func_call = 200
        self.call_counter = [0]
    
    def str2intseq(self, code: str) -> List[int]:
        """Convert program string to integer sequence"""
        return self.tokens.string_to_indices(code)
    
    def intseq2str(self, intseq: List[int]) -> str:
        """Convert integer sequence to program string"""
        return self.tokens.indices_to_string(intseq)
    
    def code2intseq(self, code: str) -> List[int]:
        """Convert code string to integer sequence (alias for str2intseq)"""
        return self.str2intseq(code)
    
    def parse(self, code: str, **kwargs) -> KarelProgram:
        """Parse a program string"""
        self.call_counter = [0]
        return self.parser.parse(code)
    
    def run(self, world, code: str, **kwargs) -> List[np.ndarray]:
        """
        Run a program on a Karel world
        
        Args:
            world: Karel world instance
            code: Program string to execute
            
        Returns:
            List of states representing execution history
        """
        self.call_counter = [0]
        
        try:
            # Clear world history if it has the method
            if hasattr(world, 'clear_history'):
                world.clear_history()
            
            # Execute program
            execution_trace = self.parser.execute(code, world)
            
            # Return state history (compatibility with original)
            if hasattr(world, 's_h'):
                return world.s_h
            else:
                return execution_trace
                
        except (ParseError, ExecutionError) as e:
            # For compatibility, raise RuntimeError like original
            raise RuntimeError(str(e))
    
    def execute_tokens(self, tokens: List[str], world) -> List[np.ndarray]:
        """Execute a list of tokens on a Karel world"""
        try:
            if hasattr(world, 'clear_history'):
                world.clear_history()
            
            execution_trace = self.parser.execute_tokens(tokens, world)
            
            if hasattr(world, 's_h'):
                return world.s_h
            else:
                return execution_trace
                
        except (ParseError, ExecutionError) as e:
            raise RuntimeError(str(e))
    
    def execute_indices(self, indices: List[int], world) -> List[np.ndarray]:
        """Execute a list of token indices on a Karel world"""
        try:
            if hasattr(world, 'clear_history'):
                world.clear_history()
            
            execution_trace = self.parser.execute_indices(indices, world)
            
            if hasattr(world, 's_h'):
                return world.s_h
            else:
                return execution_trace
                
        except (ParseError, ExecutionError) as e:
            raise RuntimeError(str(e))
    
    def validate_program(self, code: str) -> Tuple[bool, str]:
        """Validate a program string"""
        return self.parser.validate_program(code)
    
    def generate_random_program(
        self,
        max_length: int = 20,
        include_control_flow: bool = True,
        include_conditions: bool = True
    ) -> str:
        """
        Generate a random valid Karel program
        
        Args:
            max_length: Maximum number of tokens in program body
            include_control_flow: Whether to include IF, WHILE, REPEAT
            include_conditions: Whether to include condition checks
            
        Returns:
            Random program string
        """
        program_tokens = ['DEF', 'run', 'm(']
        
        # Generate random statement sequence
        body_tokens = self._generate_random_statements(
            max_length - 4,  # Account for DEF run m( m)
            include_control_flow,
            include_conditions
        )
        
        program_tokens.extend(body_tokens)
        program_tokens.append('m)')
        
        return ' '.join(program_tokens)
    
    def _generate_random_statements(
        self,
        max_tokens: int,
        include_control_flow: bool,
        include_conditions: bool,
        depth: int = 0
    ) -> List[str]:
        """Generate random statement tokens"""
        if max_tokens <= 0 or depth > 3:  # Limit nesting depth
            return []
        
        statements = []
        remaining_tokens = max_tokens
        
        while remaining_tokens > 0:
            # Choose statement type
            if include_control_flow and remaining_tokens > 5 and self.rng.random() < 0.3:
                # Generate control flow statement
                if self.rng.random() < 0.4:  # REPEAT
                    stmt_tokens, consumed = self._generate_repeat_statement(
                        remaining_tokens, include_control_flow, include_conditions, depth + 1
                    )
                elif self.rng.random() < 0.7:  # IF
                    stmt_tokens, consumed = self._generate_if_statement(
                        remaining_tokens, include_control_flow, include_conditions, depth + 1
                    )
                else:  # WHILE
                    stmt_tokens, consumed = self._generate_while_statement(
                        remaining_tokens, include_control_flow, include_conditions, depth + 1
                    )
            else:
                # Generate action statement
                stmt_tokens = [self.rng.choice(self.tokens.action_tokens)]
                consumed = 1
            
            statements.extend(stmt_tokens)
            remaining_tokens -= consumed
            
            if consumed == 0:  # Avoid infinite loop
                break
        
        return statements
    
    def _generate_repeat_statement(self, max_tokens: int, include_cf: bool, include_cond: bool, depth: int) -> Tuple[List[str], int]:
        """Generate REPEAT statement"""
        if max_tokens < 5:
            return [], 0
        
        count = self.rng.choice(['R=2', 'R=3', 'R=4', 'R=5'])
        body_tokens = self._generate_random_statements(
            min(max_tokens - 4, 6),  # Limit body size
            include_cf and depth < 2,
            include_cond,
            depth
        )
        
        tokens = ['REPEAT', count, 'r('] + body_tokens + ['r)']
        return tokens, len(tokens)
    
    def _generate_if_statement(self, max_tokens: int, include_cf: bool, include_cond: bool, depth: int) -> Tuple[List[str], int]:
        """Generate IF statement"""
        if max_tokens < 7:
            return [], 0
        
        # Choose condition
        condition = self.rng.choice(self.tokens.condition_tokens)
        negate = include_cond and self.rng.random() < 0.3
        
        # Generate body
        body_tokens = self._generate_random_statements(
            min(max_tokens - 6 - (1 if negate else 0), 4),
            include_cf and depth < 2,
            include_cond,
            depth
        )
        
        if negate:
            tokens = ['IF', 'c(', 'not', condition, 'c)', 'i('] + body_tokens + ['i)']
        else:
            tokens = ['IF', 'c(', condition, 'c)', 'i('] + body_tokens + ['i)']
        
        return tokens, len(tokens)
    
    def _generate_while_statement(self, max_tokens: int, include_cf: bool, include_cond: bool, depth: int) -> Tuple[List[str], int]:
        """Generate WHILE statement"""
        if max_tokens < 7:
            return [], 0
        
        # Choose condition
        condition = self.rng.choice(self.tokens.condition_tokens)
        negate = include_cond and self.rng.random() < 0.2  # Less likely to negate in while
        
        # Generate body (smaller to avoid infinite loops)
        body_tokens = self._generate_random_statements(
            min(max_tokens - 6 - (1 if negate else 0), 3),
            False,  # No nested control flow in while
            include_cond,
            depth
        )
        
        if negate:
            tokens = ['WHILE', 'c(', 'not', condition, 'c)', 'w('] + body_tokens + ['w)']
        else:
            tokens = ['WHILE', 'c(', condition, 'c)', 'w('] + body_tokens + ['w)']
        
        return tokens, len(tokens)
    
    def generate_program_dataset(
        self,
        num_programs: int,
        max_length: int = 20,
        seed: Optional[int] = None
    ) -> List[Tuple[str, List[int]]]:
        """
        Generate a dataset of random programs
        
        Args:
            num_programs: Number of programs to generate
            max_length: Maximum program length
            seed: Random seed
            
        Returns:
            List of (program_string, token_indices) tuples
        """
        if seed is not None:
            old_state = self.rng.get_state()
            self.rng.seed(seed)
        
        dataset = []
        
        for _ in range(num_programs):
            # Generate random program
            program_str = self.generate_random_program(max_length)
            
            # Convert to indices
            indices = self.str2intseq(program_str)
            
            # Validate program
            valid, _ = self.validate_program(program_str)
            if valid:
                dataset.append((program_str, indices))
        
        if seed is not None:
            self.rng.set_state(old_state)
        
        return dataset
    
    def get_vocabulary_info(self) -> Dict[str, Any]:
        """Get vocabulary information"""
        return {
            'vocab_size': len(self.int2token),
            'action_tokens': self.action_functions,
            'condition_tokens': self.conditional_functions,
            'num_actions': len(self.action_functions),
            'num_conditions': len(self.conditional_functions),
            'token_to_idx': self.token2int,
            'idx_to_token': {i: token for i, token in enumerate(self.int2token)},
            'padding_idx': self.tokens.get_padding_index()
        }
    
    def print_program(self, code: str):
        """Print a formatted program"""
        print("Karel Program:")
        print("-" * 40)
        
        tokens = self.tokens.string_to_tokens(code)
        indent = 0
        
        for token in tokens:
            if token in ['r)', 'i)', 'e)', 'w)', 'm)']:
                indent -= 2
            
            print("  " * indent + token)
            
            if token in ['r(', 'i(', 'e(', 'w(', 'm(']:
                indent += 2
    
    def get_program_complexity(self, code: str) -> Dict[str, int]:
        """Analyze program complexity"""
        tokens = self.tokens.string_to_tokens(code)
        
        complexity = {
            'total_tokens': len(tokens),
            'actions': 0,
            'conditions': 0,
            'control_structures': 0,
            'nesting_depth': 0,
            'loops': 0,
            'conditionals': 0
        }
        
        current_depth = 0
        max_depth = 0
        
        for token in tokens:
            if token in self.action_functions:
                complexity['actions'] += 1
            elif token in self.conditional_functions:
                complexity['conditions'] += 1
            elif token in ['REPEAT', 'WHILE']:
                complexity['control_structures'] += 1
                complexity['loops'] += 1
            elif token in ['IF', 'IFELSE']:
                complexity['control_structures'] += 1
                complexity['conditionals'] += 1
            elif token in ['r(', 'i(', 'e(', 'w(', 'm(']:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif token in ['r)', 'i)', 'e)', 'w)', 'm)']:
                current_depth -= 1
        
        complexity['nesting_depth'] = max_depth
        return complexity


# Factory function for compatibility with original codebase
def get_DSL_option_v2(dsl_type: str = 'prob', seed: Optional[int] = None, environment: str = 'karel') -> KarelDSL:
    """
    Factory function to create Karel DSL instance
    
    Args:
        dsl_type: Type of DSL (for compatibility, not used)
        seed: Random seed
        environment: Environment name
        
    Returns:
        KarelDSL instance
    """
    return KarelDSL(seed=seed, environment=environment)


# Create global instance for backward compatibility
default_dsl = KarelDSL()


if __name__ == "__main__":
    # Example usage and testing
    dsl = KarelDSL(seed=42)
    
    print("Karel DSL Information:")
    print("=" * 50)
    
    vocab_info = dsl.get_vocabulary_info()
    print(f"Vocabulary size: {vocab_info['vocab_size']}")
    print(f"Number of actions: {vocab_info['num_actions']}")
    print(f"Number of conditions: {vocab_info['num_conditions']}")
    
    print(f"\nAction tokens: {vocab_info['action_tokens']}")
    print(f"Condition tokens: {vocab_info['condition_tokens']}")
    
    # Test random program generation
    print(f"\nGenerating random programs:")
    print("-" * 30)
    
    for i in range(3):
        program = dsl.generate_random_program(max_length=15)
        print(f"\nProgram {i+1}:")
        dsl.print_program(program)
        
        # Test conversion
        indices = dsl.str2intseq(program)
        recovered = dsl.intseq2str(indices)
        print(f"Indices: {indices}")
        print(f"Recovered: {recovered}")
        print(f"Match: {program == recovered}")
        
        # Analyze complexity
        complexity = dsl.get_program_complexity(program)
        print(f"Complexity: {complexity}")
        
        # Validate
        valid, error = dsl.validate_program(program)
        print(f"Valid: {valid}")
        if not valid:
            print(f"Error: {error}")