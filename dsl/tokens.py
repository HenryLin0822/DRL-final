"""
Token definitions for Karel DSL

This module defines all the tokens used in the Karel Domain Specific Language,
including their mappings and categories.
"""

from typing import Dict, List, Tuple
from enum import IntEnum


class TokenType(IntEnum):
    """Token types for organizing the DSL vocabulary"""
    PROGRAM_STRUCTURE = 0  # DEF, run, m(, m)
    CONTROL_FLOW = 1       # IF, WHILE, REPEAT, etc.
    CONDITIONS = 2         # frontIsClear, markersPresent, etc.
    ACTIONS = 3           # move, turnLeft, etc.
    CONSTANTS = 4         # R=2, R=3, etc.
    BRACKETS = 5          # c(, c), w(, w), etc.
    LOGICAL = 6           # not, and, or
    PADDING = 7           # Padding token


class KarelTokens:
    """
    Karel DSL Token Management
    
    This class handles all token definitions, mappings, and conversions
    for the Karel Domain Specific Language.
    """
    
    def __init__(self):
        # Core token definitions
        self._define_tokens()
        self._create_mappings()
    
    def _define_tokens(self):
        """Define all DSL tokens"""
        
        # Program structure tokens
        self.structure_tokens = [
            'DEF',           # 0
            'run',           # 1
            'm(',            # 2
            'm)',            # 3
        ]
        
        # Action tokens (primitive operations)
        self.action_tokens = [
            'move',          # 4
            'turnLeft',      # 5
            'turnRight',     # 6
            'pickMarker',    # 7
            'putMarker',     # 8
        ]
        
        # Control flow tokens
        self.control_tokens = [
            'REPEAT',        # 9
            'r(',            # 10
            'r)',            # 11
            'IF',            # 16
            'IFELSE',        # 17
            'ELSE',          # 18
            'i(',            # 19
            'i)',            # 20
            'e(',            # 21
            'e)',            # 22
            'WHILE',         # 31
            'w(',            # 32
            'w)',            # 33
        ]
        
        # Constant tokens (for REPEAT)
        self.constant_tokens = [
            'R=2',           # 12
            'R=3',           # 13
            'R=4',           # 14
            'R=5',           # 15
        ]
        
        # Condition tokens (perception functions)
        self.condition_tokens = [
            'frontIsClear',      # 23
            'leftIsClear',       # 24
            'rightIsClear',      # 25
            'markersPresent',    # 26
            'noMarkersPresent',  # 27
        ]
        
        # Logical tokens
        self.logical_tokens = [
            'not',           # 28
            'c(',            # 29
            'c)',            # 30
        ]
        
        # Padding token (for sequence padding)
        self.padding_tokens = [
            '<PAD>',         # 34
        ]
        
        # Combine all tokens in order
        self.all_tokens = (
            self.structure_tokens +     # 0-3
            self.action_tokens +        # 4-8
            self.control_tokens[:3] +   # 9-11 (REPEAT, r(, r))
            self.constant_tokens +      # 12-15
            self.control_tokens[3:9] +  # 16-22 (IF, IFELSE, ELSE, i(, i), e(, e))
            self.condition_tokens +     # 23-27
            self.logical_tokens +       # 28-30
            self.control_tokens[9:] +   # 31-33 (WHILE, w(, w))
            self.padding_tokens         # 34
        )
        
        self.vocab_size = len(self.all_tokens)
    
    def _create_mappings(self):
        """Create token-to-index and index-to-token mappings"""
        self.token_to_idx = {token: idx for idx, token in enumerate(self.all_tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.all_tokens)}
        
        # Create reverse lookup for token types
        self.token_types = {}
        
        for token in self.structure_tokens:
            self.token_types[token] = TokenType.PROGRAM_STRUCTURE
        
        for token in self.action_tokens:
            self.token_types[token] = TokenType.ACTIONS
        
        for token in self.control_tokens:
            self.token_types[token] = TokenType.CONTROL_FLOW
        
        for token in self.condition_tokens:
            self.token_types[token] = TokenType.CONDITIONS
        
        for token in self.constant_tokens:
            self.token_types[token] = TokenType.CONSTANTS
        
        for token in self.logical_tokens:
            self.token_types[token] = TokenType.LOGICAL
        
        for token in ['r(', 'r)', 'i(', 'i)', 'e(', 'e)', 'c(', 'c)', 'w(', 'w)']:
            self.token_types[token] = TokenType.BRACKETS
        
        for token in self.padding_tokens:
            self.token_types[token] = TokenType.PADDING
    
    def string_to_tokens(self, program_string: str) -> List[str]:
        """Convert program string to list of tokens"""
        return program_string.strip().split()
    
    def tokens_to_string(self, tokens: List[str]) -> str:
        """Convert list of tokens to program string"""
        return ' '.join(tokens)
    
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices"""
        return [self.token_to_idx.get(token, self.token_to_idx['<PAD>']) for token in tokens]
    
    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens"""
        return [self.idx_to_token.get(idx, '<PAD>') for idx in indices]
    
    def string_to_indices(self, program_string: str) -> List[int]:
        """Convert program string directly to indices"""
        tokens = self.string_to_tokens(program_string)
        return self.tokens_to_indices(tokens)
    
    def indices_to_string(self, indices: List[int]) -> str:
        """Convert indices directly to program string"""
        tokens = self.indices_to_tokens(indices)
        # Remove padding tokens
        tokens = [token for token in tokens if token != '<PAD>']
        return self.tokens_to_string(tokens)
    
    def filter_padding(self, tokens: List[str]) -> List[str]:
        """Remove padding tokens from token list"""
        return [token for token in tokens if token != '<PAD>']
    
    def filter_padding_indices(self, indices: List[int]) -> List[int]:
        """Remove padding token indices from index list"""
        padding_idx = self.token_to_idx['<PAD>']
        return [idx for idx in indices if idx != padding_idx]
    
    def get_token_type(self, token: str) -> TokenType:
        """Get the type of a token"""
        return self.token_types.get(token, TokenType.PADDING)
    
    def get_tokens_by_type(self, token_type: TokenType) -> List[str]:
        """Get all tokens of a specific type"""
        return [token for token, ttype in self.token_types.items() if ttype == token_type]
    
    def is_valid_program_start(self, tokens: List[str]) -> bool:
        """Check if tokens start with valid program structure"""
        if len(tokens) < 3:
            return False
        return tokens[0] == 'DEF' and tokens[1] == 'run' and tokens[2] == 'm('
    
    def is_valid_program_end(self, tokens: List[str]) -> bool:
        """Check if tokens end with valid program structure"""
        if len(tokens) == 0:
            return False
        return tokens[-1] == 'm)'
    
    def add_program_wrapper(self, stmt_tokens: List[str]) -> List[str]:
        """Wrap statement tokens with program structure"""
        return ['DEF', 'run', 'm('] + stmt_tokens + ['m)']
    
    def extract_statement(self, program_tokens: List[str]) -> List[str]:
        """Extract statement tokens from program (remove wrapper)"""
        if self.is_valid_program_start(program_tokens) and self.is_valid_program_end(program_tokens):
            return program_tokens[3:-1]  # Remove DEF run m( ... m)
        return program_tokens
    
    def validate_tokens(self, tokens: List[str]) -> Tuple[bool, str]:
        """Validate that all tokens are in vocabulary"""
        for token in tokens:
            if token not in self.token_to_idx:
                return False, f"Unknown token: {token}"
        return True, "All tokens valid"
    
    def get_action_indices(self) -> List[int]:
        """Get indices of action tokens"""
        return [self.token_to_idx[token] for token in self.action_tokens]
    
    def get_condition_indices(self) -> List[int]:
        """Get indices of condition tokens"""
        return [self.token_to_idx[token] for token in self.condition_tokens]
    
    def get_padding_index(self) -> int:
        """Get padding token index"""
        return self.token_to_idx['<PAD>']
    
    def pad_sequence(self, indices: List[int], max_length: int) -> List[int]:
        """Pad sequence to max_length with padding tokens"""
        if len(indices) >= max_length:
            return indices[:max_length]
        else:
            padding_needed = max_length - len(indices)
            padding_idx = self.get_padding_index()
            return indices + [padding_idx] * padding_needed
    
    def truncate_at_padding(self, indices: List[int]) -> List[int]:
        """Truncate sequence at first padding token"""
        padding_idx = self.get_padding_index()
        try:
            pad_start = indices.index(padding_idx)
            return indices[:pad_start]
        except ValueError:
            return indices  # No padding found
    
    def get_vocab_info(self) -> Dict:
        """Get vocabulary information"""
        return {
            'vocab_size': self.vocab_size,
            'num_actions': len(self.action_tokens),
            'num_conditions': len(self.condition_tokens),
            'num_constants': len(self.constant_tokens),
            'padding_idx': self.get_padding_index(),
            'token_types': {ttype.name: len(self.get_tokens_by_type(ttype)) 
                          for ttype in TokenType}
        }
    
    def print_vocabulary(self):
        """Print the complete vocabulary"""
        print("Karel DSL Vocabulary:")
        print("=" * 50)
        
        for token_type in TokenType:
            tokens = self.get_tokens_by_type(token_type)
            if tokens:
                print(f"\n{token_type.name}:")
                for token in tokens:
                    idx = self.token_to_idx[token]
                    print(f"  {idx:2d}: {token}")
        
        print(f"\nTotal vocabulary size: {self.vocab_size}")


# Create global instance
karel_tokens = KarelTokens()

# Export commonly used functions
def string_to_indices(program_string: str) -> List[int]:
    """Convert program string to indices"""
    return karel_tokens.string_to_indices(program_string)

def indices_to_string(indices: List[int]) -> str:
    """Convert indices to program string"""
    return karel_tokens.indices_to_string(indices)

def get_vocab_size() -> int:
    """Get vocabulary size"""
    return karel_tokens.vocab_size

def get_padding_index() -> int:
    """Get padding token index"""
    return karel_tokens.get_padding_index()

def pad_sequence(indices: List[int], max_length: int) -> List[int]:
    """Pad sequence to max_length"""
    return karel_tokens.pad_sequence(indices, max_length)


if __name__ == "__main__":
    # Example usage and testing
    tokens = KarelTokens()
    tokens.print_vocabulary()
    
    # Test basic conversion
    program = "DEF run m( move turnLeft move m)"
    print(f"\nTest program: {program}")
    
    indices = tokens.string_to_indices(program)
    print(f"Indices: {indices}")
    
    recovered = tokens.indices_to_string(indices)
    print(f"Recovered: {recovered}")
    
    print(f"\nVocabulary info: {tokens.get_vocab_info()}")