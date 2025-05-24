"""
Token definitions for Karel DSL - CORRECTED COMPLETE VERSION

This module defines ALL the tokens used in the Karel Domain Specific Language,
including the previously missing keywords: DO, END, THEN, TIMES, and proper number tokens.

This fixes the critical vocabulary gaps that caused training on corrupted data.
"""

from typing import Dict, List, Tuple
from enum import IntEnum


class TokenType(IntEnum):
    """Token types for organizing the DSL vocabulary"""
    PROGRAM_STRUCTURE = 0  # DEF, run, m(, m)
    CONTROL_FLOW = 1       # IF, WHILE, REPEAT, etc.
    CONDITIONS = 2         # frontIsClear, markersPresent, etc.
    ACTIONS = 3           # move, turnLeft, etc.
    CONSTANTS = 4         # 1, 2, 3, 4, 5, etc.
    BRACKETS = 5          # c(, c), w(, w), etc.
    LOGICAL = 6           # not, and, or
    PADDING = 7           # Padding token


class KarelTokens:
    """
    Karel DSL Token Management - COMPLETE VERSION
    
    This class handles all token definitions, mappings, and conversions
    for the Karel Domain Specific Language with ALL necessary tokens included.
    """
    
    def __init__(self):
        # Core token definitions
        self._define_tokens()
        self._create_mappings()
    
    def _define_tokens(self):
        """Define all DSL tokens including previously missing ones"""
        
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
        
        # Control flow tokens - FIXED: Added all missing keywords
        self.control_tokens = [
            'REPEAT',        # 9
            'TIMES',         # 10 - ADDED: Was missing, caused "REPEAT 3 TIMES" to fail
            'r(',            # 11
            'r)',            # 12
            'IF',            # 13
            'THEN',          # 14 - ADDED: Was missing, caused "IF...THEN" to fail  
            'IFELSE',        # 15
            'ELSE',          # 16
            'i(',            # 17
            'i)',            # 18
            'e(',            # 19
            'e)',            # 20
            'WHILE',         # 21
            'DO',            # 22 - ADDED: Was missing, caused "WHILE...DO" to fail
            'END',           # 23 - ADDED: Was missing, caused "...END" to fail
            'w(',            # 24
            'w)',            # 25
        ]
        
        # Number tokens - FIXED: Use simple numbers AND keep R=X for compatibility
        self.constant_tokens = [
            '1',             # 26 - ADDED: For "REPEAT 1 TIMES"
            '2',             # 27 - ADDED: For "REPEAT 2 TIMES"
            '3',             # 28 - ADDED: For "REPEAT 3 TIMES"
            '4',             # 29 - ADDED: For "REPEAT 4 TIMES"
            '5',             # 30 - ADDED: For "REPEAT 5 TIMES"
            'R=2',           # 31 - Keep for backward compatibility
            'R=3',           # 32 - Keep for backward compatibility
            'R=4',           # 33 - Keep for backward compatibility
            'R=5',           # 34 - Keep for backward compatibility
        ]
        
        # Condition tokens (perception functions)
        self.condition_tokens = [
            'frontIsClear',      # 35
            'leftIsClear',       # 36
            'rightIsClear',      # 37
            'markersPresent',    # 38
            'noMarkersPresent',  # 39
        ]
        
        # Logical tokens
        self.logical_tokens = [
            'not',           # 40
            'c(',            # 41
            'c)',            # 42
        ]
        
        # Padding token (for sequence padding)
        self.padding_tokens = [
            '<PAD>',         # 43
        ]
        
        # Combine all tokens in order - COMPLETE VOCABULARY
        self.all_tokens = (
            self.structure_tokens +     # 0-3
            self.action_tokens +        # 4-8
            self.control_tokens +       # 9-25
            self.constant_tokens +      # 26-34
            self.condition_tokens +     # 35-39
            self.logical_tokens +       # 40-42
            self.padding_tokens         # 43
        )
        
        self.vocab_size = len(self.all_tokens)
        
        # Verify we have all critical tokens
        critical_tokens = ['DO', 'END', 'THEN', 'TIMES', '1', '2', '3', '4', '5']
        missing = [token for token in critical_tokens if token not in self.all_tokens]
        if missing:
            raise ValueError(f"CRITICAL ERROR: Still missing tokens: {missing}")
        
        print(f"✅ Complete Karel vocabulary initialized with {self.vocab_size} tokens")
        print(f"✅ All critical tokens included: {critical_tokens}")
    
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
        
        # Bracket tokens
        bracket_tokens = ['r(', 'r)', 'i(', 'i)', 'e(', 'e)', 'c(', 'c)', 'w(', 'w)']
        for token in bracket_tokens:
            if token in self.all_tokens:
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
        """Convert tokens to indices with proper error handling"""
        indices = []
        for token in tokens:
            if token in self.token_to_idx:
                indices.append(self.token_to_idx[token])
            else:
                print(f"WARNING: Unknown token '{token}' mapped to <PAD>")
                indices.append(self.token_to_idx['<PAD>'])
        return indices
    
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
        unknown_tokens = []
        for token in tokens:
            if token not in self.token_to_idx:
                unknown_tokens.append(token)
        
        if unknown_tokens:
            return False, f"Unknown tokens: {unknown_tokens}"
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
    
    def test_critical_programs(self):
        """Test that critical Karel programs tokenize correctly"""
        test_programs = [
            "move",
            "WHILE frontIsClear DO move END",
            "REPEAT 3 TIMES move END", 
            "IF frontIsClear THEN move ELSE turnLeft END",
            "DEF run m( WHILE frontIsClear DO move END m)"
        ]
        
        print("\n🧪 Testing critical program tokenization:")
        print("=" * 50)
        
        all_passed = True
        for program in test_programs:
            tokens = self.string_to_tokens(program)
            indices = self.tokens_to_indices(tokens)
            reconstructed = self.indices_to_string(indices)
            
            # Check for issues
            unknown_tokens = [token for token in tokens if token not in self.token_to_idx]
            has_padding = self.get_padding_index() in indices
            perfect_match = program == reconstructed
            
            status = "✅" if perfect_match and not unknown_tokens and not has_padding else "❌"
            print(f"\n{status} Program: '{program}'")
            print(f"   Tokens: {tokens}")
            print(f"   Indices: {indices}")
            print(f"   Reconstructed: '{reconstructed}'")
            
            if unknown_tokens:
                print(f"   ❌ Unknown tokens: {unknown_tokens}")
                all_passed = False
            if has_padding:
                print(f"   ❌ Unexpected padding in indices")
                all_passed = False
            if not perfect_match:
                print(f"   ❌ Reconstruction mismatch")
                all_passed = False
            if perfect_match and not unknown_tokens and not has_padding:
                print(f"   ✅ Perfect tokenization")
        
        print(f"\n{'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        return all_passed
    
    def print_vocabulary(self):
        """Print the complete vocabulary organized by type"""
        print("Complete Karel DSL Vocabulary:")
        print("=" * 50)
        
        for token_type in TokenType:
            tokens = self.get_tokens_by_type(token_type)
            if tokens:
                print(f"\n{token_type.name}:")
                for token in tokens:
                    idx = self.token_to_idx[token]
                    print(f"  {idx:2d}: '{token}'")
        
        print(f"\nTotal vocabulary size: {self.vocab_size}")
        print(f"Padding index: {self.get_padding_index()}")


# Create global instance
karel_tokens = KarelTokens()

# Export commonly used functions for backward compatibility
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
    # Test the corrected tokenization
    print("🚀 CORRECTED KAREL DSL TOKENS")
    print("=" * 60)
    
    tokens = KarelTokens()
    
    # Print vocabulary
    tokens.print_vocabulary()
    
    # Test critical programs
    success = tokens.test_critical_programs()
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"   Vocabulary size: {tokens.vocab_size} (was 35, now complete)")
    print(f"   Padding index: {tokens.get_padding_index()}")
    print(f"   Critical tests: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if success:
        print(f"\n🎉 VOCABULARY IS NOW COMPLETE!")
        print(f"   Replace your dsl/tokens.py with this file")
        print(f"   Then retrain your model from scratch")
    else:
        print(f"\n⚠️  VOCABULARY STILL HAS ISSUES!")
        print(f"   Check the test output above for details")