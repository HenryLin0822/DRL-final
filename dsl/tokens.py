"""
Token definitions for Karel DSL - FINAL COMPLETE VERSION

This includes ALL possible Karel condition tokens to prevent any missing token issues.
"""

from typing import Dict, List, Tuple
from enum import IntEnum


class TokenType(IntEnum):
    """Token types for organizing the DSL vocabulary"""
    PROGRAM_STRUCTURE = 0
    CONTROL_FLOW = 1
    CONDITIONS = 2
    ACTIONS = 3
    CONSTANTS = 4
    BRACKETS = 5
    LOGICAL = 6
    PADDING = 7


class KarelTokens:
    """Complete Karel DSL Token Management - FINAL VERSION"""
    
    def __init__(self):
        self._define_tokens()
        self._create_mappings()
    
    def _define_tokens(self):
        """Define ALL DSL tokens"""
        
        # Program structure tokens
        self.structure_tokens = [
            'DEF',           # 0
            'run',           # 1
            'm(',            # 2
            'm)',            # 3
        ]
        
        # Action tokens
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
            'TIMES',         # 10
            'r(',            # 11
            'r)',            # 12
            'IF',            # 13
            'THEN',          # 14
            'IFELSE',        # 15
            'ELSE',          # 16
            'i(',            # 17
            'i)',            # 18
            'e(',            # 19
            'e)',            # 20
            'WHILE',         # 21
            'DO',            # 22
            'END',           # 23
            'w(',            # 24
            'w)',            # 25
        ]
        
        # Number tokens
        self.constant_tokens = [
            '1',             # 26
            '2',             # 27
            '3',             # 28
            '4',             # 29
            '5',             # 30
            'R=2',           # 31
            'R=3',           # 32
            'R=4',           # 33
            'R=5',           # 34
        ]
        
        # COMPLETE condition tokens - FIXED: Added ALL missing conditions
        self.condition_tokens = [
            'frontIsClear',      # 35
            'leftIsClear',       # 36
            'rightIsClear',      # 37
            'backIsClear',       # 38 - ADDED
            'markersPresent',    # 39
            'noMarkersPresent',  # 40
            'facingNorth',       # 41 - ADDED
            'facingSouth',       # 42 - ADDED
            'facingEast',        # 43 - ADDED
            'facingWest',        # 44 - ADDED
            'notFacingNorth',    # 45 - ADDED (this was the missing one!)
            'notFacingSouth',    # 46 - ADDED
            'notFacingEast',     # 47 - ADDED
            'notFacingWest',     # 48 - ADDED
        ]
        
        # Logical tokens
        self.logical_tokens = [
            'not',           # 49
            'c(',            # 50
            'c)',            # 51
        ]
        
        # Padding token
        self.padding_tokens = [
            '<PAD>',         # 52
        ]
        
        # Combine all tokens
        self.all_tokens = (
            self.structure_tokens +     # 0-3
            self.action_tokens +        # 4-8
            self.control_tokens +       # 9-25
            self.constant_tokens +      # 26-34
            self.condition_tokens +     # 35-48
            self.logical_tokens +       # 49-51
            self.padding_tokens         # 52
        )
        
        self.vocab_size = len(self.all_tokens)
        
        # Verify critical tokens
        critical_tokens = ['DO', 'END', 'THEN', 'TIMES', 'notFacingNorth']
        missing = [token for token in critical_tokens if token not in self.all_tokens]
        if missing:
            raise ValueError(f"CRITICAL ERROR: Still missing tokens: {missing}")
        
        print(f"✅ FINAL complete vocabulary: {self.vocab_size} tokens")
        print(f"✅ All critical tokens included: {critical_tokens}")
    
    def _create_mappings(self):
        """Create token-to-index and index-to-token mappings"""
        self.token_to_idx = {token: idx for idx, token in enumerate(self.all_tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.all_tokens)}
        
        # Create token types mapping
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
        for token in self.padding_tokens:
            self.token_types[token] = TokenType.PADDING
        
        # Bracket tokens
        bracket_tokens = ['r(', 'r)', 'i(', 'i)', 'e(', 'e)', 'c(', 'c)', 'w(', 'w)']
        for token in bracket_tokens:
            if token in self.all_tokens:
                self.token_types[token] = TokenType.BRACKETS
    
    def string_to_tokens(self, program_string: str) -> List[str]:
        return program_string.strip().split()
    
    def tokens_to_string(self, tokens: List[str]) -> str:
        return ' '.join(tokens)
    
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        indices = []
        for token in tokens:
            if token in self.token_to_idx:
                indices.append(self.token_to_idx[token])
            else:
                print(f"WARNING: Unknown token '{token}' mapped to <PAD>")
                indices.append(self.token_to_idx['<PAD>'])
        return indices
    
    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.idx_to_token.get(idx, '<PAD>') for idx in indices]
    
    def string_to_indices(self, program_string: str) -> List[int]:
        tokens = self.string_to_tokens(program_string)
        return self.tokens_to_indices(tokens)
    
    def indices_to_string(self, indices: List[int]) -> str:
        tokens = self.indices_to_tokens(indices)
        tokens = [token for token in tokens if token != '<PAD>']
        return self.tokens_to_string(tokens)
    
    def get_padding_index(self) -> int:
        return self.token_to_idx['<PAD>']
    
    def get_vocab_info(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'num_actions': len(self.action_tokens),
            'num_conditions': len(self.condition_tokens),
            'padding_idx': self.get_padding_index(),
        }
    
    def test_all_default_programs(self):
        """Test the exact programs used in inference"""
        test_programs = [
            "WHILE frontIsClear DO move END",
            "REPEAT 3 TIMES move END",
            "IF frontIsClear THEN move ELSE turnLeft END",
            "WHILE notFacingNorth DO turnLeft END"  # This was failing!
        ]
        
        print("\n🧪 Testing ALL default inference programs:")
        print("=" * 50)
        
        all_passed = True
        for program in test_programs:
            tokens = self.string_to_tokens(program)
            indices = self.tokens_to_indices(tokens)
            reconstructed = self.indices_to_string(indices)
            
            unknown_tokens = [token for token in tokens if token not in self.token_to_idx]
            has_padding = self.get_padding_index() in indices
            perfect_match = program == reconstructed
            
            status = "✅" if perfect_match and not unknown_tokens else "❌"
            print(f"\n{status} Program: '{program}'")
            
            if unknown_tokens:
                print(f"   ❌ Unknown tokens: {unknown_tokens}")
                all_passed = False
            if not perfect_match:
                print(f"   ❌ Mismatch: '{reconstructed}'")
                all_passed = False
            if perfect_match and not unknown_tokens:
                print(f"   ✅ Perfect tokenization")
        
        return all_passed


# Create global instance
karel_tokens = KarelTokens()

# Export functions
def string_to_indices(program_string: str) -> List[int]:
    return karel_tokens.string_to_indices(program_string)

def indices_to_string(indices: List[int]) -> str:
    return karel_tokens.indices_to_string(indices)

def get_vocab_size() -> int:
    return karel_tokens.vocab_size

def get_padding_index() -> int:
    return karel_tokens.get_padding_index()

def pad_sequence(indices: List[int], max_length: int) -> List[int]:
    if len(indices) >= max_length:
        return indices[:max_length]
    else:
        padding_needed = max_length - len(indices)
        padding_idx = get_padding_index()
        return indices + [padding_idx] * padding_needed


if __name__ == "__main__":
    print("🚀 FINAL COMPLETE KAREL TOKENS")
    print("=" * 50)
    
    tokens = KarelTokens()
    success = tokens.test_all_default_programs()
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Vocabulary size: {tokens.vocab_size}")
    print(f"   All tests: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if success:
        print(f"\n🎉 VOCABULARY IS NOW COMPLETELY FIXED!")
        print(f"   No more missing token warnings should appear")
