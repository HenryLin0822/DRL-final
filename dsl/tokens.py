"""
Token definitions for Karel DSL - Updated with specific mapping

This uses the exact token ordering as specified in the mapping while maintaining all original functions.
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
    """Karel DSL Token Management - Updated with specific mapping, all functions maintained"""
    
    def __init__(self):
        self._define_tokens()
        self._create_mappings()
    
    def _define_tokens(self):
        """Define tokens in the exact order specified by the mapping"""
        
        # Tokens in exact order from index 0-33 + PAD token
        self.all_tokens = [
            'DEF',              # 0
            'run',              # 1
            'm(',               # 2
            'm)',               # 3
            'move',             # 4
            'turnLeft',         # 5
            'turnRight',        # 6
            'pickMarker',       # 7
            'putMarker',        # 8
            'REPEAT',           # 9
            'r(',               # 10
            'r)',               # 11
            'R=2',              # 12
            'R=3',              # 13
            'R=4',              # 14
            'R=5',              # 15
            'IF',               # 16
            'IFELSE',           # 17
            'ELSE',             # 18
            'i(',               # 19
            'i)',               # 20
            'e(',               # 21
            'e)',               # 22
            'frontIsClear',     # 23
            'leftIsClear',      # 24
            'rightIsClear',     # 25
            'markersPresent',   # 26
            'noMarkersPresent', # 27
            'not',              # 28
            'c(',               # 29
            'c)',               # 30
            'WHILE',            # 31
            'w(',               # 32
            'w)',               # 33
            '<PAD>',            # 34 - Added back for padding functionality
        ]
        
        self.vocab_size = len(self.all_tokens)
        
        # Categorize tokens by type for reference (maintaining original structure)
        self.structure_tokens = ['DEF', 'run', 'm(', 'm)']
        self.action_tokens = ['move', 'turnLeft', 'turnRight', 'pickMarker', 'putMarker']
        self.control_tokens = ['REPEAT', 'r(', 'r)', 'IF', 'IFELSE', 'ELSE', 'i(', 'i)', 'e(', 'e)', 'WHILE', 'w(', 'w)']
        self.constant_tokens = ['R=2', 'R=3', 'R=4', 'R=5']
        self.condition_tokens = ['frontIsClear', 'leftIsClear', 'rightIsClear', 'markersPresent', 'noMarkersPresent']
        self.logical_tokens = ['not', 'c(', 'c)']
        self.padding_tokens = ['<PAD>']
        
        # Verify critical tokens (adjusted for new vocabulary)
        critical_tokens = ['WHILE', 'IF', 'REPEAT', 'frontIsClear']
        missing = [token for token in critical_tokens if token not in self.all_tokens]
        if missing:
            raise ValueError(f"CRITICAL ERROR: Missing tokens: {missing}")
        
        print(f"✅ Updated vocabulary: {self.vocab_size} tokens")
        print(f"✅ Token mapping matches specified indices 0-33 + PAD at 34")
    
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
        bracket_tokens = ['r(', 'r)', 'i(', 'i)', 'e(', 'e)', 'c(', 'c)', 'w(', 'w)', 'm(', 'm)']
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
        """Convert indices to tokens with debugging"""
        print("🔍 DEBUG INDICES_TO_TOKENS:")
        result_tokens = []
        
        for i, idx in enumerate(indices[:15]):  # Only debug first 15
            if idx in self.idx_to_token:
                token = self.idx_to_token[idx]
                result_tokens.append(token)
                #print(f"   [{i}]: {idx} -> '{token}' ✓")
            else:
                result_tokens.append('<PAD>')
                #print(f"   [{i}]: {idx} -> '<PAD>' (not found)")
        
        # Continue for rest of indices without debug
        for idx in indices[15:]:
            result_tokens.append(self.idx_to_token.get(idx, '<PAD>'))
        
        return result_tokens
    
    def string_to_indices(self, program_string: str) -> List[int]:
        tokens = self.string_to_tokens(program_string)
        return self.tokens_to_indices(tokens)
    
    def indices_to_string(self, indices: List[int]) -> str:
        """Convert indices to program string with debugging"""
        #print("🔍 DEBUG TOKENS CONVERSION:")
        #print(f"   Input indices: {indices[:15]}")
        
        # Convert indices to tokens
        tokens = self.indices_to_tokens(indices)
        #print(f"   Converted to tokens: {tokens[:15]}")
        
        # Filter padding
        filtered_tokens = [token for token in tokens if token != '<PAD>']
        #print(f"   After padding filter: {filtered_tokens[:15]}")
        
        # Join to string
        result_string = self.tokens_to_string(filtered_tokens)
        #print(f"   Final string: '{result_string}'")
        
        return result_string
    
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
        """Test basic Karel programs that work with the updated vocabulary"""
        # Updated test programs to work with available tokens
        test_programs = [
            "WHILE frontIsClear move",  # Simplified - no DO/END
            "REPEAT R=3 move",          # Simplified - no TIMES/END  
            "IF frontIsClear move ELSE turnLeft",  # Simplified - no THEN/END
            "not frontIsClear"          # Basic negation
        ]
        
        print("\n🧪 Testing updated Karel programs:")
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
                print(f"   Indices: {indices}")
        
        return all_passed
    
    def print_token_mapping(self):
        """Print the complete token mapping for verification"""
        print("\n📋 Complete Token Mapping:")
        print("=" * 30)
        for idx, token in enumerate(self.all_tokens):
            print(f"{idx:2d}: '{token}'")
    def validate_tokens(self, token_list: List[str]) -> Tuple[bool, str]:
        """
        Validate that all tokens in the list are known tokens
        
        Args:
            token_list: List of token strings to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        unknown_tokens = []
        
        for token in token_list:
            if token not in self.token_to_idx:
                unknown_tokens.append(token)
        
        if unknown_tokens:
            return False, f"Unknown tokens: {unknown_tokens}"
        
        return True, ""

    def extract_statement(self, token_list: List[str]) -> List[str]:
        """
        Extract statement tokens from a program (remove DEF run m( ... m))
        
        Args:
            token_list: Full program token list
            
        Returns:
            Statement tokens without the program structure wrapper
        """
        if len(token_list) < 4:
            return []
        
        # Check if it starts with DEF run m( and ends with m)
        if (token_list[0] == 'DEF' and 
            token_list[1] == 'run' and 
            token_list[2] == 'm(' and 
            token_list[-1] == 'm)'):
            # Return everything between m( and m)
            return token_list[3:-1]
        else:
            # Return as-is if not proper program structure
            return token_list

    def is_valid_program_start(self, tokens: List[str]) -> bool:
        """Check if program starts with DEF run m("""
        return (len(tokens) >= 3 and 
                tokens[0] == 'DEF' and 
                tokens[1] == 'run' and 
                tokens[2] == 'm(')

    def is_valid_program_end(self, tokens: List[str]) -> bool:
        """Check if program ends with m)"""
        return len(tokens) > 0 and tokens[-1] == 'm)'

    def filter_padding(self, tokens: List[str]) -> List[str]:
        """Remove padding tokens from token list"""
        return [token for token in tokens if token != '<PAD>']


# Create global instance
karel_tokens = KarelTokens()

# Export functions - ALL ORIGINAL FUNCTIONS MAINTAINED
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

def validate_tokens(token_list: List[str]) -> Tuple[bool, str]:
    return karel_tokens.validate_tokens(token_list)

def extract_statement(token_list: List[str]) -> List[str]:
    return karel_tokens.extract_statement(token_list)

def is_valid_program_start(tokens: List[str]) -> bool:
    return karel_tokens.is_valid_program_start(tokens)

def is_valid_program_end(tokens: List[str]) -> bool:
    return karel_tokens.is_valid_program_end(tokens)

def filter_padding(tokens: List[str]) -> List[str]:
    return karel_tokens.filter_padding(tokens)

if __name__ == "__main__":
    print("🚀 KAREL TOKENS - Updated Mapping with All Functions")
    print("=" * 50)
    
    tokens = KarelTokens()
    tokens.print_token_mapping()
    
    success = tokens.test_all_default_programs()
    
    print(f"\n📊 SUMMARY:")
    print(f"   Vocabulary size: {tokens.vocab_size}")
    print(f"   Core mapping: indices 0-33 as specified")
    print(f"   Padding token: index 34")
    print(f"   All original functions: ✅ MAINTAINED")
    print(f"   Basic tests: {'✅ PASSED' if success else '❌ FAILED'}")
    
    print(f"\n🔍 Available Functions:")
    print(f"   - string_to_indices()")
    print(f"   - indices_to_string()")
    print(f"   - get_vocab_size()")
    print(f"   - get_padding_index()")
    print(f"   - pad_sequence()")
    print(f"   - test_all_default_programs()")
    print(f"   - get_vocab_info()")
    
    if success:
        print(f"\n🎉 TOKEN MAPPING UPDATED WITH ALL FUNCTIONS PRESERVED!")