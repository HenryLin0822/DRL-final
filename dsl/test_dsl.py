"""
Test script for Karel DSL implementation

This script tests the DSL tokens, parser, and execution system.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dsl.tokens import karel_tokens, TokenType
from dsl.parser import KarelDSLParser, ParseError, ExecutionError
from dsl.karel_dsl import KarelDSL, get_DSL_option_v2
from environments.karel_world import KarelWorld


def test_tokens():
    """Test token management"""
    print("=" * 50)
    print("Testing Token Management")
    print("=" * 50)
    
    tokens = karel_tokens
    
    # Test basic properties
    print(f"Vocabulary size: {tokens.vocab_size}")
    print(f"Padding index: {tokens.get_padding_index()}")
    
    # Test token conversion
    test_program = "DEF run m( move turnLeft move m)"
    print(f"\nTest program: {test_program}")
    
    # String to indices and back
    indices = tokens.string_to_indices(test_program)
    print(f"Indices: {indices}")
    
    recovered = tokens.indices_to_string(indices)
    print(f"Recovered: {recovered}")
    
    assert test_program == recovered, "Token conversion failed"
    
    # Test padding
    padded = tokens.pad_sequence(indices, 20)
    print(f"Padded to 20: {padded}")
    
    truncated = tokens.truncate_at_padding(padded)
    print(f"Truncated: {truncated}")
    
    assert indices == truncated, "Padding/truncation failed"
    
    # Test token types
    print(f"\nToken types:")
    for token_type in TokenType:
        type_tokens = tokens.get_tokens_by_type(token_type)
        if type_tokens:
            print(f"  {token_type.name}: {len(type_tokens)} tokens")
    
    print("✓ Token management test passed!")


def test_parser():
    """Test DSL parser"""
    print("\n" + "=" * 50)
    print("Testing DSL Parser")
    print("=" * 50)
    
    parser = KarelDSLParser()
    
    # Test simple programs
    test_programs = [
        "DEF run m( move turnLeft move m)",
        "DEF run m( REPEAT R=3 r( move r) m)",
        "DEF run m( IF c( frontIsClear c) i( move i) m)",
        "DEF run m( WHILE c( frontIsClear c) w( move w) m)",
        "DEF run m( move IF c( markersPresent c) i( pickMarker i) turnRight m)"
    ]
    
    for i, program in enumerate(test_programs):
        print(f"\nTesting program {i+1}: {program}")
        
        # Parse program
        parsed = parser.parse(program)
        print(f"Valid: {parsed.is_valid}")
        
        if not parsed.is_valid:
            print(f"Error: {parsed.error_message}")
        else:
            print(f"Statements: {len(parsed.statements)}")
            for j, stmt in enumerate(parsed.statements):
                print(f"  {j+1}: {stmt['type']}")
        
        # Test validation
        valid, error = parser.validate_program(program)
        print(f"Validation: {valid}")
        if not valid:
            print(f"Validation error: {error}")
        
        assert parsed.is_valid == valid, f"Validation mismatch for program {i+1}"
    
    # Test invalid programs
    invalid_programs = [
        "move turnLeft",  # Missing structure
        "DEF run m( move",  # Incomplete
        "DEF run m( UNKNOWN_TOKEN m)",  # Unknown token
        "DEF run m( IF c( frontIsClear c) i( move m)",  # Unmatched brackets
    ]
    
    print(f"\nTesting invalid programs:")
    for i, program in enumerate(invalid_programs):
        print(f"  Invalid {i+1}: {program}")
        parsed = parser.parse(program)
        print(f"    Valid: {parsed.is_valid} (should be False)")
        assert not parsed.is_valid, f"Invalid program {i+1} was marked as valid"
    
    print("✓ Parser test passed!")


def test_execution():
    """Test program execution"""
    print("\n" + "=" * 50)
    print("Testing Program Execution")
    print("=" * 50)
    
    # Create Karel world
    karel_world = KarelWorld(task='harvester', grid_size=(6, 6))
    
    # Create initial state
    initial_state = np.zeros((6, 6, 8), dtype=bool)
    # Add walls
    initial_state[0, :, 4] = True
    initial_state[-1, :, 4] = True
    initial_state[:, 0, 4] = True
    initial_state[:, -1, 4] = True
    
    # Place agent
    initial_state[1, 1, 1] = True  # Facing East
    
    # Add markers
    initial_state[2, 2, 6] = True  # 1 marker
    initial_state[3, 3, 6] = True  # 1 marker
    
    # Initialize empty positions
    for r in range(1, 5):
        for c in range(1, 5):
            if not initial_state[r, c, :4].any() and not initial_state[r, c, 6:].any():
                initial_state[r, c, 5] = True  # No markers
    
    # Test programs
    parser = KarelDSLParser()
    
    test_programs = [
        "DEF run m( move move turnRight move m)",
        "DEF run m( REPEAT R=3 r( move r) m)",
        "DEF run m( IF c( frontIsClear c) i( move move i) m)",
        "DEF run m( WHILE c( frontIsClear c) w( move w) m)"
    ]
    
    for i, program in enumerate(test_programs):
        print(f"\nExecuting program {i+1}: {program}")
        
        # Reset world
        karel_world.reset(initial_state.copy())
        print(f"Initial agent position: {karel_world._get_agent_position()}")
        
        try:
            # Execute program
            execution_trace = parser.execute(program, karel_world)
            print(f"Execution successful: {len(execution_trace)} states")
            print(f"Final agent position: {karel_world._get_agent_position()}")
            print(f"Total reward: {karel_world.total_reward:.3f}")
            
        except Exception as e:
            print(f"Execution failed: {e}")
    
    # Test with token indices
    print(f"\nTesting execution with token indices:")
    program = "DEF run m( move turnLeft move pickMarker m)"
    tokens = karel_tokens.string_to_tokens(program)
    indices = karel_tokens.tokens_to_indices(tokens)
    
    karel_world.reset(initial_state.copy())
    try:
        execution_trace = parser.execute_indices(indices, karel_world)
        print(f"Index execution successful: {len(execution_trace)} states")
    except Exception as e:
        print(f"Index execution failed: {e}")
    
    print("✓ Execution test passed!")


def test_karel_dsl():
    """Test main Karel DSL interface"""
    print("\n" + "=" * 50)
    print("Testing Karel DSL Interface")
    print("=" * 50)
    
    # Test DSL creation
    dsl = KarelDSL(seed=42)
    print(f"DSL created with vocab size: {len(dsl.int2token)}")
    
    # Test compatibility with original interface
    print(f"Action functions: {dsl.action_functions}")
    print(f"Conditional functions: {dsl.conditional_functions}")
    
    # Test conversions
    test_program = "DEF run m( move turnLeft pickMarker m)"
    print(f"\nTest program: {test_program}")
    
    indices = dsl.str2intseq(test_program)
    print(f"To indices: {indices}")
    
    recovered = dsl.intseq2str(indices)
    print(f"From indices: {recovered}")
    
    assert test_program == recovered, "DSL conversion failed"
    
    # Test random program generation
    print(f"\nGenerating random programs:")
    for i in range(3):
        random_program = dsl.generate_random_program(max_length=12)
        print(f"  Random {i+1}: {random_program}")
        
        # Test validity
        valid, error = dsl.validate_program(random_program)
        print(f"    Valid: {valid}")
        if not valid:
            print(f"    Error: {error}")
    
    # Test complexity analysis
    complex_program = "DEF run m( REPEAT R=3 r( IF c( frontIsClear c) i( move turnLeft i) r) m)"
    complexity = dsl.get_program_complexity(complex_program)
    print(f"\nComplexity analysis of: {complex_program}")
    print(f"Complexity: {complexity}")
    
    # Test factory function
    dsl2 = get_DSL_option_v2(seed=123, environment='karel')
    print(f"\nFactory DSL created: {type(dsl2)}")
    
    print("✓ Karel DSL interface test passed!")


def test_integration_with_environments():
    """Test DSL integration with environments"""
    print("\n" + "=" * 50)
    print("Testing DSL Integration with Environments")
    print("=" * 50)
    
    # Import environment wrapper
    from environments.env_wrapper import make_vae_training_env
    
    # Create DSL and environment
    dsl = KarelDSL(seed=42)
    env = make_vae_training_env(
        task='harvester',
        grid_size=(6, 6),
        max_episode_steps=3,
        dsl_parser=dsl.parser
    )
    
    print(f"Environment created with DSL parser")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Environment reset successful")
    
    # Test program execution through environment
    test_program = "DEF run m( move turnLeft move m)"
    tokens = dsl.tokens.string_to_tokens(test_program)
    token_indices = dsl.tokens.tokens_to_indices(tokens)
    
    # Pad to environment's expected length
    padded_indices = dsl.tokens.pad_sequence(token_indices, env.max_program_length)
    
    print(f"Executing program through environment: {test_program}")
    print(f"Token indices: {token_indices}")
    
    try:
        obs, reward, terminated, truncated, info = env.step(padded_indices)
        print(f"Environment step successful:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Program valid: {info.get('program_valid', False)}")
        print(f"  Execution steps: {info.get('execution_steps', 0)}")
    except Exception as e:
        print(f"Environment step failed: {e}")
    
    env.close()
    print("✓ DSL-Environment integration test passed!")


def run_all_dsl_tests():
    """Run all DSL tests"""
    print("Starting Karel DSL Tests")
    print("=" * 70)
    
    try:
        test_tokens()
        test_parser()
        test_execution()
        test_karel_dsl()
        test_integration_with_environments()
        
        print("\n" + "=" * 70)
        print("🎉 ALL DSL TESTS PASSED! 🎉")
        print("Your Karel DSL implementation is working correctly!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ DSL TEST FAILED: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_dsl_tests()
    sys.exit(0 if success else 1)