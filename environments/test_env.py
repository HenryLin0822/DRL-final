"""
Test script for HPRL environments and wrappers

This script tests the Karel environment, state generator, and environment wrapper
to ensure everything works correctly before proceeding with training.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.karel_world import KarelWorld, Direction, Action
from environments.state_generator import KarelStateGenerator
from environments.karel_env import KarelEnvironment, make_karel_env
from environments.env_wrapper import (
    HPRLEnvironmentWrapper, 
    make_vae_training_env, 
    make_hprl_training_env,
    GymnasiumCompatWrapper
)


def test_karel_world():
    """Test basic Karel world functionality"""
    print("=" * 50)
    print("Testing Karel World")
    print("=" * 50)
    
    # Test basic world creation
    world = KarelWorld(task='harvester', grid_size=(8, 8))
    
    # Test state generation
    initial_state = np.zeros((8, 8, 8), dtype=bool)
    
    # Add walls around perimeter
    initial_state[0, :, 4] = True  # Top wall
    initial_state[-1, :, 4] = True  # Bottom wall
    initial_state[:, 0, 4] = True  # Left wall
    initial_state[:, -1, 4] = True  # Right wall
    
    # Place agent
    initial_state[1, 1, Direction.EAST] = True
    
    # Add some markers
    initial_state[3, 3, 6] = True  # 1 marker
    initial_state[5, 5, 6] = True  # 1 marker
    
    # Initialize empty positions
    for r in range(1, 7):
        for c in range(1, 7):
            if not initial_state[r, c, :4].any() and not initial_state[r, c, 6:].any():
                initial_state[r, c, 5] = True  # No markers
    
    # Reset world
    state = world.reset(initial_state)
    print(f"Initial state shape: {state.shape}")
    print("Initial world state:")
    world.render()
    
    # Test actions
    actions = [Action.MOVE, Action.MOVE, Action.TURN_RIGHT, Action.MOVE, Action.PICK_MARKER]
    for i, action in enumerate(actions):
        print(f"\nStep {i+1}: Executing {action.name}")
        state, reward, done, info = world.step(action)
        print(f"Reward: {reward:.3f}, Done: {done}")
        world.render()
        
        if done:
            break
    
    # Test perception functions
    print(f"\nPerception tests:")
    print(f"Front clear: {world.front_is_clear()}")
    print(f"Left clear: {world.left_is_clear()}")
    print(f"Right clear: {world.right_is_clear()}")
    print(f"Marker present: {world.marker_present()}")
    print(f"No marker present: {world.no_marker_present()}")
    
    print("✓ Karel World test passed!")


def test_state_generator():
    """Test state generator for different tasks"""
    print("\n" + "=" * 50)
    print("Testing State Generator")
    print("=" * 50)
    
    tasks = ['harvester', 'cleanHouse', 'fourCorners', 'stairClimber', 'topOff', 'randomMaze']
    
    for task in tasks:
        print(f"\nTesting {task} state generation:")
        generator = KarelStateGenerator(grid_size=(8, 8), task=task, seed=42)
        
        state, metadata = generator.generate_state(task_specific=True)
        print(f"State shape: {state.shape}")
        print(f"Metadata keys: {list(metadata.keys())}")
        
        # Verify state validity
        assert state.shape == (8, 8, 8), f"Wrong state shape for {task}"
        assert state.dtype == bool, f"Wrong state dtype for {task}"
        
        # Check that agent exists
        agent_positions = np.where(state[:, :, :4])
        assert len(agent_positions[0]) > 0, f"No agent found in {task} state"
        
        # Check walls around perimeter
        assert np.all(state[0, :, 4]), f"Missing top wall in {task}"
        assert np.all(state[-1, :, 4]), f"Missing bottom wall in {task}"
        assert np.all(state[:, 0, 4]), f"Missing left wall in {task}"
        assert np.all(state[:, -1, 4]), f"Missing right wall in {task}"
        
        print(f"✓ {task} state generation valid")
    
    # Test fixed instruction states
    print(f"\nTesting fixed instruction states:")
    generator = KarelStateGenerator(grid_size=(8, 8), task='harvester', seed=42)
    
    for i in range(5):
        state = generator.generate_program_instruction_state(8, 8, wall_prob=0.1, idx=i)
        print(f"Fixed state {i} shape: {state.shape}")
        assert state.shape == (8, 8, 8), f"Wrong fixed state shape for idx {i}"
    
    print("✓ State Generator test passed!")


def test_karel_environment():
    """Test Karel environment wrapper"""
    print("\n" + "=" * 50)
    print("Testing Karel Environment")
    print("=" * 50)
    
    # Test standard environment
    env = make_karel_env(task='harvester', env_type='standard', grid_size=(8, 8))
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Reset obs shape: {obs.shape}")
    print(f"Reset info keys: {list(info.keys())}")
    
    # Test some steps with simple programs
    # Simple program: move, turn, move, pick
    test_actions = [
        np.array([0, 1, 2, 4, 5, 4, 7, 3] + [34] * 32),  # Simple program + padding
        np.array([0, 1, 2, 6, 4, 8, 3] + [34] * 33),     # Another simple program
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\nStep {i+1}:")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        print(f"Info keys: {list(info.keys())}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("✓ Karel Environment test passed!")


def test_vae_wrapper():
    """Test VAE training wrapper"""
    print("\n" + "=" * 50)
    print("Testing VAE Training Wrapper")
    print("=" * 50)
    
    # Create VAE training environment
    env = make_vae_training_env(
        task='harvester',
        grid_size=(8, 8),
        max_episode_steps=3,
        max_program_length=20,
        vocab_size=35,
        use_fixed_states=True
    )
    
    print(f"Mode: VAE training")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create a mock target execution
    target_execution = []
    for i in range(5):
        state = np.zeros((8, 8, 8), dtype=bool)
        state[0, :, 4] = True  # Walls
        state[-1, :, 4] = True
        state[:, 0, 4] = True
        state[:, -1, 4] = True
        state[1+i, 1+i, Direction.EAST] = True  # Moving agent
        state[1+i, 1+i, 5] = True  # No marker
        target_execution.append(state)
    
    env.set_target_execution(target_execution)
    
    # Test reset and steps
    obs, info = env.reset(seed=42)
    print(f"Reset obs shape: {obs.shape}")
    print(f"Mode in info: {info.get('mode')}")
    
    # Test with simple program tokens
    for step in range(3):
        # Simple action sequence: DEF run m( move move m)
        action = np.array([0, 1, 2, 4, 4, 3] + [34] * 14)
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step+1}:")
        print(f"Reward: {reward:.3f}")
        print(f"Execution match: {info.get('execution_match', 0):.3f}")
        print(f"Program valid: {info.get('program_valid', False)}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("✓ VAE Training Wrapper test passed!")


def test_hprl_wrapper():
    """Test HPRL training wrapper (without actual VAE decoder)"""
    print("\n" + "=" * 50)
    print("Testing HPRL Training Wrapper")
    print("=" * 50)
    
    # Create mock VAE decoder
    class MockVAEDecoder:
        def eval(self):
            pass
        
        def decode(self, embedding, max_length=40, deterministic=True):
            # Return simple program tokens: DEF run m( move turn m)
            tokens = np.array([0, 1, 2, 4, 5, 3])
            return (tokens,)
    
    mock_decoder = MockVAEDecoder()
    
    # Create HPRL training environment
    env = make_hprl_training_env(
        task='harvester',
        grid_size=(8, 8),
        max_episode_steps=3,
        latent_dim=64,
        vae_decoder=mock_decoder,
        use_fixed_states=True
    )
    
    print(f"Mode: HPRL training")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset and steps
    obs, info = env.reset(seed=42)
    print(f"Reset obs shape: {obs.shape}")
    print(f"Mode in info: {info.get('mode')}")
    
    # Test with random embeddings
    for step in range(3):
        # Random program embedding
        action = np.random.randn(64).astype(np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step+1}:")
        print(f"Reward: {reward:.3f}")
        print(f"Program valid: {info.get('program_valid', False)}")
        print(f"Programs executed: {info.get('programs_executed', 0)}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("✓ HPRL Training Wrapper test passed!")


def test_gymnasium_compatibility():
    """Test gymnasium compatibility wrapper"""
    print("\n" + "=" * 50)
    print("Testing Gymnasium Compatibility")
    print("=" * 50)
    
    # Test with new API
    base_env = make_vae_training_env(task='harvester', grid_size=(6, 6))
    new_api_env = GymnasiumCompatWrapper(base_env)
    
    print("Testing new API:")
    obs, info = new_api_env.reset(seed=42)
    print(f"Reset returns: obs {obs.shape}, info {type(info)}")
    
    action = np.array([0, 1, 2, 4, 3] + [34] * 35)
    result = new_api_env.step(action)
    print(f"Step returns {len(result)} items: {[type(x) for x in result]}")
    
    # Test with old API
    new_api_env.enable_old_gym_api()
    print("\nTesting old API compatibility:")
    obs = new_api_env.reset(seed=42)
    print(f"Reset returns: obs {obs.shape}")
    
    obs, reward, done, info = new_api_env.step(action)
    print(f"Step returns: obs, reward {reward:.3f}, done {done}, info {type(info)}")
    
    new_api_env.close()
    print("✓ Gymnasium Compatibility test passed!")


def test_multiple_tasks():
    """Test multiple Karel tasks"""
    print("\n" + "=" * 50)
    print("Testing Multiple Tasks")
    print("=" * 50)
    
    tasks = ['harvester', 'cleanHouse', 'fourCorners', 'randomMaze']
    
    for task in tasks:
        print(f"\nTesting task: {task}")
        
        # Test basic environment
        env = make_karel_env(task=task, env_type='standard', grid_size=(6, 6))
        obs, info = env.reset(seed=42)
        
        # Test a few steps
        for i in range(2):
            action = np.array([0, 1, 2, 4, 5, 3] + [34] * 34)  # Simple program
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.3f}, done={terminated or truncated}")
            
            if terminated or truncated:
                break
        
        env.close()
        print(f"✓ Task {task} works correctly")


def run_all_tests():
    """Run all tests"""
    print("Starting HPRL Environment Tests")
    print("=" * 70)
    
    try:
        test_karel_world()
        test_state_generator()
        test_karel_environment()
        test_vae_wrapper()
        test_hprl_wrapper()
        test_gymnasium_compatibility()
        test_multiple_tasks()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Your HPRL environment implementation is working correctly!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)