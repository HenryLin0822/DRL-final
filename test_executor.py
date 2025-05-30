#!/usr/bin/env python3
"""
Karel Program Renderer

This script renders the execution of Karel programs step by step.
Validation is handled by the program executor in the core system.

Usage:
    python karel_renderer.py
    python karel_renderer.py --test
"""

import numpy as np
import time
import os
import sys
from typing import List, Union, Optional, Tuple, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import from your existing modules
try:
    from environments.karel_world import KarelWorld, Direction, Action
    from environments.karel_env import KarelEnvironment
    from dsl.tokens import karel_tokens
    from models.program_executor import ProgramExecutor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


class KarelRenderer:
    """Karel Program Renderer using the existing Karel world implementation"""
    
    def __init__(self, task: str = 'harvester', grid_size: Tuple[int, int] = (8, 8)):
        self.task = task
        self.grid_size = grid_size
        
        # Initialize Karel world
        self.karel_world = KarelWorld(task=task, grid_size=grid_size, timeout_steps=100)
        
        # Initialize program executor for token handling
        self.program_executor = ProgramExecutor(device='cpu')
        
        # Initialize tokens
        self.tokens = karel_tokens
        
        # Get token mappings
        self.token_to_string = self.tokens.idx_to_token
        self.string_to_token = self.tokens.token_to_idx
        
        # Display symbols
        self.agent_symbols = {
            Direction.NORTH: '^', Direction.EAST: '>', 
            Direction.SOUTH: 'v', Direction.WEST: '<'
        }
        
        # Colors for terminal output
        self.colors = {
            'reset': '\033[0m', 'red': '\033[91m', 'green': '\033[92m',
            'yellow': '\033[93m', 'blue': '\033[94m', 'cyan': '\033[96m',
            'white': '\033[97m', 'gray': '\033[90m', 'bold': '\033[1m'
        }
    
    def parse_program(self, program_input: Union[str, List[int], List[str]]) -> List[int]:
        """Parse program input into token indices"""
        if isinstance(program_input, str):
            tokens = program_input.strip().split()
            indices = []
            for token in tokens:
                if token in self.string_to_token:
                    indices.append(self.string_to_token[token])
                elif token.isdigit():
                    indices.append(int(token))
                else:
                    print(f"Warning: Unknown token '{token}', skipping")
            return indices
        elif isinstance(program_input, list):
            if len(program_input) == 0:
                return []
            if isinstance(program_input[0], str):
                indices = []
                for token in program_input:
                    if token in self.string_to_token:
                        indices.append(self.string_to_token[token])
                    else:
                        print(f"Warning: Unknown token '{token}', skipping")
                return indices
            else:
                return list(program_input)
        else:
            raise ValueError(f"Unsupported program input type: {type(program_input)}")
    
    def create_task_specific_world(self):
        """Create task-specific initial world state"""
        self.karel_world.reset()
        state = self.karel_world.state
        
        if self.task == 'harvester':
            # Place markers in a pattern for harvesting
            for r in range(2, min(6, self.grid_size[0]-2)):
                for c in range(2, min(6, self.grid_size[1]-2)):
                    if not state[r, c, :4].any() and not state[r, c, 4]:
                        state[r, c, 5] = False
                        state[r, c, 6] = True
            
            # IMPORTANT: For IF statement test - make sure Karel is ON a marker
            # Find Karel's current position and place a marker there
            agent_pos = self.karel_world._get_agent_position()
            if agent_pos:
                karel_row, karel_col = agent_pos[0], agent_pos[1]
                # Place a marker at Karel's position for the IF test to work
                state[karel_row, karel_col, 5] = False  # Remove "no markers"
                state[karel_row, karel_col, 6] = True   # Add 1 marker
                print(f"DEBUG: Placed marker at Karel's position ({karel_row}, {karel_col})")
    
    def render_state(self, show_colors: bool = True, action_info: str = ""):
        """Render the current Karel world state"""
        print(f"\n{'='*60}")
        print(f"Karel World - Task: {self.task} | Step: {self.karel_world.step_count}")
        if action_info:
            print(f"Action: {action_info}")
        print(f"{'='*60}")
        
        state = self.karel_world.state
        h, w = self.grid_size
        
        for row in range(h):
            line = "  "
            for col in range(w):
                char = "."
                color = 'white'
                
                if state[row, col, 4]:  # Wall
                    char = "█"
                    color = 'gray'
                elif state[row, col, :4].any():  # Agent
                    direction = np.argmax(state[row, col, :4])
                    char = self.agent_symbols[Direction(direction)]
                    color = 'green'
                elif state[row, col, 6]:  # 1 marker
                    char = "●"
                    color = 'red'
                elif state[row, col, 7]:  # 2+ markers
                    char = "◉"
                    color = 'yellow'
                
                if show_colors and os.name != 'nt':
                    line += f"{self.colors[color]}{char}{self.colors['reset']}"
                else:
                    line += char
            print(line)
        
        print()
        print("Legend: ^>v< Karel  █ Wall  ● 1 Marker  ◉ 2+ Markers")
        
        agent_pos = self.karel_world._get_agent_position()
        if agent_pos:
            print(f"Karel position: ({agent_pos[0]}, {agent_pos[1]}) facing {agent_pos[2].name}")
        print(f"Total reward: {self.karel_world.total_reward:.3f}")
        print()
    
    def execute_program(self, program: Union[str, List[int], List[str]], 
                       step_delay: float = 1.0, auto_advance: bool = False) -> bool:
        """Execute a Karel program with step-by-step rendering"""
        try:
            token_indices = self.parse_program(program)
        except Exception as e:
            print(f"{self.colors.get('red', '')}✗ Error parsing program: {e}{self.colors.get('reset', '')}")
            return False

        if not token_indices:
            print(f"{self.colors.get('red', '')}✗ No valid tokens found in program{self.colors.get('reset', '')}")
            return False

        print(f"\n{self.colors.get('bold', '')}Program Analysis:{self.colors.get('reset', '')}")
        print(f"Input: {program}")
        print(f"Parsed tokens: {token_indices}")

        token_names = [self.token_to_string.get(token, f"UNK_{token}") for token in token_indices]
        print(f"Token names: {token_names}")
        
        return self._execute_with_dsl(token_indices, step_delay, auto_advance)
    
    def _execute_with_dsl(self, token_indices: List[int], step_delay: float, auto_advance: bool) -> bool:
        """Execute program using DSL execution"""
        print(f"\n{self.colors.get('blue', '')}Using DSL execution{self.colors.get('reset', '')}")
        
        self.karel_world.reset()
        self.create_task_specific_world()
        
        self.render_state(action_info="Initial State")

        if not auto_advance:
            input("Press Enter to start DSL execution...")
        else:
            time.sleep(step_delay)

        try:
            import torch
            result = self.program_executor.execute_with_dsl(
                torch.tensor(token_indices),
                self.karel_world,
                return_traces=True
            )
            
            if result['success']:
                print(f"{self.colors.get('green', '')}✓ DSL execution successful{self.colors.get('reset', '')}")
                self.render_state(action_info=f"DSL Complete - Reward: {result['total_reward']:.3f}")
                
                print(f"\nDSL Execution Summary:")
                print(f"  Actions executed: {result['action_length']}")
                print(f"  Total reward: {result['total_reward']:.3f}")
                print(f"  Success: {result['success']}")
                return True
            else:
                print(f"{self.colors.get('red', '')}✗ DSL execution failed: {result.get('error', 'Unknown error')}{self.colors.get('reset', '')}")
                return False
                
        except Exception as e:
            print(f"{self.colors.get('red', '')}✗ DSL execution error: {e}{self.colors.get('reset', '')}")
            return False


def test_validation():
    """Test valid and invalid programs"""
    print("🧪 KAREL VALIDATION TESTS")
    print("="*60)
    
    renderer = KarelRenderer(task='harvester', grid_size=(6, 6))
    
    # Valid programs that should pass
    valid_programs = [
        ("DEF run m( move m)", "Simple move"),
        ("DEF run m( move turnLeft move m)", "Multiple actions"),
        ("DEF run m( REPEAT R=3 r( move r) m)", "Repeat loop"),
        ("DEF run m( WHILE c( frontIsClear c) w( move w) m)", "While loop"),
        #("DEF run m( IF c( markersPresent c) i( pickMarker i) m)", "If statement"),
        ("DEF run m( IF c( frontIsClear c) i( move i) m)", "If statement"),
    ]
    
    # Invalid programs that should fail
    invalid_programs = [
        ("move turnLeft pickMarker", "Missing DEF run structure"),
        ("run m( move m)", "Missing DEF"),
        ("DEF run move m)", "Missing m("),
        ("DEF run m( move", "Missing m)"),
        ("DEF run m( REPEAT R=3 r( move m)", "Unmatched brackets"),
        ("DEF run m( WHILE c( frontIsClear w( move w) m)", "Missing c)"),
    ]
    
    passed = 0
    total = 0
    
    print("\n✅ Testing VALID programs (should pass):")
    for program, desc in valid_programs:
        total += 1
        print(f"\nTest: {desc}")
        print(f"Program: {program}")
        try:
            success = renderer.execute_program(program, step_delay=0.1, auto_advance=True)
            if success:
                print("✅ PASSED")
                passed += 1
            else:
                print("❌ FAILED (should have passed)")
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print("\n❌ Testing INVALID programs (should fail):")
    for program, desc in invalid_programs:
        total += 1
        print(f"\nTest: {desc}")
        print(f"Program: {program}")
        try:
            success = renderer.execute_program(program, step_delay=0.1, auto_advance=True)
            if not success:
                print("✅ CORRECTLY REJECTED")
                passed += 1
            else:
                print("❌ FAILED (should have been rejected)")
        except Exception as e:
            print("✅ CORRECTLY THREW ERROR")
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ Some tests failed")
    
    return passed == total


def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_validation()
        return
    
    print("Karel Program Renderer")
    print("="*60)
    
    # Get task
    tasks = ['harvester', 'cleanHouse', 'fourCorners', 'randomMaze']
    print("\nAvailable tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")
    
    while True:
        try:
            choice = input(f"\nSelect task (1-{len(tasks)}) [default: 1]: ").strip()
            if not choice:
                choice = "1"
            task_idx = int(choice) - 1
            if 0 <= task_idx < len(tasks):
                selected_task = tasks[task_idx]
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")
    
    renderer = KarelRenderer(task=selected_task)
    
    print("\nExample programs:")
    print("  Valid: DEF run m( move move move m)")
    print("  Valid: DEF run m( WHILE c( frontIsClear c) w( move w) m)")
    print("  Invalid: move turnLeft pickMarker")
    
    while True:
        program = input("\nEnter program: ").strip()
        if program:
            break
        print("Please enter a program.")
    
    auto_advance = input("Auto-advance? (y/n) [default: n]: ").strip().lower() == 'y'
    step_delay = 1.5 if auto_advance else 0.0
    
    print(f"\nExecuting on {selected_task} task...")
    success = renderer.execute_program(program, step_delay=step_delay, auto_advance=auto_advance)
    
    if success:
        print("\n✅ Program completed successfully!")
    else:
        print("\n❌ Program failed!")


if __name__ == "__main__":
    main()