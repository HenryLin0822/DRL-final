"""
Core Karel World Implementation

This module contains the core Karel world mechanics, handling state representation,
action execution, and reward calculation for various Karel tasks.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from enum import IntEnum
import copy


class Direction(IntEnum):
    """Karel's facing directions"""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Action(IntEnum):
    """Karel's primitive actions"""
    MOVE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    PICK_MARKER = 3
    PUT_MARKER = 4


class KarelWorld:
    """
    Core Karel World implementation with support for multiple tasks.
    
    State representation: (H, W, 8) boolean array
    - Channels 0-3: Agent direction (one-hot)
    - Channel 4: Walls
    - Channel 5: No markers (0 markers)
    - Channel 6: 1 marker
    - Channel 7: 2+ markers (for topOff task)
    """
    
    def __init__(
        self, 
        task: str = 'harvester',
        grid_size: Tuple[int, int] = (8, 8),
        max_markers: int = 2,
        timeout_steps: int = 100
    ):
        self.task = task
        self.h, self.w = grid_size
        self.max_markers = max_markers
        self.timeout_steps = timeout_steps
        
        # State representation
        self.state = np.zeros((self.h, self.w, 8), dtype=bool)
        self.initial_state = None
        self.metadata = {}
        
        # Execution tracking
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        
        # History for visualization/debugging
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
    def reset(self, initial_state: Optional[np.ndarray] = None, metadata: Optional[Dict] = None) -> np.ndarray:
        """Reset the Karel world to initial state"""
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            self._generate_random_state()
            
        self.initial_state = self.state.copy()
        self.metadata = metadata or {}
        
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        
        self.state_history = [self.state.copy()]
        self.action_history = []
        self.reward_history = []
        
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one primitive action in the Karel world"""
        if self.done:
            return self.state.copy(), 0.0, True, {'timeout': True}
            
        self.step_count += 1
        
        # Execute action
        success = self._execute_action(Action(action))
        
        # Calculate reward
        reward = self._calculate_reward() if success else -0.1
        
        # Check termination conditions
        self.done = self._check_done() or (self.step_count >= self.timeout_steps)
        
        # Update history
        self.state_history.append(self.state.copy())
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.total_reward += reward
        
        info = {
            'success': success,
            'timeout': self.step_count >= self.timeout_steps,
            'step_count': self.step_count,
            'total_reward': self.total_reward
        }
        
        return self.state.copy(), reward, self.done, info
    
    def _execute_action(self, action: Action) -> bool:
        """Execute a primitive Karel action"""
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return False
            
        row, col, direction = agent_pos
        
        try:
            if action == Action.MOVE:
                return self._move_forward(row, col, direction)
            elif action == Action.TURN_LEFT:
                return self._turn_left(row, col, direction)
            elif action == Action.TURN_RIGHT:
                return self._turn_right(row, col, direction)
            elif action == Action.PICK_MARKER:
                return self._pick_marker(row, col)
            elif action == Action.PUT_MARKER:
                return self._put_marker(row, col)
            else:
                return False
        except Exception:
            return False
    
    def _move_forward(self, row: int, col: int, direction: Direction) -> bool:
        """Move Karel forward in the current direction"""
        # Calculate new position
        new_row, new_col = row, col
        if direction == Direction.NORTH:
            new_row = row - 1
        elif direction == Direction.EAST:
            new_col = col + 1
        elif direction == Direction.SOUTH:
            new_row = row + 1
        elif direction == Direction.WEST:
            new_col = col - 1
            
        # Check bounds and walls
        if (new_row < 0 or new_row >= self.h or 
            new_col < 0 or new_col >= self.w or
            self.state[new_row, new_col, 4]):  # Wall
            return False
            
        # Move agent
        self.state[row, col, :4] = False
        self.state[new_row, new_col, direction] = True
        return True
    
    def _turn_left(self, row: int, col: int, direction: Direction) -> bool:
        """Turn Karel left"""
        new_direction = Direction((direction - 1) % 4)
        self.state[row, col, :4] = False
        self.state[row, col, new_direction] = True
        return True
    
    def _turn_right(self, row: int, col: int, direction: Direction) -> bool:
        """Turn Karel right"""
        new_direction = Direction((direction + 1) % 4)
        self.state[row, col, :4] = False
        self.state[row, col, new_direction] = True
        return True
    
    def _pick_marker(self, row: int, col: int) -> bool:
        """Pick up a marker from current position"""
        if self.state[row, col, 5]:  # No markers
            return False
            
        # Update marker count
        if self.state[row, col, 6]:  # 1 marker -> 0 markers
            self.state[row, col, 6] = False
            self.state[row, col, 5] = True
        elif self.state[row, col, 7]:  # 2+ markers -> 1 marker
            self.state[row, col, 7] = False
            self.state[row, col, 6] = True
        else:
            return False
            
        return True
    
    def _put_marker(self, row: int, col: int) -> bool:
        """Put a marker at current position"""
        # Update marker count
        if self.state[row, col, 5]:  # 0 markers -> 1 marker
            self.state[row, col, 5] = False
            self.state[row, col, 6] = True
        elif self.state[row, col, 6]:  # 1 marker -> 2+ markers
            if self.max_markers > 1:
                self.state[row, col, 6] = False
                self.state[row, col, 7] = True
            else:
                return False  # Can't place more than 1 marker
        else:
            return False  # Already at max markers
            
        return True
    
    def _get_agent_position(self) -> Optional[Tuple[int, int, Direction]]:
        """Get agent's current position and direction"""
        agent_positions = np.where(self.state[:, :, :4])
        if len(agent_positions[0]) == 0:
            return None
        
        row, col, direction = agent_positions[0][0], agent_positions[1][0], agent_positions[2][0]
        return row, col, Direction(direction)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on the current task"""
        if self.task == 'harvester':
            return self._harvester_reward()
        elif self.task == 'cleanHouse':
            return self._clean_house_reward()
        elif self.task == 'fourCorners':
            return self._four_corners_reward()
        elif self.task == 'stairClimber':
            return self._stair_climber_reward()
        elif self.task == 'topOff':
            return self._top_off_reward()
        elif self.task == 'randomMaze':
            return self._random_maze_reward()
        else:
            return 0.0
    
    def _harvester_reward(self) -> float:
        """Reward for harvester task: collect all markers"""
        total_markers = np.sum(self.state[:, :, 6:])  # Count all markers
        max_possible = (self.h - 2) * (self.w - 2)  # Exclude walls
        
        if hasattr(self, '_prev_markers'):
            reward = self._prev_markers - total_markers  # Reward for collecting
        else:
            reward = 0.0
            
        self._prev_markers = total_markers
        return reward
    
    def _clean_house_reward(self) -> float:
        """Reward for cleanHouse task: clean specific marked positions"""
        if 'marker_positions' not in self.metadata:
            return 0.0
            
        cleaned = 0
        for pos in self.metadata['marker_positions']:
            if self.state[pos[0], pos[1], 5]:  # No marker (cleaned)
                cleaned += 1
                
        progress = cleaned / len(self.metadata['marker_positions'])
        
        if hasattr(self, '_prev_progress'):
            reward = progress - self._prev_progress
        else:
            reward = 0.0
            
        self._prev_progress = progress
        return reward
    
    def _four_corners_reward(self) -> float:
        """Reward for fourCorners task: place markers at corners"""
        corners = [(1, 1), (1, self.w-2), (self.h-2, 1), (self.h-2, self.w-2)]
        correct_markers = sum(1 for corner in corners if self.state[corner[0], corner[1], 6])
        
        # Penalize incorrect markers
        total_markers = np.sum(self.state[:, :, 6:])
        incorrect_markers = total_markers - correct_markers
        
        if incorrect_markers > 0:
            return -1.0  # Task failed
            
        progress = correct_markers / 4.0
        
        if hasattr(self, '_prev_corner_progress'):
            reward = progress - self._prev_corner_progress
        else:
            reward = 0.0
            
        self._prev_corner_progress = progress
        return reward
    
    def _stair_climber_reward(self) -> float:
        """Reward for stairClimber task: reach the marker"""
        if 'marker_positions' not in self.metadata:
            return 0.0
            
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return 0.0
            
        target_pos = self.metadata['marker_positions'][0]
        distance = abs(agent_pos[0] - target_pos[0]) + abs(agent_pos[1] - target_pos[1])
        
        if hasattr(self, '_prev_distance'):
            reward = self._prev_distance - distance  # Reward for getting closer
        else:
            reward = 0.0
            
        self._prev_distance = distance
        return reward * 0.1  # Scale reward
    
    def _top_off_reward(self) -> float:
        """Reward for topOff task: fill bottom row and reach end"""
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return 0.0
            
        bottom_row = self.h - 2
        score = 0
        
        # Check consecutive filled positions from left
        for col in range(1, agent_pos[1] + 1):
            if 'expected_marker_positions' in self.metadata:
                if (bottom_row, col) in self.metadata['expected_marker_positions']:
                    if not self.state[bottom_row, col, 5]:  # Has marker
                        score += 1
                    else:
                        break
                        
        # Bonus for reaching the end
        if agent_pos[0] == bottom_row and agent_pos[1] == self.w - 2:
            score += 1
            
        progress = score / (self.w - 1)
        
        if hasattr(self, '_prev_topoff_progress'):
            reward = progress - self._prev_topoff_progress
        else:
            reward = 0.0
            
        self._prev_topoff_progress = progress
        return reward
    
    def _random_maze_reward(self) -> float:
        """Reward for randomMaze task: reach the marker"""
        if 'marker_positions' not in self.metadata:
            return 0.0
            
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return 0.0
            
        target_pos = self.metadata['marker_positions'][0]
        if agent_pos[0] == target_pos[0] and agent_pos[1] == target_pos[1]:
            return 1.0  # Reached goal
        else:
            return 0.0
    
    def _check_done(self) -> bool:
        """Check if the task is completed"""
        if self.task == 'harvester':
            return np.sum(self.state[:, :, 6:]) == 0  # All markers collected
        elif self.task == 'cleanHouse':
            if 'marker_positions' not in self.metadata:
                return False
            return all(self.state[pos[0], pos[1], 5] for pos in self.metadata['marker_positions'])
        elif self.task == 'fourCorners':
            corners = [(1, 1), (1, self.w-2), (self.h-2, 1), (self.h-2, self.w-2)]
            correct = sum(1 for corner in corners if self.state[corner[0], corner[1], 6])
            total_markers = np.sum(self.state[:, :, 6:])
            return correct == 4 and total_markers == 4
        elif self.task == 'randomMaze':
            if 'marker_positions' not in self.metadata:
                return False
            agent_pos = self._get_agent_position()
            target_pos = self.metadata['marker_positions'][0]
            return (agent_pos is not None and 
                   agent_pos[0] == target_pos[0] and 
                   agent_pos[1] == target_pos[1])
        else:
            return False
    
    def _generate_random_state(self):
        """Generate a random initial state (placeholder implementation)"""
        # Reset state
        self.state.fill(False)
        
        # Add walls around perimeter
        self.state[0, :, 4] = True  # Top wall
        self.state[-1, :, 4] = True  # Bottom wall
        self.state[:, 0, 4] = True  # Left wall
        self.state[:, -1, 4] = True  # Right wall
        
        # Place agent randomly (not on walls)
        agent_row = np.random.randint(1, self.h - 1)
        agent_col = np.random.randint(1, self.w - 1)
        agent_dir = np.random.randint(0, 4)
        self.state[agent_row, agent_col, agent_dir] = True
        
        # Initialize all empty positions with "no markers"
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if not self.state[r, c, :4].any():  # Not agent position
                    self.state[r, c, 5] = True  # No markers
    
    # Perception functions for DSL
    def front_is_clear(self) -> bool:
        """Check if front is clear"""
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return False
            
        row, col, direction = agent_pos
        
        if direction == Direction.NORTH:
            new_row, new_col = row - 1, col
        elif direction == Direction.EAST:
            new_row, new_col = row, col + 1
        elif direction == Direction.SOUTH:
            new_row, new_col = row + 1, col
        else:  # WEST
            new_row, new_col = row, col - 1
            
        return (0 <= new_row < self.h and 0 <= new_col < self.w and 
                not self.state[new_row, new_col, 4])
    
    def left_is_clear(self) -> bool:
        """Check if left is clear"""
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return False
            
        row, col, direction = agent_pos
        left_direction = Direction((direction - 1) % 4)
        
        if left_direction == Direction.NORTH:
            new_row, new_col = row - 1, col
        elif left_direction == Direction.EAST:
            new_row, new_col = row, col + 1
        elif left_direction == Direction.SOUTH:
            new_row, new_col = row + 1, col
        else:  # WEST
            new_row, new_col = row, col - 1
            
        return (0 <= new_row < self.h and 0 <= new_col < self.w and 
                not self.state[new_row, new_col, 4])
    
    def right_is_clear(self) -> bool:
        """Check if right is clear"""
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return False
            
        row, col, direction = agent_pos
        right_direction = Direction((direction + 1) % 4)
        
        if right_direction == Direction.NORTH:
            new_row, new_col = row - 1, col
        elif right_direction == Direction.EAST:
            new_row, new_col = row, col + 1
        elif right_direction == Direction.SOUTH:
            new_row, new_col = row + 1, col
        else:  # WEST
            new_row, new_col = row, col - 1
            
        return (0 <= new_row < self.h and 0 <= new_col < self.w and 
                not self.state[new_row, new_col, 4])
    
    def marker_present(self) -> bool:
        """Check if marker is present at current position"""
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return False
            
        row, col, _ = agent_pos
        return self.state[row, col, 6] or self.state[row, col, 7]  # 1 or 2+ markers
    
    def no_marker_present(self) -> bool:
        """Check if no marker is present at current position"""
        return not self.marker_present()
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        return self.state.copy()
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the current state"""
        if mode == 'human':
            self._print_state()
            return None
        elif mode == 'rgb_array':
            return self._state_to_image()
        else:
            return self.state.copy()
    
    def _print_state(self):
        """Print ASCII representation of the state"""
        agent_symbols = {Direction.NORTH: '^', Direction.EAST: '>', 
                        Direction.SOUTH: 'v', Direction.WEST: '<'}
        
        for row in range(self.h):
            line = ""
            for col in range(self.w):
                if self.state[row, col, 4]:  # Wall
                    line += "█"
                elif self.state[row, col, :4].any():  # Agent
                    direction = np.argmax(self.state[row, col, :4])
                    line += agent_symbols[Direction(direction)]
                elif self.state[row, col, 6] or self.state[row, col, 7]:  # Marker
                    line += "●"
                else:  # Empty
                    line += "."
            print(line)
        print()
    
    def _state_to_image(self) -> np.ndarray:
        """Convert state to RGB image (placeholder)"""
        # This would return an RGB image representation
        # For now, return a simple grayscale version
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        for row in range(self.h):
            for col in range(self.w):
                if self.state[row, col, 4]:  # Wall
                    img[row, col] = [64, 64, 64]  # Gray
                elif self.state[row, col, :4].any():  # Agent
                    img[row, col] = [0, 255, 0]  # Green
                elif self.state[row, col, 6] or self.state[row, col, 7]:  # Marker
                    img[row, col] = [255, 0, 0]  # Red
                else:  # Empty
                    img[row, col] = [255, 255, 255]  # White
                    
        return img