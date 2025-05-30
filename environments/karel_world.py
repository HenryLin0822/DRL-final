"""
Enhanced Karel World Implementation

This module contains the Karel world mechanics with all original tasks and reward functions
while maintaining simplified architecture features.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from enum import IntEnum
from collections import deque
import copy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.state_generator import KarelStateGenerator
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
    Enhanced Karel World implementation with all original tasks and reward functions.
    
    State representation: (H, W, 8) boolean array
    - Channels 0-3: Agent direction (one-hot)
    - Channel 4: Walls
    - Channel 5: No markers (0 markers)
    - Channel 6: 1 marker
    - Channel 7: 2+ markers
    """
    
    def __init__(
        self, 
        task: str = 'harvester',
        grid_size = (8, 8),
        timeout_steps: int = 500,
        reward_diff: bool = False,
        final_reward_scale: bool = True
    ):
        self.task = task
        self.h, self.w = grid_size
        self.timeout_steps = timeout_steps
        self.reward_diff = reward_diff
        self.final_reward_scale = final_reward_scale
        self.state_generator = KarelStateGenerator(task=self.task, grid_size = (8,8))
        # Maximum markers per cell based on task
        if task in ['topOff', 'topOff_sparse', 'randomMaze_key2door', 'randomMaze_key2door_sparse', 
                   'randomMaze_key2doorSpace', 'randomMaze_key2doorSpace_sparse', 
                   'doorkey', 'doorkey_sparse', 'seeder', 'seeder_sparse']:
            self.max_markers = 2
        else:
            self.max_markers = 1
        
        # State representation
        self.state, self.metadata = self.state_generator.generate_state(task_specific=True)
        
        self.initial_state = self.state.copy()
        #print("initial state:")
        #print(self.initial_state[:,:,6])
        
        # Execution tracking
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        
        # Progress tracking for rewards
        self.progress_ratio = 0.0
        self.prev_pos_reward = 0.0
        self.init_pos_reward = 0.0
        
        # Snake-specific tracking
        self.snake_body = deque([(1, 1), (1, 2)])
        self.snake_len = 2
        self.snake_marker_pointer = 0
        
        # Position tracking for oneStroke
        self.pos_h = []
        self.pos_h_set = set()
        
        # History for visualization/debugging
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
    def reset(self, initial_state: Optional[np.ndarray] = None, metadata: Optional[Dict] = None) -> np.ndarray:
        """Reset the Karel world to initial state"""
        if self.initial_state is not None:
            self.state = self.initial_state.copy()
        else:
            self._generate_random_state()
        #print("in reset:", self.state[:,:,6])    
        self.initial_state = self.state.copy()
        
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        
        # Reset progress tracking
        self.progress_ratio = 0.0
        self.prev_pos_reward = 0.0
        self.init_pos_reward = 0.0
        
        # Reset snake tracking
        self.snake_body = deque([(1, 1), (1, 2)])
        self.snake_len = 2
        if 'marker_pointer' in self.metadata:
            self.snake_marker_pointer = self.metadata['marker_pointer']
        
        # Reset position tracking
        agent_pos = self._get_agent_position()
        if agent_pos:
            self.pos_h = [(agent_pos[0], agent_pos[1])]
            self.pos_h_set = {(agent_pos[0], agent_pos[1])}
        else:
            self.pos_h = []
            self.pos_h_set = set()
        
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
        #print("before step:")
        #print("one marker:", self.state[:,:,6])
        success = self._execute_action(Action(action))
        
        # Calculate reward
        reward = self._calculate_reward() if success else 0.0 # from -0.1 to 0
        
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
        #if(reward<0):
        #    print("xxx")
        return self.state.copy(), reward, self.done, info
    
    def _execute_action(self, action: Action) -> bool:
        """Execute a primitive Karel action"""
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return False
            
        row, col, direction = agent_pos
        print(action)
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
            print("move failed, wall or out of bounds")
            return False
        
        # Special handling for snake body collision
        if self.task in ['snake', 'snake_sparse'] and self.state[new_row, new_col, 7]:
            self.done = True
            return False
            
        # Move agent
        self.state[row, col, :4] = False
        self.state[new_row, new_col, direction] = True
        
        # Update position tracking for oneStroke
        if self.task in ['oneStroke', 'oneStroke_sparse']:
            self.pos_h.append((new_row, new_col))
            self.pos_h_set.add((new_row, new_col))
            # Mark passed cell as wall
            self.state[row, col, 7] = False
            self.state[row, col, 6] = False
            self.state[row, col, 5] = False
            self.state[row, col, 4] = True  # Wall
        
        # Snake-specific movement handling
        if self.task in ['snake', 'snake_sparse'] and not self.done:
            self._handle_snake_movement(row, col, new_row, new_col)
        #print("True")
        return True
    
    def _handle_snake_movement(self, old_row: int, old_col: int, new_row: int, new_col: int):
        """Handle snake-specific movement mechanics"""
        # Add old position to snake body
        if (new_row, new_col) not in self.snake_body:
            self.snake_body.append((old_row, old_col))
        else:
            self.done = True
            return
            
        # Check if marker eaten
        if self.state[new_row, new_col, 6] and self.snake_len < 22:
            self.snake_len += 1
            self.state[new_row, new_col, 6] = False
            
            # Generate new marker
            if 'marker_list' in self.metadata:
                attempts = 0
                while attempts < 50:
                    marker_pos = self.metadata['marker_list'][self.snake_marker_pointer]
                    self.snake_marker_pointer = (self.snake_marker_pointer + 1) % len(self.metadata['marker_list'])
                    if np.sum(self.state[marker_pos[0], marker_pos[1], :]) == 0:
                        self.state[marker_pos[0], marker_pos[1], 6] = True
                        break
                    attempts += 1
        
        # Mark old position with double marker (snake body)
        self.state[old_row, old_col, 7] = True
        self.state[old_row, old_col, 6] = False
        self.state[old_row, old_col, 5] = False
        
        # Remove tail if snake too long
        if len(self.snake_body) > self.snake_len:
            tail_pos = self.snake_body.popleft()
            self.state[tail_pos[0], tail_pos[1], :] = False
            self.state[tail_pos[0], tail_pos[1], 5] = True  # Empty
    
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
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return 0.0
        #print("get reward")    
        if self.task == 'harvester' or self.task == 'harvester_sparse':
            return self._harvester_reward(agent_pos)
        elif self.task == 'cleanHouse' or self.task == 'cleanHouse_sparse':
            return self._clean_house_reward(agent_pos)
        elif self.task == 'fourCorners' or self.task == 'fourCorners_sparse':
            return self._four_corners_reward(agent_pos)
        elif self.task == 'randomMaze' or self.task == 'randomMaze_sparse':
            return self._random_maze_reward(agent_pos)
        elif self.task == 'stairClimber' or self.task == 'stairClimber_sparse':
            return self._stair_climber_reward(agent_pos)
        elif self.task == 'topOff' or self.task == 'topOff_sparse':
            return self._top_off_reward(agent_pos)
        elif self.task == 'randomMaze_key2door' or self.task == 'randomMaze_key2door_sparse':
            return self._key2door_reward(agent_pos)
        elif self.task == 'randomMaze_key2doorSpace' or self.task == 'randomMaze_key2doorSpace_sparse':
            return self._key2door_space_reward(agent_pos)
        elif self.task == 'oneStroke' or self.task == 'oneStroke_sparse':
            return self._one_stroke_reward(agent_pos)
        elif self.task == 'doorkey' or self.task == 'doorkey_sparse':
            return self._doorkey_reward(agent_pos)
        elif self.task == 'seeder' or self.task == 'seeder_sparse':
            return self._seeder_reward(agent_pos)
        elif self.task == 'snake' or self.task == 'snake_sparse':
            return self._snake_reward(agent_pos)
        else:
            return 0.0
    
    def _harvester_reward(self, agent_pos) -> float:
        """Reward for harvester task: collect all markers"""
        if self.done:
            return 0.0
            
        total_markers = np.sum(self.state[:, :, 6:])
        max_markers = (self.h - 2) * (self.w - 2)
        print("total_markers:", total_markers, "max_markers:", max_markers)
        current_progress_ratio = (max_markers - total_markers) / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        
        done = total_markers == 0
        reward = reward if self.task == 'harvester' else float(done)
        self.done = self.done or done
        return reward
    
    def _clean_house_reward(self, agent_pos) -> float:
        """Reward for cleanHouse task: clean positions that had markers initially"""
        if self.done:
            return 0.0
        
        # Compare current state with initial state to find which positions had markers
        initial_markers = self.initial_state[:, :, 6] | self.initial_state[:, :, 7]
        current_markers = self.state[:, :, 6] | self.state[:, :, 7]
        print("sum:",np.sum(initial_markers))
        # Count positions that had markers initially but are now empty
        cleaned_positions = initial_markers & (~current_markers)
        total_initial_markers = np.sum(initial_markers)
        
        if total_initial_markers == 0:
            print("No initial markers found, returning 0 reward")
            return 0.0
            
        current_progress_ratio = np.sum(cleaned_positions) / float(total_initial_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        
        done = np.sum(cleaned_positions) == total_initial_markers
        reward = reward if self.task == 'cleanHouse' else float(done)
        self.done = self.done or done
        return reward
    
    def _four_corners_reward(self, agent_pos) -> float:
        """Reward for fourCorners task: place markers at corners"""
        if self.done:
            return 0.0
        print("metadata",self.metadata)
        correct_markers = 0
        if self.state[1, 1, 6]:
            correct_markers += 1
        if self.state[self.h-2, 1, 6]:
            correct_markers += 1
        if self.state[self.h-2, self.w-2, 6]:
            correct_markers += 1
        if self.state[1, self.w-2, 6]:
            correct_markers += 1
        print("correct_markers:", correct_markers)
        #total_markers = np.sum(self.state[:, :, 6:])
        total_markers = 4
        print("total_markers:", total_markers)
        incorrect_markers = total_markers - correct_markers
        
        current_progress_ratio = correct_markers / 4.0
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        
        if incorrect_markers > 0 and reward > 0.0:
            reward = 0.0
            
        done = correct_markers == 4 or incorrect_markers > 0
        
        if self.task == 'fourCorners_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward
    
    def _random_maze_reward(self, agent_pos) -> float:
        """Reward for randomMaze task: reach the marker"""
        print("metadata",self.metadata)
        print("self.done",self.done)
        if self.done:
            return 0.0
            
        if 'marker_positions' not in self.metadata or len(self.metadata['marker_positions']) == 0:
            #print("metadata",self.metadata)
            return 0.0
            
        # Find initial marker position from initial state
        x, y = np.where(self.initial_state[:, :, 6] > 0)
        print("x, y", x, y)
        if len(x) != 1:
            return 0.0
            
        marker_pos = np.array([x[0], y[0]])
        distance_to_goal = -1 * (abs(agent_pos[0] - marker_pos[0]) + abs(agent_pos[1] - marker_pos[1]))
        print("distance_to_goal", distance_to_goal)
        done = distance_to_goal == 0
        reward = float(done)
        self.done = self.done or done
        #print(reward)
        return reward
    
    def _stair_climber_reward(self, agent_pos) -> float:
        if self.done:
            return 0.0                     # episode already finished
        #print("agent_pos", agent_pos)
        # where is the goal?
        gx, gy = np.where(self.initial_state[:, :, 6]>0)
        #print("gx, gy", gx, gy)
        reached = (agent_pos[0] == gx[0]) and (agent_pos[1] == gy[0])

        if reached:
            self.done = True               # terminate the episode
        return 1.0 if reached else 0.0
    
    def _top_off_reward(self, agent_pos) -> float:
        """Reward for topOff task: fill bottom row and reach end"""
        if self.done:
            return 0.0
            
        score = 0
        
        for c in range(1, agent_pos[1] + 1):
            if 'not_expected_marker_positions' in self.metadata:
                if (self.h-2, c) in self.metadata['not_expected_marker_positions']:
                    if self.state[self.h-2, c, 7]:
                        score += 1
                    else:
                        break
            elif 'expected_marker_positions' in self.metadata:
                if (self.h-2, c) in self.metadata['expected_marker_positions']:
                    if self.state[self.h-2, c, 5]:
                        score += 1
                    else:
                        break
                        
        if (self.w - 2 == agent_pos[1] and self.h - 2 == agent_pos[0]) and score == self.w - 2:
            score += 1
            
        current_progress_ratio = score / (self.w - 1)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        
        done = False
        if 'not_expected_marker_positions' in self.metadata:
            expected_filled = sum([self.state[pos[0], pos[1], 7] for pos in self.metadata['not_expected_marker_positions']])
            done = (expected_filled == len(self.metadata['not_expected_marker_positions']) and 
                   (self.w - 2 == agent_pos[1] and self.h - 2 == agent_pos[0]) and 
                   current_progress_ratio == 1.0)
        
        reward = reward if self.task == 'topOff' else float(done)
        if self.task == 'topOff_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward
    
    def _key2door_reward(self, agent_pos) -> float:
        """Reward for randomMaze_key2door task"""
        if self.done:
            return 0.0
            
        total_markers = np.sum(self.state[:, :, 6:])
        error_markers = total_markers - 2
        score = 0
        
        if self.state[6, 3, 5]:  # Key picked
            score += 0.5
        if self.state[6, 3, 5] and self.state[1, 6, 7]:  # Key picked and door marked
            score += 0.5
            
        if error_markers > 0:
            score -= error_markers * 0.0001
            
        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        
        done = (current_progress_ratio == 1.0)
        reward = reward if self.task == 'randomMaze_key2door' else float(done)
        if self.task == 'randomMaze_key2door_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward
    
    def _key2door_space_reward(self, agent_pos) -> float:
        """Reward for randomMaze_key2doorSpace task"""
        if self.done:
            return 0.0
            
        total_markers = np.sum(self.state[:, :, 6:])
        error_markers = total_markers - 2
        score = 0
        
        if self.state[6, 3, 5]:  # Key picked
            score += 0.5
        if self.state[6, 3, 5] and self.state[1, 6, 7]:  # Key picked and door marked
            score += 0.5
            
        if error_markers > 0:
            score -= error_markers * 0.0001
            
        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        
        done = (current_progress_ratio == 1.0)
        reward = reward if self.task == 'randomMaze_key2doorSpace' else float(done)
        if self.task == 'randomMaze_key2doorSpace_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward
    
    def _one_stroke_reward(self, agent_pos) -> float:
        """Reward for oneStroke task: traverse all cells exactly once"""
        if self.done:
            return 0.0
            
        pos_tuple = tuple(agent_pos[:2])
        
        is_overlap = pos_tuple in self.pos_h_set and pos_tuple != self.pos_h[-1] if self.pos_h else False
        is_hit_wall = (len(self.action_history) > 0 and self.action_history[-1] == 0 and 
                      pos_tuple == self.pos_h[-1] if self.pos_h else False)
        traverse_length = len(self.pos_h_set)
        
        max_markers = (self.w - 2) * (self.h - 2)
        
        current_progress_ratio = traverse_length / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        
        done = is_overlap or is_hit_wall or traverse_length == max_markers
        reward = reward if self.task == 'oneStroke' else float(done)
        self.done = self.done or done
        return reward
    
    def _doorkey_reward(self, agent_pos) -> float:
        """Reward for doorkey task: generic key-door puzzle"""
        if self.done:
            return 0.0
            
        if 'key' not in self.metadata or 'target' not in self.metadata:
            return 0.0
            
        total_markers = np.sum(self.state[:, :, 6:])
        error_markers = total_markers - 2
        score = 0
        
        key_pos = self.metadata['key']
        target_pos = self.metadata['target']
        
        if self.state[key_pos[0], key_pos[1], 5]:  # Key picked
            score += 0.5
        if self.state[key_pos[0], key_pos[1], 5] and self.state[target_pos[0], target_pos[1], 7]:  # Key picked and target marked
            score += 0.5
            
        # Open door if key picked
        if self.state[key_pos[0], key_pos[1], 5] and 'door_positions' in self.metadata:
            for door_pos in self.metadata['door_positions']:
                self.state[door_pos[0], door_pos[1], 4] = False
                
        if error_markers > 0:
            score -= error_markers * 0.0001
            
        current_progress_ratio = score
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        
        done = (current_progress_ratio == 1.0)
        reward = reward if self.task == 'doorkey' else float(done)
        if self.task == 'doorkey_sparse':
            reward = reward if done and not self.done else 0
        self.done = self.done or done
        return reward
    
    def _seeder_reward(self, agent_pos) -> float:
        """Reward for seeder task: place exactly one marker in each valid cell"""
        if self.done:
            return 0.0
            
        existing_marker_num = len(self.metadata.get('existing_marker', []))
        max_markers = (self.w - 2) * (self.h - 2) - existing_marker_num
        
        total_one_markers = np.sum(self.state[:, :, 6])
        total_two_markers = np.sum(self.state[:, :, 7])
        
        score = total_one_markers - existing_marker_num
        
        current_progress_ratio = score / float(max_markers)
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        
        done = (total_one_markers == max_markers and total_two_markers == 0) or total_two_markers > 0
        reward = reward if self.task == 'seeder' else float(done)
        self.done = self.done or done
        return reward
    
    def _snake_reward(self, agent_pos) -> float:
        """Reward for snake task: eat markers to grow while avoiding body collision"""
        if self.done:
            return 0.0
            
        is_hit_body = self.state[agent_pos[0], agent_pos[1], 7]
        
        current_progress_ratio = (self.snake_len - 2) / 20.0
        reward = current_progress_ratio - self.progress_ratio
        self.progress_ratio = current_progress_ratio
        
        done = is_hit_body or current_progress_ratio >= 0.99
        reward = reward if self.task == 'snake' else float(done)
        self.done = self.done or done
        return reward
    
    def _check_done(self) -> bool:
        """Check if the task is completed"""
        if self.task == 'harvester' or self.task == 'harvester_sparse':
            #print("one marker:", self.state[:,:,6])
            return np.sum(self.state[:, :, 6:]) == 0
        elif self.task == 'cleanHouse' or self.task == 'cleanHouse_sparse':
            if 'marker_positions' not in self.metadata:
                return False
            return all(self.state[pos[0], pos[1], 5] for pos in self.metadata['marker_positions'])
        elif self.task == 'fourCorners' or self.task == 'fourCorners_sparse':
            corners = [(1, 1), (1, self.w-2), (self.h-2, 1), (self.h-2, self.w-2)]
            correct = sum(1 for corner in corners if self.state[corner[0], corner[1], 6])
            total_markers = np.sum(self.state[:, :, 6:])
            return correct == 4 or total_markers > 4
        elif self.task == 'randomMaze' or self.task == 'randomMaze_sparse':
            # Find initial marker position
            x, y = np.where(self.initial_state[:, :, 6] > 0)
            if len(x) != 1:
                return False
            marker_pos = (x[0], y[0])
            agent_pos = self._get_agent_position()
            return (agent_pos is not None and 
                   agent_pos[0] == marker_pos[0] and 
                   agent_pos[1] == marker_pos[1])
        elif self.task == 'stairClimber' or self.task == 'stairClimber_sparse':
            # Find initial marker position
            x, y = np.where(self.initial_state[:, :, 6] > 0)
            if len(x) != 1:
                return False
            marker_pos = (x[0], y[0])
            agent_pos = self._get_agent_position()
            return (agent_pos is not None and 
                   agent_pos[0] == marker_pos[0] and 
                   agent_pos[1] == marker_pos[1])
        elif self.task == 'topOff' or self.task == 'topOff_sparse':
            if 'not_expected_marker_positions' in self.metadata:
                expected_filled = sum([self.state[pos[0], pos[1], 7] for pos in self.metadata['not_expected_marker_positions']])
                agent_pos = self._get_agent_position()
                return (expected_filled == len(self.metadata['not_expected_marker_positions']) and 
                       agent_pos is not None and
                       self.w - 2 == agent_pos[1] and self.h - 2 == agent_pos[0])
            return False
        elif self.task == 'randomMaze_key2door' or self.task == 'randomMaze_key2door_sparse':
            return self.state[6, 3, 5] and self.state[1, 6, 7]
        elif self.task == 'randomMaze_key2doorSpace' or self.task == 'randomMaze_key2doorSpace_sparse':
            return self.state[6, 3, 5] and self.state[1, 6, 7]
        elif self.task == 'oneStroke' or self.task == 'oneStroke_sparse':
            pos_tuple = tuple(self._get_agent_position()[:2]) if self._get_agent_position() else None
            if pos_tuple is None:
                return False
            is_overlap = pos_tuple in self.pos_h_set and pos_tuple != self.pos_h[-1] if self.pos_h else False
            is_hit_wall = (len(self.action_history) > 0 and self.action_history[-1] == 0 and 
                          pos_tuple == self.pos_h[-1] if self.pos_h else False)
            traverse_length = len(self.pos_h_set)
            max_markers = (self.w - 2) * (self.h - 2)
            return is_overlap or is_hit_wall or traverse_length == max_markers
        elif self.task == 'doorkey' or self.task == 'doorkey_sparse':
            if 'key' not in self.metadata or 'target' not in self.metadata:
                return False
            key_pos = self.metadata['key']
            target_pos = self.metadata['target']
            return self.state[key_pos[0], key_pos[1], 5] and self.state[target_pos[0], target_pos[1], 7]
        elif self.task == 'seeder' or self.task == 'seeder_sparse':
            existing_marker_num = len(self.metadata.get('existing_marker', []))
            max_markers = (self.w - 2) * (self.h - 2) - existing_marker_num
            total_one_markers = np.sum(self.state[:, :, 6])
            total_two_markers = np.sum(self.state[:, :, 7])
            return (total_one_markers == max_markers and total_two_markers == 0) or total_two_markers > 0
        elif self.task == 'snake' or self.task == 'snake_sparse':
            agent_pos = self._get_agent_position()
            if agent_pos is None:
                return False
            is_hit_body = self.state[agent_pos[0], agent_pos[1], 7]
            current_progress_ratio = (self.snake_len - 2) / 20.0
            return is_hit_body or current_progress_ratio >= 0.99
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
                elif self.state[row, col, 6]:  # 1 marker
                    line += "●"
                elif self.state[row, col, 7]:  # 2+ markers
                    line += "◉"
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
                elif self.state[row, col, 6]:  # 1 marker
                    img[row, col] = [255, 0, 0]  # Red
                elif self.state[row, col, 7]:  # 2+ markers
                    img[row, col] = [255, 165, 0]  # Orange
                else:  # Empty
                    img[row, col] = [255, 255, 255]  # White
                    
        return img


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    world = KarelWorld(task='harvester', grid_size=(6, 6))
    
    # Create a simple test state
    test_state = np.zeros((6, 6, 8), dtype=bool)
    
    # Add walls around perimeter
    test_state[0, :, 4] = True
    test_state[-1, :, 4] = True
    test_state[:, 0, 4] = True
    test_state[:, -1, 4] = True
    
    # Place agent at (1, 1) facing north
    test_state[1, 1, 0] = True
    
    # Place some markers
    test_state[2, 2, 6] = True  # 1 marker
    test_state[3, 3, 6] = True  # 1 marker
    
    # Initialize empty cells
    for r in range(1, 5):
        for c in range(1, 5):
            if not test_state[r, c, :].any():
                test_state[r, c, 5] = True
    
    # Reset world with test state
    state = world.reset(test_state)
    print("Initial state:")
    world.render()
    
    # Test some actions
    actions = [Action.MOVE, Action.MOVE, Action.PICK_MARKER, Action.TURN_RIGHT, Action.MOVE]
    
    for i, action in enumerate(actions):
        state, reward, done, info = world.step(action.value)
        print(f"Step {i+1}: Action={action.name}, Reward={reward:.3f}, Done={done}")
        world.render()
        
        if done:
            break
    
    print(f"Total reward: {world.total_reward:.3f}")
