"""
Karel State Generator

This module generates initial states for various Karel tasks.
"""
from enum import IntEnum
import numpy as np
from typing import Tuple, Dict, List, Optional, Any

class Direction(IntEnum):
    """Karel's facing directions"""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class KarelStateGenerator:
    """
    Generates initial states for Karel tasks
    """
    
    def __init__(
        self,
        grid_size = (8, 8),
        task: str = 'harvester',
        seed: Optional[int] = None
    ):
        self.h, self.w = grid_size
        self.task = task
        self.rng = np.random.RandomState(seed)
        # Store metadata separately to avoid returning tuples
        self.last_metadata = {}
    
    def generate_state(self, task_specific: bool = True) -> np.ndarray:
        """
        Generate an initial state for the specified task
        
        Args:
            task_specific: Whether to generate task-specific states
            
        Returns:
            state: Initial Karel state as numpy array
            
        Note: Metadata is stored in self.last_metadata
        """
        if task_specific:
            if self.task == 'harvester':
                return self._generate_harvester_state()
            elif self.task == 'cleanHouse':
                return self._generate_clean_house_state()
            elif self.task == 'fourCorners':
                return self._generate_four_corners_state()
            elif self.task == 'stairClimber':
                return self._generate_stair_climber_state()
            elif self.task == 'topOff':
                return self._generate_top_off_state()
            elif self.task == 'randomMaze':
                return self._generate_random_maze_state()
            else:
                return self._generate_random_state()
        else:
            return self._generate_random_state()
    
    def get_last_metadata(self) -> Dict[str, Any]:
        """Get metadata from the last state generation"""
        return self.last_metadata
    
    def _generate_random_state(self) -> np.ndarray:
        """Generate a random Karel state"""
        state = np.zeros((self.h, self.w, 8), dtype=bool)
        
        # Add walls around perimeter
        state[0, :, 4] = True  # Top wall
        state[-1, :, 4] = True  # Bottom wall
        state[:, 0, 4] = True  # Left wall
        state[:, -1, 4] = True  # Right wall
        
        # Place agent randomly
        agent_row = self.rng.randint(1, self.h - 1)
        agent_col = self.rng.randint(1, self.w - 1)
        agent_dir = self.rng.randint(0, 4)
        state[agent_row, agent_col, agent_dir] = True
        
        # Initialize all positions with "no markers"
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if not state[r, c, :4].any():  # Not agent position
                    state[r, c, 5] = True  # No markers
        
        self.last_metadata = {}
        return state, self.last_metadata
    
    def _generate_harvester_state(self) -> np.ndarray:
        """Generate initial state for harvester task"""
        state = np.zeros((self.h, self.w, 8), dtype=bool)
        
        # Add walls around perimeter
        state[0, :, 4] = True
        state[-1, :, 4] = True
        state[:, 0, 4] = True
        state[:, -1, 4] = True
        
        # Place agent randomly
        agent_row = self.rng.randint(1, self.h - 1)
        agent_col = self.rng.randint(1, self.w - 1)
        agent_dir = self.rng.randint(0, 4)
        state[agent_row, agent_col, agent_dir] = True
        
        # Fill entire grid with markers (except agent position and walls)
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if not state[r, c, :4].any():  # Not agent position
                    state[r, c, 6] = True  # 1 marker
                else:
                    state[r, c, 5] = True  # No markers at agent position
        
        self.last_metadata = {'total_markers': (self.h - 2) * (self.w - 2) - 1}
        return state, self.last_metadata
    
    def _generate_clean_house_state(self) -> np.ndarray:
        """Generate initial state for cleanHouse task"""
        state = np.zeros((self.h, self.w, 8), dtype=bool)
        
        # Add walls around perimeter
        state[0, :, 4] = True
        state[-1, :, 4] = True
        state[:, 0, 4] = True
        state[:, -1, 4] = True
        
        # Place agent at a fixed position
        agent_row, agent_col = 1, 1
        agent_dir = Direction.EAST
        state[agent_row, agent_col, agent_dir] = True
        
        # Generate random marker positions (3-5 markers)
        num_markers = self.rng.randint(3, 6)
        marker_positions = []
        
        available_positions = []
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if (r, c) != (agent_row, agent_col):  # Not agent position
                    available_positions.append((r, c))
        
        selected_positions = self.rng.choice(
            len(available_positions), 
            size=min(num_markers, len(available_positions)),
            replace=False
        )
        
        # Place markers and initialize empty positions
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if not state[r, c, :4].any():  # Not agent position
                    if (r, c) in [available_positions[i] for i in selected_positions]:
                        state[r, c, 6] = True  # 1 marker
                        marker_positions.append((r, c))
                    else:
                        state[r, c, 5] = True  # No markers
                else:
                    state[r, c, 5] = True  # No markers at agent position
        
        self.last_metadata = {'marker_positions': marker_positions}
        return state, self.last_metadata
    
    def _generate_four_corners_state(self) -> np.ndarray:
        """Generate initial state for fourCorners task"""
        state = np.zeros((self.h, self.w, 8), dtype=bool)
        
        # Add walls around perimeter
        state[0, :, 4] = True
        state[-1, :, 4] = True
        state[:, 0, 4] = True
        state[:, -1, 4] = True
        
        # Place agent randomly (not at corners)
        while True:
            agent_row = self.rng.randint(2, self.h - 2)
            agent_col = self.rng.randint(2, self.w - 2)
            # Avoid corners
            corners = [(1, 1), (1, self.w-2), (self.h-2, 1), (self.h-2, self.w-2)]
            if (agent_row, agent_col) not in corners:
                break
        
        agent_dir = self.rng.randint(0, 4)
        state[agent_row, agent_col, agent_dir] = True
        
        # Initialize all positions with "no markers"
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                state[r, c, 5] = True  # No markers
        
        # Agent position should also have no markers
        state[agent_row, agent_col, 5] = True
        
        self.last_metadata = {'corners': [(1, 1), (1, self.w-2), (self.h-2, 1), (self.h-2, self.w-2)]}
        return state, self.last_metadata
    
    def _generate_stair_climber_state(self) -> np.ndarray:
        """Generate initial state for stairClimber task"""
        state = np.zeros((self.h, self.w, 8), dtype=bool)
        
        # Add walls around perimeter
        state[0, :, 4] = True
        state[-1, :, 4] = True
        state[:, 0, 4] = True
        state[:, -1, 4] = True
        
        # Create stairs pattern
        for r in range(1, self.h - 1):
            for c in range(1, min(r, self.w - 1)):
                if c < self.w - 1:
                    state[r, c, 4] = True  # Wall for stairs
        
        # Place agent at bottom left
        agent_row, agent_col = self.h - 2, 1
        # Find valid position for agent
        while state[agent_row, agent_col, 4]:  # If there's a wall
            agent_col += 1
            if agent_col >= self.w - 1:
                agent_row -= 1
                agent_col = 1
        
        agent_dir = Direction.NORTH
        state[agent_row, agent_col, agent_dir] = True
        
        # Place marker at top of stairs
        marker_row, marker_col = 1, self.h - 2
        if marker_col >= self.w - 1:
            marker_col = self.w - 2
        
        # Initialize all non-wall positions with "no markers"
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if not state[r, c, 4] and not state[r, c, :4].any():  # Not wall, not agent
                    if (r, c) == (marker_row, marker_col):
                        state[r, c, 6] = True  # Marker at goal
                    else:
                        state[r, c, 5] = True  # No markers
        
        # Agent position
        state[agent_row, agent_col, 5] = True
        
        # Valid positions for agent (not on walls)
        valid_positions = []
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if not state[r, c, 4]:  # Not a wall
                    valid_positions.append((r, c))
        
        self.last_metadata = {
            'marker_positions': [(marker_row, marker_col)],
            'agent_valid_positions': valid_positions
        }
        return state, self.last_metadata
    
    def _generate_top_off_state(self) -> np.ndarray:
        """Generate initial state for topOff task"""
        state = np.zeros((self.h, self.w, 8), dtype=bool)
        
        # Add walls around perimeter
        state[0, :, 4] = True
        state[-1, :, 4] = True
        state[:, 0, 4] = True
        state[:, -1, 4] = True
        
        # Place agent at bottom left
        agent_row, agent_col = self.h - 2, 1
        agent_dir = Direction.EAST
        state[agent_row, agent_col, agent_dir] = True
        
        # Generate pattern for bottom row
        bottom_row = self.h - 2
        expected_positions = []
        not_expected_positions = []
        
        for c in range(1, self.w - 1):
            if c != agent_col:  # Not agent position
                if self.rng.random() < 0.5:  # 50% chance
                    expected_positions.append((bottom_row, c))
                else:
                    not_expected_positions.append((bottom_row, c))
        
        # Initialize all positions
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if (r, c) in not_expected_positions:
                    state[r, c, 6] = True  # Already has marker
                else:
                    state[r, c, 5] = True  # No markers
        
        self.last_metadata = {
            'expected_marker_positions': expected_positions,
            'not_expected_marker_positions': not_expected_positions
        }
        return state, self.last_metadata
    
    def _generate_random_maze_state(self) -> np.ndarray:
        """Generate initial state for randomMaze task"""
        state = np.zeros((self.h, self.w, 8), dtype=bool)
        
        # Add walls around perimeter
        state[0, :, 4] = True
        state[-1, :, 4] = True
        state[:, 0, 4] = True
        state[:, -1, 4] = True
        
        # Add some random internal walls to create maze-like structure
        for r in range(2, self.h - 2):
            for c in range(2, self.w - 2):
                if self.rng.random() < 0.2:  # 20% chance of wall
                    state[r, c, 4] = True
        
        # Place agent randomly
        while True:
            agent_row = self.rng.randint(1, self.h - 1)
            agent_col = self.rng.randint(1, self.w - 1)
            if not state[agent_row, agent_col, 4]:  # Not on wall
                break
        
        agent_dir = self.rng.randint(0, 4)
        state[agent_row, agent_col, agent_dir] = True
        
        # Place marker at random position (not agent position)
        while True:
            marker_row = self.rng.randint(1, self.h - 1)
            marker_col = self.rng.randint(1, self.w - 1)
            if (not state[marker_row, marker_col, 4] and  # Not on wall
                (marker_row, marker_col) != (agent_row, agent_col)):  # Not agent pos
                break
        
        # Initialize all non-wall positions
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if not state[r, c, 4] and not state[r, c, :4].any():  # Not wall, not agent
                    if (r, c) == (marker_row, marker_col):
                        state[r, c, 6] = True  # Goal marker
                    else:
                        state[r, c, 5] = True  # No markers
        
        # Agent position
        state[agent_row, agent_col, 5] = True
        
        self.last_metadata = {'marker_positions': [(marker_row, marker_col)]}
        return state, self.last_metadata
    
    def generate_program_instruction_state(
        self, 
        height: int, 
        width: int, 
        wall_prob: float = 0.1, 
        idx: int = 0
    ) -> np.ndarray:
        """
        Generate a fixed state for program instruction (for consistent CNN input)
        
        This creates simple, predictable states that can be used as fixed input
        during HPRL training to ensure consistent observation space.
        """
        state = np.zeros((height, width, 8), dtype=bool)
        
        # Add walls around perimeter
        state[0, :, 4] = True
        state[-1, :, 4] = True
        state[:, 0, 4] = True
        state[:, -1, 4] = True
        
        # Use idx to create different but deterministic patterns
        rng_local = np.random.RandomState(idx * 42)  # Fixed seed based on idx
        
        # Place agent at fixed position based on idx
        agent_row = 1 + (idx % (height - 2))
        agent_col = 1 + ((idx // (height - 2)) % (width - 2))
        agent_dir = idx % 4
        state[agent_row, agent_col, agent_dir] = True
        
        # Add some walls based on idx
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if (r, c) != (agent_row, agent_col):
                    if rng_local.random() < wall_prob:
                        state[r, c, 4] = True
                    else:
                        state[r, c, 5] = True  # No markers
        
        # Agent position has no markers
        state[agent_row, agent_col, 5] = True
        
        # Store metadata for this function too
        self.last_metadata = {
            'agent_position': (agent_row, agent_col, agent_dir),
            'wall_probability': wall_prob,
            'idx': idx
        }
        
        return state, self.last_metadata