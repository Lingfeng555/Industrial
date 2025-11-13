import numpy as np
from typing import List

class TrajectoryHistory:    
    def __init__(self, max_points: int = 20):
        self.max_points = max_points
        self.world_positions: List[np.ndarray] = []
    
    def add_pose(self, pose):
        self.world_positions.append(pose.translation.copy())
        
        if len(self.world_positions) > self.max_points:
            self.world_positions = self.world_positions[-self.max_points:]
    
    def clear(self):
        self.world_positions.clear()
    
    def get_world_positions(self) -> np.ndarray:
        return np.array(self.world_positions)
    
    def get_recent(self, n: int) -> np.ndarray:
        return np.array(self.world_positions)[-n:]
    
    def __len__(self) -> int:
        return len(self.world_positions)
