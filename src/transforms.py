import numpy as np

class Pose:
    def __init__(self, quaternion: np.ndarray, translation: np.ndarray):
        self.quaternion = quaternion / np.linalg.norm(quaternion)
        self.translation = translation
        self._rotation_matrix = None
        self._transform_matrix = None
    
    @classmethod
    def from_array(cls, pose_array: np.ndarray) -> 'Pose':
        return cls(
            quaternion=pose_array[:4],
            translation=pose_array[4:7]
        )
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        if self._rotation_matrix is None:
            self._rotation_matrix = self._quaternion_to_rotation()
        return self._rotation_matrix
    
    @property
    def transform_matrix(self) -> np.ndarray:
        if self._transform_matrix is None:
            T = np.identity(4)
            T[:3, :3] = self.rotation_matrix
            T[:3, 3] = self.translation
            self._transform_matrix = T
        return self._transform_matrix
    
    def _quaternion_to_rotation(self) -> np.ndarray:
        qx, qy, qz, qw = self.quaternion
        
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return R
    
    def get_yaw(self) -> float:
        R = self.rotation_matrix
        return np.arctan2(R[1, 0], R[0, 0])


class CoordinateTransformer:  
    def world_to_vehicle_2d(
        self, 
        world_points: np.ndarray, 
        current_pose: Pose
    ) -> np.ndarray:
        if world_points.shape[1] >= 2:
            world_points_2d = world_points[:, :2]
        else:
            raise ValueError(f"Expected at least 2 columns, got {world_points.shape[1]}")
        
        R_vehicle_to_world = current_pose.rotation_matrix[:2, :2]
        R_world_to_vehicle = R_vehicle_to_world.T

        delta = world_points_2d - current_pose.translation[:2]
        vehicle_points = (R_world_to_vehicle @ delta.T).T
        
        return vehicle_points
    
    def world_to_vehicle_3d(
        self, 
        world_points: np.ndarray, 
        current_pose: Pose
    ) -> np.ndarray:
        R_world_to_vehicle = current_pose.rotation_matrix.T
        delta = world_points - current_pose.translation
        vehicle_points = (R_world_to_vehicle @ delta.T).T
        
        return vehicle_points
