import numpy as np

class Transformation:
    @staticmethod
    def pose_to_matrix(pose):
        qx, qy, qz, qw, tx, ty, tz = pose
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, tx],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw, ty],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2, tz],
            [0, 0, 0, 1]
        ])

        return R
    
    @staticmethod
    def get_world_position_from_pose(pose):
        return np.array([pose[4], pose[5], pose[6]])