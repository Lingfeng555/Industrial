"""Main visualization controller."""

import numpy as np
from typing import List, Optional
from once_dataset import ONCEDataset

from .transforms import Pose, CoordinateTransformer
from .trajectory import TrajectoryHistory
from .renderers import MatplotlibRenderer, Renderer


class FrameData:
    """Container for frame data."""
    
    def __init__(self, dataset: ONCEDataset, frame_idx: int, split_name: str):
        """Extract and prepare frame data from dataset."""
        frame = dataset.frame_by_index(frame_idx)
        
        # Extract LiDAR data
        lidar_data = frame.get("lidar_data", {})
        lidar_points = lidar_data.get("points", None)
        
        self.lidar_points = self._to_numpy(lidar_points)[:, :3] if lidar_points is not None else None
        self.boxes_3d = lidar_data.get("3D_bboxes", [])
        
        # Extract pose
        self.pose = self._extract_pose(frame)
        
        # Extract camera data
        self.camera_images = self._extract_camera_images(frame)
        self.entities = frame.get("entities", [])
        
        # Metadata
        self.split_name = split_name
        self.frame_idx = frame_idx
    
    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        try:
            return tensor.numpy()
        except Exception:
            return np.asarray(tensor)
    
    @staticmethod
    def _extract_pose(frame: dict) -> Optional[np.ndarray]:
        """Extract pose from frame, trying multiple sources."""
        # Try frame-level pose first
        pose = frame.get("pose", None)
        if pose is not None:
            return np.array(pose)
        
        # Fallback to camera pose
        camera_data = frame.get("camera_data", {})
        if camera_data:
            try:
                first_cam = next(iter(camera_data.keys()))
                pose = camera_data[first_cam].get("position", None)
                if pose is not None:
                    return np.array(pose)
            except StopIteration:
                pass
        
        # Fallback to lidar pose
        lidar_data = frame.get("lidar_data", {})
        pose = lidar_data.get("position", None)
        if pose is not None:
            return np.array(pose)
        
        return None
    
    def _extract_camera_images(self, frame: dict) -> List[dict]:
        """Extract and prepare camera images (optimized)."""
        camera_data = frame.get("camera_data", {})
        if not camera_data:
            return []
        
        images = []
        # Only extract cameras we actually display
        cam_ids = ["cam05", "cam06", "cam09", "cam03", "cam01", "cam07", "cam08"]
        
        for cam_id in cam_ids:
            cam_dict = camera_data.get(cam_id, {})
            if not cam_dict:
                continue
                
            image_tensor = cam_dict.get("image_tensor", None)
            if image_tensor is None:
                continue
            
            # Convert to numpy and transpose CxHxW -> HxWxC
            image_np = self._to_numpy(image_tensor)
            if image_np.ndim == 3 and image_np.shape[0] in [1, 3, 4]:  # CxHxW format
                image_np = image_np.transpose(1, 2, 0)
            
            boxes_2d = cam_dict.get("2D_bboxes", [])
            
            images.append({
                'cam': cam_id,
                'image': image_np,
                'boxes': boxes_2d
            })
        
        return images
    
    def to_dict(self) -> dict:
        """Convert to dictionary for rendering."""
        return {
            'lidar_points': self.lidar_points,
            'boxes_3d': self.boxes_3d,
            'pose': self.pose,
            'camera_images': self.camera_images,
            'entities': self.entities,
            'split_name': self.split_name
        }


class ONCEVisualizer:
    """Main visualizer for ONCE dataset."""
    
    def __init__(self):
        """
        Initialize visualizer.
        
        Args:
            trajectory_yaw_offset_deg: Yaw offset for trajectory alignment
            playback_delay: Delay between frames in seconds
            use_open3d: Whether to use Open3D 3D viewer
            show_debug: Whether to show debug information
        """        
        # Core components
        self.trajectory_history = TrajectoryHistory(max_points=10)
        self.transformer = CoordinateTransformer()
        
        # Renderers
        self.renderers: List[Renderer] = []
    
    def visualize_split(self, dataset: ONCEDataset, split_name: str):
        """
        Visualize a dataset split.
        
        Args:
            dataset: ONCEDataset instance
            split_name: Name of the split being visualized
        """
        if len(dataset) == 0:
            print(f"Split '{split_name}' contains 0 frames. Skipping.")
            return
        
        # Initialize renderers
        self._initialize_renderers(split_name)
        
        # Clear trajectory history for new split
        self.trajectory_history.clear()
        
        try:
            # Process each frame
            for frame_idx in range(len(dataset)):
                frame_data = FrameData(dataset, frame_idx, split_name)
                
                # Update trajectory history
                if frame_data.pose is not None:
                    pose = Pose.from_array(frame_data.pose)
                    self.trajectory_history.add_pose(pose)
                
                # Render frame with all renderers
                frame_dict = frame_data.to_dict()
                for renderer in self.renderers:
                    renderer.render(frame_dict, frame_idx)
        
        finally:
            # Clean up renderers
            self._cleanup_renderers()
    
    def _initialize_renderers(self, split_name: str):
        """Initialize all renderers."""
        self.renderers.clear()
        
        # Matplotlib renderer (always enabled)
        matplotlib_renderer = MatplotlibRenderer(
            trajectory_history=self.trajectory_history,
            transformer=self.transformer
        )
        self.renderers.append(matplotlib_renderer)
    
    def _cleanup_renderers(self):
        """Clean up all renderers."""
        for renderer in self.renderers:
            renderer.close()
        self.renderers.clear()
