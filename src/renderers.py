import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from .transforms import Pose, CoordinateTransformer
from .trajectory import TrajectoryHistory


class Renderer(ABC):
    @abstractmethod
    def render(self, frame_data: Dict, frame_idx: int):
        pass
    
    @abstractmethod
    def close(self):
        pass


class MatplotlibRenderer(Renderer):    
    def __init__(
        self,
        trajectory_history: TrajectoryHistory,
        transformer: CoordinateTransformer,
        figsize: Tuple[int, int] = (20, 10),
    ):
        self.trajectory_history = trajectory_history
        self.transformer = transformer

        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=figsize)
        self.gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.2)
        
        self.grid_rows, self.grid_cols = 3, 4
        self.right_gs = self.gs[0, 1].subgridspec(
            self.grid_rows, self.grid_cols, 
            hspace=0.05, wspace=0.05
        )
        
        self.camera_order = ['cam05', 'cam06', 'cam09', 'cam03', 'cam01', 'cam07', 'cam08']
        self.camera_positions = [
            (0, 1), (0, 2),
            (1, 0), (1, 2), (1, 3),
            (2, 1), (2, 2)
        ]

        self.sensors_img = plt.imread(os.path.join(os.getcwd(), 'assets/images/sensors.png'))

        self.lidar_color = (0.8, 0.6, 0.0)
        self.trajectory_color = 'cyan'
        self.current_position_color = 'lime'
        self.forward_arrow_color = 'red'
        self.bbox_color = 'r'
        
        self.trajectory_line_width = 2
        self.max_trajectory_points = 50
        self.forward_arrow_length = 5.0
        
        self.use_distance_coloring = True
        self.distance_colormap = 'turbo'
        self.max_distance = 80.0 
        
        self._lidar_scatter = None
        self._trajectory_line = None
        self._current_pos_scatter = None
        self._forward_arrow = None
        self._bbox_lines = []
        self._camera_axes = {}
        self._camera_images = {}
        self._initialize_axes()
        self._forward_arrow.set_visible(True)
    
    def render(self, frame_data: Dict, frame_idx: int):
        lidar_points = frame_data.get('lidar_points')
        boxes_3d = frame_data.get('boxes_3d', [])
        camera_images = frame_data.get('camera_images', [])
        entities = frame_data.get('entities', [])
        pose_array = frame_data.get('pose')
        split_name = frame_data.get('split_name', 'unknown')
        
        ax_lidar = self.fig.axes[0]
        
        self._render_lidar_view(
            ax_lidar, lidar_points, boxes_3d, pose_array
        )

        self._render_camera_grid(camera_images, entities)
        
        self.fig.suptitle(f"Split {split_name} — frame {frame_idx}")
        plt.pause(0.01)
    
    def _initialize_axes(self):
        ax_lidar = self.fig.add_subplot(self.gs[0, 0])
        ax_lidar.set_title("LiDAR (top-down)")
        ax_lidar.set_xlabel('X')
        ax_lidar.set_ylabel('Y')
        
        if self.use_distance_coloring:
            self._lidar_scatter = ax_lidar.scatter([], [], s=1, c=[], cmap=self.distance_colormap, 
                                                  vmin=0, vmax=self.max_distance)
        else:
            self._lidar_scatter = ax_lidar.scatter([], [], s=1, color=self.lidar_color)
        self._trajectory_line, = ax_lidar.plot([], [], c=self.trajectory_color, 
                                               linewidth=self.trajectory_line_width, marker='.')
        self._current_pos_scatter = ax_lidar.scatter([0], [0], color=self.current_position_color, 
                                                     s=50, marker='o', edgecolors='white', 
                                                     linewidths=2, zorder=5)
        self._forward_arrow = ax_lidar.arrow(0, 0, 0, -self.forward_arrow_length,
                                            head_width=1.5, head_length=1.0,
                                            fc=self.forward_arrow_color, 
                                            ec=self.forward_arrow_color,
                                            linewidth=2, zorder=6, visible=False)

        cam_idx = 0
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                ax = self.fig.add_subplot(self.right_gs[r, c])
                ax.axis('off')
                
                if (r, c) == (1, 1):
                    if self.sensors_img is not None:
                        ax.imshow(self.sensors_img)
                    continue
                
                if (r, c) in self.camera_positions and cam_idx < len(self.camera_order):
                    cam_id = self.camera_order[cam_idx]
                    ax.set_title(cam_id, fontsize=8)
                    self._camera_axes[cam_id] = ax
                    self._camera_images[cam_id] = None
                    cam_idx += 1
    
    def _render_lidar_view(
        self, 
        ax, 
        lidar_points: np.ndarray, 
        boxes_3d: List,
        pose_array: Optional[np.ndarray],
    ):
        self._lidar_scatter.set_offsets(lidar_points[:, :2])
        
        if self.use_distance_coloring:
            distances = np.sqrt(lidar_points[:, 0]**2 + lidar_points[:, 1]**2)
            self._lidar_scatter.set_array(distances)
        
        current_pose = Pose.from_array(pose_array)
        self._render_trajectory_2d(current_pose)
        
        for line in self._bbox_lines:
            line.remove()
        self._bbox_lines.clear()
        
        for bbox in boxes_3d:
            corners = self._bbox7_to_corners(np.asarray(bbox))
            xs, ys = corners[:, 0], corners[:, 1]
            line, = ax.plot(
                list(xs[:4]) + [xs[0]], 
                list(ys[:4]) + [ys[0]], 
                c=self.bbox_color
            )
            self._bbox_lines.append(line)
        
        ax.relim()
        ax.autoscale_view()
    
    def _bbox7_to_corners(self, box7):
        cx, cy, cz, l, w, h, yaw = box7
        dx = l / 2.0
        dy = w / 2.0
        dz = h / 2.0
        local = np.array([
            [ dx,  dy, -dz],
            [ dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx,  dy, -dz],
            [ dx,  dy,  dz],
            [ dx, -dy,  dz],
            [-dx, -dy,  dz],
            [-dx,  dy,  dz]
        ])
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        R = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
        corners = (R @ local.T).T + np.array([cx, cy, cz])
        return corners

    def _render_trajectory_2d(self, current_pose: Pose):
        world_positions = self.trajectory_history.get_world_positions()
        vehicle_positions = self.transformer.world_to_vehicle_2d(
            world_positions, current_pose
        )
        
        n_points = min(self.max_trajectory_points, len(vehicle_positions))
        traj_to_plot = vehicle_positions[-n_points:]
        self._trajectory_line.set_data(traj_to_plot[:, 0], traj_to_plot[:, 1])
        self._current_pos_scatter.set_offsets([[0.0, 0.0]])
    
    def _render_camera_grid(self, camera_images: List[Dict], entities: List):
        images_by_cam = {img['cam']: img for img in camera_images}
        
        for cam_id, ax in self._camera_axes.items():
            for patch in ax.patches[:]:
                patch.remove()
            for text in ax.texts[:]:
                text.remove()
            
            img_entry = images_by_cam.get(cam_id)
            if img_entry is None:
                if self._camera_images[cam_id] is not None:
                    self._camera_images[cam_id].set_data(np.zeros((10, 10, 3)))
                continue
            
            if self._camera_images[cam_id] is None:
                self._camera_images[cam_id] = ax.imshow(img_entry['image'])
            else:
                self._camera_images[cam_id].set_data(img_entry['image'])
            
            boxes_2d = img_entry.get('boxes', [])
            if boxes_2d:
                for box_idx, box in enumerate(boxes_2d):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    if entities and box_idx < len(entities):
                        label = entities[box_idx]
                        ax.text(
                            x1, max(y1 - 5, 0), label,
                            color='yellow', fontsize=5,
                            backgroundcolor='none'
                        )
    
    def close(self):
        plt.close(self.fig)