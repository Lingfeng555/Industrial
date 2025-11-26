import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
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
        training_mode: bool = False,
    ):
        self.trajectory_history = trajectory_history
        self.transformer = transformer
        self.training_mode = training_mode

        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=figsize)
        
        if training_mode:
            # Training layout: cameras | lidar+trajectory | metrics+steering
            self.gs = self.fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 0.8], wspace=0.15)
        else:
            self.gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.2)
        
        self.grid_rows, self.grid_cols = 3, 4
        
        if training_mode:
            self.right_gs = self.gs[0, 0].subgridspec(
                self.grid_rows, self.grid_cols, 
                hspace=0.05, wspace=0.05
            )
        else:
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
        
        # Training mode specific
        self._loss_history = {'recon': [], 'action': [], 'reward': []}
        self._ax_metrics = None
        self._ax_steering = None
        self._ax_trajectory = None  # Reference to trajectory axis for direction arrows
        self._loss_lines = {}
        self._steering_arrow_pred = None
        self._steering_arrow_gt = None
        self._pred_direction_arrow = None
        self._gt_direction_arrow = None
        
        self._initialize_axes()
        if self._forward_arrow is not None:
            self._forward_arrow.set_visible(True)
    
    def render(self, frame_data: Dict, frame_idx: int):
        lidar_points = frame_data.get('lidar_points')
        boxes_3d = frame_data.get('boxes_3d', [])
        camera_images = frame_data.get('camera_images', [])
        entities = frame_data.get('entities', [])
        pose_array = frame_data.get('pose')
        split_name = frame_data.get('split_name', 'unknown')
        
        # Training-specific data
        training_data = frame_data.get('training_data', None)
        
        if not self.training_mode:
            ax_lidar = self.fig.axes[0]
            if lidar_points is not None:
                self._render_lidar_view(
                    ax_lidar, lidar_points, boxes_3d, pose_array
                )

        self._render_camera_grid(camera_images, entities)
        
        if self.training_mode and training_data:
            self._render_training_metrics(training_data, frame_idx)
        
        title = f"Split {split_name} — frame {frame_idx}"
        if self.training_mode and training_data:
            epoch = training_data.get('epoch', 0)
            title = f"Training Epoch {epoch} — frame {frame_idx}"
        
        self.fig.suptitle(title)
        plt.pause(0.001)
    
    def _initialize_axes(self):
        if self.training_mode:
            # Lidar/trajectory in middle
            ax_lidar = self.fig.add_subplot(self.gs[0, 1])
            self._ax_trajectory = ax_lidar  # Store reference for direction arrows
            ax_lidar.set_title("Trajectory & Direction Prediction")
            ax_lidar.set_xlabel('X')
            ax_lidar.set_ylabel('Y')
            ax_lidar.set_xlim(-50, 50)
            ax_lidar.set_ylim(-50, 50)
            ax_lidar.set_aspect('equal')
            ax_lidar.grid(True, alpha=0.3)
            
            # Trajectory visualization (no lidar in training mode typically)
            self._trajectory_line, = ax_lidar.plot([], [], c=self.trajectory_color, 
                                                   linewidth=self.trajectory_line_width, marker='.', label='Trajectory')
            self._current_pos_scatter = ax_lidar.scatter([0], [0], color=self.current_position_color, 
                                                         s=100, marker='o', edgecolors='white', 
                                                         linewidths=2, zorder=5, label='Current')
            ax_lidar.legend(loc='upper right', fontsize=8)
            
            # Right side: metrics and steering
            right_gs = self.gs[0, 2].subgridspec(3, 1, hspace=0.35, height_ratios=[1, 1, 0.8])
            
            # Loss metrics plot
            self._ax_metrics = self.fig.add_subplot(right_gs[0, 0])
            self._ax_metrics.set_title("Training Losses & Reward", fontsize=10)
            self._ax_metrics.set_xlabel('Step')
            self._ax_metrics.set_ylabel('Loss')
            self._loss_lines['recon'], = self._ax_metrics.plot([], [], 'g-', label='Reconstruction', linewidth=1.5)
            self._loss_lines['action'], = self._ax_metrics.plot([], [], 'y-', label='Action', linewidth=1.5)
            self._ax_metrics.legend(loc='upper right', fontsize=8)
            self._ax_metrics.grid(True, alpha=0.3)
            
            # Reward plot
            self._ax_reward = self.fig.add_subplot(right_gs[1, 0])
            self._ax_reward.set_title("Reward (neg. L1 distance)", fontsize=10)
            self._ax_reward.set_xlabel('Step')
            self._ax_reward.set_ylabel('Reward')
            self._loss_lines['reward'], = self._ax_reward.plot([], [], 'c-', label='Reward', linewidth=1.5)
            self._ax_reward.legend(loc='upper right', fontsize=8)
            self._ax_reward.grid(True, alpha=0.3)
            
            # Steering visualization
            self._ax_steering = self.fig.add_subplot(right_gs[2, 0])
            self._ax_steering.set_title("Steering Prediction", fontsize=10)
            self._ax_steering.set_xlim(-1.5, 1.5)
            self._ax_steering.set_ylim(-0.5, 1.5)
            self._ax_steering.set_aspect('equal')
            self._ax_steering.axis('off')
            
            # Draw steering wheel
            circle = plt.Circle((0, 0.5), 0.8, fill=False, color='gray', linewidth=2)
            self._ax_steering.add_patch(circle)
            
        else:
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

        # Camera grid
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
    
    def _render_training_metrics(self, training_data: Dict, frame_idx: int):
        """Render training-specific metrics"""
        recon_loss = training_data.get('recon_loss', 0)
        action_loss = training_data.get('action_loss', 0)
        reward = training_data.get('reward', 0)
        pred_acc = training_data.get('pred_acc', 0)
        pred_steer = training_data.get('pred_steer', 0)
        gt_acc = training_data.get('gt_acc', 0)
        gt_steer = training_data.get('gt_steer', 0)
        pred_direction = training_data.get('pred_direction', [0, 1])
        gt_direction = training_data.get('gt_direction', [0, 1])
        pred_speed = training_data.get('pred_speed', 1.0)
        gt_speed_from_trainer = training_data.get('gt_speed', 1.0)
        
        # Update loss history
        self._loss_history['recon'].append(recon_loss)
        self._loss_history['action'].append(action_loss)
        self._loss_history['reward'].append(reward)
        
        # Keep only last 500 points
        max_history = 500
        for key in self._loss_history:
            if len(self._loss_history[key]) > max_history:
                self._loss_history[key] = self._loss_history[key][-max_history:]
        
        # Update loss plot
        x_data = list(range(len(self._loss_history['recon'])))
        self._loss_lines['recon'].set_data(x_data, self._loss_history['recon'])
        self._loss_lines['action'].set_data(x_data, self._loss_history['action'])
        self._ax_metrics.relim()
        self._ax_metrics.autoscale_view()
        
        # Update reward plot
        self._loss_lines['reward'].set_data(x_data, self._loss_history['reward'])
        self._ax_reward.relim()
        self._ax_reward.autoscale_view()
        
        # Update steering visualization
        # Clear previous arrows
        for patch in self._ax_steering.patches[:]:
            if isinstance(patch, FancyArrow):
                patch.remove()
        
        # Use gt_speed from trainer
        gt_speed = gt_speed_from_trainer
        
        # Scale arrows by speed (base + speed factor)
        base_arrow = 0.2
        speed_factor = 0.05  # Scale factor for speed contribution
        
        # Draw ground truth steering (cyan) - direction from steer, length from speed
        # Note: Positive steer is Left (CCW), so we negate x component for display (Left = negative X)
        gt_arrow_len = base_arrow + gt_speed * speed_factor
        gt_dx = -gt_arrow_len * np.sin(gt_steer)
        gt_dy = gt_arrow_len * np.cos(gt_steer)
        self._ax_steering.arrow(0, 0.5, gt_dx, gt_dy, head_width=0.1, head_length=0.05,
                                fc='cyan', ec='cyan', linewidth=2, label='GT')
        
        # Draw predicted steering (yellow) - direction from steer, length from predicted speed
        pred_arrow_len = base_arrow + pred_speed * speed_factor
        pred_dx = -pred_arrow_len * np.sin(pred_steer)
        pred_dy = pred_arrow_len * np.cos(pred_steer)
        self._ax_steering.arrow(0, 0.5, pred_dx, pred_dy, head_width=0.08, head_length=0.04,
                                fc='yellow', ec='yellow', linewidth=2, label='Pred')
        
        # Add text info
        for text in self._ax_steering.texts[:]:
            text.remove()
        
        self._ax_steering.text(-1.4, -0.3, f"Pred Acc: {pred_acc:.4f}", fontsize=9, color='yellow')
        self._ax_steering.text(-1.4, -0.45, f"Pred Steer: {np.degrees(pred_steer):.1f}°", fontsize=9, color='yellow')
        self._ax_steering.text(0.3, -0.3, f"GT Acc: {gt_acc:.4f}", fontsize=9, color='cyan')
        self._ax_steering.text(0.3, -0.45, f"GT Steer: {np.degrees(gt_steer):.1f}°", fontsize=9, color='cyan')
        self._ax_steering.text(-0.6, -0.6, f"Reward: {reward:.3f}", fontsize=10, color='lime', fontweight='bold')
        
        # Update trajectory with direction arrows
        if len(self.trajectory_history) > 0 and self._ax_trajectory is not None:
            world_positions = self.trajectory_history.get_world_positions()
            if len(world_positions) > 1:
                # Center on current position
                current_pos = world_positions[-1]
                relative_positions = world_positions - current_pos
                self._trajectory_line.set_data(relative_positions[:, 0], relative_positions[:, 1])
                
                # Clear previous direction arrows from trajectory plot
                for patch in self._ax_trajectory.patches[:]:
                    if isinstance(patch, FancyArrow):
                        patch.remove()
                
                # Use gt_speed from trainer (consistent with how pred_speed is computed)
                gt_speed = gt_speed_from_trainer
                
                # Scale factor to make arrows visible (speeds can be small)
                # We use a base scale + proportional to speed
                base_scale = 3.0
                speed_scale = 2.0  # multiplier for speed component
                
                # Draw ground truth direction (cyan) - with actual magnitude
                gt_dir = np.array(gt_direction)
                gt_arrow_length = base_scale + gt_speed * speed_scale
                self._ax_trajectory.arrow(0, 0, gt_dir[0] * gt_arrow_length, gt_dir[1] * gt_arrow_length,
                                         head_width=2.0, head_length=1.5,
                                         fc='cyan', ec='cyan', linewidth=2, zorder=10,
                                         alpha=0.8)
                
                # Draw predicted direction (yellow) - with predicted magnitude
                pred_dir = np.array(pred_direction)
                pred_arrow_length = base_scale + pred_speed * speed_scale
                self._ax_trajectory.arrow(0, 0, pred_dir[0] * pred_arrow_length, pred_dir[1] * pred_arrow_length,
                                         head_width=1.5, head_length=1.0,
                                         fc='yellow', ec='yellow', linewidth=2, zorder=11,
                                         alpha=0.8)
                
                # Add legend for arrows with speed info
                for text in self._ax_trajectory.texts[:]:
                    text.remove()
                self._ax_trajectory.text(25, 45, f"→ GT Dir (spd: {gt_speed:.2f})", fontsize=8, color='cyan')
                self._ax_trajectory.text(25, 40, f"→ Pred Dir (spd: {pred_speed:.2f})", fontsize=8, color='yellow')
    
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