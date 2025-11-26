"""
Dreamer-based Reinforcement Learning Trainer for Multi-Camera Autonomous Driving
Predicts: acceleration (speed change) and steering angle from camera observations

Training directly from dataset (supervised learning):
- Sample sequences of frames from dataset
- Train world model (VAE + RNN) to encode observations and predict actions
- The RNN learns temporal dynamics from sequential frames
"""

import os
import tqdm
from args import args

import torch
from torch import nn
from torch.distributions import Normal, Independent
import torch.optim as optim
import numpy as np
import math

from src.once_dataset import ONCEDataset
from src.Models.networks import (
    CNNRepresentationFusion, VAEConvDecoder, VAEConvEncoder, 
    LatentDynamicsModel, RecurrentModel, Actor, Critic, ActionPredictor
)
from src.trajectory import TrajectoryHistory
from src.transforms import Pose, CoordinateTransformer
from src.renderers import MatplotlibRenderer


def compute_actions_from_trajectory(trajectory: TrajectoryHistory):
    """
    Compute acceleration and steering angle from TrajectoryHistory.
    
    Args:
        trajectory: TrajectoryHistory with at least 3 positions
    
    Returns:
        acceleration: Speed change between frames
        steering_angle: Angle between previous and current direction vectors
    """
    if len(trajectory) < 3:
        return 0.0, 0.0
    
    positions = trajectory.get_recent(3)  # Get last 3 positions
    p0, p1, p2 = positions[0], positions[1], positions[2]
    
    # Compute velocity vectors (2D, using x and y)
    v1 = p1[:2] - p0[:2]  # Previous direction
    v2 = p2[:2] - p1[:2]  # Current direction
    
    # Acceleration: difference in speed magnitude
    speed1 = np.linalg.norm(v1)
    speed2 = np.linalg.norm(v2)
    acceleration = speed2 - speed1
    
    # Clamp acceleration (noise reduction)
    if abs(acceleration) < 0.1:
        acceleration = 0.0
    
    # Steering angle: angle between direction vectors
    if speed1 > 1e-6 and speed2 > 1e-6:
        # Normalize vectors
        v1_norm = v1 / speed1
        v2_norm = v2 / speed2
        
        # Compute angle using cross product (for sign) and dot product (for magnitude)
        cross = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
        dot = np.dot(v1_norm, v2_norm)
        steering_angle = math.atan2(cross, dot)
    else:
        steering_angle = 0.0
    
    return acceleration, steering_angle


def predict_next_position(trajectory: TrajectoryHistory, acceleration: float, steering: float):
    """
    Predict the next position given current trajectory and action.
    
    Args:
        trajectory: TrajectoryHistory with at least 2 positions
        acceleration: Predicted acceleration
        steering: Predicted steering angle
    
    Returns:
        predicted_position: (x, y) predicted next position
        current_direction: (dx, dy) current direction vector (normalized)
        current_speed: current speed magnitude
    """
    if len(trajectory) < 2:
        return np.array([0.0, 0.0]), np.array([0.0, 1.0]), 0.0
    
    positions = trajectory.get_recent(2)
    p0, p1 = positions[0], positions[1]
    
    # Current direction and speed
    direction = p1[:2] - p0[:2]
    current_speed = np.linalg.norm(direction)
    
    if current_speed > 1e-6:
        direction_norm = direction / current_speed
    else:
        direction_norm = np.array([0.0, 1.0])  # Default forward
    
    # Apply steering to rotate direction
    cos_s, sin_s = np.cos(steering), np.sin(steering)
    new_direction = np.array([
        direction_norm[0] * cos_s - direction_norm[1] * sin_s,
        direction_norm[0] * sin_s + direction_norm[1] * cos_s
    ])
    
    # Apply acceleration to get new speed
    # Clamp acceleration (noise reduction)
    if abs(acceleration) < 0.1:
        acceleration = 0.0
        
    new_speed = current_speed + acceleration
    new_speed = max(0.0, new_speed)  # Speed can't be negative
    
    # Predict next position
    predicted_position = p1[:2] + new_direction * new_speed
    
    return predicted_position, new_direction, new_speed


def compute_reward(predicted_pos: np.ndarray, gt_pos: np.ndarray) -> float:
    """
    Compute reward based on L1 distance between predicted and ground truth position.
    Reward is negative L1 distance (closer = higher reward).
    """
    l1_distance = np.abs(predicted_pos - gt_pos[:2]).sum()
    return -l1_distance


class MultiCameraDreamerTrainer:
    """
    Dreamer-based trainer for multi-camera autonomous driving.
    Learns to predict acceleration and steering from camera observations.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.num_cameras = 7
        
        # Encoders for each camera
        self.encoders = nn.ModuleList([
            VAEConvEncoder(config.observationShape, config.encodedObsSize, config.encoder)
            for _ in range(self.num_cameras)
        ]).to(self.device)
        
        # Decoders for each camera
        self.decoders = nn.ModuleList([
            VAEConvDecoder(config.fullStateSize, config.observationShape, config.decoder)
            for _ in range(self.num_cameras)
        ]).to(self.device)
        
        # Camera fusion network
        self.representation_fusion = CNNRepresentationFusion(
            inputShape=(self.num_cameras, config.encodedObsSize),
            outputSize=config.fusedSize
        ).to(self.device)
        
        # VAE latent model (reparameterization)
        self.latent_model = LatentDynamicsModel().to(self.device)
        
        # Recurrent model for temporal dynamics
        self.recurrent_model = RecurrentModel(
            recurrentSize=config.recurrentSize,
            latentSize=config.fusedSize,
            actionSize=2,  # acceleration + steering
            config=config.recurrent
        ).to(self.device)
        
        # Action predictor (predicts acceleration and steering)
        self.action_predictor = ActionPredictor(
            inputSize=config.fullStateSize,
            hiddenSize=config.actionPredictorHiddenSize
        ).to(self.device)
        
        # Initial recurrent state
        self.initial_recurrent_state = nn.Parameter(
            torch.zeros(1, config.recurrentSize)
        ).to(self.device)
        
        # Visualization components (initialized in train if needed)
        self.renderer = None
        self.trajectory = None
        self.transformer = None
        
        # Metrics tracking
        self.reward_history = []
        
        print(f"MultiCameraDreamerTrainer initialized with {self.num_cameras} cameras")

    def _init_visualization(self):
        """Initialize the MatplotlibRenderer for training visualization"""
        self.trajectory = TrajectoryHistory(max_points=50)
        self.transformer = CoordinateTransformer()
        self.renderer = MatplotlibRenderer(
            trajectory_history=self.trajectory,
            transformer=self.transformer,
            figsize=(22, 10),
            training_mode=True
        )

    def _prepare_camera_images_for_render(self, frame_data: dict) -> list:
        """Convert frame data to format expected by renderer"""
        camera_data = frame_data.get("camera_data", {})
        camera_images = []
        
        for cam_id, cam_dict in camera_data.items():
            image_tensor = cam_dict.get("image_tensor", None)
            if image_tensor is None:
                continue
            
            # Convert tensor to numpy (C, H, W) -> (H, W, C)
            image_np = image_tensor.numpy()
            if image_np.ndim == 3 and image_np.shape[0] in [1, 3, 4]:
                image_np = image_np.transpose(1, 2, 0)
            
            boxes_2d = cam_dict.get("2D_bboxes", [])
            
            camera_images.append({
                'cam': cam_id,
                'image': image_np,
                'boxes': boxes_2d
            })
        
        return camera_images

    def train(self, dataset: ONCEDataset, epochs=10, visualize=False):
        """
        Train world model directly from dataset sequences.
        For each sequence of batchLength frames:
        - Encode observations with VAE
        - Process through RNN with actions
        - Learn to reconstruct observations and predict actions
        """
        reconstruction_loss_fn = nn.L1Loss()
        
        # Collect all parameters
        all_params = []
        all_params.extend(self.encoders.parameters())
        all_params.extend(self.decoders.parameters())
        all_params.extend(self.representation_fusion.parameters())
        all_params.extend(self.recurrent_model.parameters())
        all_params.extend(self.action_predictor.parameters())
        
        optimizer = optim.Adam(all_params, lr=self.config.lr)
        
        # Initialize visualization if needed
        if visualize:
            self._init_visualization()
        
        # Training loop
        for epoch in range(epochs):
            epoch_recon_loss = 0.0
            epoch_action_loss = 0.0
            epoch_count = 0
            
            # Reset recurrent state and trajectory at start of each epoch
            recurrent_state = self.initial_recurrent_state.clone()
            trajectory = TrajectoryHistory(max_points=10)
            prev_action = torch.zeros(1, 2, device=self.device)
            
            # Also reset visualization trajectory
            if visualize and self.trajectory is not None:
                self.trajectory.clear()
            
            # Process dataset sequentially (maintaining temporal continuity)
            progress_bar = tqdm.tqdm(range(len(dataset)), desc=f"Epoch {epoch+1}/{epochs}")
            
            for idx in progress_bar:
                frame_data = dataset.frame_by_index(idx)
                camera_data = frame_data.get("camera_data", {})
                camera_names = list(camera_data.keys())
                
                if not camera_names:
                    continue
                
                # Load and process all camera images
                camera_tensors = []
                for cam_idx, cam_name in enumerate(camera_names):
                    if cam_idx >= self.num_cameras:
                        break
                    img_tensor = camera_data[cam_name]["image_tensor"]
                    img_tensor = torch.nn.functional.interpolate(
                        img_tensor.unsqueeze(0),
                        size=(self.config.observationShape[1], self.config.observationShape[2]),
                        mode='bilinear',
                        align_corners=False
                    )
                    camera_tensors.append(img_tensor)
                
                # Pad if fewer cameras
                while len(camera_tensors) < self.num_cameras:
                    camera_tensors.append(torch.zeros(1, *self.config.observationShape, device=self.device))
                
                # Stack cameras: (1, num_cameras, C, H, W)
                observation = torch.cat(camera_tensors, dim=0).unsqueeze(0).to(self.device)
                
                # Get pose from camera data and update trajectory
                pose_array = camera_data[camera_names[0]].get("position", None)
                if pose_array is not None:
                    pose = Pose.from_array(np.array(pose_array))
                    trajectory.add_pose(pose)
                    # Also update visualization trajectory
                    if visualize and self.trajectory is not None:
                        self.trajectory.add_pose(pose)
                
                # Compute ground truth action from trajectory history
                gt_acc, gt_steer = compute_actions_from_trajectory(trajectory)
                gt_action = torch.tensor([[gt_acc, gt_steer]], dtype=torch.float32, device=self.device)
                
                # Get current direction and speed for visualization
                # gt_speed should be the speed AFTER applying gt_acc (i.e., current_speed + gt_acc)
                if len(trajectory) >= 2:
                    recent_pos = trajectory.get_recent(2)
                    gt_velocity = recent_pos[1][:2] - recent_pos[0][:2]
                    current_speed = np.linalg.norm(gt_velocity)
                    
                    if current_speed > 1e-6:
                        current_direction_norm = gt_velocity / current_speed
                    else:
                        current_direction_norm = np.array([0.0, 1.0])
                    
                    # Calculate GT next direction by applying GT steering to current direction
                    # This makes it comparable to predicted direction (which is also next direction)
                    cos_gt, sin_gt = np.cos(gt_steer), np.sin(gt_steer)
                    gt_direction = np.array([
                        current_direction_norm[0] * cos_gt - current_direction_norm[1] * sin_gt,
                        current_direction_norm[0] * sin_gt + current_direction_norm[1] * cos_gt
                    ])
                    
                    # gt_speed is the speed after applying ground truth acceleration
                    gt_speed = max(0.0, current_speed + gt_acc)
                else:
                    gt_direction = np.array([0.0, 1.0])
                    gt_speed = 0.0
                
                # Encode each camera
                encoded_representations = []
                for cam_idx in range(self.num_cameras):
                    cam_obs = observation[:, cam_idx]  # (1, C, H, W)
                    mu, logvar = self.encoders[cam_idx](cam_obs)
                    encoded = self.latent_model(mu, logvar)
                    encoded_representations.append(encoded)
                
                # Fuse camera representations
                representation_tensor = torch.stack(encoded_representations, dim=1)
                fused_representation = self.representation_fusion(representation_tensor)
                
                # Update recurrent state (detach to prevent BPTT across entire dataset)
                # prev_action is from previous iteration
                recurrent_state = self.recurrent_model(
                    recurrent_state.detach(), 
                    fused_representation, 
                    prev_action
                )
                
                # Full state = recurrent + fused observation
                full_state = torch.cat([recurrent_state, fused_representation], dim=-1)
                
                # Predict action
                pred_acc_mean, pred_acc_std, pred_steer_mean, pred_steer_std = self.action_predictor(full_state)
                
                # Compute predicted next position and reward
                pred_pos, pred_direction, pred_speed = predict_next_position(
                    trajectory, 
                    pred_acc_mean.item(), 
                    pred_steer_mean.item()
                )
                
                # Get ground truth next position (current position since we're looking at where we ended up)
                if len(trajectory) >= 1:
                    gt_next_pos = trajectory.get_recent(1)[0][:2]  # Current position is the "next" from previous frame's perspective
                    reward = compute_reward(pred_pos, gt_next_pos)
                else:
                    reward = 0.0
                    gt_next_pos = np.array([0.0, 0.0])
                
                # Action loss
                acc_dist = Normal(pred_acc_mean, pred_acc_std)
                steer_dist = Normal(pred_steer_mean, pred_steer_std)
                action_loss = -acc_dist.log_prob(gt_action[:, 0]).mean()
                action_loss += -steer_dist.log_prob(gt_action[:, 1]).mean()
                
                # Reconstruction loss
                recon_loss = torch.tensor(0.0, device=self.device)
                reconstructed = []
                for cam_idx in range(self.num_cameras):
                    recon = self.decoders[cam_idx](full_state)
                    reconstructed.append(recon)
                    recon_loss += reconstruction_loss_fn(recon, observation[:, cam_idx])
                
                # Total loss
                total_loss = recon_loss + self.config.action_loss_weight * action_loss
                
                # Backprop
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_recon_loss += recon_loss.item()
                epoch_action_loss += action_loss.item()
                epoch_count += 1
                self.reward_history.append(reward)
                
                # Store current action as previous for next iteration
                prev_action = gt_action.detach()
                
                progress_bar.set_postfix({
                    "Recon": f"{recon_loss.item():.4f}",
                    "Action": f"{action_loss.item():.4f}",
                    "Reward": f"{reward:.3f}",
                    "Acc": f"{pred_acc_mean.item():.3f}",
                    "Steer": f"{math.degrees(pred_steer_mean.item()):.1f}°"
                })
                
                # Visualization using MatplotlibRenderer
                if visualize and self.renderer is not None:
                    camera_images = self._prepare_camera_images_for_render(frame_data)
                    
                    render_data = {
                        'camera_images': camera_images,
                        'pose': pose_array,
                        'entities': frame_data.get("entities", []),
                        'split_name': 'train',
                        'training_data': {
                            'epoch': epoch + 1,
                            'recon_loss': recon_loss.item(),
                            'action_loss': action_loss.item(),
                            'reward': reward,
                            'pred_acc': pred_acc_mean.item(),
                            'pred_steer': pred_steer_mean.item(),
                            'gt_acc': gt_acc,
                            'gt_steer': gt_steer,
                            'pred_direction': pred_direction.tolist(),
                            'gt_direction': gt_direction.tolist(),
                            'pred_speed': pred_speed,
                            'gt_speed': gt_speed
                        }
                    }
                    self.renderer.render(render_data, idx)
            
            # Log epoch metrics
            if epoch_count > 0:
                avg_recon = epoch_recon_loss / epoch_count
                avg_action = epoch_action_loss / epoch_count
                print(f"\nEpoch {epoch+1}/{epochs} - Avg Recon Loss: {avg_recon:.4f}, Avg Action Loss: {avg_action:.4f}")
        
        # Clean up visualization
        if visualize and self.renderer is not None:
            self.renderer.close()

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'encoders': self.encoders.state_dict(),
            'decoders': self.decoders.state_dict(),
            'representation_fusion': self.representation_fusion.state_dict(),
            'recurrent_model': self.recurrent_model.state_dict(),
            'action_predictor': self.action_predictor.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoders.load_state_dict(checkpoint['encoders'])
        self.decoders.load_state_dict(checkpoint['decoders'])
        self.representation_fusion.load_state_dict(checkpoint['representation_fusion'])
        self.recurrent_model.load_state_dict(checkpoint['recurrent_model'])
        self.action_predictor.load_state_dict(checkpoint['action_predictor'])
        print(f"Checkpoint loaded from {path}")


def main():
    # Configuration
    class Config:
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Image dimensions (downscaled for efficiency)
            self.observationShape = (3, 192, 102)  # (C, H, W) - 10x smaller than original
            
            # Encoding sizes
            self.encodedObsSize = 512
            self.fusedSize = 1024
            self.recurrentSize = 512
            self.fullStateSize = self.recurrentSize + self.fusedSize
            
            # Training params
            self.lr = 1e-4
            self.action_loss_weight = 1.0
            self.batchSize = 16
            self.batchLength = 8  # Sequence length for world model training
            
            # Action predictor hidden size
            self.actionPredictorHiddenSize = 256
            
            # Encoder config (matches VAEConvEncoder expectations)
            self.encoder = type('obj', (object,), {
                'activation': 'GELU',
            })()
            
            # Decoder config (matches VAEConvDecoder expectations)
            self.decoder = type('obj', (object,), {
                'activation': 'GELU',
            })()
            
            # Recurrent config (matches RecurrentModel expectations)
            self.recurrent = type('obj', (object,), {
                'activation': 'GELU',
                'hiddenSize': 512
            })()
    
    config = Config()
    print(f"Using device: {config.device}")
    
    print(f"\n{'='*60}")
    print(f"Multi-Camera Dreamer Training")
    print(f"Predicting: Acceleration & Steering Angle")
    print(f"{'='*60}\n")
    
    # Load dataset
    dataset = ONCEDataset(
        data_path=args.data_path,
        split="train",
        data_type="camera",
        level="frame",
        logger_name="DreamerTraining",
        show_logs=True
    )
    
    # Create trainer
    trainer = MultiCameraDreamerTrainer(config)
    
    print("\nStarting training with visualization...")
    print("Press 'q' in the visualization window to quit.\n")
    
    # Train
    trainer.train(dataset, epochs=20, visualize=True)
    
    # Save checkpoint
    trainer.save_checkpoint("dreamer_checkpoint.pth")


if __name__ == "__main__":
    main()
