"""
Dreamer Model Inference Script
Loads a trained checkpoint and runs inference on the dataset with visualization.
Saves rendered figures for later use in BentoML serving.
"""

import os
import tqdm
import torch
import numpy as np
import math
from pathlib import Path

from args import args
from trainer import (
    MultiCameraDreamerTrainer, Config, 
    compute_actions_from_trajectory, 
    predict_next_position,
    compute_reward
    
)
from src.once_dataset import ONCEDataset
from src.trajectory import TrajectoryHistory
from src.transforms import Pose, CoordinateTransformer
from src.renderers import MatplotlibRenderer


class DreamerInference:
    """
    Inference engine for trained Dreamer model.
    Loads checkpoint and performs inference with visualization.
    """
    
    def __init__(self, checkpoint_path: str, config: Config):
        self.config = config
        self.device = config.device
        
        # Initialize trainer (loads all model components)
        self.trainer = MultiCameraDreamerTrainer(config)
        
        # Load checkpoint
        self.trainer.load_checkpoint(checkpoint_path)
        
        # Set models to eval mode
        self.trainer.encoders.eval()
        self.trainer.decoders.eval()
        self.trainer.representation_fusion.eval()
        self.trainer.recurrent_model.eval()
        self.trainer.action_predictor.eval()
        
        # Visualization components
        self.trajectory = None
        self.transformer = None
        self.renderer = None
        
        # Output directory for saved figures
        self.output_dir = Path("inference_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        print(f"✓ Saving outputs to {self.output_dir}")
    
    def _init_visualization(self):
        """Initialize the MatplotlibRenderer for inference visualization"""
        self.trajectory = TrajectoryHistory(max_points=50)
        self.transformer = CoordinateTransformer()
        self.renderer = MatplotlibRenderer(
            trajectory_history=self.trajectory,
            transformer=self.transformer,
            figsize=(22, 10),
            training_mode=True  # Show metrics
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
    
    def _save_figure(self, frame_idx: int):
        """Save current matplotlib figure to disk"""
        if self.renderer is None:
            return
        
        output_path = self.output_dir / f"frame_{frame_idx:06d}.png"
        self.renderer.fig.savefig(
            output_path, 
            dpi=100, 
            bbox_inches='tight',
            pad_inches=0.1
        )
    
    @torch.no_grad()
    def run_inference(self, dataset: ONCEDataset, max_frames: int = None, save_figures: bool = True):
        """
        Run inference on dataset with visualization and figure saving.
        
        Args:
            dataset: ONCEDataset to run inference on
            max_frames: Maximum number of frames to process (None = all)
            visualize: Whether to show live visualization
            save_figures: Whether to save figures to disk
        """
        if save_figures:
            visualize = True
        # Initialize visualization if needed
        if visualize:
            self._init_visualization()
        
        # Initialize recurrent state and trajectory
        recurrent_state = self.trainer.initial_recurrent_state.clone()
        trajectory = TrajectoryHistory(max_points=10)
        prev_action = torch.zeros(1, 2, device=self.device)
        
        # Metrics tracking
        rewards = []
        pred_actions = []
        gt_actions = []
        
        # Process frames
        num_frames = len(dataset) if max_frames is None else min(max_frames, len(dataset))
        progress_bar = tqdm.tqdm(range(num_frames), desc="Running Inference")
        
        for idx in progress_bar:
            frame_data = dataset.frame_by_index(idx)
            camera_data = frame_data.get("camera_data", {})
            camera_names = list(camera_data.keys())
            
            if not camera_names:
                continue
            
            # Load and process all camera images
            camera_tensors = []
            for cam_idx, cam_name in enumerate(camera_names):
                if cam_idx >= self.trainer.num_cameras:
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
            while len(camera_tensors) < self.trainer.num_cameras:
                camera_tensors.append(torch.zeros(1, *self.config.observationShape, device=self.device))
            
            # Stack cameras: (1, num_cameras, C, H, W)
            observation = torch.cat(camera_tensors, dim=0).unsqueeze(0).to(self.device)
            
            # Get pose and update trajectory
            pose_array = camera_data[camera_names[0]].get("position", None)
            if pose_array is not None:
                pose = Pose.from_array(np.array(pose_array))
                trajectory.add_pose(pose)
                if visualize and self.trajectory is not None:
                    self.trajectory.add_pose(pose)
            
            # Compute ground truth action
            gt_acc, gt_steer = compute_actions_from_trajectory(trajectory)
            gt_action = torch.tensor([[gt_acc, gt_steer]], dtype=torch.float32, device=self.device)
            
            # Get current direction and speed
            if len(trajectory) >= 2:
                recent_pos = trajectory.get_recent(2)
                gt_velocity = recent_pos[1][:2] - recent_pos[0][:2]
                current_speed = np.linalg.norm(gt_velocity)
                
                if current_speed > 1e-6:
                    current_direction_norm = gt_velocity / current_speed
                else:
                    current_direction_norm = np.array([0.0, 1.0])
                
                # Calculate GT next direction
                cos_gt, sin_gt = np.cos(gt_steer), np.sin(gt_steer)
                gt_direction = np.array([
                    current_direction_norm[0] * cos_gt - current_direction_norm[1] * sin_gt,
                    current_direction_norm[0] * sin_gt + current_direction_norm[1] * cos_gt
                ])
                
                gt_speed = max(0.0, current_speed + gt_acc)
            else:
                gt_direction = np.array([0.0, 1.0])
                gt_speed = 0.0
            
            # Encode each camera
            encoded_representations = []
            for cam_idx in range(self.trainer.num_cameras):
                cam_obs = observation[:, cam_idx]
                mu, logvar = self.trainer.encoders[cam_idx](cam_obs)
                encoded = self.trainer.latent_model(mu, logvar)
                encoded_representations.append(encoded)
            
            # Fuse camera representations
            representation_tensor = torch.stack(encoded_representations, dim=1)
            fused_representation = self.trainer.representation_fusion(representation_tensor)
            
            # Update recurrent state
            recurrent_state = self.trainer.recurrent_model(
                recurrent_state.detach(), 
                fused_representation, 
                prev_action
            )
            
            # Full state
            full_state = torch.cat([recurrent_state, fused_representation], dim=-1)
            
            # Predict action
            pred_acc_mean, pred_acc_std, pred_steer_mean, pred_steer_std = \
                self.trainer.action_predictor(full_state)
            
            # Compute predicted next position and reward
            pred_pos, pred_direction, pred_speed = predict_next_position(
                trajectory, 
                pred_acc_mean.item(), 
                pred_steer_mean.item()
            )
            
            # Get ground truth next position and compute reward
            if len(trajectory) >= 1:
                gt_next_pos = trajectory.get_recent(1)[0][:2]
                reward = compute_reward(pred_pos, gt_next_pos)
            else:
                reward = 0.0
                gt_next_pos = np.array([0.0, 0.0])
            
            # Track metrics
            rewards.append(reward)
            pred_actions.append([pred_acc_mean.item(), pred_steer_mean.item()])
            gt_actions.append([gt_acc, gt_steer])
            
            # Update prev_action
            prev_action = torch.tensor([[pred_acc_mean.item(), pred_steer_mean.item()]], 
                                      dtype=torch.float32, device=self.device)
            
            progress_bar.set_postfix({
                "Reward": f"{reward:.3f}",
                "Acc": f"{pred_acc_mean.item():.3f}",
                "Steer": f"{math.degrees(pred_steer_mean.item()):.1f}°"
            })
            
            # Visualization
            if visualize and self.renderer is not None:
                camera_images = self._prepare_camera_images_for_render(frame_data)
                
                render_data = {
                    'camera_images': camera_images,
                    'pose': pose_array,
                    'entities': frame_data.get("entities", []),
                    'split_name': 'inference',
                    'training_data': {
                        'epoch': 'Inference',
                        'recon_loss': 0.0,
                        'action_loss': 0.0,
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
                
                # Save figure
                if save_figures:
                    self._save_figure(idx)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("Inference Summary")
        print("="*60)
        print(f"Total frames processed: {num_frames}")
        print(f"Average reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
        
        pred_actions = np.array(pred_actions)
        gt_actions = np.array(gt_actions)
        
        print(f"\nAcceleration:")
        print(f"    Predicted: {pred_actions[:, 0].mean():.4f} ± {pred_actions[:, 0].std():.4f}")
        print(f"    Ground Truth: {gt_actions[:, 0].mean():.4f} ± {gt_actions[:, 0].std():.4f}")
        print(f"    MAE: {np.abs(pred_actions[:, 0] - gt_actions[:, 0]).mean():.4f}")
        
        print(f"\nSteering (degrees):")
        print(f"    Predicted: {np.degrees(pred_actions[:, 1]).mean():.2f}° ± {np.degrees(pred_actions[:, 1]).std():.2f}°")
        print(f"    Ground Truth: {np.degrees(gt_actions[:, 1]).mean():.2f}° ± {np.degrees(gt_actions[:, 1]).std():.2f}°")
        print(f"    MAE: {np.degrees(np.abs(pred_actions[:, 1] - gt_actions[:, 1])).mean():.2f}°")
        
        if save_figures:
            print(f"\nSaved {num_frames} figures to {self.output_dir}")
        
        # Clean up
        if visualize and self.renderer is not None:
            self.renderer.close()
        
        return {
            'rewards': rewards,
            'pred_actions': pred_actions,
            'gt_actions': gt_actions
        }


def main():
    # Configuration
    config = Config()
    print(f"Using device: {config.device}")
    
    print(f"\n{'='*60}")
    print(f"Dreamer Model Inference")
    print(f"{'='*60}\n")
    
    # Checkpoint path
    checkpoint_path = "dreamer_checkpoint.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using trainer.py")
        return
    
    # Load dataset
    dataset = ONCEDataset(
        data_path=args.data_path,
        split="val",  # Use validation split for inference
        data_type="camera",
        level="frame",
        logger_name="DreamerInference",
        show_logs=True
    )
    
    # Create inference engine
    inference = DreamerInference(checkpoint_path, config)
    
    print("\nRunning inference with visualization...")
    print("Figures will be saved to 'inference_outputs/' directory")
    print("Press 'q' in the visualization window to quit.\n")
    
    # Run inference (process first 1000 frames as example)
    results = inference.run_inference(
        dataset, 
        max_frames=1000,
        save_figures=True
    )


if __name__ == "__main__":
    main()