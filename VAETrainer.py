import os
import tqdm
from args import args

import torch
from torch import nn
import torch.optim as optim
import cv2
import numpy as np

from src.once_dataset import ONCEDataset
from src.Models.networks import CNNRepresentationFusion, VAEConvDecoder, VAEConvEncoder, LatentDynamicsModel

class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize, latentSize, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)()

        self.linear = nn.Linear(latentSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize)

    def forward(self, recurrentState, latentState):
        h = self.activation(self.linear(latentState))
        return self.recurrent(h, recurrentState)

class replay_buffer:
    def __init__(self, length):
        self.length = length
        self.buffer = []

    def add(self, item):
        if len(self.buffer) >= self.length:
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self, index):
        return self.buffer[index]
    
    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
    
    def __iter__(self):
        for item in self.buffer:
            yield item

    def __getitem__(self, index):
        return self.buffer[index]

class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        num_models = 7
        self.encoders = []
        for i in range(num_models):
            encoder = VAEConvEncoder(config.observationShape, config.encodedObsSize, config.encoder)
            self.encoders.append(encoder.to(self.device))
        
        self.decoders = []
        for i in range(num_models):
            decoder = VAEConvDecoder(config.encodedObsSize * 4, config.observationShape, config.decoder)
            self.decoders.append(decoder.to(self.device))
        
        self.representation_fusion = CNNRepresentationFusion(
            inputShape=(num_models, config.encodedObsSize),
            outputSize=config.encodedObsSize * 4
            ).to(self.device)

        self.latent_model = LatentDynamicsModel().to(self.device)

        self.recurrent_model = RecurrentModel(
            recurrentSize=config.encodedObsSize * 4,
            latentSize=config.encodedObsSize * 4,
            config=config.recurrent
            ).to(self.device)

        self.replay_buffer = replay_buffer(length=5)
        self.initial_state = nn.Parameter(torch.zeros(1, config.encodedObsSize * 4)).to(self.device)

        print("VAE Trainer initialized with 7 encoders and decoders.")

    def train(self, dataset: ONCEDataset, epochs = 10, visualize=False):
        loss_fn = nn.L1Loss()
        
        # Collect all parameters from encoders and decoders
        all_params = []
        for enc, dec in zip(self.encoders, self.decoders):
            all_params.extend(list(enc.parameters()))
            all_params.extend(list(dec.parameters()))
            all_params.extend(list(self.representation_fusion.parameters()))
            all_params.extend(list(self.latent_model.parameters()))
            all_params.extend(list(self.recurrent_model.parameters()))
        
        optimizer = optim.Adam(all_params, lr=1e-4)
        progress_bar = tqdm.tqdm(total=epochs * len(dataset), desc="Training VAE")

        for epoch in range(epochs):

            self.replay_buffer.clear()

            for idx in range(len(dataset)):
                frame_data = dataset.frame_by_index(idx)
                camera_data = frame_data.get("camera_data", {})
            
                # Get list of camera names and corresponding tensors
                camera_names = list(camera_data.keys())
                camera_tensors = [camera_data[cam]["image_tensor"] for cam in camera_names]

                loss = torch.tensor(0.0, device=self.device)
                reconstructed = []
                processed_originals = []
                encoded_representations = []
            
                # Process all cameras
                for cam_idx, camera_tensor in enumerate(camera_tensors):
                    camera_tensor = camera_tensor.unsqueeze(0).to(self.device)  # Already a tensor, add batch dim

                    # Resize to match config shape
                    camera_tensor = torch.nn.functional.interpolate(
                        camera_tensor, 
                        size=(self.config.observationShape[1], self.config.observationShape[2]),
                        mode='bilinear',
                        align_corners=False
                    )
                    processed_originals.append(camera_tensor)

                    # Encode and decode
                    mu, logvar = self.encoders[cam_idx](camera_tensor)
                    encoded = self.latent_model(mu, logvar)
                    encoded_representations.append(encoded)

                if encoded_representations == []:
                    continue
                    
                representation_tensor = torch.stack(encoded_representations, dim=1)
                fused_representation = self.representation_fusion(representation_tensor)
                self.replay_buffer.add(fused_representation)

                # Expand initial state to match batch size
                batch_size = fused_representation.shape[0]
                latent_representation = self.initial_state.expand(batch_size, -1).contiguous()

                # Process replay buffer: detach historical frames but keep current frame's gradients
                for frame_idx, frame in enumerate(self.replay_buffer):
                    # Detach historical frames to prevent backprop through them
                    if frame_idx < len(self.replay_buffer) - 1:
                        frame = frame.detach()
                    
                    latent_representation = self.recurrent_model(latent_representation, frame)

                # Decode the final predicted state
                for cam_idx in range(len(processed_originals)):
                    reconstructed_frame = self.decoders[cam_idx](latent_representation)
                    reconstructed.append(reconstructed_frame)
                    # Compute loss against processed (resized) originals
                    loss += loss_fn(reconstructed_frame[cam_idx:cam_idx+1], processed_originals[cam_idx])

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(1)
                # Visualization
                if visualize and idx % 10 == 0:  # Show every 10th frame
                    self._visualize(processed_originals, reconstructed, loss.item())
    
    def _visualize(self, originals, reconstructed, loss):
        """Show all cameras in 2 columns - originals on left, reconstructed on right"""
        
        # Process all cameras
        camera_comparisons = []
        for orig_tensor, recon_tensor in zip(originals, reconstructed):
            # Convert tensors to numpy arrays [H, W, C]
            orig = orig_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
            recon = recon_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
            
            # Clip to valid range and convert to uint8
            orig = np.clip(orig * 255, 0, 255).astype(np.uint8)
            recon = np.clip(recon * 255, 0, 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
            recon = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
            
            # Create side-by-side comparison for this camera
            h, w = orig.shape[:2]
            comparison = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
            comparison[:, :w] = orig
            comparison[:, w+10:] = recon
            
            camera_comparisons.append(comparison)
        
        # Arrange in 2 columns - split cameras into left and right columns
        num_cameras = len(camera_comparisons)
        rows_per_col = (num_cameras + 1) // 2  # Ceiling division
        
        # Split into two columns
        left_column = camera_comparisons[:rows_per_col]
        right_column = camera_comparisons[rows_per_col:]
        
        # Pad right column if needed
        if len(right_column) < len(left_column):
            # Create blank image matching the size
            h, w = left_column[0].shape[:2]
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            right_column.append(blank)
        
        # Stack each column vertically
        left_stack = np.vstack(left_column)
        right_stack = np.vstack(right_column)
        
        # Combine columns horizontally
        full_comparison = np.hstack([left_stack, right_stack])
        
        # Add text label at the top
        cv2.putText(full_comparison, f"Loss: {loss:.4f} | Press 'q' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('VAE Training - All Cameras (Original | Reconstructed)', full_comparison)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True

def main():   
    # Create a simple config object
    class Config:
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Use much smaller resolution to avoid huge linear layers
            self.observationShape = (3, 1920 // 10, 1020 // 10)
            self.encodedObsSize = 1024  # Reduced from 256
            
            # Encoder config - detailed channels/kernels per user
            self.encoder = type('obj', (object,), {
                'activation': 'Tanh',
                'depth': 16,
                'kernelSize': 4,
                'stride': 2,
                'convChannels': [32, 64, 128, 256],
                'convKernelSizes': [4, 4, 4, 4],
                'convStrides': [2, 2, 2, 2]
            })

            # Decoder config - detailed channels/kernels per user
            self.decoder = type('obj', (object,), {
                'activation': 'Tanh',
                'depth': 16,
                'kernelSize': 5,
                'stride': 2,
                'convChannels': [256, 128, 64, 32],
                'convKernelSizes': [5, 5, 6, 6],
                'convStrides': [2, 2, 2, 2],
                'inputHeight': 96,
                'inputWidth': 96
            })

            self.recurrent = type('obj', (object,), {
                'hiddenSize': 200,
                'activation': 'Tanh',
            })
    
    config = Config()
    print(f"Using device: {config.device}")
    
    # Train on training split only
    print(f"\n{'='*60}")
    print(f"Training VAE on train split")
    print(f"{'='*60}\n")
    
    dataset = ONCEDataset(
        data_path=args.data_path,
        split="train",
        data_type="camera",  # Only need camera data for VAE
        level="frame",
        logger_name=f"ONCEDataset_train",
        show_logs=True
    )
    
    trainer = VAETrainer(config)
    print("Starting training with visualization... Press 'q' in the window to quit.\n")
    trainer.train(dataset, epochs=5, visualize=True)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()