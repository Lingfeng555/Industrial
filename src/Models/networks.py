import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough, Beta
from torch.distributions.utils import probs_to_logits
from src.Models.utils import sequentialModel1D


class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize, latentSize, actionSize, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)()

        self.linear = nn.Linear(latentSize + actionSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize)

    def forward(self, recurrentState, latentState, action):
        x = torch.cat((latentState, action), -1)
        h = self.activation(self.linear(x))
        return self.recurrent(h, recurrentState)

class VAESampler(nn.Module):
    def __init__(self, inputSize, latentSize, config):
        super().__init__()
        self.latentSize = latentSize
        self.encoder = VAEConvEncoder(inputSize[0], latentSize, config)
        self.decoder = VAEConvDecoder(latentSize, inputSize[0], config)

    def forward(self, x):
        mean, std = self.encoder(x)
        distribution = torch.distributions.Normal(mean, std)
        sample = distribution.rsample()
        reconstruction = self.decoder(sample)

        return reconstruction, distribution

class PriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits
    

class PosteriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class RewardModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        layers = []
        curr = inputSize
        for _ in range(self.config.numLayers):
            layers.append(nn.Linear(curr, self.config.hiddenSize))
            layers.append(getattr(nn, self.config.activation)())
            curr = self.config.hiddenSize
        layers.append(nn.Linear(curr, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 1, self.config.activation)

    def forward(self, x):
        return Bernoulli(logits=self.network(x).squeeze(-1))

class LatentDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar)
        z = mu + eps*sigma

        return z

class VAEConvEncoder(nn.Module):
    def __init__(self, inputShape, outputSize, config):
        super().__init__()
        self.config = config
        activation = getattr(nn, self.config.activation)()
        channels, height, width = inputShape
        self.outputSize = outputSize

        self.convolutionalNet = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            activation,
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            activation,
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            activation,
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            activation,
            nn.AdaptiveAvgPool2d((1, 1))  # Output: (batch, 256, 1, 1)
        )
        
        self.fc = nn.Linear(256, outputSize)
        self.mu = nn.Linear(outputSize, outputSize)
        self.logvar = nn.Linear(outputSize, outputSize)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.convolutionalNet(x)  # (batch, 256, 1, 1)
        x = x.view(batch_size, -1)    # (batch, 256)
        x = self.fc(x)                # (batch, outputSize)

        mu = self.mu(x)
        logvar = self.logvar(x)
        
        return mu, logvar
    


class VAEConvDecoder(nn.Module):
    def __init__(self, inputSize, outputShape, config):
        super().__init__()
        self.config = config
        self.channels, self.height, self.width = outputShape
        activation = getattr(nn, self.config.activation)()
        
        # Project latent vector to spatial feature maps
        self.fc = nn.Linear(inputSize, 256 * 4 * 4)
        
        self.network = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=0),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0)
        )

    def forward(self, x):
        # Reshape from (batch, latent_dim) -> (batch, 256, 4, 4)
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        
        output = self.network(x)
        # Resize to match target dimensions if needed
        if output.shape[2] != self.height or output.shape[3] != self.width:
            output = torch.nn.functional.interpolate(output, size=(self.height, self.width), mode='bilinear', align_corners=False)
        return output
    
class CNNRepresentationFusion(nn.Module):
    def __init__(self, inputShape, outputSize):
        super().__init__()
        num_cameras, latent_dim = inputShape
        
        # Treat each camera's latent as a channel
        self.conv_net = nn.Sequential(
            nn.Conv1d(num_cameras, 32, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Linear(64, outputSize)

    def forward(self, x):
        # x shape: (batch, num_cameras, latent_dim)
        # Conv1d expects: (batch, channels, sequence_length)
        # So x is already in the right format with num_cameras as channels
        x = self.conv_net(x)  # Output: (batch, 64, 1)
        x = x.squeeze(-1)      # Output: (batch, 64)
        x = self.fc(x)         # Output: (batch, outputSize)
        return x


class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, actionLow, actionHigh, device, config):
        super().__init__()
        actionSize *= 2
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, actionSize, self.config.activation)
        self.register_buffer("actionScale", ((torch.tensor(actionHigh, device=device) - torch.tensor(actionLow, device=device)) / 2.0))
        self.register_buffer("actionBias", ((torch.tensor(actionHigh, device=device) + torch.tensor(actionLow, device=device)) / 2.0))

    def forward(self, x, training=False):
        logStdMin, logStdMax = -5, 2
        mean, logStd = self.network(x).chunk(2, dim=-1)
        logStd = logStdMin + (logStdMax - logStdMin)/2*(torch.tanh(logStd) + 1) # (-1, 1) to (min, max)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh*self.actionScale + self.actionBias
        if training:
            logprobs = distribution.log_prob(sample)
            logprobs -= torch.log(self.actionScale*(1 - sampleTanh.pow(2)) + 1e-6)
            entropy = distribution.entropy()
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action


class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))


class ActionPredictor(nn.Module):
    """Predicts acceleration and steering angle from latent state"""
    def __init__(self, inputSize, hiddenSize=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
        )
        
        # Acceleration head (continuous, can be negative for braking)
        self.acceleration_head = nn.Sequential(
            nn.Linear(hiddenSize, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # mean and log_std
        )
        
        # Steering angle head (continuous, bounded)
        self.steering_head = nn.Sequential(
            nn.Linear(hiddenSize, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # mean and log_std
        )
    
    def forward(self, x):
        features = self.network(x)
        
        # Acceleration prediction
        acc_params = self.acceleration_head(features)
        acc_mean, acc_logstd = acc_params.chunk(2, dim=-1)
        acc_std = torch.exp(acc_logstd.clamp(-10, 2))
        
        # Steering prediction
        steer_params = self.steering_head(features)
        steer_mean, steer_logstd = steer_params.chunk(2, dim=-1)
        steer_std = torch.exp(steer_logstd.clamp(-10, 2))
        
        return acc_mean.squeeze(-1), acc_std.squeeze(-1), steer_mean.squeeze(-1), steer_std.squeeze(-1)
