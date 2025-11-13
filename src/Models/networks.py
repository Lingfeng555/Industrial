import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough, Beta
from torch.distributions.utils import probs_to_logits
from utils import sequentialModel1D


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


class VAEConvEncoder(nn.Module):
    def __init__(self, inputShape, outputSize, config):
        super().__init__()
        self.config = config
        activation = getattr(nn, self.config.activation)()
        channels, height, width = inputShape
        self.outputSize = outputSize

        self.convolutionalNet = nn.Sequential(
            nn.Conv2d(channels,            self.config.depth*1, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*1, self.config.depth*2, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*2, self.config.depth*4, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*4, self.config.depth*8, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Flatten(),
            nn.Linear(self.config.depth*8*(height // (self.config.stride ** 4))*(width // (self.config.stride ** 4)), outputSize), activation)

    def forward(self, x):
        return self.convolutionalNet(x).view(-1, self.outputSize)


class VAEConvDecoder(nn.Module):
    def __init__(self, inputSize, outputShape, config):
        super().__init__()
        self.config = config
        self.channels, self.height, self.width = outputShape
        activation = getattr(nn, self.config.activation)()

        self.network = nn.Sequential(
            nn.Linear(inputSize, self.config.depth*32),
            nn.Unflatten(1, (self.config.depth*32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(self.config.depth*32, self.config.depth*4, self.config.kernelSize,     self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*4,  self.config.depth*2, self.config.kernelSize,     self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*2,  self.config.depth*1, self.config.kernelSize + 1, self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*1,  self.channels,       self.config.kernelSize + 1, self.config.stride))

    def forward(self, x):
        return self.network(x)


class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, actionLow, actionHigh, device, config):
        super().__init__()
        self.config = config
        self.action_dim = actionSize
        
        hidden_dim = config.get('actor_hidden_dim', 256)
        self.shared_net = nn.Sequential(
            nn.Linear(inputSize, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_alpha = nn.Linear(hidden_dim, actionSize)
        self.fc_beta = nn.Linear(hidden_dim, actionSize)
        self.sp = nn.Softplus()
        
        action_low_tensor = torch.tensor(actionLow, dtype=torch.float32, device=device)
        action_high_tensor = torch.tensor(actionHigh, dtype=torch.float32, device=device)
        self.register_buffer("actionScale", action_high_tensor - action_low_tensor)
        self.register_buffer("actionLow", action_low_tensor)
    
    def forward(self, x, training=False):
        features = self.shared_net(x)
        
        alpha = self.sp(self.fc_alpha(features)) + 1.0
        beta = self.sp(self.fc_beta(features)) + 1.0
        
        #this was a recommendation by chatgpt
        alpha = torch.clamp(alpha, min=1.0, max=15.0)
        beta = torch.clamp(beta, min=1.0, max=15.0)
        
        distribution = Beta(alpha, beta)
        
        if training:
            sample = distribution.rsample()
        else:
            sample = alpha / (alpha + beta)
            
        #this was also a recommendation by chatgpt
        sample = torch.clamp(sample, min=1e-6, max=1.0 - 1e-6)

        action = sample * self.actionScale + self.actionLow
        
        if training:
            logprobs = distribution.log_prob(sample)

            # this took me way too long to figure out
            # change of variables for Tanh squashing
            # see appendix C of https://arxiv.org/pdf/1801.01290.pdf

            # this is a change of variable that accounts for the previously done transformations
            # this is accounted for the division of the jacobian from the transformation
            # which in log space is a subtraction
            # specifically the tanh squashing, with a correction for the scale
            # I WANT TO KILL MYSELF
            # the +1e-6 is to prevent log(0)
            # sorry for the mess, but if you are interested, this is called the log jacobian transformation formula
            logprobs = logprobs - torch.log(self.actionScale + 1e-6)

            # final log-probability of the executed (squashed+scaled) action
            logprobs = logprobs.sum(dim=-1, keepdim=False)

            # entropy because we want to measue randomness in the original space
            # this is shannon's information theory entropy
            # We pass this to the loss function to either maximize or minimize it
            # how am I going to explain this to Asier ._.?
            entropy = distribution.entropy() + torch.log(self.actionScale + 1e-6)
            
            # we add the log jacobian to the entropy because, even though
            # they are independent, we will add them to the loss when optimizing
            # remember that the log_jacobian represents the expected change of rate
            entropy = entropy.sum(dim=-1, keepdim=False)
            
            return action, logprobs, entropy
        else:
            return action


class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config

        layers = []
        currentsize = inputSize
        for _ in range(self.config.numLayers):
            layers.append(nn.Linear(currentsize, self.config.hiddenSize))
            layers.append(getattr(nn, self.config.activation)())
            currentsize = self.config.hiddenSize
        layers.append(nn.Linear(currentsize, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))
