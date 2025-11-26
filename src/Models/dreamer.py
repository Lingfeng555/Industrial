import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough, Normal
import numpy as np
import os

from networks import RecurrentModel, PriorNet, PosteriorNet, RewardModel, ContinueModel, VAEConvEncoder, VAEConvDecoder, CNNRepresentationFusion, Actor, Critic
from utils import computeLambdaValues, Moments
from buffer import ReplayBuffer
import imageio


class Dreamer:
    def __init__(self, observationShape, actionSize, actionLow, actionHigh, device, config):
        self.observationShape   = observationShape
        self.actionSize         = actionSize
        self.config             = config
        self.device             = device

        self.recurrentSize  = config.recurrentSize
        self.latentSize     = config.latentLength*config.latentClasses
        self.fullStateSize  = config.recurrentSize + self.latentSize

        self.numCameras      = 7
        self.fusedSize       = config.encodedObsSize * 4  # Output size from CNNRepresentationFusion
        
        self.actor           = Actor(self.fullStateSize, actionSize, actionLow, actionHigh, device,                                  config.actor          ).to(self.device)
        self.critic          = Critic(self.fullStateSize,                                                                            config.critic         ).to(self.device)
        
        # Create separate encoder/decoder for each camera (using list comprehension, NOT multiplication)
        self.encoder         = nn.ModuleList([
            VAEConvEncoder(observationShape, self.config.encodedObsSize, config.encoder).to(self.device) 
            for _ in range(self.numCameras)
        ])
        self.decoder         = nn.ModuleList([
            VAEConvDecoder(self.fullStateSize, observationShape, config.decoder).to(self.device) 
            for _ in range(self.numCameras)
        ])
        
        # Fusion network: takes (numCameras, encodedObsSize) input, outputs fusedSize
        self.representationfusion = CNNRepresentationFusion(
            inputShape=(self.numCameras, self.config.encodedObsSize), 
            outputSize=self.fusedSize
        ).to(self.device)
        
        self.recurrentModel  = RecurrentModel(config.recurrentSize, self.latentSize, actionSize,                                     config.recurrentModel ).to(self.device)
        self.priorNet        = PriorNet(config.recurrentSize, config.latentLength, config.latentClasses,                             config.priorNet       ).to(self.device)
        # PosteriorNet needs to take fusedSize instead of encodedObsSize for multi-camera input
        self.posteriorNet    = PosteriorNet(config.recurrentSize + self.fusedSize, config.latentLength, config.latentClasses,        config.posteriorNet   ).to(self.device)
        self.rewardPredictor = RewardModel(self.fullStateSize,                                                                       config.reward         ).to(self.device)
        if config.useContinuationPrediction:
            self.continuePredictor  = ContinueModel(self.fullStateSize,                                                              config.continuation   ).to(self.device)

        self.buffer         = ReplayBuffer(observationShape, actionSize, config.buffer, device)
        self.valueMoments   = Moments(device)

        # Collect world model parameters from all encoders and decoders
        self.worldModelParameters = []
        for enc in self.encoder:
            self.worldModelParameters.extend(enc.parameters())
        for dec in self.decoder:
            self.worldModelParameters.extend(dec.parameters())
        self.worldModelParameters.extend(self.representationfusion.parameters())
        self.worldModelParameters.extend(self.recurrentModel.parameters())
        self.worldModelParameters.extend(self.priorNet.parameters())
        self.worldModelParameters.extend(self.posteriorNet.parameters())
        self.worldModelParameters.extend(self.rewardPredictor.parameters())
        if self.config.useContinuationPrediction:
            self.worldModelParameters += list(self.continuePredictor.parameters())

        self.worldModelOptimizer    = torch.optim.Adam(self.worldModelParameters,   lr=self.config.worldModelLR)
        self.actorOptimizer         = torch.optim.Adam(self.actor.parameters(),     lr=self.config.actorLR)
        self.criticOptimizer        = torch.optim.Adam(self.critic.parameters(),    lr=self.config.criticLR)

        self.totalEpisodes      = 0
        self.totalEnvSteps      = 0
        self.totalGradientSteps = 0


    def worldModelTraining(self, data):
        # data.observations shape: (batchSize, batchLength, numCameras, *observationShape)
        # Encode each camera separately and then fuse
        batchSize = self.config.batchSize
        batchLength = self.config.batchLength
        
        # Encode all cameras for all timesteps
        encoded_all_cameras = []
        for cam_idx in range(self.numCameras):
            # Extract observations for this camera: (batchSize, batchLength, C, H, W)
            cam_obs = data.observations[:, :, cam_idx]
            # Encode: flatten batch and time, then reshape
            encoded = self.encoder[cam_idx](cam_obs.view(-1, *self.observationShape))
            encoded = encoded.view(batchSize, batchLength, -1)
            encoded_all_cameras.append(encoded)
        
        # Stack cameras: (batchSize, batchLength, numCameras, encodedObsSize)
        encoded_stacked = torch.stack(encoded_all_cameras, dim=2)
        
        # Fuse camera representations for each timestep
        fusedObservations = []
        for t in range(batchLength):
            # (batchSize, numCameras, encodedObsSize)
            fused = self.representationfusion(encoded_stacked[:, t])
            fusedObservations.append(fused)
        fusedObservations = torch.stack(fusedObservations, dim=1)  # (batchSize, batchLength, fusedSize)
        
        previousRecurrentState  = torch.zeros(batchSize, self.recurrentSize, device=self.device)
        previousLatentState     = torch.zeros(batchSize, self.latentSize, device=self.device)

        recurrentStates, priorsLogits, posteriors, posteriorsLogits = [], [], [], []
        for t in range(1, self.config.batchLength):
            recurrentState              = self.recurrentModel(previousRecurrentState, previousLatentState, data.actions[:, t-1])
            _, priorLogits              = self.priorNet(recurrentState)
            posterior, posteriorLogits  = self.posteriorNet(torch.cat((recurrentState, fusedObservations[:, t]), -1))

            recurrentStates.append(recurrentState)
            priorsLogits.append(priorLogits)
            posteriors.append(posterior)
            posteriorsLogits.append(posteriorLogits)

            previousRecurrentState = recurrentState
            previousLatentState    = posterior

        recurrentStates             = torch.stack(recurrentStates,              dim=1) # (batchSize, batchLength-1, recurrentSize)
        priorsLogits                = torch.stack(priorsLogits,                 dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        posteriors                  = torch.stack(posteriors,                   dim=1) # (batchSize, batchLength-1, latentLength*latentClasses)
        posteriorsLogits            = torch.stack(posteriorsLogits,             dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        fullStates                  = torch.cat((recurrentStates, posteriors), dim=-1) # (batchSize, batchLength-1, recurrentSize + latentLength*latentClasses)

        # Reconstruct each camera separately
        reconstructionLoss = torch.tensor(0.0, device=self.device)
        for cam_idx in range(self.numCameras):
            reconstructionMeans = self.decoder[cam_idx](fullStates.view(-1, self.fullStateSize)).view(
                batchSize, self.config.batchLength-1, *self.observationShape
            )
            reconstructionDistribution = Independent(Normal(reconstructionMeans, 1), len(self.observationShape))
            # data.observations shape: (batchSize, batchLength, numCameras, C, H, W)
            reconstructionLoss += -reconstructionDistribution.log_prob(data.observations[:, 1:, cam_idx]).mean()

        rewardDistribution  =  self.rewardPredictor(fullStates)
        rewardLoss          = -rewardDistribution.log_prob(data.rewards[:, 1:].squeeze(-1)).mean()

        priorDistribution       = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits              ), 1)
        priorDistributionSG     = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits.detach()     ), 1)
        posteriorDistribution   = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits          ), 1)
        posteriorDistributionSG = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits.detach() ), 1)

        priorLoss       = kl_divergence(posteriorDistributionSG, priorDistribution  )
        posteriorLoss   = kl_divergence(posteriorDistribution  , priorDistributionSG)
        freeNats        = torch.full_like(priorLoss, self.config.freeNats)

        priorLoss       = self.config.betaPrior*torch.maximum(priorLoss, freeNats)
        posteriorLoss   = self.config.betaPosterior*torch.maximum(posteriorLoss, freeNats)
        klLoss          = (priorLoss + posteriorLoss).mean()

        worldModelLoss =  reconstructionLoss + rewardLoss + klLoss # I think that the reconstruction loss is relatively a bit too high (11k) 
        
        if self.config.useContinuationPrediction:
            continueDistribution = self.continuePredictor(fullStates)
            continueLoss         = nn.BCELoss(continueDistribution.probs, 1 - data.dones[:, 1:])
            worldModelLoss      += continueLoss.mean()

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.worldModelOptimizer.step()

        klLossShiftForGraphing = (self.config.betaPrior + self.config.betaPosterior)*self.config.freeNats
        metrics = {
            "worldModelLoss"        : worldModelLoss.item() - klLossShiftForGraphing,
            "reconstructionLoss"    : reconstructionLoss.item(),
            "rewardPredictorLoss"   : rewardLoss.item(),
            "klLoss"                : klLoss.item() - klLossShiftForGraphing}
        return fullStates.view(-1, self.fullStateSize).detach(), metrics


    def behaviorTraining(self, fullState):
        recurrentState, latentState = torch.split(fullState, (self.recurrentSize, self.latentSize), -1)
        fullStates, logprobs, entropies = [], [], []
        for _ in range(self.config.imaginationHorizon):
            action, logprob, entropy = self.actor(fullState.detach(), training=True)
            recurrentState = self.recurrentModel(recurrentState, latentState, action)
            latentState, _ = self.priorNet(recurrentState)

            fullState = torch.cat((recurrentState, latentState), -1)
            fullStates.append(fullState)
            logprobs.append(logprob)
            entropies.append(entropy)
        fullStates  = torch.stack(fullStates,    dim=1) # (batchSize*batchLength, imaginationHorizon, recurrentSize + latentLength*latentClasses)
        logprobs    = torch.stack(logprobs[1:],  dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        entropies   = torch.stack(entropies[1:], dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        
        predictedRewards = self.rewardPredictor(fullStates[:, :-1]).mean
        values           = self.critic(fullStates).mean
        continues        = self.continuePredictor(fullStates).mean if self.config.useContinuationPrediction else torch.full_like(predictedRewards, self.config.discount)
        lambdaValues     = computeLambdaValues(predictedRewards, values, continues, self.config.lambda_)

        _, inverseScale = self.valueMoments(lambdaValues)
        advantages      = (lambdaValues - values[:, :-1])/inverseScale

        actorLoss = -torch.mean(advantages.detach()*logprobs + self.config.entropyScale*entropies)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.actorOptimizer.step()

        valueDistributions  =  self.critic(fullStates[:, :-1].detach())
        criticLoss          = -torch.mean(valueDistributions.log_prob(lambdaValues.detach()))

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.criticOptimizer.step()

        metrics = {
            "actorLoss"     : actorLoss.item(),
            "criticLoss"    : criticLoss.item(),
            "entropies"     : entropies.mean().item(),
            "logprobs"      : logprobs.mean().item(),
            "advantages"    : advantages.mean().item(),
            "criticValues"  : values.mean().item()}
        return metrics


    @torch.no_grad()
    def environmentInteraction(self, env, numEpisodes, seed=None, evaluation=False, saveVideo=False, filename="videos/unnamedVideo", fps=30, macroBlockSize=16):
        scores = []
        for i in range(numEpisodes):
            recurrentState, latentState = torch.zeros(1, self.recurrentSize, device=self.device), torch.zeros(1, self.latentSize, device=self.device)
            action = torch.zeros(1, self.actionSize).to(self.device)

            observation = env.reset(seed= (seed + self.totalEpisodes if seed else None))

            # Encode each camera with its corresponding encoder
            encoded_cameras = []
            for cam_idx, camera in enumerate(observation):
                if cam_idx >= self.numCameras:
                    break
                encoded = self.encoder[cam_idx](torch.from_numpy(camera).float().unsqueeze(0).to(self.device))
                encoded_cameras.append(encoded)

            # Stack and fuse: (1, numCameras, encodedObsSize)
            encoded_stacked = torch.stack(encoded_cameras, dim=1)
            encoded_observation = self.representationfusion(encoded_stacked).squeeze(0)

            currentScore, stepCount, done, frames = 0, 0, False, []
            while not done:
                recurrentState      = self.recurrentModel(recurrentState, latentState, action)
                latentState, _      = self.posteriorNet(torch.cat((recurrentState, encoded_observation.view(1, -1)), -1))

                action          = self.actor(torch.cat((recurrentState, latentState), -1))
                actionNumpy     = action.cpu().numpy().reshape(-1)

                nextObservation, reward, done = env.step(actionNumpy)
                if not evaluation:
                    self.buffer.add(observation, actionNumpy, reward, nextObservation, done)

                if saveVideo and i == 0:
                    frame = env.render()
                    targetHeight = (frame.shape[0] + macroBlockSize - 1)//macroBlockSize*macroBlockSize # getting rid of imagio warning
                    targetWidth = (frame.shape[1] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
                    frames.append(np.pad(frame, ((0, targetHeight - frame.shape[0]), (0, targetWidth - frame.shape[1]), (0, 0)), mode='edge'))

                # Encode next observation from all cameras
                encoded_cameras = []
                for cam_idx, camera in enumerate(nextObservation):
                    if cam_idx >= self.numCameras:
                        break
                    encoded = self.encoder[cam_idx](torch.from_numpy(camera).float().unsqueeze(0).to(self.device))
                    encoded_cameras.append(encoded)

                encoded_stacked = torch.stack(encoded_cameras, dim=1)
                encoded_observation = self.representationfusion(encoded_stacked).squeeze(0)
                observation = nextObservation
                
                currentScore += reward
                stepCount += 1
                if done:
                    scores.append(currentScore)
                    if not evaluation:
                        self.totalEpisodes += 1
                        self.totalEnvSteps += stepCount

                    if saveVideo and i == 0:
                        finalFilename = f"{filename}_reward_{currentScore:.0f}.mp4"
                        with imageio.get_writer(finalFilename, fps=fps) as video:
                            for frame in frames:
                                video.append_data(frame)
                    break
        return sum(scores)/numEpisodes if numEpisodes else None
    

    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        checkpoint = {
            'encoder'               : [encoder.state_dict() for encoder in self.encoder],
            'decoder'               : [decoder.state_dict() for decoder in self.decoder],
            'representationfusion'  : self.representationfusion.state_dict(),
            'recurrentModel'        : self.recurrentModel.state_dict(),
            'priorNet'              : self.priorNet.state_dict(),
            'posteriorNet'          : self.posteriorNet.state_dict(),
            'rewardPredictor'       : self.rewardPredictor.state_dict(),
            'actor'                 : self.actor.state_dict(),
            'critic'                : self.critic.state_dict(),
            'worldModelOptimizer'   : self.worldModelOptimizer.state_dict(),
            'criticOptimizer'       : self.criticOptimizer.state_dict(),
            'actorOptimizer'        : self.actorOptimizer.state_dict(),
            'totalEpisodes'         : self.totalEpisodes,
            'totalEnvSteps'         : self.totalEnvSteps,
            'totalGradientSteps'    : self.totalGradientSteps}
        if self.config.useContinuationPrediction:
            checkpoint['continuePredictor'] = self.continuePredictor.state_dict()
        torch.save(checkpoint, checkpointPath)


    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpointPath}")
        
        checkpoint = torch.load(checkpointPath, map_location=self.device)
        
        # Load each encoder/decoder individually
        for idx, encoder_state in enumerate(checkpoint['encoder']):
            self.encoder[idx].load_state_dict(encoder_state)
        for idx, decoder_state in enumerate(checkpoint['decoder']):
            self.decoder[idx].load_state_dict(decoder_state)
        
        self.representationfusion.load_state_dict(checkpoint['representationfusion'])
        self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
        self.priorNet.load_state_dict(checkpoint['priorNet'])
        self.posteriorNet.load_state_dict(checkpoint['posteriorNet'])
        self.rewardPredictor.load_state_dict(checkpoint['rewardPredictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.worldModelOptimizer.load_state_dict(checkpoint['worldModelOptimizer'])
        self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
        self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
        self.totalEpisodes = checkpoint['totalEpisodes']
        self.totalEnvSteps = checkpoint['totalEnvSteps']
        self.totalGradientSteps = checkpoint['totalGradientSteps']
        if self.config.useContinuationPrediction:
            self.continuePredictor.load_state_dict(checkpoint['continuePredictor'])
