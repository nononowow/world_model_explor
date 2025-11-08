# World Model Architecture

## Overview

This project implements world models following the architecture proposed in "World Models" (Ha & Schmidhuber, 2018) and extends it for autonomous driving applications.

## Core Components

### 1. Variational Autoencoder (VAE)

The VAE encodes high-dimensional observations (images) into a low-dimensional latent space.

**Architecture:**
- Encoder: Convolutional layers → Latent distribution (μ, σ)
- Decoder: Latent sample → Reconstructed observation
- Loss: Reconstruction loss + KL divergence

**Files:**
- `src/vae/basic_vae.py`: Basic VAE implementation
- `src/vae/improved_vae.py`: β-VAE and ResNet-based variants
- `src/world_model/vae_model.py`: World Models VAE
- `src/driving/driving_vae.py`: Multi-modal VAE for driving

### 2. Dynamics Model (MDN-RNN)

The RNN predicts future latent states given current latent state and action.

**Architecture:**
- LSTM layers
- Mixture Density Network output (multiple Gaussians)
- Captures uncertainty in predictions

**Files:**
- `src/rnn/mdn_rnn.py`: MDN-RNN implementation
- `src/rnn/transformer_dynamics.py`: Transformer-based alternative
- `src/world_model/rnn_model.py`: World Models RNN
- `src/driving/driving_dynamics.py`: Driving dynamics model

### 3. Controller

The controller maps latent states to actions.

**Architecture:**
- Feedforward neural network
- Input: Latent state (optionally + RNN hidden state)
- Output: Action

**Files:**
- `src/controller/simple_controller.py`: Basic controller
- `src/world_model/controller.py`: World Models controller
- `src/driving/driving_controller.py`: Driving controller

## Training Pipeline

### Phase 1: Data Collection
- Collect random rollouts from environment
- Store observations, actions, rewards

### Phase 2: VAE Training
- Train VAE to encode observations
- Learn efficient latent representations

### Phase 3: RNN Training
- Encode collected observations to latent space
- Train RNN to predict next latent state given current state and action

### Phase 4: Controller Training
- Train controller in latent space
- Can use evolutionary strategies or gradient-based methods
- Evaluate using world model (no environment interaction needed)

## Autonomous Driving Extensions

### Multi-Modal Inputs
- Camera images (vision)
- Vehicle state (speed, steering angle, etc.)

### Safety Features
- Uncertainty quantification
- Safety constraints in planning
- Conservative actions when uncertain

### Planning
- Model Predictive Control (MPC)
- Cross Entropy Method (CEM)
- Uncertainty-aware planning

## File Structure

```
src/
├── vae/              # VAE implementations
├── rnn/              # Dynamics models
├── controller/       # Policy networks
├── world_model/      # Classic World Models
├── planning/         # Planning algorithms
├── driving/          # Driving-specific code
├── environments/     # Environment wrappers
└── utils/            # Utilities
```

## References

- Ha, D., & Schmidhuber, J. (2018). World Models. arXiv:1803.10122
- Hafner, D., et al. (2019). Dreamer: Learning Models by Imagination. arXiv:1912.01603
- Hafner, D., et al. (2019). Learning Latent Dynamics for Planning from Pixels. arXiv:1811.04551

