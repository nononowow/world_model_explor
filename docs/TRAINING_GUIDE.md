# Training Guide

## Quick Start

### 1. Train Basic VAE

```bash
python scripts/train_vae.py --dataset mnist --epochs 50
```

### 2. Train World Model

```bash
python scripts/train_world_model.py --env CarRacing-v0 --collect_episodes 1000
```

### 3. Train Driving Model

```bash
python scripts/train_driving_model.py --env SimpleDrivingEnv
```

## Detailed Training

### VAE Training

The VAE learns to compress observations into a latent space.

**Key Parameters:**
- `latent_dim`: Dimension of latent space (default: 32)
- `beta`: Weight for KL divergence (default: 1.0, higher for more disentanglement)
- `lr`: Learning rate (default: 1e-3)

**Tips:**
- Start with smaller latent dimensions for faster training
- Increase beta gradually for better disentanglement
- Monitor reconstruction quality and KL divergence

### RNN Training

The RNN learns to predict future latent states.

**Key Parameters:**
- `hidden_dim`: LSTM hidden dimension (default: 256)
- `num_mixtures`: Number of Gaussian mixtures (default: 5)
- `sequence_length`: Length of training sequences (default: 16)

**Tips:**
- Longer sequences help learn long-term dependencies
- More mixtures capture more complex uncertainty
- Use gradient clipping to stabilize training

### Controller Training

The controller learns to act in latent space.

**Methods:**
1. **Evolutionary Strategies**: Population-based optimization
   - `population_size`: Number of candidates (default: 64)
   - `num_generations`: Number of iterations (default: 20)

2. **Gradient-Based**: Policy gradients (not fully implemented)

**Tips:**
- Evolutionary strategies are more robust but slower
- Start with smaller population for faster iteration
- Evaluate on world model (fast) before testing on real environment

## Hyperparameter Tuning

### VAE
- **latent_dim**: 16-64 (larger = more capacity, slower)
- **beta**: 0.1-10.0 (higher = more disentangled)
- **learning_rate**: 1e-4 to 1e-3

### RNN
- **hidden_dim**: 128-512
- **num_mixtures**: 3-10
- **sequence_length**: 8-32

### Controller
- **hidden_dims**: [256, 256] or [256, 256, 128]
- **population_size**: 32-128
- **num_generations**: 10-50

## Monitoring Training

### VAE Metrics
- Reconstruction loss (should decrease)
- KL divergence (should stabilize)
- Visual inspection of reconstructions

### RNN Metrics
- Prediction loss (should decrease)
- Uncertainty (variance of predictions)
- Visual inspection of predicted sequences

### Controller Metrics
- Episode reward (should increase)
- Episode length
- Success rate

## Common Issues

### VAE not reconstructing well
- Increase model capacity (more layers/channels)
- Decrease beta (less regularization)
- Check data preprocessing (normalization)

### RNN predictions are poor
- Increase sequence length
- More training data
- Check latent space quality (VAE must be good first)

### Controller not learning
- Ensure VAE and RNN are well-trained first
- Increase population size or generations
- Check reward function

## Experiment Tracking

Use TensorBoard or Weights & Biases to track training:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/VAE', loss, epoch)
```

