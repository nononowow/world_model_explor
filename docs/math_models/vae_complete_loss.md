# VAE Complete Loss Function

## Overview

The complete VAE loss function combines reconstruction loss and KL divergence loss. Understanding this combined objective is crucial for training effective VAEs.

## Complete Loss Formula

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

where:
- `L_recon`: Reconstruction loss (negative log-likelihood)
- `L_KL`: KL divergence loss (regularization)
- `β`: Weighting factor (typically 1.0, or varied in β-VAE)

## Detailed Formulation

### Standard VAE (β = 1)

$$\mathcal{L} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + \text{KL}(q(z|x) || p(z))$$

### β-VAE (β ≠ 1)

$$\mathcal{L} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + \beta \cdot \text{KL}(q(z|x) || p(z))$$

## Component Breakdown

### 1. Reconstruction Term

$$-\mathbb{E}_{q(z|x)}[\log p(x|z)]$$

- **Purpose**: Maximize likelihood of data under decoder
- **Effect**: Encourages good reconstructions
- **Computation**: Sample `z ~ q(z|x)`, compute `log p(x|z)`

### 2. KL Divergence Term

$$\text{KL}(q(z|x) || p(z))$$

- **Purpose**: Regularize latent distribution
- **Effect**: Keeps `q(z|x)` close to prior `p(z)`
- **Computation**: Analytical for Gaussian distributions

## Trade-off

The two terms create a trade-off:

- **High reconstruction, low KL**: Model focuses on reconstruction, latent space may be unstructured
- **Low reconstruction, high KL**: Model has structured latent space but poor reconstructions
- **Balanced**: Good reconstructions with structured latent space

## β Parameter

### β = 1 (Standard VAE)

- Balanced trade-off
- Good for general use
- May have entangled representations

### β > 1 (β-VAE)

- Stronger regularization
- More disentangled representations
- May sacrifice reconstruction quality
- Common values: 2, 4, 8

### β < 1

- Weaker regularization
- Better reconstructions
- Less structured latent space
- Rarely used

## Implementation

### Basic Implementation

```python
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Compute complete VAE loss.
    
    Args:
        recon_x: Reconstructed data
        x: Original data
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL term
    """
    # Reconstruction loss (BCE for images)
    recon_loss = F.binary_cross_entropy(
        recon_x, x, reduction='sum'
    )
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
```

### With Different Reconstruction Losses

```python
def vae_loss_flexible(recon_x, x, mu, logvar, beta=1.0, 
                     recon_loss_type='bce'):
    """Flexible VAE loss with different reconstruction losses."""
    
    # Reconstruction loss
    if recon_loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(
            recon_x, x, reduction='sum'
        )
    elif recon_loss_type == 'mse':
        recon_loss = F.mse_loss(
            recon_x, x, reduction='sum'
        )
    else:
        raise ValueError(f"Unknown loss type: {recon_loss_type}")
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
```

## Training Dynamics

### Early Training

- Reconstruction loss dominates (large)
- KL loss is small (latent space not yet structured)
- Model learns to reconstruct

### Mid Training

- Both terms contribute
- Latent space starts to structure
- Reconstructions improve

### Late Training

- KL loss stabilizes
- Reconstruction loss decreases
- Balance between terms

## Monitoring

### Metrics to Track

1. **Total Loss**: Overall objective
2. **Reconstruction Loss**: Reconstruction quality
3. **KL Loss**: Latent space structure
4. **Ratio**: `KL / Recon` (should be reasonable, e.g., 0.1-1.0)

### Visual Inspection

- **Reconstructions**: Should be clear and accurate
- **Latent Space**: Should be structured (visualize with t-SNE)
- **Samples**: Generated samples should be diverse

## Common Issues

### 1. KL Collapse

**Symptom**: KL loss → 0, poor reconstructions
**Cause**: Model ignores latent code
**Solution**: Increase β, check encoder gradients

### 2. Reconstruction Collapse

**Symptom**: Poor reconstructions, high KL
**Cause**: β too large, over-regularization
**Solution**: Decrease β, check decoder capacity

### 3. Unbalanced Losses

**Symptom**: One term dominates
**Solution**: Adjust β, normalize terms, or use annealing

## Loss Annealing

Gradually increase β during training:

$$\beta(t) = \min(1.0, \frac{t}{T})$$

where `t` is training step and `T` is annealing steps.

This helps:
- Start with reconstruction focus
- Gradually add regularization
- Avoid KL collapse

## Normalization

Sometimes losses are normalized:

$$\mathcal{L} = \frac{\mathcal{L}_{\text{recon}}}{N} + \beta \cdot \frac{\mathcal{L}_{\text{KL}}}{D}$$

where:
- `N`: Number of data dimensions
- `D`: Latent dimension

This makes the terms more comparable.

## Free Bits

To prevent KL collapse, use "free bits":

$$\mathcal{L}_{\text{KL}} = \max(\text{KL}(q||p), \lambda)$$

where `λ` is a threshold. This ensures KL doesn't go below a minimum.

## References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.
- Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR.

