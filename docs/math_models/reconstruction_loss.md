# Reconstruction Loss

## Overview

The reconstruction loss measures how well a VAE can reconstruct the input data from the latent representation. It's the first term in the VAE objective function.

## General Form

The reconstruction loss is:

$$\mathcal{L}_{\text{recon}} = -\mathbb{E}_{q(z|x)}[\log p(x|z)]$$

This is the negative log-likelihood of the data under the decoder distribution.

## Interpretation

- **High likelihood** → Good reconstruction → Low loss
- **Low likelihood** → Poor reconstruction → High loss

The decoder `p_θ(x|z)` models the distribution of data given a latent code.

## Common Loss Functions

The choice of reconstruction loss depends on the data type and decoder output distribution.

### 1. Binary Cross-Entropy (BCE)

**For**: Binary data or normalized images (pixel values in [0,1])

**Distribution**: Bernoulli

$$p(x|z) = \prod_i x_i^{ŷ_i} (1-x_i)^{1-ŷ_i}$$

**Loss**:

$$\mathcal{L}_{\text{BCE}} = -\sum_i [x_i \log ŷ_i + (1-x_i) \log(1-ŷ_i)]$$

where `ŷ = decoder(z)` is the decoder output.

**Implementation**:
```python
recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
```

### 2. Mean Squared Error (MSE)

**For**: Continuous data, real-valued images

**Distribution**: Gaussian with fixed variance

$$p(x|z) = \mathcal{N}(x; μ_θ(z), σ^2 I)$$

**Loss** (assuming `σ² = 1`):

$$\mathcal{L}_{\text{MSE}} = \frac{1}{2} \sum_i (x_i - ŷ_i)^2$$

**Implementation**:
```python
recon_loss = F.mse_loss(recon_x, x, reduction='sum')
```

### 3. L1 Loss (Mean Absolute Error)

**For**: Robust to outliers, sparse reconstructions

**Loss**:

$$\mathcal{L}_{\text{L1}} = \sum_i |x_i - ŷ_i|$$

**Implementation**:
```python
recon_loss = F.l1_loss(recon_x, x, reduction='sum')
```

### 4. Gaussian with Learned Variance

**For**: When uncertainty in reconstruction matters

**Distribution**: `p(x|z) = N(μ_θ(z), σ_θ²(z))`

**Loss**:

$$\mathcal{L} = \frac{1}{2} \sum_i \left[ \frac{(x_i - μ_i)^2}{σ_i^2} + \log σ_i^2 \right]$$

## Per-Pixel vs Per-Image

### Per-Pixel Loss

Sum/average over all pixels:

$$\mathcal{L} = \frac{1}{HW} \sum_{i,j} \ell(x_{i,j}, ŷ_{i,j})$$

### Per-Image Loss

Average over batch:

$$\mathcal{L} = \frac{1}{B} \sum_b \ell(x_b, ŷ_b)$$

## Reduction Modes

- **`sum`**: Sum over all dimensions (used in VAE paper)
- **`mean`**: Average over all dimensions
- **`none`**: No reduction (returns per-sample loss)

## Normalization

### Image Data

Images are typically normalized to [0, 1]:
- Divide by 255: `x = x / 255.0`
- Use sigmoid output: `ŷ = sigmoid(decoder(z))`

### Other Data

- Standardize: `(x - μ) / σ`
- Normalize: `(x - min) / (max - min)`

## Weighted Reconstruction

Sometimes we weight different parts of the reconstruction:

$$\mathcal{L} = \sum_i w_i \ell(x_i, ŷ_i)$$

For example:
- Higher weight on important regions
- Lower weight on background
- Class-specific weights

## Multi-Modal Reconstruction

For multi-modal inputs (e.g., image + state):

$$\mathcal{L}_{\text{recon}} = \mathcal{L}_{\text{image}} + λ \mathcal{L}_{\text{state}}$$

where `λ` balances the two terms.

## Relationship to Autoencoders

In standard autoencoders:
- Reconstruction loss is the only term
- No probabilistic interpretation
- Direct `L2` or `L1` distance

In VAEs:
- Reconstruction loss is the likelihood term
- Probabilistic interpretation
- Part of a larger objective (with KL divergence)

## Common Issues

### 1. Blurry Reconstructions

**Cause**: Using MSE with natural images
**Solution**: Use perceptual loss, adversarial loss, or learned metrics

### 2. Mode Collapse

**Cause**: Decoder ignores latent code
**Solution**: Ensure KL term is working, check gradient flow

### 3. Overfitting

**Cause**: Model memorizes training data
**Solution**: Increase KL weight, add regularization

## Advanced Losses

### Perceptual Loss

Uses features from a pre-trained network:

$$\mathcal{L}_{\text{perceptual}} = ||\phi(x) - \phi(ŷ)||^2$$

where `φ` is a feature extractor (e.g., VGG).

### Adversarial Loss

Combines with a discriminator:

$$\mathcal{L}_{\text{adv}} = \log D(x) + \log(1 - D(ŷ))$$

## Implementation Example

```python
def vae_reconstruction_loss(recon_x, x, loss_type='bce'):
    """
    Compute reconstruction loss for VAE.
    
    Args:
        recon_x: Reconstructed data [B, C, H, W]
        x: Original data [B, C, H, W]
        loss_type: 'bce', 'mse', or 'l1'
    """
    if loss_type == 'bce':
        return F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif loss_type == 'mse':
        return F.mse_loss(recon_x, x, reduction='sum')
    elif loss_type == 'l1':
        return F.l1_loss(recon_x, x, reduction='sum')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
```

## References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.
- Goodfellow et al. (2016). Deep Learning. MIT Press.

