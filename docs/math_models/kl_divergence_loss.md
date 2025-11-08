# KL Divergence Loss

## Overview

Kullback-Leibler (KL) Divergence measures how different one probability distribution is from another. In VAEs, it's used as a regularization term to keep the learned latent distribution close to a prior distribution.

## Definition

For two probability distributions `P` and `Q`, the KL divergence is:

$$\text{KL}(P || Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

For discrete distributions:

$$\text{KL}(P || Q) = \sum_i p(i) \log \frac{p(i)}{q(i)}$$

## Key Properties

1. **Non-negative**: `KL(P || Q) ≥ 0`
2. **Zero if equal**: `KL(P || Q) = 0` if and only if `P = Q`
3. **Not symmetric**: `KL(P || Q) ≠ KL(Q || P)` in general
4. **Not a metric**: Doesn't satisfy triangle inequality

## In VAE Context

In Variational Autoencoders, we use:

$$\text{KL}(q_φ(z|x) || p(z))$$

where:
- `q_φ(z|x)`: Approximate posterior (encoder output)
- `p(z)`: Prior distribution (typically `N(0, I)`)

### Purpose

1. **Regularization**: Prevents the encoder from collapsing to a point
2. **Latent Space Structure**: Encourages a structured, interpretable latent space
3. **Generalization**: Helps the model generalize to new data

## KL Divergence for Gaussian Distributions

When both distributions are Gaussian, we can compute KL divergence analytically.

### Case: Two Multivariate Gaussians

If:
- `q(z) = N(μ_q, Σ_q)`
- `p(z) = N(μ_p, Σ_p)`

Then:

$$\text{KL}(q || p) = \frac{1}{2} \left[ \text{tr}(Σ_p^{-1} Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) - k + \log \frac{\det(Σ_p)}{\det(Σ_q)} \right]$$

where `k` is the dimensionality.

### Special Case: VAE (Diagonal Covariance)

In VAEs, we typically use:
- `q(z|x) = N(μ, diag(σ²))` (diagonal covariance)
- `p(z) = N(0, I)` (standard normal)

This simplifies to:

$$\text{KL}(q(z|x) || p(z)) = \frac{1}{2} \sum_{i=1}^k \left[ σ_i^2 + μ_i^2 - 1 - \log σ_i^2 \right]$$

where `k` is the latent dimension.

### Derivation

Starting from the general formula with `μ_p = 0` and `Σ_p = I`:

$$\text{KL}(q || p) = \frac{1}{2} \left[ \text{tr}(I^{-1} \text{diag}(σ²)) + (0 - μ)^T I^{-1} (0 - μ) - k + \log \frac{\det(I)}{\det(\text{diag}(σ²))} \right]$$

$$= \frac{1}{2} \left[ \sum_i σ_i^2 + μ^T μ - k - \log \prod_i σ_i^2 \right]$$

$$= \frac{1}{2} \sum_i \left[ σ_i^2 + μ_i^2 - 1 - \log σ_i^2 \right]$$

## Implementation

### Using Log-Variance

In practice, we work with `logvar = log(σ²)`:

```python
# KL divergence for diagonal Gaussian
kl_loss = 0.5 * torch.sum(
    torch.exp(logvar) + mu.pow(2) - 1 - logvar,
    dim=1
)
```

Or element-wise:

```python
kl_loss = 0.5 * (sigma_sq + mu.pow(2) - 1 - logvar).sum(dim=1)
```

where `sigma_sq = torch.exp(logvar)`.

## Interpretation

### Term by Term

1. **`σ²`**: Penalizes large variances (encourages compact latent space)
2. **`μ²`**: Penalizes means far from zero (centers latent space at origin)
3. **`-log σ²`**: Prevents variance from collapsing to zero
4. **`-1`**: Normalization constant

### Effect on Latent Space

- **Small KL**: Latent space is close to standard normal
- **Large KL**: Latent space deviates significantly from prior
- **Zero variance**: If `σ → 0`, KL → ∞ (prevents collapse)

## β-VAE Extension

In β-VAE, we weight the KL term:

$$\mathcal{L} = \mathbb{E}[\log p(x|z)] - β \cdot \text{KL}(q(z|x) || p(z))$$

- **β = 1**: Standard VAE
- **β > 1**: Stronger regularization (more disentangled)
- **β < 1**: Weaker regularization

## Numerical Stability

### Issues

- `log(σ²)` can be unstable when `σ²` is very small
- `exp(logvar)` can overflow

### Solutions

1. **Clamp logvar**: `logvar = clamp(logvar, min=-10, max=10)`
2. **Use log-sum-exp trick** for numerical stability
3. **Add epsilon**: `logvar = logvar + eps` to prevent log(0)

## Relationship to Information Theory

KL divergence is related to:
- **Cross-entropy**: `H(P, Q) = H(P) + KL(P || Q)`
- **Mutual information**: `I(X; Z) = KL(p(x,z) || p(x)p(z))`
- **Entropy**: `H(P) = -KL(P || Uniform)`

## References

- Kullback & Leibler (1951). On Information and Sufficiency. Annals of Mathematical Statistics.
- Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.

