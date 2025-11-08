# Reparameterization Trick

## Overview

The reparameterization trick is a method that allows us to backpropagate through random sampling operations, making it possible to train VAEs with gradient descent.

## The Problem

In VAEs, we need to sample from the latent distribution:

$$z \sim q_φ(z|x) = \mathcal{N}(μ_φ(x), σ_φ(x))$$

However, directly sampling `z ~ N(μ, σ)` is not differentiable with respect to `μ` and `σ`, which prevents gradient-based optimization.

## The Solution

Instead of sampling directly, we:

1. Sample from a fixed distribution: `ε ~ N(0, 1)`
2. Transform it deterministically: `z = μ + σ ⊙ ε`

This is differentiable because:
- `ε` is independent of parameters `φ`
- The transformation `z = μ + σ ⊙ ε` is differentiable

## Mathematical Formulation

### Standard Formulation

$$z = μ_φ(x) + σ_φ(x) ⊙ ε$$

where:
- `ε ~ N(0, I)` (standard normal, independent of `φ`)
- `⊙` denotes element-wise multiplication
- `μ_φ(x)` and `σ_φ(x)` are outputs of the encoder

### Why It Works

The key insight is that we can express any Gaussian distribution as a transformation of a standard Gaussian:

If `ε ~ N(0, 1)`, then `μ + σ·ε ~ N(μ, σ²)`

This transformation is:
- **Deterministic** (given `ε`)
- **Differentiable** (with respect to `μ` and `σ`)
- **Preserves distribution** (still samples from `N(μ, σ²)`)

## Gradient Flow

### Without Reparameterization (Broken)

```
x → Encoder → (μ, σ) → Sample z ~ N(μ, σ) → Decoder → x̂
                    ↑
              No gradient flow!
```

### With Reparameterization (Works)

```
x → Encoder → (μ, σ) ──┐
                       ├─→ z = μ + σ·ε → Decoder → x̂
ε ~ N(0,1) ────────────┘
              ↑
        Gradient flows!
```

## Implementation

### Pseudocode

```python
# Forward pass
mu, logvar = encoder(x)  # Encoder outputs mean and log-variance
std = exp(0.5 * logvar)  # Standard deviation
eps = sample_from_N(0, 1)  # Sample from standard normal
z = mu + std * eps  # Reparameterization
x_recon = decoder(z)  # Decode

# Backward pass
# Gradients can flow through z → mu, std → encoder
```

### Why Log-Variance?

In practice, we often output `logvar` instead of `σ` because:
- Ensures `σ > 0` (since `σ = exp(0.5 * logvar)`)
- More numerically stable
- Easier to optimize

## Generalization

The reparameterization trick works for other distributions too:

### Location-Scale Family

For any distribution in the location-scale family:

$$z = μ + σ ⊙ ε$$

where `ε` follows a standard distribution (e.g., standard normal, standard uniform).

### Other Distributions

- **Uniform**: `z = a + (b-a) ⊙ ε` where `ε ~ Uniform(0,1)`
- **Exponential**: Requires inverse CDF
- **Beta, Gamma**: More complex transformations

## Key Properties

1. **Differentiability**: Enables gradient-based optimization
2. **Unbiased**: Preserves the original distribution
3. **Low Variance**: Reduces variance in gradient estimates (compared to score function estimator)

## Comparison with Alternatives

### Score Function Estimator (REINFORCE)

- Higher variance
- Works for discrete variables
- Doesn't require reparameterization

### Reparameterization Trick

- Lower variance
- Requires continuous, reparameterizable distributions
- More efficient for continuous latent variables

## In VAE Context

The reparameterization trick is essential for VAEs because:

1. **Training**: Allows backpropagation through the sampling operation
2. **Efficiency**: Lower variance gradients lead to faster convergence
3. **Flexibility**: Works with continuous latent spaces

## References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.
- Rezende et al. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. ICML.

