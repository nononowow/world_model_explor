# Mixture Density Networks

## Overview

Mixture Density Networks (MDN) are neural networks that output parameters of a mixture distribution rather than point predictions. This allows modeling complex, multi-modal distributions and capturing uncertainty.

## The Problem

Standard neural networks output point estimates:
- **Regression**: Single value prediction
- **Classification**: Single class prediction

But many problems have:
- **Multiple valid outputs**: Several possible outcomes
- **Uncertainty**: We don't know which outcome will occur
- **Multi-modality**: Output distribution has multiple peaks

## Solution: Mixture Distribution

Instead of predicting a single value, predict a **mixture of distributions**:

$$p(y|x) = \sum_{i=1}^K \pi_i(x) \cdot \mathcal{N}(y; \mu_i(x), \sigma_i^2(x))$$

where:
- `K`: Number of mixture components
- `π_i(x)`: Mixing weights (probabilities, sum to 1)
- `μ_i(x)`: Mean of component `i`
- `σ_i²(x)`: Variance of component `i`

## Architecture

### Output Layer

For `K` components and `D`-dimensional output:

1. **Mixing weights**: `K` values → softmax → probabilities
2. **Means**: `K × D` values → no activation (can be any real number)
3. **Variances**: `K × D` values → exp → positive values

### Network Structure

```
Input x → Hidden Layers → Output Layer
                              ├─→ π (K values) → Softmax
                              ├─→ μ (K×D values)
                              └─→ σ² (K×D values) → Exp
```

## Mathematical Formulation

### Mixture Distribution

$$p(y|x) = \sum_{i=1}^K \pi_i \mathcal{N}(y; \mu_i, \Sigma_i)$$

For diagonal covariance (common case):

$$p(y|x) = \sum_{i=1}^K \pi_i \prod_{d=1}^D \mathcal{N}(y_d; \mu_{i,d}, \sigma_{i,d}^2)$$

### Likelihood

Given data `(x, y)`, the likelihood is:

$$L = \prod_{n=1}^N p(y_n | x_n) = \prod_{n=1}^N \sum_{i=1}^K \pi_i(x_n) \mathcal{N}(y_n; \mu_i(x_n), \sigma_i^2(x_n))$$

### Negative Log-Likelihood Loss

$$-\log L = -\sum_{n=1}^N \log \sum_{i=1}^K \pi_i(x_n) \mathcal{N}(y_n; \mu_i(x_n), \sigma_i^2(x_n))$$

## Implementation

### Forward Pass

```python
def mdn_forward(x, num_mixtures, output_dim):
    """
    MDN forward pass.
    
    Args:
        x: Input [batch_size, input_dim]
        num_mixtures: Number of mixture components K
        output_dim: Dimension of output D
    
    Returns:
        pi: Mixing weights [batch_size, K]
        mu: Means [batch_size, K, D]
        sigma: Standard deviations [batch_size, K, D]
    """
    # Network output
    output = network(x)  # [batch_size, K*(2*D + 1)]
    
    # Reshape
    output = output.view(batch_size, num_mixtures, 2*output_dim + 1)
    
    # Split
    pi_logits = output[:, :, 0]  # [batch_size, K]
    mu = output[:, :, 1:1+output_dim]  # [batch_size, K, D]
    logvar = output[:, :, 1+output_dim:]  # [batch_size, K, D]
    
    # Apply activations
    pi = F.softmax(pi_logits, dim=1)  # Probabilities
    sigma = torch.exp(0.5 * logvar)  # Standard deviation
    
    return pi, mu, sigma
```

### Loss Function

```python
def mdn_loss(pi, mu, sigma, target):
    """
    Compute negative log-likelihood for MDN.
    
    Args:
        pi: Mixing weights [batch_size, K]
        mu: Means [batch_size, K, D]
        sigma: Standard deviations [batch_size, K, D]
        target: Target values [batch_size, D]
    """
    batch_size, num_mixtures, output_dim = mu.shape
    
    # Expand target for broadcasting
    target = target.unsqueeze(1).expand(-1, num_mixtures, -1)
    
    # Compute log probability for each component
    log_prob = -0.5 * (
        output_dim * np.log(2 * np.pi)
        + torch.sum(torch.log(sigma**2), dim=-1)
        + torch.sum((target - mu)**2 / sigma**2, dim=-1)
    )  # [batch_size, K]
    
    # Weight by mixing weights
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    
    # Log-sum-exp for numerical stability
    max_log_prob = torch.max(weighted_log_prob, dim=1, keepdim=True)[0]
    log_sum_exp = max_log_prob + torch.log(
        torch.sum(torch.exp(weighted_log_prob - max_log_prob), dim=1, keepdim=True) + 1e-8
    )
    
    # Negative log-likelihood
    nll = -torch.sum(log_sum_exp)
    
    return nll
```

## Sampling

To sample from the mixture:

1. Sample component: `i ~ Categorical(π)`
2. Sample from component: `y ~ N(μ_i, σ_i²)`

```python
def sample_from_mdn(pi, mu, sigma):
    """Sample from mixture distribution."""
    # Sample component
    mix_dist = Categorical(pi)
    component = mix_dist.sample()  # [batch_size]
    
    # Sample from selected component
    batch_size = pi.shape[0]
    samples = []
    for b in range(batch_size):
        comp_idx = component[b].item()
        sample = torch.normal(mu[b, comp_idx], sigma[b, comp_idx])
        samples.append(sample)
    
    return torch.stack(samples)
```

## In World Models (MDN-RNN)

### Application

MDN-RNN predicts future latent states:
- **Input**: Current latent state `z_t` and action `a_t`
- **Output**: Distribution over next latent state `z_{t+1}`
- **Uncertainty**: Captures stochasticity in environment

### Why MDN?

1. **Uncertainty**: Environment is stochastic
2. **Multi-modality**: Multiple possible futures
3. **Planning**: Need to reason about uncertainty

### Architecture

```
(z_t, a_t) → LSTM → Hidden State → MDN Layer
                                    ├─→ π (mixture weights)
                                    ├─→ μ (means)
                                    └─→ σ² (variances)
```

## Choosing Number of Mixtures

### Too Few (K = 1)

- Single Gaussian
- Can't capture multi-modality
- Underfits complex distributions

### Too Many (K >> needed)

- Overparameterized
- May overfit
- Slower training and inference

### Rule of Thumb

- Start with `K = 5` (common in world models)
- Increase if distribution is very complex
- Decrease if model is overfitting

## Numerical Stability

### Issues

1. **Log of small values**: `log(σ²)` when `σ²` is very small
2. **Exponentiation**: `exp(logvar)` can overflow
3. **Division**: `1/σ²` when `σ²` is very small

### Solutions

1. **Clamp logvar**: `logvar = clamp(logvar, min=-10, max=10)`
2. **Add epsilon**: `pi = pi + eps` before taking log
3. **Log-sum-exp trick**: Used in loss computation

## Advantages

1. **Uncertainty Quantification**: Naturally captures uncertainty
2. **Multi-modality**: Can model complex distributions
3. **Flexibility**: Works with any base distribution (Gaussian, etc.)

## Disadvantages

1. **Complexity**: More parameters than point prediction
2. **Training**: More difficult to train
3. **Inference**: Sampling required (not deterministic)

## References

- Bishop (1994). Mixture Density Networks. Technical Report.
- Ha & Schmidhuber (2018). World Models. arXiv:1803.10122.

