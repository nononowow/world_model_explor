# Variational Inference

## Overview

Variational Inference (VI) is a method for approximating complex probability distributions. In the context of VAEs, it's used to approximate the posterior distribution `p(z|x)` where `z` is a latent variable and `x` is an observation.

## The Problem

Given observations `x`, we want to infer the latent variables `z` that generated them. The true posterior is:

$$p(z|x) = \frac{p(x|z) p(z)}{p(x)}$$

However, computing `p(x) = \int p(x|z) p(z) dz` is often intractable (especially for high-dimensional `z`).

## Solution: Variational Approximation

Instead of computing the exact posterior, we approximate it with a simpler distribution `q_φ(z|x)` parameterized by `φ`.

### Variational Family

We choose a family of distributions `Q` (e.g., Gaussian) and find the best approximation:

$$q^*(z|x) = \arg\min_{q \in Q} \text{KL}(q(z|x) || p(z|x))$$

### Evidence Lower Bound (ELBO)

Minimizing KL divergence is equivalent to maximizing the Evidence Lower Bound:

$$\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) || p(z))$$

**Derivation:**

$$\log p(x) = \log \int p(x|z) p(z) dz$$

$$= \log \int q(z|x) \frac{p(x|z) p(z)}{q(z|x)} dz$$

$$= \log \mathbb{E}_{q(z|x)}\left[\frac{p(x|z) p(z)}{q(z|x)}\right]$$

Using Jensen's inequality:

$$\geq \mathbb{E}_{q(z|x)}\left[\log \frac{p(x|z) p(z)}{q(z|x)}\right]$$

$$= \mathbb{E}_{q(z|x)}[\log p(x|z)] + \mathbb{E}_{q(z|x)}\left[\log \frac{p(z)}{q(z|x)}\right]$$

$$= \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) || p(z))$$

## Interpretation

The ELBO has two terms:

1. **Reconstruction term**: `𝔼[log p(x|z)]`
   - Measures how well we can reconstruct `x` from `z`
   - Encourages `q(z|x)` to place mass on `z` values that explain `x` well

2. **Regularization term**: `-KL(q(z|x) || p(z))`
   - Keeps `q(z|x)` close to the prior `p(z)`
   - Prevents overfitting to training data
   - Encourages a compact latent space

## In VAE

In Variational Autoencoders:

- **Encoder**: Learns `q_φ(z|x)` (approximate posterior)
- **Decoder**: Learns `p_θ(x|z)` (likelihood)
- **Prior**: `p(z) = N(0, I)` (standard Gaussian)
- **Approximate Posterior**: `q_φ(z|x) = N(μ_φ(x), σ_φ(x))`

## Key Insights

1. **Tightness of Bound**: The gap between ELBO and true log-likelihood is exactly the KL divergence
2. **Trade-off**: Better reconstruction vs. simpler latent space
3. **Amortization**: Instead of optimizing `q(z|x)` for each `x`, we learn a function (encoder) that maps any `x` to `q(z|x)`

## References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.
- Blei et al. (2017). Variational Inference: A Review for Statisticians. JASA.

