# Mathematical Foundations Study Guide

## Learning Path for World Models

This directory contains detailed mathematical explanations of key concepts used in world models. Follow this study guide to master the fundamentals before diving into implementation.

## Recommended Study Order

### Week 1: Variational Autoencoder Foundations

1. **Start Here**: [Variational Inference](variational_inference.md)
   - Understand the fundamental problem VAE solves
   - Learn about approximate posterior distributions
   - Time: 2-3 hours

2. **Core Concept**: [Reparameterization Trick](reparameterization_trick.md)
   - Essential for training VAEs
   - Enables backpropagation through sampling
   - Time: 1-2 hours

3. **Key Loss Component**: [KL Divergence Loss](kl_divergence_loss.md)
   - Regularization term in VAE
   - Measures difference between distributions
   - Time: 2-3 hours

4. **Reconstruction**: [Reconstruction Loss](reconstruction_loss.md)
   - Measures how well VAE reconstructs inputs
   - Different loss functions for different data types
   - Time: 1-2 hours

5. **Put It Together**: [VAE Complete Loss](vae_complete_loss.md)
   - Combine reconstruction and KL divergence
   - Understand the trade-off (β-VAE)
   - Time: 1 hour

### Week 2: Sequence Modeling and Uncertainty

6. **RNN Basics**: [LSTM and GRU](lstm_gru.md)
   - Recurrent neural networks for sequences
   - Long-term dependencies
   - Time: 2-3 hours

7. **Uncertainty Modeling**: [Mixture Density Networks](mixture_density_networks.md)
   - Modeling complex output distributions
   - Multiple modes and uncertainty
   - Time: 2-3 hours

8. **Combined**: [MDN-RNN](mdn_rnn.md)
   - RNN with mixture density output
   - Predicting future latent states
   - Time: 2-3 hours

### Week 3: World Models Architecture

9. **Planning**: [Model Predictive Control](model_predictive_control.md)
   - Planning in latent space
   - Optimization over action sequences
   - Time: 2-3 hours

10. **Optimization**: [Cross Entropy Method](cross_entropy_method.md)
    - Population-based optimization
    - Used in CEM planner
    - Time: 1-2 hours

11. **Complete System**: [World Models Training](world_models_training.md)
    - Three-stage training process
    - Latent space planning
    - Time: 2-3 hours

## Quick Reference

### Essential Formulas

- **VAE Loss**: `L = L_recon + β * L_KL`
- **KL Divergence**: `KL(q||p) = ∫ q(z) log(q(z)/p(z)) dz`
- **Reparameterization**: `z = μ + σ * ε` where `ε ~ N(0,1)`
- **Mixture Density**: `p(x) = Σ π_i * N(x; μ_i, σ_i)`

### Key Concepts Checklist

- [ ] Understand variational inference
- [ ] Master reparameterization trick
- [ ] Know KL divergence and its role
- [ ] Understand reconstruction losses
- [ ] Grasp LSTM/GRU mechanics
- [ ] Learn mixture density networks
- [ ] Understand MDN-RNN architecture
- [ ] Know MPC and planning
- [ ] Understand CEM optimization
- [ ] Grasp complete world model training

## Study Tips

1. **Read with Code**: After reading each concept, look at the corresponding implementation
2. **Derive Formulas**: Try to derive key formulas yourself
3. **Visualize**: Draw diagrams to understand the concepts
4. **Practice**: Implement simple versions from scratch
5. **Connect**: Understand how concepts connect in the full system

## Time Investment

- **Total Time**: ~25-35 hours
- **Per Day**: 2-3 hours for 2 weeks
- **Intensive**: 4-5 hours per day for 1 week

## Prerequisites

- Basic calculus (derivatives, integrals)
- Linear algebra (matrices, vectors)
- Probability theory (distributions, expectations)
- Basic neural networks (forward/backward pass)

## Next Steps After Mastery

1. Implement each concept from scratch
2. Train models and observe the math in action
3. Experiment with hyperparameters
4. Read original papers with full understanding
5. Extend and improve upon the concepts

---

**Start with**: [Variational Inference](variational_inference.md)

