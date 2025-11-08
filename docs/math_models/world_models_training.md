# World Models Training

## Overview

World Models training follows a three-stage process that separates the learning of different components. This document explains the complete training pipeline and the mathematical foundations.

## Three-Stage Training

### Stage 1: Data Collection

**Goal**: Collect diverse experience from the environment.

**Process**:
1. Run random policy (or simple heuristic) in environment
2. Collect tuples: `(observation, action, reward, next_observation)`
3. Store in dataset

**Mathematical Formulation**:

Collect dataset:
$$\mathcal{D} = \{(x_t, a_t, r_t, x_{t+1})\}_{t=1}^T$$

where:
- `x_t`: Observation at time `t`
- `a_t`: Action taken
- `r_t`: Reward received
- `x_{t+1}`: Next observation

### Stage 2: VAE Training

**Goal**: Learn to encode observations into a compact latent space.

**Process**:
1. Train VAE on collected observations `{x_t}`
2. Learn encoder `q_φ(z|x)` and decoder `p_θ(x|z)`

**Loss Function**:

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{x \sim \mathcal{D}} \left[ -\mathbb{E}_{q(z|x)}[\log p(x|z)] + \beta \cdot \text{KL}(q(z|x) || p(z)) \right]$$

**After Training**:
- Can encode observations: `z = encode(x)`
- Can decode latent states: `x = decode(z)`

### Stage 3: RNN Training

**Goal**: Learn to predict future latent states.

**Process**:
1. Encode all observations: `z_t = encode(x_t)`
2. Train RNN on sequences: `(z_t, a_t, z_{t+1})`

**Loss Function**:

$$\mathcal{L}_{\text{RNN}} = -\sum_{t=1}^{T-1} \log p(z_{t+1} | z_t, a_t, h_{t-1})$$

For MDN-RNN:

$$= -\sum_{t=1}^{T-1} \log \sum_{i=1}^K \pi_{t,i} \mathcal{N}(z_{t+1}; \mu_{t,i}, \sigma_{t,i}^2)$$

**After Training**:
- Can predict: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)`

### Stage 4: Controller Training

**Goal**: Learn to act optimally in latent space.

**Process**:
1. Train controller `π(a|z)` to maximize expected reward
2. Use world model (VAE + RNN) for fast evaluation

**Objective**:

$$\max_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^H r_t \right]$$

where trajectories `τ` are generated using the world model.

## Complete Training Pipeline

### Algorithm

```
1. Collect Data:
   for episode in range(num_episodes):
       for step in range(max_steps):
           a = random_action()
           x, r, done = env.step(a)
           store(x, a, r)

2. Train VAE:
   for epoch in range(vae_epochs):
       for batch in observations:
           recon, mu, logvar = vae(x)
           loss = recon_loss + KL_loss
           optimize(vae, loss)

3. Train RNN:
   encode all observations to latent space
   for epoch in range(rnn_epochs):
       for sequence in latent_sequences:
           pi, mu, logvar = rnn(z, a)
           loss = mdn_loss(pi, mu, logvar, z_next)
           optimize(rnn, loss)

4. Train Controller:
   for generation in range(num_generations):
       population = sample_controllers()
       for controller in population:
           reward = evaluate_in_world_model(controller)
       elites = select_top_k(population, k)
       update_distribution(elites)
```

## Latent Space Planning

### Why Latent Space?

1. **Efficiency**: Lower-dimensional than observation space
2. **Speed**: Fast predictions from world model
3. **Abstraction**: Captures essential information

### Planning Process

```
1. Encode current observation: z_0 = encode(x_0)
2. Plan in latent space:
   a* = argmax_a sum_{t=0}^{H-1} r(z_t, a_t)
   where z_{t+1} = predict(z_t, a_t)
3. Execute first action: a_0
4. Observe next state: x_1
5. Repeat
```

## Controller Training Methods

### 1. Evolutionary Strategies

**Idea**: Evolve a population of controllers.

**Algorithm**:
1. Initialize population of controllers
2. Evaluate each in world model
3. Select best performers
4. Create new population from elites
5. Repeat

**Advantages**:
- Simple
- Works with any reward function
- Robust to local optima

**Disadvantages**:
- Sample inefficient
- May need many evaluations

### 2. Policy Gradients

**Idea**: Use gradients to improve policy.

**Objective**:

$$\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t | z_t) \cdot R_t \right]$$

where `R_t` is the return (cumulative reward).

**Advantages**:
- Sample efficient
- Can use value functions

**Disadvantages**:
- Requires differentiable policy
- More complex

## World Model Evaluation

### In Model (Fast)

Evaluate controller using world model:
- No environment interaction needed
- Fast (can evaluate many controllers)
- May have model errors

### In Environment (Slow)

Evaluate controller in real environment:
- Accurate
- Slow (real-time interaction)
- Used for final evaluation

## Key Insights

### 1. Separation of Concerns

- **VAE**: Handles high-dimensional observations
- **RNN**: Handles temporal dynamics
- **Controller**: Handles decision-making

### 2. Latent Space Benefits

- **Abstraction**: Learns useful representations
- **Efficiency**: Faster than pixel space
- **Generalization**: Can plan for unseen scenarios

### 3. Model-Based Advantages

- **Sample efficiency**: Learn from collected data
- **Fast planning**: No environment interaction needed
- **Safety**: Can test policies safely in model

## Common Issues

### 1. VAE Quality

**Problem**: Poor VAE → poor latent space → poor planning
**Solution**: Ensure VAE is well-trained before RNN training

### 2. Model Errors

**Problem**: World model doesn't match real environment
**Solution**: Collect more diverse data, improve model capacity

### 3. Distribution Shift

**Problem**: Controller explores regions not seen during training
**Solution**: Use uncertainty estimates, add regularization

## Hyperparameters

### Data Collection

- **Episodes**: 100-1000 (more = better, but slower)
- **Steps per episode**: Environment-dependent

### VAE Training

- **Epochs**: 50-100
- **Batch size**: 32-128
- **Learning rate**: 1e-4 to 1e-3
- **β**: 1.0 (or higher for disentanglement)

### RNN Training

- **Epochs**: 50-100
- **Sequence length**: 8-32
- **Batch size**: 32-128
- **Learning rate**: 1e-3 to 1e-4

### Controller Training

- **Generations**: 10-50 (evolutionary)
- **Population size**: 32-128
- **Horizon**: 12-16 (for planning)

## References

- Ha & Schmidhuber (2018). World Models. arXiv:1803.10122.
- Hafner et al. (2019). Dreamer: Learning Models by Imagination. arXiv:1912.01603.

