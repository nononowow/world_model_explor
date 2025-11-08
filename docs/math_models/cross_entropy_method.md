# Cross Entropy Method (CEM)

## Overview

The Cross Entropy Method (CEM) is a population-based optimization algorithm that iteratively refines a distribution over solutions. It's commonly used in MPC for world models because it's simple, robust, and doesn't require gradients.

## Basic Idea

1. **Initialize**: Start with a distribution over solutions
2. **Sample**: Generate candidate solutions
3. **Evaluate**: Score each candidate
4. **Select**: Keep top-K (elite) solutions
5. **Update**: Refit distribution to elites
6. **Repeat**: Until convergence

## Algorithm

### Pseudocode

```
1. Initialize distribution: μ, σ
2. For iteration = 1 to max_iterations:
    3. Sample N candidates from N(μ, σ)
    4. Evaluate all candidates
    5. Select top K elites (best performers)
    6. Update: μ = mean(elites), σ = std(elites)
    7. Optionally: smooth update with α
8. Return best candidate
```

## Mathematical Formulation

### Initialization

Start with a distribution over action sequences:

$$a \sim \mathcal{N}(\mu_0, \Sigma_0)$$

Typically:
- `μ_0 = 0` (zero mean)
- `Σ_0 = I` (identity covariance)

### Sampling

At iteration `t`, sample `N` candidates:

$$a_i^{(t)} \sim \mathcal{N}(\mu_t, \Sigma_t), \quad i = 1, ..., N$$

### Evaluation

Evaluate each candidate using the world model:

$$J_i = \sum_{k=0}^{H-1} r(z_k, a_{i,k})$$

where `z_k` are predicted states from the world model.

### Selection

Keep top `K` elites (best `K` candidates):

$$\mathcal{E}_t = \{a_i^{(t)} : J_i \text{ in top } K\}$$

### Update

Update distribution parameters:

$$\mu_{t+1} = \frac{1}{K} \sum_{a \in \mathcal{E}_t} a$$

$$\Sigma_{t+1} = \frac{1}{K} \sum_{a \in \mathcal{E}_t} (a - \mu_{t+1})(a - \mu_{t+1})^T$$

### Smoothing (Optional)

To prevent premature convergence:

$$\mu_{t+1} = \alpha \cdot \mu_{t+1} + (1-\alpha) \cdot \mu_t$$

$$\Sigma_{t+1} = \alpha \cdot \Sigma_{t+1} + (1-\alpha) \cdot \Sigma_t$$

where `α ∈ [0, 1]` is a smoothing factor (typically 0.25).

## Implementation

### Basic CEM

```python
def cem_plan(world_model, initial_latent, reward_fn, 
             horizon=12, num_samples=1000, num_elite=100, 
             num_iterations=5, alpha=0.25):
    """
    CEM planning.
    
    Args:
        world_model: Dict with 'vae', 'rnn'
        initial_latent: Current latent state
        reward_fn: Function (latent, action) -> reward
        horizon: Planning horizon
        num_samples: Number of candidates to sample
        num_elite: Number of elites to keep
        num_iterations: Number of CEM iterations
        alpha: Smoothing factor
    
    Returns:
        action: Optimal first action
    """
    action_dim = 3  # Example: steering, throttle, brake
    device = initial_latent.device
    
    # Initialize distribution
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)
    
    for iteration in range(num_iterations):
        # Sample candidates
        actions = torch.normal(
            mean.unsqueeze(0).expand(num_samples, -1, -1),
            std.unsqueeze(0).expand(num_samples, -1, -1)
        )
        actions = torch.clamp(actions, -1, 1)  # Clip to valid range
        
        # Evaluate candidates
        rewards = evaluate_sequences(
            world_model, initial_latent, actions, reward_fn, device
        )
        
        # Select elites
        elite_indices = torch.argsort(rewards, descending=True)[:num_elite]
        elite_actions = actions[elite_indices]
        
        # Update distribution
        new_mean = elite_actions.mean(dim=0)
        new_std = elite_actions.std(dim=0)
        
        # Smooth update
        mean = alpha * new_mean + (1 - alpha) * mean
        std = alpha * new_std + (1 - alpha) * std
        std = torch.clamp(std, min=0.1)  # Prevent collapse
    
    # Return first action of best sequence
    best_idx = torch.argmax(rewards)
    return actions[best_idx, 0].cpu().numpy()

def evaluate_sequences(world_model, initial_latent, actions, reward_fn, device):
    """Evaluate action sequences using world model."""
    num_samples, horizon, action_dim = actions.shape
    
    z = initial_latent.unsqueeze(0).expand(num_samples, -1)
    hidden = None
    total_rewards = torch.zeros(num_samples, device=device)
    
    world_model['rnn'].eval()
    
    with torch.no_grad():
        for k in range(horizon):
            action = actions[:, k, :].to(device)
            
            # Compute reward
            rewards = reward_fn(z, action)
            total_rewards += rewards
            
            # Predict next state
            z, hidden = world_model['rnn'].predict(z, action.unsqueeze(1), hidden)
            z = z.squeeze(1)
    
    return total_rewards
```

## Hyperparameters

### Population Size

- **Small (N=100-500)**: Fast, but may miss good solutions
- **Medium (N=1000)**: Good balance (common)
- **Large (N=5000+)**: Better coverage, but slower

### Elite Size

- **Small (K=10-50)**: Fast convergence, but may get stuck
- **Medium (K=100)**: Good balance (common)
- **Large (K=500+)**: Slower convergence, more exploration

### Iterations

- **Few (3-5)**: Fast, but may not converge
- **Medium (5-10)**: Good balance
- **Many (10+)**: Better solutions, but slower

### Smoothing Factor

- **α = 0.0**: No smoothing (may be unstable)
- **α = 0.25**: Common choice
- **α = 1.0**: No smoothing (aggressive updates)

## Advantages

1. **No gradients**: Doesn't require differentiable reward
2. **Robust**: Works well with noisy evaluations
3. **Simple**: Easy to implement
4. **Parallelizable**: Can evaluate candidates in parallel

## Disadvantages

1. **Sample inefficient**: Needs many evaluations
2. **Local optima**: May get stuck
3. **High-dimensional**: Performance degrades in high dimensions

## Variants

### CMA-ES

Covariance Matrix Adaptation Evolution Strategy:
- Adapts full covariance matrix (not just diagonal)
- Better for high-dimensional problems
- More complex, but often better performance

### Separable CEM

For high-dimensional problems:
- Treat each dimension independently
- Faster, but less expressive

## In World Models

CEM is commonly used because:
1. **Fast planning**: Can evaluate many sequences quickly
2. **No gradients**: Reward function doesn't need to be differentiable
3. **Robust**: Handles stochasticity in world model

## References

- Rubinstein (1997). Optimization of Computer Simulation Models with Rare Events. European Journal of Operations Research.
- Chua et al. (2018). Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS.

