# Model Predictive Control (MPC)

## Overview

Model Predictive Control (MPC) is an optimization-based control method that uses a model to predict future states and optimize actions over a planning horizon. In world models, MPC is used to plan optimal actions in the latent space.

## Basic Idea

1. **Predict**: Use world model to predict future states
2. **Optimize**: Find action sequence that maximizes reward
3. **Execute**: Apply first action, then replan

## Mathematical Formulation

### Objective

At time `t`, we want to find the optimal action sequence `a_{t:t+H-1}` that maximizes:

$$J = \sum_{k=0}^{H-1} r(z_{t+k}, a_{t+k})$$

subject to:
- Dynamics: `z_{t+k+1} = f(z_{t+k}, a_{t+k})`
- Constraints: `a_{t+k} ∈ A`

where:
- `H`: Planning horizon
- `r(z, a)`: Reward function
- `f`: Dynamics model (world model)
- `A`: Action space

### Optimization Problem

$$\max_{a_{t:t+H-1}} \sum_{k=0}^{H-1} r(\hat{z}_{t+k}, a_{t+k})$$

where `ẑ_{t+k}` are predicted states from the world model.

## Algorithm

### Receding Horizon Control

```
For each timestep t:
    1. Get current state z_t
    2. Solve optimization: a*_{t:t+H-1} = argmax J
    3. Execute first action: a_t = a*_t
    4. Observe next state z_{t+1}
    5. Repeat
```

### Key Properties

1. **Receding horizon**: Only first action is executed
2. **Replanning**: Re-optimize at each step
3. **Feedback**: Uses actual state (not predicted) for next optimization

## In World Models

### Latent Space Planning

Instead of planning in observation space, plan in latent space:

1. **Encode**: `z_t = encode(x_t)`
2. **Plan**: Optimize actions in latent space
3. **Predict**: Use RNN to predict future latent states
4. **Decode**: Can decode to see predicted observations

### Advantages

- **Efficiency**: Latent space is lower-dimensional
- **Speed**: Fast predictions from world model
- **No environment**: Don't need to interact with real environment

## Implementation

### Gradient-Based MPC

```python
def mpc_plan(world_model, initial_latent, reward_fn, horizon=12, num_iterations=10):
    """
    MPC planning using gradient-based optimization.
    
    Args:
        world_model: Dict with 'vae', 'rnn', optionally 'controller'
        initial_latent: Current latent state [latent_dim]
        reward_fn: Function (latent, action) -> reward
        horizon: Planning horizon
        num_iterations: Optimization iterations
    
    Returns:
        action: Optimal first action [action_dim]
    """
    # Initialize actions
    actions = torch.zeros(horizon, action_dim, requires_grad=True)
    optimizer = torch.optim.Adam([actions], lr=0.1)
    
    initial_latent = initial_latent.unsqueeze(0)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Rollout
        z = initial_latent
        hidden = None
        total_reward = 0
        
        world_model['rnn'].train()
        
        for k in range(horizon):
            action = actions[k:k+1].unsqueeze(0)
            
            # Reward
            reward = reward_fn(z, action)
            total_reward += reward
            
            # Predict next state
            z, hidden = world_model['rnn'].predict(z.squeeze(0), action.squeeze(0), hidden)
            z = z.unsqueeze(0)
        
        # Maximize reward (minimize negative)
        loss = -total_reward
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_([actions], max_norm=1.0)
        
        optimizer.step()
        
        # Clip actions to valid range
        with torch.no_grad():
            actions.clamp_(-1, 1)
    
    # Return first action
    return actions[0].detach().cpu().numpy()
```

## Reward Function

### From Controller

```python
def controller_reward_fn(controller, latent, action):
    """Reward based on controller output."""
    controller_action = controller(latent)
    reward = -F.mse_loss(action, controller_action)
    return reward
```

### Goal-Based

```python
def goal_reward_fn(goal_latent, latent, action):
    """Reward for reaching goal."""
    distance = F.mse_loss(latent, goal_latent)
    reward = -distance
    return reward
```

### Custom

```python
def custom_reward_fn(latent, action):
    """Custom reward function."""
    # Decode to observation if needed
    # obs = world_model['vae'].decode(latent)
    # Compute reward based on observation
    reward = compute_reward(latent, action)
    return reward
```

## Cross-Entropy Method (CEM)

Alternative to gradient-based optimization:

1. **Sample**: Generate candidate action sequences
2. **Evaluate**: Score each sequence using world model
3. **Select**: Keep top-K sequences (elites)
4. **Update**: Refit distribution to elites
5. **Repeat**: Iterate until convergence

See [Cross Entropy Method](cross_entropy_method.md) for details.

## Hyperparameters

### Planning Horizon

- **Short (H=4-8)**: Fast, but may miss long-term effects
- **Medium (H=12-16)**: Good balance (common in world models)
- **Long (H=20+)**: Better long-term planning, but slower

### Optimization

- **Iterations**: 5-20 (more = better, but slower)
- **Learning rate**: 0.01-0.1 (for gradient-based)
- **Population size**: 100-1000 (for CEM)

## Advantages

1. **Model-based**: Uses learned world model
2. **Robust**: Replanning handles model errors
3. **Flexible**: Easy to change reward function
4. **Efficient**: Fast planning in latent space

## Disadvantages

1. **Model quality**: Requires good world model
2. **Computational cost**: Optimization at each step
3. **Local optima**: May get stuck in local optima

## Extensions

### Uncertainty-Aware MPC

Consider uncertainty in predictions:

$$J = \mathbb{E}[R] - \lambda \cdot \text{Var}(R)$$

where `λ` trades off expected reward vs. risk.

### Constrained MPC

Add constraints:

$$\max J \text{ s.t. } g(z, a) \leq 0$$

for safety or physical constraints.

## References

- Rawlings & Mayne (2009). Model Predictive Control: Theory and Design.
- Hafner et al. (2019). Learning Latent Dynamics for Planning from Pixels. NeurIPS.

