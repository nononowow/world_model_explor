# MDN-RNN

## Overview

MDN-RNN (Mixture Density Network RNN) combines an LSTM with a Mixture Density Network output layer. It's a key component of world models, used to predict future latent states with uncertainty.

## Architecture

### Components

1. **LSTM**: Processes sequences of (latent_state, action) pairs
2. **MDN Output Layer**: Outputs parameters of a mixture of Gaussians

### Structure

```
Input Sequence: [(z_1, a_1), (z_2, a_2), ..., (z_T, a_T)]
                    ↓
              LSTM Layers
                    ↓
            Hidden States: [h_1, h_2, ..., h_T]
                    ↓
            MDN Output Layer
                    ↓
    [π_t, μ_t, σ_t] for each timestep t
```

## Mathematical Formulation

### Forward Pass

For each timestep `t`:

1. **LSTM Update**:
   $$h_t, C_t = \text{LSTM}([z_t, a_t], h_{t-1}, C_{t-1})$$

2. **MDN Output**:
   $$\pi_t, \mu_t, \sigma_t = \text{MDN}(h_t)$$

3. **Distribution**:
   $$p(z_{t+1} | z_t, a_t, h_{t-1}) = \sum_{i=1}^K \pi_{t,i} \mathcal{N}(z_{t+1}; \mu_{t,i}, \sigma_{t,i}^2)$$

### Prediction

To predict next latent state:

1. **Sample from mixture**:
   $$z_{t+1} \sim \sum_{i=1}^K \pi_{t,i} \mathcal{N}(\mu_{t,i}, \sigma_{t,i}^2)$$

2. **Or take weighted mean** (deterministic):
   $$z_{t+1} = \sum_{i=1}^K \pi_{t,i} \mu_{t,i}$$

## Loss Function

### Negative Log-Likelihood

Given a sequence of latent states `z_1, ..., z_T` and actions `a_1, ..., a_{T-1}`:

$$\mathcal{L} = -\sum_{t=1}^{T-1} \log p(z_{t+1} | z_t, a_t, h_{t-1})$$

$$= -\sum_{t=1}^{T-1} \log \sum_{i=1}^K \pi_{t,i} \mathcal{N}(z_{t+1}; \mu_{t,i}, \sigma_{t,i}^2)$$

### Implementation

```python
def mdn_rnn_loss(pi, mu, logvar, target_latent):
    """
    Compute MDN-RNN loss.
    
    Args:
        pi: [batch_size, seq_len, num_mixtures]
        mu: [batch_size, seq_len, num_mixtures, latent_dim]
        logvar: [batch_size, seq_len, num_mixtures, latent_dim]
        target_latent: [batch_size, seq_len, latent_dim]
    """
    batch_size, seq_len, num_mixtures, latent_dim = mu.shape
    
    # Reshape for computation
    pi = pi.view(-1, num_mixtures)
    mu = mu.view(-1, num_mixtures, latent_dim)
    logvar = logvar.view(-1, num_mixtures, latent_dim)
    target = target_latent.view(-1, latent_dim)
    
    # Expand target
    target = target.unsqueeze(1).expand(-1, num_mixtures, -1)
    
    # Log probability for each component
    log_prob = -0.5 * (
        latent_dim * np.log(2 * np.pi)
        + torch.sum(logvar, dim=-1)
        + torch.sum((target - mu)**2 / torch.exp(logvar), dim=-1)
    )
    
    # Weighted log probability
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    
    # Log-sum-exp
    max_log_prob = torch.max(weighted_log_prob, dim=-1, keepdim=True)[0]
    log_sum_exp = max_log_prob + torch.log(
        torch.sum(torch.exp(weighted_log_prob - max_log_prob), dim=-1, keepdim=True) + 1e-8
    )
    
    # Negative log-likelihood
    nll = -torch.sum(log_sum_exp)
    
    return nll
```

## Training

### Data Preparation

1. **Collect rollouts**: `(observation, action, next_observation)` tuples
2. **Encode observations**: Use trained VAE to get latent states `z`
3. **Create sequences**: `(z_t, a_t, z_{t+1})` for `t = 1, ..., T-1`

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        latents, actions, next_latents = batch
        
        # Forward pass
        pi, mu, logvar, hidden = mdn_rnn(latents, actions)
        
        # Loss
        loss = mdn_rnn_loss(pi, mu, logvar, next_latents)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdn_rnn.parameters(), max_norm=1.0)
        optimizer.step()
```

## Prediction

### Single Step

```python
def predict_next(mdn_rnn, current_latent, action, hidden=None):
    """
    Predict next latent state.
    
    Args:
        current_latent: [latent_dim] or [batch_size, latent_dim]
        action: [action_dim] or [batch_size, action_dim]
        hidden: LSTM hidden state
    
    Returns:
        next_latent: Predicted next latent state
        hidden: Updated hidden state
    """
    # Add sequence dimension
    current_latent = current_latent.unsqueeze(1)
    action = action.unsqueeze(1)
    
    # Forward pass
    pi, mu, logvar, hidden = mdn_rnn(current_latent, action, hidden)
    
    # Sample from mixture
    from torch.distributions import MixtureSameFamily, Normal, Categorical
    
    pi = pi.squeeze(1)  # [batch_size, num_mixtures]
    mu = mu.squeeze(1)  # [batch_size, num_mixtures, latent_dim]
    logvar = logvar.squeeze(1)
    
    # Create mixture distribution
    mix_dist = Categorical(pi)
    comp_dist = Independent(Normal(mu, torch.exp(0.5 * logvar)), 1)
    mixture = MixtureSameFamily(mix_dist, comp_dist)
    
    # Sample
    next_latent = mixture.sample()
    
    return next_latent, hidden
```

### Multi-Step Rollout

```python
def rollout(mdn_rnn, initial_latent, actions, hidden=None):
    """
    Rollout multiple steps.
    
    Args:
        initial_latent: [latent_dim]
        actions: [horizon, action_dim]
        hidden: Initial hidden state
    
    Returns:
        predicted_latents: [horizon, latent_dim]
    """
    predicted_latents = []
    current_latent = initial_latent
    
    for action in actions:
        next_latent, hidden = predict_next(mdn_rnn, current_latent, action, hidden)
        predicted_latents.append(next_latent)
        current_latent = next_latent
    
    return torch.stack(predicted_latents)
```

## Why MDN-RNN?

### Uncertainty Modeling

- **Stochastic environments**: Real environments are not deterministic
- **Multiple futures**: Different actions lead to different outcomes
- **Planning**: Need to reason about uncertainty for robust planning

### Advantages over Deterministic RNN

1. **Captures uncertainty**: Knows when predictions are uncertain
2. **Multi-modal predictions**: Can predict multiple possible outcomes
3. **Better planning**: Uncertainty helps with robust decision-making

## Hyperparameters

### LSTM

- **Hidden dimension**: 256-512 (common: 256)
- **Number of layers**: 1-2 (usually 1)
- **Dropout**: 0.1-0.3

### MDN

- **Number of mixtures**: 3-10 (common: 5)
- **Output dimension**: Same as latent dimension

### Training

- **Sequence length**: 8-32 (common: 16)
- **Batch size**: 32-128
- **Learning rate**: 1e-3 to 1e-4
- **Gradient clipping**: 1.0

## Common Issues

### 1. Mode Collapse

**Symptom**: All mixture components converge to same distribution
**Solution**: Initialize means to be different, use more mixtures

### 2. Overconfident Predictions

**Symptom**: Variances become very small
**Solution**: Add minimum variance constraint, regularize

### 3. Poor Long-Term Predictions

**Symptom**: Predictions degrade over long horizons
**Solution**: Train on longer sequences, use scheduled sampling

## In World Models

MDN-RNN is used to:
1. **Predict future states**: Given current state and action
2. **Model uncertainty**: Capture stochasticity
3. **Enable planning**: Controller can plan in latent space
4. **Fast simulation**: No need for environment interaction

## References

- Bishop (1994). Mixture Density Networks. Technical Report.
- Ha & Schmidhuber (2018). World Models. arXiv:1803.10122.

