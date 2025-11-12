"""
Mixture Density Network RNN (MDN-RNN) implementation.

The MDN-RNN predicts future latent states given current latent state and action.
It outputs a mixture of Gaussians to model uncertainty in predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily, Independent
from torch.distributions.categorical import Categorical


class MDNRNN(nn.Module):
    """
    Mixture Density Network RNN for predicting future latent states.
    
    Given a sequence of (latent_state, action) pairs, predicts the next latent state
    as a mixture of Gaussians to capture uncertainty.
    """
    
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, num_mixtures=5):
        """
        Args:
            latent_dim: Dimension of latent state from VAE
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension of LSTM
            num_mixtures: Number of Gaussian mixtures
        """
        super(MDNRNN, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        
        # Input: latent_state + action
        input_dim = latent_dim + action_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # MDN output layers
        # For each mixture: mean (latent_dim), logvar (latent_dim), weight (1)
        self.mdn_linear = nn.Linear(hidden_dim, num_mixtures * (2 * latent_dim + 1))
    
    def forward(self, latent_states, actions, hidden=None):
        """
        Forward pass through MDN-RNN.
        
        Args:
            latent_states: [batch_size, seq_len, latent_dim]
            actions: [batch_size, seq_len, action_dim]
            hidden: LSTM hidden state (optional)
        
        Returns:
            pi: Mixture weights [batch_size, seq_len, num_mixtures]
            mu: Mixture means [batch_size, seq_len, num_mixtures, latent_dim]
            logvar: Mixture log variances [batch_size, seq_len, num_mixtures, latent_dim]
            hidden: Updated LSTM hidden state
        """
        batch_size, seq_len = latent_states.shape[:2]
        # Concatenate latent state and action
        inputs = torch.cat([latent_states, actions], dim=-1)  # [batch, seq, latent_dim + action_dim]
        
        # LSTM forward
        lstm_out, hidden = self.lstm(inputs, hidden)  # [batch, seq, hidden_dim]
        
        # MDN output
        mdn_out = self.mdn_linear(lstm_out)  # [batch, seq, num_mixtures * (2*latent_dim + 1)]
        
        # Reshape and split into components
        mdn_out = mdn_out.view(batch_size, seq_len, self.num_mixtures, 2 * self.latent_dim + 1)
        
        # Split into pi, mu, logvar
        pi_logits = mdn_out[..., 0]  # [batch, seq, num_mixtures]
        pi = F.softmax(pi_logits, dim=-1)
        
        mu = mdn_out[..., 1:1 + self.latent_dim]  # [batch, seq, num_mixtures, latent_dim]
        logvar = mdn_out[..., 1 + self.latent_dim:]  # [batch, seq, num_mixtures, latent_dim]
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return pi, mu, logvar, hidden
    
    def get_distribution(self, pi, mu, logvar):
        """
        Create a mixture distribution from MDN outputs.
        
        Args:
            pi: Mixture weights [batch_size, seq_len, num_mixtures]
            mu: Mixture means [batch_size, seq_len, num_mixtures, latent_dim]
            logvar: Mixture log variances [batch_size, seq_len, num_mixtures, latent_dim]
        
        Returns:
            Mixture distribution
        """
        batch_size, seq_len, num_mixtures, latent_dim = mu.shape
        
        # Reshape for distribution
        pi = pi.view(-1, num_mixtures)  # [batch*seq, num_mixtures]
        mu = mu.view(-1, num_mixtures, latent_dim)  # [batch*seq, num_mixtures, latent_dim]
        logvar = logvar.view(-1, num_mixtures, latent_dim)  # [batch*seq, num_mixtures, latent_dim]
        
        # Create categorical distribution for mixture weights
        mix_dist = Categorical(pi)
        
        # Create normal distributions for each mixture component
        comp_dist = Independent(Normal(mu, torch.exp(0.5 * logvar)), 1)
        
        # Create mixture distribution
        mixture = MixtureSameFamily(mix_dist, comp_dist)
        
        return mixture
    
    def predict(self, latent_state, action, hidden=None):
        """
        Predict next latent state given current state and action.
        
        Args:
            latent_state: [batch_size, latent_dim]
            action: [batch_size, action_dim]
            hidden: LSTM hidden state (optional)
        
        Returns:
            next_latent: Predicted next latent state [batch_size, latent_dim]
            hidden: Updated LSTM hidden state
        """
        
        # Forward pass
        pi, mu, logvar, hidden = self.forward(latent_state, action, hidden)
        
        # Sample from mixture (or take mean)
        # Option 1: Sample from mixture
        mixture = self.get_distribution(pi, mu, logvar)
        next_latent = mixture.sample()  # [batch*1, latent_dim]
        next_latent = next_latent.view(-1, self.latent_dim)  # [batch, latent_dim]
        
        # Option 2: Take weighted mean (deterministic)
        # next_latent = torch.sum(pi.squeeze(1).unsqueeze(-1) * mu.squeeze(1), dim=1)
        
        return next_latent, hidden


def mdn_rnn_loss(pi, mu, logvar, target_latent):
    """
    Compute MDN-RNN loss (negative log-likelihood).
    
    Args:
        pi: Mixture weights [batch_size, seq_len, num_mixtures]
        mu: Mixture means [batch_size, seq_len, num_mixtures, latent_dim]
        logvar: Mixture log variances [batch_size, seq_len, num_mixtures, latent_dim]
        target_latent: Target next latent states [batch_size, seq_len, latent_dim]
    
    Returns:
        loss: Negative log-likelihood loss
    """
    batch_size, seq_len, num_mixtures, latent_dim = mu.shape
    
    # Reshape for computation
    pi = pi.view(-1, num_mixtures)  # [batch*seq, num_mixtures]
    mu = mu.view(-1, num_mixtures, latent_dim)  # [batch*seq, num_mixtures, latent_dim]
    logvar = logvar.view(-1, num_mixtures, latent_dim)  # [batch*seq, num_mixtures, latent_dim]
    target = target_latent.view(-1, latent_dim)  # [batch*seq, latent_dim]
    
    # Expand target for broadcasting
    target = target.unsqueeze(1).expand(-1, num_mixtures, -1)  # [batch*seq, num_mixtures, latent_dim]
    
    # Compute log probability for each mixture component
    # log N(target | mu, var) = -0.5 * sum(log(2*pi*var) + (target-mu)^2/var)
    log_prob = -0.5 * (
        latent_dim * torch.log(2 * torch.tensor(3.14159, device=mu.device))
        + torch.sum(logvar, dim=-1)  # sum over latent_dim
        + torch.sum((target - mu).pow(2) / torch.exp(logvar), dim=-1)  # sum over latent_dim
    )  # [batch*seq, num_mixtures]
    
    # Weight by mixture weights and sum
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)  # [batch*seq, num_mixtures]
    
    # Log-sum-exp trick for numerical stability
    max_log_prob = torch.max(weighted_log_prob, dim=-1, keepdim=True)[0]
    log_sum_exp = max_log_prob + torch.log(
        torch.sum(torch.exp(weighted_log_prob - max_log_prob), dim=-1, keepdim=True) + 1e-8
    )
    
    # Negative log-likelihood
    nll = -torch.sum(log_sum_exp)
    
    return nll

