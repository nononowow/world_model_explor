"""
Transformer-based dynamics model.

Alternative to RNN for modeling temporal dynamics in latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerDynamics(nn.Module):
    """
    Transformer-based dynamics model.
    
    Predicts future latent states using self-attention mechanism.
    """
    
    def __init__(self, latent_dim=32, action_dim=3, d_model=256, nhead=8, 
                 num_layers=4, dim_feedforward=1024, dropout=0.1, num_mixtures=5):
        """
        Args:
            latent_dim: Dimension of latent state
            action_dim: Dimension of action space
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            num_mixtures: Number of Gaussian mixtures for output
        """
        super(TransformerDynamics, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_mixtures = num_mixtures
        
        # Input projection: (latent + action) -> d_model
        self.input_proj = nn.Linear(latent_dim + action_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: d_model -> mixture parameters
        # For each mixture: mean (latent_dim), logvar (latent_dim), weight (1)
        self.output_proj = nn.Linear(d_model, num_mixtures * (2 * latent_dim + 1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, latent_states, actions):
        """
        Forward pass through transformer dynamics model.
        
        Args:
            latent_states: [batch_size, seq_len, latent_dim]
            actions: [batch_size, seq_len, action_dim]
        
        Returns:
            pi: Mixture weights [batch_size, seq_len, num_mixtures]
            mu: Mixture means [batch_size, seq_len, num_mixtures, latent_dim]
            logvar: Mixture log variances [batch_size, seq_len, num_mixtures, latent_dim]
        """
        batch_size, seq_len = latent_states.shape[:2]
        
        # Concatenate latent state and action
        inputs = torch.cat([latent_states, actions], dim=-1)  # [batch, seq, latent_dim + action_dim]
        
        # Project to model dimension
        x = self.input_proj(inputs)  # [batch, seq, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer forward
        x = self.transformer(x)  # [batch, seq, d_model]
        
        # Output projection
        output = self.output_proj(x)  # [batch, seq, num_mixtures * (2*latent_dim + 1)]
        
        # Reshape and split into components
        output = output.view(batch_size, seq_len, self.num_mixtures, 2 * self.latent_dim + 1)
        
        # Split into pi, mu, logvar
        pi_logits = output[..., 0]  # [batch, seq, num_mixtures]
        pi = F.softmax(pi_logits, dim=-1)
        
        mu = output[..., 1:1 + self.latent_dim]  # [batch, seq, num_mixtures, latent_dim]
        logvar = output[..., 1 + self.latent_dim:]  # [batch, seq, num_mixtures, latent_dim]
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return pi, mu, logvar
    
    def predict(self, latent_state, action, context_latents=None, context_actions=None):
        """
        Predict next latent state given current state and action.
        
        Args:
            latent_state: [batch_size, latent_dim]
            action: [batch_size, action_dim]
            context_latents: Optional context sequence [batch_size, context_len, latent_dim]
            context_actions: Optional context actions [batch_size, context_len, action_dim]
        
        Returns:
            next_latent: Predicted next latent state [batch_size, latent_dim]
        """
        batch_size = latent_state.shape[0]
        
        # Build sequence
        if context_latents is not None and context_actions is not None:
            # Use context + current
            latents = torch.cat([context_latents, latent_state.unsqueeze(1)], dim=1)
            actions = torch.cat([context_actions, action.unsqueeze(1)], dim=1)
        else:
            # Just current
            latents = latent_state.unsqueeze(1)  # [batch, 1, latent_dim]
            actions = action.unsqueeze(1)  # [batch, 1, action_dim]
        
        # Forward pass
        pi, mu, logvar = self.forward(latents, actions)
        
        # Take the last timestep
        pi = pi[:, -1, :]  # [batch, num_mixtures]
        mu = mu[:, -1, :, :]  # [batch, num_mixtures, latent_dim]
        logvar = logvar[:, -1, :, :]  # [batch, num_mixtures, latent_dim]
        
        # Sample from mixture (or take weighted mean)
        # Option 1: Sample
        from torch.distributions import Normal, MixtureSameFamily, Independent, Categorical
        mix_dist = Categorical(pi)
        comp_dist = Independent(Normal(mu, torch.exp(0.5 * logvar)), 1)
        mixture = MixtureSameFamily(mix_dist, comp_dist)
        next_latent = mixture.sample()  # [batch, latent_dim]
        
        # Option 2: Weighted mean (deterministic)
        # next_latent = torch.sum(pi.unsqueeze(-1) * mu, dim=1)
        
        return next_latent

