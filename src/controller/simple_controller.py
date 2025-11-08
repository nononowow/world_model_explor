"""
Simple controller/policy network implementation.

The controller takes latent states (and optionally hidden states from RNN)
and outputs actions. Can be trained using evolutionary strategies or
gradient-based methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleController(nn.Module):
    """
    Simple feedforward controller that maps latent states to actions.
    
    This is a basic policy network that can be used with world models.
    """
    
    def __init__(self, latent_dim=32, action_dim=3, hidden_dims=[256, 256]):
        """
        Args:
            latent_dim: Dimension of latent state
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(SimpleController, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Actions typically in [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, latent_state):
        """
        Forward pass through controller.
        
        Args:
            latent_state: Latent state [batch_size, latent_dim]
        
        Returns:
            action: Action [batch_size, action_dim]
        """
        return self.network(latent_state)
    
    def act(self, latent_state, deterministic=True):
        """
        Get action from latent state.
        
        Args:
            latent_state: Latent state [batch_size, latent_dim] or [latent_dim]
            deterministic: If True, return mean action; if False, add noise
        
        Returns:
            action: Action
        """
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
        
        action = self.forward(latent_state)
        
        if not deterministic:
            # Add exploration noise
            noise = torch.randn_like(action) * 0.1
            action = action + noise
            action = torch.clamp(action, -1, 1)
        
        if action.shape[0] == 1:
            action = action.squeeze(0)
        
        return action


class RNNController(nn.Module):
    """
    Controller that uses both latent state and RNN hidden state.
    
    This is useful when you want the controller to have memory
    of past states through the RNN hidden state.
    """
    
    def __init__(self, latent_dim=32, rnn_hidden_dim=256, action_dim=3, hidden_dims=[256]):
        """
        Args:
            latent_dim: Dimension of latent state
            rnn_hidden_dim: Dimension of RNN hidden state
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(RNNController, self).__init__()
        
        # Concatenate latent state and RNN hidden state
        input_dim = latent_dim + rnn_hidden_dim
        
        # Build network layers
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, latent_state, rnn_hidden):
        """
        Forward pass through controller.
        
        Args:
            latent_state: Latent state [batch_size, latent_dim]
            rnn_hidden: RNN hidden state [batch_size, rnn_hidden_dim]
        
        Returns:
            action: Action [batch_size, action_dim]
        """
        # Flatten RNN hidden if it's a tuple (h, c)
        if isinstance(rnn_hidden, tuple):
            rnn_hidden = rnn_hidden[0]  # Use hidden state, not cell state
            rnn_hidden = rnn_hidden[-1]  # Take last layer if multi-layer
        
        # Concatenate
        combined = torch.cat([latent_state, rnn_hidden], dim=-1)
        return self.network(combined)

