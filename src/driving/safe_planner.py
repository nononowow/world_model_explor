"""
Uncertainty-aware planning for safe autonomous driving.

Implements safety constraints and uncertainty quantification in planning.
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.planning.mpc_planner import CEMPlanner


class SafePlanner(CEMPlanner):
    """
    Safe planner that considers uncertainty and safety constraints.
    """
    
    def __init__(self, action_dim, horizon=12, num_samples=1000, num_elite=100,
                 num_iterations=5, alpha=0.25, safety_threshold=0.1):
        """
        Args:
            action_dim: Dimension of action space
            horizon: Planning horizon
            num_samples: Number of action sequences to sample
            num_elite: Number of elite sequences
            num_iterations: Number of CEM iterations
            alpha: Smoothing factor
            safety_threshold: Uncertainty threshold for safety
        """
        super(SafePlanner, self).__init__(
            action_dim, horizon, num_samples, num_elite, num_iterations, alpha
        )
        self.safety_threshold = safety_threshold
    
    def plan(self, world_model, initial_latent, reward_fn, device='cuda'):
        """
        Plan with safety constraints.
        
        Args:
            world_model: World model dict
            initial_latent: Initial latent state
            reward_fn: Reward function
            device: Device
        
        Returns:
            action: Safe action
        """
        # Get base plan
        action = super().plan(world_model, initial_latent, reward_fn, device)
        
        # Check safety and adjust if needed
        action = self._ensure_safety(world_model, initial_latent, action, device)
        
        return action
    
    def _ensure_safety(self, world_model, initial_latent, action, device):
        """
        Ensure action is safe by checking uncertainty.
        
        Args:
            world_model: World model dict
            initial_latent: Initial latent state
            action: Proposed action
            device: Device
        
        Returns:
            safe_action: Adjusted action if needed
        """
        # Predict next state with uncertainty
        world_model['rnn'].eval()
        with torch.no_grad():
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            initial_latent_tensor = initial_latent.unsqueeze(0).to(device) if isinstance(initial_latent, np.ndarray) else initial_latent
            
            # Get prediction with uncertainty
            pi, mu, logvar, _ = world_model['rnn'](
                initial_latent_tensor.unsqueeze(1),
                action_tensor.unsqueeze(1)
            )
            
            # Compute uncertainty (variance of mixture)
            var = torch.exp(logvar)  # [batch, seq, num_mixtures, latent_dim]
            mean_var = (pi.unsqueeze(-1) * var).sum(dim=2)  # Weighted variance
            uncertainty = mean_var.mean().item()
        
        # If uncertainty is too high, reduce action magnitude
        if uncertainty > self.safety_threshold:
            # Scale down action (be more conservative)
            action = action * 0.5
        
        return action


def compute_safety_reward(latent, action, world_model, device='cuda'):
    """
    Compute reward that includes safety penalty.
    
    Args:
        latent: Current latent state
        action: Action
        world_model: World model dict
        device: Device
    
    Returns:
        reward: Safety-aware reward
    """
    # Base reward (from controller or other source)
    base_reward = 0.0  # Would come from actual reward function
    
    # Safety penalty based on uncertainty
    world_model['rnn'].eval()
    with torch.no_grad():
        if isinstance(latent, np.ndarray):
            latent = torch.FloatTensor(latent).unsqueeze(0).to(device)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).unsqueeze(0).to(device)
        
        pi, mu, logvar, _ = world_model['rnn'](
            latent.unsqueeze(1),
            action.unsqueeze(1)
        )
        
        # Compute uncertainty
        var = torch.exp(logvar)
        mean_var = (pi.unsqueeze(-1) * var).sum(dim=2)
        uncertainty = mean_var.mean().item()
    
    # Penalty for high uncertainty
    safety_penalty = -uncertainty * 10.0
    
    return base_reward + safety_penalty

