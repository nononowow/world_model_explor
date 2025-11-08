"""
Model Predictive Control (MPC) planner.

Uses the world model to plan optimal actions by optimizing over
a sequence of actions in the latent space.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable


class CEMPlanner:
    """
    Cross Entropy Method (CEM) planner for MPC.
    
    Optimizes action sequences by iteratively refining a distribution
    over actions using the world model.
    """
    
    def __init__(self, action_dim, horizon=12, num_samples=1000, num_elite=100, 
                 num_iterations=5, alpha=0.25):
        """
        Args:
            action_dim: Dimension of action space
            horizon: Planning horizon (number of steps to plan ahead)
            num_samples: Number of action sequences to sample
            num_elite: Number of elite sequences to keep
            num_iterations: Number of CEM iterations
            alpha: Smoothing factor for updating mean
        """
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elite = num_elite
        self.num_iterations = num_iterations
        self.alpha = alpha
    
    def plan(self, world_model, initial_latent, reward_fn, device='cuda'):
        """
        Plan optimal action sequence using CEM.
        
        Args:
            world_model: World model (dict with 'vae', 'rnn', optionally 'controller')
            initial_latent: Initial latent state [latent_dim]
            reward_fn: Function that computes reward given (latent, action) -> reward
            device: Device to run on
        
        Returns:
            action: Optimal first action [action_dim]
        """
        # Initialize action distribution
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device)
        
        for iteration in range(self.num_iterations):
            # Sample action sequences
            actions = torch.normal(mean.unsqueeze(0).expand(self.num_samples, -1, -1),
                                 std.unsqueeze(0).expand(self.num_samples, -1, -1))
            actions = torch.clamp(actions, -1, 1)  # Clip to valid action range
            
            # Evaluate sequences
            rewards = self._evaluate_sequences(world_model, initial_latent, actions, reward_fn, device)
            
            # Select elite sequences
            elite_indices = torch.argsort(rewards, descending=True)[:self.num_elite]
            elite_actions = actions[elite_indices]
            
            # Update distribution
            mean = self.alpha * elite_actions.mean(dim=0) + (1 - self.alpha) * mean
            std = self.alpha * elite_actions.std(dim=0) + (1 - self.alpha) * std
            std = torch.clamp(std, min=0.1)  # Prevent collapse
        
        # Return first action of best sequence
        best_idx = torch.argmax(rewards)
        return actions[best_idx, 0].cpu().numpy()
    
    def _evaluate_sequences(self, world_model, initial_latent, actions, reward_fn, device):
        """
        Evaluate action sequences using world model.
        
        Args:
            world_model: World model dict
            initial_latent: Initial latent state [latent_dim]
            actions: Action sequences [num_samples, horizon, action_dim]
            reward_fn: Reward function
            device: Device
        
        Returns:
            rewards: Total rewards for each sequence [num_samples]
        """
        num_samples, horizon, _ = actions.shape
        
        # Expand initial latent
        z = initial_latent.unsqueeze(0).expand(num_samples, -1).to(device)
        hidden = None
        
        total_rewards = torch.zeros(num_samples, device=device)
        
        world_model['rnn'].eval()
        
        with torch.no_grad():
            for t in range(horizon):
                # Get actions for this timestep
                action = actions[:, t, :].to(device)
                
                # Compute reward
                rewards = reward_fn(z, action)
                total_rewards += rewards
                
                # Predict next latent state
                z, hidden = world_model['rnn'].predict(z, action.unsqueeze(1), hidden)
                z = z.squeeze(1)
        
        return total_rewards


class MPCPlanner:
    """
    Model Predictive Control planner.
    
    Uses gradient-based optimization to find optimal actions.
    """
    
    def __init__(self, action_dim, horizon=12, num_iterations=10, lr=0.1):
        """
        Args:
            action_dim: Dimension of action space
            horizon: Planning horizon
            num_iterations: Number of optimization iterations
            lr: Learning rate for optimization
        """
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.lr = lr
    
    def plan(self, world_model, initial_latent, reward_fn, device='cuda'):
        """
        Plan optimal action sequence using gradient-based optimization.
        
        Args:
            world_model: World model dict
            initial_latent: Initial latent state [latent_dim]
            reward_fn: Function that computes reward given (latent, action) -> reward
            device: Device
        
        Returns:
            action: Optimal first action [action_dim]
        """
        # Initialize actions
        actions = torch.zeros(self.horizon, self.action_dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([actions], lr=self.lr)
        
        initial_latent = initial_latent.to(device)
        
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Rollout with current actions
            z = initial_latent.unsqueeze(0)
            hidden = None
            total_reward = 0
            
            world_model['rnn'].train()  # Enable gradients
            
            for t in range(self.horizon):
                action = actions[t:t+1].unsqueeze(0)
                
                # Compute reward
                reward = reward_fn(z, action)
                total_reward += reward
                
                # Predict next latent
                z, hidden = world_model['rnn'].predict(z.squeeze(0), action.squeeze(0), hidden)
                z = z.unsqueeze(0)
            
            # Maximize reward (minimize negative reward)
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


def create_reward_fn_from_controller(controller, goal_latent=None):
    """
    Create a reward function from a controller.
    
    Args:
        controller: Controller network
        goal_latent: Optional goal latent state
    
    Returns:
        reward_fn: Function (latent, action) -> reward
    """
    def reward_fn(latent, action):
        """
        Compute reward based on controller output.
        
        Args:
            latent: Latent state [batch_size, latent_dim]
            action: Action [batch_size, action_dim]
        
        Returns:
            reward: Scalar reward
        """
        # Option 1: Reward based on controller's action matching
        controller_action = controller(latent)
        reward = -F.mse_loss(action, controller_action)
        
        # Option 2: If goal is provided, reward proximity to goal
        if goal_latent is not None:
            goal_reward = -F.mse_loss(latent, goal_latent.unsqueeze(0).expand_as(latent))
            reward = reward + 0.5 * goal_reward
        
        return reward
    
    return reward_fn

